import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *
from .model import BaseModel
from .enhanced_modules import AdaptiveTemperatureLayer, CrossModalAttentionLayer, ConsistencyLoss, RelationAwareGating


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer"""
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class EnhancedMoEAdaptorLayer(nn.Module):
    """Enhanced MoE-Adaptor with Adaptive Temperature Scaling"""
    def __init__(self, n_exps, layers, num_relations, dropout=0.0, noise=True):
        super(EnhancedMoEAdaptorLayer, self).__init__()
        self.n_exps = n_exps
        self.noisy_gating = noise
        
        # Original experts
        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        
        # Enhanced: Adaptive temperature scaling
        self.adaptive_temp = AdaptiveTemperatureLayer(num_relations, hidden_dim=64)
        
    def noisy_top_k_gating(self, x, relation_ids=None, train=None, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        
        # Enhanced: Use adaptive temperature if relation_ids provided
        if relation_ids is not None:
            temperatures = self.adaptive_temp(relation_ids)
            gates = F.softmax(logits / temperatures, dim=-1)
        else:
            gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x, relation_ids=None):
        gates = self.noisy_top_k_gating(x, relation_ids, self.training)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2), expert_outputs, gates


class EnhancedModalFusionLayer(nn.Module):
    """Enhanced Modal Fusion with Cross-Modal Attention"""
    def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim):
        super(EnhancedModalFusionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.multi = multi
        self.img_dim = img_dim
        self.text_dim = txt_dim

        # Original fusion layers
        modal1 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(in_dim, out_dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.img_dim, out_dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.text_dim, out_dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        
        # Enhanced: Cross-modal attention
        self.cross_modal_attn = CrossModalAttentionLayer(
            struct_dim=out_dim, 
            img_dim=out_dim, 
            txt_dim=out_dim,
            hidden_dim=out_dim,
            num_heads=4
        )

    def forward(self, modal1_emb, modal2_emb, modal3_emb, use_cross_attn=True):
        batch_size = modal1_emb.size(0)
        x_mm = []
        
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            
            # Enhanced: Apply cross-modal attention
            if use_cross_attn:
                x_modal1, x_modal2, x_modal3, cross_attn_weights = self.cross_modal_attn(
                    x_modal1, x_modal2, x_modal3
                )
            
            x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)
            attention_scores = self.ent_attn(x_stack).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)
            x_mm.append(context_vectors)
            
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        
        if use_cross_attn:
            return x_mm, attention_weights, cross_attn_weights
        else:
            return x_mm, attention_weights


class EnhancedRelMoE(BaseModel):
    """Enhanced RelMoE with Adaptive Temperature and Cross-Modal Attention"""
    def __init__(self, args):
        super(EnhancedRelMoE, self).__init__(args)

        # Store args for enhanced features and xAI
        self.args = args
        self.use_adaptive_temp = getattr(args, 'use_adaptive_temp', True)
        self.use_cross_modal_attn = getattr(args, 'use_cross_modal_attn', True)
        self.consistency_weight = getattr(args, 'consistency_weight', 0.1)

        # Embedding space: None (real), 'complex' (2x), 'quaternion' (4x)
        space = getattr(args, 'space', None)
        if isinstance(space, str):
            space = space.lower()
        if space == 'complex':
            self.space_mult = 2
        elif space == 'quaternion':
            self.space_mult = 4
        else:
            self.space_mult = 1

        # Effective embedding sizes
        eff_dim = args.dim * self.space_mult
        eff_r_dim = args.r_dim * self.space_mult

        # Entity / relation embeddings
        self.entity_embeddings = nn.Embedding(len(args.entity2id), eff_dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)

        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), eff_r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        # Load pretrained embeddings if requested (and expand if using complex/quaternion)
        if args.pre_trained:
            ent_vec = torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float()
            rel_vec = torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()
            if self.space_mult > 1:
                ent_vec = ent_vec.repeat(1, self.space_mult)
                rel_vec = rel_vec.repeat(1, self.space_mult)
            self.entity_embeddings = nn.Embedding.from_pretrained(ent_vec, freeze=False)
            rel_cat = torch.cat((rel_vec, -1 * rel_vec), dim=0)
            if self.space_mult > 1:
                rel_cat = rel_cat.repeat(1, self.space_mult)
            self.relation_embeddings = nn.Embedding.from_pretrained(rel_cat, freeze=False)

        self.rel_gate = nn.Embedding(2 * len(args.relation2id), 1, padding_idx=None)

        # Multi-modal embeddings setup (same as original)
        if args.dataset == "DB15K":
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "MKG" in args.dataset:
            img = args.img.to(self.device).view(args.img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 32))
            txt = txt.view(txt.size(0), -1)
        elif "TIVA" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 32, 64))
            img = img.view(img.size(0), -1)
            txt = args.desp.to(self.device)
            txt = txt.view(txt.size(0), -1)
        elif "Kuai" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 12, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "WN9" in args.dataset:
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            img = torch.tensor(img).to(torch.float32)
            txt = args.desp.to(self.device)
            txt = txt.view(txt.size(0), -1)
            txt = torch.tensor(txt).to(torch.float32)
        elif "FB15K-237" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 12, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)

        # If using complex/quaternion, expand image/text features by repeating channels
        if self.space_mult > 1:
            img = img.repeat(1, self.space_mult)
            txt = txt.repeat(1, self.space_mult)

        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
        self.img_relation_embeddings = nn.Embedding(2 * len(args.relation2id), eff_r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)

        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
        self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), eff_r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        # Dimensions (expanded if complex/quaternion)
        self.real_dim = args.dim
        self.dim = eff_dim
        self.img_dim = self.img_entity_embeddings.weight.data.shape[1]
        self.txt_dim = self.txt_entity_embeddings.weight.data.shape[1]
        self.fuse_out_dim = self.dim

        # Score function layers use effective (expanded) dims
        self.TuckER_S = TuckERLayer(self.dim, eff_r_dim)
        self.TuckER_I = TuckERLayer(self.img_dim, eff_r_dim)
        self.TuckER_D = TuckERLayer(self.txt_dim, eff_r_dim)
        self.TuckER_MM = TuckERLayer(self.dim, self.fuse_out_dim)

        # Enhanced: MoE layers with adaptive temperature
        num_relations = 2 * len(args.relation2id)
        if self.use_adaptive_temp:
            self.visual_moe = EnhancedMoEAdaptorLayer(
                n_exps=args.n_exp,
                layers=[self.img_dim, self.img_dim],
                num_relations=num_relations
            )
            self.text_moe = EnhancedMoEAdaptorLayer(
                n_exps=args.n_exp,
                layers=[self.txt_dim, self.txt_dim],
                num_relations=num_relations
            )
            self.structure_moe = EnhancedMoEAdaptorLayer(
                n_exps=args.n_exp,
                layers=[self.dim, self.dim],
                num_relations=num_relations
            )
            self.mm_moe = EnhancedMoEAdaptorLayer(
                n_exps=args.n_exp,
                layers=[self.fuse_out_dim, self.fuse_out_dim],
                num_relations=num_relations
            )
        else:
            # Fall back to original MoE
            from .RelMoE import MoEAdaptorLayer
            self.visual_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.img_dim, self.img_dim])
            self.text_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.txt_dim, self.txt_dim])
            self.structure_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.dim, self.dim])
            self.mm_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.fuse_out_dim, self.fuse_out_dim])

        # Enhanced: Modal fusion with cross-attention
        if self.use_cross_modal_attn:
            self.fuse_e = EnhancedModalFusionLayer(
                in_dim=self.dim,
                out_dim=self.fuse_out_dim,
                multi=2,
                img_dim=self.img_dim,
                txt_dim=self.txt_dim
            )
            self.fuse_r = EnhancedModalFusionLayer(
                in_dim=eff_r_dim,
                out_dim=self.fuse_out_dim,
                multi=2,
                img_dim=eff_r_dim,
                txt_dim=eff_r_dim
            )
        else:
            # Fall back to original fusion
            from .RelMoE import ModalFusionLayer
            self.fuse_e = ModalFusionLayer(
                in_dim=self.dim,
                out_dim=self.fuse_out_dim,
                multi=2,
                img_dim=self.img_dim,
                txt_dim=self.txt_dim
            )
            self.fuse_r = ModalFusionLayer(
                in_dim=eff_r_dim,
                out_dim=self.fuse_out_dim,
                multi=2,
                img_dim=eff_r_dim,
                txt_dim=eff_r_dim
            )

        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

        # Enhanced: Consistency loss
        self.consistency_loss = ConsistencyLoss(
            temperature=0.5,
            lambda_consist=self.consistency_weight
        )

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        rel_gate = self.rel_gate(relation)
        
        # Enhanced: Pass relation IDs to MoE if using adaptive temperature
        if self.use_adaptive_temp:
            e_embed, disen_str, atten_s = self.structure_moe(self.entity_embeddings(head), relation)
            e_img_embed, disen_img, atten_i = self.visual_moe(self.img_entity_embeddings(head), relation)
            e_txt_embed, disen_txt, atten_t = self.text_moe(self.txt_entity_embeddings(head), relation)
        else:
            e_embed, disen_str, atten_s = self.structure_moe(self.entity_embeddings(head), rel_gate)
            e_img_embed, disen_img, atten_i = self.visual_moe(self.img_entity_embeddings(head), rel_gate)
            e_txt_embed, disen_txt, atten_t = self.text_moe(self.txt_entity_embeddings(head), rel_gate)
        
        r_embed = self.relation_embeddings(relation)
        r_img_embed = self.img_relation_embeddings(relation)
        r_txt_embed = self.txt_relation_embeddings(relation)
        
        # Enhanced: Modal fusion with cross-attention
        if self.use_cross_modal_attn:
            e_mm_embed, attn_f, cross_attn_weights = self.fuse_e(e_embed, e_img_embed, e_txt_embed)
            r_mm_embed, _, _ = self.fuse_r(r_embed, r_img_embed, r_txt_embed)
        else:
            e_mm_embed, attn_f = self.fuse_e(e_embed, e_img_embed, e_txt_embed)
            r_mm_embed, _ = self.fuse_r(r_embed, r_img_embed, r_txt_embed)
        
        # Score computation
        pred_s = self.TuckER_S(e_embed, r_embed)
        pred_i = self.TuckER_I(e_img_embed, r_img_embed)
        pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)
        pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)
        
        # Compute all embeddings for scoring
        if self.use_adaptive_temp:
            all_s, _, _ = self.structure_moe(self.entity_embeddings.weight, torch.zeros_like(self.entity_embeddings.weight[:, 0], dtype=torch.long))
            all_v, _, _ = self.visual_moe(self.img_entity_embeddings.weight, torch.zeros_like(self.img_entity_embeddings.weight[:, 0], dtype=torch.long))
            all_t, _, _ = self.text_moe(self.txt_entity_embeddings.weight, torch.zeros_like(self.txt_entity_embeddings.weight[:, 0], dtype=torch.long))
        else:
            all_s, _, _ = self.structure_moe(self.entity_embeddings.weight)
            all_v, _, _ = self.visual_moe(self.img_entity_embeddings.weight)
            all_t, _, _ = self.text_moe(self.txt_entity_embeddings.weight)
        
        if self.use_cross_modal_attn:
            all_f, _, _ = self.fuse_e(all_s, all_v, all_t)
        else:
            all_f, _ = self.fuse_e(all_s, all_v, all_t)
        
        pred_s = torch.mm(pred_s, all_s.transpose(1, 0))
        pred_i = torch.mm(pred_i, all_v.transpose(1, 0))
        pred_d = torch.mm(pred_d, all_t.transpose(1, 0))
        pred_mm = torch.mm(pred_mm, all_f.transpose(1, 0))

        pred_s = torch.sigmoid(pred_s)
        pred_i = torch.sigmoid(pred_i)
        pred_d = torch.sigmoid(pred_d)
        pred_mm = torch.sigmoid(pred_mm)
        
        if not self.training:
            return [pred_s, pred_i, pred_d, pred_mm], [atten_s, atten_i, atten_t, attn_f]
        else:
            return [pred_s, pred_i, pred_d, pred_mm], [disen_str, disen_img, disen_txt]

    def get_batch_embeddings(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        
        if self.use_adaptive_temp:
            _, disen_str, _ = self.structure_moe(self.entity_embeddings(head), relation)
            _, disen_img, _ = self.visual_moe(self.img_entity_embeddings(head), relation)
            _, disen_txt, _ = self.text_moe(self.txt_entity_embeddings(head), relation)
        else:
            rel_gate = self.rel_gate(relation)
            _, disen_str, _ = self.structure_moe(self.entity_embeddings(head), rel_gate)
            _, disen_img, _ = self.visual_moe(self.img_entity_embeddings(head), rel_gate)
            _, disen_txt, _ = self.text_moe(self.txt_entity_embeddings(head), rel_gate)
        
        return [disen_str, disen_img, disen_txt]

    def loss_func(self, output, target):
        # Original losses
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)
        
        main_loss = loss_s + loss_i + loss_d + loss_mm
        
        # Enhanced: Add consistency loss
        if self.training and self.consistency_weight > 0:
            consistency_loss = self.consistency_loss(output)
            return main_loss + consistency_loss
        else:
            return main_loss

    def generate_explanations(self, batch_inputs, target, methods=['attention', 'gradcam', 'modality']):
        """
        Generate explanations using xAI modules
        Args:
            batch_inputs: Input batch [batch, 3]
            target: Target labels [batch, n_entities]
            methods: List of explanation methods to use
        Returns:
            Dictionary containing all explanations
        """
        from .xai_modules import ExplainabilityManager
        
        # Create explainability manager if not exists
        if not hasattr(self, '_explainability_manager'):
            self._explainability_manager = ExplainabilityManager(self, self.args)
            self._explainability_manager.setup_hooks()
        
        # Generate explanations
        explanations = self._explainability_manager.generate_explanations(
            batch_inputs, target, methods
        )
        
        return explanations
    
    def forward_with_ablation(self, batch_inputs, ablate_modal='struct'):
        """
        Forward pass with one modality ablated (set to zero)
        Used for ablation studies in xAI
        """
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        rel_gate = self.rel_gate(relation)
        
        # Get embeddings
        if self.use_adaptive_temp:
            e_embed, disen_str, atten_s = self.structure_moe(self.entity_embeddings(head), relation)
            e_img_embed, disen_img, atten_i = self.visual_moe(self.img_entity_embeddings(head), relation)
            e_txt_embed, disen_txt, atten_t = self.text_moe(self.txt_entity_embeddings(head), relation)
        else:
            e_embed, disen_str, atten_s = self.structure_moe(self.entity_embeddings(head), rel_gate)
            e_img_embed, disen_img, atten_i = self.visual_moe(self.img_entity_embeddings(head), rel_gate)
            e_txt_embed, disen_txt, atten_t = self.text_moe(self.txt_entity_embeddings(head), rel_gate)
        
        # Ablate specified modality
        if ablate_modal == 'struct':
            e_embed = torch.zeros_like(e_embed)
        elif ablate_modal == 'img':
            e_img_embed = torch.zeros_like(e_img_embed)
        elif ablate_modal == 'txt':
            e_txt_embed = torch.zeros_like(e_txt_embed)
        
        r_embed = self.relation_embeddings(relation)
        r_img_embed = self.img_relation_embeddings(relation)
        r_txt_embed = self.txt_relation_embeddings(relation)
        
        # Modal fusion
        if self.use_cross_modal_attn:
            e_mm_embed, attn_f, cross_attn_weights = self.fuse_e(e_embed, e_img_embed, e_txt_embed)
            r_mm_embed, _, _ = self.fuse_r(r_embed, r_img_embed, r_txt_embed)
        else:
            e_mm_embed, attn_f = self.fuse_e(e_embed, e_img_embed, e_txt_embed)
            r_mm_embed, _ = self.fuse_r(r_embed, r_img_embed, r_txt_embed)
        
        # Score computation
        pred_s = self.TuckER_S(e_embed, r_embed)
        pred_i = self.TuckER_I(e_img_embed, r_img_embed)
        pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)
        pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)
        
        # Compute all embeddings for scoring
        if self.use_adaptive_temp:
            all_s, _, _ = self.structure_moe(self.entity_embeddings.weight, torch.zeros_like(self.entity_embeddings.weight[:, 0], dtype=torch.long))
            all_v, _, _ = self.visual_moe(self.img_entity_embeddings.weight, torch.zeros_like(self.img_entity_embeddings.weight[:, 0], dtype=torch.long))
            all_t, _, _ = self.text_moe(self.txt_entity_embeddings.weight, torch.zeros_like(self.txt_entity_embeddings.weight[:, 0], dtype=torch.long))
        else:
            all_s, _, _ = self.structure_moe(self.entity_embeddings.weight)
            all_v, _, _ = self.visual_moe(self.img_entity_embeddings.weight)
            all_t, _, _ = self.text_moe(self.txt_entity_embeddings.weight)
        
        if self.use_cross_modal_attn:
            all_f, _, _ = self.fuse_e(all_s, all_v, all_t)
        else:
            all_f, _ = self.fuse_e(all_s, all_v, all_t)
        
        pred_s = torch.mm(pred_s, all_s.transpose(1, 0))
        pred_i = torch.mm(pred_i, all_v.transpose(1, 0))
        pred_d = torch.mm(pred_d, all_t.transpose(1, 0))
        pred_mm = torch.mm(pred_mm, all_f.transpose(1, 0))

        pred_s = torch.sigmoid(pred_s)
        pred_i = torch.sigmoid(pred_i)
        pred_d = torch.sigmoid(pred_d)
        pred_mm = torch.sigmoid(pred_mm)
        
        return [pred_s, pred_i, pred_d, pred_mm]
