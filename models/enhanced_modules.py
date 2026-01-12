import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QuaternionEmbeddingWrapper(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # 4 phần quaternion: r, i, j, k
        self.real = nn.Embedding(num_embeddings, embedding_dim)
        self.i = nn.Embedding(num_embeddings, embedding_dim)
        self.j = nn.Embedding(num_embeddings, embedding_dim)
        self.k = nn.Embedding(num_embeddings, embedding_dim)
        
        # Linear projection về real space
        self.project = nn.Linear(embedding_dim * 4, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for emb in [self.real, self.i, self.j, self.k]:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.project.weight)

    def forward(self, indices):
        r = self.real(indices)
        i = self.i(indices)
        j = self.j(indices)
        k = self.k(indices)
        
        quat_concat = torch.cat([r, i, j, k], dim=-1)  # [batch, 4*dim]
        out = self.project(quat_concat)
        return out

class AdaptiveTemperatureLayer(nn.Module):
    """
    Adaptive Temperature Scaling for relation-aware gating
    Learns dynamic temperature for each relation type to improve MoE gating
    """
    def __init__(self, num_relations, hidden_dim=64, init_temp=1.0):
        super(AdaptiveTemperatureLayer, self).__init__()
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.init_temp = init_temp
        
        # Temperature prediction network
        self.temp_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive temperature
        )
        
        # Relation embedding for temperature computation
        self.rel_temp_embed = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_normal_(self.rel_temp_embed.weight)
        
        # Initialize to reasonable temperature values
        with torch.no_grad():
            self.temp_predictor[-2].bias.data.fill_(math.log(math.exp(init_temp) - 1))
    
    def forward(self, relation_ids):
        """
        Args:
            relation_ids: [batch_size] tensor of relation IDs
        Returns:
            temperatures: [batch_size, 1] tensor of adaptive temperatures
        """
        rel_temp_emb = self.rel_temp_embed(relation_ids)
        temperatures = self.temp_predictor(rel_temp_emb)
        return temperatures + 0.1  # Ensure minimum temperature


class CrossModalAttentionLayer(nn.Module):
    """
    Cross-Modal Attention mechanism for better multi-modal fusion
    Uses attention to better integrate information across modalities
    """
    def __init__(self, struct_dim, img_dim, txt_dim, hidden_dim=256, num_heads=4):
        super(CrossModalAttentionLayer, self).__init__()
        self.struct_dim = struct_dim
        self.img_dim = img_dim  
        self.txt_dim = txt_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project all modalities to same dimension for attention
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        
        # Multi-head cross-attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projections back to original dimensions
        self.struct_out_proj = nn.Linear(hidden_dim, struct_dim)
        self.img_out_proj = nn.Linear(hidden_dim, img_dim)
        self.txt_out_proj = nn.Linear(hidden_dim, txt_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, struct_emb, img_emb, txt_emb):
        """
        Args:
            struct_emb: [batch_size, struct_dim] structural embeddings
            img_emb: [batch_size, img_dim] image embeddings  
            txt_emb: [batch_size, txt_dim] text embeddings
        Returns:
            enhanced embeddings for each modality
        """
        batch_size = struct_emb.size(0)
        
        # Project to common dimension
        struct_proj = self.struct_proj(struct_emb)  # [batch_size, hidden_dim]
        img_proj = self.img_proj(img_emb)          # [batch_size, hidden_dim]
        txt_proj = self.txt_proj(txt_emb)          # [batch_size, hidden_dim]
        
        # Stack modalities for attention
        # [batch_size, 3, hidden_dim] - (struct, img, txt)
        modal_stack = torch.stack([struct_proj, img_proj, txt_proj], dim=1)
        
        # Apply self-attention across modalities
        attended, attn_weights = self.multihead_attn(
            modal_stack, modal_stack, modal_stack
        )
        
        # Add residual connection and layer norm
        attended = self.layer_norm(attended + modal_stack)
        attended = self.dropout(attended)
        
        # Split back to individual modalities
        enhanced_struct = attended[:, 0, :]  # [batch_size, hidden_dim]
        enhanced_img = attended[:, 1, :]     # [batch_size, hidden_dim]
        enhanced_txt = attended[:, 2, :]     # [batch_size, hidden_dim]
        
        # Project back to original dimensions
        enhanced_struct = self.struct_out_proj(enhanced_struct) + struct_emb
        enhanced_img = self.img_out_proj(enhanced_img) + img_emb  
        enhanced_txt = self.txt_out_proj(enhanced_txt) + txt_emb
        
        return enhanced_struct, enhanced_img, enhanced_txt, attn_weights


class ConsistencyLoss(nn.Module):
    """
    Consistency loss to encourage similar predictions across modalities
    Helps improve robustness and multi-modal alignment
    """
    def __init__(self, temperature=0.5, lambda_consist=0.1):
        super(ConsistencyLoss, self).__init__()
        self.temperature = temperature
        self.lambda_consist = lambda_consist
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, predictions):
        """
        Args:
            predictions: List of [pred_s, pred_i, pred_d, pred_mm] prediction tensors
        Returns:
            consistency_loss: scalar tensor
        """
        pred_s, pred_i, pred_d, pred_mm = predictions
        
        # Apply temperature scaling for smoother distributions
        pred_s_soft = F.softmax(pred_s / self.temperature, dim=-1)
        pred_i_soft = F.softmax(pred_i / self.temperature, dim=-1)
        pred_d_soft = F.softmax(pred_d / self.temperature, dim=-1)
        pred_mm_soft = F.softmax(pred_mm / self.temperature, dim=-1)
        
        # Compute pairwise KL divergences
        consist_loss = 0.0
        count = 0
        
        # Between structural and visual
        consist_loss += self.kl_div(F.log_softmax(pred_s / self.temperature, dim=-1), pred_i_soft)
        consist_loss += self.kl_div(F.log_softmax(pred_i / self.temperature, dim=-1), pred_s_soft)
        
        # Between structural and textual  
        consist_loss += self.kl_div(F.log_softmax(pred_s / self.temperature, dim=-1), pred_d_soft)
        consist_loss += self.kl_div(F.log_softmax(pred_d / self.temperature, dim=-1), pred_s_soft)
        
        # Between visual and textual
        consist_loss += self.kl_div(F.log_softmax(pred_i / self.temperature, dim=-1), pred_d_soft)
        consist_loss += self.kl_div(F.log_softmax(pred_d / self.temperature, dim=-1), pred_i_soft)
        
        # Between each modality and fused
        consist_loss += self.kl_div(F.log_softmax(pred_s / self.temperature, dim=-1), pred_mm_soft)
        consist_loss += self.kl_div(F.log_softmax(pred_i / self.temperature, dim=-1), pred_mm_soft)
        consist_loss += self.kl_div(F.log_softmax(pred_d / self.temperature, dim=-1), pred_mm_soft)
        
        return self.lambda_consist * consist_loss / 9.0  # Average over all pairs


class RelationAwareGating(nn.Module):
    """
    Enhanced relation-aware gating mechanism with adaptive temperature
    """
    def __init__(self, input_dim, num_relations, hidden_dim=64):
        super(RelationAwareGating, self).__init__()
        self.input_dim = input_dim
        self.num_relations = num_relations
        
        # Adaptive temperature
        self.temp_layer = AdaptiveTemperatureLayer(num_relations, hidden_dim)
        
        # Relation-specific gating weights
        self.rel_gate_weights = nn.Embedding(num_relations, input_dim)
        nn.init.xavier_normal_(self.rel_gate_weights.weight)
        
    def forward(self, x, relation_ids):
        """
        Args:
            x: [batch_size, input_dim] input embeddings
            relation_ids: [batch_size] relation IDs
        Returns:
            gated_x: [batch_size, input_dim] gated embeddings
            temperatures: [batch_size, 1] adaptive temperatures
        """
        # Get adaptive temperature
        temperatures = self.temp_layer(relation_ids)
        
        # Get relation-specific gates
        rel_gates = self.rel_gate_weights(relation_ids)  # [batch_size, input_dim]
        
        # Apply gating with temperature
        gate_logits = x * rel_gates
        gates = torch.sigmoid(gate_logits / temperatures)
        
        # Gated output
        gated_x = gates * x
        
        return gated_x, temperatures
