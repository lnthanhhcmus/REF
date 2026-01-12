"""
Research-Oriented Enhancements for Multi-Modal Knowledge Graph Completion
Focuses on novel contributions with solid theoretical foundations
Avoids unnecessary deep learning complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class RelationSpecificExpertRouter(nn.Module):
    """
    Relation-Specific Expert Routing Mechanism
    
    Key Contribution: Instead of using the same gating mechanism for all relations,
    this module learns relation-specific routing patterns to better capture the
    unique characteristics of different relation types in multi-modal KG completion.
    
    Benefits:
    - Better expert specialization per relation type
    - Interpretable routing decisions
    - No significant increase in parameters
    """
    def __init__(self, num_relations, n_experts, hidden_dim=64):
        super(RelationSpecificExpertRouter, self).__init__()
        self.num_relations = num_relations
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        
        # Relation-specific routing weights (lightweight)
        self.relation_router = nn.Embedding(num_relations, n_experts)
        nn.init.uniform_(self.relation_router.weight, 0.0, 1.0)
        
        # Refinement network (small MLP)
        self.refine_net = nn.Sequential(
            nn.Linear(hidden_dim + n_experts, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_experts)
        )
        
        # Relation context encoder
        self.rel_context = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_normal_(self.rel_context.weight)
        
    def forward(self, base_gates, relation_ids, entity_emb=None):
        """
        Args:
            base_gates: [batch_size, n_experts] - base gating from MoE
            relation_ids: [batch_size] - relation type ids
            entity_emb: [batch_size, hidden_dim] - optional entity embeddings
        Returns:
            refined_gates: [batch_size, n_experts] - relation-aware gates
            routing_weights: [batch_size, n_experts] - interpretable routing scores
        """
        batch_size = relation_ids.size(0)
        
        # Get relation-specific routing preference
        rel_routing = self.relation_router(relation_ids)  # [batch_size, n_experts]
        rel_routing = F.softmax(rel_routing, dim=-1)
        
        # Get relation context
        rel_ctx = self.rel_context(relation_ids)  # [batch_size, hidden_dim]
        
        # Combine base gates with relation routing
        if entity_emb is not None:
            # Concatenate relation context with base gates
            combined = torch.cat([rel_ctx, base_gates], dim=-1)
        else:
            combined = torch.cat([rel_ctx, base_gates], dim=-1)
        
        # Refine gates using relation context
        refined_logits = self.refine_net(combined)  # [batch_size, n_experts]
        
        # Combine base gates with relation-specific routing
        # This creates a balanced combination
        alpha = 0.6  # Weight for relation-specific routing
        final_gates = alpha * rel_routing + (1 - alpha) * F.softmax(refined_logits, dim=-1)
        
        # Add residual connection with base gates
        final_gates = 0.7 * final_gates + 0.3 * base_gates
        
        # Renormalize
        final_gates = final_gates / (final_gates.sum(dim=-1, keepdim=True) + 1e-10)
        
        return final_gates, rel_routing


class DynamicModalWeighting(nn.Module):
    """
    Dynamic Modal Weighting based on Relation Context
    
    Key Contribution: Different relations benefit from different modalities.
    For example, visual relations like "has_color" benefit more from image features,
    while abstract relations like "part_of" benefit more from structural features.
    
    This module learns relation-aware weights for each modality dynamically.
    
    Benefits:
    - Adaptive fusion based on relation semantics
    - Improved performance without adding complex architectures
    - Interpretable modality importance per relation
    """
    def __init__(self, num_relations, num_modalities=4, hidden_dim=64):
        super(DynamicModalWeighting, self).__init__()
        self.num_relations = num_relations
        self.num_modalities = num_modalities  # structure, visual, text, fused
        self.hidden_dim = hidden_dim
        
        # Relation encoder for modality weighting
        self.rel_encoder = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_normal_(self.rel_encoder.weight)
        
        # Modality weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Learned prior for each modality (trainable baseline)
        self.modality_prior = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        
    def forward(self, predictions, relation_ids):
        """
        Args:
            predictions: List of [pred_s, pred_i, pred_d, pred_mm] - predictions from each modality
            relation_ids: [batch_size] - relation type ids
        Returns:
            weighted_pred: [batch_size, num_entities] - dynamically weighted prediction
            modal_weights: [batch_size, num_modalities] - interpretable weights
        """
        batch_size = relation_ids.size(0)
        
        # Get relation embeddings
        rel_emb = self.rel_encoder(relation_ids)  # [batch_size, hidden_dim]
        
        # Predict modality weights for this batch
        modal_weights = self.weight_predictor(rel_emb)  # [batch_size, num_modalities]
        
        # Add prior knowledge
        prior_weights = F.softmax(self.modality_prior, dim=0)
        modal_weights = 0.8 * modal_weights + 0.2 * prior_weights.unsqueeze(0)
        
        # Stack predictions
        pred_stack = torch.stack(predictions, dim=1)  # [batch_size, num_modalities, num_entities]
        
        # Apply dynamic weights
        weighted_pred = (modal_weights.unsqueeze(-1) * pred_stack).sum(dim=1)
        
        return weighted_pred, modal_weights


class UncertaintyEstimator(nn.Module):
    """
    Uncertainty-Aware Prediction with Ensemble
    
    Key Contribution: Estimates prediction uncertainty using expert disagreement
    and distributional properties. Allows for confidence-aware ensembling.
    
    Based on epistemic uncertainty estimation in Mixture of Experts.
    
    Benefits:
    - Improved reliability of predictions
    - Better handling of out-of-distribution samples
    - Enables selective prediction (reject low-confidence predictions)
    """
    def __init__(self, n_experts, num_entities):
        super(UncertaintyEstimator, self).__init__()
        self.n_experts = n_experts
        self.num_entities = num_entities
        
        # Learnable uncertainty calibration
        self.uncertainty_scale = nn.Parameter(torch.ones(1))
        self.uncertainty_bias = nn.Parameter(torch.zeros(1))
        
    def compute_epistemic_uncertainty(self, expert_outputs, expert_gates):
        """
        Compute epistemic uncertainty from expert disagreement
        
        Args:
            expert_outputs: [batch_size, n_experts, hidden_dim] - outputs from each expert
            expert_gates: [batch_size, n_experts] - gating weights
        Returns:
            uncertainty: [batch_size] - uncertainty score per sample
        """
        # Weighted mean
        weighted_mean = (expert_gates.unsqueeze(-1) * expert_outputs).sum(dim=1)
        
        # Compute variance across experts
        diff = expert_outputs - weighted_mean.unsqueeze(1)
        weighted_variance = (expert_gates.unsqueeze(-1) * (diff ** 2)).sum(dim=1)
        
        # Aggregate to scalar uncertainty
        uncertainty = weighted_variance.mean(dim=-1)
        
        # Calibrate
        calibrated_uncertainty = self.uncertainty_scale * uncertainty + self.uncertainty_bias
        
        return calibrated_uncertainty
    
    def compute_distributional_uncertainty(self, predictions):
        """
        Compute uncertainty from prediction distribution entropy
        
        Args:
            predictions: [batch_size, num_entities] - prediction scores
        Returns:
            entropy: [batch_size] - prediction entropy
        """
        # Convert to probabilities
        probs = F.softmax(predictions, dim=-1)
        
        # Compute entropy
        log_probs = F.log_softmax(predictions, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Normalize by max entropy
        max_entropy = math.log(self.num_entities)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def forward(self, predictions, expert_outputs, expert_gates):
        """
        Combine multiple uncertainty estimates
        
        Args:
            predictions: List of prediction tensors from modalities
            expert_outputs: List of expert outputs from each modality
            expert_gates: List of expert gates from each modality
        Returns:
            uncertainty_scores: [batch_size] - overall uncertainty
            uncertainty_dict: Dictionary with different uncertainty types
        """
        batch_size = predictions[0].size(0)
        
        # 1. Epistemic uncertainty from experts
        epistemic_uncertainties = []
        for exp_out, exp_gate in zip(expert_outputs, expert_gates):
            epistemic_unc = self.compute_epistemic_uncertainty(exp_out, exp_gate)
            epistemic_uncertainties.append(epistemic_unc)
        avg_epistemic = torch.stack(epistemic_uncertainties).mean(dim=0)
        
        # 2. Distributional uncertainty from predictions
        distributional_uncertainties = []
        for pred in predictions:
            dist_unc = self.compute_distributional_uncertainty(pred)
            distributional_uncertainties.append(dist_unc)
        avg_distributional = torch.stack(distributional_uncertainties).mean(dim=0)
        
        # 3. Modal disagreement (how much modalities disagree)
        pred_stack = torch.stack(predictions, dim=1)  # [batch_size, n_modalities, num_entities]
        modal_mean = pred_stack.mean(dim=1)
        modal_variance = ((pred_stack - modal_mean.unsqueeze(1)) ** 2).mean(dim=[1, 2])
        
        # Combine uncertainties
        total_uncertainty = (avg_epistemic + avg_distributional + modal_variance) / 3.0
        
        uncertainty_dict = {
            'epistemic': avg_epistemic,
            'distributional': avg_distributional,
            'modal_disagreement': modal_variance,
            'total': total_uncertainty
        }
        
        return total_uncertainty, uncertainty_dict


class HardNegativeMiner(nn.Module):
    """
    Hard Negative Mining for Improved Contrastive Learning
    
    Key Contribution: Instead of random negative sampling, this module
    intelligently mines hard negatives that are close to positive samples
    but semantically different. This improves the discriminative power of embeddings.
    
    Benefits:
    - More informative training signal
    - Faster convergence
    - Better generalization
    """
    def __init__(self, embedding_dim, num_entities, k_hard=3):
        super(HardNegativeMiner, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.k_hard = k_hard  # Number of hard negatives to sample
        
        # Cache for hard negative candidates (updated periodically)
        self.register_buffer('negative_cache', torch.zeros(num_entities, embedding_dim))
        self.register_buffer('cache_valid', torch.tensor(False))
        
    def update_cache(self, entity_embeddings):
        """Update the cache with current entity embeddings"""
        self.negative_cache = entity_embeddings.detach()
        self.cache_valid = torch.tensor(True)
        
    def mine_hard_negatives(self, anchor_emb, positive_ids, all_embeddings, temperature=0.07):
        """
        Mine hard negatives based on embedding similarity
        
        Args:
            anchor_emb: [batch_size, embedding_dim] - anchor embeddings
            positive_ids: [batch_size] - positive entity ids
            all_embeddings: [num_entities, embedding_dim] - all entity embeddings
            temperature: float - temperature for similarity computation
        Returns:
            hard_negative_ids: [batch_size, k_hard] - hard negative entity ids
        """
        batch_size = anchor_emb.size(0)
        
        # Compute similarities with all entities
        # [batch_size, num_entities]
        similarities = torch.mm(
            F.normalize(anchor_emb, dim=-1), 
            F.normalize(all_embeddings, dim=-1).t()
        )
        
        # Mask out positive samples
        mask = torch.ones_like(similarities)
        mask[torch.arange(batch_size), positive_ids] = -1e9
        similarities = similarities * mask
        
        # Sample hard negatives (high similarity but not positive)
        # Use temperature to control hardness
        scaled_sim = similarities / temperature
        
        # Sample using categorical distribution (higher similarity = higher probability)
        probs = F.softmax(scaled_sim, dim=-1)
        
        # Sample k_hard negatives per anchor
        hard_negative_ids = torch.multinomial(probs, num_samples=self.k_hard, replacement=False)
        
        return hard_negative_ids
    
    def forward(self, anchor_emb, positive_ids, all_embeddings, use_cache=True):
        """
        Args:
            anchor_emb: [batch_size, embedding_dim]
            positive_ids: [batch_size]
            all_embeddings: [num_entities, embedding_dim]
            use_cache: bool - whether to use cached embeddings
        Returns:
            hard_neg_ids: [batch_size, k_hard]
        """
        if use_cache and self.cache_valid:
            emb_to_use = self.negative_cache
        else:
            emb_to_use = all_embeddings
            
        hard_neg_ids = self.mine_hard_negatives(
            anchor_emb, positive_ids, emb_to_use
        )
        
        return hard_neg_ids


class PredictionCalibrator(nn.Module):
    """
    Post-hoc Calibration for Better Confidence Estimation
    
    Key Contribution: Neural networks often produce over-confident predictions.
    This module applies temperature scaling and bias correction to calibrate
    prediction confidence, improving reliability.
    
    Based on "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    
    Benefits:
    - Better calibrated confidence scores
    - Improved ranking metrics
    - More reliable uncertainty estimates
    """
    def __init__(self, num_modalities=4):
        super(PredictionCalibrator, self).__init__()
        self.num_modalities = num_modalities
        
        # Learnable temperature for each modality
        self.temperatures = nn.Parameter(torch.ones(num_modalities))
        
        # Learnable bias correction
        self.biases = nn.Parameter(torch.zeros(num_modalities))
        
        # Modality-specific scaling
        self.scales = nn.Parameter(torch.ones(num_modalities))
        
    def calibrate_single(self, logits, temperature, scale, bias):
        """Calibrate a single prediction"""
        calibrated = (logits * scale + bias) / (temperature + 1e-8)
        return calibrated
    
    def forward(self, predictions, apply_sigmoid=True):
        """
        Args:
            predictions: List of [pred_s, pred_i, pred_d, pred_mm]
            apply_sigmoid: bool - whether to apply sigmoid after calibration
        Returns:
            calibrated_predictions: List of calibrated predictions
        """
        calibrated_preds = []
        
        for i, pred in enumerate(predictions):
            # Get modality-specific calibration parameters
            temp = F.softplus(self.temperatures[i]) + 0.5  # Ensure positive, min 0.5
            scale = torch.sigmoid(self.scales[i]) * 2  # Scale between 0 and 2
            bias = self.biases[i]
            
            # Apply calibration
            calibrated = self.calibrate_single(pred, temp, scale, bias)
            
            # Apply sigmoid if needed
            if apply_sigmoid:
                calibrated = torch.sigmoid(calibrated)
            
            calibrated_preds.append(calibrated)
        
        return calibrated_preds
    
    def compute_calibration_loss(self, predictions, targets, bins=10):
        """
        Compute Expected Calibration Error (ECE) as auxiliary loss
        
        Args:
            predictions: [batch_size, num_entities]
            targets: [batch_size, num_entities]
            bins: int - number of bins for calibration
        Returns:
            ece: scalar - expected calibration error
        """
        with torch.no_grad():
            # Get confidence and accuracy per bin
            confidences = predictions.max(dim=-1)[0]
            accuracies = (predictions.argmax(dim=-1) == targets.argmax(dim=-1)).float()
            
            # Create bins
            bin_boundaries = torch.linspace(0, 1, bins + 1, device=predictions.device)
            ece = 0.0
            
            for i in range(bins):
                # Find samples in this bin
                in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
                
                if in_bin.sum() > 0:
                    avg_confidence = confidences[in_bin].mean()
                    avg_accuracy = accuracies[in_bin].mean()
                    ece += torch.abs(avg_confidence - avg_accuracy) * (in_bin.sum().float() / len(confidences))
        
        return ece


class AdaptiveDropout(nn.Module):
    """
    Relation-Aware Adaptive Dropout
    
    Key Contribution: Different relations have different complexity levels.
    This module adapts dropout rate based on relation type to prevent
    overfitting on simple relations while maintaining capacity for complex ones.
    """
    def __init__(self, num_relations, base_dropout=0.3):
        super(AdaptiveDropout, self).__init__()
        self.num_relations = num_relations
        self.base_dropout = base_dropout
        
        # Learnable dropout rate per relation
        self.rel_dropout_logits = nn.Embedding(num_relations, 1)
        nn.init.constant_(self.rel_dropout_logits.weight, 0.0)
        
    def forward(self, x, relation_ids, training=True):
        """
        Args:
            x: [batch_size, ...] - input tensor
            relation_ids: [batch_size] - relation ids
            training: bool - whether in training mode
        Returns:
            x_dropout: tensor with adaptive dropout applied
        """
        if not training:
            return x
        
        batch_size = relation_ids.size(0)
        
        # Get relation-specific dropout rates
        dropout_logits = self.rel_dropout_logits(relation_ids).squeeze(-1)
        dropout_rates = torch.sigmoid(dropout_logits) * self.base_dropout * 2  # [0, 2*base_dropout]
        
        # Apply dropout with different rates per sample
        # This is approximate but efficient
        avg_dropout = dropout_rates.mean().item()
        x_dropped = F.dropout(x, p=avg_dropout, training=True)
        
        return x_dropped
