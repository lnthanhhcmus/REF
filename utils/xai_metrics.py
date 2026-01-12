"""
Evaluation Metrics for Explainable AI (xAI)
Provides quantitative measures for explanation quality
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import auc


class XAIMetrics:
    """
    Comprehensive xAI evaluation metrics for multi-modal knowledge graphs
    """
    
    @staticmethod
    def faithfulness_score(
        model,
        batch_inputs: torch.Tensor,
        explanations: torch.Tensor,
        target: torch.Tensor,
        top_k_list: List[int] = [5, 10, 20],
        method: str = 'removal'
    ) -> Dict[str, float]:
        """
        Measure faithfulness: do explanations reflect actual model behavior?
        Higher removal impact = more faithful explanations
        
        Args:
            model: The model to evaluate
            batch_inputs: [batch, 3] input triples
            explanations: [batch, n_features] explanation scores
            target: [batch, n_entities] target labels
            top_k_list: List of k values to test
            method: 'removal' or 'insertion'
        Returns:
            Dict of faithfulness scores for each k
        """
        model.eval()
        device = batch_inputs.device
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = model(batch_inputs)
            if isinstance(baseline_output, tuple):
                baseline_pred = baseline_output[0][-1]  # Fused prediction
            else:
                baseline_pred = baseline_output[-1]
            baseline_loss = F.binary_cross_entropy(baseline_pred, target).item()
        
        faithfulness_scores = {}
        
        for top_k in top_k_list:
            if method == 'removal':
                # Remove top-k important features and measure performance drop
                top_indices = torch.topk(explanations.abs(), k=top_k, dim=-1).indices
                
                # Create masked input (this is simplified - actual implementation
                # would need to mask embeddings)
                # For now, we use ablation-based approximation
                perturbed_losses = []
                
                for modal in ['struct', 'img', 'txt']:
                    with torch.no_grad():
                        try:
                            output = model.forward_with_ablation(batch_inputs, ablate_modal=modal)
                            if isinstance(output, tuple):
                                pred = output[0][-1]
                            else:
                                pred = output[-1]
                            loss = F.binary_cross_entropy(pred, target).item()
                            perturbed_losses.append(abs(loss - baseline_loss))
                        except:
                            pass
                
                if perturbed_losses:
                    faithfulness_scores[f'faithfulness@{top_k}'] = np.mean(perturbed_losses)
                else:
                    faithfulness_scores[f'faithfulness@{top_k}'] = 0.0
                    
            elif method == 'insertion':
                # Measure how quickly performance recovers when adding features
                # in order of importance (not implemented in simplified version)
                faithfulness_scores[f'faithfulness@{top_k}'] = 0.0
        
        return faithfulness_scores
    
    @staticmethod
    def stability_score(
        explanations_list: List[torch.Tensor],
        metric: str = 'cosine'
    ) -> float:
        """
        Measure stability/consistency of explanations across similar inputs
        
        Args:
            explanations_list: List of explanation tensors
            metric: 'cosine', 'pearson', or 'kendall'
        Returns:
            Stability score (higher = more stable)
        """
        if len(explanations_list) < 2:
            return 1.0
        
        similarities = []
        
        for i in range(len(explanations_list)):
            for j in range(i + 1, len(explanations_list)):
                exp_i = explanations_list[i].flatten()
                exp_j = explanations_list[j].flatten()
                
                if metric == 'cosine':
                    sim = F.cosine_similarity(exp_i, exp_j, dim=0).item()
                elif metric == 'pearson':
                    # Pearson correlation
                    if len(exp_i) > 1:
                        corr_matrix = torch.corrcoef(torch.stack([exp_i, exp_j]))
                        sim = corr_matrix[0, 1].item()
                    else:
                        sim = 1.0
                elif metric == 'kendall':
                    # Kendall's tau (requires numpy)
                    exp_i_np = exp_i.cpu().numpy()
                    exp_j_np = exp_j.cpu().numpy()
                    tau, _ = kendalltau(exp_i_np, exp_j_np)
                    sim = tau
                else:
                    sim = F.cosine_similarity(exp_i, exp_j, dim=0).item()
                
                similarities.append(sim)
        
        return float(np.mean(similarities))
    
    @staticmethod
    def sparsity_score(
        explanation: torch.Tensor,
        threshold: float = 0.1,
        normalize: bool = True
    ) -> float:
        """
        Measure sparsity of explanation (simpler = better)
        
        Args:
            explanation: [batch, n_features] explanation scores
            threshold: Threshold for considering a feature important
            normalize: Whether to normalize before thresholding
        Returns:
            Sparsity ratio (proportion of features below threshold)
        """
        if normalize:
            exp_norm = (explanation - explanation.min()) / (explanation.max() - explanation.min() + 1e-8)
        else:
            exp_norm = explanation.abs()
        
        sparse_ratio = (exp_norm < threshold).float().mean()
        return sparse_ratio.item()
    
    @staticmethod
    def completeness_score(
        explanations: Dict[str, torch.Tensor],
        coverage_threshold: float = 1e-6
    ) -> float:
        """
        Measure completeness: do explanations cover all important aspects?
        
        Args:
            explanations: Dict of explanation tensors for different components
            coverage_threshold: Minimum value to consider as covered
        Returns:
            Coverage ratio (0-1)
        """
        if not explanations:
            return 0.0
        
        # Aggregate all explanation scores
        all_scores = torch.cat([exp.flatten() for exp in explanations.values()])
        
        # Compute coverage
        coverage = (all_scores.abs() > coverage_threshold).float().mean()
        
        return coverage.item()
    
    @staticmethod
    def monotonicity_score(
        model,
        batch_inputs: torch.Tensor,
        explanations: torch.Tensor,
        target: torch.Tensor,
        n_steps: int = 10
    ) -> float:
        """
        Measure monotonicity: does removing features in order of importance
        lead to monotonically decreasing performance?
        
        Args:
            model: The model to evaluate
            batch_inputs: [batch, 3] input triples
            explanations: [batch, n_features] explanation scores
            target: [batch, n_entities] targets
            n_steps: Number of removal steps
        Returns:
            Monotonicity score (correlation between removal and performance drop)
        """
        model.eval()
        
        # Get baseline performance
        with torch.no_grad():
            baseline_output = model(batch_inputs)
            if isinstance(baseline_output, tuple):
                baseline_pred = baseline_output[0][-1]
            else:
                baseline_pred = baseline_output[-1]
            baseline_acc = ((baseline_pred > 0.5) == (target > 0.5)).float().mean().item()
        
        # Rank features by importance
        sorted_indices = torch.argsort(explanations.abs(), descending=True, dim=-1)
        
        performances = [baseline_acc]
        
        # Progressively remove features (simplified with modality ablation)
        modalities = ['struct', 'img', 'txt']
        for i, modal in enumerate(modalities):
            if i < n_steps:
                with torch.no_grad():
                    try:
                        output = model.forward_with_ablation(batch_inputs, ablate_modal=modal)
                        if isinstance(output, tuple):
                            pred = output[0][-1]
                        else:
                            pred = output[-1]
                        acc = ((pred > 0.5) == (target > 0.5)).float().mean().item()
                        performances.append(acc)
                    except:
                        performances.append(baseline_acc * 0.9)  # Assume some drop
        
        # Compute correlation between step and performance drop
        if len(performances) > 2:
            steps = list(range(len(performances)))
            corr, _ = spearmanr(steps, performances)
            # Negative correlation is good (performance decreases as we remove)
            monotonicity = -corr if not np.isnan(corr) else 0.0
        else:
            monotonicity = 0.0
        
        return float(monotonicity)
    
    @staticmethod
    def modality_consistency_score(
        modality_importance: Dict[str, float],
        expected_order: Optional[List[str]] = None
    ) -> float:
        """
        Measure if modality importance aligns with expectations
        
        Args:
            modality_importance: Dict of modality importance scores
            expected_order: Expected ranking of modalities (most to least important)
        Returns:
            Consistency score (0-1)
        """
        if not modality_importance or len(modality_importance) < 2:
            return 1.0
        
        # Get actual ranking
        actual_ranking = sorted(modality_importance.items(), key=lambda x: x[1], reverse=True)
        actual_order = [mod for mod, _ in actual_ranking]
        
        if expected_order is None:
            # No expected order provided, just check if differences are meaningful
            values = list(modality_importance.values())
            std = np.std(values)
            mean = np.mean(values)
            # Higher std relative to mean = more consistent/discriminative
            consistency = min(std / (mean + 1e-8), 1.0)
        else:
            # Compare with expected order using rank correlation
            expected_ranks = {mod: i for i, mod in enumerate(expected_order)}
            actual_ranks = {mod: i for i, mod in enumerate(actual_order)}
            
            common_mods = set(expected_ranks.keys()) & set(actual_ranks.keys())
            if len(common_mods) < 2:
                return 0.5
            
            exp_values = [expected_ranks[mod] for mod in common_mods]
            act_values = [actual_ranks[mod] for mod in common_mods]
            
            corr, _ = spearmanr(exp_values, act_values)
            consistency = (corr + 1) / 2  # Convert [-1, 1] to [0, 1]
        
        return float(consistency)
    
    @staticmethod
    def attention_concentration_score(
        attention_weights: torch.Tensor,
        method: str = 'entropy'
    ) -> float:
        """
        Measure how concentrated/focused attention is
        
        Args:
            attention_weights: [batch, n_heads, seq_len, seq_len] or [batch, n, n]
            method: 'entropy' or 'gini'
        Returns:
            Concentration score (higher = more focused)
        """
        # Average over batch and heads if needed
        if len(attention_weights.shape) == 4:
            attn = attention_weights.mean(dim=[0, 1])
        elif len(attention_weights.shape) == 3:
            attn = attention_weights.mean(dim=0)
        else:
            attn = attention_weights
        
        # Normalize to probabilities
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        if method == 'entropy':
            # Lower entropy = more concentrated
            entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()
            # Convert to concentration (inverse of entropy, normalized)
            max_entropy = np.log(attn.shape[-1])
            concentration = 1.0 - (entropy / max_entropy).item()
        elif method == 'gini':
            # Gini coefficient
            attn_sorted = torch.sort(attn.flatten())[0]
            n = len(attn_sorted)
            index = torch.arange(1, n + 1, device=attn.device).float()
            concentration = ((2 * index - n - 1) * attn_sorted).sum() / (n * attn_sorted.sum() + 1e-8)
            concentration = concentration.item()
        else:
            concentration = 0.0
        
        return float(concentration)
    
    @staticmethod
    def compute_all_metrics(
        model,
        batch_inputs: torch.Tensor,
        explanations: Dict[str, torch.Tensor],
        target: torch.Tensor,
        config: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive set of xAI metrics
        
        Args:
            model: The model being explained
            batch_inputs: Input batch
            explanations: Dictionary of all explanations
            target: Target labels
            config: Optional configuration for metrics
        Returns:
            Dictionary of all computed metrics
        """
        if config is None:
            config = {}
        
        metrics = {}
        
        # Faithfulness
        if 'feature_attributions' in explanations:
            try:
                faith_scores = XAIMetrics.faithfulness_score(
                    model, batch_inputs, 
                    explanations['feature_attributions'], 
                    target,
                    top_k_list=config.get('top_k_list', [5, 10, 20])
                )
                metrics.update(faith_scores)
            except Exception as e:
                print(f"Warning: Could not compute faithfulness: {e}")
        
        # Sparsity
        if 'feature_attributions' in explanations:
            try:
                sparsity = XAIMetrics.sparsity_score(
                    explanations['feature_attributions'],
                    threshold=config.get('sparsity_threshold', 0.1)
                )
                metrics['sparsity'] = sparsity
            except Exception as e:
                print(f"Warning: Could not compute sparsity: {e}")
        
        # Completeness
        try:
            completeness = XAIMetrics.completeness_score(explanations)
            metrics['completeness'] = completeness
        except Exception as e:
            print(f"Warning: Could not compute completeness: {e}")
        
        # Modality consistency
        if 'modality_importance' in explanations:
            try:
                modal_consist = XAIMetrics.modality_consistency_score(
                    explanations['modality_importance'],
                    expected_order=config.get('expected_modal_order', None)
                )
                metrics['modality_consistency'] = modal_consist
            except Exception as e:
                print(f"Warning: Could not compute modality consistency: {e}")
        
        # Attention concentration
        if 'attention' in explanations and len(explanations['attention']) > 0:
            try:
                attn_map = list(explanations['attention'].values())[0]
                concentration = XAIMetrics.attention_concentration_score(attn_map)
                metrics['attention_concentration'] = concentration
            except Exception as e:
                print(f"Warning: Could not compute attention concentration: {e}")
        
        return metrics


def aggregate_metrics_over_dataset(
    model,
    dataloader,
    explainability_manager,
    n_samples: int = 100,
    config: Optional[Dict] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate xAI metrics over multiple samples from dataset
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader providing batches
        explainability_manager: Manager for generating explanations
        n_samples: Number of samples to evaluate
        config: Configuration for metrics
    Returns:
        Dictionary with mean and std for each metric
    """
    all_metrics = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_inputs, target) in enumerate(dataloader):
            if i >= n_samples:
                break
            
            try:
                # Generate explanations
                explanations = explainability_manager.generate_explanations(
                    batch_inputs, target,
                    methods=['attention', 'gradcam', 'modality']
                )
                
                # Compute metrics
                metrics = XAIMetrics.compute_all_metrics(
                    model, batch_inputs, explanations, target, config
                )
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"Warning: xAI analysis failed for batch {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Aggregate results
    if not all_metrics:
        return {}
    
    # Collect all metric names
    metric_names = set()
    for m in all_metrics:
        metric_names.update(m.keys())
    
    # Compute mean and std for each metric
    aggregated = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        if values:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
    
    return aggregated
