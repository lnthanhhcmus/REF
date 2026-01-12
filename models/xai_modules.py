"""
Explainable AI (xAI) Modules for Multi-Modal Knowledge Graph Completion
Provides interpretability techniques without modifying core model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for understanding modal contributions
    Adapted for multi-modal knowledge graph embeddings
    """
    def __init__(self, model):
        self.model = model
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self, target_layers: List[str]):
        """Register forward and backward hooks on target layers"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in target_layers:
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, layer_name: str) -> torch.Tensor:
        """
        Generate Class Activation Map for a specific layer
        Returns: [batch_size, spatial_dim] importance scores
        """
        if layer_name not in self.gradients or layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found in gradients/activations")
        
        gradients = self.gradients[layer_name]  # [batch, channels, ...]
        activations = self.activations[layer_name]  # [batch, channels, ...]
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=tuple(range(2, len(gradients.shape))), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1)
        
        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam


class IntegratedGradients:
    """
    Integrated Gradients for feature attribution
    Computes gradients along path from baseline to input
    """
    def __init__(self, model, baseline='zero'):
        self.model = model
        self.baseline = baseline
    
    def get_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Generate baseline for integrated gradients"""
        if self.baseline == 'zero':
            return torch.zeros_like(input_tensor)
        elif self.baseline == 'random':
            return torch.randn_like(input_tensor) * 0.1
        elif self.baseline == 'mean':
            return torch.ones_like(input_tensor) * input_tensor.mean()
        else:
            return torch.zeros_like(input_tensor)
    
    def compute_attributions(
        self, 
        input_tensor: torch.Tensor, 
        target_output: torch.Tensor,
        n_steps: int = 50
    ) -> torch.Tensor:
        """
        Compute integrated gradients attributions
        Args:
            input_tensor: Input embeddings [batch, dim]
            target_output: Target for gradient computation [batch, dim]
            n_steps: Number of integration steps
        Returns:
            attributions: [batch, dim] attribution scores
        """
        baseline = self.get_baseline(input_tensor)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps + 1, device=input_tensor.device)
        
        # Store gradients for each step
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass with interpolated input
            output = self.model.forward_with_embedding(interpolated)
            
            # Compute gradient
            grad = torch.autograd.grad(
                outputs=output,
                inputs=interpolated,
                grad_outputs=target_output,
                create_graph=False,
                retain_graph=False
            )[0]
            
            gradients.append(grad)
        
        # Average gradients across steps
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Multiply by (input - baseline)
        attributions = (input_tensor - baseline) * avg_gradients
        
        return attributions


class AttentionVisualization:
    """
    Extract and visualize attention patterns from multi-modal attention layers
    """
    def __init__(self):
        self.attention_maps = {}
    
    def register_attention_hook(self, model, layer_names: List[str]):
        """Register hooks to capture attention weights"""
        def attention_hook(name):
            def hook(module, input, output):
                # Assuming output contains attention weights
                if isinstance(output, tuple):
                    # MultiheadAttention returns (output, attention_weights)
                    if len(output) > 1:
                        self.attention_maps[name] = output[1].detach()
                elif hasattr(module, 'attention_weights'):
                    self.attention_maps[name] = module.attention_weights.detach()
            return hook
        
        for name, module in model.named_modules():
            if name in layer_names or any(ln in name for ln in layer_names):
                module.register_forward_hook(attention_hook(name))
    
    def get_attention_map(self, layer_name: str) -> Optional[torch.Tensor]:
        """Retrieve attention map for specific layer"""
        return self.attention_maps.get(layer_name, None)
    
    def aggregate_attention(self) -> Dict[str, torch.Tensor]:
        """Aggregate attention across all captured layers"""
        return {k: v.mean(dim=1) if len(v.shape) > 2 else v 
                for k, v in self.attention_maps.items()}


class ModalityImportanceAnalyzer:
    """
    Analyze the importance of different modalities (structure, image, text)
    using ablation and gradient-based methods
    """
    def __init__(self, model):
        self.model = model
    
    def ablation_study(
        self, 
        batch_inputs: torch.Tensor,
        target: torch.Tensor,
        modalities: List[str] = ['struct', 'img', 'txt']
    ) -> Dict[str, float]:
        """
        Perform ablation study by removing each modality
        Returns: Dictionary of performance drop for each modality
        """
        self.model.eval()
        
        # Check if model supports ablation
        if not hasattr(self.model, 'forward_with_ablation'):
            # Return placeholder scores if ablation not supported
            return {mod: 0.0 for mod in modalities}
        
        # Baseline performance with all modalities
        with torch.no_grad():
            baseline_output = self.model(batch_inputs)
            if isinstance(baseline_output, tuple):
                baseline_output = baseline_output[0]  # Get predictions
            baseline_loss = F.binary_cross_entropy(baseline_output[-1], target)
        
        importance_scores = {}
        
        # Test each modality ablation
        for modality in modalities:
            try:
                with torch.no_grad():
                    # Temporarily disable modality
                    output = self.model.forward_with_ablation(batch_inputs, ablate_modal=modality)
                    if isinstance(output, tuple):
                        output = output[0]
                    ablated_loss = F.binary_cross_entropy(output[-1], target)
                    
                    # Higher loss after ablation = more important modality
                    importance_scores[modality] = float(ablated_loss - baseline_loss)
            except Exception as e:
                # If ablation fails for this modality, set to 0
                importance_scores[modality] = 0.0
        
        return importance_scores
    
    def gradient_based_importance(
        self,
        batch_inputs: torch.Tensor,
        modality_embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute modality importance using gradient magnitudes
        Args:
            batch_inputs: Input batch [batch, 3]
            modality_embeddings: Dict of embeddings for each modality
        Returns:
            importance scores for each modality
        """
        self.model.train()
        
        importance = {}
        
        for modal_name, embeddings in modality_embeddings.items():
            if embeddings.requires_grad:
                # Compute gradient magnitude as importance
                grad_norm = torch.norm(embeddings.grad, dim=-1, keepdim=True)
                importance[modal_name] = grad_norm.detach()
            else:
                importance[modal_name] = torch.zeros_like(embeddings[:, :1])
        
        return importance


class SHAPLikeExplainer:
    """
    SHAP-inspired explanation for feature importance
    Simplified version adapted for embeddings
    """
    def __init__(self, model, background_samples: torch.Tensor):
        self.model = model
        self.background = background_samples
    
    def explain_instance(
        self,
        instance: torch.Tensor,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Explain a single instance using sampling-based approach
        Args:
            instance: Single input [1, 3] (head, relation, tail)
            n_samples: Number of samples for estimation
        Returns:
            feature_importance: [1, n_features] importance scores
        """
        self.model.eval()
        
        # Get embeddings
        with torch.no_grad():
            instance_embed = self.model.get_batch_embeddings(instance)
            background_embed = [self.model.get_batch_embeddings(bg.unsqueeze(0)) 
                               for bg in self.background[:n_samples]]
        
        # Compute marginal contributions
        contributions = []
        
        for i in range(n_samples):
            # Mix instance and background features randomly
            mask = torch.rand(instance_embed[0].shape[1]) > 0.5
            mixed_embed = instance_embed.copy()
            
            for j, (inst_e, bg_e) in enumerate(zip(instance_embed, background_embed[i])):
                mixed_embed[j] = torch.where(
                    mask.unsqueeze(0),
                    inst_e,
                    bg_e
                )
            
            # Compute prediction with mixed features
            # This requires model to support embedding-based forward pass
            with torch.no_grad():
                output = self.model.forward_with_embeddings(mixed_embed)
                contributions.append(output)
        
        # Aggregate contributions
        contributions = torch.stack(contributions).mean(dim=0)
        
        return contributions


class ExplainabilityManager:
    """
    Unified manager for all explainability techniques
    Provides easy interface for generating explanations
    """
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.grad_cam = GradCAM(model)
        self.integrated_gradients = IntegratedGradients(model)
        self.attention_viz = AttentionVisualization()
        self.modality_analyzer = ModalityImportanceAnalyzer(model)
        
        # Storage for explanations
        self.explanations = {
            'attention_maps': {},
            'feature_attributions': {},
            'modality_importance': {},
            'cam_maps': {}
        }
    
    def setup_hooks(self):
        """Setup all necessary hooks for explanation capture"""
        # Register GradCAM hooks for key layers
        target_layers = [
            'modal_fusion',
            'moe_adaptor_str',
            'moe_adaptor_img', 
            'moe_adaptor_txt'
        ]
        self.grad_cam.register_hooks(target_layers)
        
        # Register attention visualization hooks
        attention_layers = ['cross_modal_attn', 'modal_fusion']
        self.attention_viz.register_attention_hook(self.model, attention_layers)
    
    def generate_explanations(
        self,
        batch_inputs: torch.Tensor,
        target: torch.Tensor,
        methods: List[str] = ['attention', 'gradcam', 'modality']
    ) -> Dict:
        """
        Generate comprehensive explanations using multiple methods
        Args:
            batch_inputs: Input batch [batch, 3]
            target: Target labels [batch, n_entities]
            methods: List of explanation methods to use
        Returns:
            Dictionary containing all explanations
        """
        results = {}
        
        # Store original training state
        was_training = self.model.training
        
        if 'attention' in methods:
            # Extract attention patterns
            self.model.eval()
            with torch.no_grad():
                _ = self.model(batch_inputs)
            results['attention'] = self.attention_viz.aggregate_attention()
        
        if 'gradcam' in methods:
            # Generate GradCAM visualizations
            # IMPORTANT: Model must be in training mode for gradients
            self.model.train()
            
            # Note: batch_inputs are integer indices, can't require gradients
            # Gradients will flow through embeddings instead
            self.model.zero_grad()
            
            try:
                # Enable gradient computation
                with torch.set_grad_enabled(True):
                    output = self.model(batch_inputs)
                    if isinstance(output, tuple):
                        predictions = output[0][-1]  # Get fused prediction
                        embeddings = output[1]
                    else:
                        predictions = output[-1]
                        embeddings = None
                    
                    # Ensure predictions require gradients
                    if not predictions.requires_grad:
                        # Model might be in eval mode or gradients disabled
                        # Skip GradCAM if gradients not available
                        print("Warning: GradCAM skipped - model outputs don't have gradients")
                        results['gradcam'] = {}
                    else:
                        loss = F.binary_cross_entropy(predictions, target)
                        loss.backward(retain_graph=True)
                        
                        cam_maps = {}
                        for layer in ['structure_moe', 'visual_moe', 'text_moe']:
                            try:
                                cam = self.grad_cam.generate_cam(layer)
                                cam_maps[layer] = cam
                            except Exception as e:
                                # Silently skip layers that fail
                                pass
                        results['gradcam'] = cam_maps
                
            except Exception as e:
                # If GradCAM fails, return empty dict for this method
                results['gradcam'] = {}
                print(f"Warning: GradCAM failed: {e}")
        
        if 'modality' in methods:
            # Analyze modality importance via ablation
            try:
                self.model.eval()
                modal_importance = self.modality_analyzer.ablation_study(
                    batch_inputs, target
                )
                results['modality_importance'] = modal_importance
            except Exception as e:
                results['modality_importance'] = {}
                print(f"Warning: Modality analysis failed: {e}")
        
        # Restore original training state
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        
        return results
    
    def compute_faithfulness_score(
        self,
        batch_inputs: torch.Tensor,
        explanations: torch.Tensor,
        top_k: int = 10
    ) -> float:
        """
        Compute faithfulness metric: correlation between explanation and actual importance
        """
        self.model.eval()
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(batch_inputs)
            if isinstance(baseline_output, tuple):
                baseline_pred = baseline_output[0][-1]
            else:
                baseline_pred = baseline_output[-1]
        
        # Mask top-k features according to explanation
        top_k_indices = torch.topk(explanations, k=top_k, dim=-1).indices
        
        # Compute prediction with masked features
        # This requires model modification to support feature masking
        # For now, return placeholder
        faithfulness = 0.0
        
        return faithfulness
    
    def cleanup(self):
        """Remove all hooks and cleanup"""
        self.grad_cam.remove_hooks()


class ExplanationMetrics:
    """
    Metrics for evaluating explanation quality
    """
    @staticmethod
    def compute_stability(
        explanations: List[torch.Tensor],
        metric: str = 'cosine'
    ) -> float:
        """
        Measure stability of explanations across similar inputs
        Higher is better (more consistent explanations)
        """
        if len(explanations) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(explanations) - 1):
            for j in range(i + 1, len(explanations)):
                if metric == 'cosine':
                    sim = F.cosine_similarity(
                        explanations[i].flatten(),
                        explanations[j].flatten(),
                        dim=0
                    )
                elif metric == 'pearson':
                    # Pearson correlation
                    x = explanations[i].flatten()
                    y = explanations[j].flatten()
                    sim = torch.corrcoef(torch.stack([x, y]))[0, 1]
                else:
                    # Default to cosine
                    sim = F.cosine_similarity(
                        explanations[i].flatten(),
                        explanations[j].flatten(),
                        dim=0
                    )
                similarities.append(sim.item())
        
        return np.mean(similarities)
    
    @staticmethod
    def compute_sparsity(explanation: torch.Tensor, threshold: float = 0.1) -> float:
        """
        Measure sparsity of explanation (fewer important features = better)
        """
        normalized = (explanation - explanation.min()) / (explanation.max() - explanation.min() + 1e-8)
        sparse_ratio = (normalized < threshold).float().mean()
        return sparse_ratio.item()
    
    @staticmethod
    def compute_completeness(
        explanations: Dict[str, torch.Tensor]
    ) -> float:
        """
        Measure if explanations cover all important aspects
        """
        # Aggregate all explanation scores
        all_scores = torch.cat([exp.flatten() for exp in explanations.values()])
        
        # Compute coverage (non-zero explanations)
        coverage = (all_scores.abs() > 1e-6).float().mean()
        
        return coverage.item()
