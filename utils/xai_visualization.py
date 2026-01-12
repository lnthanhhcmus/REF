"""
Visualization utilities for Explainable AI (xAI) outputs
Provides plotting and analysis tools for interpretability
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json
import os


class ExplanationVisualizer:
    """
    Visualize xAI explanations in various formats
    """
    def __init__(self, entity2id: Dict, relation2id: Dict, save_dir: str = 'explanations'):
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        modality_names: List[str] = ['Struct', 'Image', 'Text'],
        save_name: Optional[str] = None,
        title: str = 'Cross-Modal Attention Weights'
    ):
        """
        Plot attention weights as heatmap
        Args:
            attention_weights: [batch, n_modalities, n_modalities] or [n_modalities, n_modalities]
            modality_names: Names of modalities
            save_name: Filename to save plot
        """
        # Average over batch if needed
        if len(attention_weights.shape) == 3:
            attn = attention_weights.mean(dim=0).cpu().numpy()
        else:
            attn = attention_weights.cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            attn,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=modality_names,
            yticklabels=modality_names,
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.title(title)
        plt.xlabel('Key Modality')
        plt.ylabel('Query Modality')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_modality_importance(
        self,
        importance_scores: Dict[str, float],
        save_name: Optional[str] = None,
        title: str = 'Modality Importance Scores'
    ):
        """
        Plot bar chart of modality importance
        Args:
            importance_scores: Dict mapping modality names to importance scores
            save_name: Filename to save plot
        """
        modalities = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        # Create color palette
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(modalities)]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(modalities, scores, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Modality', fontsize=12)
        plt.ylabel('Importance Score (Loss Increase)', fontsize=12)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_gradcam_overlay(
        self,
        gradcam_maps: Dict[str, torch.Tensor],
        save_name: Optional[str] = None,
        title: str = 'GradCAM Activation Maps'
    ):
        """
        Visualize GradCAM activation maps for different modalities
        Args:
            gradcam_maps: Dict mapping layer names to CAM tensors
            save_name: Filename to save plot
        """
        n_maps = len(gradcam_maps)
        if n_maps == 0:
            return
        
        fig, axes = plt.subplots(1, n_maps, figsize=(5 * n_maps, 4))
        if n_maps == 1:
            axes = [axes]
        
        for idx, (layer_name, cam_tensor) in enumerate(gradcam_maps.items()):
            # Average over batch
            cam = cam_tensor.mean(dim=0).cpu().numpy()
            
            # Reshape if 1D
            if len(cam.shape) == 1:
                size = int(np.sqrt(len(cam)))
                if size * size == len(cam):
                    cam = cam.reshape(size, size)
                else:
                    # Pad to square
                    size = int(np.ceil(np.sqrt(len(cam))))
                    padded = np.zeros(size * size)
                    padded[:len(cam)] = cam
                    cam = padded.reshape(size, size)
            
            im = axes[idx].imshow(cam, cmap='jet', interpolation='bilinear')
            axes[idx].set_title(layer_name.replace('_', ' ').title())
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_attribution(
        self,
        attributions: torch.Tensor,
        top_k: int = 20,
        feature_names: Optional[List[str]] = None,
        save_name: Optional[str] = None,
        title: str = 'Top Feature Attributions'
    ):
        """
        Plot top-k feature attributions
        Args:
            attributions: [n_features] attribution scores
            top_k: Number of top features to show
            feature_names: Optional names for features
            save_name: Filename to save plot
        """
        # Get top-k features
        attr_np = attributions.cpu().numpy().flatten()
        top_indices = np.argsort(np.abs(attr_np))[-top_k:][::-1]
        top_values = attr_np[top_indices]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in top_indices]
        else:
            feature_names = [feature_names[i] for i in top_indices]
        
        # Color by positive/negative
        colors = ['#2ECC71' if v > 0 else '#E74C3C' for v in top_values]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_k), top_values, color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(range(top_k), feature_names)
        plt.xlabel('Attribution Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_triple_explanation(
        self,
        triple: Tuple[int, int, int],
        explanations: Dict,
        save_name: Optional[str] = None
    ):
        """
        Create comprehensive explanation visualization for a single triple
        Args:
            triple: (head, relation, tail) entity IDs
            explanations: Dictionary of explanation outputs
            save_name: Filename to save plot
        """
        head_id, rel_id, tail_id = triple
        head_name = self.id2entity.get(head_id, f'Entity_{head_id}')
        rel_name = self.id2relation.get(rel_id, f'Relation_{rel_id}')
        tail_name = self.id2entity.get(tail_id, f'Entity_{tail_id}')
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(
            f'Explanation for Triple: ({head_name}, {rel_name}, {tail_name})',
            fontsize=16,
            fontweight='bold'
        )
        
        # Panel 1: Modality Importance (if available)
        if 'modality_importance' in explanations:
            ax1 = fig.add_subplot(gs[0, :])
            modal_imp = explanations['modality_importance']
            modalities = list(modal_imp.keys())
            scores = list(modal_imp.values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(modalities)]
            ax1.bar(modalities, scores, color=colors, alpha=0.8, edgecolor='black')
            ax1.set_title('Modality Importance (Ablation Study)', fontweight='bold')
            ax1.set_ylabel('Impact Score')
            ax1.grid(axis='y', alpha=0.3)
        
        # Panel 2-4: GradCAM maps (if available)
        if 'gradcam' in explanations and len(explanations['gradcam']) > 0:
            gradcam_maps = explanations['gradcam']
            for idx, (layer_name, cam_tensor) in enumerate(list(gradcam_maps.items())[:3]):
                ax = fig.add_subplot(gs[1, idx])
                cam = cam_tensor.mean(dim=0).cpu().numpy()
                
                # Reshape to 2D if needed
                if len(cam.shape) == 1:
                    size = int(np.ceil(np.sqrt(len(cam))))
                    padded = np.zeros(size * size)
                    padded[:len(cam)] = cam
                    cam = padded.reshape(size, size)
                
                im = ax.imshow(cam, cmap='jet', interpolation='bilinear')
                ax.set_title(f'{layer_name.replace("_", " ").title()}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Panel 5: Attention weights (if available)
        if 'attention' in explanations and len(explanations['attention']) > 0:
            ax5 = fig.add_subplot(gs[2, :2])
            # Get first attention map
            attn_map = list(explanations['attention'].values())[0]
            if len(attn_map.shape) == 3:
                attn_map = attn_map.mean(dim=0)
            
            sns.heatmap(
                attn_map.cpu().numpy(),
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                ax=ax5,
                cbar_kws={'label': 'Attention'}
            )
            ax5.set_title('Cross-Modal Attention Pattern', fontweight='bold')
        
        # Panel 6: Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        summary_text = "Summary Statistics:\n\n"
        if 'modality_importance' in explanations:
            modal_imp = explanations['modality_importance']
            most_important = max(modal_imp, key=modal_imp.get)
            summary_text += f"Most Important Modality:\n  {most_important}\n\n"
        
        if 'gradcam' in explanations:
            summary_text += f"GradCAM Layers Analyzed:\n  {len(explanations['gradcam'])}\n\n"
        
        if 'attention' in explanations:
            summary_text += f"Attention Maps Captured:\n  {len(explanations['attention'])}\n"
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_explanations_json(
        self,
        triple: Tuple[int, int, int],
        explanations: Dict,
        filename: str
    ):
        """
        Save explanations to JSON file for later analysis
        Args:
            triple: (head, relation, tail) entity IDs
            explanations: Dictionary of explanations
            filename: Output JSON filename
        """
        head_id, rel_id, tail_id = triple
        
        export_data = {
            'triple': {
                'head_id': int(head_id),
                'relation_id': int(rel_id),
                'tail_id': int(tail_id),
                'head_name': self.id2entity.get(head_id, f'Entity_{head_id}'),
                'relation_name': self.id2relation.get(rel_id, f'Relation_{rel_id}'),
                'tail_name': self.id2entity.get(tail_id, f'Entity_{tail_id}')
            },
            'explanations': {}
        }
        
        # Convert tensors to lists for JSON serialization
        for key, value in explanations.items():
            if isinstance(value, dict):
                export_data['explanations'][key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        export_data['explanations'][key][sub_key] = sub_value.cpu().tolist()
                    else:
                        export_data['explanations'][key][sub_key] = sub_value
            elif isinstance(value, torch.Tensor):
                export_data['explanations'][key] = value.cpu().tolist()
            else:
                export_data['explanations'][key] = value
        
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Explanations saved to: {filepath}")


class ExplanationComparer:
    """
    Compare explanations across different models or configurations
    """
    def __init__(self, save_dir: str = 'comparisons'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def compare_modality_importance(
        self,
        explanations_list: List[Dict],
        model_names: List[str],
        save_name: Optional[str] = None
    ):
        """
        Compare modality importance across multiple models
        Args:
            explanations_list: List of explanation dicts from different models
            model_names: Names of models
            save_name: Filename to save comparison plot
        """
        # Extract modality importance from each model
        modalities = list(explanations_list[0]['modality_importance'].keys())
        n_models = len(model_names)
        
        # Prepare data
        data = {modal: [] for modal in modalities}
        for exp in explanations_list:
            for modal in modalities:
                data[modal].append(exp['modality_importance'].get(modal, 0))
        
        # Plot grouped bar chart
        x = np.arange(len(modalities))
        width = 0.8 / n_models
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        for i, model_name in enumerate(model_names):
            values = [data[modal][i] for modal in modalities]
            ax.bar(x + i * width, values, width, label=model_name, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Modality', fontsize=12)
        ax.set_ylabel('Importance Score', fontsize=12)
        ax.set_title('Modality Importance Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(modalities)
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

