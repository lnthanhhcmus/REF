import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from models.EnhancedRelMoE import EnhancedRelMoE
from models.model import *
from models.modules import MIEstimator
from utils.data_loader import *
from utils.data_util import load_data


def parse_args():
    config_args = {
        'lr': 0.0005,
        'dropout_gat': 0.3,
        'dropout': 0.3,
        'cuda': 0,
        'epochs_gat': 3000,
        'epochs': 2000,
        'weight_decay_gat': 1e-5,
        'weight_decay': 0,
        'seed': 10010,
        'model': 'EnhancedRelMoE',
        'num-layers': 3,
        'dim': 256,
        'r_dim': 256,
        'k_w': 10,
        'k_h': 20,
        'n_heads': 2,
        'dataset': 'DB15K',
        'pre_trained': 0,
        'encoder': 0,
        'image_features': 1,
        'text_features': 1,
        'patience': 5,
        'eval_freq': 100,
        'lr_reduce_freq': 500,
        'gamma': 1.0,
        'bias': 1,
        'neg_num': 2,
        'neg_num_gat': 2,
        'alpha': 0.2,
        'alpha_gat': 0.2,
        'out_channels': 32,
        'kernel_size': 3,
        'batch_size': 1024,
        'save': 1,
        'n_exp': 3,
        'mu': 0.0001,
        'img_dim': 256,
        'txt_dim': 256,
        # Enhanced model parameters
        'use_adaptive_temp': 1,
        'use_cross_modal_attn': 1,
        'consistency_weight': 0.1,
        # Score function space parameter
        'space': 'none',  # Options: None (TuckER), 'complex', 'quaternion'
        # xAI parameters (Explainable AI)
        'use_xai': 0,  # Enable explainability features (Not Implemented)
        'xai_freq': 200,  # Frequency of xAI analysis (epochs)
        'xai_n_samples': 10,  # Number of samples to explain
        'xai_save_vis': 1,  # Save xAI visualizations
        'xai_compute_metrics': 1,  # Compute xAI evaluation metrics
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        if param == 'space':
            parser.add_argument(f"--{param}", default=val, type=str)
        else:
            parser.add_argument(f"--{param}", default=val, type=type(val))
    args = parser.parse_args()
    return args

args = parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
torch.cuda.set_device(args.cuda)
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

entity2id, relation2id, img_features, text_features, train_data, val_data, test_data = load_data(args.dataset)
print("Training data {:04d}".format(len(train_data[0])))

corpus = ConvECorpus(args, train_data, val_data, test_data, entity2id, relation2id)

if args.image_features:
    args.img = F.normalize(torch.Tensor(img_features), p=2, dim=1)
if args.text_features:
    args.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)
args.entity2id = entity2id
args.relation2id = relation2id

model_name = {
    'EnhancedRelMoE': EnhancedRelMoE,
}
time.sleep(5)

def train_decoder(args):
    if args.model == "EnhancedRelMoE":
        print("Using Enhanced RelMoE with:")
        print(f"  - Adaptive Temperature: {args.use_adaptive_temp}")
        print(f"  - Cross-Modal Attention: {args.use_cross_modal_attn}")
        print(f"  - Consistency Weight: {args.consistency_weight}")
        if args.space:
            print(f"  - Score Function Space: {args.space}")
        if args.use_xai:
            print(f"  - xAI Enabled: Freq={args.xai_freq}, Samples={args.xai_n_samples}")

    model = model_name[args.model](args)
    
    # xAI: Setup visualization tools if enabled
    xai_visualizer = None
    if args.use_xai and args.xai_save_vis:
        from utils.xai_visualization import ExplanationVisualizer
        xai_visualizer = ExplanationVisualizer(
            entity2id=entity2id,
            relation2id=relation2id,
            save_dir=f'explanations/{args.dataset}/{args.model}'
        )
        print(f"xAI visualizations will be saved to: explanations/{args.dataset}/{args.model}")
    
    args.dim = model.dim
    args.img_dim = model.img_dim
    args.txt_dim = model.txt_dim
    estimator = MIEstimator(args)
    # print(str(model))
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    
    optimizer_mi = torch.optim.Adam(params=estimator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)
        estimator = estimator.to(args.device)

    # Train Model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = model.init_metric_dict()
    corpus.batch_size = args.batch_size
    corpus.neg_num = args.neg_num
    training_range = tqdm(range(args.epochs))
    for epoch in training_range:
        model.train()
        epoch_loss = []
        epoch_mi_loss = []
        t = time.time()
        corpus.shuffle()
        epoch_losses = []  # Store losses for curriculum learning
        
        for batch_num in range(corpus.max_batch_num):
            # Training the KGC model
            estimator.eval()
           
            optimizer.zero_grad()
            
            train_indices, train_values = corpus.get_batch(batch_num)
            train_indices = torch.LongTensor(train_indices)
            if args.cuda is not None and int(args.cuda) >= 0:
                train_indices = train_indices.to(args.device)
                train_values = train_values.to(args.device)
            
            # Forward pass
            if args.model == "UltraEnhancedRelMoE":
                output, embeddings, load_losses = model.forward(train_indices)
                loss = model.ultra_loss_func(output, train_values, embeddings, load_losses) + args.mu * estimator(embeddings)
            else:
                output, embeddings = model.forward(train_indices)
                loss = model.loss_func(output, train_values) + args.mu * estimator(embeddings)
            
            loss.backward()
            
            # Gradient clipping for stability
            if args.model == "UltraEnhancedRelMoE":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Store loss for curriculum learning
            epoch_losses.append(loss.detach())
            
            # Train the estimator
            estimator.train()
            optimizer_mi.zero_grad()
            with torch.no_grad():
                embeddings = model.get_batch_embeddings(train_indices)
            estimator_loss = estimator.train_estimator(embeddings)
            estimator_loss.backward()
            optimizer_mi.step()
            epoch_loss.append(loss.data.item())
            epoch_mi_loss.append(estimator_loss.item())
        
        training_range.set_postfix(loss="main: {:.5} mi: {:.5}".format(sum(epoch_loss), sum(epoch_mi_loss)))
        
        lr_scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            print("Epoch {:04d} , average loss {:.4f} , epoch_time {:.4f}\n".format(
                epoch + 1, sum(epoch_loss) / len(epoch_loss), time.time() - t))
            model.eval()
            with torch.no_grad():
                val_metrics, _ = corpus.get_validation_pred(model, 'test')
            if val_metrics['MRR'] > best_test_metrics['MRR']:
                best_test_metrics['MRR'] = val_metrics['MRR']
            if val_metrics['MR'] < best_test_metrics['MR']:
                best_test_metrics['MR'] = val_metrics['MR']
            if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
                best_test_metrics['Hits@1'] = val_metrics['Hits@1']
            if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
                best_test_metrics['Hits@3'] = val_metrics['Hits@3']
            if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
                best_test_metrics['Hits@10'] = val_metrics['Hits@10']
            if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
                best_test_metrics['Hits@100'] = val_metrics['Hits@100']
            print('\n'.join(['Epoch: {:04d}'.format(epoch + 1), model.format_metrics(val_metrics, 'test')]))
            print("\n\n")
            
            # xAI: Generate explanations periodically
            if args.use_xai and (epoch + 1) % args.xai_freq == 0:
                print(f"\n{'='*60}")
                print(f"Generating xAI Explanations at Epoch {epoch + 1}")
                print(f"{'='*60}\n")
                
                try:
                    # Get a sample batch for explanation
                    sample_indices, sample_values = corpus.get_batch(0)
                    sample_indices = torch.LongTensor(sample_indices[:args.xai_n_samples])
                    sample_values = sample_values[:args.xai_n_samples]
                    
                    if args.cuda is not None and int(args.cuda) >= 0:
                        sample_indices = sample_indices.to(args.device)
                        sample_values = sample_values.to(args.device)
                    
                    # Generate explanations
                    model.eval()
                    with torch.no_grad():
                        explanations = model.generate_explanations(
                            sample_indices,
                            sample_values,
                            methods=['attention', 'gradcam', 'modality']
                        )
                    
                    # Print modality importance
                    if 'modality_importance' in explanations:
                        print("\nModality Importance Scores:")
                        for modal, score in explanations['modality_importance'].items():
                            print(f"  {modal:8s}: {score:8.5f}")
                    
                    # Visualize if enabled
                    if xai_visualizer is not None and args.xai_save_vis:
                        # Save attention heatmap
                        if 'attention' in explanations and len(explanations['attention']) > 0:
                            for layer_name, attn_weights in explanations['attention'].items():
                                xai_visualizer.plot_attention_heatmap(
                                    attn_weights,
                                    save_name=f'attention_epoch{epoch+1}_{layer_name}.png',
                                    title=f'Attention Weights - Epoch {epoch+1}'
                                )
                        
                        # Save modality importance
                        if 'modality_importance' in explanations:
                            xai_visualizer.plot_modality_importance(
                                explanations['modality_importance'],
                                save_name=f'modality_importance_epoch{epoch+1}.png',
                                title=f'Modality Importance - Epoch {epoch+1}'
                            )
                        
                        # Save GradCAM visualizations
                        if 'gradcam' in explanations and len(explanations['gradcam']) > 0:
                            xai_visualizer.plot_gradcam_overlay(
                                explanations['gradcam'],
                                save_name=f'gradcam_epoch{epoch+1}.png',
                                title=f'GradCAM Activations - Epoch {epoch+1}'
                            )
                    
                    # Compute xAI metrics if enabled
                    if args.xai_compute_metrics:
                        from utils.xai_metrics import XAIMetrics
                        xai_metrics = XAIMetrics.compute_all_metrics(
                            model, sample_indices, explanations, sample_values
                        )
                        print("\nxAI Evaluation Metrics:")
                        for metric_name, metric_value in xai_metrics.items():
                            print(f"  {metric_name:25s}: {metric_value:.4f}")
                    
                    print(f"\n{'='*60}\n")
                    
                except Exception as e:
                    print(f"Warning: xAI analysis failed: {e}")
                    import traceback
                    traceback.print_exc()


    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        estimator.eval()
        with torch.no_grad():
            best_test_metrics, _ = corpus.get_validation_pred(model, 'test')
    print('\n'.join(['Val set results:', model.format_metrics(best_val_metrics, 'val')]))
    print('\n'.join(['Test set results:', model.format_metrics(best_test_metrics, 'test')]))
    print("\n\n\n\n\n\n")

    if args.save:
        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}.pth')
        print('Saved model!')


if __name__ == '__main__':
    train_decoder(args)
