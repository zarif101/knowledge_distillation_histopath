#!/usr/bin/env python
"""
Evaluation script for trained models.

Computes comprehensive metrics including:
- MSE, R², Pearson/Spearman correlation (regression)
- Accuracy, F1, Precision, Recall (classification)
- Spatial SSIM (HEST only)

Usage:
    # Evaluate HEST model
    python pipeline/evaluate.py --dataset hest --model_path ./results/model.pt \
        --data_path /path/to/data --config_path ./results/config.json \
        --output_dir ./results/eval --compute_spatial
    
    # Evaluate WSI model
    python pipeline/evaluate.py --dataset wsiclass --model_path ./results/model.pt \
        --data_path /path/to/data --config_path ./results/config.json \
        --output_dir ./results/eval
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config.json from training')
    parser.add_argument('--dataset', type=str, choices=['hest', 'wsiclass'],
                        help='Dataset type (auto-detected from config if provided)')
    parser.add_argument('--data_path', type=str,
                        help='Path to evaluation data')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to metadata (for wsiclass)')
    
    # Benchmark
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Use predefined benchmark configuration')
    
    # Evaluation split
    parser.add_argument('--eval_split_path', type=str, default=None,
                        help='Path to evaluation split file')
    parser.add_argument('--use_val', action='store_true',
                        help='Use validation set from training config')
    
    # Gene list (HEST)
    parser.add_argument('--gene_list_path', type=str, default=None,
                        help='Path to gene list (auto-detected from config)')
    
    # Spatial metrics
    parser.add_argument('--compute_spatial', action='store_true',
                        help='Compute spatial SSIM metrics (HEST only)')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    return parser.parse_args()


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    from scipy.stats import pearsonr, spearmanr
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # R² (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Per-gene correlations
    n_genes = y_true.shape[1]
    pearson_corrs = []
    spearman_corrs = []
    
    for g in range(n_genes):
        try:
            pc, _ = pearsonr(y_true[:, g], y_pred[:, g])
            sc, _ = spearmanr(y_true[:, g], y_pred[:, g])
        except:
            pc, sc = 0, 0
        pearson_corrs.append(pc if not np.isnan(pc) else 0)
        spearman_corrs.append(sc if not np.isnan(sc) else 0)
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'pearson_mean': float(np.mean(pearson_corrs)),
        'pearson_std': float(np.std(pearson_corrs)),
        'spearman_mean': float(np.mean(spearman_corrs)),
        'spearman_std': float(np.std(spearman_corrs)),
        'pearson_per_gene': pearson_corrs,
        'spearman_per_gene': spearman_corrs
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    # If y_pred is logits/probabilities, convert to class predictions
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred_classes)),
        'f1_macro': float(f1_score(y_true, y_pred_classes, average='macro')),
        'f1_weighted': float(f1_score(y_true, y_pred_classes, average='weighted')),
        'precision_macro': float(precision_score(y_true, y_pred_classes, average='macro')),
        'recall_macro': float(recall_score(y_true, y_pred_classes, average='macro'))
    }


def run_inference(model, data_loader, device) -> tuple:
    """Run inference and collect predictions."""
    model.eval()
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 2:
                imgs, targets = batch
            else:
                imgs, targets = batch[0], batch[1]
            
            imgs = imgs.to(device)
            preds = model(imgs)
            
            all_true.append(targets.numpy())
            all_pred.append(preds.cpu().numpy())
    
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    
    return y_true, y_pred


def main():
    args = parse_args()
    
    # Load config if provided
    config = {}
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {args.config_path}")
    
    # Load benchmark config if specified
    if args.benchmark:
        from utils.benchmarks import get_benchmark
        benchmark_config = get_benchmark(args.benchmark)
        config.update(benchmark_config)
    
    # Determine dataset type
    dataset = args.dataset or config.get('dataset')
    if dataset is None:
        raise ValueError("Must specify --dataset or provide config with dataset")
    
    # Determine data path
    data_path = args.data_path or config.get('data_path')
    if data_path is None:
        raise ValueError("Must specify --data_path")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path, map_location=device)
    if hasattr(model, 'module'):  # DataParallel wrapper
        model = model.module
    model = model.to(device)
    model.eval()
    
    # Get evaluation samples
    if args.eval_split_path:
        from utils.split_utils import load_split_file
        eval_samples = load_split_file(args.eval_split_path)
    elif args.use_val and 'val_split_path' in config:
        from utils.split_utils import load_split_file
        eval_samples = load_split_file(config['val_split_path'])
    else:
        # Use all samples or a default val split
        from utils.split_utils import get_samples_from_paths, get_train_val_split
        all_samples = get_samples_from_paths(data_path)
        _, eval_samples = get_train_val_split(
            all_samples, 
            val_ratio=0.2, 
            seed=config.get('seed', 42)
        )
    
    print(f"Evaluating on {len(eval_samples)} samples")
    
    # Load gene list for HEST
    gene_list = None
    if dataset == 'hest':
        gene_list_path = args.gene_list_path or config.get('gene_list_path')
        if gene_list_path is None:
            # Try to find gene_list.pkl in same directory as config
            config_dir = Path(args.config_path).parent if args.config_path else None
            if config_dir and (config_dir / 'gene_list.pkl').exists():
                gene_list_path = str(config_dir / 'gene_list.pkl')
        
        if gene_list_path:
            from utils.gene_filtering import load_gene_list
            gene_list = load_gene_list(gene_list_path)
            print(f"Loaded {len(gene_list)} genes")
    
    # Get transforms
    from modeling.models import get_transforms
    transforms = get_transforms(config.get('model', 'tinyvit'))
    
    # Create evaluation dataset and loader
    if dataset == 'hest':
        from utils.data_utils import STPatchDatasetHEST
        eval_dataset = STPatchDatasetHEST(
            data_path=data_path,
            sample_ids=eval_samples,
            gene_list=gene_list,
            transform=transforms
        )
    else:
        from utils.data_utils import WSIClassificationDataset
        metadata_path = args.metadata_path or config.get('metadata_path')
        eval_dataset = WSIClassificationDataset(
            patches_path=data_path,
            metadata_path=metadata_path,
            sample_ids=eval_samples,
            transform=transforms
        )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Run inference
    print("Running inference...")
    y_true, y_pred = run_inference(model, eval_loader, device)
    print(f"Predictions shape: {y_pred.shape}")
    
    # Compute metrics
    print("Computing metrics...")
    if dataset == 'hest':
        metrics = compute_regression_metrics(y_true, y_pred)
        
        # Compute spatial metrics if requested
        if args.compute_spatial:
            print("Computing spatial SSIM metrics...")
            from utils.spatial_metrics import compute_spatial_metrics
            
            # Need AnnData for spatial coords - load from first sample
            import scanpy as sc
            sample_path = Path(data_path) / eval_samples[0]
            h5ad_files = list(sample_path.glob("*.h5ad"))
            if h5ad_files:
                adata = sc.read_h5ad(h5ad_files[0])
                spatial_metrics = compute_spatial_metrics(
                    y_true, y_pred, adata, gene_list
                )
                metrics['spatial'] = spatial_metrics
    else:
        metrics = compute_classification_metrics(y_true, y_pred)
    
    # Add metadata
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['model_path'] = str(args.model_path)
    metrics['n_samples'] = len(eval_samples)
    metrics['dataset'] = dataset
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    metrics_json = convert_for_json(metrics)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save per-gene metrics as CSV (for HEST)
    if dataset == 'hest' and gene_list:
        import pandas as pd
        gene_metrics = pd.DataFrame({
            'gene': gene_list,
            'pearson': metrics['pearson_per_gene'],
            'spearman': metrics['spearman_per_gene']
        })
        if 'spatial' in metrics and 'ssim_per_gene' in metrics['spatial']:
            gene_metrics['ssim'] = metrics['spatial']['ssim_per_gene']
        
        gene_metrics_path = output_dir / 'per_gene_metrics.csv'
        gene_metrics.to_csv(gene_metrics_path, index=False)
        print(f"Saved per-gene metrics to {gene_metrics_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    if dataset == 'hest':
        print(f"MSE:              {metrics['mse']:.6f}")
        print(f"R²:               {metrics['r2']:.4f}")
        print(f"Pearson (mean):   {metrics['pearson_mean']:.4f} ± {metrics['pearson_std']:.4f}")
        print(f"Spearman (mean):  {metrics['spearman_mean']:.4f} ± {metrics['spearman_std']:.4f}")
        if 'spatial' in metrics:
            print(f"SSIM (mean):      {metrics['spatial']['ssim_mean']:.4f} ± {metrics['spatial']['ssim_std']:.4f}")
    else:
        print(f"Accuracy:         {metrics['accuracy']:.4f}")
        print(f"F1 (macro):       {metrics['f1_macro']:.4f}")
        print(f"F1 (weighted):    {metrics['f1_weighted']:.4f}")
        print(f"Precision:        {metrics['precision_macro']:.4f}")
        print(f"Recall:           {metrics['recall_macro']:.4f}")
    print("="*50)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

