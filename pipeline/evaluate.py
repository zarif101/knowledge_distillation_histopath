#!/usr/bin/env python
"""
Evaluation script for trained models (HEST and custom datasets).

HEST (default): Provide patches_path and adata_path
Custom datasets: Register an adapter and use --dataset your_adapter_name

Usage:
    # HEST evaluation
    python pipeline/evaluate.py \
        --model_path ./results/model_epoch99 \
        --patches_path /path/to/patches/ \
        --adata_path /path/to/st/ \
        --config_path ./results/config.json \
        --output_dir ./results/eval \
        --compute_spatial
    
    # Custom dataset evaluation
    python pipeline/evaluate.py \
        --dataset my_custom_adapter \
        --model_path ./results/model_epoch99 \
        --data_path /path/to/my/data \
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

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='hest',
                        help='Dataset adapter to use. Default "hest".')
    
    # Model
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config.json from training')
    
    # HEST paths
    parser.add_argument('--patches_path', type=str, default=None)
    parser.add_argument('--adata_path', type=str, default=None)
    
    # Custom adapter path
    parser.add_argument('--data_path', type=str, default=None)
    
    # Evaluation split
    parser.add_argument('--eval_split_path', type=str, default=None)
    parser.add_argument('--use_val', action='store_true',
                        help='Use validation set from training')
    
    # HEST gene list
    parser.add_argument('--gene_list_path', type=str, default=None)
    
    # Spatial metrics (HEST only)
    parser.add_argument('--compute_spatial', action='store_true',
                        help='Compute spatial SSIM metrics (HEST only)')
    
    # Save predictions
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predicted expression as h5ad files (HEST only)')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    return parser.parse_args()


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr
    
    mse = float(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    n_outputs = y_true.shape[1] if len(y_true.shape) > 1 else 1
    pearson_corrs, spearman_corrs = [], []
    
    for i in range(n_outputs):
        col_true = y_true[:, i] if n_outputs > 1 else y_true
        col_pred = y_pred[:, i] if n_outputs > 1 else y_pred
        try:
            pc, _ = pearsonr(col_true, col_pred)
            sc, _ = spearmanr(col_true, col_pred)
        except:
            pc, sc = 0, 0
        pearson_corrs.append(pc if not np.isnan(pc) else 0)
        spearman_corrs.append(sc if not np.isnan(sc) else 0)
    
    return {
        'mse': mse, 'r2': r2,
        'pearson_mean': float(np.mean(pearson_corrs)),
        'pearson_std': float(np.std(pearson_corrs)),
        'spearman_mean': float(np.mean(spearman_corrs)),
        'spearman_std': float(np.std(spearman_corrs)),
        'pearson_per_output': pearson_corrs,
        'spearman_per_output': spearman_corrs
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred
    
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred_classes)),
        'f1_macro': float(f1_score(y_true, y_pred_classes, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)),
        'precision_macro': float(precision_score(y_true, y_pred_classes, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred_classes, average='macro', zero_division=0))
    }


def run_inference(model, data_loader, device):
    model.eval()
    all_true, all_pred = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            imgs, targets = batch
            imgs = imgs.to(device)
            preds = model(imgs)
            all_true.append(targets.numpy())
            all_pred.append(preds.cpu().numpy())
    
    return np.concatenate(all_true), np.concatenate(all_pred)


def get_sample_ids_from_h5_dir(directory: str) -> list:
    return sorted([f.stem for f in Path(directory).glob("*.h5")])


def save_predictions_as_h5ad(
    y_pred: np.ndarray,
    eval_samples: list,
    adata_path: str,
    patches_path: str,
    gene_list: list,
    output_dir: Path
):
    """
    Save predictions as h5ad files, one per sample.
    
    Maps predictions back to samples by loading each sample's adata and matching
    patch counts. Creates new h5ad files with predicted expression values.
    """
    import scanpy as sc
    from utils.data_utils import read_h5_patches, align_st_patches
    
    predictions_dir = output_dir / 'predictions'
    predictions_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving predictions to {predictions_dir}...")
    
    pred_idx = 0
    for sample_id in eval_samples:
        # Load original adata to get structure
        adata_file = Path(adata_path) / f"{sample_id}.h5ad"
        if not adata_file.exists():
            print(f"Warning: {adata_file} not found, skipping {sample_id}")
            continue
        
        # Load patches and adata to get alignment
        patch_path = Path(patches_path) / f"{sample_id}.h5"
        if not patch_path.exists():
            print(f"Warning: {patch_path} not found, skipping {sample_id}")
            continue
        
        patches, barcodes = read_h5_patches(str(patch_path))
        adata = sc.read_h5ad(adata_file)
        patches_aligned, adata_aligned = align_st_patches((patches, barcodes), adata)
        
        n_spots = adata_aligned.shape[0]
        
        # Extract predictions for this sample
        sample_preds = y_pred[pred_idx:pred_idx + n_spots]
        pred_idx += n_spots
        
        # Create new adata with predicted expression
        # Apply same preprocessing as dataset (normalize_total, log1p) for structure consistency
        pred_adata = adata_aligned.copy()
        pred_adata.var_names = pred_adata.var_names.str.upper()
        sc.pp.normalize_total(pred_adata)
        sc.pp.log1p(pred_adata)
        
        # Filter to selected genes (same as dataset does)
        # This keeps genes in adata.var_names order (filtered)
        pred_adata = pred_adata[:, pred_adata.var_names.isin(gene_list)]
        
        # Get available genes in adata order (this is the order predictions are in)
        available_genes = list(pred_adata.var_names)
        
        # Map predictions: predictions are in gene_list order, need to reorder to adata order
        # Create mapping from gene_list indices to adata column positions
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
        adata_col_indices = [gene_to_idx[g] for g in available_genes if g in gene_to_idx]
        
        # Extract and reorder predictions to match adata gene order
        sample_preds_reordered = sample_preds[:, adata_col_indices]
        
        # Set predicted expression (predictions are already in normalized/log1p space)
        # Convert to dense if sparse
        if hasattr(pred_adata.X, 'toarray'):
            pred_adata.X = pred_adata.X.toarray()
        pred_adata.X = sample_preds_reordered
        
        # Save
        output_file = predictions_dir / f"{sample_id}_predicted.h5ad"
        pred_adata.write(output_file)
        print(f"  Saved {sample_id}: {n_spots} spots, {len(available_genes)} genes -> {output_file}")
    
    print(f"Predictions saved to: {predictions_dir}")


def run_hest_eval(args, output_dir, device, config):
    """Evaluate HEST model."""
    from utils.gene_filtering import load_gene_list
    from utils.split_utils import load_split_file, get_train_val_split
    from utils.data_utils import STPatchDatasetHEST
    from modeling.models import get_transforms
    
    patches_path = args.patches_path or config.get('patches_path')
    adata_path = args.adata_path or config.get('adata_path')
    
    if not patches_path or not adata_path:
        raise ValueError("HEST requires patches_path and adata_path")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path, map_location=device)
    if hasattr(model, 'module'):
        model = model.module
    model = model.to(device)
    model.eval()
    
    # Get eval samples
    config_dir = Path(args.config_path).parent if args.config_path else None
    
    if args.eval_split_path:
        eval_samples = load_split_file(args.eval_split_path)
    elif args.use_val and config_dir and (config_dir / 'val_split.json').exists():
        eval_samples = load_split_file(config_dir / 'val_split.json')
    else:
        all_samples = get_sample_ids_from_h5_dir(patches_path)
        _, eval_samples = get_train_val_split(all_samples, seed=config.get('seed', 42))
    
    print(f"Evaluating on {len(eval_samples)} samples")
    
    # Load gene list
    gene_list_path = args.gene_list_path
    if not gene_list_path and config_dir:
        candidate = config_dir / 'gene_list.pkl'
        if candidate.exists():
            gene_list_path = str(candidate)
    
    if not gene_list_path:
        raise ValueError("Gene list required for HEST evaluation")
    
    gene_list = load_gene_list(gene_list_path)
    print(f"Using {len(gene_list)} genes")
    
    # Create dataset
    transforms = get_transforms(config.get('model', 'tinyvit'))
    eval_dataset = STPatchDatasetHEST(patches_path, adata_path, eval_samples, gene_list, transforms)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, 
                             num_workers=args.num_workers, pin_memory=True)
    
    # Inference
    print("Running inference...")
    y_true, y_pred = run_inference(model, eval_loader, device)
    
    # Metrics
    metrics = compute_regression_metrics(y_true, y_pred)
    
    # Spatial metrics
    if args.compute_spatial:
        print("Computing spatial SSIM...")
        from utils.spatial_metrics import compute_spatial_metrics
        import scanpy as sc
        adata = sc.read_h5ad(Path(adata_path) / f"{eval_samples[0]}.h5ad")
        spatial = compute_spatial_metrics(y_true, y_pred, adata, gene_list)
        metrics['spatial'] = spatial
    
    # Save predictions as h5ad files
    if args.save_predictions:
        save_predictions_as_h5ad(
            y_pred, eval_samples, adata_path, patches_path, 
            gene_list, output_dir
        )
    
    return metrics, gene_list


def run_custom_eval(args, output_dir, device, config):
    """Evaluate custom dataset model."""
    from utils.dataset_adapter import get_dataset_adapter
    from modeling.models import get_transforms
    
    adapter = get_dataset_adapter(args.dataset)
    print(f"Using adapter: {args.dataset}")
    
    # Load model
    model = torch.load(args.model_path, map_location=device)
    if hasattr(model, 'module'):
        model = model.module
    model = model.to(device)
    model.eval()
    
    # Load data
    transforms = get_transforms(config.get('model', 'tinyvit'))
    _, eval_loader = adapter.load_data(
        train_samples=[], val_samples=[],
        transforms=transforms,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Inference
    y_true, y_pred = run_inference(model, eval_loader, device)
    
    # Metrics based on task type
    if adapter.task_type == 'regression':
        metrics = compute_regression_metrics(y_true, y_pred)
    else:
        metrics = compute_classification_metrics(y_true, y_pred)
    
    return metrics, None


def main():
    args = parse_args()
    
    # Load config
    config = {}
    if args.config_path:
        with open(args.config_path) as f:
            config = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Route to handler
    if args.dataset.lower() == 'hest':
        metrics, gene_list = run_hest_eval(args, output_dir, device, config)
    else:
        metrics, gene_list = run_custom_eval(args, output_dir, device, config)
    
    # Add metadata
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['model_path'] = str(args.model_path)
    metrics['dataset'] = args.dataset
    
    # Save
    def to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json(v) for v in obj]
        return obj
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(to_json(metrics), f, indent=2)
    
    # Per-gene CSV for HEST
    if gene_list and 'pearson_per_output' in metrics:
        import pandas as pd
        df = pd.DataFrame({
            'gene': gene_list,
            'pearson': metrics['pearson_per_output'],
            'spearman': metrics['spearman_per_output']
        })
        if 'spatial' in metrics and 'ssim_per_gene' in metrics['spatial']:
            df['ssim'] = metrics['spatial']['ssim_per_gene']
        df.to_csv(output_dir / 'per_gene_metrics.csv', index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if 'mse' in metrics:
        print(f"MSE:           {metrics['mse']:.6f}")
        print(f"R²:            {metrics['r2']:.4f}")
        print(f"Pearson:       {metrics['pearson_mean']:.4f} ± {metrics['pearson_std']:.4f}")
        print(f"Spearman:      {metrics['spearman_mean']:.4f} ± {metrics['spearman_std']:.4f}")
        if 'spatial' in metrics:
            print(f"SSIM:          {metrics['spatial']['ssim_mean']:.4f} ± {metrics['spatial']['ssim_std']:.4f}")
    else:
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"F1 (macro):    {metrics['f1_macro']:.4f}")
    
    print("="*50)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
