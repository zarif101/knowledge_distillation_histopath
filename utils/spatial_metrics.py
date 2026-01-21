"""
Spatial metrics for evaluating HEST predictions.

Includes Structural Similarity Index (SSIM) for spatial gene expression maps.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings


def build_spatial_maps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    adata,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build 2D spatial maps for each gene from spot coordinates.
    
    Args:
        y_true: Ground truth array of shape (N_spots, N_genes)
        y_pred: Predicted array of shape (N_spots, N_genes)
        adata: AnnData object with spatial coordinates
        
    Returns:
        Tuple of (pred_maps, true_maps) - lists of 2D arrays per gene
    """
    # Get integer row/col for each spot
    if 'array_row' in adata.obs and 'array_col' in adata.obs:
        rows = adata.obs['array_row'].values.astype(int)
        cols = adata.obs['array_col'].values.astype(int)
    elif 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
        # Normalize coordinates to grid indices
        rows = coords[:, 1].astype(int)
        cols = coords[:, 0].astype(int)
        # Shift to start from 0
        rows = rows - rows.min()
        cols = cols - cols.min()
    else:
        raise ValueError("AnnData must have 'array_row'/'array_col' in obs or 'spatial' in obsm")
    
    # Determine grid size
    H, W = rows.max() + 1, cols.max() + 1
    G = y_true.shape[1]
    
    pred_maps = []
    true_maps = []
    
    for g in range(G):
        # Create H×W maps for this gene
        true_map = np.zeros((H, W), dtype=np.float32)
        pred_map = np.zeros((H, W), dtype=np.float32)
        
        for i, (r, c) in enumerate(zip(rows, cols)):
            # Handle zeros (add small epsilon for SSIM stability)
            true_val = y_true[i, g] if y_true[i, g] != 0 else 1e-6
            true_map[r, c] = true_val
            pred_map[r, c] = y_pred[i, g]
        
        pred_maps.append(pred_map)
        true_maps.append(true_map)
    
    return pred_maps, true_maps


def compute_ssim_per_gene(
    true_maps: List[np.ndarray],
    pred_maps: List[np.ndarray],
    data_range: Optional[float] = None
) -> List[float]:
    """
    Compute SSIM for each gene's spatial map.
    
    Args:
        true_maps: List of ground truth 2D maps per gene
        pred_maps: List of predicted 2D maps per gene
        data_range: Data range for SSIM calculation (auto-computed if None)
        
    Returns:
        List of SSIM scores per gene
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError("scikit-image is required for SSIM. Install with: pip install scikit-image")
    
    ssim_scores = []
    
    for true_map, pred_map in zip(true_maps, pred_maps):
        # Compute data range if not provided
        if data_range is None:
            dr = max(true_map.max() - true_map.min(), pred_map.max() - pred_map.min())
            dr = max(dr, 1e-6)  # Avoid division by zero
        else:
            dr = data_range
        
        # Compute SSIM
        try:
            score = ssim(true_map, pred_map, data_range=dr)
        except Exception as e:
            warnings.warn(f"SSIM computation failed: {e}. Using 0.")
            score = 0.0
        
        ssim_scores.append(score)
    
    return ssim_scores


def compute_spatial_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    adata,
    gene_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute all spatial metrics for predictions.
    
    Args:
        y_true: Ground truth array (N_spots, N_genes)
        y_pred: Predicted array (N_spots, N_genes)
        adata: AnnData object with spatial coordinates
        gene_names: Optional list of gene names for reporting
        
    Returns:
        Dictionary with spatial metrics
    """
    # Build spatial maps
    pred_maps, true_maps = build_spatial_maps(y_true, y_pred, adata)
    
    # Compute SSIM per gene
    ssim_scores = compute_ssim_per_gene(true_maps, pred_maps)
    
    # Aggregate results
    results = {
        'ssim_mean': float(np.mean(ssim_scores)),
        'ssim_std': float(np.std(ssim_scores)),
        'ssim_median': float(np.median(ssim_scores)),
        'ssim_min': float(np.min(ssim_scores)),
        'ssim_max': float(np.max(ssim_scores)),
        'ssim_per_gene': ssim_scores
    }
    
    if gene_names is not None:
        results['ssim_by_gene'] = {
            gene: score for gene, score in zip(gene_names, ssim_scores)
        }
    
    return results


def evaluate_with_spatial(
    model,
    val_loader,
    loss_fn,
    device,
    adata,
    gene_names: Optional[List[str]] = None
) -> Dict:
    """
    Full evaluation including spatial SSIM metrics.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to run on
        adata: AnnData object with spatial coordinates
        gene_names: Optional gene names
        
    Returns:
        Dictionary with all metrics including spatial SSIM
    """
    import torch
    
    model.eval()
    val_loss = 0
    all_true, all_pred = [], []
    
    with torch.no_grad():
        for imgs, y_true in val_loader:
            imgs, y_true = imgs.to(device), y_true.to(device)
            preds = model(imgs)
            val_loss += loss_fn(preds, y_true).item()
            all_true.append(y_true.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Concatenate across batches
    y_true_arr = np.concatenate(all_true, axis=0)
    y_pred_arr = np.concatenate(all_pred, axis=0)
    
    # Compute spatial metrics
    spatial_metrics = compute_spatial_metrics(
        y_true_arr, y_pred_arr, adata, gene_names
    )
    
    # Add loss
    spatial_metrics['val_loss'] = val_loss
    
    return spatial_metrics

