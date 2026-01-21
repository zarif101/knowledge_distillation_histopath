"""
Gene filtering strategies for HEST spatial transcriptomics.

Supports:
- Random selection
- Highly Variable Genes (HVG) via scanpy
- Spatially Variable Genes (SVG) via Moran's I
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Union
import warnings


def filter_random(
    adata,
    n_genes: int = 100,
    seed: int = 42
) -> List[str]:
    """
    Randomly select n_genes from the AnnData object.
    
    Args:
        adata: AnnData object with gene expression data
        n_genes: Number of genes to select
        seed: Random seed for reproducibility
        
    Returns:
        List of selected gene names
    """
    np.random.seed(seed)
    all_genes = list(adata.var_names)
    n_genes = min(n_genes, len(all_genes))
    selected = np.random.choice(all_genes, size=n_genes, replace=False)
    return list(selected)


def filter_highly_variable(
    adata,
    n_genes: int = 100,
    flavor: str = 'seurat_v3',
    **kwargs
) -> List[str]:
    """
    Select highly variable genes using scanpy.
    
    Args:
        adata: AnnData object with gene expression data
        n_genes: Number of top HVGs to select
        flavor: Method for HVG selection ('seurat', 'seurat_v3', 'cell_ranger')
        **kwargs: Additional arguments passed to sc.pp.highly_variable_genes
        
    Returns:
        List of selected gene names
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for HVG filtering. Install with: pip install scanpy")
    
    # Work on a copy to avoid modifying original
    adata_copy = adata.copy()
    
    # Normalize if not already done (required for HVG)
    if 'log1p' not in adata_copy.uns:
        sc.pp.normalize_total(adata_copy, target_sum=1e4)
        sc.pp.log1p(adata_copy)
    
    # Compute HVGs
    sc.pp.highly_variable_genes(
        adata_copy,
        n_top_genes=n_genes,
        flavor=flavor,
        **kwargs
    )
    
    # Get highly variable genes
    hvg_mask = adata_copy.var['highly_variable']
    selected_genes = list(adata_copy.var_names[hvg_mask])
    
    return selected_genes[:n_genes]


def filter_spatially_variable(
    adata,
    n_genes: int = 100,
    **kwargs
) -> List[str]:
    """
    Select spatially variable genes using Moran's I statistic.
    
    Args:
        adata: AnnData object with spatial coordinates in obsm['spatial']
        n_genes: Number of top SVGs to select
        **kwargs: Additional arguments passed to sq.gr.spatial_autocorr
        
    Returns:
        List of selected gene names
    """
    try:
        import squidpy as sq
    except ImportError:
        raise ImportError("squidpy is required for spatial variability filtering. Install with: pip install squidpy")
    
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for spatial variability filtering. Install with: pip install scanpy")
    
    # Work on a copy
    adata_copy = adata.copy()
    
    # Ensure spatial coordinates exist
    if 'spatial' not in adata_copy.obsm:
        raise ValueError("AnnData must have spatial coordinates in obsm['spatial']")
    
    # Normalize if needed
    if 'log1p' not in adata_copy.uns:
        sc.pp.normalize_total(adata_copy, target_sum=1e4)
        sc.pp.log1p(adata_copy)
    
    # Build spatial neighbors graph
    sq.gr.spatial_neighbors(adata_copy, coord_type='generic')
    
    # Compute Moran's I for all genes
    sq.gr.spatial_autocorr(
        adata_copy,
        mode='moran',
        **kwargs
    )
    
    # Sort by Moran's I and select top genes
    moranI_df = adata_copy.uns['moranI'].sort_values(by='I', ascending=False)
    selected_genes = list(moranI_df.head(n_genes).index)
    
    return selected_genes


def get_gene_list(
    adata,
    strategy: str = 'random',
    n_genes: int = 100,
    seed: int = 42,
    **kwargs
) -> List[str]:
    """
    Get a gene list using the specified filtering strategy.
    
    Args:
        adata: AnnData object
        strategy: One of 'random', 'hvg', 'svg' (or 'morans_i')
        n_genes: Number of genes to select
        seed: Random seed (for random strategy)
        **kwargs: Additional arguments for specific strategies
        
    Returns:
        List of selected gene names
    """
    strategy = strategy.lower()
    
    if strategy == 'random':
        return filter_random(adata, n_genes=n_genes, seed=seed)
    elif strategy in ['hvg', 'highly_variable']:
        return filter_highly_variable(adata, n_genes=n_genes, **kwargs)
    elif strategy in ['svg', 'spatially_variable', 'morans_i', 'moran']:
        return filter_spatially_variable(adata, n_genes=n_genes, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'random', 'hvg', or 'svg'")


def get_gene_list_from_multiple_samples(
    adata_list: List,
    strategy: str = 'random',
    n_genes: int = 100,
    seed: int = 42,
    aggregation: str = 'intersection',
    **kwargs
) -> List[str]:
    """
    Get a consensus gene list from multiple AnnData objects.
    
    Args:
        adata_list: List of AnnData objects
        strategy: Filtering strategy
        n_genes: Number of genes per sample
        seed: Random seed
        aggregation: How to combine gene lists ('intersection' or 'union')
        **kwargs: Additional arguments for specific strategies
        
    Returns:
        List of selected gene names
    """
    gene_sets = []
    for i, adata in enumerate(adata_list):
        genes = get_gene_list(
            adata,
            strategy=strategy,
            n_genes=n_genes,
            seed=seed + i,  # Different seed per sample for random
            **kwargs
        )
        gene_sets.append(set(genes))
    
    if aggregation == 'intersection':
        final_genes = set.intersection(*gene_sets)
    elif aggregation == 'union':
        final_genes = set.union(*gene_sets)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return list(final_genes)


def save_gene_list(gene_list: List[str], path: Union[str, Path]) -> None:
    """Save gene list to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(gene_list, f)
    print(f"Saved {len(gene_list)} genes to {path}")


def load_gene_list(path: Union[str, Path]) -> List[str]:
    """Load gene list from pickle file."""
    with open(path, 'rb') as f:
        gene_list = pickle.load(f)
    return gene_list

