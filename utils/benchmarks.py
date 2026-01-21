"""
Predefined benchmark configurations for reproducible HEST experiments.

Includes benchmarks for multiple tissue types:
- Lung, Breast, Colon, Prostate

Each tissue type has benchmarks for different gene filtering strategies:
- Random, Highly Variable (HVG), Spatially Variable (SVG/Moran's I)

Usage:
    python pipeline/finetune.py --benchmark lung_hvg_100 --output_dir ./results
    python pipeline/finetune.py --list_benchmarks
"""

from typing import Dict, Any, List, Optional


# ============================================================================
# Base path - UPDATE THIS if your data is in a different location
# ============================================================================
HEST_BASE_PATH = "/multimodal/dist"


# ============================================================================
# Helper to generate benchmarks for a tissue type
# ============================================================================
def _make_tissue_benchmarks(tissue_name: str, folder_name: str) -> Dict[str, Dict[str, Any]]:
    """Generate standard benchmarks for a tissue type."""
    base = f"{HEST_BASE_PATH}/{folder_name}"
    
    return {
        f"{tissue_name}_random_50": {
            "description": f"{tissue_name.upper()} with 50 randomly selected genes",
            "filter_strategy": "random",
            "n_genes": 50,
            "patches_path": f"{base}/patches/",
            "adata_path": f"{base}/st/",
            "hyperparams": {"learning_rate": 1e-4, "batch_size": 32, "epochs": 100, "weight_decay": 1e-5},
            "seed": 42
        },
        f"{tissue_name}_random_100": {
            "description": f"{tissue_name.upper()} with 100 randomly selected genes",
            "filter_strategy": "random",
            "n_genes": 100,
            "patches_path": f"{base}/patches/",
            "adata_path": f"{base}/st/",
            "hyperparams": {"learning_rate": 1e-4, "batch_size": 32, "epochs": 100, "weight_decay": 1e-5},
            "seed": 42
        },
        f"{tissue_name}_hvg_50": {
            "description": f"{tissue_name.upper()} with 50 highly variable genes",
            "filter_strategy": "hvg",
            "n_genes": 50,
            "patches_path": f"{base}/patches/",
            "adata_path": f"{base}/st/",
            "hyperparams": {"learning_rate": 1e-4, "batch_size": 32, "epochs": 100, "weight_decay": 1e-5},
            "seed": 42
        },
        f"{tissue_name}_hvg_100": {
            "description": f"{tissue_name.upper()} with 100 highly variable genes",
            "filter_strategy": "hvg",
            "n_genes": 100,
            "patches_path": f"{base}/patches/",
            "adata_path": f"{base}/st/",
            "hyperparams": {"learning_rate": 1e-4, "batch_size": 32, "epochs": 100, "weight_decay": 1e-5},
            "seed": 42
        },
        f"{tissue_name}_svg_50": {
            "description": f"{tissue_name.upper()} with 50 spatially variable genes (Moran's I)",
            "filter_strategy": "svg",
            "n_genes": 50,
            "patches_path": f"{base}/patches/",
            "adata_path": f"{base}/st/",
            "hyperparams": {"learning_rate": 1e-4, "batch_size": 32, "epochs": 100, "weight_decay": 1e-5},
            "seed": 42
        },
        f"{tissue_name}_svg_100": {
            "description": f"{tissue_name.upper()} with 100 spatially variable genes (Moran's I)",
            "filter_strategy": "svg",
            "n_genes": 100,
            "patches_path": f"{base}/patches/",
            "adata_path": f"{base}/st/",
            "hyperparams": {"learning_rate": 1e-4, "batch_size": 32, "epochs": 100, "weight_decay": 1e-5},
            "seed": 42
        },
    }


# ============================================================================
# Generate all benchmarks
# ============================================================================
BENCHMARKS: Dict[str, Dict[str, Any]] = {}

# Add benchmarks for each tissue type
BENCHMARKS.update(_make_tissue_benchmarks("lung", "hest_data_lung"))
BENCHMARKS.update(_make_tissue_benchmarks("breast", "hest_data_breast"))
BENCHMARKS.update(_make_tissue_benchmarks("colon", "hest_data_colon"))
BENCHMARKS.update(_make_tissue_benchmarks("prostate", "hest_data_prostate"))


# ============================================================================
# API Functions
# ============================================================================

def get_benchmark(name: str) -> Dict[str, Any]:
    """
    Get a benchmark configuration by name.
    
    Args:
        name: Name of the benchmark (e.g., 'lung_hvg_100')
        
    Returns:
        Dictionary with benchmark configuration
    """
    if name not in BENCHMARKS:
        available = list(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    
    return BENCHMARKS[name].copy()


def list_benchmarks() -> List[str]:
    """List all available benchmark names."""
    return list(BENCHMARKS.keys())


def print_benchmark_info(name: Optional[str] = None) -> None:
    """Print information about benchmarks."""
    if name is not None:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}")
            return
        config = BENCHMARKS[name]
        print(f"\n=== {name} ===")
        print(f"Description: {config.get('description', 'N/A')}")
        print(f"Gene Filter: {config['filter_strategy']} (n={config.get('n_genes', 'N/A')})")
        print(f"Paths:")
        print(f"  patches_path: {config.get('patches_path')}")
        print(f"  adata_path: {config.get('adata_path')}")
        print(f"Hyperparams: {config.get('hyperparams', {})}")
    else:
        print("\n" + "="*60)
        print("AVAILABLE BENCHMARKS")
        print("="*60)
        
        # Group by tissue
        tissues = ["lung", "breast", "colon", "prostate"]
        for tissue in tissues:
            print(f"\n{tissue.upper()}:")
            for bname in sorted(BENCHMARKS.keys()):
                if bname.startswith(tissue):
                    config = BENCHMARKS[bname]
                    strategy = config['filter_strategy'].upper()
                    n_genes = config['n_genes']
                    print(f"  {bname:25s} - {strategy} {n_genes} genes")
        
        print("\n" + "-"*60)
        print("Usage:")
        print("  python pipeline/finetune.py --benchmark lung_hvg_100 --output_dir ./results")
        print("  python pipeline/distill.py --benchmark breast_svg_100 --output_dir ./results")
        print("="*60)


if __name__ == "__main__":
    print_benchmark_info()
