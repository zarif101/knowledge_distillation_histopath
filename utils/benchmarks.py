"""
Predefined benchmark configurations for reproducible HEST experiments.

Users can run standardized benchmarks with:
    python pipeline/finetune.py --benchmark hest_hvg_100 --output_dir ./results
"""

from typing import Dict, Any, List, Optional


# ============================================================================
# HEST Benchmark Configurations
# ============================================================================

BENCHMARKS: Dict[str, Dict[str, Any]] = {
    
    "hest_random_50": {
        "description": "HEST with 50 randomly selected genes",
        "filter_strategy": "random",
        "n_genes": 50,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        # FILL IN: paths to your HEST data
        "patches_path": "<FILL_IN_HEST_PATCHES_PATH>",  # e.g., "/data/hest/patches/"
        "adata_path": "<FILL_IN_HEST_ADATA_PATH>",      # e.g., "/data/hest/st/"
        "seed": 42
    },
    
    "hest_random_100": {
        "description": "HEST with 100 randomly selected genes",
        "filter_strategy": "random",
        "n_genes": 100,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        "patches_path": "<FILL_IN_HEST_PATCHES_PATH>",
        "adata_path": "<FILL_IN_HEST_ADATA_PATH>",
        "seed": 42
    },
    
    "hest_hvg_50": {
        "description": "HEST with 50 highly variable genes",
        "filter_strategy": "hvg",
        "n_genes": 50,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        "patches_path": "<FILL_IN_HEST_PATCHES_PATH>",
        "adata_path": "<FILL_IN_HEST_ADATA_PATH>",
        "seed": 42
    },
    
    "hest_hvg_100": {
        "description": "HEST with 100 highly variable genes",
        "filter_strategy": "hvg",
        "n_genes": 100,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        "patches_path": "<FILL_IN_HEST_PATCHES_PATH>",
        "adata_path": "<FILL_IN_HEST_ADATA_PATH>",
        "seed": 42
    },
    
    "hest_svg_50": {
        "description": "HEST with 50 spatially variable genes (Moran's I)",
        "filter_strategy": "svg",
        "n_genes": 50,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        "patches_path": "<FILL_IN_HEST_PATCHES_PATH>",
        "adata_path": "<FILL_IN_HEST_ADATA_PATH>",
        "seed": 42
    },
    
    "hest_svg_100": {
        "description": "HEST with 100 spatially variable genes (Moran's I)",
        "filter_strategy": "svg",
        "n_genes": 100,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        "patches_path": "<FILL_IN_HEST_PATCHES_PATH>",
        "adata_path": "<FILL_IN_HEST_ADATA_PATH>",
        "seed": 42
    },
    
}


def get_benchmark(name: str) -> Dict[str, Any]:
    """
    Get a benchmark configuration by name.
    
    Args:
        name: Name of the benchmark
        
    Returns:
        Dictionary with benchmark configuration
    """
    if name not in BENCHMARKS:
        available = list(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    
    config = BENCHMARKS[name].copy()
    
    # Validate that paths are filled in
    for key in ['patches_path', 'adata_path']:
        if key in config and isinstance(config[key], str) and config[key].startswith('<FILL_IN'):
            raise ValueError(
                f"Benchmark '{name}' has placeholder path for '{key}'. "
                f"Please fill in the actual path in utils/benchmarks.py"
            )
    
    return config


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
        print(f"Hyperparams: {config.get('hyperparams', {})}")
        print(f"Paths:")
        print(f"  patches_path: {config.get('patches_path', 'N/A')}")
        print(f"  adata_path: {config.get('adata_path', 'N/A')}")
    else:
        print("\n=== Available HEST Benchmarks ===\n")
        for bname, config in BENCHMARKS.items():
            desc = config.get('description', 'No description')
            print(f"  {bname:20s} - {desc}")
        print("\nUsage: python pipeline/finetune.py --benchmark <name> --output_dir ./results")
        print("       python pipeline/distill.py --benchmark <name> --output_dir ./results")


if __name__ == "__main__":
    print_benchmark_info()
