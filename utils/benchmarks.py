"""
Predefined benchmark configurations for reproducible experiments.

Users can run standardized benchmarks with:
    python pipeline/finetune.py --benchmark hest_hvg_100
"""

from typing import Dict, Any, List, Optional


# ============================================================================
# Benchmark Configuration Registry
# ============================================================================

BENCHMARKS: Dict[str, Dict[str, Any]] = {
    
    # ========================================================================
    # HEST Benchmarks - Spatial Gene Expression Prediction
    # ========================================================================
    
    "hest_random_50": {
        "description": "HEST with 50 randomly selected genes",
        "dataset": "hest",
        "filter_strategy": "random",
        "n_genes": 50,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        # FILL IN: paths to your HEST data
        "data_path": "<FILL_IN_HEST_DATA_PATH>",
        "seed": 42
    },
    
    "hest_random_100": {
        "description": "HEST with 100 randomly selected genes",
        "dataset": "hest",
        "filter_strategy": "random",
        "n_genes": 100,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        "data_path": "<FILL_IN_HEST_DATA_PATH>",
        "seed": 42
    },
    
    "hest_hvg_50": {
        "description": "HEST with 50 highly variable genes",
        "dataset": "hest",
        "filter_strategy": "hvg",
        "n_genes": 50,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        "data_path": "<FILL_IN_HEST_DATA_PATH>",
        "seed": 42
    },
    
    "hest_hvg_100": {
        "description": "HEST with 100 highly variable genes",
        "dataset": "hest",
        "filter_strategy": "hvg",
        "n_genes": 100,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        "data_path": "<FILL_IN_HEST_DATA_PATH>",
        "seed": 42
    },
    
    "hest_svg_50": {
        "description": "HEST with 50 spatially variable genes (Moran's I)",
        "dataset": "hest",
        "filter_strategy": "svg",
        "n_genes": 50,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        "data_path": "<FILL_IN_HEST_DATA_PATH>",
        "seed": 42
    },
    
    "hest_svg_100": {
        "description": "HEST with 100 spatially variable genes (Moran's I)",
        "dataset": "hest",
        "filter_strategy": "svg",
        "n_genes": 100,
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 1e-5
        },
        "data_path": "<FILL_IN_HEST_DATA_PATH>",
        "seed": 42
    },
    
    # ========================================================================
    # WSI Classification Benchmarks
    # ========================================================================
    
    "wsiclass_default": {
        "description": "WSI Classification with default settings",
        "dataset": "wsiclass",
        "hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 64,
            "epochs": 50,
            "weight_decay": 1e-5
        },
        # FILL IN: paths to your WSI data
        "data_path": "<FILL_IN_WSI_PATCHES_PATH>",
        "metadata_path": "<FILL_IN_WSI_METADATA_PATH>",
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
    for key in ['data_path', 'metadata_path']:
        if key in config and config[key].startswith('<FILL_IN'):
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
        print(f"Dataset: {config.get('dataset', 'N/A')}")
        if 'filter_strategy' in config:
            print(f"Gene Filter: {config['filter_strategy']} (n={config.get('n_genes', 'N/A')})")
        print(f"Hyperparams: {config.get('hyperparams', {})}")
    else:
        print("\n=== Available Benchmarks ===\n")
        for bname, config in BENCHMARKS.items():
            desc = config.get('description', 'No description')
            dataset = config.get('dataset', 'unknown')
            print(f"  {bname:20s} [{dataset:8s}] - {desc}")
        print("\nUse --benchmark <name> to run a predefined benchmark")


if __name__ == "__main__":
    print_benchmark_info()

