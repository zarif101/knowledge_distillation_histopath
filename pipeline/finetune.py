#!/usr/bin/env python
"""
Fine-tuning script for HEST and custom datasets.

HEST (default): Just provide patches_path and adata_path
Custom datasets: Register an adapter and use --dataset your_adapter_name

Usage:
    # HEST (default) - any HEST-format dataset (lung, skin, breast, etc.)
    python pipeline/finetune.py \
        --patches_path /path/to/patches/ \
        --adata_path /path/to/st/ \
        --filter_strategy hvg \
        --n_genes 100 \
        --output_dir ./results
    
    # Custom dataset via adapter
    python pipeline/finetune.py \
        --dataset my_custom_adapter \
        --data_path /path/to/my/data \
        --output_dir ./results
    
    # Using predefined benchmark
    python pipeline/finetune.py --benchmark hest_hvg_100 --output_dir ./results
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune models on HEST or custom datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset selection (default: hest)
    parser.add_argument('--dataset', type=str, default='hest',
                        help='Dataset adapter to use. Default "hest" for HEST format. '
                             'For custom datasets, register an adapter and use its name.')
    
    # HEST-specific paths (used when --dataset hest)
    parser.add_argument('--patches_path', type=str,
                        help='[HEST] Path to patches directory containing H5 files')
    parser.add_argument('--adata_path', type=str,
                        help='[HEST] Path to adata directory containing h5ad files')
    
    # Generic data path (for custom adapters)
    parser.add_argument('--data_path', type=str,
                        help='[Custom] Generic data path passed to your adapter')
    
    # Benchmark
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Use predefined benchmark configuration')
    parser.add_argument('--list_benchmarks', action='store_true',
                        help='List available benchmarks and exit')
    
    # Model selection
    parser.add_argument('--model', type=str, default='tinyvit',
                        choices=['tinyvit', 'uni2', 'virchow2'],
                        help='Model architecture')
    
    # Gene filtering (HEST)
    parser.add_argument('--filter_strategy', type=str, default='random',
                        choices=['random', 'hvg', 'svg'],
                        help='[HEST] Gene filtering strategy')
    parser.add_argument('--n_genes', type=int, default=100,
                        help='[HEST] Number of genes to select')
    parser.add_argument('--gene_list_path', type=str, default=None,
                        help='[HEST] Path to pre-computed gene list (pkl)')
    
    # Split configuration
    parser.add_argument('--train_split_path', type=str, default=None,
                        help='Path to train split file')
    parser.add_argument('--val_split_path', type=str, default=None,
                        help='Path to validation split file')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation ratio if no split files')
    
    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def save_config(args, output_dir: Path, extra_info: dict = None):
    """Save full configuration for reproducibility."""
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()
    config['pytorch_version'] = torch.__version__
    if extra_info:
        config.update(extra_info)
    
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Saved config to {config_path}")


def get_sample_ids_from_h5_dir(directory: str) -> list:
    """Get sample IDs from H5 filenames."""
    return sorted([f.stem for f in Path(directory).glob("*.h5")])


def run_hest(args, output_dir, device):
    """Run training for HEST format dataset."""
    from utils.split_utils import get_train_val_split, save_split_file
    from utils.gene_filtering import get_gene_list, save_gene_list, load_gene_list
    from modeling.models import get_model, build_model_with_head
    from utils.train_HEST import finetune_HEST_data
    
    # Validate HEST paths
    if args.patches_path is None or args.adata_path is None:
        raise ValueError("HEST requires --patches_path and --adata_path")
    
    # Get samples and splits
    all_samples = get_sample_ids_from_h5_dir(args.patches_path)
    print(f"Found {len(all_samples)} samples in {args.patches_path}")
    
    train_samples, val_samples = get_train_val_split(
        samples=all_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_split_path=args.train_split_path,
        val_split_path=args.val_split_path
    )
    
    save_split_file(train_samples, output_dir / 'train_split.json')
    save_split_file(val_samples, output_dir / 'val_split.json')
    
    # Gene filtering
    if args.gene_list_path:
        gene_list = load_gene_list(args.gene_list_path)
        print(f"Loaded {len(gene_list)} genes from {args.gene_list_path}")
    else:
        import scanpy as sc
        adata_file = Path(args.adata_path) / f"{train_samples[0]}.h5ad"
        print(f"Computing gene list from {adata_file}")
        adata = sc.read_h5ad(adata_file)
        gene_list = get_gene_list(adata, strategy=args.filter_strategy, 
                                   n_genes=args.n_genes, seed=args.seed)
        save_gene_list(gene_list, output_dir / 'gene_list.pkl')
        print(f"Selected {len(gene_list)} genes using '{args.filter_strategy}'")
    
    # Save config
    save_config(args, output_dir, {'n_genes_actual': len(gene_list)})
    
    # Build model
    model, transforms = get_model(args.model)
    model = build_model_with_head(model, len(gene_list), task='regression')
    model = model.to(device)
    
    # Train
    hyperparams = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers
    }
    
    finetune_HEST_data(
        patches_path=args.patches_path,
        adata_path=args.adata_path,
        train_samples=train_samples,
        val_samples=val_samples,
        gene_list=gene_list,
        log_dir=str(output_dir) + '/',
        model=model,
        transforms=transforms,
        hyperparams_dict=hyperparams
    )
    
    return {'gene_list': gene_list, 'train_samples': train_samples, 'val_samples': val_samples}


def run_custom(args, output_dir, device):
    """Run training for custom dataset via adapter."""
    from utils.dataset_adapter import get_dataset_adapter
    from utils.split_utils import get_train_val_split, save_split_file
    from modeling.models import get_model, build_model_with_head
    
    # Get adapter
    adapter = get_dataset_adapter(args.dataset)
    print(f"Using adapter: {args.dataset} (task: {adapter.task_type})")
    
    # Get default hyperparams from adapter, override with CLI args
    hyperparams = adapter.get_default_hyperparams()
    hyperparams.update({
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers
    })
    
    # For custom adapters, user provides samples or we need a way to get them
    # This is adapter-specific - adapter.load_data handles it
    train_samples = []  # Adapter will handle sample loading
    val_samples = []
    
    # Save config
    save_config(args, output_dir)
    
    # Build model
    model, transforms = get_model(args.model)
    
    # Load data via adapter
    train_loader, val_loader = adapter.load_data(
        train_samples=train_samples,
        val_samples=val_samples,
        transforms=transforms,
        data_path=args.data_path,
        batch_size=hyperparams['batch_size'],
        num_workers=hyperparams['num_workers']
    )
    
    # Get num outputs and build model head
    num_outputs = adapter.get_num_outputs()
    model = build_model_with_head(model, num_outputs, task=adapter.task_type)
    model = model.to(device)
    
    # Get loss function
    loss_fn = adapter.get_loss_function()
    
    # Training loop
    from utils.train_HEST import train_single, eval_single
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'],
                                  weight_decay=hyperparams['weight_decay'])
    
    log_path = str(output_dir / 'log.txt')
    train_single(model, train_loader, val_loader, loss_fn, optimizer,
                 hyperparams['epochs'], device, log_path, str(output_dir) + '/')
    
    return {}


def main():
    args = parse_args()
    
    # Handle --list_benchmarks
    if args.list_benchmarks:
        from utils.benchmarks import print_benchmark_info
        print_benchmark_info()
        return
    
    # Load benchmark config if specified
    if args.benchmark:
        from utils.benchmarks import get_benchmark
        benchmark_config = get_benchmark(args.benchmark)
        print(f"\n=== Using benchmark: {args.benchmark} ===")
        print(f"Description: {benchmark_config.get('description', 'N/A')}\n")
        
        args.patches_path = benchmark_config.get('patches_path', args.patches_path)
        args.adata_path = benchmark_config.get('adata_path', args.adata_path)
        args.filter_strategy = benchmark_config.get('filter_strategy', args.filter_strategy)
        args.n_genes = benchmark_config.get('n_genes', args.n_genes)
        args.seed = benchmark_config.get('seed', args.seed)
        
        hp = benchmark_config.get('hyperparams', {})
        args.learning_rate = hp.get('learning_rate', args.learning_rate)
        args.batch_size = hp.get('batch_size', args.batch_size)
        args.epochs = hp.get('epochs', args.epochs)
        args.weight_decay = hp.get('weight_decay', args.weight_decay)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Route to appropriate handler
    if args.dataset.lower() == 'hest':
        run_hest(args, output_dir, device)
    else:
        run_custom(args, output_dir, device)
    
    print(f"\n=== Training complete! ===")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
