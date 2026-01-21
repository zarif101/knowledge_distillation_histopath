#!/usr/bin/env python
"""
Unified knowledge distillation script for HEST and WSI Classification.

Distills knowledge from large foundation models (UNI2, Virchow2) to TinyViT.

Usage:
    # HEST with knowledge distillation
    python pipeline/distill.py --dataset hest --data_path /path/to/data \
        --teacher uni2 --student tinyvit \
        --filter_strategy hvg --n_genes 100 --output_dir ./results/distill_hest
    
    # WSI Classification with distillation
    python pipeline/distill.py --dataset wsiclass --data_path /path/to/patches \
        --metadata_path /path/to/metadata.csv \
        --teacher virchow2 --student tinyvit --output_dir ./results/distill_wsi
    
    # Using predefined benchmark
    python pipeline/distill.py --benchmark hest_hvg_100 --output_dir ./results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge distillation for histopathology models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, choices=['hest', 'wsiclass'],
                        help='Dataset type')
    parser.add_argument('--data_path', type=str,
                        help='Path to dataset')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to metadata (for wsiclass)')
    
    # Benchmark (alternative to manual config)
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Use predefined benchmark configuration')
    parser.add_argument('--list_benchmarks', action='store_true',
                        help='List available benchmarks and exit')
    
    # Model selection
    parser.add_argument('--teacher', type=str, default='uni2',
                        choices=['uni2', 'virchow2'],
                        help='Teacher model')
    parser.add_argument('--student', type=str, default='tinyvit',
                        choices=['tinyvit'],
                        help='Student model')
    
    # Distillation type
    parser.add_argument('--distill_type', type=str, default='feature',
                        choices=['feature', 'logit'],
                        help='Type of distillation')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for distillation loss vs task loss')
    
    # Gene filtering (HEST only)
    parser.add_argument('--filter_strategy', type=str, default='random',
                        choices=['random', 'hvg', 'svg'],
                        help='Gene filtering strategy')
    parser.add_argument('--n_genes', type=int, default=100,
                        help='Number of genes to select')
    parser.add_argument('--gene_list_path', type=str, default=None,
                        help='Path to pre-computed gene list (pkl)')
    
    # Split configuration
    parser.add_argument('--train_split_path', type=str, default=None,
                        help='Path to train split file')
    parser.add_argument('--val_split_path', type=str, default=None,
                        help='Path to validation split file')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation ratio (if no split files)')
    
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


def save_config(args, output_dir: Path, gene_list=None):
    """Save full configuration for reproducibility."""
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()
    config['pytorch_version'] = torch.__version__
    config['mode'] = 'distillation'
    
    if gene_list is not None:
        config['n_genes_actual'] = len(gene_list)
    
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Saved config to {config_path}")


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
        
        # Override args with benchmark values
        args.dataset = benchmark_config.get('dataset', args.dataset)
        args.data_path = benchmark_config.get('data_path', args.data_path)
        args.metadata_path = benchmark_config.get('metadata_path', args.metadata_path)
        args.filter_strategy = benchmark_config.get('filter_strategy', args.filter_strategy)
        args.n_genes = benchmark_config.get('n_genes', args.n_genes)
        args.seed = benchmark_config.get('seed', args.seed)
        
        # Override hyperparams
        hp = benchmark_config.get('hyperparams', {})
        args.learning_rate = hp.get('learning_rate', args.learning_rate)
        args.batch_size = hp.get('batch_size', args.batch_size)
        args.epochs = hp.get('epochs', args.epochs)
        args.weight_decay = hp.get('weight_decay', args.weight_decay)
    
    # Validate required args
    if args.dataset is None:
        raise ValueError("Must specify --dataset or --benchmark")
    if args.data_path is None:
        raise ValueError("Must specify --data_path or --benchmark with data_path")
    
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
    
    # Import utilities
    from utils.split_utils import get_samples_from_paths, get_train_val_split
    from utils.gene_filtering import get_gene_list, save_gene_list, load_gene_list
    from modeling.models import get_model, get_transforms
    
    # Get samples and splits
    all_samples = get_samples_from_paths(args.data_path)
    train_samples, val_samples = get_train_val_split(
        samples=all_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_split_path=args.train_split_path,
        val_split_path=args.val_split_path
    )
    
    # Handle gene filtering (HEST only)
    gene_list = None
    if args.dataset == 'hest':
        if args.gene_list_path:
            gene_list = load_gene_list(args.gene_list_path)
            print(f"Loaded {len(gene_list)} genes from {args.gene_list_path}")
        else:
            import scanpy as sc
            sample_path = Path(args.data_path) / train_samples[0]
            h5ad_files = list(sample_path.glob("*.h5ad"))
            if h5ad_files:
                adata = sc.read_h5ad(h5ad_files[0])
                gene_list = get_gene_list(
                    adata,
                    strategy=args.filter_strategy,
                    n_genes=args.n_genes,
                    seed=args.seed
                )
                gene_list_save_path = output_dir / 'gene_list.pkl'
                save_gene_list(gene_list, gene_list_save_path)
                print(f"Selected {len(gene_list)} genes using {args.filter_strategy} strategy")
    
    # Save config
    save_config(args, output_dir, gene_list)
    
    # Get teacher and student models
    teacher_model, teacher_transforms = get_model(args.teacher, args.dataset)
    student_model, student_transforms = get_model(args.student, args.dataset)
    
    # Use teacher transforms for consistency
    transforms = teacher_transforms
    
    # Determine number of outputs
    if args.dataset == 'hest':
        num_outputs = len(gene_list)
    else:
        import pandas as pd
        metadata = pd.read_csv(args.metadata_path)
        num_outputs = metadata['label'].nunique()
    
    # Build models with heads
    from modeling.models import build_model_with_head
    teacher_model = build_model_with_head(teacher_model, num_outputs, args.dataset)
    student_model = build_model_with_head(student_model, num_outputs, args.dataset)
    
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    # Hyperparameters dict
    hyperparams = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'temperature': args.temperature,
        'alpha': args.alpha
    }
    
    # Run distillation based on dataset and type
    if args.dataset == 'hest':
        if args.distill_type == 'feature':
            from utils.train_HEST import distill_HEST_data_featurelevel
            distill_HEST_data_featurelevel(
                data_path=args.data_path,
                train_samples=train_samples,
                val_samples=val_samples,
                gene_list=gene_list,
                log_dir=str(output_dir),
                teacher_model=teacher_model,
                student_model=student_model,
                transforms=transforms,
                hyperparams_dict=hyperparams
            )
        else:
            from utils.train_HEST import distill_HEST_data
            distill_HEST_data(
                data_path=args.data_path,
                train_samples=train_samples,
                val_samples=val_samples,
                gene_list=gene_list,
                log_dir=str(output_dir),
                teacher_model=teacher_model,
                student_model=student_model,
                transforms=transforms,
                hyperparams_dict=hyperparams
            )
    else:  # wsiclass
        from utils.train_WSICLASS import distill_WSICLASS_data
        distill_WSICLASS_data(
            patches_path=args.data_path,
            metadata_path=args.metadata_path,
            train_samples=train_samples,
            val_samples=val_samples,
            log_dir=str(output_dir),
            teacher_model=teacher_model,
            student_model=student_model,
            transforms=transforms,
            hyperparams_dict=hyperparams
        )
    
    print(f"\n=== Distillation complete! ===")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

