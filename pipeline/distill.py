#!/usr/bin/env python
"""
Knowledge distillation script for HEST and custom datasets.

Distills knowledge from large foundation models (UNI2, Virchow2) to TinyViT.

HEST (default): Just provide patches_path and adata_path
Custom datasets: Register an adapter and use --dataset your_adapter_name

Usage:
    # HEST (default) - any HEST-format dataset
    python pipeline/distill.py \
        --patches_path /path/to/patches/ \
        --adata_path /path/to/st/ \
        --teacher uni2 \
        --student tinyvit \
        --filter_strategy hvg \
        --n_genes 100 \
        --output_dir ./results
    
    # Custom dataset via adapter
    python pipeline/distill.py \
        --dataset my_custom_adapter \
        --data_path /path/to/my/data \
        --teacher uni2 \
        --student tinyvit \
        --output_dir ./results
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
        description="Knowledge distillation for HEST or custom datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset selection (default: hest)
    parser.add_argument('--dataset', type=str, default='hest',
                        help='Dataset adapter to use. Default "hest" for HEST format.')
    
    # HEST-specific paths
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
    parser.add_argument('--teacher', type=str, default='uni2',
                        choices=['uni2', 'virchow2'],
                        help='Teacher model (foundation model)')
    parser.add_argument('--student', type=str, default='tinyvit',
                        choices=['tinyvit'],
                        help='Student model (lightweight)')
    
    # Distillation settings
    parser.add_argument('--distill_type', type=str, default='feature',
                        choices=['feature', 'logit'],
                        help='Type of distillation')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for distillation loss vs task loss')
    
    # Gene filtering (HEST)
    parser.add_argument('--filter_strategy', type=str, default='random',
                        choices=['random', 'hvg', 'svg'],
                        help='[HEST] Gene filtering strategy')
    parser.add_argument('--n_genes', type=int, default=100,
                        help='[HEST] Number of genes to select')
    parser.add_argument('--gene_list_path', type=str, default=None,
                        help='[HEST] Path to pre-computed gene list (pkl)')
    
    # Split configuration
    parser.add_argument('--train_split_path', type=str, default=None)
    parser.add_argument('--val_split_path', type=str, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    
    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def save_config(args, output_dir: Path, extra_info: dict = None):
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()
    config['pytorch_version'] = torch.__version__
    config['mode'] = 'distillation'
    if extra_info:
        config.update(extra_info)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)


def get_sample_ids_from_h5_dir(directory: str) -> list:
    return sorted([f.stem for f in Path(directory).glob("*.h5")])


def run_hest(args, output_dir, device):
    """Run distillation for HEST format dataset."""
    from utils.split_utils import get_train_val_split, save_split_file
    from utils.gene_filtering import get_gene_list, save_gene_list, load_gene_list
    from modeling.models import get_model, build_model_with_head
    
    if args.patches_path is None or args.adata_path is None:
        raise ValueError("HEST requires --patches_path and --adata_path")
    
    # Get samples and splits
    all_samples = get_sample_ids_from_h5_dir(args.patches_path)
    print(f"Found {len(all_samples)} samples")
    
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
    else:
        import scanpy as sc
        adata_file = Path(args.adata_path) / f"{train_samples[0]}.h5ad"
        adata = sc.read_h5ad(adata_file)
        gene_list = get_gene_list(adata, strategy=args.filter_strategy,
                                   n_genes=args.n_genes, seed=args.seed)
        save_gene_list(gene_list, output_dir / 'gene_list.pkl')
    
    print(f"Using {len(gene_list)} genes")
    save_config(args, output_dir, {'n_genes_actual': len(gene_list)})
    
    # Build models
    teacher_model, teacher_transforms = get_model(args.teacher)
    student_model, _ = get_model(args.student)
    
    teacher_model = build_model_with_head(teacher_model, len(gene_list), task='regression')
    student_model = build_model_with_head(student_model, len(gene_list), task='regression')
    
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    # Train
    hyperparams = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'temperature': args.temperature,
        'alpha': args.alpha
    }
    
    if args.distill_type == 'feature':
        from utils.train_HEST import distill_HEST_data_featurelevel
        distill_HEST_data_featurelevel(
            patches_path=args.patches_path,
            adata_path=args.adata_path,
            train_samples=train_samples,
            val_samples=val_samples,
            gene_list=gene_list,
            log_dir=str(output_dir) + '/',
            teacher_model=teacher_model,
            student_model=student_model,
            transforms=teacher_transforms,
            hyperparams_dict=hyperparams
        )
    else:
        from utils.train_HEST import distill_HEST_data
        distill_HEST_data(
            patches_path=args.patches_path,
            adata_path=args.adata_path,
            train_samples=train_samples,
            val_samples=val_samples,
            gene_list=gene_list,
            log_dir=str(output_dir) + '/',
            teacher_model=teacher_model,
            student_model=student_model,
            transforms=teacher_transforms,
            hyperparams_dict=hyperparams
        )


def run_custom(args, output_dir, device):
    """Run distillation for custom dataset via adapter."""
    from utils.dataset_adapter import get_dataset_adapter
    from modeling.models import get_model, build_model_with_head
    from utils.train_HEST import train_student_teacher
    from utils import custom_losses
    
    adapter = get_dataset_adapter(args.dataset)
    print(f"Using adapter: {args.dataset} (task: {adapter.task_type})")
    
    save_config(args, output_dir)
    
    # Build models
    teacher_model, teacher_transforms = get_model(args.teacher)
    student_model, _ = get_model(args.student)
    
    # Load data
    train_loader, val_loader = adapter.load_data(
        train_samples=[],
        val_samples=[],
        transforms=teacher_transforms,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    num_outputs = adapter.get_num_outputs()
    teacher_model = build_model_with_head(teacher_model, num_outputs, task=adapter.task_type)
    student_model = build_model_with_head(student_model, num_outputs, task=adapter.task_type)
    
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    # Distillation loss
    loss_fn = custom_losses.DistillationLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    
    train_student_teacher(teacher_model, student_model, train_loader, val_loader,
                          loss_fn, optimizer, args.epochs, device,
                          str(output_dir / 'log.txt'), str(output_dir) + '/')


def main():
    args = parse_args()
    
    if args.list_benchmarks:
        from utils.benchmarks import print_benchmark_info
        print_benchmark_info()
        return
    
    if args.benchmark:
        from utils.benchmarks import get_benchmark
        benchmark_config = get_benchmark(args.benchmark)
        print(f"\n=== Using benchmark: {args.benchmark} ===\n")
        
        args.patches_path = benchmark_config.get('patches_path', args.patches_path)
        args.adata_path = benchmark_config.get('adata_path', args.adata_path)
        args.filter_strategy = benchmark_config.get('filter_strategy', args.filter_strategy)
        args.n_genes = benchmark_config.get('n_genes', args.n_genes)
        args.seed = benchmark_config.get('seed', args.seed)
        
        hp = benchmark_config.get('hyperparams', {})
        for k, v in hp.items():
            if hasattr(args, k):
                setattr(args, k, v)
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.dataset.lower() == 'hest':
        run_hest(args, output_dir, device)
    else:
        run_custom(args, output_dir, device)
    
    print(f"\n=== Distillation complete! ===")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
