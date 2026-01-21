#!/usr/bin/env python3
"""
Run All Paper Experiments

This script runs all experiments needed to reproduce the paper results:
- Multiple tissue types (lung, breast, colon, prostate, skin)
- Multiple gene selection strategies (random, HVG, SVG)
- Multiple model configurations (TinyViT baseline, TinyViT + distillation)

Usage:
    # Run all experiments
    python run_experiments.py --data_root /path/to/hest_data --output_dir ./outputs

    # Run specific tissue only
    python run_experiments.py --data_root /path/to/hest_data --tissues lung breast

    # Run specific strategies only
    python run_experiments.py --data_root /path/to/hest_data --strategies hvg svg

    # Dry run (show what would be run without executing)
    python run_experiments.py --data_root /path/to/hest_data --dry_run

Requirements:
    - GPU with sufficient memory (recommended: 24GB+ for distillation)
    - HEST data downloaded and organized by tissue type
    - HuggingFace token set for foundation model access (UNI2, Virchow2)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


# =============================================================================
# Experiment Configuration
# =============================================================================

# Tissue types and their data paths (relative to data_root)
TISSUES = {
    'lung': {
        'patches_subdir': 'hest_data_lung/patches',
        'adata_subdir': 'hest_data_lung/st',
    },
    'breast': {
        'patches_subdir': 'hest_data_breast/patches',
        'adata_subdir': 'hest_data_breast/st',
    },
    'colon': {
        'patches_subdir': 'hest_data_colon/patches',
        'adata_subdir': 'hest_data_colon/st',
    },
    'prostate': {
        'patches_subdir': 'hest_data_prostate/patches',
        'adata_subdir': 'hest_data_prostate/st',
    },
    'skin': {
        'patches_subdir': 'hest_data_skin/patches',
        'adata_subdir': 'hest_data_skin/st',
    },
}

# Gene selection strategies
STRATEGIES = ['random', 'hvg', 'svg']

# Gene counts to test
GENE_COUNTS = [50, 100]

# Model configurations
MODEL_CONFIGS = {
    'tinyvit_baseline': {
        'script': 'finetune',
        'model': 'tinyvit',
        'description': 'TinyViT baseline (no distillation)',
    },
    'tinyvit_uni_distill': {
        'script': 'distill',
        'model': 'tinyvit',
        'teacher': 'uni2',
        'description': 'TinyViT with UNI2 distillation',
    },
    'tinyvit_virchow_distill': {
        'script': 'distill',
        'model': 'tinyvit',
        'teacher': 'virchow2',
        'description': 'TinyViT with Virchow2 distillation',
    },
}

# Training hyperparameters
DEFAULT_HYPERPARAMS = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'seed': 42,
}


# =============================================================================
# Experiment Runner
# =============================================================================

def build_experiment_list(
    tissues: List[str],
    strategies: List[str],
    gene_counts: List[int],
    model_configs: List[str],
) -> List[Dict[str, Any]]:
    """Build list of all experiments to run."""
    experiments = []
    
    for tissue in tissues:
        for strategy in strategies:
            for n_genes in gene_counts:
                for model_name in model_configs:
                    config = MODEL_CONFIGS[model_name]
                    
                    experiment = {
                        'name': f'{tissue}_{strategy}_{n_genes}_{model_name}',
                        'tissue': tissue,
                        'strategy': strategy,
                        'n_genes': n_genes,
                        'model_name': model_name,
                        'model_config': config,
                    }
                    experiments.append(experiment)
    
    return experiments


def build_command(
    experiment: Dict[str, Any],
    data_root: Path,
    output_dir: Path,
    hyperparams: Dict[str, Any],
) -> List[str]:
    """Build the command to run an experiment."""
    tissue_config = TISSUES[experiment['tissue']]
    model_config = experiment['model_config']
    
    # Determine which script to use
    script = model_config['script']
    
    # Build base command
    cmd = [
        sys.executable, '-m', f'pipeline.{script}',
        '--patches_path', str(data_root / tissue_config['patches_subdir']),
        '--adata_path', str(data_root / tissue_config['adata_subdir']),
        '--model', model_config['model'],
        '--filter_strategy', experiment['strategy'],
        '--n_genes', str(experiment['n_genes']),
        '--output_dir', str(output_dir / experiment['name']),
        '--epochs', str(hyperparams.get('epochs', 50)),
        '--batch_size', str(hyperparams.get('batch_size', 32)),
        '--lr', str(hyperparams.get('learning_rate', 1e-4)),
        '--seed', str(hyperparams.get('seed', 42)),
    ]
    
    # Add teacher model for distillation
    if script == 'distill' and 'teacher' in model_config:
        cmd.extend(['--teacher', model_config['teacher']])
    
    return cmd


def run_experiment(
    experiment: Dict[str, Any],
    data_root: Path,
    output_dir: Path,
    hyperparams: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment and return result."""
    cmd = build_command(experiment, data_root, output_dir, hyperparams)
    
    result = {
        'experiment': experiment['name'],
        'command': ' '.join(cmd),
        'status': 'pending',
        'start_time': None,
        'end_time': None,
        'error': None,
    }
    
    if dry_run:
        result['status'] = 'dry_run'
        return result
    
    print(f"\n{'='*60}")
    print(f"Running: {experiment['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result['start_time'] = datetime.now().isoformat()
    
    try:
        # Change to project root directory for imports to work
        project_root = Path(__file__).parent.parent
        
        process = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=False,  # Show output in real-time
            text=True,
        )
        
        if process.returncode == 0:
            result['status'] = 'success'
        else:
            result['status'] = 'failed'
            result['error'] = f'Exit code: {process.returncode}'
            
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    result['end_time'] = datetime.now().isoformat()
    
    return result


def run_evaluation(
    experiment: Dict[str, Any],
    data_root: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run evaluation for a completed experiment."""
    tissue_config = TISSUES[experiment['tissue']]
    exp_output_dir = output_dir / experiment['name']
    
    cmd = [
        sys.executable, '-m', 'pipeline.evaluate',
        '--model_path', str(exp_output_dir / 'best_model.pt'),
        '--patches_path', str(data_root / tissue_config['patches_subdir']),
        '--adata_path', str(data_root / tissue_config['adata_subdir']),
        '--output_dir', str(exp_output_dir),
        '--compute_spatial',  # Enable SSIM computation
    ]
    
    result = {
        'experiment': f'{experiment["name"]}_eval',
        'command': ' '.join(cmd),
        'status': 'pending',
    }
    
    if dry_run:
        result['status'] = 'dry_run'
        return result
    
    print(f"\nEvaluating: {experiment['name']}")
    
    try:
        project_root = Path(__file__).parent.parent
        process = subprocess.run(cmd, cwd=str(project_root), capture_output=False)
        result['status'] = 'success' if process.returncode == 0 else 'failed'
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run all paper experiments for reproducibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments
    python run_experiments.py --data_root /data/hest

    # Run only lung and breast tissues
    python run_experiments.py --data_root /data/hest --tissues lung breast

    # Run only HVG and SVG strategies
    python run_experiments.py --data_root /data/hest --strategies hvg svg

    # Run only baseline models (no distillation)
    python run_experiments.py --data_root /data/hest --models tinyvit_baseline

    # Dry run to see what would be executed
    python run_experiments.py --data_root /data/hest --dry_run
        """
    )
    
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing HEST data folders')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save experiment results')
    parser.add_argument('--tissues', nargs='+', default=list(TISSUES.keys()),
                        choices=list(TISSUES.keys()),
                        help='Tissue types to run experiments for')
    parser.add_argument('--strategies', nargs='+', default=STRATEGIES,
                        choices=STRATEGIES,
                        help='Gene selection strategies to test')
    parser.add_argument('--gene_counts', nargs='+', type=int, default=GENE_COUNTS,
                        help='Number of genes to select')
    parser.add_argument('--models', nargs='+', default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model configurations to run')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip evaluation after training')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--resume', action='store_true',
                        help='Skip experiments that already have results')
    
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate data root
    if not data_root.exists():
        print(f"Error: Data root does not exist: {data_root}")
        sys.exit(1)
    
    # Build hyperparameters
    hyperparams = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
    }
    
    # Build experiment list
    experiments = build_experiment_list(
        tissues=args.tissues,
        strategies=args.strategies,
        gene_counts=args.gene_counts,
        model_configs=args.models,
    )
    
    print(f"\n{'='*60}")
    print(f"StainFactor Paper Experiments")
    print(f"{'='*60}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Tissues: {args.tissues}")
    print(f"Strategies: {args.strategies}")
    print(f"Gene counts: {args.gene_counts}")
    print(f"Models: {args.models}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("DRY RUN - Commands that would be executed:\n")
    
    # Track results
    all_results = []
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {experiment['name']}")
        
        # Check if should skip (resume mode)
        if args.resume:
            metrics_file = output_dir / experiment['name'] / 'metrics.json'
            if metrics_file.exists():
                print(f"  Skipping (results exist)")
                continue
        
        # Run training
        result = run_experiment(
            experiment, data_root, output_dir, hyperparams, args.dry_run
        )
        all_results.append(result)
        
        if args.dry_run:
            print(f"  {result['command']}")
            continue
        
        # Run evaluation if training succeeded
        if result['status'] == 'success' and not args.skip_eval:
            eval_result = run_evaluation(
                experiment, data_root, output_dir, args.dry_run
            )
            all_results.append(eval_result)
    
    # Save experiment log
    log_file = output_dir / 'experiment_log.json'
    with open(log_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'args': vars(args),
            'experiments': all_results,
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    success = sum(1 for r in all_results if r['status'] == 'success')
    failed = sum(1 for r in all_results if r['status'] == 'failed')
    errors = sum(1 for r in all_results if r['status'] == 'error')
    
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"\nLog saved to: {log_file}")
    
    if not args.dry_run:
        print(f"\nNext step: Generate figures with:")
        print(f"  python reproduce/generate_figures.py --results_dir {output_dir}")


if __name__ == '__main__':
    main()

