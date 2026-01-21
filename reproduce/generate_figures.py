#!/usr/bin/env python3
"""
Generate Paper Figures

This script generates all figures for the paper from experiment results.

Usage:
    # Generate from default results directory (from run_experiments.py)
    python generate_figures.py

    # Generate from custom results directory
    python generate_figures.py --results_dir /path/to/your/results

    # Generate specific figures only
    python generate_figures.py --figures performance pathways

Requirements:
    Results directory must contain experiment outputs with:
    - metrics.json (evaluation metrics)
    - gene_list.pkl (genes used, for pathway analysis)
    - config.json (experiment configuration)
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Try to import gseapy for pathway analysis
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

TISSUES = ['lung', 'breast', 'colon', 'prostate', 'skin']
STRATEGIES = ['random', 'hvg', 'svg']
GENE_COUNTS = [50, 100]
MODELS = ['tinyvit_baseline', 'tinyvit_uni_distill', 'tinyvit_virchow_distill']

# Display names
STRATEGY_NAMES = {
    'random': 'Random',
    'hvg': 'Highly Variable',
    'svg': 'Spatially Variable'
}

MODEL_NAMES = {
    'tinyvit_baseline': 'TinyViT (Baseline)',
    'tinyvit_uni_distill': 'TinyViT + UNI2',
    'tinyvit_virchow_distill': 'TinyViT + Virchow2'
}

TISSUE_NAMES = {
    'lung': 'Lung',
    'breast': 'Breast',
    'colon': 'Colon',
    'prostate': 'Prostate',
    'skin': 'Skin'
}

# Colors
STRATEGY_COLORS = {
    'random': '#E74C3C',
    'hvg': '#3498DB',
    'svg': '#2ECC71'
}

MODEL_COLORS = {
    'tinyvit_baseline': '#95A5A6',
    'tinyvit_uni_distill': '#9B59B6',
    'tinyvit_virchow_distill': '#E67E22'
}


# =============================================================================
# Data Loading
# =============================================================================

def load_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all experiment results from a results directory.
    
    Expected structure:
        results_dir/
            {tissue}_{strategy}_{n_genes}_{model}/
                metrics.json
                config.json
                gene_list.pkl
    """
    records = []
    
    # Find all metrics.json files
    metrics_files = list(results_dir.rglob('metrics.json'))
    
    if not metrics_files:
        return pd.DataFrame()
    
    for metrics_file in metrics_files:
        try:
            # Load metrics
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Try to load config
            config_file = metrics_file.parent / 'config.json'
            config = {}
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            
            # Parse experiment info from directory name
            exp_name = metrics_file.parent.name
            
            record = {
                'experiment': exp_name,
                'path': str(metrics_file.parent),
                **metrics,
                **config,
            }
            
            # Try to extract metadata from experiment name
            # Expected format: {tissue}_{strategy}_{n_genes}_{model}
            parts = exp_name.split('_')
            
            for tissue in TISSUES:
                if tissue in exp_name.lower():
                    record['tissue'] = tissue
                    break
            
            for strategy in STRATEGIES:
                if strategy in exp_name.lower():
                    record['strategy'] = strategy
                    break
            
            for n in GENE_COUNTS:
                if f'_{n}_' in exp_name or exp_name.endswith(f'_{n}'):
                    record['n_genes'] = n
                    break
            
            for model in MODELS:
                if model in exp_name.lower():
                    record['model'] = model
                    break
            
            records.append(record)
            
        except Exception as e:
            print(f"Warning: Could not load {metrics_file}: {e}")
    
    return pd.DataFrame(records)


def load_gene_lists(results_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    """Load gene lists organized by tissue and strategy."""
    gene_lists = {}
    
    for gene_file in results_dir.rglob('gene_list.pkl'):
        try:
            with open(gene_file, 'rb') as f:
                genes = pickle.load(f)
            
            exp_name = gene_file.parent.name
            
            # Parse tissue and strategy
            tissue = None
            strategy = None
            
            for t in TISSUES:
                if t in exp_name.lower():
                    tissue = t
                    break
            
            for s in STRATEGIES:
                if s in exp_name.lower():
                    strategy = s
                    break
            
            if tissue and strategy:
                if tissue not in gene_lists:
                    gene_lists[tissue] = {}
                gene_lists[tissue][strategy] = genes if isinstance(genes, list) else list(genes)
                
        except Exception as e:
            print(f"Warning: Could not load {gene_file}: {e}")
    
    return gene_lists


def validate_results(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Check if results have required data."""
    if df.empty:
        return False
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Warning: Missing columns in results: {missing}")
        return False
    
    return True


# =============================================================================
# Figure A: ST Inference Performance
# =============================================================================

def plot_st_performance(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Generate the ST Inference Performance figure.
    
    Shows performance comparison across:
    - Gene selection strategies
    - Model architectures
    - Tissue types
    """
    if not validate_results(df, ['tissue', 'strategy', 'r2']):
        print("Error: Insufficient data for ST performance figure")
        print("Required: tissue, strategy, r2 columns")
        return None
    
    # Setup figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    tissues_present = [t for t in TISSUES if t in df['tissue'].unique()]
    strategies_present = [s for s in STRATEGIES if s in df['strategy'].unique()]
    
    if not tissues_present or not strategies_present:
        print("Error: No valid tissue/strategy combinations found")
        return None
    
    # Panel A: Strategy comparison across tissues
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(tissues_present))
    width = 0.25
    
    for i, strategy in enumerate(strategies_present):
        strategy_data = df[df['strategy'] == strategy]
        means = []
        stds = []
        
        for tissue in tissues_present:
            tissue_vals = strategy_data[strategy_data['tissue'] == tissue]['r2']
            means.append(tissue_vals.mean() if len(tissue_vals) > 0 else 0)
            stds.append(tissue_vals.std() if len(tissue_vals) > 1 else 0)
        
        ax1.bar(x + i * width, means, width, yerr=stds,
                label=STRATEGY_NAMES.get(strategy, strategy),
                color=STRATEGY_COLORS.get(strategy, '#333333'),
                capsize=3, edgecolor='black', linewidth=0.5)
    
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([TISSUE_NAMES.get(t, t) for t in tissues_present])
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('A. Gene Selection Strategy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel B: Model comparison (if model column exists)
    ax2 = fig.add_subplot(gs[1, 0])
    
    if 'model' in df.columns:
        models_present = [m for m in MODELS if m in df['model'].unique()]
        
        for i, model in enumerate(models_present):
            model_data = df[df['model'] == model]
            means = [model_data[model_data['tissue'] == t]['r2'].mean() 
                     for t in tissues_present]
            
            ax2.bar(x + i * width, means, width,
                    label=MODEL_NAMES.get(model, model),
                    color=MODEL_COLORS.get(model, '#333333'),
                    edgecolor='black', linewidth=0.5)
        
        ax2.legend(loc='upper right', fontsize=9)
    else:
        # Just show overall performance by tissue
        means = [df[df['tissue'] == t]['r2'].mean() for t in tissues_present]
        ax2.bar(x, means, 0.5, color='steelblue', edgecolor='black')
    
    ax2.set_xticks(x + width if 'model' in df.columns else x)
    ax2.set_xticklabels([TISSUE_NAMES.get(t, t) for t in tissues_present], 
                        rotation=45, ha='right')
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('B. Model Architecture Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Panel C: Gene count effect (if n_genes column exists)
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 'n_genes' in df.columns:
        gene_counts_present = sorted(df['n_genes'].dropna().unique())
        
        for i, n_genes in enumerate(gene_counts_present):
            gene_data = df[df['n_genes'] == n_genes]
            means = [gene_data[gene_data['tissue'] == t]['r2'].mean() 
                     for t in tissues_present]
            
            ax3.bar(x + i * 0.35, means, 0.35,
                    label=f'{int(n_genes)} genes',
                    color=plt.cm.viridis(i / max(1, len(gene_counts_present) - 1)),
                    edgecolor='black', linewidth=0.5)
        
        ax3.legend(loc='upper right')
    else:
        means = [df[df['tissue'] == t]['r2'].mean() for t in tissues_present]
        ax3.bar(x, means, 0.5, color='steelblue', edgecolor='black')
    
    ax3.set_xticks(x + 0.175 if 'n_genes' in df.columns else x)
    ax3.set_xticklabels([TISSUE_NAMES.get(t, t) for t in tissues_present], 
                        rotation=45, ha='right')
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('C. Effect of Gene Count', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.suptitle('ST Inference Performance', fontsize=18, fontweight='bold', y=0.98)
    
    # Save
    output_path = output_dir / 'st_inference_performance.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# Pathway Analysis Figures
# =============================================================================

def run_enrichment(gene_list: List[str], gene_sets: List[str] = None) -> Optional[pd.DataFrame]:
    """Run gene set enrichment analysis."""
    if not GSEAPY_AVAILABLE:
        print("Warning: gseapy not available. Install with: pip install gseapy")
        return None
    
    if gene_sets is None:
        gene_sets = ['GO_Biological_Process_2021', 'KEGG_2021_Human']
    
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism='human',
            outdir=None,
            no_plot=True,
            cutoff=0.05
        )
        return enr.results
    except Exception as e:
        print(f"Enrichment failed: {e}")
        return None


def plot_pathways_for_tissue(
    gene_lists: Dict[str, List[str]], 
    tissue: str,
    output_dir: Path
) -> Optional[Path]:
    """Generate pathway figure for a single tissue."""
    
    if not gene_lists:
        print(f"No gene lists found for {tissue}")
        return None
    
    strategies_present = [s for s in STRATEGIES if s in gene_lists]
    
    if not strategies_present:
        print(f"No valid strategies found for {tissue}")
        return None
    
    # Run enrichment for each strategy
    enrichment_results = {}
    for strategy in strategies_present:
        genes = gene_lists[strategy]
        print(f"  Running enrichment for {strategy} ({len(genes)} genes)...")
        results = run_enrichment(genes)
        if results is not None and len(results) > 0:
            enrichment_results[strategy] = results.head(5)
    
    if not enrichment_results:
        print(f"No enrichment results for {tissue}")
        return None
    
    # Create figure
    n_strategies = len(enrichment_results)
    fig, axes = plt.subplots(1, n_strategies, figsize=(6 * n_strategies, 8))
    
    if n_strategies == 1:
        axes = [axes]
    
    for idx, (strategy, df) in enumerate(enrichment_results.items()):
        ax = axes[idx]
        
        top5 = df.head(5).copy()
        
        if 'Adjusted P-value' in top5.columns:
            top5['neg_log_pval'] = -np.log10(top5['Adjusted P-value'].clip(lower=1e-10))
        else:
            continue
        
        if 'Term' in top5.columns:
            top5['Term_short'] = top5['Term'].apply(
                lambda x: x[:40] + '...' if len(str(x)) > 40 else x
            )
        else:
            continue
        
        y_pos = np.arange(len(top5))
        color = STRATEGY_COLORS.get(strategy, '#333333')
        
        ax.barh(y_pos, top5['neg_log_pval'], color=color,
                edgecolor='black', linewidth=0.5, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top5['Term_short'], fontsize=10)
        ax.invert_yaxis()
        
        ax.set_xlabel('-log₁₀(Adjusted P-value)', fontsize=11)
        ax.set_title(f'{STRATEGY_NAMES.get(strategy, strategy)} Genes',
                     fontsize=14, fontweight='bold', color=color)
        
        ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Top 5 Pathways per Gene Set — {TISSUE_NAMES.get(tissue, tissue)}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / f'pathways_{tissue}.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_all_pathways(gene_lists: Dict[str, Dict[str, List[str]]], output_dir: Path):
    """Generate pathway figures for all tissues."""
    for tissue in TISSUES:
        if tissue in gene_lists:
            print(f"\nProcessing pathways for {TISSUE_NAMES.get(tissue, tissue)}...")
            plot_pathways_for_tissue(gene_lists[tissue], tissue, output_dir)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate paper figures from experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected Results Format:
    results_dir/
        {experiment_name}/
            metrics.json       # Must contain: r2, pearson_mean, etc.
            gene_list.pkl      # List of gene names (for pathway analysis)
            config.json        # Experiment configuration (optional)

Examples:
    # Generate from default location (after running run_experiments.py)
    python generate_figures.py

    # Generate from custom results
    python generate_figures.py --results_dir /path/to/my/results

    # Generate only performance figure
    python generate_figures.py --figures performance
        """
    )
    
    parser.add_argument('--results_dir', type=str, default='./outputs',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Directory to save generated figures')
    parser.add_argument('--figures', nargs='+', 
                        default=['performance', 'pathways'],
                        choices=['performance', 'pathways', 'all'],
                        help='Which figures to generate')
    
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check results directory exists
    if not results_dir.exists():
        print(f"\nError: Results directory does not exist: {results_dir}")
        print("\nTo generate figures, you need experiment results.")
        print("Either:")
        print("  1. Run experiments first:")
        print("     python run_experiments.py --data_root /path/to/hest_data")
        print("  2. Provide your own results directory:")
        print("     python generate_figures.py --results_dir /path/to/your/results")
        print("\nExpected results format:")
        print("  results_dir/")
        print("    {experiment}/")
        print("      metrics.json    # Required: r2, pearson_mean, etc.")
        print("      gene_list.pkl   # Required for pathway analysis")
        print("      config.json     # Optional: experiment config")
        sys.exit(1)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    
    # Load results
    print(f"\nLoading results from: {results_dir}")
    df = load_all_results(results_dir)
    
    if df.empty:
        print(f"\nError: No results found in {results_dir}")
        print("\nMake sure the directory contains experiment outputs with metrics.json files.")
        print("Run experiments first with: python run_experiments.py --data_root /path/to/data")
        sys.exit(1)
    
    print(f"Loaded {len(df)} experiment results")
    
    # Normalize figure selection
    figures_to_generate = args.figures
    if 'all' in figures_to_generate:
        figures_to_generate = ['performance', 'pathways']
    
    generated = []
    
    # Generate ST Performance figure
    if 'performance' in figures_to_generate:
        print("\n" + "="*60)
        print("Generating ST Inference Performance Figure")
        print("="*60)
        
        path = plot_st_performance(df, output_dir)
        if path:
            generated.append(path)
    
    # Generate Pathway figures
    if 'pathways' in figures_to_generate:
        print("\n" + "="*60)
        print("Generating Pathway Analysis Figures")
        print("="*60)
        
        if not GSEAPY_AVAILABLE:
            print("\nWarning: gseapy not installed. Skipping pathway analysis.")
            print("Install with: pip install gseapy")
        else:
            gene_lists = load_gene_lists(results_dir)
            
            if not gene_lists:
                print("\nNo gene lists found for pathway analysis.")
                print("Make sure experiments saved gene_list.pkl files.")
            else:
                plot_all_pathways(gene_lists, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("Figure Generation Complete")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    if generated:
        print("\nGenerated figures:")
        for path in generated:
            print(f"  - {path}")
    
    # List all files in output directory
    all_outputs = list(output_dir.glob('*.pdf')) + list(output_dir.glob('*.png'))
    if all_outputs:
        print(f"\nAll outputs ({len(all_outputs)} files):")
        for f in sorted(all_outputs):
            print(f"  - {f.name}")


if __name__ == '__main__':
    main()

