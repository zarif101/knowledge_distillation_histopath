# StainFactor

An open-source framework for **assessing the impact of target selection methods on virtual staining model performance**.

The framework includes plug-and-play Python scripts that train and evaluate model architectures including foundation models, knowledge distillation, and lightweight models for virtual staining datasets filtered using a variety of target set selection strategies. Users can choose to assess custom existing trained models or retrain using standardized preprocessing workflows, enabling flexible usage to suit diverse model evaluations. The package supports evaluation on standardized benchmark datasets as well as additional datasets implemented by users.

---

## Table of Contents

- [Key Features](#key-features)
- [Benchmark Data](#benchmark-data)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [HEST Data Format](#hest-data-format)
- [Gene Filtering Strategies](#gene-filtering-strategies)
- [Models](#models)
- [CLI Reference](#cli-reference)
- [Predefined Benchmarks](#predefined-benchmarks)
- [Output Structure](#output-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Custom Datasets](#custom-datasets)
- [Train/Validation Splits](#trainvalidation-splits)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Reproducing Paper Figures](#reproducing-paper-figures)

---

## Key Features

- **Foundation Models**: UNI2, Virchow2
- **Lightweight Models**: TinyViT (5M params)
- **Knowledge Distillation**: Transfer knowledge from large teachers to small students
- **Target Selection Strategies**: Random, Highly Variable Genes (HVG), Spatially Variable Genes (SVG/Moran's I)
- **Standardized Benchmarks**: Pre-configured experiments across multiple tissue types
- **Custom Datasets**: Extensible adapter system for your own data formats

---

## Benchmark Data

The paper benchmarks use the publicly available [HEST dataset](https://huggingface.co/datasets/MahmoodLab/hest) (Jaume et al., 2024), which provides spatial transcriptomics data paired with H&E whole slide images across multiple tissue types.

**Tissue types used in the paper:**
- Lung, Breast, Colon, Prostate, Skin

See `reproduce/README.md` for detailed data download instructions.

---

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone <repository_url>
cd stainfactor

# Create environment from specification
conda env create -f environment.yml
conda activate stainfactor
```

### Option 2: Pip

```bash
# Clone the repository
git clone <repository_url>
cd stainfactor

# Install dependencies
pip install torch torchvision timm scanpy h5py scikit-learn scipy pandas numpy
pip install huggingface_hub    # For foundation models (UNI2, Virchow2)
pip install scikit-image       # For SSIM spatial metrics
pip install gseapy             # For pathway analysis (figure reproduction)
```

---

## Quick Start

### 1. Fine-tune a Model

```bash
python -m pipeline.finetune \
    --patches_path /data/hest/patches/ \
    --adata_path /data/hest/st/ \
    --model tinyvit \
    --filter_strategy hvg \
    --n_genes 100 \
    --output_dir ./results/finetune
```

### 2. Knowledge Distillation

```bash
python -m pipeline.distill \
    --patches_path /data/hest/patches/ \
    --adata_path /data/hest/st/ \
    --teacher uni2 \
    --student tinyvit \
    --filter_strategy hvg \
    --n_genes 100 \
    --output_dir ./results/distill
```

### 3. Evaluate

```bash
python -m pipeline.evaluate \
    --model_path ./results/finetune/model_epoch99 \
    --patches_path /data/hest/patches/ \
    --adata_path /data/hest/st/ \
    --config_path ./results/finetune/config.json \
    --output_dir ./results/eval \
    --compute_spatial
```

---

## HEST Data Format

The pipeline is designed for **HEST-format** spatial transcriptomics data. This format works for any tissue type (lung, breast, skin, etc.):

```
your_data/
├── patches/                    # --patches_path
│   ├── SAMPLE_001.h5          # H5 file with 'img' and 'barcode' arrays
│   ├── SAMPLE_002.h5
│   └── ...
│
└── st/                         # --adata_path
    ├── SAMPLE_001.h5ad        # AnnData with gene expression + spatial coords
    ├── SAMPLE_002.h5ad
    └── ...
```

**H5 patch files** contain:
- `img`: Array of patch images
- `barcode`: Array of spot barcodes

**h5ad files** are standard AnnData objects with gene expression matrices.

---

## Gene Filtering Strategies

Choose which genes to predict using `--filter_strategy`:

| Strategy | Flag | Description |
|----------|------|-------------|
| **Random** | `--filter_strategy random` | Randomly select N genes |
| **Highly Variable** | `--filter_strategy hvg` | Top N genes by variance (scanpy) |
| **Spatially Variable** | `--filter_strategy svg` | Top N genes by Moran's I (squidpy) |
| **Custom** | `--gene_list_path genes.pkl` | Provide your own gene list |

```bash
# Example: Compare strategies
python -m pipeline.finetune --filter_strategy random --n_genes 100 --output_dir ./random
python -m pipeline.finetune --filter_strategy hvg --n_genes 100 --output_dir ./hvg
python -m pipeline.finetune --filter_strategy svg --n_genes 100 --output_dir ./svg
```

---

## Models

| Model | Type | Flag | Description |
|-------|------|------|-------------|
| TinyViT | Lightweight | `--model tinyvit` | 5M params, fast inference |
| UNI2 | Foundation | `--model uni2` | MahmoodLab's foundation model |
| Virchow2 | Foundation | `--model virchow2` | Paige AI's foundation model |

---

## CLI Reference

### `pipeline.finetune`

Fine-tune a model on HEST data.

| Argument | Description | Default |
|----------|-------------|---------|
| `--patches_path` | Directory with H5 patch files | Required |
| `--adata_path` | Directory with h5ad expression files | Required |
| `--output_dir` | Output directory | Required |
| `--model` | Model: `tinyvit`, `uni2`, `virchow2` | `tinyvit` |
| `--filter_strategy` | Gene selection: `random`, `hvg`, `svg` | `random` |
| `--n_genes` | Number of genes | `100` |
| `--gene_list_path` | Pre-computed gene list (pkl) | None |
| `--train_split_path` | Custom train samples file | None |
| `--val_split_path` | Custom val samples file | None |
| `--val_ratio` | Validation ratio | `0.2` |
| `--batch_size` | Batch size | `32` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--epochs` | Training epochs | `100` |
| `--seed` | Random seed | `42` |

### `pipeline.distill`

Knowledge distillation from teacher to student.

| Argument | Description | Default |
|----------|-------------|---------|
| `--teacher` | Teacher model: `uni2`, `virchow2` | `uni2` |
| `--student` | Student model: `tinyvit` | `tinyvit` |
| `--distill_type` | `feature` or `logit` level | `feature` |
| `--temperature` | Distillation temperature | `4.0` |
| `--alpha` | Distillation loss weight | `0.5` |
| *(plus all finetune args)* | | |

### `pipeline.evaluate`

Evaluate a trained model.

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to trained model | Required |
| `--config_path` | Config from training (auto-loads paths) | None |
| `--eval_split_path` | Custom evaluation samples | None |
| `--use_val` | Use val split from training | False |
| `--compute_spatial` | Compute SSIM metrics | False |
| *(plus patches_path, adata_path, gene_list_path)* | | |

---

## Predefined Benchmarks

Benchmarks are **shortcuts** that auto-fill data paths, gene filtering strategy, and hyperparameters. Instead of typing all arguments manually, just use `--benchmark`:

```bash
# Without benchmark (verbose):
python -m pipeline.finetune \
    --patches_path /multimodal/dist/hest_data_lung/patches/ \
    --adata_path /multimodal/dist/hest_data_lung/st/ \
    --filter_strategy hvg --n_genes 100 \
    --learning_rate 1e-4 --batch_size 32 --epochs 100 \
    --output_dir ./results

# With benchmark (simple):
python -m pipeline.finetune --benchmark lung_hvg_100 --output_dir ./results
```

### Available Benchmarks

4 tissue types × 3 strategies × 2 gene counts = **24 benchmarks**

| Tissue | Benchmarks |
|--------|------------|
| **Lung** | `lung_random_50`, `lung_random_100`, `lung_hvg_50`, `lung_hvg_100`, `lung_svg_50`, `lung_svg_100` |
| **Breast** | `breast_random_50`, `breast_random_100`, `breast_hvg_50`, `breast_hvg_100`, `breast_svg_50`, `breast_svg_100` |
| **Colon** | `colon_random_50`, `colon_random_100`, `colon_hvg_50`, `colon_hvg_100`, `colon_svg_50`, `colon_svg_100` |
| **Prostate** | `prostate_random_50`, `prostate_random_100`, `prostate_hvg_50`, `prostate_hvg_100`, `prostate_svg_50`, `prostate_svg_100` |

### List All Benchmarks

```bash
python -m pipeline.finetune --list_benchmarks
```

### Override Benchmark Settings

CLI arguments override benchmark defaults:

```bash
# Use lung_hvg_100 but with different epochs and learning rate
python -m pipeline.finetune --benchmark lung_hvg_100 --epochs 50 --learning_rate 5e-5 --output_dir ./results
```

### Configure Paths

Benchmarks use paths defined in `utils/benchmarks.py`. The default base path is:
```
/multimodal/dist/hest_data_{tissue}/
├── patches/
└── st/
```

To change the base path, edit `HEST_BASE_PATH` in `utils/benchmarks.py`.

---

## Output Structure

Every run produces a standardized output:

```
output_dir/
├── config.json           # Full configuration (reproducibility)
├── train_split.json      # Training sample IDs
├── val_split.json        # Validation sample IDs
├── gene_list.pkl         # Selected genes
├── log.txt               # Training log
├── model_epoch0          # Checkpoints
├── model_epoch1
└── ...

eval/
├── metrics.json          # All metrics
└── per_gene_metrics.csv  # Per-gene breakdown
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean squared error |
| R² | Coefficient of determination |
| Pearson | Mean Pearson correlation across genes |
| Spearman | Mean Spearman correlation across genes |
| SSIM | Structural similarity on spatial maps (with `--compute_spatial`) |

---

## Custom Datasets

For data **not in HEST format**, use the adapter system:

### Step 1: Create Your Adapter

See `examples/custom_dataset_adapter.py` for a complete template.

```python
from utils.dataset_adapter import DatasetAdapter, register_dataset

@register_dataset("my_data")
class MyAdapter(DatasetAdapter):
    @property
    def task_type(self):
        return "regression"  # or "classification"
    
    def load_data(self, train_samples, val_samples, transforms, **kwargs):
        # Load YOUR data in YOUR format
        # Return (train_loader, val_loader)
        ...
    
    def get_num_outputs(self, **kwargs):
        return 100  # number of targets
    
    def get_loss_function(self):
        return torch.nn.MSELoss()
```

### Step 2: Use Your Adapter

```bash
# Import your adapter, then run with --dataset flag
python -m pipeline.finetune \
    --dataset my_data \
    --data_path /path/to/your/data \
    --output_dir ./results
```

**Key point:** Your data can be in **ANY format**. The adapter handles all loading/processing.

---

## Train/Validation Splits

**Default:** Random 80/20 split with seed.

**Custom splits:** Provide JSON or text files with sample IDs:

```bash
python -m pipeline.finetune \
    --train_split_path ./splits/train.json \
    --val_split_path ./splits/val.json \
    ...
```

Splits are automatically saved to `output_dir/` for reproducibility.

---

## Examples

### Example 1: Quick Run with Benchmarks

The easiest way to get started - just specify a benchmark:

```bash
# Train TinyViT on lung data with highly variable genes
python -m pipeline.finetune --benchmark lung_hvg_100 --output_dir ./results/lung_hvg

# Train with distillation from UNI2
python -m pipeline.distill --benchmark breast_svg_50 --teacher uni2 --output_dir ./results/breast_distill
```

### Example 2: Compare Gene Selection Strategies

```bash
# Compare all three strategies on the same tissue
for strategy in random hvg svg; do
    python -m pipeline.finetune \
        --patches_path /data/hest_lung/patches/ \
        --adata_path /data/hest_lung/st/ \
        --filter_strategy $strategy \
        --n_genes 100 \
        --output_dir ./results/lung_${strategy}
done

# Evaluate all three
for strategy in random hvg svg; do
    python -m pipeline.evaluate \
        --model_path ./results/lung_${strategy}/model_epoch99 \
        --config_path ./results/lung_${strategy}/config.json \
        --output_dir ./results/lung_${strategy}/eval
done
```

### Example 3: Compare Models (Baseline vs Distilled)

```bash
# Train TinyViT baseline
python -m pipeline.finetune \
    --benchmark colon_hvg_100 \
    --model tinyvit \
    --output_dir ./results/tinyvit_baseline

# Train TinyViT with UNI2 distillation
python -m pipeline.distill \
    --benchmark colon_hvg_100 \
    --teacher uni2 --student tinyvit \
    --output_dir ./results/tinyvit_uni_distill

# Train TinyViT with Virchow2 distillation
python -m pipeline.distill \
    --benchmark colon_hvg_100 \
    --teacher virchow2 --student tinyvit \
    --output_dir ./results/tinyvit_virchow_distill

# Compare results
for model in tinyvit_baseline tinyvit_uni_distill tinyvit_virchow_distill; do
    echo "=== $model ==="
    cat ./results/${model}/eval/metrics.json
done
```

### Example 4: Use Your Own Gene List

```python
# Create a custom gene list
import pickle
my_genes = ['TP53', 'EGFR', 'KRAS', 'BRCA1', 'MYC', ...]  # your genes
with open('my_genes.pkl', 'wb') as f:
    pickle.dump(my_genes, f)
```

```bash
# Use the custom gene list
python -m pipeline.finetune \
    --patches_path /data/patches/ \
    --adata_path /data/st/ \
    --gene_list_path my_genes.pkl \
    --output_dir ./results/custom_genes
```

### Example 5: Cross-Tissue Comparison

Run the same configuration across multiple tissues:

```bash
for tissue in lung breast colon prostate; do
    python -m pipeline.finetune \
        --benchmark ${tissue}_hvg_100 \
        --output_dir ./results/${tissue}_hvg_100
    
    python -m pipeline.evaluate \
        --model_path ./results/${tissue}_hvg_100/model_epoch99 \
        --config_path ./results/${tissue}_hvg_100/config.json \
        --output_dir ./results/${tissue}_hvg_100/eval \
        --compute_spatial
done
```

### Example 6: Full Pipeline with SSIM Spatial Metrics

```bash
# 1. Fine-tune foundation model
python -m pipeline.finetune \
    --patches_path /data/hest_breast/patches/ \
    --adata_path /data/hest_breast/st/ \
    --model uni2 \
    --filter_strategy svg \
    --n_genes 100 \
    --output_dir ./results/uni2_breast

# 2. Distill to lightweight TinyViT
python -m pipeline.distill \
    --patches_path /data/hest_breast/patches/ \
    --adata_path /data/hest_breast/st/ \
    --teacher uni2 \
    --student tinyvit \
    --gene_list_path ./results/uni2_breast/gene_list.pkl \
    --output_dir ./results/distill_breast

# 3. Evaluate with spatial SSIM metrics
python -m pipeline.evaluate \
    --model_path ./results/uni2_breast/model_epoch99 \
    --config_path ./results/uni2_breast/config.json \
    --output_dir ./results/uni2_breast/eval \
    --compute_spatial

python -m pipeline.evaluate \
    --model_path ./results/distill_breast/model_epoch99 \
    --config_path ./results/distill_breast/config.json \
    --output_dir ./results/distill_breast/eval \
    --compute_spatial
```

### Example 7: Hyperparameter Tuning

Override default hyperparameters:

```bash
# Longer training with lower learning rate
python -m pipeline.finetune \
    --benchmark lung_hvg_100 \
    --epochs 200 \
    --learning_rate 5e-5 \
    --batch_size 64 \
    --output_dir ./results/lung_tuned

# Adjust distillation parameters
python -m pipeline.distill \
    --benchmark lung_hvg_100 \
    --teacher uni2 \
    --temperature 2.0 \
    --alpha 0.7 \
    --output_dir ./results/lung_distill_tuned
```

### Example 8: Reproducible Experiments with Fixed Splits

```bash
# First run - saves splits automatically
python -m pipeline.finetune \
    --benchmark prostate_svg_50 \
    --seed 42 \
    --output_dir ./results/run1

# Reproduce with same splits
python -m pipeline.finetune \
    --benchmark prostate_svg_50 \
    --train_split_path ./results/run1/train_split.json \
    --val_split_path ./results/run1/val_split.json \
    --output_dir ./results/run2
```

---

## Project Structure

```
knowledge_distillation_histopath/
├── pipeline/
│   ├── finetune.py          # Fine-tuning script
│   ├── distill.py           # Knowledge distillation script
│   └── evaluate.py          # Evaluation script
│
├── utils/
│   ├── gene_filtering.py    # Random, HVG, SVG selection
│   ├── split_utils.py       # Train/val split management
│   ├── spatial_metrics.py   # SSIM calculations
│   ├── benchmarks.py        # Predefined benchmark configs
│   ├── dataset_adapter.py   # Custom dataset system
│   ├── data_utils.py        # HEST dataset class
│   ├── train_HEST.py        # Training loops
│   └── custom_losses.py     # Distillation losses
│
├── modeling/
│   ├── models.py            # Model loaders
│   └── tinyvit/             # TinyViT architecture
│
├── examples/
│   └── custom_dataset_adapter.py  # Template for custom data
│
├── reproduce/                     # Figure reproduction scripts
│   ├── README.md                  # Instructions
│   ├── run_experiments.py         # Step 1: Train all models
│   └── generate_figures.py        # Step 2: Generate paper figures
│
└── environment.yml                # Conda environment specification
```

---

## Environment Setup

For a complete, reproducible environment, use the provided `environment.yml`:

```bash
# Create conda environment
conda env create -f environment.yml
conda activate stainfactor

# Verify installation
python -c "import torch; print(torch.__version__)"
```

---

## Reproducing Paper Figures

The `reproduce/` directory contains scripts to regenerate all results figures:

```bash
cd reproduce

# Step 1: Run all experiments (trains models, runs evaluation)
python run_experiments.py --data_root /path/to/hest_data --output_dir ./outputs

# Step 2: Generate figures from results
python generate_figures.py --results_dir ./outputs --output_dir ./figures
```

**Using your own results:** If you ran experiments separately, point to your results:

```bash
python generate_figures.py --results_dir /path/to/your/results
```

See `reproduce/README.md` for detailed instructions, expected data formats, and customization options.
