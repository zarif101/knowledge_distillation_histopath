# StainFactor

An open-source framework for **assessing the impact of target selection methods on virtual staining model performance**.

The framework includes plug-and-play Python scripts that train and evaluate model architectures including foundation models, knowledge distillation, and lightweight models for virtual staining datasets filtered using a variety of target set selection strategies. Users can choose to assess custom existing trained models or retrain using standardized preprocessing workflows, enabling flexible usage to suit diverse model evaluations. The package supports evaluation on standardized benchmark datasets as well as additional datasets implemented by users.

## Key Features

- **Foundation Models**: UNI2, Virchow2
- **Lightweight Models**: TinyViT (5M params)
- **Knowledge Distillation**: Transfer knowledge from large teachers to small students
- **Target Selection Strategies**: Random, Highly Variable Genes (HVG), Spatially Variable Genes (SVG/Moran's I)
- **Standardized Benchmarks**: Pre-configured experiments across multiple tissue types
- **Custom Datasets**: Extensible adapter system for your own data formats

---

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd stainfactor

# Install dependencies
pip install torch torchvision timm scanpy h5py scikit-learn scipy pandas numpy
pip install huggingface_hub    # For foundation models (UNI2, Virchow2)
pip install scikit-image       # For SSIM spatial metrics

# Optional: for Moran's I spatial gene selection
pip install squidpy
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

### Compare Gene Selection Strategies

```bash
for strategy in random hvg svg; do
    python -m pipeline.finetune \
        --patches_path /data/patches/ \
        --adata_path /data/st/ \
        --filter_strategy $strategy \
        --n_genes 100 \
        --output_dir ./results/${strategy}
done
```

### Full Pipeline: Fine-tune → Distill → Evaluate

```bash
# 1. Fine-tune foundation model
python -m pipeline.finetune \
    --patches_path /data/patches/ --adata_path /data/st/ \
    --model uni2 --filter_strategy hvg --n_genes 100 \
    --output_dir ./results/uni2

# 2. Distill to lightweight model
python -m pipeline.distill \
    --patches_path /data/patches/ --adata_path /data/st/ \
    --teacher uni2 --student tinyvit \
    --gene_list_path ./results/uni2/gene_list.pkl \
    --output_dir ./results/distill

# 3. Evaluate both
python -m pipeline.evaluate \
    --model_path ./results/uni2/model_epoch99 \
    --config_path ./results/uni2/config.json \
    --output_dir ./results/uni2/eval --compute_spatial

python -m pipeline.evaluate \
    --model_path ./results/distill/model_epoch99 \
    --config_path ./results/distill/config.json \
    --output_dir ./results/distill/eval --compute_spatial
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
└── examples/
    └── custom_dataset_adapter.py  # Template for custom data
```

---

## License

[Add license]

## Citation

[Add citation]
