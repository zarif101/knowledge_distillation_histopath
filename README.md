# Virtual Staining Target Selection Framework

An open-source framework for assessing the impact of target selection methods on virtual staining model performance. The framework includes plug-and-play Python scripts that train and evaluate model architectures including foundation models, knowledge distillation, and lightweight models for virtual staining datasets filtered using a variety of target set selection strategies.

Users can choose to assess custom existing trained models or retrain using standardized preprocessing workflows, enabling flexible usage to suit diverse model evaluations. The package supports evaluation on standardized benchmark datasets as well as additional datasets implemented by users.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Scripts Reference](#scripts-reference)
5. [Target Selection Strategies](#target-selection-strategies)
6. [Standardized Benchmarks](#standardized-benchmarks)
7. [Train/Validation Splits](#trainvalidation-splits)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Output Structure](#output-structure)
10. [Custom Datasets](#custom-datasets)
11. [Examples](#examples)

---

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd knowledge_distillation_histopath

# Install dependencies
pip install torch torchvision timm scanpy h5py scikit-learn scipy pandas numpy
pip install huggingface_hub  # Required for foundation models (UNI2, VIRCHOW2)
pip install scikit-image     # Required for SSIM spatial metrics

# Optional: for Moran's I spatial gene selection
pip install squidpy
```

---

## Quick Start

### 1. Fine-tune a Foundation Model

```bash
python -m pipeline.finetune \
    --dataset HEST \
    --model_name UNI2 \
    --patches_path data/patches/ \
    --adata_path data/st/ \
    --filter_strategy highly_variable \
    --n_genes 100 \
    --output_dir runs/finetune_uni2_hvg/ \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --epochs 50 \
    --hf_path hf_key.txt
```

### 2. Distill into a Lightweight Model

```bash
python -m pipeline.distill \
    --dataset HEST \
    --teacher_model UNI2 \
    --teacher_path runs/finetune_uni2_hvg/checkpoints/model_epoch50 \
    --student_model TINYVIT \
    --patches_path data/patches/ \
    --adata_path data/st/ \
    --filter_strategy highly_variable \
    --n_genes 100 \
    --output_dir runs/distill_tinyvit/ \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --epochs 50 \
    --hf_path hf_key.txt
```

### 3. Evaluate a Trained Model

```bash
python -m pipeline.evaluate \
    --dataset HEST \
    --model_name UNI2 \
    --model_path runs/finetune_uni2_hvg/checkpoints/model_epoch50 \
    --patches_path data/patches/ \
    --adata_path data/st/ \
    --gene_list_path runs/finetune_uni2_hvg/gene_list.pkl \
    --output_dir runs/finetune_uni2_hvg/eval/ \
    --compute_spatial \
    --hf_path hf_key.txt
```

---

## Core Concepts

### Supported Models

| Model | Type | Size | Description |
|-------|------|------|-------------|
| `UNI2` | Foundation | ~300M params | MahmoodLab's UNI2 Vision Transformer |
| `VIRCHOW2` | Foundation | ~300M params | Paige AI's Virchow2 model |
| `TINYVIT` | Lightweight | 5M params | Compact model for deployment |
| `TINYVIT11M` | Lightweight | 11M params | Larger TinyViT variant |

### Supported Datasets

| Dataset | Task | Input | Output |
|---------|------|-------|--------|
| `HEST` | Virtual Staining / Gene Expression | Histopathology patches | Gene expression values |
| `WSICLASS` | Classification | Whole slide image patches | Class labels |

### Workflow Options

1. **Fine-tune Only**: Train a foundation model or lightweight model directly on your task
2. **Knowledge Distillation**: Train a lightweight student model to mimic a larger teacher model
3. **Evaluate Only**: Assess an existing trained model on test data

---

## Scripts Reference

### `pipeline.finetune`

Fine-tune a model on your dataset.

```bash
python -m pipeline.finetune \
    --dataset {HEST,WSICLASS} \
    --model_name {UNI2,VIRCHOW2,TINYVIT,TINYVIT11M} \
    --patches_path PATH \
    --output_dir PATH \
    --hf_path PATH \
    [additional options...]
```

**Key Arguments:**

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset type: `HEST` or `WSICLASS` |
| `--model_name` | Model architecture to train |
| `--patches_path` | Directory containing patch files |
| `--output_dir` | Where to save all outputs |
| `--hf_path` | Path to HuggingFace API key file |
| `--filter_strategy` | Gene selection strategy (HEST only) |
| `--n_genes` | Number of genes to select |
| `--train_split` | Custom training samples file |
| `--val_split` | Custom validation samples file |
| `--train_layers` | `all` or `final` (freeze strategy) |
| `--batch_size` | Training batch size |
| `--learning_rate` | Learning rate |
| `--epochs` | Number of training epochs |

### `pipeline.distill`

Distill a teacher model into a student model.

```bash
python -m pipeline.distill \
    --dataset {HEST,WSICLASS} \
    --teacher_model {UNI2,VIRCHOW2,...} \
    --teacher_path PATH \
    --student_model {TINYVIT,TINYVIT11M,...} \
    --patches_path PATH \
    --output_dir PATH \
    --hf_path PATH \
    [additional options...]
```

**Additional Arguments:**

| Argument | Description |
|----------|-------------|
| `--teacher_model` | Teacher model architecture |
| `--teacher_path` | Path to trained teacher checkpoint |
| `--student_model` | Student model architecture |
| `--distill_level` | `output` (match predictions) or `feature` (match representations) |

### `pipeline.evaluate`

Evaluate a trained model with comprehensive metrics.

```bash
python -m pipeline.evaluate \
    --dataset {HEST,WSICLASS} \
    --model_name {UNI2,VIRCHOW2,...} \
    --model_path PATH \
    --patches_path PATH \
    --output_dir PATH \
    --hf_path PATH \
    [additional options...]
```

**Key Arguments:**

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to trained model checkpoint |
| `--gene_list_path` | Gene list used for training (HEST) |
| `--eval_split` | `test` (30% holdout) or `all` samples |
| `--eval_split_path` | Custom evaluation samples file |
| `--compute_spatial` | Compute SSIM spatial metrics (HEST) |

---

## Target Selection Strategies

For HEST (virtual staining) tasks, you can select which genes to predict using different strategies:

### Available Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| **Custom** | `--filter_strategy custom --gene_list_path genes.pkl` | Provide your own gene list |
| **Random** | `--filter_strategy random --n_genes 100` | Randomly select N genes |
| **Highly Variable** | `--filter_strategy highly_variable --n_genes 100` | Top N genes by variance (HVG) |
| **Spatially Variable** | `--filter_strategy spatially_variable --n_genes 100` | Top N genes by Moran's I |

### Example: Compare Different Strategies

```bash
# Train with highly variable genes
python -m pipeline.finetune --dataset HEST --model_name UNI2 \
    --filter_strategy highly_variable --n_genes 100 \
    --output_dir runs/hvg_100/ ...

# Train with spatially variable genes
python -m pipeline.finetune --dataset HEST --model_name UNI2 \
    --filter_strategy spatially_variable --n_genes 100 \
    --output_dir runs/svg_100/ ...

# Train with random genes
python -m pipeline.finetune --dataset HEST --model_name UNI2 \
    --filter_strategy random --n_genes 100 --seed 42 \
    --output_dir runs/random_100/ ...
```

---

## Standardized Benchmarks

The framework includes pre-defined benchmark configurations for reproducible experiments. Benchmarks specify the dataset, gene selection strategy, hyperparameters, and recommended models.

### Using a Benchmark

```bash
# List all available benchmarks
python -m pipeline.finetune --list_benchmarks

# Run a benchmark (auto-fills dataset, paths, genes, hyperparameters)
python -m pipeline.finetune \
    --benchmark HEST_HVG100 \
    --model_name UNI2 \
    --output_dir runs/benchmark_hvg100/ \
    --hf_path hf_key.txt
```

### Available Benchmarks

| Benchmark | Dataset | Strategy | Genes | Description |
|-----------|---------|----------|-------|-------------|
| `HEST_HVG50` | HEST | Highly Variable | 50 | HVG baseline |
| `HEST_HVG100` | HEST | Highly Variable | 100 | HVG standard |
| `HEST_HVG250` | HEST | Highly Variable | 250 | HVG extended |
| `HEST_SVG50` | HEST | Spatially Variable | 50 | Moran's I selection |
| `HEST_SVG100` | HEST | Spatially Variable | 100 | Moran's I selection |
| `HEST_RANDOM50` | HEST | Random | 50 | Random baseline |
| `HEST_RANDOM100` | HEST | Random | 100 | Random baseline |
| `HEST_LUAD_HVG100` | HEST | Highly Variable | 100 | Lung adenocarcinoma |
| `HEST_BREAST_HVG100` | HEST | Highly Variable | 100 | Breast cancer |
| `HEST_SKIN_HVG100` | HEST | Highly Variable | 100 | Skin tissue |
| `WSICLASS_EXAMPLE` | WSICLASS | N/A | N/A | WSI classification |

### Benchmark Configuration

Each benchmark includes:
- **Dataset**: Which dataset adapter to use
- **Data paths**: Paths to patches, adata, metadata
- **Gene selection**: Filter strategy and number of genes
- **Splits**: Pre-defined or seed-based train/val/test splits
- **Hyperparameters**: Batch size, learning rate, epochs
- **Recommended models**: Models known to work well

### Overriding Benchmark Settings

You can override any benchmark setting with CLI arguments:

```bash
# Use HEST_HVG100 but with different learning rate and epochs
python -m pipeline.finetune \
    --benchmark HEST_HVG100 \
    --model_name UNI2 \
    --learning_rate 0.001 \
    --epochs 50 \
    --output_dir runs/custom_benchmark/ \
    --hf_path hf_key.txt
```

### Setting Up Benchmark Paths

Benchmarks have placeholder paths (`<FILL_IN>`) that need to be configured. Edit `utils/benchmarks.py` and replace the placeholders:

```python
# Before
patches_path="<FILL_IN>/hest_data/patches/",
adata_path="<FILL_IN>/hest_data/st/",

# After
patches_path="/data/hest/patches/",
adata_path="/data/hest/st/",
```

---

## Train/Validation Splits

### Default Behavior

By default, the framework automatically splits your data 70/30 train/validation using a random seed:

```bash
python -m pipeline.finetune ... --seed 42  # Reproducible 70/30 split
```

### Custom Splits

Provide your own split files (one sample ID per line):

```bash
python -m pipeline.finetune ... \
    --train_split splits/train_samples.txt \
    --val_split splits/val_samples.txt
```

**Split file format:**
```
SAMPLE_001
SAMPLE_002
SAMPLE_003
```

### Auto-Saved Splits

When training, splits are automatically saved to your output directory for reproducibility:
- `output_dir/train_samples.txt`
- `output_dir/val_samples.txt`

### Evaluation Splits

```bash
# Use the default test split (30% holdout with same seed)
python -m pipeline.evaluate ... --eval_split test --seed 42

# Evaluate on all samples
python -m pipeline.evaluate ... --eval_split all

# Use custom evaluation samples
python -m pipeline.evaluate ... --eval_split_path splits/test_samples.txt
```

---

## Evaluation Metrics

### HEST (Virtual Staining) Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error between predicted and true expression |
| **R²** | Coefficient of determination |
| **Pearson r** | Pearson correlation coefficient (mean/median across genes) |
| **Spearman r** | Spearman rank correlation (mean/median across genes) |
| **SSIM** | Structural Similarity Index on spatial expression maps |

**Per-gene breakdown** is provided for all metrics.

### WSICLASS (Classification) Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **F1 Macro** | Macro-averaged F1 score |
| **F1 Weighted** | Weighted F1 score |
| **Per-class Accuracy** | Accuracy for each class |

### Spatial Metrics (SSIM)

SSIM measures structural similarity between spatial gene expression maps:

```bash
python -m pipeline.evaluate --dataset HEST ... --compute_spatial
```

This builds 2D spatial maps from spot coordinates and computes SSIM between true and predicted maps for each gene.

---

## Output Structure

All scripts produce a standardized output directory:

```
output_dir/
│
├── config.json              # Complete configuration for reproducibility
│                            # Contains all arguments, timestamp, paths
│
├── train_samples.txt        # Training sample IDs (auto-saved)
├── val_samples.txt          # Validation sample IDs (auto-saved)
│
├── gene_list.pkl            # Gene list used (HEST only)
│                            # Auto-generated or copied from custom
│
├── checkpoints/             # Model checkpoints
│   ├── model_epoch0
│   ├── model_epoch1
│   ├── ...
│   └── model_epoch{best}
│
└── eval/                    # Evaluation outputs (if evaluated)
    ├── eval_config.json     # Evaluation configuration
    ├── metrics.json         # All metrics (overall + per-gene)
    └── per_gene_metrics.csv # Per-gene metrics as CSV
```

### Config File Example

```json
{
  "dataset": "HEST",
  "model_name": "UNI2",
  "filter_strategy": "highly_variable",
  "n_genes": 100,
  "batch_size": 16,
  "learning_rate": 0.0001,
  "epochs": 50,
  "seed": 42,
  "timestamp": "2026-01-21T10:30:00",
  "output_paths": {
    "root": "runs/experiment_1/",
    "checkpoints": "runs/experiment_1/checkpoints/",
    "config": "runs/experiment_1/config.json",
    "gene_list": "runs/experiment_1/gene_list.pkl"
  }
}
```

### Metrics Output Example

```json
{
  "overall": {
    "mse": 0.0234,
    "r2": 0.8756,
    "mean_pearson_r": 0.7823,
    "median_pearson_r": 0.8012,
    "mean_spearman_r": 0.7654,
    "median_spearman_r": 0.7891,
    "mean_ssim": 0.6543,
    "median_ssim": 0.6821,
    "n_samples": 15420,
    "n_genes": 100
  },
  "per_gene": [
    {"gene": "EGFR", "mse": 0.012, "r2": 0.91, "pearson_r": 0.85, "ssim": 0.72},
    {"gene": "TP53", "mse": 0.018, "r2": 0.87, "pearson_r": 0.81, "ssim": 0.68},
    ...
  ]
}
```

---

## Custom Datasets

The framework uses a **Dataset Adapter** system that makes it easy to add your own datasets. You create a Python class that implements a few required methods, and the framework handles the rest.

### Using a Custom Dataset

```bash
python -m pipeline.finetune \
    --dataset MYDATASET \
    --custom_adapter path/to/my_adapter.py \
    --patches_path my_data/patches/ \
    --labels_path my_data/labels.csv \
    --output_dir runs/my_experiment/ \
    --hf_path hf_key.txt
```

### Creating a Custom Adapter

Create a Python file with your adapter class:

```python
# my_adapter.py

from utils.dataset_adapter import DatasetAdapter, register_dataset
from torch.utils.data import Dataset
import torch
import numpy as np
import os

# Step 1: Define your PyTorch Dataset
class MyDataset(Dataset):
    def __init__(self, patches_path, labels_path, samples, transforms=None):
        self.patches_path = patches_path
        self.samples = samples
        self.transforms = transforms
        # Load your labels here
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load and return (image, target) pair
        sample_id = self.samples[idx]
        patch = np.load(f"{self.patches_path}/{sample_id}.npy")
        patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        if self.transforms:
            patch = self.transforms(patch)
        target = torch.tensor([0.5])  # Your target here
        return patch, target

# Step 2: Create the Adapter
@register_dataset
class MyAdapter(DatasetAdapter):
    name = "MYDATASET"           # Used in --dataset arg
    task_type = "regression"     # or "classification"
    
    def get_sample_ids(self, data_path, **kwargs):
        """Return list of sample IDs from your data directory."""
        files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        return [f.replace('.npy', '') for f in files]
    
    def get_num_outputs(self, **kwargs):
        """Return number of output dimensions (genes/classes)."""
        return 1
    
    def create_dataset(self, samples, transforms, patches_path, labels_path=None, **kwargs):
        """Create and return your PyTorch Dataset."""
        return MyDataset(patches_path, labels_path, samples, transforms)
    
    def validate_args(self, args):
        """Validate required arguments."""
        if not args.labels_path:
            raise ValueError("--labels_path is required")
    
    def get_required_args(self):
        """List of required CLI argument names."""
        return ['labels_path']
```

### Adapter Interface Reference

Your adapter class must:

1. **Inherit from `DatasetAdapter`**
2. **Set class attributes:**
   - `name`: Unique identifier (used in `--dataset` CLI arg)
   - `task_type`: Either `"regression"` or `"classification"`

3. **Implement required methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `get_sample_ids(data_path, **kwargs)` | List all available sample IDs | `List[str]` |
| `get_num_outputs(**kwargs)` | Number of outputs (genes/classes) | `int` |
| `create_dataset(samples, transforms, **kwargs)` | Create PyTorch Dataset | `Dataset` |

4. **Optional methods:**

| Method | Description | Default |
|--------|-------------|---------|
| `validate_args(args)` | Validate CLI arguments | No-op |
| `get_required_args()` | List required arg names | `[]` |
| `get_output_names(**kwargs)` | Names of outputs | `None` |
| `get_default_loss()` | Default loss function | `'mse'` or `'ce'` |
| `get_default_batch_size()` | Default batch size | `16` |

### Example: Complete Custom Adapter

See `examples/custom_dataset_adapter.py` for a complete, well-documented example.

### Built-in Adapters

The framework includes these built-in dataset adapters:

| Name | Task | Description |
|------|------|-------------|
| `HEST` | Regression | Spatial transcriptomics gene expression |
| `WSICLASS` | Classification | Whole slide image classification |

You can view their implementations in `utils/dataset_adapter.py` as reference.

---

## Examples

### Example 1: Full Pipeline (Fine-tune → Distill → Evaluate)

```bash
# Step 1: Fine-tune UNI2 on highly variable genes
python -m pipeline.finetune \
    --dataset HEST \
    --model_name UNI2 \
    --patches_path data/patches/ \
    --adata_path data/st/ \
    --filter_strategy highly_variable \
    --n_genes 100 \
    --output_dir runs/uni2_hvg/ \
    --batch_size 16 \
    --epochs 50 \
    --hf_path hf_key.txt

# Step 2: Distill into TinyViT
python -m pipeline.distill \
    --dataset HEST \
    --teacher_model UNI2 \
    --teacher_path runs/uni2_hvg/checkpoints/model_epoch50 \
    --student_model TINYVIT \
    --patches_path data/patches/ \
    --adata_path data/st/ \
    --gene_list_path runs/uni2_hvg/gene_list.pkl \
    --filter_strategy custom \
    --output_dir runs/tinyvit_distilled/ \
    --batch_size 16 \
    --epochs 50 \
    --hf_path hf_key.txt

# Step 3: Evaluate both models
python -m pipeline.evaluate \
    --dataset HEST \
    --model_name UNI2 \
    --model_path runs/uni2_hvg/checkpoints/model_epoch50 \
    --patches_path data/patches/ \
    --adata_path data/st/ \
    --gene_list_path runs/uni2_hvg/gene_list.pkl \
    --output_dir runs/uni2_hvg/eval/ \
    --compute_spatial \
    --hf_path hf_key.txt

python -m pipeline.evaluate \
    --dataset HEST \
    --model_name TINYVIT \
    --model_path runs/tinyvit_distilled/checkpoints/model_epoch50 \
    --patches_path data/patches/ \
    --adata_path data/st/ \
    --gene_list_path runs/uni2_hvg/gene_list.pkl \
    --output_dir runs/tinyvit_distilled/eval/ \
    --compute_spatial \
    --hf_path hf_key.txt
```

### Example 2: Run Standardized Benchmarks

```bash
# List available benchmarks
python -m pipeline.finetune --list_benchmarks

# Run the HVG100 benchmark with UNI2
python -m pipeline.finetune \
    --benchmark HEST_HVG100 \
    --model_name UNI2 \
    --output_dir runs/benchmark_uni2_hvg100/ \
    --hf_path hf_key.txt

# Distill to TinyViT using same benchmark
python -m pipeline.distill \
    --benchmark HEST_HVG100 \
    --teacher_model UNI2 \
    --teacher_path runs/benchmark_uni2_hvg100/checkpoints/model_epoch100 \
    --student_model TINYVIT \
    --output_dir runs/benchmark_tinyvit_hvg100/ \
    --hf_path hf_key.txt

# Evaluate both
python -m pipeline.evaluate \
    --benchmark HEST_HVG100 \
    --model_name UNI2 \
    --model_path runs/benchmark_uni2_hvg100/checkpoints/model_epoch100 \
    --gene_list_path runs/benchmark_uni2_hvg100/gene_list.pkl \
    --output_dir runs/benchmark_uni2_hvg100/eval/ \
    --compute_spatial \
    --hf_path hf_key.txt
```

### Example 3: Compare Target Selection Strategies (Benchmark Study)

```bash
# Run experiments with different gene selection strategies using benchmarks
for benchmark in HEST_RANDOM100 HEST_HVG100 HEST_SVG100; do
    python -m pipeline.finetune \
        --benchmark $benchmark \
        --model_name UNI2 \
        --output_dir runs/${benchmark}_uni2/ \
        --hf_path hf_key.txt
    
    python -m pipeline.evaluate \
        --benchmark $benchmark \
        --model_name UNI2 \
        --model_path runs/${benchmark}_uni2/checkpoints/model_epoch100 \
        --gene_list_path runs/${benchmark}_uni2/gene_list.pkl \
        --output_dir runs/${benchmark}_uni2/eval/ \
        --compute_spatial \
        --hf_path hf_key.txt
done
```

### Example 4: Compare Target Selection Strategies (Manual)

```bash
# Run experiments with different gene selection strategies
for strategy in random highly_variable spatially_variable; do
    python -m pipeline.finetune \
        --dataset HEST \
        --model_name UNI2 \
        --patches_path data/patches/ \
        --adata_path data/st/ \
        --filter_strategy $strategy \
        --n_genes 100 \
        --output_dir runs/compare_${strategy}/ \
        --epochs 50 \
        --hf_path hf_key.txt
    
    python -m pipeline.evaluate \
        --dataset HEST \
        --model_name UNI2 \
        --model_path runs/compare_${strategy}/checkpoints/model_epoch50 \
        --patches_path data/patches/ \
        --adata_path data/st/ \
        --gene_list_path runs/compare_${strategy}/gene_list.pkl \
        --output_dir runs/compare_${strategy}/eval/ \
        --compute_spatial \
        --hf_path hf_key.txt
done
```

### Example 5: Train Lightweight Model Without Distillation

```bash
# Train TinyViT directly (no teacher)
python -m pipeline.finetune \
    --dataset HEST \
    --model_name TINYVIT \
    --patches_path data/patches/ \
    --adata_path data/st/ \
    --filter_strategy highly_variable \
    --n_genes 100 \
    --output_dir runs/tinyvit_direct/ \
    --batch_size 32 \
    --epochs 100 \
    --hf_path hf_key.txt
```

### Example 6: Using Custom Train/Val Splits

```bash
# Create split files
echo -e "SAMPLE_001\nSAMPLE_002\nSAMPLE_003" > splits/train.txt
echo -e "SAMPLE_004\nSAMPLE_005" > splits/val.txt

# Train with custom splits
python -m pipeline.finetune \
    --dataset HEST \
    --model_name UNI2 \
    --patches_path data/patches/ \
    --adata_path data/st/ \
    --train_split splits/train.txt \
    --val_split splits/val.txt \
    --filter_strategy highly_variable \
    --n_genes 100 \
    --output_dir runs/custom_split/ \
    --hf_path hf_key.txt
```

### Example 7: WSI Classification

```bash
# Fine-tune for WSI classification
python -m pipeline.finetune \
    --dataset WSICLASS \
    --model_name UNI2 \
    --patches_path wsi_data/patches/ \
    --metadata_path wsi_data/metadata.csv \
    --num_classes 5 \
    --output_dir runs/wsi_classify/ \
    --loss_fn ce \
    --epochs 50 \
    --hf_path hf_key.txt

# Evaluate
python -m pipeline.evaluate \
    --dataset WSICLASS \
    --model_name UNI2 \
    --model_path runs/wsi_classify/checkpoints/model_epoch50 \
    --patches_path wsi_data/patches/ \
    --metadata_path wsi_data/metadata.csv \
    --num_classes 5 \
    --output_dir runs/wsi_classify/eval/ \
    --hf_path hf_key.txt
```

### Example 8: Using a Custom Dataset

```bash
# Create your custom adapter (see examples/custom_dataset_adapter.py)

# Fine-tune with custom dataset
python -m pipeline.finetune \
    --dataset MYDATASET \
    --custom_adapter examples/custom_dataset_adapter.py \
    --model_name UNI2 \
    --patches_path my_data/patches/ \
    --labels_path my_data/labels.csv \
    --output_dir runs/custom_experiment/ \
    --batch_size 16 \
    --epochs 50 \
    --hf_path hf_key.txt

# Distill to lightweight model
python -m pipeline.distill \
    --dataset MYDATASET \
    --custom_adapter examples/custom_dataset_adapter.py \
    --teacher_model UNI2 \
    --teacher_path runs/custom_experiment/checkpoints/model_epoch50 \
    --student_model TINYVIT \
    --patches_path my_data/patches/ \
    --labels_path my_data/labels.csv \
    --output_dir runs/custom_distill/ \
    --batch_size 16 \
    --epochs 50 \
    --hf_path hf_key.txt

# Evaluate
python -m pipeline.evaluate \
    --dataset MYDATASET \
    --custom_adapter examples/custom_dataset_adapter.py \
    --model_name TINYVIT \
    --model_path runs/custom_distill/checkpoints/model_epoch50 \
    --patches_path my_data/patches/ \
    --labels_path my_data/labels.csv \
    --output_dir runs/custom_distill/eval/ \
    --hf_path hf_key.txt
```

---

## Data Format Requirements

### HEST Dataset

**Patches Directory:**
```
patches_path/
├── SAMPLE_001.h5    # H5 file with 'img' and 'barcode' datasets
├── SAMPLE_002.h5
└── ...
```

**AnnData Directory:**
```
adata_path/
├── SAMPLE_001.h5ad  # AnnData with expression matrix and spatial coords
├── SAMPLE_002.h5ad
└── ...
```

### WSICLASS Dataset

**Patches Directory:**
```
patches_path/
├── SAMPLE_001.npy   # Shape: (N_patches, H, W, 3)
├── SAMPLE_002.npy
└── ...
```

**Metadata CSV:**
```csv
sample_id,class
SAMPLE_001,0
SAMPLE_002,1
SAMPLE_003,2
```

---

## License

[Add license information]

## Citation

[Add citation information]

