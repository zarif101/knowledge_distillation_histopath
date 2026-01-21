# Figure Reproduction

This directory contains scripts to reproduce all results figures from the StainFactor paper.

## Workflow

### Step 1: Run Experiments

Train all models and run evaluations:

```bash
python run_experiments.py --data_root /path/to/hest_data --output_dir ./outputs
```

This will:
- Train TinyViT baseline models
- Train TinyViT with UNI2 distillation
- Train TinyViT with Virchow2 distillation
- For each tissue type (lung, breast, colon, prostate, skin)
- For each gene selection strategy (random, HVG, SVG)
- For each gene count (50, 100)

**Options:**

```bash
# Run specific tissues only
python run_experiments.py --data_root /data/hest --tissues lung breast

# Run specific strategies only
python run_experiments.py --data_root /data/hest --strategies hvg svg

# Run only baseline models (faster, no distillation)
python run_experiments.py --data_root /data/hest --models tinyvit_baseline

# Dry run (see what would be executed without running)
python run_experiments.py --data_root /data/hest --dry_run

# Resume interrupted experiments (skip completed ones)
python run_experiments.py --data_root /data/hest --resume
```

### Step 2: Generate Figures

After experiments complete, generate all paper figures:

```bash
python generate_figures.py --results_dir ./outputs --output_dir ./figures
```

**Options:**

```bash
# Generate only performance figure
python generate_figures.py --figures performance

# Generate only pathway figures
python generate_figures.py --figures pathways
```

---

## Using Your Own Results

If you ran experiments differently and want to generate the same figures, point to your results directory:

```bash
python generate_figures.py --results_dir /path/to/your/results
```

### Required Results Format

Your results directory must follow this structure:

```
results_dir/
├── experiment_1/
│   ├── metrics.json       # Required
│   ├── gene_list.pkl      # Required for pathway analysis
│   └── config.json        # Optional
├── experiment_2/
│   ├── metrics.json
│   ├── gene_list.pkl
│   └── config.json
└── ...
```

### `metrics.json` Format

```json
{
    "r2": 0.65,
    "mse": 0.023,
    "pearson_mean": 0.72,
    "spearman_mean": 0.68
}
```

### `gene_list.pkl` Format

Python pickle file containing a list of gene names:

```python
['GENE1', 'GENE2', 'GENE3', ...]
```

### Experiment Naming Convention

For automatic parsing, name your experiment directories as:

```
{tissue}_{strategy}_{n_genes}_{model}
```

Examples:
- `lung_hvg_100_tinyvit_baseline`
- `breast_svg_50_tinyvit_uni_distill`

If your naming differs, the scripts will still work but may not correctly categorize results by tissue/strategy/model.

---

## Output

Generated figures are saved as both PDF and PNG:

| Figure | Filename | Description |
|--------|----------|-------------|
| Figure A | `st_inference_performance.pdf` | ST Inference Performance comparison |
| Pathway - Lung | `pathways_lung.pdf` | Top 5 pathways for lung tissue |
| Pathway - Breast | `pathways_breast.pdf` | Top 5 pathways for breast tissue |
| Pathway - Colon | `pathways_colon.pdf` | Top 5 pathways for colon tissue |
| Pathway - Prostate | `pathways_prostate.pdf` | Top 5 pathways for prostate tissue |
| Pathway - Skin | `pathways_skin.pdf` | Top 5 pathways for skin tissue |

---

## Prerequisites

### Environment Setup

```bash
# Using conda (recommended)
conda env create -f ../environment.yml
conda activate stainfactor

# Or using pip
pip install torch torchvision timm scanpy h5py scikit-learn scipy
pip install huggingface_hub scikit-image gseapy matplotlib seaborn
```

### Data Requirements

- HEST data organized by tissue type
- HuggingFace token set for foundation model access (for distillation)

### Hardware Requirements

- GPU with 16GB+ VRAM for baseline training
- GPU with 24GB+ VRAM for distillation (running both teacher and student)

---

## Troubleshooting

### "No results found"

Make sure you've run the experiments first:

```bash
python run_experiments.py --data_root /path/to/data
```

Or point to an existing results directory:

```bash
python generate_figures.py --results_dir /path/to/results
```

### "gseapy not available"

Install gseapy for pathway analysis:

```bash
pip install gseapy
```

### Experiments failing

Check the `experiment_log.json` in your output directory for detailed error messages. Common issues:
- Data path incorrect
- Insufficient GPU memory
- Missing HuggingFace authentication for foundation models
