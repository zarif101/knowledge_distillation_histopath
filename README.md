# Evaluation of Histopathology Foundation Model Knowledge Distillation

> **Note:** This repo is actively being built out.

## Goals

- Enable knowledge distillation of histopathology foundation models into smaller, task-specific **student models** of various scales.
- Enable **simple benchmarking** of distilled task-specific models on representative datasets.

## Basic Usage (to be further updated)

### **Current Capabilities**
- **Fine-tune** a foundation model (FM) on **HEST ST prediction** (evaluate different FMs).
- **Fine-tune** a foundation model (FM) on **WSI class prediction** (evaluate different FMs).
- **Distill** a fine-tuned FM into a **smaller lightweight model** for **HEST ST prediction**.
- **Distill** a fine-tuned FM into a **smaller lightweight model** for **WSI class prediction**.

### **Current Models**
- **Foundation Models (FMs):** `UNI2`, `Virchow`
- **Lightweight Models:** `TinyVIT` (only supported currently)

### **TODOs**
- Fine-tune & distill for **ROSIE virtual staining dataset**.
- Add more **FM and lightweight model** options.
- Improve **documentation and organization**.
- Move **loss functions** to a single file instead of defining them separately for each model.

## **Usage Examples**

The main interaction with this package is through the scripts in the **`pipeline/`** directory.

### **Fine-tuning a Foundation Model on HEST ST Prediction**
```bash
python -m pipeline.finetune_HEST --model_name UNI2 --dataset_name HEST \
    --patches_path hest_data/patches/ --adata_path hest_data/st/ \
    --gene_list_path hest_data/luad_ncbi_top100_genes_v2.pkl \
    --log_dir saved_models/example_model_run/ --num_classes 100 \
    --batch_size 16 --learning_rate 0.0001 --epochs 100 \
    --loss_fn mse --hf_path package/hf_secret_key.txt
```

### ** Fine-tuning a Foundation Model on WSI Classification
```bash
python -m pipeline.finetune_WSICLASS --model_name UNI2 --dataset_name WSICLASS \
    --patches_path wsidataset/patches_path/ --metadata_path wsidataset/metadata.csv \
    --log_dir saved_models/example_model_run/ --num_classes 10 \
    --learning_rate 0.0001 --epochs 100 --loss_fn ce \
    --hf_path package/hf_secret_key.txt
```

### ** Distilling a Fine-tuned FM for HEST ST Prediction
```bash
python -m pipeline.distill_HEST --teacher_path UNI2.h5 --dataset_name HEST \
    --patches_path hest_data/patches/ --adata_path hest_data/st/ \
    --log_dir saved_models/example_model_run/ --num_classes 10 \
    --learning_rate 0.0001 --batch_size 16 --epochs 100 --loss_fn ce \
    --hf_path package/hf_secret_key.txt
```
### ** Distilling a Fine-tuned FM for WSI Classification
```bash
python -m pipeline.distill_WSICLASS --teacher_path UNI2.h5 --dataset_name WSICLASS \
    --patches_path wsidataset/patches_path/ --metadata_path wsidataset/metadata.csv \
    --log_dir saved_models/example_model_run/ --num_classes 10 \
    --learning_rate 0.0001 --epochs 100 --loss_fn ce \
    --hf_path package/hf_secret_key.txt
```

