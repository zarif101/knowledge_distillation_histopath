'''
Script that can be run from the command line to distill teacher into student model, for now just task-specific. Input the model paths, dataset, hyperparams, etc., and will distill the teacher into the student. on the dataset.
TODOS:
- Add support for different models (ex: finetuning a control student model), datasets, losses
- Add support for feature-level distillation
- Add support for student model loading from disk, ex: if they have pretrained it somehow.

Example usage:
python distill.py --teacher_path UNI2 \
                   --dataset_name HEST \
                   --patches_path ../hest_data/patches/ \
                   --adata_path ../hest_data/hest_data/st/ \
                   --gene_list_path ../hest_data/luad_ncbi_top100_genes_simple.pkl \
                   --log_dir ../saved_models/example_model_run/ \
                   --num_classes 100 \
                   --batch_size 16 \
                   --learning_rate 0.0001 \
                   --epochs 100
                   -- loss_fn mse
                   -- hf_path hf_secret_key.txt
'''

import sys
import os

import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import h5py
import pickle
import torch.nn.functional as F
import tinyvit
from tinyvit.models import tiny_vit_5m_224
import torch.nn as nn
import torchvision
import torch_geometric
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from ..utils import data_utils
from ..modeling import models
from ..utils import train
from .utils import custom_losses

# Model loading function mapping
MODEL_LOADERS = {
    "UNI2": models.load_model_and_transform_UNI2,
    "TINYVIT":models.load_tiny_vit_5m_224
    # Future models can be added here
}

# Dataset fine-tuning function mapping
DATASET_FUNCTIONS = {
    "HEST": train.distill_HEST_data,
    # Future dataset functions can be added here
}

LOSS_FUNCTIONS = {
    "mse": torch.nn.MSELoss,
    # Future loss functions
}


def distill(teacher_path, teacher_model, student_model, dataset_name, patches_path, adata_path, gene_list_path, log_dir, num_classes, batch_size, learning_rate, epochs, loss_fn,
            hf_path):
    """Fine-tune a selected model on a selected dataset."""
    
    if student_model not in MODEL_LOADERS:
        raise ValueError(f"Student model '{model_name}' is not recognized. Available models: {list(MODEL_LOADERS.keys())}")
    if teacher_model not in MODEL_LOADERS:
        raise ValueError(f"Student model '{model_name}' is not recognized. Available models: {list(MODEL_LOADERS.keys())}")

    if dataset_name not in DATASET_FUNCTIONS:
        raise ValueError(f"Dataset '{dataset_name}' is not recognized. Available datasets: {list(DATASET_FUNCTIONS.keys())}")

    loss_fn = LOSS_FUNCTIONS[loss_fn]
    loss_fn = loss_fn()

    with open(hf_path, "r") as file:
        hf_key = file.readline().strip()

    # Load teacher model model
    teacher_loader = MODEL_LOADERS[model_name]
    teacher_model_temp, transforms = model_loader(hf_key, num_classes)
    teacher_model=torch.load(teacher_path)
    teacher_model.eval()
    del teacher_model_temp

    # Load student model
    student_model_loader = MODEL_LOADERS[model_name]
    student_model = student_model_loader(num_classes)

    # List and split dataset
    files = [q for q in os.listdir(patches_path) if 'ZEN' not in q]
    samples = [item.split('.')[0] for item in files]
    train_items, val_items = train_test_split(samples, test_size=0.3, random_state=42)

    # Hyperparameters
    hyperparams_dict = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs
    }

    # Select dataset-specific training function
    dataset_function = DATASET_FUNCTIONS[dataset_name]
    dataset_function(
        patches_path, adata_path, train_items, val_items, gene_list_path, log_dir, teacher_model, student_model, transforms, loss_fn, hyperparams_dict
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill a teacher model into a student model, for a specific dataset/task.")
    
    parser.add_argument("--teacher_path", type=str, required=True, choices=MODEL_LOADERS.keys(), help="Path to teacher model (full model saved)")
    parser.add_argument("--teacher_model", type=str, required=True, choices=MODEL_LOADERS.keys(), help="Name of teacher model (used to get transforms)")
    parser.add_argument("--student_model", type=str, required=True, choices=MODEL_LOADERS.keys(), help="Name of student model")
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_FUNCTIONS.keys(), help="Dataset to fine-tune on.")
    parser.add_argument("--patches_path", type=str, required=True, help="Path to the patches directory.")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to the AnnData directory.")
    parser.add_argument("--gene_list_path", type=str, required=True, help="Path to the gene list pickle file.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory to save model logs.")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of output classes.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--loss_fn", type=str, default="mse", help="Loss function to use", choices=LOSS_FUNCTIONS.keys())
    parser.add_argument("--hf_path", type=str, help="Math to HF secret key (needed if using an HF model like UNI2)")

    args = parser.parse_args()

    finetune(
        teacher_path=args.teacher_path,
        teacher_model=args.teacher_model,
        student_model=args.student_model
        dataset_name=args.dataset_name,
        patches_path=args.patches_path,
        adata_path=args.adata_path,
        gene_list_path=args.gene_list_path,
        log_dir=args.log_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        loss_fn=args.loss_fn,
        hf_path=args.hf_path
    )
