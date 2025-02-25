'''
Script that can be run from the command line to finetune models. Input the model, dataset, hyperparams, etc., and will finetune the model on the dataset.
TODOS:
- Add support for different models (ex: finetuning a control student model), datasets, losses

Example usage:
python finetune_WSICLASS.py --model_name UNI2 \
                   --dataset_name HEST \
                   --patches_path ../wsidata/patches/ \
                   --metadata_path ../wsidata/metadata.csv \
                   --num_classes 10
                   --log_dir ../saved_models/example_model_run/ \
                   --num_classes 100 \
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
from ..utils import train_WSICLASS
from ..utils import custom_losses

# Model loading function mapping
MODEL_LOADERS = {
    "UNI2": models.load_model_and_transform_UNI2,
    "VIRCHOW2": models.load_model_and_transform_VIRCHOW2
    # Future models can be added here
}

# Dataset fine-tuning function mapping
DATASET_FUNCTIONS = {
    "HEST": train_WSICLASS.finetune_WSICLASS_data,
    # Future dataset functions can be added here
}

LOSS_FUNCTIONS = {
    "mse": torch.nn.MSELoss,
    # Future loss functions
}


def finetune(model_name, dataset_name, patches_path, metadata_path, num_classes, log_dir, num_classes, batch_size, learning_rate, epochs, loss_fn,
            hf_path):
    """Fine-tune a selected model on a selected dataset."""
    
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Model '{model_name}' is not recognized. Available models: {list(MODEL_LOADERS.keys())}")

    if dataset_name not in DATASET_FUNCTIONS:
        raise ValueError(f"Dataset '{dataset_name}' is not recognized. Available datasets: {list(DATASET_FUNCTIONS.keys())}")

    loss_fn = LOSS_FUNCTIONS[loss_fn]
    loss_fn = loss_fn()

    with open(hf_path, "r") as file:
        hf_key = file.readline().strip()

    # Load selected model
    model_loader = MODEL_LOADERS[model_name]
    encoder_model, transforms = model_loader(hf_key, 0)
    last_layer = list(encoder_model.children())[-1]  # Access last layer
    feature_dim = last_layer.out_features if hasattr(last_layer, 'out_features') else None
    model = models.WSIMILClassifier(encoder_model, feature_dim, num_classes)

    # List and split dataset
    files = [q for q in os.listdir(patches_path) if 'ZEN' not in q]
    samples = [item.split('.')[0] for item in files]
    train_items, val_items = train_test_split(samples, test_size=0.3, random_state=42)

    # Hyperparameters
    hyperparams_dict = {
        "learning_rate": learning_rate,
        "epochs": epochs
    }

    # Select dataset-specific training function
    dataset_function = DATASET_FUNCTIONS[dataset_name]
    dataset_function(
        patches_path, adata_path, train_items, val_items, log_dir, model, transforms, loss_fn, hyperparams_dict
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on a selected dataset.")
    
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_LOADERS.keys(), help="Model to fine-tune.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_FUNCTIONS.keys(), help="Dataset to fine-tune on.")
    parser.add_argument("--num_classes", type=int, required=True, choices=DATASET_FUNCTIONS.keys(), help="Total num classes")
    parser.add_argument("--patches_path", type=str, required=True, help="Path to the patches directory.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata CSV.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory to save model logs.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--loss_fn", type=str, default="mse", help="Loss function to use", choices=LOSS_FUNCTIONS.keys())
    parser.add_argument("--hf_path", type=str, help="Path to HF secret key (needed if using an HF model like UNI2)")

    args = parser.parse_args()

    finetune(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        patches_path=args.patches_path,
        metadata_path=args.metadata_path,
        num_classes=args.num_classes,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        loss_fn=args.loss_fn,
        hf_path=args.hf_path
    )
