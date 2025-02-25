import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import pickle
import torch.nn.functional as F
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from ..utils import data_utils
from ..modeling import models
from ..utils import train_WSICLASS
from ..utils import custom_losses

# Model loading function mapping
MODEL_LOADERS = {
    "UNI2": models.load_model_and_transform_UNI2,
    "VIRCHOW2": models.load_model_and_transform_VIRCHOW2,
    # Add other models here if needed
}

# Dataset training function mapping
DATASET_FUNCTIONS = {
    "WSICLASS": train_WSICLASS.distill_WSICLASS_data,  # Function for knowledge distillation
    # Add other datasets here if needed
}

# Loss function mapping
LOSS_FUNCTIONS = {
    "mse": torch.nn.MSELoss,
    # Add more loss functions here if needed
}

def distill(teacher_path, teacher_model, student_model, dataset_name, patches_path, metadata_path, num_classes, log_dir,
            batch_size, learning_rate, epochs, loss_fn, hf_path):
    """
    Distill a teacher model into a student model for WSI MIL classification.
    """
    
    if teacher_model not in MODEL_LOADERS:
        raise ValueError(f"Teacher model '{teacher_model}' is not recognized. Available models: {list(MODEL_LOADERS.keys())}")
    if student_model not in MODEL_LOADERS:
        raise ValueError(f"Student model '{student_model}' is not recognized. Available models: {list(MODEL_LOADERS.keys())}")
    if dataset_name not in DATASET_FUNCTIONS:
        raise ValueError(f"Dataset '{dataset_name}' is not recognized. Available datasets: {list(DATASET_FUNCTIONS.keys())}")

    # Select loss function
    loss_fn = LOSS_FUNCTIONS[loss_fn]()
    
    # Load Hugging Face API key if needed
    if hf_path:
        with open(hf_path, "r") as file:
            hf_key = file.readline().strip()
    else:
        hf_key = None

    # Load teacher model
    teacher_loader = MODEL_LOADERS[teacher_model]
    #teacher_model_instance, _ = teacher_loader(hf_key, 0)
    #last_layer = list(teacher_model_instance.children())[-1]  # Access last layer
    #feature_dim = last_layer.out_features if hasattr(last_layer, 'out_features') else None
    #teacher_model_instance = models.WSIMILClassifier(teacher_model_instance, feature_dim, num_classes)
    teacher_model_instance = torch.load(teacher_path)
    teacher_model_instance.eval()

    # Load student model
    student_loader = MODEL_LOADERS[student_model]
    student_encoder, transforms = student_loader(hf_key, 0)
    last_layer = list(student_encoder.children())[-1]  # Access last layer
    feature_dim = last_layer.out_features if hasattr(last_layer, 'out_features') else None
    student_model_instance = models.WSIMILClassifier(student_encoder, feature_dim, num_classes)
    
    

    # List and split dataset
    files = [q for q in os.listdir(patches_path) if 'ZEN' not in q]
    samples = [item.split('.')[0] for item in files]
    train_items, val_items = train_test_split(samples, test_size=0.3, random_state=42)

    # Hyperparameters
    hyperparams_dict = {
        "learning_rate": learning_rate,
        "epochs": epochs
    }

    # Select dataset-specific distillation function
    dataset_function = DATASET_FUNCTIONS[dataset_name]
    dataset_function(
        patches_path, metadata_path, train_items, val_items, log_dir, teacher_model_instance, student_model_instance,
        transforms, loss_fn, hyperparams_dict
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill a teacher model into a student model for WSI MIL classification.")
    
    parser.add_argument("--teacher_path", type=str, required=True, help="Path to the teacher model checkpoint.")
    parser.add_argument("--teacher_model", type=str, required=True, choices=MODEL_LOADERS.keys(), help="Name of the teacher model.")
    parser.add_argument("--student_model", type=str, required=True, choices=MODEL_LOADERS.keys(), help="Name of the student model.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_FUNCTIONS.keys(), help="Dataset to distill on.")
    parser.add_argument("--patches_path", type=str, required=True, help="Path to the patches directory.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata CSV.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory to save model logs.")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of output classes.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--loss_fn", type=str, default="mse", choices=LOSS_FUNCTIONS.keys(), help="Loss function to use.")
    parser.add_argument("--hf_path", type=str, help="Path to HF secret key (needed if using a Hugging Face model like UNI2).")

    args = parser.parse_args()

    distill(
        teacher_path=args.teacher_path,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
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
