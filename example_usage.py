'''
Example usage of this framework
'''

import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tinyvit.models import tiny_vit_5m_224
import torch.nn as nn
import torchvision
import torch_geometric
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import scanpy as sc
import h5py
import pickle
import torch.nn.functional as F

import data_utils
import models
import train
import custom_losses

def finetune_UNI2_HEST():
    # ../hest_data is where HEST samples were downloaded into using their documentation/standard process. Selected only LUAD in this example case.
    patches_path="../hest_data/patches/"
    adata_path='../hest_data/hest_data/st/'
    gene_list_path="../hest_data/luad_ncbi_top100_genes_simple.pkl"
    log_dir="../saved_models/example_model_run/"
    num_classes=100
    hyperparams_dict = {"batch_size":16, "learning_rate":0.0001, "epochs":100}
    loss_fn = torch.nn.MSELoss()
    with open("hf_secret_key.txt", "r") as file:
        hf_key = file.readline().strip()
        
    model, transforms = models.load_model_and_transform_UNI2(hf_key, num_classes)
    
    #If want to freeze some layers while finetuning, would modify the model here (ie: freeze chosen layers)
    
    files = [q for q in os.listdir(patches_path) if 'ZEN' not in q]
    samples=[item.split('.')[0] for item in files]
    train_items,val_items=train_test_split(samples,test_size=0.3, random_state=42)

    train.finetune_HEST_data(patches_path, adata_path, train_items, val_items, gene_list_path, log_dir, model, transforms,
                            loss_fn, hyperparams_dict)

def distill_tinyvit_UNI2_HEST():
    # ../hest_data is where HEST samples were downloaded into using their documentation/standard process. Selected only LUAD in this example case.
    patches_path="../hest_data/patches/"
    adata_path='../hest_data/hest_data/st/'
    gene_list_path="../hest_data/luad_ncbi_top100_genes_simple.pkl"
    log_dir="../saved_models/example_model_run/"
    num_classes=100
    hyperparams_dict = {"batch_size":16, "learning_rate":0.0001, "epochs":100}
    loss_fn = custom_losses.distillation_loss
    
    with open("hf_secret_key.txt", "r") as file:
        hf_key = file.readline().strip()
        
    teacher_model, transforms = models.load_model_and_transform_UNI2(hf_key, num_classes)
    student_model = models.load_tiny_vit_5m_224(num_classes=num_classes)

    files = [q for q in os.listdir(patches_path) if 'ZEN' not in q]
    samples=[item.split('.')[0] for item in files]
    train_items,val_items=train_test_split(samples,test_size=0.3, random_state=42)

    train.distill_HEST_data(patches_path, adata_path, train_items, val_items, gene_list_path, log_dir, teacher_model,
                            student_model, transforms, loss_fn, hyperparams_dict)