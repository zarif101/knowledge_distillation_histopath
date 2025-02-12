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
import finetune


def finetune_UNI2_HEST():
    # ../hest_data is where HEST samples were downloaded into using their documentation/standard process. Selected only LUAD in this example case.
    patches_path="../hest_data/patches/"
    adata_path='../hest_data/hest_data/st/'
    gene_list_path="../hest_data/luad_ncbi_top100_genes_simple.pkl"
    log_dir="../saved_models/example_model_run/"
    
    with open("hf_secret_key.txt", "r") as file:
        hf_key = file.readline().strip()
        
    model, transforms = models.load_model_and_transform_UNI2(hf_key)

    files = [q for q in os.listdir(patches_path) if 'ZEN' not in q]
    samples=[item.split('.')[0] for item in files]
    train_items,val_items=train_test_split(samples,test_size=0.3, random_state=42)

    finetune.finetune_HEST_data(patches_path, adata_path, train_items, val_items, gene_list_path, gene_list_path, log_dir, model, transforms)
'''
To implement:
1. KD (invariant to output/level, allow control of that)
2. Add some more models
'''