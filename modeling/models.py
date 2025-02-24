'''
Methods to load all models, including teacher models (large) and student models (smaller). Can also load associated transforms for pretrained teacher models.

TODOs: Make flexible to allow loading from pretrained checkpoint/if saved locally on device
'''
from huggingface_hub import login
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
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

def load_model_and_transform_UNI2(huggingface_key, num_classes):
    '''
    Load UNI2 model and associated transforms.

    huggingface_key (ex: hf12p312312): private key from HuggingFace required to access this model (get from HuggingFace if you don't have one)

    Example usage:
    hf_key = "hf131231990123"
    uni_model, uni_transforms = load_model_and_transform_UNI2(hf_key)
    '''
    timm_kwargs = {
       'img_size': 224, 
       'patch_size': 14, 
       'depth': 24,
       'num_heads': 24,
       'init_values': 1e-5, 
       'embed_dim': 1536,
       'mlp_ratio': 2.66667*2,
       'num_classes': num_classes, 
       'no_embed_class': True,
       'mlp_layer': timm.layers.SwiGLUPacked, 
       'act_layer': torch.nn.SiLU, 
       'reg_tokens': 8, 
       'dynamic_img_size': True
      }
    uni_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            #transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    return model, uni_transforms

def load_tiny_vit_5m_224(pretrained=False,num_classes=100):
    return tiny_vit_5m_224(pretrained=pretrained,num_classes=num_classes)
    