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
    login(huggingface_key)
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

# By default, cannot "add in" an output layer with defined size to the end of Virchow2, using the HF import. Thus, adding one in ourselves, trainable layer.
class Virchow2Extended(nn.Module):
    def __init__(self, base_model, output_dim):
        super().__init__()
        self.base_model = base_model  # Pretrained Virchow2 model
        self.fc = nn.Linear(2560, output_dim)  # Map embedding to output_dim

    def forward(self, image):
        output = self.base_model(image)  # size: 1 x 261 x 1280

        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280

        # Compute final embedding
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

        # Pass through the linear layer
        out = self.fc(embedding)  # size: 1 x output_dim
        return out
class Virchow2Emb(nn.Module):
    def __init__(self, base_model, output_dim):
        super().__init__()
        self.base_model = base_model  # Pretrained Virchow2 model
        self.fc = nn.Linear(2560, output_dim)  # Map embedding to output_dim

    def forward(self, image):
        output = self.base_model(image)  # size: 1 x 261 x 1280

        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280

        # Compute final embedding
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        return embedding
        
def load_model_and_transform_VIRCHOW2(huggingface_key, num_classes):
    '''
    Load Virchow2 model and associated transforms. Adds on a classification layer at the end, as default Virchow is just a frozen feature extractor.

    huggingface_key (ex: hf12p312312): private key from HuggingFace required to access this model (get from HuggingFace if you don't have one)

    Example usage:
    hf_key = "hf131231990123"
    virchow_model, virchow_transforms = load_model_and_transform_VIRCHOW2(hf_key, num_classes=100)
    '''
    from timm.layers import SwiGLUPacked
    login(huggingface_key)
    base_model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    if num_classes!=0:
        model=Virchow2Extended(base_model=base_model,output_dim=num_classes)
    else:
        model=Virchow2Emb(base_model=base_model)
    virchow_transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
    return model, virchow_transforms


class WSIMILClassifier(nn.Module):
    def __init__(self, encoder_model, feature_dim, n_classes):
        """
        Simple WSI MIL classification model.

        Args:
            encoder_model (nn.Module): Patch encoder model (e.g., ResNet, ViT).
            feature_dim: size of embedding from encoder model, ex: 2048
            output_size (int): Number of output classes.
        """
        super(WSIMILClassifier, self).__init__()
        self.encoder = encoder_model  # Patch encoder

        # Fully connected layer for bag-level classification
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input patches of shape (N, 3, 224, 224), representing one WSI.

        Returns:
            torch.Tensor: Softmax probabilities of shape (1, output_size).
        """
        # Encode each patch
        patch_features = self.encoder(x)  # Shape: (N, feature_dim)
        # Mean pooling across all patches (bag-level embedding)
        bag_embedding = patch_features.mean(dim=0, keepdim=True)  # Shape: (1, feature_dim)

        # Classification layer
        logits = self.fc(bag_embedding)  # Shape: (1, output_size)
        probs = F.softmax(logits, dim=-1)  # Apply softmax for class probabilities
        return probs  # Output shape: (1, output_size)

    