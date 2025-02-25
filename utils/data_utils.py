'''
Methods and classes to handle data loading and processing.
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

def read_h5_patches(path):
    with h5py.File(path, 'r') as h5f:
        img = h5f['img'][:]  # Read the dataset into a NumPy array
        barcode=h5f['barcode'][:]
    return img
    
class STPatchDatasetHEST(Dataset):
    '''
    Dataset class to load ST patches and corresponding expression profiles, from the HEST dataset(https://github.com/mahmoodlab/HEST)
    
    patches_path (ex: "hest_data/patches_path): path to patch images saved in H5 arrays. Each H5 contains all patches for given sample.
    adata_path (ex: "hest_data/st/"): path to ST patch expression saved in AnnData stored as h5ad. Each h5ad contains expression AnnData for given sample
    samples (ex: ["NCBI516", "NCBI512"...]): list with specific sample IDs to include, such as if only including training samples in this dataset.
    gene_list_path (ex: "luad_top100_genes.pkl"): path to list with gene names to include in each expression profile.
    transforms: torchvision transform composition for images 
    
    Example usage: 
    patches_path='hest_data/patches/'
    adata_path='hest_data/st/'
    gene_list_path="luad_top100_genes"
    train_items,val_items=train_test_split(samples,test_size=0.3, random_state=42)
    transforms = transforms.Compose(
    [
        transforms.Resize(224),
        #transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
    train_dset=PatchDataset(patches_path, adata_path, train_items, gene_list_path, transform)
    
    '''
    def __init__(self, patches_path, adata_path, samples, gene_list_path, transforms):
        super().__init__()
        self.patches_path = patches_path 
        self.adata_path = adata_path
        self.samples = samples # list of samples, ex; ['TCGA-DS-31423','TCGA-RF-32321']
        self.all_items=[]
        self.sample2patches={}
        self.gene_list=pickle.load(open(gene_list_path,'rb')) 
        for item in samples:
            patch_path=self.patches_path+item+'.h5'
            patches=read_h5_patches(patch_path)
            
            adata=sc.read_h5ad(self.adata_path+item+'.h5ad')
            adata.var_names = adata.var_names.str.upper()
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            adata = adata[:, adata.var_names.isin(self.gene_list)]
            exp=adata.X.toarray()
            self.all_items+=[(patches[i],exp[i]) for i in range(len(patches))]#sample, index
        self.transforms=transforms
    def __len__(self):#total # of patches
        return len(self.all_items)
    def __getitem__(self, idx):
        patch,exp = self.all_items[idx]
        #patch=np.load(self.patches_path+sample+'/'+patch_name)
        patch=patch/255
        patch=torch.from_numpy(patch).permute(2, 0, 1).to(torch.float32)
        if self.transforms:
            patch_trans=self.transforms(patch)
        else: 
            patch_trans=patch

        #if exp.shape[0] < 100: #for some reason happened during training
        #    exp = torch.nn.functional.pad(exp, (0, 100 - exp.shape[0]), mode='constant', value=0)
        #    print('!!!PADDED!!!',idx,exp)
        return patch_trans,exp

class WSIClassDataset(Dataset):
    '''
    Dataset class to load WSI "bags" and corresponding slide-level labels, from arbitrary dataset. Needs to include a dir where there is one npy per WSI, with the npy being shaped (N, X, X, 3), N=number of patches in the image, X being the shape of a single patch.
    
    patches_path (ex: "wsi_dataset/patches/"): path to directory with WSI patches saved in .npy files. Nothing else should be in the directory. The name of each file should be ID.npy, where ID is a unique identifier for that sample. 
    metadata_path (ex: "wsi_dataset/metadata.txt"): path to txt/CSV file. Must include one column called "sample_id" and another called "class" where "sample_id" is the identifier for each WSI, and "class" is an integer value (0-C, where C=total # of classes) to be used as the image-level label.
    samples (ex: ["NCBI516", "NCBI512"...]): list with specific sample IDs to include, such as if only including training samples in this dataset.
    transforms: torchvision transform composition for images 
    
    Example usage: 
    patches_path='wsi_dataset/patches/'
    metadata_path='wsi_dataset/metadata.txt'
    train_items,val_items=train_test_split(samples,test_size=0.3, random_state=42)
    transforms = transforms.Compose(
    [
        transforms.Resize(224),
        #transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
    train_dset=WSIClassDataset(patches_path, metadata_path, train_items, transform)
    
    '''
    def __init__(self, patches_path, metadata_path, samples, transforms):
        super().__init__()
        self.patches_path = patches_path 
        self.metadata_path = metadata_path
        self.metadata=pd.read_csv(metadata_path)
        self.samples = samples # list of samples, ex; ['TCGA-DS-31423','TCGA-RF-32321']
        self.num_classes=max(self.metadata["class"])
        self.sample_ids=self.metadata["sample_id"]
        self.transforms=transforms
    def __len__(self):#total # of patches
        return len(self.samples)
    def __getitem__(self, idx):
        sample=self.samples[idx]
        patches=np.load(self.patches_path+sample+".npy")
        patches=patches/255
        patches=torch.from_numpy(patches).permute(2, 0, 1).to(torch.float32)
        if self.transforms:
            patch_trans=self.transforms(patch)
        else: 
            patch_trans=patch
        label = int(self.metadata[self.metadata["sample_id"]==sample]["class"].values[0])
        label_onehot = F.one_hot(label, num_classes=self.num_classes).float()
        return patch_trans,label_onehot
