'''
Methods to train models: finetune pretrained models, train "controls" (student model without a teacher), train knowledge distillation. Envisioned usage example: finetune foundation model on specific dataset, prior to doing knowledge distillation to distill the finetuned foundation model into a smaller lightweight model.
Currently assuming dataset with single image patches as X, regression targets as y (ex: ST from patches).
TODOs: 
- Support eval functions per epoch (ex: correlation)
- Support different types of alignment (ex: feature level instead of just output level)
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
from scipy.stats import pearsonr

from . import data_utils
#import data_utils
from ..modeling import models
from . import custom_losses

def train_single(model, train_loader, val_loader, loss_fn, optim, epochs, device, log_path, save_dir):
    with open(log_path, "a") as f: # get logging ready
        f.write("Train Loss,Val Loss")
    best_val_loss = float('inf')  # Track best validation loss

    # Temp for Testing #
    '''
    val_loss,val_true,val_pred = eval_single(model, val_loader, loss_fn, device)
    val_tru_arr=np.concatenate(val_true)
    val_pred_arr=np.concatenate(val_pred)
    corrs=[]
    for i in range(100):
        corr = pearsonr(val_tru_arr[:,i],val_pred_arr[:,i])
        corrs.append(corr.statistic)
    corr=sum(corrs)/len(corrs)
    '''
    #End Temp for Testing #
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx,data in enumerate(train_loader):
            optim.zero_grad()
            imgs,y_true=data
            imgs = imgs.to(device)
            y_true = y_true.to(device)
            logits=model(imgs)
            loss=loss_fn(logits,y_true)
            
            loss.backward()
            train_loss+=loss.item()
            optim.step()
            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
        train_loss/=len(train_loader)
        val_loss,val_true,val_pred = eval_single(model, val_loader, loss_fn, device)
        
        #val_tru_arr=np.squeeze(val_true)
        #val_pred_arr=np.squeeze(val_pred)
        val_tru_arr=np.concatenate(val_true)
        val_pred_arr=np.concatenate(val_pred)
        corrs=[]
        for i in range(len(torch.squeeze(y_true))):
            corr = pearsonr(val_tru_arr[:,i],val_pred_arr[:,i])
            corrs.append(corr.statistic)
        corr=sum(corrs)/len(corrs)
        with open(log_path, "a") as f:
            f.write(f"{train_loss},{val_loss},{corr}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, save_dir + 'model_epoch' + str(epoch))

def train_student_teacher(teacher_model, student_model, train_loader, val_loader, loss_fn, optim, epochs, device, log_path, save_dir):
    with open(log_path, "a") as f: # get logging ready
        #f.write("Train Loss,Val Loss,Val Pearson")
        f.write("Train Loss,Val Loss")
    temperature=2#arbitrary, should test
    best_val_loss = float('inf')  # Track best validation loss

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx,data in enumerate(train_loader):
            #print(len(data))
            optim.zero_grad()
            imgs,exp_true=data
            imgs = imgs.to(device)
            exp_true = exp_true.to(device)
            teacher_pred = teacher_model(imgs)
            student_pred = student_model(imgs)
            loss = loss_fn(student_pred, teacher_pred, exp_true, temperature)
            
            loss.backward()
            train_loss+=loss.item()
            optim.step()
            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
        train_loss/=len(train_loader)
        #val_loss, val_corr = eval(teacher_model, student_model, val_loader, loss_fn, device)
        val_loss = eval_student_teacher(teacher_model, student_model, val_loader, loss_fn, device)
        
        with open(log_path, "a") as f:
            #f.write(f"{train_loss},{val_loss},{val_corr}\n")
            f.write(f"{train_loss},{val_loss}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model, save_dir + 'model_epoch' + str(epoch))

def eval_student_teacher(teacher_model, student_model, val_loader, loss_fn, device):
    temperature=2#arbitrary, should test
    student_model.eval()
    val_loss=0
    all_true=[]
    all_pred=[]
    with torch.no_grad():
        for batch_idx,data in enumerate(val_loader):
            imgs,exp_true=data
            imgs = imgs.to(device)
            exp_true = exp_true.to(device)
            teacher_pred = teacher_model(imgs)
            student_pred = student_model(imgs)
            loss = loss_fn(student_pred, teacher_pred, exp_true, temperature)
            val_loss+=loss.item()
            all_true.append(exp_true.detach().cpu().numpy())
            all_pred.append(student_pred.detach().cpu().numpy()) 
    val_loss/=len(val_loader)
    student_model.train()
    
    #avg_corr='NO_AVG_FORNOW'
    #print('VAL AVG CORR', avg_corr)
    #return val_loss, avg_corr
    return val_loss
'''Feats'''
def train_student_teacher_featurelevel(teacher_model, student_model, train_loader, val_loader, loss_fn, optim, epochs, device, log_path, save_dir):
    with open(log_path, "a") as f: # get logging ready
        #f.write("Train Loss,Val Loss,Val Pearson")
        f.write("Train Loss,Val Loss")
    temperature=2#arbitrary, should test
    best_val_loss = float('inf')  # Track best validation loss

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx,data in enumerate(train_loader):
            #print(len(data))
            optim.zero_grad()
            imgs,exp_true=data
            imgs = imgs.to(device)
            exp_true = exp_true.to(device)
            teacher_feats = teacher_model(imgs)
            student_pred = student_model(imgs)
            student_feats=student_model.forward_features(imgs)
            loss = loss_fn(student_feats, student_pred, teacher_feats, exp_true, temperature)
            
            loss.backward()
            train_loss+=loss.item()
            optim.step()
            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print('EPOCH DONE')
        train_loss/=len(train_loader)
        #val_loss, val_corr = eval(teacher_model, student_model, val_loader, loss_fn, device)
        val_loss = eval_student_teacher_featurelevel(teacher_model, student_model, val_loader, loss_fn, device)
        
        with open(log_path, "a") as f:
            #f.write(f"{train_loss},{val_loss},{val_corr}\n")
            f.write(f"{train_loss},{val_loss}\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('About to save')
            torch.save(student_model,save_dir+'model_epoch'+str(epoch))
            print("SAVED")
def eval_student_teacher_featurelevel(teacher_model, student_model, val_loader, loss_fn, device):
    temperature=2#arbitrary, should test
    student_model.eval()
    val_loss=0
    all_true=[]
    all_pred=[]
    with torch.no_grad():
        for batch_idx,data in enumerate(val_loader):
            imgs,exp_true=data
            imgs = imgs.to(device)
            exp_true = exp_true.to(device)
            teacher_feats = teacher_model(imgs)
            #student_feats, student_pred = student_model(imgs)
            student_pred = student_model(imgs)
            student_feats=student_model.forward_features(imgs)
            loss = loss_fn(student_feats, student_pred, teacher_feats, exp_true, temperature)
            val_loss+=loss.item()
            all_true.append(exp_true.detach().cpu().numpy())
            all_pred.append(student_pred.detach().cpu().numpy()) 
    val_loss/=len(val_loader)
    student_model.train()
    return val_loss
'''Feats'''

def eval_single(model, val_loader, loss_fn, device):
    model.eval()
    val_loss=0
    all_true=[]
    all_pred=[]
    with torch.no_grad():
        for batch_idx,data in enumerate(val_loader):
            imgs,y_true=data
            imgs = imgs.to(device)
            y_true = y_true.to(device)
            preds=model(imgs)
            loss=loss_fn(preds,y_true)
            val_loss+=loss.item()
            all_true.append(y_true.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())
    val_loss/=len(val_loader)
    model.train()
    return val_loss, all_true, all_pred

def finetune_HEST_data(patches_path, adata_path, train_samples, val_samples, gene_list, log_dir, model, transforms,
                      hyperparams_dict, loss_fn=None):
    '''
    Finetune a foundation model (ex: UNI2) on HEST data. Task: Infer gene expression profiles from input patches.

    Args:
        patches_path: Path to directory with H5 patch files
        adata_path: Path to directory with h5ad expression files
        train_samples: List of training sample IDs
        val_samples: List of validation sample IDs
        gene_list: List of gene names OR path to pickle file
        log_dir: Directory for logs and model checkpoints
        model: Model to fine-tune
        transforms: Image transforms
        hyperparams_dict: Dict with batch_size, learning_rate, epochs
        loss_fn: Loss function (default: MSELoss)
    '''
    train_dset = data_utils.STPatchDatasetHEST(patches_path, adata_path, train_samples, gene_list, transforms)
    batch_size=hyperparams_dict['batch_size'] 
    train_loader=DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    
    val_dset=data_utils.STPatchDatasetHEST(patches_path, adata_path, val_samples, gene_list, transforms)
    val_loader=DataLoader(val_dset, batch_size=batch_size)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    LR=hyperparams_dict['learning_rate']
    optim=torch.optim.Adam(model.parameters(),lr=LR)
    epochs=hyperparams_dict['epochs']
    log_path = log_dir+"log.txt"
    model_save_dir=log_dir
    
    model=model.to(device)
    
    train_single(model, train_loader, val_loader, loss_fn, optim, epochs,
         device,log_path,model_save_dir)

def distill_HEST_data(patches_path, adata_path, train_samples, val_samples, gene_list, log_dir, teacher_model,
                      student_model, transforms, hyperparams_dict, loss_fn=None):
    '''
    Distill a task-specific foundation model into a lightweight student model. Task: Infer gene expression profiles from input patches.

    Args:
        patches_path: Path to directory with H5 patch files
        adata_path: Path to directory with h5ad expression files
        train_samples: List of training sample IDs
        val_samples: List of validation sample IDs
        gene_list: List of gene names OR path to pickle file
        log_dir: Directory for logs and model checkpoints
        teacher_model: Teacher model (frozen)
        student_model: Student model to train
        transforms: Image transforms
        hyperparams_dict: Dict with batch_size, learning_rate, epochs
        loss_fn: Distillation loss function (default: DistillationLoss)
    '''
    train_dset = data_utils.STPatchDatasetHEST(patches_path, adata_path, train_samples, gene_list, transforms)
    batch_size=hyperparams_dict['batch_size'] 
    train_loader=DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    
    val_dset=data_utils.STPatchDatasetHEST(patches_path, adata_path, val_samples, gene_list, transforms)
    val_loader=DataLoader(val_dset, batch_size=batch_size)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if loss_fn is None:
        loss_fn = custom_losses.DistillationLoss()
    LR=hyperparams_dict['learning_rate']
    optim=torch.optim.Adam(student_model.parameters(),lr=LR)
    epochs=hyperparams_dict['epochs']
    log_path = log_dir+"log.txt"
    model_save_dir=log_dir
    
    teacher_model=teacher_model.to(device)
    student_model=student_model.to(device)

    train_student_teacher(teacher_model, student_model, train_loader, val_loader, loss_fn, optim, epochs,
         device,log_path,model_save_dir)

    #This example usage is doing distillation at the output level. 


def distill_HEST_data_featurelevel(patches_path, adata_path, train_samples, val_samples, gene_list, log_dir, teacher_model,
                      student_model, transforms, hyperparams_dict, loss_fn=None):
    '''
    Distill a task-specific foundation model into a lightweight student model using feature-level alignment.
    Task: Infer gene expression profiles from input patches.

    Args:
        patches_path: Path to directory with H5 patch files
        adata_path: Path to directory with h5ad expression files
        train_samples: List of training sample IDs
        val_samples: List of validation sample IDs
        gene_list: List of gene names OR path to pickle file
        log_dir: Directory for logs and model checkpoints
        teacher_model: Teacher model (frozen)
        student_model: Student model to train
        transforms: Image transforms
        hyperparams_dict: Dict with batch_size, learning_rate, epochs
        loss_fn: Feature-level distillation loss (default: FeatureLevelDistillationLoss)
    '''
    train_dset = data_utils.STPatchDatasetHEST(patches_path, adata_path, train_samples, gene_list, transforms)
    batch_size=hyperparams_dict['batch_size'] 
    train_loader=DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    
    val_dset=data_utils.STPatchDatasetHEST(patches_path, adata_path, val_samples, gene_list, transforms)
    val_loader=DataLoader(val_dset, batch_size=batch_size)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if loss_fn is None:
        loss_fn = custom_losses.FeatureLevelDistillationLoss()
    LR=hyperparams_dict['learning_rate']
    optim=torch.optim.Adam(student_model.parameters(),lr=LR)
    epochs=hyperparams_dict['epochs']
    log_path = log_dir+"log.txt"
    model_save_dir=log_dir
    
    teacher_model=teacher_model.to(device)
    student_model=student_model.to(device)

    train_student_teacher_featurelevel(teacher_model, student_model, train_loader, val_loader, loss_fn, optim, epochs,
         device,log_path,model_save_dir)

    #This example usage is doing distillation at the output level. 



    


