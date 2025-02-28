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
        with open(log_path, "a") as f:
            f.write(f"{train_loss},{val_loss}\n")
        torch.save(model,save_dir+'model_epoch'+str(epoch))

def train_student_teacher(teacher_model, student_model, train_loader, val_loader, loss_fn, optim, epochs, device, log_path, save_dir):
    with open(log_path, "a") as f: # get logging ready
        #f.write("Train Loss,Val Loss,Val Pearson")
        f.write("Train Loss,Val Loss")
    temperature=2#arbitrary, should test
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx,data in enumerate(train_loader):
            #print(len(data))
            optim.zero_grad()
            imgs,true=data
            imgs = imgs.to(device)
            true = true.to(device)
            teacher_pred = teacher_model(imgs)
            student_pred = student_model(imgs)
            loss = loss_fn(student_pred, teacher_pred, true, temperature)
            
            loss.backward()
            train_loss+=loss.item()
            optim.step()
            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
        train_loss/=len(train_loader)
        val_loss = eval_student_teacher(teacher_model, student_model, val_loader, loss_fn, device)
        
        with open(log_path, "a") as f:
            f.write(f"{train_loss},{val_loss}\n")
        torch.save(student_model,save_dir+'model_epoch'+str(epoch))

def eval_student_teacher(teacher_model, student_model, val_loader, loss_fn, device):
    temperature=2#arbitrary, should test
    student_model.eval()
    val_loss=0
    all_true=[]
    all_pred=[]
    with torch.no_grad():
        for batch_idx,data in enumerate(val_loader):
            imgs,true=data
            imgs = imgs.to(device)
            true = true.to(device)
            teacher_pred = teacher_model(imgs)
            student_pred = student_model(imgs)
            loss = loss_fn(student_pred, teacher_pred, true, temperature)
            val_loss+=loss.item()
            all_true.append(true.detach().cpu().numpy())
            all_pred.append(student_pred.detach().cpu().numpy()) 
    val_loss/=len(val_loader)
    student_model.train()
    return val_loss

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

def finetune_WSICLASS_data(patches_path, metadata_path, train_samples, val_samples, log_dir, model, transforms, loss_fn,
                      hyperparams_dict):
        patches_path, metadata_path, samples, transforms

    '''
    Finetune a foundation model (ex: UNI2) on a WSI-level classification datset. Task: Infer stage from WSI (MIL approach).

    hyperparams_dict: learning_rate, epochs - batch size always 1 because processing as bags.
    '''
    train_dset = data_utils.WSIClassDataset(patches_path, metadata_path, train_samples, transforms)
    batch_size=hyperparams_dict['batch_size'] 
    train_loader=DataLoader(train_dset, batch_size=1, shuffle=True)
    
    val_dset=data_utils.WSIClassDataset(patches_path, metadata_path, val_samples, transforms)
    val_loader=DataLoader(val_dset, batch_size=1)

    device=torch.device('cuda')
    #loss_fn=torch.nn.MSELoss()
    loss_fn=loss_fn
    LR=hyperparams_dict['learning_rate']
    optim=torch.optim.Adam(model.parameters(),lr=LR)
    epochs=hyperparams_dict['epochs']
    log_path = log_dir+"log.txt"
    model_save_dir=log_dir
    
    model=model.to(device)
    
    train_single(model, train_loader, val_loader, loss_fn, optim, epochs,
         device,log_path,model_save_dir)

def distill_WSICLASS_data(patches_path, metadata_path, train_samples, val_samples, log_dir, teacher_model,
                      student_model, transforms, loss_fn, hyperparams_dict):
    '''
    Distill a task-specific foundation model into a lightweight student model. Task: Infer WSI class from input patches.

    hyperparams_dict: learning_rate, epochs - batch size always 1 because processing as bags.
    '''
    train_dset = data_utils.WSIClassDataset(patches_path, metadata_path, train_samples, transforms)
    batch_size=hyperparams_dict['batch_size'] 
    train_loader=DataLoader(train_dset, batch_size=1, shuffle=True)
    
    val_dset=data_utils.WSIClassDataset(patches_path, metadata_path, val_samples, transforms)
    val_loader=DataLoader(val_dset, batch_size=1)

    device=torch.device('cuda')
    loss_fn=loss_fn
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



    


