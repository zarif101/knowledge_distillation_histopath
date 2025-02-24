'''
Loss functions not standard in PyTorch (ex: knowledge distillation loss)
'''

import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torchvision
import torch_geometric
import timm
from torchvision import transforms
import torch.nn.functional as F

def distillation_loss(
    student_preds: torch.Tensor,
    teacher_preds: torch.Tensor,
    true_labels: torch.Tensor,
    T: float = 1.0,
    soft_target_loss_weight: float = 0.5,
    label_loss_weight: float = 0.5,
) -> torch.Tensor:
    """
    Computes a knowledge distillation loss for regression using MSE.
    Parameters:
    - student_preds (torch.Tensor): Predictions from the student model.
    - teacher_preds (torch.Tensor): Predictions from the teacher model.
    - true_labels (torch.Tensor): Ground truth labels.
    - T (float): Temperature parameter for softening the teacher predictions.
    - soft_target_loss_weight (float): Weight for the distillation (soft target) loss.
    - label_loss_weight (float): Weight for the true label loss.

    Returns:
    - torch.Tensor: Combined knowledge distillation loss.
    """
    # Apply temperature scaling to teacher and student predictions
    teacher_soft = teacher_preds / T
    student_soft = student_preds / T

    # Calculate the distillation loss (soft target loss) using MSE
    soft_targets_loss = F.mse_loss(student_soft, teacher_soft) * (T ** 2)

    # Calculate the true label loss using MSE
    label_loss = F.mse_loss(student_preds, true_labels)

    # Combine the losses
    loss = soft_target_loss_weight * soft_targets_loss + label_loss_weight * label_loss
    return loss