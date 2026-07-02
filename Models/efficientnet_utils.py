"""
CNN Utilities Module.

This module encapsulates all Deep Learning primitives (PyTorch loops, MONAI initializations, 
data loading) into a static utility class (`CNNUtils`). It prevents global namespace pollution 
and isolates framework-specific syntax from the high-level orchestration logic.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    roc_auc_score, 
    f1_score, 
    confusion_matrix
)
from typing import Dict, Tuple
from monai.networks.nets import EfficientNetBN


class CNNUtils:
    """
    Static utility class for 3D Convolutional Neural Network operations.
    Handles data transformations, PyTorch model generation, and the mathematical 
    execution of training/validation epochs.
    """

    @staticmethod
    def build_model(model_name: str = "efficientnet-b0", in_channels: int = 1, num_classes: int = 2) -> nn.Module:
        """
        Initializes and returns a 3D EfficientNet architecture using the MONAI medical framework.
        
        Purpose:
            EfficientNet-B0 offers the optimal trade-off between depth, width, and resolution 
            for 3D volumes, preventing CUDA Out-Of-Memory errors typically associated with 
            dense architectures like ResNet50 in 3D space.
            
        Parameters:
            model_name (str): Specifies the EfficientNet variant.
            in_channels (int): Input channel dimension. 1 for standard grayscale T1/VBM MRI.
            num_classes (int): Output dimensionality. 2 for Binary Classification.
            
        Returns:
            nn.Module: The un-trained PyTorch 3D neural network model.
        """
        return EfficientNetBN(
            model_name=model_name, 
            spatial_dims=3, 
            in_channels=in_channels, 
            num_classes=num_classes, 
            pretrained=False
        )

    @staticmethod
    def load_real_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads Raw 3D NIfTI files and prepares them for PyTorch volumetric convolutions.
        
        Purpose:
            Unlike the SVM which requires 1D flat vectors, 3D CNNs require the preservation 
            of spatial topology. Furthermore, PyTorch Conv3D layers expect a strict 5D 
            tensor format: (Batch, Channels, Depth, Height, Width).
            
        Parameters:
            csv_path (str): Path to the CSV mapping file.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - subjects: 1D array of subject identifiers.
                - X_data: 5D NumPy array holding the volumetric image tensors.
                - y_data: 1D array of binary labels.
                
        Raises:
            FileNotFoundError: If the specified CSV registry is missing.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing Data file: {csv_path}")

        df = pd.read_csv(csv_path)
        subjects, X_list, y_list = [], [], []
        
        for _, row in df.iterrows():
            subjects.append(str(row['subject_id']))
            y_list.append(int(row['label']))
            
            # img_data shape is originally (Depth, Height, Width)
            img_data = nib.load(row['file_path']).get_fdata()
            
            # WHY: np.expand_dims creates the 'Channel' axis at index 0.
            # Without this, the model will crash with: 'RuntimeError: Expected 5D input'.
            img_tensor = np.expand_dims(img_data, axis=0)
            X_list.append(img_tensor)
            
        return np.array(subjects), np.array(X_list), np.array(y_list)

    @staticmethod
    def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
        """
        Executes a complete single Forward and Backward pass (Training Epoch) across all batches.
        
        Purpose:
            To update the neural network weights by calculating the gradients of the loss 
            function with respect to the network parameters via backpropagation.
            
        Parameters:
            model (nn.Module): The active PyTorch neural network.
            loader (DataLoader): The Data provider streaming batches of tensors.
            optimizer (optim.Optimizer): The algorithm updating the weights (e.g., AdamW).
            criterion (nn.Module): The loss function to minimize (e.g., CrossEntropyLoss).
            device (torch.device): The target compute hardware (CPU or CUDA).
            
        Returns:
            float: The arithmetic mean of the loss across all iterations in this epoch.
        """
        model.train() # Explicitly set model to training mode (enables Dropout and BatchNorm)
        epoch_loss = 0.0
        
        for inputs, labels in loader:
            # WHY: non_blocking=True is a high-performance feature. It allows the GPU to 
            # continue computing the current batch while asynchronously fetching the next batch 
            # from the CPU's pinned memory, eliminating Data Starvation bottlenecks.
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad() # Clear accumulated gradients from the previous batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # Compute partial derivatives
            optimizer.step() # Apply gradient descent step
            
            epoch_loss += loss.item()
            
        return epoch_loss / max(1, len(loader))

    @staticmethod
    def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
        """
        Evaluates the network on a validation set without manipulating network parameters.
        
        Purpose:
            To monitor the model's performance on unseen data for Early Stopping checks, 
            strictly preventing gradient tracking to save massive amounts of VRAM.
            
        Parameters:
            model (nn.Module): The active PyTorch neural network.
            loader (DataLoader): The validation data batch provider.
            criterion (nn.Module): The loss function.
            device (torch.device): The target compute hardware.
            
        Returns:
            float: The average validation loss.
        """
        model.eval() # Freezes Dropout layers and forces BatchNorm to use population statistics
        val_loss = 0.0
        
        # WHY: torch.no_grad() temporarily disables the Autograd engine. This reduces 
        # memory consumption by ~50% and accelerates execution, as PyTorch no longer 
        # stores the computational graph required for backpropagation.
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        return val_loss / max(1, len(loader))

    @staticmethod
    def evaluate_dl_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculates safe classification metrics for Deep Learning predictions.
        (Logic is identical to SVM evaluation, ensuring consistent metric reporting).
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_score = float('nan')

        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
            'F1_Score': f1_score(y_true, y_pred, zero_division=0),
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'AUROC': auc_score
        }