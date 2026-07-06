"""
Module: XAI_EfficientNet.py

This module implements the Explainable AI (XAI) algorithms for 3D CNNs (EfficientNet).
It leverages memory-optimized Integrated Gradients (IG) to extract voxel-wise 
feature importance. Includes Out-of-Fold (OOF) stitching and rigorous voxel-wise 
statistical inference (Shapiro-Wilk, T-Test, Mann-Whitney U, FDR Correction) 
as per Wang et al. (2023) methodology.
"""
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from monai.data import DataLoader
from typing import Dict, List, Tuple, Any
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

class DLExplainableAI:
    """
    Explainable AI Engine for PyTorch 3D Convolutional Neural Networks.
    """

    def __init__(self, logger: Any, device: torch.device):
        """Initializes the DL XAI engine via Dependency Injection."""
        self.logger = logger
        self.device = device

    def compute_integrated_gradients(self, model: nn.Module, input_tensor: torch.Tensor, 
                                     target_class: int, baseline: torch.Tensor = None, 
                                     steps: int = 50) -> torch.Tensor:
        """
        Computes Integrated Gradients for a 3D input tensor using Riemann sum approximation.
        
        Purpose:
            Accumulates gradients sequentially to prevent GPU Out-of-Memory (OOM) errors 
            which are typical when processing multiple 3D spatial interpolations simultaneously.
        
        Args:
            model (nn.Module): The trained PyTorch model.
            input_tensor (torch.Tensor): 5D tensor (B, C, D, H, W).
            target_class (int): The index of the class to explain (e.g., 1 for AD).
            baseline (torch.Tensor): The baseline image (zeros if None).
            steps (int): The number of integral approximation steps.
            
        Returns:
            torch.Tensor: The Integrated Gradients attribution map matching the input shape.
        """
        model.eval()
        if baseline is None:
            baseline = torch.zeros_like(input_tensor, device=self.device)

        integrated_gradients = torch.zeros_like(input_tensor, device=self.device)
        step_size = 1.0 / steps

        # WHY: Looping sequentially ensures VRAM consumption remains constant (O(1) w.r.t steps)
        for i in range(1, steps + 1):
            alpha = i * step_size
            
            # Linear interpolation between baseline and input
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.detach().requires_grad_()
            
            output = model(interpolated)
            
            # Extract scores for the target class and compute gradients
            score = output[:, target_class].sum()
            model.zero_grad()
            score.backward()
            
            integrated_gradients += interpolated.grad
            
        integrated_gradients *= step_size
        integrated_gradients *= (input_tensor - baseline)
        
        return integrated_gradients

    def generate_fold_xai(self, model: nn.Module, test_loader: DataLoader, target_class: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates zero-leakage IG maps strictly on the Out-of-Fold Test Set.
        
        Args:
            model (nn.Module): The trained CNN for the current fold.
            test_loader (DataLoader): MONAI Dataloader containing only hold-out subjects.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Array of IG maps [N, D, H, W], Array of True Labels [N])
        """
        self.logger.info("Extracting Integrated Gradients on Hold-out Fold...")
        model.eval()
        
        all_ig_maps = []
        all_labels = []
        
        with torch.enable_grad():  # Autograd must be enabled to compute IG
            for batch_data in test_loader:
                inputs = batch_data["image"].to(self.device, non_blocking=True)
                labels = batch_data["label"].numpy()
                
                ig_tensor = self.compute_integrated_gradients(model, inputs, target_class=target_class)
                
                # Move to CPU, convert to NumPy, and drop the channel dimension (B, 1, D, H, W -> B, D, H, W)
                ig_maps_np = ig_tensor.cpu().detach().numpy()[:, 0, ...]
                
                all_ig_maps.append(ig_maps_np)
                all_labels.extend(labels)
                
        self.logger.debug("Fold IG extraction completed.")
        return np.concatenate(all_ig_maps, axis=0), np.array(all_labels)

    def _benjamini_hochberg_fdr(self, p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Native NumPy implementation of False Discovery Rate correction."""
        valid_idx = np.where(~np.isnan(p_values))[0]
        valid_p = p_values[valid_idx]
        
        sorted_idx = np.argsort(valid_p)
        sorted_p = valid_p[sorted_idx]
        m = len(valid_p)
        
        critical_values = (np.arange(1, m + 1) / m) * alpha
        below_critical = sorted_p <= critical_values
        
        reject_null = np.zeros_like(p_values, dtype=bool)
        if np.any(below_critical):
            max_idx = np.max(np.where(below_critical)[0])
            fdr_threshold = sorted_p[max_idx]
            reject_null[valid_idx] = p_values[valid_idx] <= fdr_threshold
            
        return reject_null

    def compute_statistical_mask(self, all_ig_maps: np.ndarray, all_labels: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Executes voxel-wise statistical analysis to isolate significant biological pathways.
        
        Pipeline Compliance:
        1. Segregates IG maps into AD and CTRL cohorts.
        2. Filters dead voxels (zero variance).
        3. Evaluates Shapiro-Wilk normality per voxel.
        4. Routes to T-Test (normal) or Mann-Whitney U (non-parametric).
        5. Applies FDR alpha thresholding.
        """
        self.logger.info("Computing Voxel-wise Statistical Significance (T-Test / MWU + FDR)...")
        
        # Flatten spatial dimensions to (N_subjects, N_voxels)
        N, D, H, W = all_ig_maps.shape
        flat_maps = all_ig_maps.reshape(N, -1)
        
        ad_maps = flat_maps[all_labels == 1]
        ctrl_maps = flat_maps[all_labels == 0]
        
        num_voxels = flat_maps.shape[1]
        p_values = np.ones(num_voxels, dtype=np.float64)  # Default to 1.0 (Not significant)
        
        # WHY: Compute variance across all subjects to skip empty background space natively
        voxel_variances = np.var(flat_maps, axis=0)
        active_voxels = np.where(voxel_variances > 1e-8)[0]
        
        self.logger.debug(f"Processing {len(active_voxels)} active voxels out of {num_voxels}.")
        
        non_parametric_count = 0
        
        for idx in active_voxels:
            ad_vals = ad_maps[:, idx]
            ctrl_vals = ctrl_maps[:, idx]
            
            # Shapiro-Wilk Normality Check
            _, p_shapiro_ad = shapiro(ad_vals)
            _, p_shapiro_ctrl = shapiro(ctrl_vals)
            
            # Fallback to Mann-Whitney U if severe non-normality is detected (p < 0.05)
            if p_shapiro_ad < 0.05 or p_shapiro_ctrl < 0.05:
                non_parametric_count += 1
                _, p_val = mannwhitneyu(ad_vals, ctrl_vals, alternative='two-sided')
            else:
                _, p_val = ttest_ind(ad_vals, ctrl_vals, equal_var=False)
                
            p_values[idx] = p_val
            
        self.logger.info(f"Fallback triggered: Mann-Whitney U used on {non_parametric_count} voxels.")
        
        # FDR Correction
        significant_mask_flat = self._benjamini_hochberg_fdr(p_values, alpha=alpha)
        
        # Reconstruct spatial dimensions
        significant_mask_3d = significant_mask_flat.reshape(D, H, W)
        
        self.logger.success("DL Statistical Significance Mask generated.")
        return significant_mask_3d

    def reconstruct_nifti(self, map_3d: np.ndarray, affine: np.ndarray, header: Any, output_path: str) -> None:
        """Saves the fully reconstructed 3D statistical mask to disk."""
        reconstructed_img = nib.Nifti1Image(map_3d.astype(np.float32), affine, header=header)
        nib.save(reconstructed_img, output_path)
        self.logger.success(f"IG Statistical Map written to disk: {output_path}")