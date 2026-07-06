"""
Module: XAI_EfficientNet.py

This module implements the Explainable AI (XAI) algorithms for 3D CNNs (EfficientNet).
It leverages memory-optimized Integrated Gradients (IG) to extract voxel-wise 
feature importance, exactly aligned with the method provided in Wang et al.'s reference implementation.

Designed for a structured package layout. Strictly decoupled from specific model classes,
accepting any valid PyTorch model spawned by the CNNPredictiveEngine.
"""
import os
import numpy as np
import nibabel as nib
import torch
from typing import Tuple, Any, List

# --- STRICT TYPE HINTING ---
# Importiamo esplicitamente la classe CustomLogger dal modulo utils 
# per imporla come contratto nell'__init__
from utils.py_logger import CustomLogger


class DLExplainableAI:
    """
    Explainable AI Engine for PyTorch 3D Convolutional Neural Networks.
    """

    def __init__(self, logger: CustomLogger, device: torch.device):
        """
        Initializes the DL XAI engine via Dependency Injection.
        Strict typing enforces the use of the project's native CustomLogger.
        
        Args:
            logger (CustomLogger): The unified project logger.
            device (torch.device): Hardware accelerator mapping.
        """
        self.logger = logger
        self.device = device

    def compute_integrated_gradients(self, model: Any, input_tensor: torch.Tensor, 
                                     target_class: int, baseline_name: str = "z", 
                                     steps: int = 100) -> np.ndarray:
        """
        Computes Integrated Gradients for a 3D input tensor using Riemann sum approximation.
        
        Purpose:
            Extracts biologically relevant activation patterns from non-linear CNNs.
            Satisfies the Axiom of Completeness.
            
        Reference (Wang et al. / Sundararajan et al., 2017):
            Implementation strictly aligned with the provided `ig.py`. 
        """
        model.eval()
        input_size = input_tensor.shape
        
        # PIPELINE COMMENT: Baseline selection mimicking ig.py
        if baseline_name == "z":                                           # zero baseline
            baseline = torch.zeros(input_size, device=self.device)
        elif baseline_name == "u":                                         # uniform noise baseline
            baseline = torch.rand(input_size, device=self.device)                    
        elif baseline_name == "g":                                         # gaussian noise baseline
            baseline = input_tensor + torch.randn(input_size, device=self.device)
        else:
            raise ValueError("Invalid baseline_name. Choose 'z', 'u', or 'g'.")

        # MATHEMATICAL COMMENT (Riemann Sum Approximation & VRAM Optimization):
        # Equation: IG(x) = (x - x') * Integral[alpha=0 to 1] gradients(x' + alpha*(x - x')) d_alpha
        # To avoid severe Out-Of-Memory (OOM) on large 3D medical volumes caused by stacking
        # all images at once, we compute the gradient one alpha step at a time in a loop.
        # This keeps VRAM usage at O(1) w.r.t 'steps'.
        
        step_size = 1.0 / steps
        all_grad_sum = torch.zeros_like(input_tensor, device=self.device)
        
        # Calculate gradients for i in range(0, steps) as per the av_grad[:-1] logic
        for i in range(steps):
            alpha = i * step_size
            
            # Linear Interpolation
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.detach().requires_grad_()
            
            out = model(interpolated)
            
            # Extract scores for the target class and backpropagate
            target_out = out[:, target_class]
            model.zero_grad()
            target_out.backward()
            
            # Accumulate gradient
            all_grad_sum += interpolated.grad

        # Average image gradient
        av_grad = all_grad_sum / steps
        
        # Diff multiplier: (input_ - baseline)
        diff = (input_tensor - baseline).detach().squeeze(0).cpu().numpy()
        av_grad_np = av_grad.detach().squeeze(0).cpu().numpy()
        
        # Final IG calculation
        i_grads = diff * av_grad_np
        
        return i_grads

    @staticmethod
    def aggregate_global_maps(maps_list: List[np.ndarray]) -> np.ndarray:
        """
        Performs element-wise arithmetic averaging over multiple out-of-fold IG arrays.
        """
        if not maps_list:
            raise ValueError("The list of maps to aggregate cannot be empty.")
        return np.mean(np.array(maps_list), axis=0)

    def reconstruct_nifti(self, map_3d: np.ndarray, affine: np.ndarray, header: Any, output_path: str) -> None:
        """Saves the fully reconstructed 3D feature map to disk."""
        self.logger.info("Reconstructing 3D volume mapping...")
        reconstructed_img = nib.Nifti1Image(map_3d.astype(np.float32), affine, header=header)
        nib.save(reconstructed_img, output_path)
        self.logger.success(f"IG Map written to disk: {output_path}")