"""
Module: XAI_EfficientNet.py

This module implements the Explainable AI (XAI) algorithms for 3D CNNs (EfficientNet).
It leverages memory-optimized Integrated Gradients (IG) to extract voxel-wise 
feature importance, exactly aligned with the method provided in Wang et al.'s reference implementation.

Designed for a structured package layout. Strictly decoupled from specific model classes,
accepting any valid PyTorch model spawned by the CNNPredictiveEngine.
"""
import numpy as np
import nibabel as nib
import torch
from typing import Tuple, Any


from utils.py_logger import CustomLogger


class EfficientNetExplainer:
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
        elif baseline_name == "g":                                         # gaussian noise baseline
            baseline = input_tensor + torch.randn(input_size, device=self.device)
        else:
            raise ValueError("Invalid baseline_name. Choose 'z' or 'g'.")

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
        diff = (input_tensor - baseline).detach().squeeze().cpu().numpy()
        av_grad_np = av_grad.detach().squeeze(0).cpu().numpy()
        
        # Final IG calculation
        i_grads = diff * av_grad_np
        
        return i_grads

    def remove_symmetric_padding(self, padded_array: np.ndarray, original_shape: Tuple[int, int, int] = (121, 145, 121)) -> np.ndarray:
        """
        Mathematically reverses MONAI's SpatialPadd(method='symmetric').
        Crops the padded 3D array back to its original physical dimensions.
        
        Args:
            padded_array (np.ndarray): The IG map with shape (160, 160, 160)
            original_shape (Tuple): The target MNI bounding box (121, 145, 121)
            
        Returns:
            np.ndarray: The cropped volume matching original MNI space.
        """
        pad_x = padded_array.shape[0] - original_shape[0]
        pad_y = padded_array.shape[1] - original_shape[1]
        pad_z = padded_array.shape[2] - original_shape[2]

        # Calculate symmetric offsets (Floor division matches MONAI's default behavior)
        start_x = pad_x // 2
        start_y = pad_y // 2
        start_z = pad_z // 2

        end_x = start_x + original_shape[0]
        end_y = start_y + original_shape[1]
        end_z = start_z + original_shape[2]

        # Perform surgical 3D crop
        cropped_array = padded_array[start_x:end_x, start_y:end_y, start_z:end_z]
        return cropped_array

    def reconstruct_nifti(self, map_3d: np.ndarray, affine: np.ndarray, output_path: str) -> None:
        """Saves the fully reconstructed 3D feature map to disk."""
        self.logger.info("Reconstructing 3D volume mapping...")
        try:
            nifti_img = nib.Nifti1Image(map_3d.astype(np.float32), affine)
            nib.save(nifti_img, output_path)
            self.logger.success(f"NIfTI saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save NIfTI volume: {e}")
            raise
        self.logger.success(f"IG Map written to disk: {output_path}")