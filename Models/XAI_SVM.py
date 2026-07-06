"""
Module: XAI_SVM.py

This module implements the analytical Explainable AI (XAI) algorithms for Linear SVMs.
It contains the SVMAnalyticalXAI class, which provides high-efficiency implementations
of Haufe's Forward Transform and Gaonkar's Analytic Significance Mapping (p-maps).
"""
import os
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Any
from sklearn.svm import SVC
from scipy.stats import norm

class SVMAnalyticalXAI:
    """
    Analytical Explainable AI Engine for Linear Support Vector Machines.
    """

    def __init__(self, logger: Any):
        """
        Initializes the SVM XAI engine via Dependency Injection.
        """
        self.logger = logger

    def compute_haufe_transform(self, model: SVC, X_train: np.ndarray) -> np.ndarray:
        """Computes Haufe's forward activation pattern from backward SVM weights."""
        self.logger.info("Computing Haufe Forward Transform via associative memory projection...")
        
        N, V = X_train.shape
        if N <= 1:
            raise ValueError("Haufe transform requires a training set size N > 1 to compute covariance.")

        W = model.coef_[0]

        X_mean = np.mean(X_train, axis=0)
        X_centered = X_train - X_mean

        # Trucco associativo per bypassare la matrice V x V
        signal_projection = X_centered @ W
        haufe_pattern = (1.0 / (N - 1)) * (X_centered.T @ signal_projection)

        self.logger.debug("Haufe forward activation map generated successfully.")
        return haufe_pattern

    def compute_gaonkar_maps(self, model: SVC, X_train: np.ndarray, y_train: np.ndarray, C: float) -> Tuple[np.ndarray, np.ndarray]:
        """Analytically estimates voxel-wise Z-score and p-value maps under the null hypothesis."""
        self.logger.info("Executing Gaonkar analytic significance mapping...")
        
        N, V = X_train.shape
        W = model.coef_[0]

        X_mean = np.mean(X_train, axis=0)
        X_centered = X_train - X_mean

        # Spazio dei campioni (N x N)
        Gram = X_centered @ X_centered.T
        M = Gram + (1.0 / C) * np.eye(N)
        
        H = np.linalg.inv(M)
        B = H @ H

        BX = B @ X_centered
        voxel_variance_projections = np.sum(X_centered * BX, axis=0)

        sigma_y_sq = np.var(y_train)
        var_W = sigma_y_sq * voxel_variance_projections
        var_W = np.where(var_W <= 0, 1e-8, var_W)

        z_map = W / np.sqrt(var_W)
        p_map = 2.0 * (1.0 - norm.cdf(np.abs(z_map)))

        self.logger.debug("Gaonkar statistical significance mappings generated successfully.")
        return z_map, p_map

    @staticmethod
    def aggregate_global_maps(maps_list: List[np.ndarray]) -> np.ndarray:
        """Performs element-wise arithmetic averaging over multiple out-of-fold XAI arrays."""
        if not maps_list:
            raise ValueError("The list of maps to aggregate cannot be empty.")
        return np.mean(np.array(maps_list), axis=0)

    def reconstruct_nifti(self, map_1d: np.ndarray, mask_path: str, output_path: str) -> None:
        """Maps a 1D masked feature array back into a 3D NIfTI grid using a spatial template."""
        self.logger.info(f"Reconstructing 1D map into 3D volume using mask template: {mask_path}")
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Template mask not found at path: {mask_path}")

        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask_bool = mask_data > 0

        active_voxel_count = np.sum(mask_bool)
        if len(map_1d) != active_voxel_count:
            raise ValueError(
                f"Dimension mismatch. 1D map length ({len(map_1d)}) "
                f"must equal mask active voxel count ({active_voxel_count})."
            )

        grid_3d = np.zeros(mask_data.shape, dtype=np.float32)
        grid_3d[mask_bool] = map_1d.astype(np.float32)

        reconstructed_img = nib.Nifti1Image(grid_3d, mask_img.affine, header=mask_img.header)
        nib.save(reconstructed_img, output_path)
        
        self.logger.success(f"3D NIfTI map written successfully to disk: {output_path}")