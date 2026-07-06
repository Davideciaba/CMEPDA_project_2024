"""
Module: XAI_SVM.py

This module implements the analytical Explainable AI (XAI) algorithms for Linear SVMs.
It contains the SVMAnalyticalXAI class, which provides high-efficiency implementations
of Haufe's Forward Transform and Gaonkar's Analytic Significance Mapping (p-maps),
strictly optimized to avoid RAM out-of-memory errors on massive 3D neuroimaging tensors.

Designed for a flat directory structure. It natively accepts models generated 
by the custom SVMPredictiveEngine.
"""
import os
import numpy as np
import nibabel as nib
from typing import List, Tuple, Any
from scipy.stats import norm

# Importazione locale (stessa cartella)
from py_logger import CustomLogger

class SVMAnalyticalXAI:
    """
    Analytical Explainable AI Engine for Linear Support Vector Machines.
    """

    def __init__(self, logger: CustomLogger):
    def __init__(self, logger: CustomLogger):
        """
        Initializes the SVM XAI engine via Dependency Injection.
        """
        self.logger = logger

    def compute_haufe_transform(self, model: Any, X_train: np.ndarray) -> np.ndarray:
        """
        Computes Haufe's forward activation pattern from backward SVM weights.
        
        MATHEMATICAL IMPLEMENTATION:
            Since standard SVM weights (W) represent a backward model, we compute 
            Activation Patterns A:
            1. SVM decision function: s_hat = X_train * W + b
            2. s_hat_centr = s_hat - mean(s_hat)
            3. X_centr = X_train - mean(X_train)
            4. A = Cov(X, s_hat) = 1 / (N - 1) * X_centr^T @ s_hat_centr
            
        Args:
            model (Any): The fitted model returned by SVMPredictiveEngine.train().
            X_train (np.ndarray): 2D feature matrix (N_samples, N_features).
            
        Returns:
            np.ndarray: Activation map A (1D array of size N_features).
        """
        self.logger.info("Computing Haufe Forward Transform...")
        
        N, V = X_train.shape
        if N <= 1:
            raise ValueError("Haufe transform requires a training set size N > 1 to compute covariance.")

        W = model.coef_[0]
        # Estrazione del bias per la decision function (se presente)
        b = model.intercept_[0] if hasattr(model, 'intercept_') and model.intercept_ is not None else 0.0

        # Step 1: s_hat = X_train W + b
        s_hat = X_train @ W + b
        
        # Step 2: Centering s_hat
        s_hat_centr = s_hat - np.mean(s_hat)

        # Step 3: Centering X_train
        X_mean = np.mean(X_train, axis=0)
        X_centr = X_train - X_mean

        # Step 4: A = Cov(X, s_hat)
        A = (1.0 / (N - 1)) * (X_centr.T @ s_hat_centr)

        self.logger.debug("Haufe forward activation map generated successfully.")
        return A

    def compute_gaonkar_maps(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, C: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analytically estimates voxel-wise Z-score and p-value maps under the null hypothesis.
        Includes High-Dimension Low-Sample-Size (HDLSS) regime checks.
        
        MATHEMATICAL IMPLEMENTATION (Gaonkar):
            1. J \in R^m : vector of ones
            2. K = X_train @ X_train^T
            3. H = K^{-1} + K^{-1} J ( -J^T K^{-1} J )^{-1} J^T K^{-1}
            4. C_matrix = X_train^T H
            5. sigma_j^2 = (4p - 4p^2) * \sum_{i=1}^m C_{j, i}^2
            6. E(w^T w) = \sum_k sigma_k^2
            7. s_j^* = w_j / (w^T w)
            8. var(s_j) = sigma_j^2 / E(w^T w)^2
            9. z_j = s_j^* / sqrt(var(s_j))
            10. p_values and FWE-correction.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (z_map, p_map_raw, p_map_fwe)
        """
        self.logger.info("Executing Gaonkar analytic significance mapping...")
        
        m, d = X_train.shape
        W = model.coef_[0]
        
        # --- QA: Gaonkar Assumptions Check ---
        ratio_md = m / d
        if ratio_md >= 0.2:
            self.logger.warning(f"Gaonkar Check Failed: Sample-to-Feature ratio (m/d) = {ratio_md:.3f} is >= 0.2.")
            
        if hasattr(model, 'support_'):
            sv_ratio = len(model.support_) / m
            if sv_ratio <= 0.95:
                self.logger.warning(f"Gaonkar Check Failed: Support Vectors percentage is {sv_ratio*100:.1f}% (Expected > 95%).")

        K = X_train @ X_train.T
        cond_K = np.linalg.cond(K)
        if cond_K >= 1e4:
            self.logger.warning(f"Gaonkar Check Failed: Gram matrix condition number is {cond_K:.2e} (Expected < 10^4).")
        
        # J \in R^m a vector with each component equal to 1
        J = np.ones((m, 1))
        
        # Using pseudo-inverse to handle singular matrices in HDLSS
        K_inv = np.linalg.pinv(K)
        
        # Term 2: K^{-1} J ( -J^T K^{-1} J )^{-1} J^T K^{-1}
        K_inv_J = K_inv @ J
        J_K_inv_J = J.T @ K_inv_J
        scalar_inv = 1.0 / (-J_K_inv_J[0, 0])
        
        # H matrix (m x m)
        H = K_inv + scalar_inv * (K_inv_J @ K_inv_J.T)
        
        # C_matrix = X^T H. Size: (d, m)
        C_matrix = X_train.T @ H
        
        # Fraction of positive class
        y_train_arr = np.array(y_train)
        classes = np.unique(y_train_arr)
        pos_class = classes[1] if len(classes) > 1 else 1
        p = np.sum(y_train_arr == pos_class) / m
        
        # \sigma_j^2 = (4p - 4p^2) \sum_{i=1}^m C_{j, i}^2
        sum_C2 = np.sum(C_matrix**2, axis=1)
        sigma2 = (4 * p - 4 * (p**2)) * sum_C2
        
        # E(w^T w) = \sum_k \sigma_k^2
        E_ww = np.sum(sigma2)
        
        # s_j^* = w_j / (w^T w)
        w_norm2 = np.dot(W, W)
        s_star = W / w_norm2
        
        # var(s_j) = \sigma_j^2 / E(w^T w)^2
        var_s = sigma2 / (E_ww**2)
        var_s = np.where(var_s <= 0, 1e-16, var_s)
        
        # z_j = s_j^* / \sqrt{var(s_j)}
        z_map = s_star / np.sqrt(var_s)
        
        # Compute the p-values and FWE-correction
        p_map_raw = 2 * norm.sf(np.abs(z_map))
        
        # Bonferroni FWE-correction
        p_map_fwe = np.minimum(p_map_raw * d, 1.0)

        self.logger.debug("Gaonkar mappings generated successfully.")
        return z_map, p_map_raw, p_map_fwe

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