"""
Module: XAI_SVM.py

This module implements the analytical Explainable AI (XAI) algorithms for Linear SVMs.
It contains the SVMAnalyticalXAI class, which provides high-efficiency implementations
of Haufe's Forward Transform and Gaonkar's Analytic Significance Mapping (p-maps),
strictly optimized to avoid RAM out-of-memory errors on massive 3D neuroimaging tensors.
"""
import os
import numpy as np
import nibabel as nib
from typing import Tuple, Any, List
from scipy.stats import norm

from utils.py_logger import CustomLogger


class SVMAnalyticalXAI:
    """
    Analytical Explainable AI Engine for Linear Support Vector Machines.
    """

    def __init__(self, logger: CustomLogger):
        """
        Initializes the SVM XAI engine via Dependency Injection.
        Strict typing enforces the use of the project's native CustomLogger.
        """
        self.logger = logger

    def compute_haufe_transform(self, model: Any, X_train: np.ndarray) -> np.ndarray:
        """
        Computes Haufe's forward activation pattern from backward SVM weights.
        
        MATHEMATICAL PIPELINE MAPPING:
            Since standard SVM weights (W) represent a backward model, we compute 
            Activation Patterns A:
            1. SVM decision function: s_hat = X_train * W + b
            2. s_hat_centr = s_hat - mean(s_hat)
            3. X_centr = X_train - mean(X_train)
            4. A = Cov(X_train, s_hat) = 1/(N-1) * sum(X_centr^T * s_hat_centr)
        """
        self.logger.info("Computing Haufe Forward Transform...")
        
        N, V = X_train.shape
        if N <= 1:
            raise ValueError("Haufe transform requires a training set size N > 1 to compute covariance.")

        W = model.coef_[0]
        # Estrazione del bias per la decision function (se presente)
        b = model.intercept_[0] if hasattr(model, 'intercept_') and model.intercept_ is not None else 0.0

        # Step 1: SVM decision function: \hat{s} = X_train W + b
        s_hat = X_train @ W + b
        
        # Step 2: Centering \hat{s}_{centr, n} = \hat{s}_n - \overline{s}_n
        s_hat_centr = s_hat - np.mean(s_hat)

        # Step 3: Centering X_{centr, n} = X_{train, n} - \overline{X_{train}}
        X_mean = np.mean(X_train, axis=0)
        X_centr = X_train - X_mean

        # Step 4: A = Cov(X, \hat{s}) = 1/(N_train - 1) * X_centr^T \hat{s}_centr
        A = (1.0 / (N - 1)) * (X_centr.T @ s_hat_centr)

        self.logger.debug("Haufe forward activation map generated successfully.")
        return A

    def compute_gaonkar_maps(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, C: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analytically estimates voxel-wise Z-score and p-value maps under the null hypothesis.
        
        MATHEMATICAL PIPELINE MAPPING (Gaonkar):
            - J \in R^m : a vector with each component equal to 1.
            - K = (XX^T)^{-1}
            - C = X^T [ K + K*J(-J^T K J)^{-1}J^T*K ]
            - \sigma_j^2 = (4p - 4p^2) \sum_i C_{i,j}^2
            - E(w^T w) = \sum_k \sigma_k^2
            - s_j^* = w_j^* / (w^{*T} w^*)
            - var(s_j) = \sigma_j^2 / E(w^T w)^2
            - z_j = s_j^* / \sqrt{var(s_j)}
            - p_values[j] = 2 * scipy.stats.norm.sf(np.abs(z_j)) + FWE-correction
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (z_map, p_map_raw, p_map_fwe)
        """
        self.logger.info("Executing Gaonkar analytic significance mapping...")
        
        m, d = X_train.shape
        W = model.coef_[0]

        # K_matrix = XX^T
        K_matrix = X_train @ X_train.T
        
        # J \in R^m a vector with each component equal to 1
        J = np.ones((m, 1))
        
        # K = (XX^T)^{-1}. Usiamo la Pseudo-Inversa per garantire stabilità numerica nei regimi HDLSS
        K_inv = np.linalg.pinv(K_matrix)
        
        # Calcolo dei sub-termini per ottimizzare le operazioni matriciali
        # K_inv_J = (XX^T)^{-1} J
        K_inv_J = K_inv @ J
        
        # J_K_inv_J = J^T (XX^T)^{-1} J
        J_K_inv_J = J.T @ K_inv_J
        
        # Inversa dello scalare (-J^T (XX^T)^{-1} J)^{-1}
        scalar_inv = 1.0 / (-J_K_inv_J[0, 0])
        
        # Costruzione della matrice interna H (m x m)
        # H = (XX^T)^{-1} + (XX^T)^{-1} J ( -J^T (XX^T)^{-1} J )^{-1} J^T (XX^T)^{-1}
        H = K_inv + scalar_inv * (K_inv_J @ K_inv_J.T)
        
        # C = X^T H. Size: (d, m) -> C_{j, i} rappresenta la feature j e il sample i
        C_matrix = X_train.T @ H
        
        # Calcolo del parametro p (frazione dei campioni etichettata come +1)
        y_train_arr = np.array(y_train)
        classes = np.unique(y_train_arr)
        pos_class = classes[1] if len(classes) > 1 else 1
        p = np.sum(y_train_arr == pos_class) / m
        
        # \sigma_j^2 = (4p - 4p^2) \sum_i C_{i,j}^2 
        # (np.sum lungo l'asse 1 somma i quadrati sui campioni per ogni voxel j)
        sum_C2 = np.sum(C_matrix**2, axis=1)
        sigma2 = (4 * p - 4 * (p**2)) * sum_C2
        
        # E(w^T w) = \sum_k \sigma_k^2
        E_ww = np.sum(sigma2)
        
        # s_j^* = w_j^* / (w^{*T} w^*)
        w_norm2 = np.dot(W, W)
        s_star = W / w_norm2
        
        # var(s_j) = \sigma_j^2 / E(w^T w)^2
        var_s = sigma2 / (E_ww**2)
        var_s = np.where(var_s <= 0, 1e-16, var_s)  # Prevenzione divisione per zero
        
        # z_j = s_j^* / \sqrt{var(s_j)}
        z_map = s_star / np.sqrt(var_s)
        
        # p_values[j] = 2 * scipy.stats.norm.sf(np.abs(z_j))
        p_map_raw = 2 * norm.sf(np.abs(z_map))
        
        # FWE-correction (Metodo Bonferroni: moltiplicazione per il numero di feature 'd')
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

        if len(map_1d) != np.sum(mask_bool):
            raise ValueError("Dimension mismatch between 1D map and mask active voxels.")

        grid_3d = np.zeros(mask_data.shape, dtype=np.float32)
        grid_3d[mask_bool] = map_1d.astype(np.float32)

        reconstructed_img = nib.Nifti1Image(grid_3d, mask_img.affine, header=mask_img.header)
        nib.save(reconstructed_img, output_path)
        
        self.logger.success(f"3D NIfTI map written successfully to disk: {output_path}")