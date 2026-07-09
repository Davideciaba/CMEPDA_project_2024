"""
Module: XAI_SVM.py

Encapsulates Explainable AI (XAI) algorithms for Linear Support Vector Machines 
in neuroimaging contexts. Implements both the Haufe Forward Transform and the 
Gaonkar Analytic Significance Maps for HDLSS (High-Dimension Low-Sample-Size) data,
including native Family-Wise Error (FWE) corrections.
"""
import os
import numpy as np
import scipy.stats as stats
import nibabel as nib
from statsmodels.stats.multitest import multipletests
from typing import Tuple, Optional

from utils.py_logger import CustomLogger

class SVMExplainer:
    """
    Computes biologically interpretable spatial patterns from backward SVM models.
    """

    def __init__(self, logger: CustomLogger):
        """Initializes the SVMExplainer with a custom logger."""
        self.logger = logger

    def compute_haufe_patterns(self, X_train: np.ndarray, decision_scores: np.ndarray) -> np.ndarray:
        """
        Computes the Haufe Forward Transform: A = Cov(X, s).
        Transforms backward SVM weights (noise suppressors) into true biological activation patterns.
        
        Args:
            X_train: Feature matrix of shape (m_samples, d_features).
            decision_scores: SVM decision function outputs (m_samples,).
            
        Returns:
            np.ndarray: Haufe activation pattern of shape (d_features,).
        """
        self.logger.info("Computing Haufe Forward Transform (Covariance mapping)...")
        
        # Center the data and the scores
        X_centr = X_train - np.mean(X_train, axis=0)
        s_centr = decision_scores - np.mean(decision_scores)
        
        m_samples = X_train.shape[0]
        if m_samples <= 1:
            self.logger.error("Insufficient samples to compute covariance.")
            raise ValueError("m_samples must be > 1 for Haufe transform.")
            
        # A = (X_centr^T @ s_centr) / (m - 1)
        haufe_map = (X_centr.T @ s_centr) / (m_samples - 1)
        
        self.logger.success("Haufe patterns successfully extracted.")
        return haufe_map

    def compute_gaonkar_maps(self, X_train: np.ndarray, y_train: np.ndarray, 
                             svm_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Analytic Statistical Significance Maps for SVM according to 
        Gaonkar & Davatzikos (2013). Avoids computationally expensive permutation tests.
        
        Args:
            X_train: Training feature matrix (m_samples, d_features).
            y_train: Binary labels in {-1, 1} or {0, 1} format (m_samples,).
            svm_weights: The optimized weight vector w* from the linear SVM (d_features,).
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Continuous Z-scores map and raw p-values map.
        """
        self.logger.info("Computing Gaonkar Analytic Significance Maps for HDLSS regime...")
        
        m_samples, d_features = X_train.shape
        
        # Check dimensional constraints
        ratio = m_samples / d_features
        self.logger.debug(f"Gaonkar Sample-to-Feature Ratio (m/d): {ratio:.4f}")
        if ratio > 0.2:
            self.logger.warning(f"m/d ratio ({ratio:.4f}) exceeds 0.2. Gaonkar assumption might be unstable.")
            
        # Ensure labels are binary and determine fraction 'p' of positive class
        unique_labels = np.unique(y_train)
        pos_class = unique_labels[1] if len(unique_labels) > 1 else 1
        p_frac = np.sum(y_train == pos_class) / m_samples
        
        # 1. Compute Gram Matrix (K = X @ X^T)
        K = X_train @ X_train.T
        cond_num = np.linalg.cond(K)
        
        self.logger.debug(f"Gram Matrix Condition Number: {cond_num:.2e}")
        
        # Safe Inversion (Pseudo-inverse if highly ill-conditioned)
        if cond_num > 1e4:
            self.logger.warning("Gram Matrix is ill-conditioned (cond > 10^4). Using Moore-Penrose Pseudo-Inverse.")
            K_inv = np.linalg.pinv(K)
        else:
            K_inv = np.linalg.inv(K)
            
        # 2. Compute Intermediate Matrix C
        J = np.ones((m_samples, 1))
        J_T_K_inv_J = float(J.T @ K_inv @ J)
        
        if J_T_K_inv_J == 0:
            raise ValueError("Math Error: J^T * K^-1 * J is zero. Cannot invert.")
            
        # term2 = K^-1 * J * (-J^T * K^-1 * J)^-1 * J^T * K^-1
        scalar_inv = 1.0 / (-J_T_K_inv_J)
        term2 = (K_inv @ J) * scalar_inv @ (J.T @ K_inv)
        
        P = K_inv + term2  # Shape: (m_samples, m_samples)
        
        # C = X^T @ P. Shape of C: (d_features, m_samples)
        C = X_train.T @ P
        
        # 3. Compute Variances
        # Sum of squared elements of C over the samples (axis=1)
        sum_C2 = np.sum(C**2, axis=1)
        sigma2 = (4 * p_frac - 4 * p_frac**2) * sum_C2  # Shape: (d_features,)
        
        E_wTw = float(np.sum(sigma2))
        
        if E_wTw <= 0:
            self.logger.error("Expected E(w^Tw) is zero or negative. Math instability detected.")
            raise ValueError("E(w^Tw) computation failed.")
            
        # 4. Compute Analytic Z-Scores
        wTw = float(svm_weights.T @ svm_weights)
        s_star = svm_weights / (wTw + 1e-15)
        
        var_s = sigma2 / (E_wTw**2)
        
        # z_j = s_j* / sqrt(var(s_j))
        z_scores = s_star / np.sqrt(var_s + 1e-15)
        
        # 5. Compute two-tailed p-values from standard normal distribution
        p_values = 2 * stats.norm.sf(np.abs(z_scores))
        
        self.logger.success("Gaonkar Z-maps and p-values successfully evaluated.")
        return z_scores, p_values

    def apply_gaonkar_fwe_correction(self, z_scores: np.ndarray, p_values: np.ndarray, 
                                     alpha: float = 0.05) -> np.ndarray:
        """
        Applies Family-Wise Error (FWE) correction using the Bonferroni method.
        Forces non-significant voxels mathematically to zero.
        
        Args:
            z_scores: Raw 1D array of Gaonkar Z-scores (d_features,).
            p_values: Raw 1D array of p-values (d_features,).
            alpha: Significance level (default 0.05).
            
        Returns:
            np.ndarray: FWE-corrected Z-scores map (d_features,).
        """
        self.logger.info(f"Applying FWE (Bonferroni) topological correction at alpha={alpha}...")
        
        # multiple comparisons correction via statsmodels
        reject_mask, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
        
        survived_voxels = np.sum(reject_mask)
        self.logger.info(f"FWE Correction complete: {survived_voxels} voxels survived the threshold.")
        
        if survived_voxels == 0:
            self.logger.warning("Zero voxels survived FWE correction. Consider inspecting uncorrected maps.")
            
        fwe_z_scores = np.zeros_like(z_scores, dtype=float)
        
        # Keep original Z-score only if the null hypothesis was rejected
        fwe_z_scores[reject_mask] = z_scores[reject_mask]
        
        return fwe_z_scores

    def reconstruct_and_save_3d(self, flat_map: np.ndarray, brain_mask: np.ndarray, 
                                affine: np.ndarray, out_path: str) -> None:
        """
        Inflates a 1D flattened feature array back into a 3D NIfTI volume using 
        the reference boolean brain mask.
        
        Args:
            flat_map: The 1D feature array (d_features,).
            brain_mask: Boolean 3D tensor of the valid brain space.
            affine: 4x4 spatial affine matrix (MNI space).
            out_path: Full export path for the .nii.gz file.
        """
        self.logger.info(f"Reconstructing 3D tensor to save at: {out_path}")
        
        # Verify sizes
        expected_features = int(np.sum(brain_mask))
        if flat_map.shape[0] != expected_features:
            self.logger.error(f"Shape mismatch: Mask has {expected_features} active voxels, flat_map has {flat_map.shape[0]}.")
            raise ValueError("1D map does not align with the 3D boolean mask.")
            
        # Inflate back to 3D background
        vol_3d = np.zeros_like(brain_mask, dtype=np.float32)
        vol_3d[brain_mask] = flat_map.astype(np.float32)
        
        # Create and save NIfTI
        img = nib.Nifti1Image(vol_3d, affine)
        
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        nib.save(img, out_path)
        
        self.logger.success("3D XAI Map cleanly exported to disk.")