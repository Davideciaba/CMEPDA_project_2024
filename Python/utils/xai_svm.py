"""
Module: xai_svm.py

Encapsulates Explainable AI (XAI) algorithms for Linear Support Vector Machines 
in neuroimaging contexts. Implements both the Haufe Forward Transform and the 
Gaonkar Analytic Significance Maps for HDLSS (High-Dimension Low-Sample-Size) data,
including native Family-Wise Error (FWE) corrections.
"""
import numpy as np
import scipy.stats as stats
import nibabel as nib
from statsmodels.stats.multitest import multipletests
from typing import Tuple

from Python.utils.py_logger import CustomLogger

class SVMExplainer:
    """
    Computes biologically interpretable spatial patterns from backward SVM models.
    
    PURPOSE:
        Raw SVM weights act as noise suppressors rather than true activation indicators.
        This class implements advanced linear algebra techniques (Haufe, Gaonkar) to 
        invert the mapping and extract statistical significance maps that correlate 
        structurally with the disease pathology.
    """

    def __init__(self, logger: CustomLogger):
        """
        Initializes the SVMExplainer with a custom logger.
        
        Args:
            logger (CustomLogger): The unified project logger.
        """
        self.logger = logger

    def compute_haufe_patterns(self, X_train: np.ndarray, decision_scores: np.ndarray) -> np.ndarray:
        """
        Computes the Haufe Forward Transform: A = Cov(X, s).
        
        PURPOSE:
            Transforms backward SVM weights (noise suppressors) into true biological 
            activation patterns by computing the covariance between the input features 
            and the model's decision scores.
        
        Args:
            X_train (np.ndarray): Feature matrix of shape (m_samples, d_features).
            decision_scores (np.ndarray): SVM decision function outputs (m_samples,).
            
        Returns:
            np.ndarray: Haufe activation pattern of shape (d_features,).
        """
        self.logger.info("Computing Haufe Forward Transform (Covariance mapping)...")
        
        # Center the data and the scores
        X_centr = X_train - np.mean(X_train, axis=0)
        s_centr = decision_scores - np.mean(decision_scores)
            
        # A = (X_centr^T @ s_centr) / (m - 1)
        haufe_map = (X_centr.T @ s_centr) / (X_train.shape[0] - 1)
        
        self.logger.success("Haufe patterns successfully extracted.")
        return haufe_map

    def compute_gaonkar_maps(self, X_train: np.ndarray, y_train: np.ndarray, 
                             svm_weights: np.ndarray, C_param: float, n_support: int,
                             correction: str = 'bonferroni', alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Analytic Statistical Significance Maps for SVM.
        
        PURPOSE:
            Implements the methodology by Gaonkar & Davatzikos (2013). Computes p-values 
            analytically for High-Dimension Low-Sample-Size (HDLSS) regimes, avoiding 
            computationally expensive permutation testing. Includes multiple comparison corrections.
        
        Args:
            X_train (np.ndarray): Training feature matrix (m_samples, d_features).
            y_train (np.ndarray): Binary labels in {-1, 1} or {0, 1} format (m_samples,).
            svm_weights (np.ndarray): The optimized weight vector w* from the linear SVM (d_features,).
            C_param (float): The regularization parameter for the SVM.
            n_support (int): The total number of support vectors from the trained SVM.
            correction (str): The multiple testing correction method ('bonferroni', 'fdr_by').
            alpha (float): The significance level for the multiple testing correction.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - z_scores: The analytic Z-score for every voxel, thresholded at alpha.
                - pvals_corrected: The p-values after FWE/FDR correction.
        """
        self.logger.info("Computing Gaonkar Analytic Significance Maps for HDLSS regime...")
        
        m_samples, d_features = X_train.shape
        
        # Check dimensional constraints
        ratio = m_samples / d_features
        self.logger.debug(f"Gaonkar Sample-to-Feature Ratio (m/d): {ratio:.4f}")
        if ratio > 0.2:
            self.logger.warning(f"m/d ratio ({ratio:.4f}) exceeds 0.2. Gaonkar assumption might be unstable.")
        
        sv_ratio = n_support / m_samples
        if sv_ratio < 0.95:
            self.logger.warning(f"HDLSS Violation: Support Vectors are only {sv_ratio:.1%} of samples (Requires >= 95.0%).")
         
        # 1. Compute Gram Matrix (K = X @ X^T)
        # Added (1/C)*I to map standard dot product into the soft-margin SVM L2 dual space
        K = X_train @ X_train.T + (1.0 / C_param) * np.eye(m_samples)
        # Use pseudoinverse for strict numerical stability
        K_inv = np.linalg.pinv(K)
            
        # 2. Compute Intermediate Matrix C
        J = np.ones((m_samples, 1))
        JT_K_inv_J = (J.T @ K_inv @ J).item()  # Scalar value
                          
        # term2 = K^-1 * J * (-J^T * K^-1 * J)^-1 * J^T * K^-1
        scalar_inv = 1.0 / (-JT_K_inv_J)
        term2 = (K_inv @ J) * scalar_inv @ (J.T @ K_inv)
        
        P = K_inv + term2  # Shape: (m_samples, m_samples)
        
        # C = X^T @ P. Shape of C: (d_features, m_samples)
        C = X_train.T @ P
        
        # Ensure labels are binary and determine fraction 'p' of positive class
        unique_labels = np.unique(y_train)
        pos_class = unique_labels[1] if len(unique_labels) > 1 else 1
        p_frac = np.sum(y_train == pos_class) / m_samples
        
        # Sum of squared elements of C over the samples (axis=1)
        sum_C2 = np.sum(C**2, axis=1)
        sigma2 = (4 * p_frac - 4 * p_frac**2) * sum_C2  # Shape: (d_features,)
        
        E_wTw = np.sum(sigma2).item()  # Scalar value
           
        # 4. Compute Analytic Z-Scores
        wTw = (svm_weights.T @ svm_weights).item()  # Scalar value
        s_star = svm_weights / (wTw + 1e-15)  # Avoid division by zero
        
        var_s = sigma2 / (E_wTw**2)
        
        # z_j = s_j* / sqrt(var(s_j))
        z_scores = s_star / np.sqrt(var_s + 1e-15)
        
        # 5. Compute two-tailed p-values from standard normal distribution
        p_values = 2 * stats.norm.sf(np.abs(z_scores))
        
        # 7. Apply Multiple Comparisons Correction (FWE / Bonferroni)
        self.logger.info(f"Applying {correction.upper()} multiple comparisons correction at alpha={alpha}...")
        reject_null, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=correction)
        
        significant_voxels = np.sum(reject_null)
        self.logger.info(f"Gaonkar Correction ({correction.upper()}): {significant_voxels}/{d_features} voxels passed significance.")
        
        # We suppress all voxels that failed the statistical correction to 0.0
        z_scores_thresholded = np.zeros_like(z_scores)
        z_scores_thresholded[reject_null] = z_scores[reject_null]
        return z_scores_thresholded, pvals_corrected

    def reconstruct_and_save_3d(self, flat_map: np.ndarray, brain_mask: np.ndarray, 
                                 affine: np.ndarray, out_path: str) -> None:
        """
        Inflates a 1D flattened feature array back into a 3D NIfTI volume.
        
        PURPOSE:
            Takes the 1D vectors outputted by the SVM/XAI processes and accurately 
            places them back into their original spatial coordinates utilizing the 
            boolean reference mask.
        
        Args:
            flat_map (np.ndarray): The 1D feature array (d_features,).
            brain_mask (np.ndarray): Boolean 3D tensor of the valid brain space.
            affine (np.ndarray): 4x4 spatial affine matrix (MNI space).
            out_path (str): Full export path for the .nii file.
            
        Raises:
            ValueError: If the 1D map dimension mismatches the mask's active voxels.
            Exception: If disk I/O fails during saving.
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
        try:
            nifti_img = nib.Nifti1Image(vol_3d, affine)
            nib.save(nifti_img, out_path)
        except Exception as e:
            self.logger.error(f"Failed to save NIfTI volume: {e}")
            raise
        
        self.logger.success("3D XAI Map cleanly exported to disk.")