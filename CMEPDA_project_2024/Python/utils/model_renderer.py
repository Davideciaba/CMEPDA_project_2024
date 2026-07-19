"""
Module: model_renderer.py

Contains ModelRenderer class, analogous to MATLAB's BrainRenderer.m.
It isolates all plotting libraries (matplotlib, seaborn) from the core mathematical engines
"""
import math
import pathlib
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.metrics import auc
from typing import List, Dict, Any, Tuple
from utils.py_logger import CustomLogger

class ModelRenderer:
    """
    Handles graphics visualization and exporting for Python models.
    
    PURPOSE:
        Acts as the central engine for all visualizations (ROC Curve,
        3D Brain Overlays, and XAI Analytics).
    """

    def __init__(self, logger: CustomLogger, output_dir: str):
        """
        Initializes the ModelRenderer object.
        
        Args:
            logger (CustomLogger): Centralized logging instance.
            output_dir (str): Base directory path to save the rendered plots.
        """
        self.logger = logger
        self.output_dir = pathlib.Path(output_dir).resolve()

    def plot_roc_curves(self, fold_artifacts: List[Dict[str, Any]], model_name: str, filename: str) -> None:
        """
        Plots the Receiver Operating Characteristic (ROC) curve for each fold.
        
        PURPOSE:
            Aggregates cross-validation performance. Plots the Mean ROC curve and its 
            ±1 Standard Deviation variance band. AUC scores are formatted as percentages.
            
        Args:
            fold_artifacts (List[Dict]): The artifacts list returned by execute_nested_cv.
            model_name (str): String label for the plot (e.g., "Linear SVM").
            filename (str): Output filename (e.g., "SVM_ROC_Curve.png").
        """
        self.logger.info(f"Rendering Nested CV ROC Curves for {model_name}...")
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        
        # Plot individual fold curves
        for artifact in fold_artifacts:
            fold_id = artifact['fold_id']
            fpr, tpr = artifact['roc_fpr'], artifact['roc_tpr']
            
            # Interpolate TPRs to a common FPR scale to compute the mean curve later
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            
            fold_auc = auc(fpr, tpr)
            aucs.append(fold_auc)
            ax.plot(fpr, tpr, lw=1.5, alpha=0.3, label=f"Fold {fold_id} (AUC = {fold_auc*100:.1f}%)")
            
        # Plot Mean ROC Curve
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(
            mean_fpr, mean_tpr, color='b',
            label=rf"Mean ROC (AUC = {mean_auc * 100:.1f}% $\pm$ {std_auc * 100:.1f}%)",
            lw=2.5, alpha=0.9
        )
        
        # Plot Standard Deviation Variance Band
        std_tpr = np.std(tprs, axis=0)
        # Prevent the standard deviation band from exceeding mathematical limits [0, 1]
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
            label=r"$\pm$ 1 Standard Deviation"
        )
        
        # Plot Random Chance Line
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Chance', alpha=0.8)
        
        # Formatting
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'Receiver Operating Characteristic - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        out_path = self.output_dir / filename
        
        try:
            fig.tight_layout()
            fig.savefig(out_path)
            self.logger.success(f"ROC Curve successfully rendered and saved at: {out_path.name}")
        except Exception as e:
            self.logger.error(f"Failed to save ROC Curve image: {e}")
        finally:
            plt.close(fig)  # Free RAM

    def _get_voxel_indices_from_mni(self, affine_mat: np.ndarray, slice_config: Any, tensor_size: Tuple[int, int, int], active_mask: np.ndarray) -> Tuple[List[int], List[float]]:
        """
        Translates a configuration into valid physical Z-axis array coordinates.
        
        PURPOSE:
            Solves the linear equation system derived from the NIfTI affine matrix 
            to map real-world millimeters into Python array slicing indices.
            
        Args:
            affine_mat (np.ndarray): 4x4 spatial affine transformation matrix.
            slice_config (Any): Scalar (step), [start, step, stop], or explicit MNI list.
            tensor_size (Tuple): Dimensions of the 3D volume.
            active_mask (np.ndarray): Boolean mask of active voxels.
            
        Returns:
            Tuple[List[int], List[float]]: Array indices for Z-axis, and their true MNI values.
        """
        # Extract Affine parameters for Z-axis
        vox_center_xy = [round(tensor_size[0] / 2), round(tensor_size[1] / 2)]
        max_z = tensor_size[2] - 1

        # Resolve Target MNI Coordinates
        if isinstance(slice_config, (int, float)):
            step_mm = float(slice_config)
            
            # Find active bounding box along Z-axi
            active_slices = np.any(active_mask, axis=(0, 1))
            active_idx = np.where(active_slices > 0)[0]
            
            if len(active_idx) == 0:
                self.logger.warning("No active voxels found in the mask.")
                return [], []

            # Matrix multiplication to extract exact MNI boundaries
            vox_bounds = np.array([
                [vox_center_xy[0], vox_center_xy[0]],
                [vox_center_xy[1], vox_center_xy[1]],
                [active_idx[0], active_idx[-1]],
                [1, 1]
            ])

            mni_bounds = affine_mat @ vox_bounds
            mni_min = np.min(mni_bounds[2, :])
            mni_max = np.max(mni_bounds[2, :])
            
            # Pad the viewing box by 1 step before and after the active region
            mni_min -= step_mm
            mni_max += step_mm

            # Fix the min to a multiple of the step (relative to Z=0)
            aligned_min = np.floor(mni_min / step_mm) * step_mm
            
            # Generate range accounting for affine scaling direction
            if mni_min < mni_max:
                mni_array = np.arange(aligned_min, mni_max + (step_mm * 0.1), step_mm)
            else:
                # Fallback if step should be negative
                mni_array = np.arange(aligned_min, mni_max - (step_mm * 0.1), -step_mm)

        elif isinstance(slice_config, list) and len(slice_config) == 3:
            # Interpreted as MATLAB's [start : step : stop]
            start_mm, step_mm, stop_mm = slice_config

            if (start_mm < stop_mm and step_mm < 0) or (start_mm > stop_mm and step_mm > 0):
                step_mm = -step_mm

            mni_array = np.arange(start_mm, stop_mm + (step_mm * 0.1), step_mm)
            
        elif isinstance(slice_config, (list, tuple, np.ndarray)):
            # Interpreted as specific discrete slices
            mni_array = np.array([float(x) for x in slice_config])
        else:
            self.logger.error("Invalid slice_config format provided.")
            raise ValueError("slice_config must be a scalar, [start, step, stop], or a list of MNI coordinates.")

        if len(mni_array) == 0:
            return [], []
        
        # Convert MNI coordinates back into matrix voxel indices
        num_mni = len(mni_array)
        mni_slices_mat = np.array([
            np.zeros(num_mni),
            np.zeros(num_mni),
            mni_array,
            np.ones(num_mni)
        ])

        # np.linalg.solve is equivalent to MATLAB's "\"
        vox_coords_mat = np.linalg.solve(affine_mat, mni_slices_mat)
        z_slices_voxel = np.round(vox_coords_mat[2, :]).astype(int)
                
        # Remove any indices outside the physical matrix volume
        valid_logical = (z_slices_voxel >= 0) & (z_slices_voxel <= max_z)
        
        z_slices_voxel = np.unique(z_slices_voxel[valid_logical])
        
        # Recalculate the physical MNI dimension for the validated and sorted voxels
        num_slices = len(z_slices_voxel)
        if num_slices == 0:
            self.logger.error("No valid voxel indices found after MNI to voxel conversion.")
            return [], []
            
        true_slices_mat = np.array([
            np.full(num_slices, vox_center_xy[0]),
            np.full(num_slices, vox_center_xy[1]),
            z_slices_voxel,
            np.ones(num_slices)
        ])
        
        true_mni_coords = affine_mat @ true_slices_mat
        z_mm_array = true_mni_coords[2, :]
                
        return z_slices_voxel.tolist(), z_mm_array.tolist()

    def plot_3d_activation_map(self, bg_nifti_path: str, stats_nifti_path: str, mask_nifti_path: str,
                               map_title: str, export_filename: str, threshold: float = 1e-15,
                               slice_config: Any = [-33.0, 3.0, 6.0]) -> None:
        """
        Extracts and plots 2D Axial slices from 3D NIfTI volumes.
        
        PURPOSE:
            Overlays 'hot' colormap statistical maps on top of grayscale anatomical 
            backgrounds, injecting transparency based on significance.
            
        Args:
            bg_nifti_path (str): Path to anatomical background volume (e.g. CTRL Subject).
            stats_nifti_path (str): Path to the XAI statistical map.
            mask_nifti_path (str): Path to the Brain Mask.
            map_title (str): Title for the generated figure.
            export_filename (str): Name of the file to save in the output_dir.
            threshold (float): Minimum absolute threshold for a voxel to be rendered.
            slice_config (Any): MNI slicing strategy configuration.
        """
        # Load all NIfTI volumes for rendering
        try:
            bg_img = nib.load(bg_nifti_path)
            stats_img = nib.load(stats_nifti_path)
            mask_img = nib.load(mask_nifti_path)
            
            bg_data = bg_img.get_fdata()
            stats_data = stats_img.get_fdata()
            mask_data = mask_img.get_fdata() > 0
            affine = bg_img.affine
        except IOError as e:
            self.logger.error(f"Failed to load NIfTI volumes for rendering: {e}")
            raise
        
        active_stat_mask = (np.abs(stats_data) >= threshold) & mask_data
        
        # Translate MNI to Voxels
        vox_indices, valid_mnis = self._get_voxel_indices_from_mni(affine, slice_config, stats_data.shape, active_stat_mask)
        
        if not vox_indices:
            self.logger.error(f"No valid brain slices found for config: {slice_config}. Bypassing render.")
            return
            
        self.logger.debug(f"Rendering {map_title} | Z-Axis Computed Slices: {len(vox_indices)}")
            
        # Matrix Layout Algorithm
        num_slices = len(vox_indices)
        cols = math.ceil(math.sqrt(num_slices + 1))
        rows = math.ceil((num_slices + 1) / cols)
        
        fig = plt.figure(figsize=(cols * 3, rows * 3), dpi=150)
        fig.patch.set_facecolor('black')
        
        # Color logic
        vmax = np.max(np.abs(stats_data[mask_data]))
        if vmax == 0: vmax = 1.0
        if vmax <= threshold:
            self.logger.warning(f"No voxels exceed the threshold ({threshold}). Aborting.")
            return
            
        norm = Normalize(vmin=-vmax, vmax=vmax, clip=True)
        cmap = plt.cm.coolwarm
        
        for idx, (vox_idx, target_mni) in enumerate(zip(vox_indices, valid_mnis)):
            # Explicitly create and attach the graphical axis for this specific slice
            ax = fig.add_subplot(rows, cols, idx + 1)
            
            # Axial slice extraction
            slice_bg = bg_data[:, :, vox_idx]
            slice_stats = stats_data[:, :, vox_idx]
                
            # Rotation
            slice_bg = np.rot90(slice_bg, k=1)
            slice_stats = np.rot90(slice_stats, k=1)
            
            # Plot Background (Grayscale)
            ax.imshow(slice_bg, cmap='gray')
            
            # Plot Overlay (Hot colormap)
            overlay_colors = cmap(norm(slice_stats))
            # Alpha mask: only show voxels exceeding the mathematical threshold
            alpha_layer = (np.abs(slice_stats) >= threshold).astype(float) * 0.75
            
            ax.imshow(overlay_colors, alpha=alpha_layer)
            ax.axis('off')
            
            ax.set_title(f"Z = {target_mni:.1f} mm", color='white', fontsize=10, fontweight='bold')
            
        # Configure Colorbar in the last available slot
        ax_cb = fig.add_subplot(rows, cols, num_slices + 1)
        ax_cb.axis('off')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax_cb, orientation='vertical', fraction=0.4, pad=0.04)
        cb.set_label('Absolute Score', color='white', fontweight='bold')
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
            
        fig.text(0.5, 0.02, f"{map_title} | Absolute Threshold {threshold:.3f}", 
                 ha='center', color='white', fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='gray', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.5'))
                 
        out_path = self.output_dir / export_filename
        try:
            fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.05, hspace=0.2)
            fig.savefig(out_path, facecolor='black', edgecolor='none')
        except Exception as e:
            self.logger.error(f"Failed to save 3D map plot: {e}")
        finally:
            plt.close(fig)


    def plot_heatmap(self, df_matrix: pd.DataFrame, filename: str, title_suffix: str = "") -> None:
        """
        Generates a comparative Heatmap evaluating feature importance across models.
        
        PURPOSE:
            Replicates the visual analytic standard proposed by Bloch et al.
            Provides a global view to identify which anatomical features are 
            consistently important across different XAI algorithms.
            
        Args:
            df_matrix (pd.DataFrame): 2D Matrix of normalized feature importances.
            filename (str): Name of the exported file.
            title_suffix (str): Contextual addition to the title.
        """
        self.logger.info(f"Building Heatmap - {title_suffix}...")
        
        fig, ax = plt.subplots(figsize=(16, 14), dpi=150)
        
        sns.heatmap(df_matrix, cmap='Blues', annot=False, 
                     cbar_kws={'label': 'Normalized Feature Importance (|Mean ROI|)'},
                    linewidths=.5, linecolor='lightgray', ax=ax)
        
        ax.set_title(f'Global Feature Importances across Models ({title_suffix})', fontsize=16, pad=20)
        ax.set_ylabel('Aspects and Features (Regions of Interest)', fontsize=14)
        ax.set_xlabel('Models and Explanation Methods', fontsize=14)
        
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
        
        out_path = self.output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.tight_layout()
            fig.savefig(out_path, dpi=300)
            self.logger.success(f"Heatmap saved to: {out_path.name}")
        except Exception as e:
            self.logger.error(f"Failed to save Heatmap: {e}")
        finally:
            plt.close(fig)

    def plot_ndcg_matrix(self, ndcg_matrix: pd.DataFrame, filename: str, title_suffix: str = "") -> None:
        """
        Generates a correlation matrix of nDCG scores between methods.
        
        PURPOSE:
            Quantifies ranking agreement. Demonstrates statistically if XAI methods 
            converge on the same biological conclusions as Ground Truth (VBM).
            
        Args:
            ndcg_matrix (pd.DataFrame): nDCG correlation scores dataframe.
            filename (str): Name of the exported file.
            title_suffix (str): Contextual addition to the title.
        """
        self.logger.info(f"Building nDCG Matrix - {title_suffix}...")
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        sns.heatmap(ndcg_matrix, cmap='Blues', annot=False, vmin=0.0, vmax=1.0,
                         cbar_kws={'label': 'nDCG Score'}, linewidths=.5, linecolor='lightgray', ax=ax)
        
        ax.set_title(f'nDCG Similarity Matrix ({title_suffix})', fontsize=16, pad=20)
        ax.set_ylabel('Reference Method', fontsize=12)
        ax.set_xlabel('Comparison Method', fontsize=12)
        
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        out_path = self.output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.tight_layout()
            fig.savefig(out_path, dpi=300)
            self.logger.success(f"nDCG Matrix saved to: {out_path.name}")
        except Exception as e:
            self.logger.error(f"Failed to save nDCG Matrix: {e}")
        finally:
            plt.close(fig)