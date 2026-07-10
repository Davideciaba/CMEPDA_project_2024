"""
Model Renderer Module.

This module houses the ModelRenderer class, analogous to MATLAB's BrainRenderer.
It isolates all plotting libraries (matplotlib, seaborn) from the core mathematical engines,
providing robust, stateful methods for exporting publication-ready MLOps graphics.
"""
import math
import pathlib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import auc
from typing import List, Dict, Any, Tuple
from Python.utils.py_logger import CustomLogger

class ModelRenderer:
    """
    Handles Graphics Visualization and Exporting for Python Models.
    Isolates rendering logic to adhere strictly to the Single Responsibility Principle.
    """

    def __init__(self, logger: CustomLogger, output_dir: str):
        """
        Initializes the ModelRenderer object.
        Args:
            logger: CustomLogger instance for execution tracking.
            output_dir: Base directory path to save the rendered plots.
        """
        self.logger = logger
        self.output_dir = pathlib.Path(output_dir).resolve()
        
        # EAFP: Attempt to create the plots directory if it doesn't exist natively
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_roc_curves(self, fold_artifacts: List[Dict[str, Any]], model_name: str, filename: str) -> None:
        """
        Plots the Receiver Operating Characteristic (ROC) curve for each fold,
        along with the Mean ROC curve and its ±1 Standard Deviation variance band.
        All AUC scores are formatted as percentages (0-100%).
        
        Args:
            fold_artifacts: The artifacts list returned by execute_nested_cv.
            model_name: String label for the plot (e.g., "Linear SVM").
            filename: Output filename (e.g., "SVM_ROC_Curve.png").
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

    def _get_voxel_indices_from_mni(self, affine_mat: np.ndarray, slice_config: Any, max_idx: int, active_mask: np.ndarray) -> Tuple[List[int], List[float]]:
        """
        Translates a configuration (scalar, [start, step, stop], or explicit list)
        into valid physical Z-axis array coordinates. Matches BrainRenderer.m exactly.
        """
        # Extract Affine parameters strictly for Z-axis (Index 2 in Python)
        translation = affine_mat[2, 3]
        scale = affine_mat[2, 2]

        # Resolve Target MNI Coordinates via Duck Typing
        if isinstance(slice_config, (int, float)):
            step = float(slice_config)
            
            # Find active bounding box along Z-axis (Compress X and Y)
            active_slices = np.any(active_mask, axis=(0, 1))
            active_idx = np.where(active_slices)[0]
            
            if len(active_idx) == 0:
                self.logger.warning("No active voxels found in the mask.")
                return [], []
                
            start_idx = active_idx[0]
            stop_idx = active_idx[-1]
            
            # Convert to MNI
            start_mni = scale * start_idx + translation
            stop_mni = scale * stop_idx + translation
            
            # Generate range accounting for affine scaling direction
            if start_mni < stop_mni:
                target_mnis = list(np.arange(start_mni, stop_mni, step))
            else:
                target_mnis = list(np.arange(start_mni, stop_mni -step))
                
            # Guarantee the absolute last slice is included (like MATLAB)
            if not np.isclose(target_mnis[-1], stop_mni, atol=1e-3):
                target_mnis.append(stop_mni)

        elif isinstance(slice_config, list) and len(slice_config) == 3:
            # Interpreted securely as MATLAB's [start : step : stop]
            start, step, stop = slice_config
            target_mnis = np.arange(start, stop, step).tolist()
        elif isinstance(slice_config, (list, tuple, np.ndarray)):
            # Interpreted as specific discrete slices
            target_mnis = [float(x) for x in slice_config]
        else:
            self.logger.error("Invalid slice_config format provided.")
            raise ValueError("slice_config must be a scalar, [start, step, stop], or a list of MNI coordinates.")

        
        
        valid_voxels = []
        valid_mnis = []
        
        # 3. Project to Voxel space and apply boundary enforcement
        for mni in target_mnis:
            vox_idx = int(round((mni - translation) / scale))
            if 0 <= vox_idx <= max_idx:
                valid_voxels.append(vox_idx)
                valid_mnis.append(mni)
                
        return valid_voxels, valid_mnis

    def plot_3d_activation_map(self, bg_nifti_path: str, stats_nifti_path: str, mask_nifti_path: str,
                               map_title: str, export_filename: str, threshold: float = 0.0,
                               slice_config: Any = [-30.5, 3.0, 60.5]) -> None:
        """
        Python replication of MATLAB's BrainRenderer.plotStatisticalOverlay.
        Extracts 2D Axial slices surgically and overlays hot colormaps natively.
        """
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

        # Z-axis boundary constraint
        max_idx = bg_data.shape[2] -1
        
        # Translate MNI to Voxels
        vox_indices, valid_mnis = self._get_voxel_indices_from_mni(affine, slice_config, max_idx, mask_data)
        
        if not vox_indices:
            self.logger.error(f"No valid brain slices found for config: {slice_config}. Bypassing render.")
            return
            #raise ValueError("No valid slices found for the provided MNI configuration.")
            
        self.logger.debug(f"Rendering {map_title} | Z-Axis Computed Slices: {len(vox_indices)}")
            
        # Matrix Layout Algorithm
        num_slices = len(vox_indices)
        cols = math.ceil(math.sqrt(num_slices + 1))
        rows = math.ceil((num_slices + 1) / cols)
        
        fig = plt.figure(figsize=(cols * 3, rows * 3), dpi=150)
        fig.patch.set_facecolor('black')
        
        # Global Coloring Logic
        vmax = np.max(np.abs(stats_data[mask_data]))
        if vmax == 0: vmax = 1.0
        if vmax <= threshold:
            self.logger.warning(f"No voxels exceed the threshold ({threshold}). Aborting.")
            # raise ValueError("No voxels exceed the threshold.")
            
        norm = Normalize(vmin=-vmax, vmax=vmax, clip=True)
        cmap = plt.cm.coolwarm
        
        for idx, (vox_idx, target_mni) in enumerate(zip(vox_indices, valid_mnis)):
            # Explicitly create and attach the graphical axis for this specific slice
            ax = fig.add_subplot(rows, cols, idx + 1)
            
            # --- Surgical 2D Axial Slice Extraction ---
            slice_bg = bg_data[:, :, vox_idx]
            slice_stats = stats_data[:, :, vox_idx]
                
            # Radiological Rotation
            slice_bg = np.rot90(slice_bg, k=1)
            slice_stats = np.rot90(slice_stats, k=1)
            
            # Plot Background (Grayscale mapping)
            ax.imshow(slice_bg, cmap='gray')
            
            # Plot Overlay (Hot colormap injection)
            overlay_colors = cmap(norm(slice_stats))
            # Alpha mask: Only show voxels exceeding the mathematical threshold
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