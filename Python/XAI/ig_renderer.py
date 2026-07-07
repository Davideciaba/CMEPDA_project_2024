"""
Module: ig_renderer.py

Provides 3D tensor visualization primitives to render Deep Learning Integrated Gradients (IG).
Uses a diverging colormap ('coolwarm') bounded by an empirical alpha-level threshold,
ensuring visual sparsity matching statistical analytical models.
"""
import os
import math
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Tuple, List

from utils.py_logger import CustomLogger

class IGRenderer:
    """
    Renderer optimized for Deep Learning Integrated Gradients (IG).
    """
    
    def __init__(self, logger: CustomLogger):
        self.logger = logger

    def _get_voxel_indices_from_mni(self, affine_mat: np.ndarray, tensor_size: Tuple[int, int, int], 
                                    active_mask: np.ndarray, step_mm: float = 5.0) -> Tuple[List[int], List[float]]:
        """Converts active bounding box to MNI millimeters and physical voxel Z-indices."""
        max_z = tensor_size[2]
        vox_center_xy = (tensor_size[0] // 2, tensor_size[1] // 2)
        
        active_slices = np.any(active_mask, axis=(0, 1))
        active_idx = np.where(active_slices)[0]
        
        if len(active_idx) == 0:
            self.logger.warning("No active voxels found. Plotting aborted.")
            return [], []
            
        z_min_vox, z_max_vox = active_idx[0], active_idx[-1]
        
        bound_min = affine_mat @ np.array([vox_center_xy[0], vox_center_xy[1], z_min_vox, 1])
        bound_max = affine_mat @ np.array([vox_center_xy[0], vox_center_xy[1], z_max_vox, 1])
        
        mni_min, mni_max = bound_min[2], bound_max[2]
        if mni_min > mni_max:
            mni_min, mni_max = mni_max, mni_min
            
        mni_min -= step_mm
        mni_max += step_mm
        
        mni_array = np.arange(mni_min, mni_max + step_mm, step_mm)
        
        affine_inv = np.linalg.inv(affine_mat)
        z_slices_voxel = []
        for z_mm in mni_array:
            vox_coords = affine_inv @ np.array([0, 0, z_mm, 1])
            z_vox = int(round(vox_coords[2]))
            if 0 <= z_vox < max_z:
                z_slices_voxel.append(z_vox)
                
        z_slices_voxel = sorted(list(set(z_slices_voxel)))
        
        z_mm_array = []
        for z_vox in z_slices_voxel:
            true_mni = affine_mat @ np.array([vox_center_xy[0], vox_center_xy[1], z_vox, 1])
            z_mm_array.append(float(true_mni[2]))
            
        return z_slices_voxel, z_mm_array

    def plot_ig_map(self, map_path: str, bg_path: str, out_fig_path: str, 
                    alpha_level: float = 0.05, step_mm: float = 3.0, 
                    map_title: str = "EfficientNet IG Map") -> None:
        """
        Renders thresholded Integrated Gradients over a raw 3D volume.
        Isolates the most extreme absolute attributions via empirical alpha percentile.
        """
        self.logger.info(f"Initiating IG Diverging Overlay Rendering (Empirical Alpha = {alpha_level})...")
        
        bg_img = nib.load(bg_path)
        bg_vol = bg_img.get_fdata()
        affine_mat = bg_img.affine
        
        ig_vol = nib.load(map_path).get_fdata()
        
        if bg_vol.shape != ig_vol.shape:
            raise ValueError("Dimension mismatch between background and IG map.")
            
        abs_ig = np.abs(ig_vol)
        
        # Calcolo Soglia Adattiva divergente (Percentile Empirico del valore assoluto)
        brain_mask = bg_vol > 0
        brain_values_abs = abs_ig[brain_mask]
        
        if len(brain_values_abs) == 0:
            self.logger.warning("Empty background volume detected. Plotting aborted.")
            return

        percentile_rank = 100.0 * (1.0 - alpha_level)
        threshold = np.percentile(brain_values_abs, percentile_rank)
        
        max_abs = np.max(abs_ig)
        if max_abs <= threshold:
            max_abs = threshold + 1e-5
            
        active_mask = abs_ig >= threshold
        z_voxels, z_mms = self._get_voxel_indices_from_mni(affine_mat, bg_vol.shape, active_mask, step_mm)
        num_slices = len(z_voxels)
        
        if num_slices == 0: return

        total_panels = num_slices + 1
        grid_cols = math.ceil(math.sqrt(total_panels))
        grid_rows = math.ceil(total_panels / grid_cols)
        
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3), facecolor='black')
        axes = axes.flatten() if total_panels > 1 else np.array([axes])
        
        cmap = plt.get_cmap('coolwarm')
        norm = Normalize(vmin=-max_abs, vmax=max_abs)
        
        for idx, (z_vox, z_mm) in enumerate(zip(z_voxels, z_mms)):
            ax = axes[idx]
            slice_bg = np.rot90(bg_vol[:, :, z_vox])
            slice_stats = np.rot90(ig_vol[:, :, z_vox])
            
            ax.imshow(slice_bg, cmap='gray')
            
            overlay_colors = cmap(norm(slice_stats))
            alpha_layer = (np.abs(slice_stats) >= threshold).astype(float) * 0.75
            
            ax.imshow(overlay_colors, alpha=alpha_layer)
            ax.axis('off')
            ax.set_title(f"Z = {z_mm:.1f} mm", color='white', fontsize=10, fontweight='bold')
            
        ax_cb = axes[num_slices]
        ax_cb.axis('off')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax_cb, orientation='vertical', fraction=0.4, pad=0.04)
        cb.set_label('Attribution (Blue: - | Red: +)', color='white', fontweight='bold')
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

        for idx in range(num_slices + 1, len(axes)):
            axes[idx].axis('off')
            
        fig.text(0.5, 0.02, f" {map_title} | Top {alpha_level*100:.1f}% Abs Attributions (|IG| \u2265 {threshold:.2f})", 
                 ha='center', color='white', fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='#333333', edgecolor='none', boxstyle='round,pad=0.5'))
                 
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        os.makedirs(os.path.dirname(os.path.abspath(out_fig_path)), exist_ok=True)
        plt.savefig(out_fig_path, facecolor=fig.get_facecolor(), dpi=300)
        plt.close(fig)
        self.logger.success(f"IG Map saved to: {out_fig_path}")