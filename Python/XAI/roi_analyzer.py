"""
Module: roi_analyzer.py

Provides analytical tools to aggregate voxel-wise Explainable AI (XAI) maps 
into structural Regions of Interest (ROIs) using standard SPM atlases.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict

from utils.py_logger import CustomLogger


class ROIAnalyzer:
    """
    Extracts regional feature importance from dense 3D XAI tensors using discrete SPM Atlases.
    """
    
    def __init__(self, logger: CustomLogger):
        self.logger = logger

    def _load_atlas_labels(self, label_csv_path: str) -> Dict[int, str]:
        if not os.path.exists(label_csv_path):
            raise FileNotFoundError(f"Atlas label CSV not found at: {label_csv_path}")
            
        df = pd.read_csv(label_csv_path)
        id_col = 'ROI_ID' if 'ROI_ID' in df.columns else df.columns[0]
        name_col = 'ROI_Name' if 'ROI_Name' in df.columns else df.columns[1]
        
        label_dict = dict(zip(df[id_col].astype(int), df[name_col].astype(str)))
        self.logger.debug(f"Loaded {len(label_dict)} ROI definitions from atlas dictionary.")
        return label_dict

    def extract_regional_importance(self, xai_map_path: str, atlas_map_path: str, 
                                    label_csv_path: str, threshold: float = 0.0) -> pd.DataFrame:
        self.logger.info("Extracting Regional Feature Importance via SPM Atlas...")
        
        roi_dict = self._load_atlas_labels(label_csv_path)
        
        xai_img = nib.load(xai_map_path)
        atlas_img = nib.load(atlas_map_path)
        
        xai_vol = xai_img.get_fdata()
        atlas_vol = np.round(atlas_img.get_fdata()).astype(int)
        
        if xai_vol.shape != atlas_vol.shape:
            self.logger.error("Spatial dimension mismatch between XAI map and SPM Atlas.")
            raise ValueError(f"XAI shape {xai_vol.shape} != Atlas shape {atlas_vol.shape}.")
            
        abs_xai = np.abs(xai_vol)
        results = []
        
        unique_atlas_ids = np.unique(atlas_vol)
        
        for roi_id in unique_atlas_ids:
            if roi_id == 0:
                continue 
                
            roi_name = roi_dict.get(roi_id, f"Unknown_Region_{roi_id}")
            
            roi_mask = (atlas_vol == roi_id)
            roi_values = abs_xai[roi_mask]
            
            total_voxels = len(roi_values)
            if total_voxels == 0:
                continue
                
            # --- METRICA RICHIESTA: Media sul numero TOTALE di voxel nella ROI ---
            mean_roi_signal = float(np.sum(roi_values) / total_voxels)
            
            # Sub-filtraggio per metriche accessorie sui soli voxel attivi (Thresholding)
            active_values = roi_values[roi_values >= threshold]
            active_count = len(active_values)
            
            results.append({
                'ROI_ID': roi_id,
                'ROI_Name': roi_name,
                'Total_Voxels': total_voxels,
                'Mean_ROI_Signal': mean_roi_signal, # La tua metrica di densità
                'Active_Voxels': active_count,
                'Sum_Active_Signal': float(np.sum(active_values)) if active_count > 0 else 0.0
            })
            
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            # Ordiniamo il DataFrame usando la tua nuova metrica globale come parametro principale
            df_results = df_results.sort_values(by='Mean_ROI_Signal', ascending=False).reset_index(drop=True)
            
        self.logger.success(f"ROI analysis complete. Evaluated {len(df_results)} anatomical regions.")
        return df_results