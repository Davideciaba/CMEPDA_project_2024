import os
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict, Tuple

from utils.py_logger import CustomLogger


class ROIAnalyzer:
    """
    Extracts regional feature importance from dense 3D XAI tensors using discrete SPM Atlases.
    Includes methods for comparing different XAI maps using ranking metrics like NDCG.
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
        """
        Calculates the feature importance for each Region of Interest defined by the SPM Atlas.
        Filters out White Matter, Ventricles, and Cerebellum to focus on Gray Matter.
        """
        self.logger.info("Extracting Regional Feature Importance via SPM Atlas...")
        
        roi_dict = self._load_atlas_labels(label_csv_path)
        
        xai_img = nib.load(xai_map_path)
        atlas_img = nib.load(atlas_map_path)
        
        xai_vol = xai_img.get_fdata()
        atlas_vol = np.round(atlas_img.get_fdata()).astype(int)
        
        if xai_vol.shape != atlas_vol.shape:
            self.logger.error("Spatial dimension mismatch between XAI map and SPM Atlas.")
            raise ValueError(f"XAI shape {xai_vol.shape} != Atlas shape {atlas_vol.shape}. Coregistration required.")
            
        abs_xai = np.abs(xai_vol)
        results = []
        
        unique_atlas_ids = np.unique(atlas_vol)
        
        # Keywords to filter out non-Gray Matter regions
        exclude_keywords = [
            'white matter', 'wm', 'ventricle', 'vent', 'cerebellum', 'cerebellar', 
            'brain-stem', 'chiasm', 'vessel', 'csf', 'unknown', 'background'
        ]
        
        for roi_id in unique_atlas_ids:
            if roi_id == 0:
                continue 
                
            roi_name = roi_dict.get(roi_id, f"Unknown_Region_{roi_id}")
            
            # Skip non-GM regions based on naming conventions
            name_lower = str(roi_name).lower()
            if any(keyword in name_lower for keyword in exclude_keywords):
                continue
                
            roi_mask = (atlas_vol == roi_id)
            roi_values = abs_xai[roi_mask]
            
            total_voxels = len(roi_values)
            if total_voxels == 0:
                continue
                
            # Metrics calculation
            mean_roi_signal = float(np.sum(roi_values) / total_voxels)
            sum_roi_signal = float(np.sum(roi_values))
            
            active_values = roi_values[roi_values >= threshold]
            active_count = len(active_values)
            
            results.append({
                'ROI_ID': roi_id,
                'ROI_Name': roi_name,
                'Total_Voxels': total_voxels,
                'Mean_ROI_Signal': mean_roi_signal,
                'Sum_ROI_Signal': sum_roi_signal,
                'Active_Voxels': active_count,
                'Sum_Active_Signal': float(np.sum(active_values)) if active_count > 0 else 0.0
            })
            
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values(by='Mean_ROI_Signal', ascending=False).reset_index(drop=True)
            
        self.logger.success(f"ROI analysis complete. Evaluated {len(df_results)} Grey Matter regions.")
        return df_results

    def _dcg(self, scores: np.ndarray) -> float:
        """
        Calculates the Discounted Cumulative Gain (DCG) for a list of scores.
        """
        return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

    def calculate_ndcg(self, predicted_scores: np.ndarray, true_scores: np.ndarray, k: int) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (nDCG) at top K.
        Compares a predicted ranking of regions against a 'ground truth' or baseline ranking.
        """
        if len(predicted_scores) == 0 or len(true_scores) == 0:
            return 0.0
            
        # Get the sorted indices for both the predicted and true scores (descending)
        pred_order = np.argsort(predicted_scores)[::-1]
        ideal_order = np.argsort(true_scores)[::-1]
        
        # Calculate ideal DCG using the top k values from the true ranking
        ideal_scores = true_scores[ideal_order][:k]
        idcg = self._dcg(ideal_scores)
        
        if idcg == 0:
            return 0.0
            
        # Calculate actual DCG using the true scores of the items predicted in the top k
        actual_scores = true_scores[pred_order][:k]
        dcg = self._dcg(actual_scores)
        
        return dcg / idcg

    def compare_maps_ndcg(self, map1_df: pd.DataFrame, map2_df: pd.DataFrame, metric: str = 'Mean_ROI_Signal', k: int = 10) -> float:
        """
        Helper method to calculate NDCG between two processed ROI DataFrames.
        map1_df is treated as the 'prediction' and map2_df as the 'truth' (or baseline).
        """
        # Ensure both dataframes contain the same regions in the same order
        merged = pd.merge(map1_df, map2_df, on='ROI_ID', suffixes=('_pred', '_true'))
        
        if merged.empty:
            self.logger.warning("Could not calculate NDCG: No overlapping ROIs found between the two maps.")
            return 0.0
            
        scores_pred = merged[f'{metric}_pred'].values
        scores_true = merged[f'{metric}_true'].values
        
        ndcg_val = self.calculate_ndcg(scores_pred, scores_true, k)
        self.logger.debug(f"Calculated NDCG@{k} for {metric}: {ndcg_val:.4f}")
        
        return ndcg_val