import os
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict, Tuple, List

from utils.py_logger import CustomLogger

class ROIAnalyzer:
    """
    Extracts regional feature importance from dense 3D XAI tensors using discrete SPM Atlases.
    Includes methods for comparing different XAI maps and aggregating across CV folds.
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
        return label_dict

    def extract_regional_importance(self, xai_map_path: str, atlas_map_path: str, 
                                    label_csv_path: str, threshold: float = 0.0) -> pd.DataFrame:
        """
        Calculates the feature importance for each Region of Interest defined by the SPM Atlas.
        Filters out White Matter, Ventricles, and Cerebellum to focus on Gray Matter.
        """
        roi_dict = self._load_atlas_labels(label_csv_path)
        xai_img = nib.load(xai_map_path)
        atlas_img = nib.load(atlas_map_path)
        
        xai_vol = xai_img.get_fdata()
        atlas_vol = np.round(atlas_img.get_fdata()).astype(int)
        
        abs_xai = np.abs(xai_vol)
        results = []
        
        unique_atlas_ids = np.unique(atlas_vol)
        
        exclude_keywords = [
            'white matter', 'wm', 'ventricle', 'vent', 'cerebellum', 'cerebellar', 
            'brain-stem', 'chiasm', 'vessel', 'csf', 'unknown', 'background'
        ]
        
        for roi_id in unique_atlas_ids:
            if roi_id == 0:
                continue 
                
            roi_name = roi_dict.get(roi_id, f"Unknown_Region_{roi_id}")
            
            name_lower = str(roi_name).lower()
            if any(keyword in name_lower for keyword in exclude_keywords):
                continue
                
            roi_mask = (atlas_vol == roi_id)
            roi_values = abs_xai[roi_mask]
            
            total_voxels = len(roi_values)
            if total_voxels == 0:
                continue
                
            mean_roi_signal = float(np.sum(roi_values) / total_voxels)
            sum_roi_signal = float(np.sum(roi_values))
            
            results.append({
                'ROI_ID': roi_id,
                'ROI_Name': roi_name,
                'Mean_ROI_Signal': mean_roi_signal,
                'Sum_ROI_Signal': sum_roi_signal
            })
            
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values(by='Mean_ROI_Signal', ascending=False).reset_index(drop=True)
            
        return df_results

    def aggregate_and_normalize_maps(self, map_paths: List[str], atlas_map_path: str, label_csv_path: str, metric: str = 'Mean_ROI_Signal') -> pd.DataFrame:
        """
        Estrae l'importanza regionale per una lista di mappe (es. i 5 fold), ne fa la media,
        e poi normalizza i risultati tra 0 e 1 (Min-Max Scaling) come nell'articolo di Bloch.
        """
        if not map_paths:
            self.logger.warning("Lista di mappe vuota fornita all'aggregatore.")
            return pd.DataFrame()

        self.logger.info(f"Aggregazione e normalizzazione di {len(map_paths)} mappe per la metrica: {metric}...")
        
        all_series = []
        for path in map_paths:
            if not os.path.exists(path):
                self.logger.warning(f"File non trovato, lo salto: {path}")
                continue
                
            df = self.extract_regional_importance(str(path), atlas_map_path, label_csv_path)
            s = df.set_index('ROI_Name')[metric]
            all_series.append(s)
            
        if not all_series:
            return pd.DataFrame()
            
        # Concatena tutti i fold come colonne e fai la media per ogni ROI
        combined_df = pd.concat(all_series, axis=1)
        mean_series = combined_df.mean(axis=1)
        
        # Normalizzazione Min-Max (0 - 1)
        min_val = mean_series.min()
        max_val = mean_series.max()
        
        if max_val > min_val:
            norm_series = (mean_series - min_val) / (max_val - min_val)
        else:
            norm_series = mean_series * 0.0
            
        result_df = norm_series.reset_index()
        result_df.columns = ['ROI_Name', 'Normalized_Importance']
        return result_df

    # ... [Manteniamo le funzioni _dcg, calculate_ndcg, compare_maps_ndcg identiche a prima] ...
    def _dcg(self, scores: np.ndarray) -> float:
        return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

    def calculate_ndcg(self, predicted_scores: np.ndarray, true_scores: np.ndarray, k: int) -> float:
        if len(predicted_scores) == 0 or len(true_scores) == 0: return 0.0
        pred_order = np.argsort(predicted_scores)[::-1]
        ideal_order = np.argsort(true_scores)[::-1]
        ideal_scores = true_scores[ideal_order][:k]
        idcg = self._dcg(ideal_scores)
        if idcg == 0: return 0.0
        actual_scores = true_scores[pred_order][:k]
        dcg = self._dcg(actual_scores)
        return dcg / idcg

    def compare_maps_ndcg(self, map1_df: pd.DataFrame, map2_df: pd.DataFrame, metric: str = 'Mean_ROI_Signal', k: int = 10) -> float:
        merged = pd.merge(map1_df, map2_df, on='ROI_ID', suffixes=('_pred', '_true'))
        if merged.empty: return 0.0
        return self.calculate_ndcg(merged[f'{metric}_pred'].values, merged[f'{metric}_true'].values, k)