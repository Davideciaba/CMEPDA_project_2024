"""
Module: roi_analyzer.py

Extracts, aggregates, and mathematically evaluates regional feature importance 
from dense 3D XAI tensors. Maps continuous voxel-level data to discrete 
Regions of Interest (ROIs) using standard SPM Atlases.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict, List

from Python.utils.py_logger import CustomLogger

class ROIAnalyzer:
    """
    Extracts regional feature importance from dense 3D XAI tensors using discrete SPM Atlases.
    
    PURPOSE:
        Acts as the analytical bridge between continuous 3D XAI maps (e.g., Haufe, Gaonkar) 
        and discrete neuroanatomical regions. Includes methods for comparing different 
        XAI maps via nDCG and aggregating results across Nested CV folds.
    """
    
    def __init__(self, logger: CustomLogger):
        """
        Initializes the ROIAnalyzer.
        
        Args:
            logger (CustomLogger): Centralized logging instance.
        """
        self.logger = logger

    def _load_atlas_labels(self, label_csv_path: str) -> Dict[int, str]:
        """
        Loads the SPM Atlas dictionary mapping integer IDs to anatomical names.
        
        Args:
            label_csv_path (str): Absolute path to the CSV containing atlas labels.
            
        Returns:
            Dict[int, str]: Dictionary mapping ROI IDs to their string names.
        """
        if not os.path.exists(label_csv_path):
            raise FileNotFoundError(f"Atlas label CSV not found at: {label_csv_path}")
            
        df = pd.read_csv(label_csv_path)
        id_col = 'ROI_ID' if 'ROI_ID' in df.columns else df.columns[0]
        name_col = 'ROI_Name' if 'ROI_Name' in df.columns else df.columns[1]
        
        label_dict = dict(zip(df[id_col].astype(int), df[name_col].astype(str)))
        return label_dict

    def extract_regional_importance(self, xai_map_path: str, atlas_map_path: str, 
                                    label_csv_path: str, use_absolute: bool = True) -> pd.DataFrame:
        """
        Calculates the feature importance for each Region of Interest.
        
        PURPOSE:
            Extracts voxel data guided by the SPM Atlas. Filters out non-relevant tissues 
            (White Matter, Ventricles, Cerebellum) to focus on Gray Matter.
            If use_absolute is True, computes metrics on absolute values (for magnitude/ranking).
            If use_absolute is False, computes metrics on raw values (keeping signs for directional impact).
            
        Args:
            xai_map_path (str): Path to the 3D XAI NIfTI map.
            atlas_map_path (str): Path to the discrete SPM Atlas NIfTI map.
            label_csv_path (str): Path to the CSV mapping atlas IDs to names.
            use_absolute (bool): Flag for absolute magnitude extraction vs directional extraction.
                
        Returns:
            pd.DataFrame: Table containing ROI_ID, ROI_Name, Mean_ROI_Signal, Sum_ROI_Signal.
        """
        roi_dict = self._load_atlas_labels(label_csv_path)
        xai_img = nib.load(xai_map_path)
        atlas_img = nib.load(atlas_map_path)
        
        xai_vol = xai_img.get_fdata()
        atlas_vol = np.round(atlas_img.get_fdata()).astype(int)
        
        # Use absolute values for global importance to avoid signal cancellation.
        # Otherwise, keep signs to see if the feature are protective or progressive
        if use_absolute:
            work_vol = np.abs(xai_vol)
        else:
            work_vol = xai_vol
            
        results = []
        
        unique_atlas_ids = np.unique(atlas_vol)
        
        # Filter out regions that are not Gray Matter or not pertinent to the study
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
            roi_values = work_vol[roi_mask]
            
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
            # Even if signs are kept, sort by absolute magnitude to keep the most "impactful" regions on top
            df_results['abs_sort'] = df_results['Mean_ROI_Signal'].abs()
            df_results = df_results.sort_values(by='abs_sort', ascending=False).drop(columns=['abs_sort']).reset_index(drop=True)
            
        return df_results

    def aggregate_and_normalize_maps(self, map_paths: List[str], atlas_map_path: str, label_csv_path: str, metric: str = 'Mean_ROI_Signal', use_absolute: bool = True) -> pd.DataFrame:
        """
        Averages regional importance across multiple cross-validation folds.
        
        PURPOSE:
            Aggregates fold-specific XAI maps into a single global representation.
            Applies Min-Max scaling [0, 1] if magnitude evaluation is requested,
            ensuring mathematical comparability across different methods.
            
        Args:
            map_paths (List[str]): List of absolute paths to the XAI NIfTI files.
            atlas_map_path (str): Path to the discrete SPM Atlas NIfTI map.
            label_csv_path (str): Path to the CSV mapping atlas IDs to names.
            metric (str): The DataFrame column to aggregate (default: 'Mean_ROI_Signal').
            use_absolute (bool): Determines magnitude vs directional aggregation.
            
        Returns:
            pd.DataFrame: Aggregated results. Contains 'Normalized_Importance' if 
                          use_absolute=True, otherwise contains the raw averaged metric.
        """
        if not map_paths:
            self.logger.warning("Lista di mappe vuota fornita all'aggregatore.")
            return pd.DataFrame()

        self.logger.info(f"Aggregazione di {len(map_paths)} mappe per la metrica: {metric} (Absolute: {use_absolute})...")
        
        all_series = []
        for path in map_paths:
            if not os.path.exists(path):
                self.logger.warning(f"File non trovato, lo salto: {path}")
                continue
                
            df = self.extract_regional_importance(str(path), atlas_map_path, label_csv_path, use_absolute=use_absolute)
            if not df.empty:
                s = df.set_index('ROI_Name')[metric]
                all_series.append(s)
            
        if not all_series:
            return pd.DataFrame()
            
        # Concatenate all folds as columns and compute the row-wise mean (across CV)
        combined_df = pd.concat(all_series, axis=1)
        mean_series = combined_df.mean(axis=1)
        
        if use_absolute:
            # Min-Max Normalization (0 - 1) required for Heatmaps
            min_val = mean_series.min()
            max_val = mean_series.max()
            if max_val > min_val:
                norm_series = (mean_series - min_val) / (max_val - min_val)
            else:
                norm_series = mean_series * 0.0
            result_df = norm_series.reset_index()
            result_df.columns = ['ROI_Name', 'Normalized_Importance']
        else:
            # Return raw averaged values preserving original signs for Diverging Bar Plots
            result_df = mean_series.reset_index()
            result_df.columns = ['ROI_Name', metric]
            
        return result_df

    def _dcg(self, scores: np.ndarray) -> float:
        """
        Computes the Discounted Cumulative Gain (DCG) for a vector of scores.
        """
        return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

    def calculate_ndcg(self, predicted_scores: np.ndarray, true_scores: np.ndarray, k: int) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (nDCG).
        
        PURPOSE:
            Evaluates the ranking quality of a predictive XAI model against the Ground Truth.
            Uses a logarithmic decay to penalize errors in the most important (Top K) regions.
            
        Args:
            predicted_scores (np.ndarray): Importance scores generated by the XAI method.
            true_scores (np.ndarray): True importance scores (e.g., from VBM).
            k (int): Number of top elements to consider.
            
        Returns:
            float: nDCG score between 0.0 (total disagreement) and 1.0 (perfect ranking).
        """
        if len(predicted_scores) == 0 or len(true_scores) == 0: return 0.0
        
        # Determine the IDEAL ranking based strictly on Ground Truth scores
        ideal_order = np.argsort(true_scores)[::-1]
        ideal_scores = true_scores[ideal_order][:k]
        idcg = self._dcg(ideal_scores)
        if idcg == 0: return 0.0
        
        # Determine the actual ranking proposed by the XAI model
        pred_order = np.argsort(predicted_scores)[::-1]
        # Retrieve the True scores ordered by the XAI's ranking logic
        actual_scores = true_scores[pred_order][:k]
        dcg = self._dcg(actual_scores)
        
        return dcg / idcg

    def compare_maps_ndcg(self, map1_df: pd.DataFrame, map2_df: pd.DataFrame, metric: str = 'Mean_ROI_Signal', k: int = 10) -> float:
        """
        Wrapper to compute nDCG directly between two Pandas DataFrames containing ROI metrics.
        """
        merged = pd.merge(map1_df, map2_df, on='ROI_ID', suffixes=('_pred', '_true'))
        if merged.empty: return 0.0
        return self.calculate_ndcg(merged[f'{metric}_pred'].values, merged[f'{metric}_true'].values, k)