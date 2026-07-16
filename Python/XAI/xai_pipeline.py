import sys
import os
import pathlib
import numpy as np
import pandas as pd

current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parents[2] 
sys.path.append(str(project_root / "Python"))

from Python.utils.py_logger import CustomLogger
from Python.utils.model_renderer import ModelRenderer
from XAI.roi_analyzer import ROIAnalyzer

def run_comparative_pipeline():
    logger = CustomLogger(name="Comparative_XAI")
    logger.add_console_handler(level="DEBUG", use_colors=True)
    logger.info("--- Booting Comparative XAI Pipeline (Net Impact |Mean| Logic) ---")
    
    vbm_path = project_root / "MATLAB" / "VBM_Pipeline" / "Results" / "Thresholded_Maps" / "TPM_Mask_FWE_corrected_map.nii"
    atlas_path = project_root / "AD_CTRL" / "labels_Neuromorphometrics.nii"
    atlas_csv_path = project_root / "AD_CTRL" / "spm_atlas_labels.csv"
    
    svm_results_dir = project_root / "Python" / "SVM_Pipeline" / "Results"
    output_dir = project_root / "Python" / "XAI" / "Plots"
    
    analyzer = ROIAnalyzer(logger)
    plotter = ModelRenderer(logger, str(output_dir))
    
    if not vbm_path.exists():
        logger.critical(f"File VBM Ground Truth non trovato: {vbm_path}")
        return

    logger.info("Estrazione importanza ROI dalla VBM (Ground Truth)...")
    df_vbm = analyzer.extract_regional_importance(str(vbm_path), str(atlas_path), str(atlas_csv_path), use_absolute=False)
    df_vbm['Abs_Mean_ROI_Signal'] = df_vbm['Mean_ROI_Signal'].abs()
    
    vbm_scores = df_vbm.set_index('ROI_Name')['Abs_Mean_ROI_Signal']
    vbm_min, vbm_max = vbm_scores.min(), vbm_scores.max()
    vbm_norm = (vbm_scores - vbm_min) / (vbm_max - vbm_min) if vbm_max > vbm_min else vbm_scores * 0.0
    
    methods_dict = {
        "SVM Raw Weights (Top 1%)": list(svm_results_dir.glob("SVM_Raw_Weights_Fold_*_Top1.nii")), 
        "SVM Raw Weights (Top 5%)": list(svm_results_dir.glob("SVM_Raw_Weights_Fold_*_Top5.nii")), 
        "SVM Haufe (Top 1%)": list(svm_results_dir.glob("SVM_Haufe_Fold_*_Top1.nii")),
        "SVM Haufe (Top 5%)": list(svm_results_dir.glob("SVM_Haufe_Fold_*_Top5.nii")),
        "SVM Gaonkar (FDR 0.1)": list(svm_results_dir.glob("SVM_Gaonkar_Fold_*_fdr01.nii")),
        "SVM Gaonkar (Bonf 0.05)": list(svm_results_dir.glob("SVM_Gaonkar_Fold_*_bonf005.nii")),
    }
    
    heatmap_columns = {"VBM (Ground Truth)": vbm_norm}
    aggregated_for_matrix = {"VBM (Ground Truth)": vbm_norm}
    
    for method_name, paths_list in methods_dict.items():
        if not paths_list:
            logger.warning(f"Nessuna mappa trovata per {method_name}.")
            continue
            
        logger.info(f"Elaborazione: {method_name}")
        df_agg_real = analyzer.aggregate_and_normalize_maps(paths_list, str(atlas_path), str(atlas_csv_path), metric='Mean_ROI_Signal', use_absolute=False)
        
        if not df_agg_real.empty:
            safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "")
            
            # Delega il salvataggio al plotter passandogli un nome file relativo a output_dir
            div_out_path = f"Directional/Diverging_Mean_{safe_name}.png"
            plotter.plot_diverging_bars(df_agg_real, score_col='Mean_ROI_Signal', title=method_name, filename=div_out_path, top_k=20)

            s_abs = df_agg_real.set_index('ROI_Name')['Mean_ROI_Signal'].abs()
            s_min, s_max = s_abs.min(), s_abs.max()
            s_norm = (s_abs - s_min) / (s_max - s_min) if s_max > s_min else s_abs * 0.0
            aggregated_for_matrix[method_name] = s_norm

        for fold_path in paths_list:
            fold_name = fold_path.stem 
            
            df_fold = analyzer.extract_regional_importance(str(fold_path), str(atlas_path), str(atlas_csv_path), use_absolute=False)
            if df_fold.empty: continue
            
            df_fold['Abs_Mean_ROI_Signal'] = df_fold['Mean_ROI_Signal'].abs()
            
            safe_fold_name = fold_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "")
            csv_out_path = project_root / "Python" / "XAI" / "Results" / "Folds" / f"{safe_fold_name}_FeatureImportance.csv"
            os.makedirs(os.path.dirname(os.path.abspath(csv_out_path)), exist_ok=True)
            df_fold.to_csv(str(csv_out_path), index=False)
            
            plot_out_path = f"Folds/Top20_AbsNetMean_{safe_fold_name}.png"
            plotter.plot_top_rois(df_fold, score_col='Abs_Mean_ROI_Signal', title=fold_name, filename=plot_out_path, top_k=20)
            
            fold_scores = df_fold.set_index('ROI_Name')['Abs_Mean_ROI_Signal']
            f_min, f_max = fold_scores.min(), fold_scores.max()
            fold_norm = (fold_scores - f_min) / (f_max - f_min) if f_max > f_min else fold_scores * 0.0
            
            fold_num = fold_name.split("_Fold_")[1].split("_")[0] if "_Fold_" in fold_name else fold_name[-1]
            heatmap_columns[f"{method_name} (F{fold_num})"] = fold_norm
    
    if len(aggregated_for_matrix) > 1:
        logger.info("Costruzione della Matrice nDCG tra tutti i metodi...")
        matrix_df = pd.DataFrame(aggregated_for_matrix).fillna(0.0)
        method_keys = list(matrix_df.columns)
        ndcg_matrix = pd.DataFrame(index=method_keys, columns=method_keys, dtype=float)
        
        full_k = len(matrix_df)
        for ref_m in method_keys:
            for comp_m in method_keys:
                true_scores = matrix_df[ref_m].values
                pred_scores = matrix_df[comp_m].values
                score = analyzer.calculate_ndcg(pred_scores, true_scores, k=full_k)
                ndcg_matrix.loc[ref_m, comp_m] = score
                
        out_matrix_path = "nDCG_Correlation_Matrix.png"
        plotter.plot_ndcg_matrix(ndcg_matrix, out_matrix_path, title_suffix=f"Net Impact |Mean| (K={full_k})")
    
    if len(heatmap_columns) > 1:
        logger.info("Costruzione della grande Heatmap di Bloch (Tutti i Fold)...")
        heatmap_matrix = pd.DataFrame(heatmap_columns).fillna(0.0)
        heatmap_matrix = heatmap_matrix.sort_values(by="VBM (Ground Truth)", ascending=False)
        
        top_k_heat = 35
        if len(heatmap_matrix) > top_k_heat:
            heatmap_matrix = heatmap_matrix.head(top_k_heat)
            
        out_heat_path = "Bloch_Heatmap_AllFolds.png"
        plotter.plot_bloch_style_heatmap(heatmap_matrix, out_heat_path, title_suffix="Net Impact |Mean|")
        
    logger.success("--- Pipeline Comparativa Completata ---")

if __name__ == "__main__":
    run_comparative_pipeline()