import sys
from pathlib import Path
from typing import Optional
import pandas as pd

from Python.utils.py_logger import CustomLogger
from Python.utils.model_renderer import ModelRenderer
from Python.utils.roi_analyzer import ROIAnalyzer
from Python.utils.reset_directory import reset_directory

def run_xai_comparison(
    enable_file_logging: bool = False, 
    output_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None
) -> None:
    current_dir = Path(__file__).parent.resolve()
    base_out = output_dir.resolve() if output_dir else current_dir

    atlas_path = input_dir / "labels_Neuromorphometrics.nii"
    atlas_csv_path = input_dir / "spm_atlas_labels.csv"

    matlab_results_dir = base_out.parent / "MATLAB_Results"
    vbm_path = matlab_results_dir / "VBM_Pipeline_Results" / "Results" / "TPM_Mask_FWE_corrected_map.nii"
    
    xai_maps_dir = base_out / "SVM_XAI_Results" / "Results"
    
    comp_base = base_out / "XAI_Comparison_Results"
    results_dir = comp_base / "Results"
    plots_dir = comp_base / "Plots"
    log_dir = comp_base / "Log_Files"
    
    log = CustomLogger(name="XAI_Comparison")
    log.add_console_handler(level="DEBUG", use_colors=True)
    
    if enable_file_logging:
        reset_directory(log_dir, log)
        log_path = log_dir / "XAI_Comparison.log"
        try:
            log.add_file_handler(str(log_path), level="DEBUG")
        except OSError as e:
            log.critical(f"I/O ERROR: Cannot write to {log_path.name}")
            log.critical("Pipeline aborted. Ensure you have write permissions.")
            sys.exit(1)
    else:
        comp_base.mkdir(parents=True, exist_ok=True)
        dummy_file = comp_base / ".dummy_write_test"
        try:
            with open(dummy_file, 'w') as f: pass
            dummy_file.unlink()
            log.info("Dummy write test passed. Filesystem allows writing.")
        except OSError as e:
            log.critical(f"I/O ERROR: Cannot write to {comp_base}.")
            log.critical(f"Pipeline aborted. Ensure you have write permissions. Details: {e}")
            sys.exit(1)

    log.info("--- Booting XAI Comparison Pipeline ---")

    reset_directory(results_dir, log)
    reset_directory(plots_dir, log)
    
    analyzer = ROIAnalyzer(log)
    plotter = ModelRenderer(log, str(plots_dir))
    
    if not atlas_path.exists() or not atlas_csv_path.exists():
        log.critical(f"FATAL: Neuromorphometrics Atlas files missing in {input_dir}.")
        sys.exit(1)

    heatmap_columns = {}
    aggregated_for_matrix = {}

    log.info("Phase 1: Validating VBM Ground Truth...")
    if not vbm_path.exists():
        log.warning(f"VBM Ground Truth file not found at: {vbm_path}")
        log.warning("Proceeding with model-to-model comparison only.")
    else:
        log.info("Extracting ROI importance from VBM (Ground Truth)...")
        try:
            df_vbm = analyzer.extract_regional_importance(str(vbm_path), str(atlas_path), str(atlas_csv_path), use_absolute=False)
            df_vbm['Abs_Mean_ROI_Signal'] = df_vbm['Mean_ROI_Signal'].abs()
            
            vbm_scores = df_vbm.set_index('ROI_Name')['Abs_Mean_ROI_Signal']
            vbm_min, vbm_max = vbm_scores.min(), vbm_scores.max()
            vbm_norm = (vbm_scores - vbm_min) / (vbm_max - vbm_min) if vbm_max > vbm_min else vbm_scores * 0.0
            
            heatmap_columns["VBM (Ground Truth)"] = vbm_norm
            aggregated_for_matrix["VBM (Ground Truth)"] = vbm_norm
            log.success("VBM (Ground Truth) processed successfully")
        except Exception as e:
            log.error(f"Failed to process VBM Ground Truth: {e}")
    
    log.info("Phase 2: Aggregating Python XAI maps...")
    methods_dict = {
        "SVM Raw Weights (Top 1%)": list(xai_maps_dir.glob("SVM_Raw_Weights_Fold_*_Top1.nii")), 
        "SVM Raw Weights (Top 5%)": list(xai_maps_dir.glob("SVM_Raw_Weights_Fold_*_Top5.nii")), 
        "SVM Haufe (Top 1%)": list(xai_maps_dir.glob("SVM_Haufe_Fold_*_Top1.nii")),
        "SVM Haufe (Top 5%)": list(xai_maps_dir.glob("SVM_Haufe_Fold_*_Top5.nii")),
        "SVM Gaonkar (FDR 0.1)": list(xai_maps_dir.glob("SVM_Gaonkar_Fold_*_fdr01.nii")),
        "SVM Gaonkar (Bonf 0.05)": list(xai_maps_dir.glob("SVM_Gaonkar_Fold_*_bonf005.nii")),
    }
    
    for method_name, paths_list in methods_dict.items():
        if not paths_list:
            log.warning(f"No maps found for {method_name}.")
            continue
            
        log.info(f"Processing: {method_name}")
        df_agg_real = analyzer.aggregate_and_normalize_maps(paths_list, str(atlas_path), str(atlas_csv_path), metric='Mean_ROI_Signal', use_absolute=False)
        
        if not df_agg_real.empty:
            safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "")
            
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
            
            # Save Fold CSVs dynamically 
            csv_out_path = results_dir / f"{safe_fold_name}_FeatureImportance.csv"
            csv_out_path.parent.mkdir(parents=True, exist_ok=True)
            df_fold.to_csv(str(csv_out_path), index=False)
            
            plot_out_path = f"Folds/Top20_AbsNetMean_{safe_fold_name}.png"
            plotter.plot_top_rois(df_fold, score_col='Abs_Mean_ROI_Signal', title=fold_name, filename=plot_out_path, top_k=20)
            
            fold_scores = df_fold.set_index('ROI_Name')['Abs_Mean_ROI_Signal']
            f_min, f_max = fold_scores.min(), fold_scores.max()
            fold_norm = (fold_scores - f_min) / (f_max - f_min) if f_max > f_min else fold_scores * 0.0
            
            fold_num = fold_name.split("_Fold_")[1].split("_")[0] if "_Fold_" in fold_name else fold_name[-1]
            heatmap_columns[f"{method_name} (F{fold_num})"] = fold_norm
    
    if len(aggregated_for_matrix) > 1:
        log.info("Phase 3: Building nDCG Correlation Matrix across all methods...")
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
    else:
        log.warning("Phase 3 Skipped: Not enough valid maps to compute nDCG Correlation Matrix (requires > 1).")

    if len(heatmap_columns) > 1:
        log.info("Phase 4: Building global Heatmap (all folds)...")
        heatmap_matrix = pd.DataFrame(heatmap_columns).fillna(0.0)
        
        if "VBM (Ground Truth)" in heatmap_matrix.columns:
            heatmap_matrix = heatmap_matrix.sort_values(by="VBM (Ground Truth)", ascending=False)
        else:
            heatmap_matrix = heatmap_matrix.sort_index()
            
        top_k_heat = 35
        if len(heatmap_matrix) > top_k_heat:
            heatmap_matrix = heatmap_matrix.head(top_k_heat)
            
        out_heat_path = "Heatmap_AllFolds.png"
        plotter.plot_heatmap(heatmap_matrix, out_heat_path, title_suffix="Net Impact |Mean|")
    else:
        log.warning("Phase 4 Skipped: Not enough valid maps to generate Heatmap (requires > 1).")

    log.success("--- XAI Comparison Completed ---")