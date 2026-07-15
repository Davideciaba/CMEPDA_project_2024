import sys
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parents[2] 
sys.path.append(str(project_root / "Python"))

from utils.py_logger import CustomLogger
from XAI.roi_analyzer import ROIAnalyzer

class XAIPlotter:
    def __init__(self, logger: CustomLogger):
        self.logger = logger

    def plot_top_rois(self, df: pd.DataFrame, score_col: str, title: str, out_path: str, top_k: int = 20):
        """
        Crea un grafico a barre orizzontali delle Top K regioni (solo valori assoluti/magnitudo).
        """
        self.logger.info(f"Generazione del grafico Top {top_k} per {title}...")
        
        df_sorted = df.sort_values(by=score_col, ascending=False).head(top_k)
        df_sorted = df_sorted.iloc[::-1]
        
        plt.figure(figsize=(12, 8))
        color = 'lightcoral'
        
        plt.barh(df_sorted['ROI_Name'], df_sorted[score_col], color=color, edgecolor='black')
        
        plt.xlabel('Feature Importance (Sum of Absolute Values)')
        plt.ylabel('Brain Regions (Gray Matter)')
        plt.title(f'Top {top_k} Regions - {title}', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_diverging_bars(self, df: pd.DataFrame, score_col: str, title: str, out_path: str, top_k: int = 20):
        """
        Crea un grafico a barre divergenti usando la MEDIA grezza (Mean ROI) per mostrare l'impatto direzionale.
        """
        self.logger.info(f"Generazione Diverging Bar Chart per {title}...")
        
        # Ordina per valore assoluto della media per trovare le Top K più importanti direzionalmente
        df['Abs_Score'] = df[score_col].abs()
        df_sorted = df.sort_values(by='Abs_Score', ascending=False).head(top_k)
        
        df_sorted = df_sorted.iloc[::-1]
        
        plt.figure(figsize=(12, 8))
        
        # Colora di rosso se positivo, blu se negativo
        colors = ['indianred' if x > 0 else 'steelblue' for x in df_sorted[score_col]]
        
        plt.barh(df_sorted['ROI_Name'], df_sorted[score_col], color=colors, edgecolor='black')
        plt.axvline(0, color='black', linewidth=1)
        
        plt.xlabel('Mean ROI Weight (Directional Impact on Classification Score)')
        plt.ylabel('Brain Regions (Gray Matter)')
        plt.title(f'Top {top_k} Directional Impact - {title}', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_bloch_style_heatmap(self, df_matrix: pd.DataFrame, out_path: str, title_suffix: str = ""):
        """
        Genera la Heatmap di Bloch usando la Feature Importance (Sum of Absolute Values).
        """
        self.logger.info(f"Generazione della Heatmap Comparativa (Stile Bloch)...")
        
        plt.figure(figsize=(16, 14))
        
        sns.heatmap(df_matrix, cmap='Blues', annot=False, 
                    cbar_kws={'label': 'Normalized Feature Importance (Abs Sum)'},
                    linewidths=.5, linecolor='lightgray')
        
        plt.title(f'Global Feature Importances across Models ({title_suffix})', fontsize=16, pad=20)
        plt.ylabel('Aspects and Features (Regions of Interest)', fontsize=14)
        plt.xlabel('Models and Explanation Methods', fontsize=14)
        
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plt.xticks(rotation=45, ha='left')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
        self.logger.success(f"Heatmap salvata in: {out_path}")

    def plot_ndcg_curves(self, ndcg_data: dict, k_values: list, out_path: str, title_suffix: str):
        """
        Plotta le curve nDCG calcolate sulla Feature Importance (Sum of Absolute Values).
        """
        self.logger.info("Generazione grafico nDCG...")
        plt.figure(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'D', 'v', '<']
        
        for i, (method_name, avg_ndcgs) in enumerate(ndcg_data.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(k_values, avg_ndcgs, color=color, marker=marker, linewidth=2, label=method_name)
                
        plt.title(f'nDCG@K vs VBM Ground Truth ({title_suffix})', fontsize=14)
        plt.xlabel('Top K Regions (K)', fontsize=12)
        plt.ylabel('nDCG Score (based on Absolute Sum)', fontsize=12)
        plt.xticks(k_values)
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.success(f"nDCG Plot salvato in: {out_path}")

    def plot_ndcg_matrix(self, ndcg_matrix: pd.DataFrame, out_path: str, title_suffix: str = ""):
        """
        Genera una matrice di correlazione nDCG tra tutti i metodi basata sulla Feature Importance.
        """
        self.logger.info("Generazione della Matrice nDCG...")
        
        plt.figure(figsize=(10, 8))
        
        ax = sns.heatmap(ndcg_matrix, cmap='Blues', annot=True, fmt=".2f", vmin=0.0, vmax=1.0,
                         cbar_kws={'label': 'nDCG Score'}, linewidths=.5, linecolor='lightgray')
        
        plt.title(f'nDCG Similarity Matrix ({title_suffix})', fontsize=16, pad=20)
        plt.ylabel('Reference Method', fontsize=12)
        plt.xlabel('Comparison Method', fontsize=12)
        
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xticks(rotation=45, ha='left')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
        self.logger.success(f"Matrice nDCG salvata in: {out_path}")


def run_comparative_pipeline():
    logger = CustomLogger(name="Comparative_XAI")
    logger.add_console_handler(level="DEBUG", use_colors=True)
    logger.info("--- Booting Comparative XAI Pipeline ---")
    
    vbm_path = project_root / "MATLAB" / "VBM_Pipeline" / "Results" / "Thresholded_Maps" / "TPM_Mask_FWE_corrected_map.nii"
    atlas_path = project_root / "AD_CTRL" / "labels_Neuromorphometrics.nii"
    atlas_csv_path = project_root / "AD_CTRL" / "spm_atlas_labels.csv"
    
    svm_results_dir = project_root / "Python" / "SVM_Pipeline" / "Results"
    output_dir = project_root / "Python" / "XAI" / "Plots"
    
    analyzer = ROIAnalyzer(logger)
    plotter = XAIPlotter(logger)
    
    if not vbm_path.exists():
        logger.critical(f"File VBM Ground Truth non trovato: {vbm_path}")
        return
        
    # --- Definizione Metriche secondo la nuova logica ---
    # Direzionalità (Diverging Bars): Media dei pesi grezzi
    metric_dir = 'Mean_ROI_Signal'
    # Importanza globale (Heatmap & nDCG): Somma dei valori assoluti
    metric_imp = 'Sum_ROI_Signal'
    k_values = [5, 10, 15, 20, 25, 30]

    logger.info("Estrazione importanza ROI dalla VBM (Ground Truth)...")
    # La VBM viene valutata come Ground Truth di Importanza Globale (Somma Abs)
    df_vbm_abs = analyzer.extract_regional_importance(str(vbm_path), str(atlas_path), str(atlas_csv_path), use_absolute=True)
    vbm_scores = df_vbm_abs.set_index('ROI_Name')[metric_imp]
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
    ndcg_data_avg = {}
    
    for method_name, paths_list in methods_dict.items():
        if not paths_list:
            logger.warning(f"Nessuna mappa trovata per {method_name}.")
            continue
            
        logger.info(f"Elaborazione: {method_name}")
        ndcg_data_avg[method_name] = {}
        
        # --- 1. DIVERGING BAR PLOT (Mean ROI, Segni mantenuti) ---
        df_agg_real = analyzer.aggregate_and_normalize_maps(paths_list, str(atlas_path), str(atlas_csv_path), metric=metric_dir, use_absolute=False)
        if not df_agg_real.empty:
            safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "")
            div_out_path = output_dir / "Directional" / f"Diverging_Mean_{safe_name}.png"
            plotter.plot_diverging_bars(df_agg_real, score_col=metric_dir, title=method_name, out_path=str(div_out_path), top_k=20)

        # --- 2. PREPARAZIONE HEATMAP E NDCG MATRICE (Somma Valori Assoluti) ---
        df_agg_abs = analyzer.aggregate_and_normalize_maps(paths_list, str(atlas_path), str(atlas_csv_path), metric=metric_imp, use_absolute=True)
        if not df_agg_abs.empty:
            aggregated_for_matrix[method_name] = df_agg_abs.set_index('ROI_Name')['Normalized_Importance']

        # --- 3. ANALISI DEI SINGOLI FOLD (Somma Valori Assoluti) ---
        fold_ndcgs = []
        for fold_path in paths_list:
            fold_name = fold_path.stem 
            
            # Estrazione ROI per il singolo fold (in valore assoluto per Feature Importance)
            df_fold_abs = analyzer.extract_regional_importance(str(fold_path), str(atlas_path), str(atlas_csv_path), use_absolute=True)
            if df_fold_abs.empty: continue
            
            # Calcolo nDCG contro la VBM
            ndcg_scores = [analyzer.compare_maps_ndcg(df_fold_abs, df_vbm_abs, metric=metric_imp, k=k) for k in k_values]
            fold_ndcgs.append(ndcg_scores)
            
            # Salvataggio Top 20 per il singolo fold
            safe_fold_name = fold_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "")
            csv_out_path = project_root / "Python" / "XAI" / "Results" / "Folds" / f"{safe_fold_name}_FeatureImportance.csv"
            os.makedirs(os.path.dirname(os.path.abspath(csv_out_path)), exist_ok=True)
            df_fold_abs.to_csv(str(csv_out_path), index=False)
            
            plot_out_path = project_root / "Python" / "XAI" / "Plots" / "Folds" / f"Top20_AbsSum_{safe_fold_name}.png"
            plotter.plot_top_rois(df_fold_abs, score_col=metric_imp, title=fold_name, out_path=str(plot_out_path), top_k=20)
            
            # Normalizzazione (0-1) e aggiunta colonna per la Heatmap grande
            fold_scores = df_fold_abs.set_index('ROI_Name')[metric_imp]
            f_min, f_max = fold_scores.min(), fold_scores.max()
            fold_norm = (fold_scores - f_min) / (f_max - f_min) if f_max > f_min else fold_scores * 0.0
            
            # Estrae solo il numero del fold dal nome file (es. SVM_Haufe_Fold_3_Top1 -> 3)
            fold_num = fold_name.split("_Fold_")[1].split("_")[0] if "_Fold_" in fold_name else fold_name[-1]
            col_name = f"{method_name} (F{fold_num})"
            heatmap_columns[col_name] = fold_norm
    
        # Media degli nDCG dei fold per il plot a linee
        if fold_ndcgs:
            ndcg_data_avg[method_name] = np.mean(fold_ndcgs, axis=0)
    
    # =====================================================================
    # GENERAZIONE GRAFICI GLOBALI (Basati su Feature Importance / Abs Sum)
    # =====================================================================
    
    # 1. Plot nDCG a linee (Fig 8)
    if ndcg_data_avg:
        out_ndcg_path = output_dir / "nDCG_FeatureImportance_Curves.png"
        plotter.plot_ndcg_curves(ndcg_data_avg, k_values, str(out_ndcg_path), title_suffix="Feature Importance")
        
    # 2. Matrice nDCG (Fig 8 heatmap)
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
                
        out_matrix_path = output_dir / "nDCG_FeatureImportance_Correlation_Matrix.png"
        plotter.plot_ndcg_matrix(ndcg_matrix, str(out_matrix_path), title_suffix=f"Feature Importance (K={full_k})")
    
    # 3. Heatmap Bloch (Fig 7)
    if len(heatmap_columns) > 1:
        logger.info("Costruzione della grande Heatmap di Bloch (Tutti i Fold)...")
        heatmap_matrix = pd.DataFrame(heatmap_columns).fillna(0.0)
        
        # Ordina le righe sempre in base alla VBM
        heatmap_matrix = heatmap_matrix.sort_values(by="VBM (Ground Truth)", ascending=False)
        
        # Mostra solo le Top 35
        top_k_heat = 35
        if len(heatmap_matrix) > top_k_heat:
            heatmap_matrix = heatmap_matrix.head(top_k_heat)
            
        out_heat_path = output_dir / "Bloch_Heatmap_FeatureImportance_AllFolds.png"
        plt.rcParams['figure.figsize'] = (16, 14) 
        plotter.plot_bloch_style_heatmap(heatmap_matrix, str(out_heat_path), title_suffix="Feature Importance")
        
    logger.success("--- Pipeline Comparativa Completata ---")

if __name__ == "__main__":
    run_comparative_pipeline()