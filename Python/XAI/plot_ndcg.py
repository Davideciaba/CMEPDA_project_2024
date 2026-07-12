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
        Crea e salva un grafico a barre orizzontali delle Top K regioni per un singolo metodo.
        """
        self.logger.info(f"Generazione del grafico Top {top_k} per {title}...")
        
        # Ordina e prendi le prime K
        df_sorted = df.sort_values(by=score_col, ascending=False).head(top_k)
        df_sorted = df_sorted.iloc[::-1] # Inverti per avere la barra più grande in alto
        
        plt.figure(figsize=(12, 8))
        color = 'skyblue' if 'Mean' in title else 'lightcoral'
        
        plt.barh(df_sorted['ROI_Name'], df_sorted[score_col], color=color, edgecolor='black')
        
        plt.xlabel('Normalized Importance (0-1)')
        plt.ylabel('Regioni del Cervello (Materia Grigia)')
        plt.title(f'Top {top_k} Regioni - {title}', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_bloch_style_heatmap(self, df_matrix: pd.DataFrame, out_path: str, title_suffix: str = ""):
        """
        Genera una heatmap stile Bloch per confrontare vari metodi.
        df_matrix: DataFrame con indici=ROI_Name, colonne=Nomi Metodi, valori=Importanza Normalizzata (0-1).
        """
        self.logger.info(f"Generazione della Heatmap Comparativa (Stile Bloch) - {title_suffix}...")
        
        plt.figure(figsize=(10, 14))
        
        # Usa 'Blues' come nell'articolo per indicare l'importanza
        sns.heatmap(df_matrix, cmap='Blues', annot=False, 
                    cbar_kws={'label': 'Normalized Feature Importance'},
                    linewidths=.5, linecolor='lightgray')
        
        plt.title(f'Global Feature Importances across Models ({title_suffix})', fontsize=14, pad=15)
        plt.ylabel('Aspects and Features (Regions of Interest)', fontsize=12)
        plt.xlabel('Models and Explanation Methods', fontsize=12)
        
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plt.xticks(rotation=45, ha='left')
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
        self.logger.success(f"Heatmap salvata in: {out_path}")

def run_comparative_pipeline():
    logger = CustomLogger(name="Comparative_XAI")
    logger.add_console_handler(level="DEBUG", use_colors=True)
    logger.info("--- Booting Comparative XAI Pipeline ---")
    
    # Percorsi hardcoded relativi alla root del progetto
    atlas_path = project_root / "AD_CTRL" / "labels_Neuromorphometrics.nii"
    atlas_csv_path = project_root / "AD_CTRL" / "spm_atlas_labels.csv"
    
    # Aggiornato per puntare alla cartella contenente i risultati SVM reali
    svm_results_dir = project_root / "Python" / "SVM_Pipeline" / "Results"
    
    analyzer = ROIAnalyzer(logger)
    plotter = XAIPlotter(logger)
    
    # ========================================================================
    # DIZIONARIO DELLE MAPPE
    # Cerca in automatico i fold per ogni metodo e soglia grazie a .glob()
    # I nomi delle chiavi diventeranno le colonne della Heatmap (es. VBM, Haufe Top1, ecc.)
    # ========================================================================
    maps_to_aggregate = {
        "VBM (Ground Truth)": [project_root / "MATLAB" / "VBM_Pipeline" / "Results" / "Thresholded Maps" / "TPM_Mask_FWE_corrected_map.nii"],
        "SVM Haufe (Top 1%)": list(svm_results_dir.glob("SVM_Haufe_Fold_*_Top1.nii")),
        "SVM Haufe (Top 5%)": list(svm_results_dir.glob("SVM_Haufe_Fold_*_Top5.nii")),
        "SVM Gaonkar (FDR 0.1)": list(svm_results_dir.glob("SVM_Gaonkar_Fold_*_fdr01.nii")),
        "SVM Gaonkar (Bonf 0.05)": list(svm_results_dir.glob("SVM_Gaonkar_Fold_*_bonf005.nii")),
    }
    
    metrics_to_plot = ['Mean_ROI_Signal', 'Sum_ROI_Signal']
    
    for metric in metrics_to_plot:
        logger.info(f"=== Creazione matrice per la metrica: {metric} ===")
        aggregated_results = {}
        
        for method_name, paths_list in maps_to_aggregate.items():
            if not paths_list:
                logger.warning(f"Nessuna mappa trovata per {method_name}. Controlla i nomi dei file o l'estensione.")
                continue
                
            # Estrae, aggrega e normalizza per ogni metodo
            norm_df = analyzer.aggregate_and_normalize_maps(paths_list, str(atlas_path), str(atlas_csv_path), metric=metric)
            
            if not norm_df.empty:
                # Creiamo un nome file sicuro rimuovendo spazi e parentesi dal nome del metodo
                safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "")
                
                # 1. Salva i risultati in un CSV per ogni metodo
                csv_out_path = project_root / "Python" / "XAI" / "Results" / f"{safe_name}_{metric}_Scores.csv"
                os.makedirs(os.path.dirname(os.path.abspath(csv_out_path)), exist_ok=True)
                norm_df.to_csv(str(csv_out_path), index=False)
                logger.info(f"Punteggi salvati in: {csv_out_path.name}")
                
                # 2. Genera il grafico a barre delle Top 20
                plot_out_path = project_root / "Python" / "XAI" / "Plots" / f"Top20_{safe_name}_{metric}.png"
                plotter.plot_top_rois(
                    df=norm_df, 
                    score_col='Normalized_Importance', 
                    title=f"{method_name} ({metric})", 
                    out_path=str(plot_out_path), 
                    top_k=20
                )
                
                # Prepara il dataframe per il join della Heatmap
                norm_df = norm_df.set_index('ROI_Name')
                norm_df = norm_df.rename(columns={'Normalized_Importance': method_name})
                aggregated_results[method_name] = norm_df
                
        if not aggregated_results:
            logger.critical(f"Nessun dato valido processato per {metric}. Salto il plot.")
            continue

        # Inner join su tutte le colonne dei metodi
        heatmap_matrix = pd.concat(aggregated_results.values(), axis=1)
        heatmap_matrix = heatmap_matrix.fillna(0.0)

        # Ordinamento basato sulla VBM se esiste, altrimenti sulla media
        if "VBM (Ground Truth)" in heatmap_matrix.columns:
            heatmap_matrix = heatmap_matrix.sort_values(by="VBM (Ground Truth)", ascending=False)
        else:
            heatmap_matrix['Row_Mean'] = heatmap_matrix.mean(axis=1)
            heatmap_matrix = heatmap_matrix.sort_values(by="Row_Mean", ascending=False).drop(columns=['Row_Mean'])

        # Top K
        top_k = 35
        if len(heatmap_matrix) > top_k:
            logger.info(f"Taglio la matrice alle Top {top_k} regioni per una visualizzazione ottimale.")
            heatmap_matrix = heatmap_matrix.head(top_k)

        # Plotting
        out_plot_path = project_root / "Python" / "XAI" / "Plots" / f"Bloch_Heatmap_Comparison_{metric}.png"
        plotter.plot_bloch_style_heatmap(heatmap_matrix, str(out_plot_path), title_suffix=metric)
        
    logger.success("--- Pipeline Comparativa Completata ---")

if __name__ == "__main__":
    run_comparative_pipeline()