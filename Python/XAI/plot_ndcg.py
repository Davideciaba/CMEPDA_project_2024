import sys
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imposta i path per importare i moduli custom
current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parents[2] 
sys.path.append(str(project_root / "Python"))

from utils.py_logger import CustomLogger
from XAI.roi_analyzer import ROIAnalyzer

class ROIPlotter:
    def __init__(self, logger: CustomLogger):
        self.logger = logger

    def plot_top_rois(self, df: pd.DataFrame, metric: str, top_k: int = 20, out_path: str = None):
        """
        Crea e mostra un grafico a barre orizzontali delle Top K regioni in base alla metrica scelta.
        """
        self.logger.info(f"Generazione del grafico per le Top {top_k} regioni basato su {metric}...")
        
        # Ordina il dataframe in modo decrescente per la metrica scelta e prendi le prime 'top_k'
        df_sorted = df.sort_values(by=metric, ascending=False).head(top_k)
        
        # Inverti l'ordine in modo che la barra più grande sia in alto nel grafico
        df_sorted = df_sorted.iloc[::-1]
        
        plt.figure(figsize=(12, 8))
        
        # Scegliamo un colore diverso a seconda della metrica
        color = 'skyblue' if 'Mean' in metric else 'lightcoral'
        
        plt.barh(df_sorted['ROI_Name'], df_sorted[metric], color=color, edgecolor='black')
        
        plt.xlabel(f'Valore {metric}')
        plt.ylabel('Regioni del Cervello (Neuromorphometrics)')
        plt.title(f'Top {top_k} Regioni - VBM ({metric})', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Salva l'immagine se è stato fornito un percorso
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            plt.savefig(out_path, dpi=300)
            self.logger.info(f"Grafico salvato in: {out_path}")
            
        # Mostra il grafico a schermo
        plt.show()


def run_vbm_roi_analysis():
    logger = CustomLogger(name="VBM_ROI_Analysis")
    logger.add_console_handler(level="DEBUG", use_colors=True)
    logger.info("--- Avvio Analisi ROI VBM ---")
    
    # Percorsi dei file
    map_path = project_root / "MATLAB" / "VBM_Pipeline" / "Results" / "Thresholded_Maps" / "TPM_Mask_FWE_corrected_map.nii"
    atlas_path = project_root / "AD_CTRL" / "labels_Neuromorphometrics.nii"
    atlas_csv_path = project_root / "AD_CTRL" / "spm_atlas_labels.csv"
    
    out_plot_mean_path = project_root / "Python" / "XAI" / "Plots" / "VBM_Top20_Mean.png"
    out_plot_sum_path = project_root / "Python" / "XAI" / "Plots" / "VBM_Top20_Sum.png"
    
    # Verifica che i file esistano
    if not map_path.exists():
        logger.critical(f"Errore: File della mappa non trovato in {map_path}")
        sys.exit(1)
    if not atlas_path.exists():
        logger.critical(f"Errore: File dell'atlante non trovato in {atlas_path}")
        sys.exit(1)
    if not atlas_csv_path.exists():
        logger.critical(f"Errore: File CSV dell'atlante non trovato in {atlas_csv_path}")
        sys.exit(1)
        
    analyzer = ROIAnalyzer(logger)
    plotter = ROIPlotter(logger)
    
    logger.info("Estrazione dei valori delle ROI dalla mappa VBM...")
    
    # Estrae l'importanza per regione dalla mappa VBM
    df_vbm = analyzer.extract_regional_importance(str(map_path), str(atlas_path), str(atlas_csv_path))
    
    # Salva i risultati in un CSV per sicurezza
    csv_out = project_root / "Python" / "XAI" / "Results" / "VBM_ROI_Scores.csv"
    os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
    df_vbm.to_csv(str(csv_out), index=False)
    logger.info(f"Risultati tabellari salvati in: {csv_out}")
    
    # Crea e mostra il grafico per la MEDIA
    plotter.plot_top_rois(df_vbm, metric='Mean_ROI_Signal', top_k=20, out_path=str(out_plot_mean_path))
    
    # Crea e mostra il grafico per la SOMMA
    plotter.plot_top_rois(df_vbm, metric='Sum_ROI_Signal', top_k=20, out_path=str(out_plot_sum_path))

    logger.success("--- Analisi Completata ---")

if __name__ == "__main__":
    run_vbm_roi_analysis()