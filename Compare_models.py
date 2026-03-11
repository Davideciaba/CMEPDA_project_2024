import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from nilearn import datasets, plotting, image
from loguru import logger

# ==========================================
# 1. Funzioni Matematiche per Analisi Voxel-wise
# ==========================================
def calculate_top_k_overlap(map1_data, map2_data, mask_data, k_percent=5.0):
    """
    Calcola la percentuale di overlap (Jaccard Index sui top K voxel) tra due mappe XAI.
    Isola i top K% voxel con i valori assoluti più alti all'interno della maschera GM.
    """
    m1_flat = np.abs(map1_data[mask_data > 0])
    m2_flat = np.abs(map2_data[mask_data > 0])
    
    k = max(1, int(len(m1_flat) * (k_percent / 100.0)))
    if k == 0: return 0.0
    
    top_idx1 = set(np.argsort(m1_flat)[-k:])
    top_idx2 = set(np.argsort(m2_flat)[-k:])
    
    overlap_count = len(top_idx1.intersection(top_idx2))
    return (overlap_count / k) * 100

def calculate_voxelwise_correlation(map1_data, map2_data, mask_data):
    """Calcola la correlazione spaziale di Pearson (Punto 5.1) tra due mappe."""
    m1_flat = map1_data[mask_data > 0]
    m2_flat = map2_data[mask_data > 0]
    
    # Previene errori se una mappa è completamente vuota (es. mascherata troppo severamente)
    if np.std(m1_flat) == 0 or np.std(m2_flat) == 0:
        return 0.0, 1.0
        
    corr, p_val = pearsonr(m1_flat, m2_flat)
    return corr, p_val

# ==========================================
# 2. Analisi per ROI (Regioni di Interesse)
# ==========================================
def extract_roi_summaries(maps_dict, atlas_img, atlas_labels, target_rois):
    """
    Calcola l'importanza media di diverse mappe XAI all'interno di specifiche ROI
    strutturali (es. Ippocampo) definite dall'atlante Harvard-Oxford.
    """
    atlas_data = atlas_img.get_fdata()
    results = []
    
    for roi_name in target_rois:
        # Trova l'indice della ROI nell'atlante
        roi_indices = [i for i, label in enumerate(atlas_labels) if roi_name.lower() in label.lower()]
        if not roi_indices:
            logger.warning(f"ROI '{roi_name}' non trovata nell'atlante.")
            continue
            
        roi_mask = np.isin(atlas_data, roi_indices)
        
        # Calcola il valore assoluto medio per ogni mappa in questa ROI
        row = {'ROI': roi_name}
        for map_name, map_data in maps_dict.items():
            row[f'Mean_Imp_{map_name}'] = np.mean(np.abs(map_data[roi_mask]))
        results.append(row)
        
    return pd.DataFrame(results)

# ==========================================
# 3. Funzioni di Visualizzazione e Analisi Performance
# ==========================================
def get_metrics_from_run(run_dir, run_label):
    """Estrae le metriche cercando prima nel summary CSV, con fallback sui singoli JSON."""
    
    # Metodo 1 (Nuovo e consigliato): Cerca il file aggregato in summary/
    summary_metrics_path = os.path.join(run_dir, "summary", "metrics_all_folds.csv")
    if os.path.exists(summary_metrics_path):
        df = pd.read_csv(summary_metrics_path)
        df['Model_Setup'] = run_label
        return df

    # Metodo 2 (Vecchio fallback): Cerca nelle cartelle dei singoli fold
    folds_dir = os.path.join(run_dir, "folds")
    if not os.path.exists(folds_dir): 
        return pd.DataFrame()
    
    rows = []
    for fold in os.listdir(folds_dir):
        met_path = os.path.join(folds_dir, fold, "metrics.json")
        if os.path.exists(met_path):
            with open(met_path, 'r') as f:
                met = json.load(f)
                met['Model_Setup'] = run_label
                met['Fold'] = fold
                rows.append(met)
                
    return pd.DataFrame(rows)

def plot_model_comparison(run_dirs_dict, title, out_path):
    """
    Genera un bar plot per confrontare le performance (Punto 5.2).
    Può essere usato per 'SVM vs EffNet' oppure per la Sensitivity Analysis 'Raw vs Res'.
    """
    dfs = []
    for label, d in run_dirs_dict.items():
        df = get_metrics_from_run(d, label)
        if not df.empty: dfs.append(df)
        
    if not dfs:
        logger.error(f"Nessun dato di metriche trovato per generare il plot: {title}")
        return
        
    df_all = pd.concat(dfs, ignore_index=True)
    metrics_to_plot = ['balanced_accuracy', 'sensitivity', 'specificity', 'auc']
    df_melt = df_all.melt(id_vars=['Model_Setup', 'Fold'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model_Setup', capsize=.1, errorbar='sd')
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.success(f"Plot salvato in: {out_path}")

def plot_xai_brain_slices(maps_dict, out_path):
    """Stampa visivamente le proiezioni ortogonali affiancate delle mappe XAI chiave."""
    # Filtriamo solo le mappe che ci interessa graficare
    keys_to_plot = ['VBM', 'SVM_Haufe', 'EffNet_IG']
    valid_keys = [k for k in keys_to_plot if k in maps_dict]
    
    fig, axes = plt.subplots(len(valid_keys), 1, figsize=(10, 4 * len(valid_keys)))
    cut_coords = (0, -20, -15) # Spazio MNI: centro ippocampo / temporale mediale
    
    for idx, key in enumerate(valid_keys):
        ax = axes[idx] if len(valid_keys) > 1 else axes
        plotting.plot_stat_map(maps_dict[key], display_mode='ortho', cut_coords=cut_coords, 
                               axes=ax, title=key, colorbar=True, cmap='cold_hot')
                               
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(f"Confronto mappe cerebrali salvato in: {out_path}")

# ==========================================
# 4. Pipeline Esecutiva Principale
# ==========================================
def main():
    logger.add("logs/comparison_pipeline.log", rotation="5 MB")
    logger.info("Avvio Pipeline di Confronto e Sensitivity Analysis (Punto 5)...")
    
    # --- CONFIGURAZIONE ---
    # Questi percorsi andranno aggiornati con le tue cartelle reali a fine addestramento
    DIR_SVM_RAW = "results/runs/run_1772536685_seed42_SVM"
    DIR_SVM_RES = "results/runs/run_1772537227_seed42_SVM"
    DIR_EFF_RES = "results/runs/run_1772536701_seed42_EfficientNet"
    
    OUT_DIR = "results/comparisons"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Mappe NIfTI aggregate (presumibilmente calcolate calcolando la media sui fold)
    # Assicurati di salvare queste mappe alla fine dell'addestramento
    MAP_PATHS = {
        'VBM': "data/maps/vbm_tstat_map.nii.gz",
        'SVM_Weights': "data/maps/svm_weights_mean.nii.gz",
        'SVM_Haufe': "data/maps/svm_haufe_mean.nii.gz",
        'SVM_Haufe_Masked': "data/maps/svm_haufe_masked_mean.nii.gz",
        'EffNet_IG': "data/maps/effnet_ig_mean.nii.gz",
        'EffNet_IG_Masked': "data/maps/effnet_ig_masked_mean.nii.gz",
        'Mask': "data/gm_mask_MNI.nii.gz"
    }

    # --- 1. SENSITIVITY ANALYSIS E CONFRONTO MODELLI (Punto 5.2) ---
    logger.info("Generazione plot Sensitivity Analysis (Raw vs Residuals)...")
    plot_model_comparison({'SVM_Raw_TIV': DIR_SVM_RAW, 'SVM_Residualized': DIR_SVM_RES}, 
                          "Sensitivity Analysis: Raw(TIV) vs Residui", 
                          os.path.join(OUT_DIR, "sensitivity_analysis_svm.png"))
                          
    logger.info("Generazione plot Confronto Modelli...")
    plot_model_comparison({'Linear SVM': DIR_SVM_RES, 'EfficientNet 3D': DIR_EFF_RES}, 
                          "Confronto Performance: SVM vs EfficientNet (sui Residui GM)", 
                          os.path.join(OUT_DIR, "performance_comparison.png"))

    # Verifica presenza file NIfTI
    missing_maps = [k for k, p in MAP_PATHS.items() if not os.path.exists(p)]
    if missing_maps:
        logger.warning(f"Le seguenti mappe NIfTI mancano, l'analisi spaziale verrà saltata: {missing_maps}")
        return

    # --- 2. CARICAMENTO DATI SPAZIALI ---
    logger.info("Caricamento mappe NIfTI...")
    mask_img = nib.load(MAP_PATHS['Mask'])
    mask_data = mask_img.get_fdata()
    
    imgs = {k: nib.load(v) for k, v in MAP_PATHS.items() if k != 'Mask'}
    
    # Allineamento rigoroso allo spazio della VBM
    vbm_img = imgs['VBM']
    for k in imgs:
        if k != 'VBM':
            imgs[k] = image.resample_to_img(imgs[k], vbm_img)
            
    data_arrays = {k: img.get_fdata() for k, img in imgs.items()}

    # --- 3. CONFRONTI VOXEL-WISE COMPLETI (Punto 5.1 e 5.2) ---
    logger.info("Calcolo metriche spaziali (Correlazioni e Overlap Top-5%)...")
    results = []
    
    comparisons_to_make = [
        ('VBM', 'SVM_Weights'),
        ('VBM', 'SVM_Haufe'),
        ('VBM', 'SVM_Haufe_Masked'),
        ('VBM', 'EffNet_IG'),
        ('VBM', 'EffNet_IG_Masked'),
        ('SVM_Haufe', 'EffNet_IG') # Confronto diretto XAI
    ]
    
    for m1, m2 in comparisons_to_make:
        corr, _ = calculate_voxelwise_correlation(data_arrays[m1], data_arrays[m2], mask_data)
        overlap = calculate_top_k_overlap(data_arrays[m1], data_arrays[m2], mask_data, k_percent=5.0)
        results.append({'Map_1': m1, 'Map_2': m2, 'Pearson_Corr': corr, 'Top_5%_Overlap': overlap})
        logger.debug(f"{m1} vs {m2} -> Corr: {corr:.3f} | Overlap: {overlap:.1f}%")
        
    df_spatial = pd.DataFrame(results)
    df_spatial.to_csv(os.path.join(OUT_DIR, "spatial_comparisons_summary.csv"), index=False)

    # --- 4. ANALISI ROI CON ATLANTE (Punto 5.1) ---
    logger.info("Avvio analisi ROI con atlante Harvard-Oxford...")
    atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    atlas_img = image.resample_to_img(nib.load(atlas.maps), vbm_img, interpolation='nearest')
    
    rois_of_interest = ['Hippocampus', 'Amygdala', 'Parahippocampal', 'Temporal Pole']
    df_roi = extract_roi_summaries(data_arrays, atlas_img, atlas.labels, rois_of_interest)
    df_roi.to_csv(os.path.join(OUT_DIR, "roi_importance_summary.csv"), index=False)
    logger.success("Riassunto Importanza per ROI salvato con successo.")

    # --- 5. PLOT GRAFICO CEREBRALE ---
    plot_xai_brain_slices(imgs, os.path.join(OUT_DIR, "xai_brain_slices.png"))
    
    logger.success("Pipeline di Confronto completata! Tutti i CSV e PNG sono salvati.")

if __name__ == "__main__":
    main()