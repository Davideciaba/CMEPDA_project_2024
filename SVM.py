# %% STEP 7: SVM ANALYSIS (LOOCV, FINAL MODEL, AND MAPPING)
# Author: Gemini
# Date: 09/10/2025
# --------------------------------------------------------------------------
# Questo script implementa l'intera pipeline di analisi per la SVM lineare
# come descritto nel documento di pipeline.
# SEZIONE 4: Esegue una LOOCV interna sul "remaining set" (80%) per il tuning.
# SEZIONE 5: Addestra il modello finale e lo valuta sul "hold-out set" (20%).
# SEZIONE 6: Aggrega i pesi in "aspetti" e crea una mappa NIfTI.
# --------------------------------------------------------------------------

import sys
import h5py
import numpy as np
from scipy.stats import mode
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger
from tqdm import tqdm
import joblib

# Assicurati di aver installato NiBabel: pip install nibabel
try:
    import nibabel as nib
except ImportError:
    logger.critical("Libreria NiBabel non trovata. Installala con: pip install nibabel")
    sys.exit(1)

# --- 0. Configurazione del Logger ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Definizione dei Percorsi dei File di Input ---
# NOTA: Questo script presuppone che i file .mat e .npz degli step precedenti siano disponibili.
mat_file_path = '../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat'
# Assumiamo di avere i dati splittati ma non ancora scalati, come da logica del documento
split_data_file = 'split_unscaled_data.npz' # !IMPORTANTE: questo file deve contenere X_rem e X_hold_test non scalati
clustering_results_file = 'hierarchical_clustering_results.npz'
# Usiamo un'immagine di riferimento per salvare il file .nii finale
reference_nii_path = '/data/GM_maps/wc1subj001.nii' # Sostituisci con un percorso valido

# --- 2. Caricamento Dati ---
logger.info("Caricamento di tutti i dati necessari dai file di preprocessing...")
try:
    # Carica i dati non scalati e gli indici
    with np.load(split_data_file) as data:
        X_rem = data['X_rem']
        y_rem = data['y_rem']
        X_hold_test = data['X_hold_test']
        y_hold_test = data['y_hold_test']
    
    # Carica i risultati del clustering per l'aggregazione finale
    with np.load(clustering_results_file) as data:
        cluster_labels_voxel = data['cluster_labels_voxel']
        K = int(data['num_clusters'])

    # Carica metadati per la ricostruzione della mappa 3D
    with h5py.File(mat_file_path, 'r') as f:
        mask3D = f['mask'][()]
        voxelIdx = f['voxelIdx'][()].ravel().astype(int) - 1 # 0-based

    n_rem, M = X_rem.shape
    logger.success("Tutti i file sono stati caricati con successo.")

except FileNotFoundError as e:
    logger.critical(f"File di input non trovato: {e}. Assicurati di aver eseguito tutti gli script precedenti.")
    sys.exit(1)

# ==========================================================================
# SEZIONE 4: LOOCV INTERNO SUL REMAINING SET (80%)
# ==========================================================================
logger.info("INIZIO SEZIONE 4: Esecuzione del LOOCV interno per il tuning di C...")

W_all_rem = np.zeros((n_rem, M))
bestC_int_all = np.zeros(n_rem)
C_candidates = [0.01, 0.1, 1, 10]

for k in tqdm(range(n_rem), desc="SVM Internal LOOCV"):
    
    # Definizione di Train e Validation Interno
    val_idx = k
    train_indices = np.delete(np.arange(n_rem), k)

    X_train_int_raw = X_rem[train_indices, :]
    y_train_int = y_rem[train_indices]
    X_val_int_raw = X_rem[val_idx, :].reshape(1, -1)
    y_val_int = y_rem[val_idx]

    # Standardizzazione basata SOLO sul train interno
    scaler_int = StandardScaler()
    X_train_int_std = scaler_int.fit_transform(X_train_int_raw)
    X_val_int_std = scaler_int.transform(X_val_int_raw)
    
    # Tuning di C
    best_c_int, best_val_acc_int = C_candidates[0], -1
    for c_value in C_candidates:
        svm_c = SVC(kernel='linear', C=c_value, probability=True, random_state=42)
        svm_c.fit(X_train_int_std, y_train_int)
        val_acc = svm_c.score(X_val_int_std, [y_val_int])
        if val_acc > best_val_acc_int:
            best_val_acc_int = val_acc
            best_c_int = c_value
    bestC_int_all[k] = best_c_int
    
    # Rifit su Train + Validation interno con il C ottimale
    X_trval_int_std = np.vstack([X_train_int_std, X_val_int_std])
    y_trval_int = np.append(y_train_int, y_val_int)
    
    svm_int_final = SVC(kernel='linear', C=best_c_int, probability=True, random_state=42)
    svm_int_final.fit(X_trval_int_std, y_trval_int)
    
    W_all_rem[k, :] = svm_int_final.coef_.flatten()

logger.success("SEZIONE 4 COMPLETATA: LOOCV interno terminato.")

# ==========================================================================
# SEZIONE 5: MODELLO FINALE E VALUTAZIONE SULL'HOLD-OUT SET (20%)
# ==========================================================================
logger.info("INIZIO SEZIONE 5: Addestramento del modello finale e valutazione.")

# 5.1.1 Scelta di C_final (la moda dei C trovati)
C_final = mode(bestC_int_all, keepdims=True).mode[0]
logger.info(f"Iperparametro finale scelto (moda dei C trovati): C = {C_final}")

# 5.1.2 Standardizzazione finale (fit su tutto X_rem)
logger.info("Standardizzazione finale: fit su tutto il remaining set (80%), transform su entrambi.")
scaler_final = StandardScaler()
X_rem_std = scaler_final.fit_transform(X_rem)
X_hold_std = scaler_final.transform(X_hold_test)

# 5.1.3 Addestramento SVM finale
logger.info(f"Addestramento del modello SVM finale su {n_rem} soggetti...")
svm_final_hold = SVC(kernel='linear', C=C_final, probability=True, random_state=42)
svm_final_hold.fit(X_rem_std, y_rem)
w_final_hold = svm_final_hold.coef_.flatten()

# 5.1.4 Predizione e valutazione su Hold-Out set
logger.info(f"Valutazione del modello finale su {len(y_hold_test)} soggetti del hold-out set...")
y_prob_svm_hold = svm_final_hold.predict_proba(X_hold_std)[:, 1]
y_pred_svm_hold = (y_prob_svm_hold >= 0.5).astype(int)

accuracy_hold = accuracy_score(y_hold_test, y_pred_svm_hold)
auc_hold = roc_auc_score(y_hold_test, y_prob_svm_hold)
logger.success(f"Performance su Hold-Out Set: Accuracy = {accuracy_hold:.4f}, AUC = {auc_hold:.4f}")

logger.success("SEZIONE 5 COMPLETATA: Modello finale addestrato e valutato.")

# ==========================================================================
# SEZIONE 6: AGGREGAZIONE DEI PESI IN ASPETTI E MAPPA NIfTI
# ==========================================================================
logger.info("INIZIO SEZIONE 6: Aggregazione dei pesi e creazione della mappa NIfTI.")

# 6.1 Aggregazione dei pesi SVM finali in "aspetti"
logger.info(f"Aggregazione dei {M} pesi voxel-wise in {K} aspetti...")
W_bar_hold = np.zeros(K)
for a in range(1, K + 1):
    vox_indices_in_aspect = np.where(cluster_labels_voxel == a)[0]
    if len(vox_indices_in_aspect) > 0:
        W_bar_hold[a - 1] = np.sum(w_final_hold[vox_indices_in_aspect])

# 6.2 Creazione della mappa NIfTI
logger.info("Creazione della mappa 3D dei pesi aggregati...")
# Crea un vettore flat con M elementi, dove ogni voxel ha il valore dell'aspetto a cui appartiene
flat_map = W_bar_hold[cluster_labels_voxel - 1] # -1 per passare da 1-based a 0-based

# Crea un volume 3D vuoto e inserisci i valori dei voxel attivi
W_map_hold_3D = np.zeros(mask3D.shape, dtype=np.float32)
W_map_hold_3D[mask3D] = flat_map

# Salvataggio del file NIfTI usando un'immagine di riferimento per l'header
try:
    ref_img = nib.load(reference_nii_path)
    affine = ref_img.affine
    header = ref_img.header
    
    nifti_image = nib.Nifti1Image(W_map_hold_3D, affine, header)
    output_nii_file = 'SVM_weight_map_aspects_holdout.nii'
    nib.save(nifti_image, output_nii_file)
    logger.success(f"Mappa NIfTI salvata con successo in: '{output_nii_file}'")
except FileNotFoundError:
    logger.error(f"File NIfTI di riferimento non trovato in '{reference_nii_path}'. Impossibile salvare la mappa.")

logger.success("SEZIONE 6 COMPLETATA: Pipeline SVM terminata.")