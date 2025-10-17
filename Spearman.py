# %% STEP 4: HIERARCHICAL CLUSTERING AND VOXEL-LEVEL MAPPING
# Author: Gemini
# Date: 08/10/2025
# --------------------------------------------------------------------------
# Questo script esegue un clustering gerarchico sui dati dei super-voxel
# per identificare gli "aspetti" e mappa le etichette finali di nuovo
# a livello dei singoli voxel. La correlazione di Spearman è calcolata
# con PyTorch per sfruttare l'accelerazione CUDA.

import sys
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from loguru import logger
import joblib
import torch

# --- 0. Configurazione del Logger ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Definizione dei Parametri e Percorsi ---
input_data_file = 'supervoxel_data.npz'
output_results_file = 'hierarchical_clustering_results.npz'
output_linkage_file = 'linkage_matrix.joblib'
HIERARCHICAL_CUT_THRESHOLD = 0.5

# --- 2. Configurazione del dispositivo PyTorch ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Utilizzo del dispositivo: {device}")

# --- 3. Funzione Correlazione di Spearman con PyTorch ---
def spearman_corr_pytorch(x):
    """
    Calcola la matrice di correlazione di Spearman usando PyTorch.
    x: tensore 2D di shape (n_samples, n_features)
    """
    # Calcola i ranghi per ogni feature (colonna)
    x_rank = x.argsort(dim=0).argsort(dim=0).float()
    
    # Calcola la correlazione di Pearson sui ranghi
    n = x_rank.size(0)
    
    # Sottrai la media
    x_mean = torch.mean(x_rank, dim=0)
    x_centered = x_rank - x_mean
    
    # Calcola la covarianza
    cov = (x_centered.t() @ x_centered) / (n - 1)
    
    # Calcola la deviazione standard
    std = torch.std(x_rank, dim=0)
    std_matrix = torch.outer(std, std)
    
    # Calcola la correlazione
    corr_mat = cov / std_matrix
    corr_mat = torch.nan_to_num(corr_mat, 0) # Gestisce divisioni per zero
    
    return corr_mat

# --- 4. Caricamento dei Dati dei Super-voxel ---
logger.info(f"Caricamento dei dati dei super-voxel da '{input_data_file}'...")
try:
    with np.load(input_data_file) as data:
        X_rem_sv = data['X_rem_sv']
        supervoxel_labels = data['supervoxel_labels']
    logger.success("Dati caricati con successo.")
    logger.info(f"Matrice X_rem_sv: {X_rem_sv.shape} | Tipo Dati: {X_rem_sv.dtype}")
    logger.info(f"Voxel-to-Supervoxel map: {supervoxel_labels.shape}")

except FileNotFoundError:
    logger.critical(f"File '{input_data_file}' non trovato. Eseguire prima lo script 'create_supervoxels.py'.")
    sys.exit(1)

# --- 5. Calcolo della Matrice di Correlazione di Spearman (PyTorch) ---
logger.info(f"Calcolo della matrice di correlazione di Spearman tra {X_rem_sv.shape[1]} super-voxel (PyTorch)...")

# Converte i dati in tensori e li sposta sul dispositivo
X_rem_sv_tensor = torch.from_numpy(X_rem_sv).to(device)

# Calcola la correlazione
corr_mat_tensor = spearman_corr_pytorch(X_rem_sv_tensor)

# Riporta il risultato sulla CPU come array NumPy
corr_mat = corr_mat_tensor.cpu().numpy()

logger.success(f"Matrice di correlazione calcolata. Dimensioni: {corr_mat.shape}")

# --- 6. Costruzione della Matrice di Distanza ---
logger.info("Conversione della matrice di correlazione in matrice di distanza (1 - |corr|)...")

# --- FIX: Aggiungi questa riga per "clippare" i valori ---
# Forza i valori della matrice di correlazione a rimanere nell'intervallo [-1, 1]
# per prevenire errori di calcolo dovuti a imprecisioni numeriche.
np.clip(corr_mat, -1.0, 1.0, out=corr_mat)
# -----------------------------------------------------------

dist_mat = 1 - np.abs(corr_mat)
logger.success(f"Matrice di distanza creata. Dimensioni: {dist_mat.shape}")

# --- 7. Esecuzione del Clustering Gerarchico ---
logger.info("Esecuzione del clustering gerarchico...")
logger.info("Condensamento della matrice di distanza nel formato vettoriale...")
condensed_dist = squareform(dist_mat, checks=False)

logger.info("Calcolo della matrice di linkage con metodo 'average'...")
Z = linkage(condensed_dist, method='average')
logger.success("Calcolo della matrice di linkage completato.")

logger.info(f"Formazione dei cluster tagliando il dendrogramma alla soglia H = {HIERARCHICAL_CUT_THRESHOLD}...")
cluster_labels_sv = fcluster(Z, HIERARCHICAL_CUT_THRESHOLD, criterion='distance')
K = cluster_labels_sv.max()
logger.success(f"Trovati K = {K} cluster (aspetti) a livello di super-voxel.")

# --- 8. Estensione delle Etichette a Livello Voxel ---
logger.info("Assegnazione delle etichette finali a ogni singolo voxel...")
cluster_labels_voxel = cluster_labels_sv[supervoxel_labels]
logger.success(f"Mappatura a livello voxel completata. Creato array di dimensioni: {cluster_labels_voxel.shape}")

# --- 9. Salvataggio dei Risultati ---
logger.info(f"Salvataggio dei risultati in '{output_results_file}' e '{output_linkage_file}'...")
np.savez_compressed(output_results_file,
                    cluster_labels_supervoxel=cluster_labels_sv,
                    cluster_labels_voxel=cluster_labels_voxel,
                    num_clusters=K)

joblib.dump(Z, output_linkage_file)
logger.success("Script completato con successo!")