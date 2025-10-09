# %% STEP 4: HIERARCHICAL CLUSTERING AND VOXEL-LEVEL MAPPING
# Author: Gemini
# Date: 08/10/2025
# --------------------------------------------------------------------------
# Questo script esegue un clustering gerarchico sui dati dei super-voxel
# (già in formato float32) per identificare gli "aspetti" e mappa
# le etichette finali di nuovo a livello dei singoli voxel.

import sys
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from loguru import logger
import joblib

# --- 0. Configurazione del Logger ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Definizione dei Parametri e Percorsi ---
input_data_file = 'supervoxel_data.npz'
output_results_file = 'hierarchical_clustering_results.npz'
output_linkage_file = 'linkage_matrix.joblib'
HIERARCHICAL_CUT_THRESHOLD = 0.5

# --- 2. Caricamento dei Dati dei Super-voxel ---
logger.info(f"Caricamento dei dati dei super-voxel da '{input_data_file}'...")
try:
    with np.load(input_data_file) as data:
        X_rem_sv = data['X_rem_sv']
        supervoxel_labels = data['supervoxel_labels']
    # Logga le informazioni, incluso il tipo di dati per conferma
    logger.success("Dati caricati con successo.")
    logger.info(f"Matrice X_rem_sv: {X_rem_sv.shape} | Tipo Dati: {X_rem_sv.dtype}")
    logger.info(f"Voxel-to-Supervoxel map: {supervoxel_labels.shape}")

except FileNotFoundError:
    logger.critical(f"File '{input_data_file}' non trovato. Eseguire prima lo script 'create_supervoxels.py'.")
    sys.exit(1)

# --- 3. Calcolo della Matrice di Correlazione di Spearman ---
logger.info(f"Calcolo della matrice di correlazione di Spearman tra {X_rem_sv.shape[1]} super-voxel...")

# ---- SEMPLIFICAZIONE ----
# Non è più necessaria la conversione a float32 perché i dati sono già nel formato corretto.
# Si passa direttamente l'array caricato alla funzione.
corr_mat, _ = spearmanr(X_rem_sv)
# -------------------------

logger.success(f"Matrice di correlazione calcolata. Dimensioni: {corr_mat.shape}")

# --- 4. Costruzione della Matrice di Distanza ---
logger.info("Conversione della matrice di correlazione in matrice di distanza (1 - |corr|)...")
dist_mat = 1 - np.abs(corr_mat)
logger.success(f"Matrice di distanza creata. Dimensioni: {dist_mat.shape}")

# --- 5. Esecuzione del Clustering Gerarchico ---
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

# --- 6. Estensione delle Etichette a Livello Voxel ---
logger.info("Assegnazione delle etichette finali a ogni singolo voxel...")
cluster_labels_voxel = cluster_labels_sv[supervoxel_labels]
logger.success(f"Mappatura a livello voxel completata. Creato array di dimensioni: {cluster_labels_voxel.shape}")

# --- 7. Salvataggio dei Risultati ---
logger.info(f"Salvataggio dei risultati in '{output_results_file}' e '{output_linkage_file}'...")
np.savez_compressed(output_results_file,
                    cluster_labels_supervoxel=cluster_labels_sv,
                    cluster_labels_voxel=cluster_labels_voxel,
                    num_clusters=K)

joblib.dump(Z, output_linkage_file)
logger.success("Script completato con successo!")