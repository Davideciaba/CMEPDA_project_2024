# %% STEP 3: SUPER-VOXEL CLUSTERING (CON SCALATURA INTERNA)
# Author: Gemini
# Date: 08/10/2025
# --------------------------------------------------------------------------
# Questo script carica i dati NON SCALATI, li standardizza, raggruppa
# i voxel in super-voxel, aggrega i segnali e salva il risultato.

import os
import sys
import h5py
import numpy as np
import pandas as pd
from loguru import logger
import torch
from sklearn.preprocessing import StandardScaler # Aggiunto per la scalatura interna

# ... (le funzioni kmeans_pytorch_batched e la configurazione del logger rimangono invariate) ...
# Assicurati che la funzione kmeans_pytorch_batched sia presente qui come prima

# --- 0. Configurazione del Logger ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Definizione dei Parametri e Percorsi ---
M_PRIME = 10000
BATCH_SIZE = 8192
mat_file_path = '../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat'
split_data_file = 'split_data_and_indices.npz' 
output_data_file = 'supervoxel_data.npz'
output_kmeans_model_file = 'kmeans_supervoxel_model.pt'

# --- 2. Configurazione del dispositivo PyTorch ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Utilizzo del dispositivo: {DEVICE}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- 3. Funzione K-Means con PyTorch (Modalità Batch) ---
def kmeans_pytorch_batched(X, n_clusters, max_iter=300, tol=1e-4, batch_size=4096):
    # (Incolla qui la funzione kmeans_pytorch_batched che abbiamo definito in precedenza)
    # ...
    indices = torch.randperm(X.shape[0])[:n_clusters]
    centroids = X[indices]
    for iter_idx in range(max_iter):
        labels = torch.zeros(X.shape[0], dtype=torch.long, device=DEVICE)
        for i in range(0, X.shape[0], batch_size):
            end = i + batch_size
            batch_X = X[i:end]
            distances_batch = torch.cdist(batch_X, centroids)
            labels[i:end] = torch.argmin(distances_batch, dim=1)
        new_centroids = torch.zeros_like(centroids)
        for c in range(n_clusters):
            cluster_members = X[labels == c]
            if len(cluster_members) > 0:
                new_centroids[c] = cluster_members.mean(dim=0)
        centroid_shift = torch.norm(new_centroids - centroids)
        centroids = new_centroids
        if centroid_shift < tol:
            break
    return labels, centroids

# --- 4. Caricamento dei Dati NON SCALATI ---
logger.info(f"Caricamento dei dati NON SCALATI da '{split_data_file}'...")
try:
    with np.load(split_data_file) as data:
        X_rem = data['X_rem']
        y_rem = data['y_rem']
        X_hold_test = data['X_hold_test']
        y_hold_test = data['y_hold_test']
    
    with h5py.File(mat_file_path, 'r') as f:
        mask3D = f['mask'][()] 
        voxelIdx = f['voxelIdx'][()].ravel().astype(int) - 1
    logger.success('Dati non scalati caricati con successo.')
except FileNotFoundError as e:
    logger.critical(f"File non trovato: {e}. Eseguire prima 'Split.py'.")
    sys.exit(1)

# --- 5. Standardizzazione dei Dati ---
logger.info("Standardizzazione dei dati (fit solo su X_rem)...")
scaler = StandardScaler()
X_rem_scaled = scaler.fit_transform(X_rem)
X_hold_test_scaled = scaler.transform(X_hold_test) # Usa lo scaler fittato su rem
logger.success("Standardizzazione completata.")

# --- 6. Recupero Coordinate 3D dei Voxel ---
logger.info("Conversione degli indici lineari dei voxel in coordinate 3D...")
M = X_rem.shape[1]
coords_tuple = np.unravel_index(voxelIdx, mask3D.shape, order='F')
coords_np = np.column_stack(coords_tuple).astype(np.float32)
logger.success(f"Generate coordinate 3D per {coords_np.shape[0]} voxel.")

# --- 7. Raggruppamento Spaziale con K-means ---
logger.info(f"Avvio K-means (PyTorch) per raggruppare {M} voxel in {M_PRIME} super-voxel...")
coords_tensor = torch.from_numpy(coords_np).to(DEVICE)
supervoxel_labels_tensor, _ = kmeans_pytorch_batched(coords_tensor, n_clusters=M_PRIME, batch_size=BATCH_SIZE)
supervoxel_labels = supervoxel_labels_tensor.cpu().numpy()
logger.success("Clustering K-means completato.")

# --- 8. Aggregazione del Segnale per Super-voxel (sui dati scalati) ---
logger.info("Aggregazione delle intensità dei voxel in segnali di super-voxel...")
def aggregate_to_supervoxels(X_data, labels, n_clusters):
    df = pd.DataFrame(X_data.T)
    df['cluster'] = labels
    supervoxel_df = df.groupby('cluster').mean()
    return supervoxel_df.reindex(range(n_clusters)).fillna(0).T.values

X_rem_sv = aggregate_to_supervoxels(X_rem_scaled, supervoxel_labels, M_PRIME).astype(np.float32)
X_hold_test_sv = aggregate_to_supervoxels(X_hold_test_scaled, supervoxel_labels, M_PRIME).astype(np.float32)
logger.success("Aggregazione completata.")

# --- 9. Salvataggio dei Dati Basati su Super-voxel ---
logger.info(f"Salvataggio dei dati aggregati in '{output_data_file}'...")
np.savez_compressed(output_data_file,
                    X_rem_sv=X_rem_sv, y_rem=y_rem,
                    X_hold_test_sv=X_hold_test_sv, y_hold_test=y_hold_test,
                    supervoxel_labels=supervoxel_labels)
logger.success("Script completato con successo!")