# %% STEP 3: SUPER-VOXEL CLUSTERING (Accelerated with Intel Extension)
# Author: Gemini
# Date: 08/10/2025
# --------------------------------------------------------------------------
# Questo script raggruppa i voxel in super-voxel, aggrega i segnali
# di intensità e salva il risultato. Utilizza Intel(R) Extension for 
# Scikit-learn per accelerare drasticamente il clustering K-means.

# ---- NUOVE RIGHE PER ACCELERAZIONE INTEL ----
from sklearnex import patch_sklearn
patch_sklearn()
# ---------------------------------------------

import os
import sys
import h5py
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans

# --- 0. Configurazione del Logger ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Definizione dei Parametri e Percorsi ---
M_PRIME = 10000
mat_file_path = '../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat'
split_data_file = 'split_scaled_data.npz'
output_data_file = 'supervoxel_data.npz'
output_kmeans_model_file = 'kmeans_supervoxel_model.joblib'

# --- 2. Caricamento dei Dati ---
logger.info("Caricamento dei dati di input...")
try:
    with np.load(split_data_file) as data:
        X_rem_scaled = data['X_rem']
        y_rem = data['y_rem']
        X_hold_test_scaled = data['X_hold_test']
        y_hold_test = data['y_hold_test']
    logger.success(f'Caricato "{split_data_file}" con successo.')

    with h5py.File(mat_file_path, 'r') as f:
        # Usa il nome corretto 'mask' come discusso
        mask3D = f['mask'][()] 
        voxelIdx = f['voxelIdx'][()].ravel().astype(int)
    logger.success(f'Caricati metadati da "{mat_file_path}" con successo.')
except FileNotFoundError as e:
    logger.critical(f"File non trovato: {e}. Assicurarsi di aver eseguito gli script precedenti.")
    sys.exit(1)
except KeyError as e:
    logger.critical(f"Errore di chiave nel file .mat: {e}. Assicurarsi che il file contenga le variabili necessarie con i nomi corretti.")
    sys.exit(1)

# --- 3. Recupero Coordinate 3D dei Voxel ---
logger.info("Conversione degli indici lineari dei voxel in coordinate 3D...")
M = X_rem_scaled.shape[1]
coords_tuple = np.unravel_index(voxelIdx - 1, mask3D.shape, order='F')
coords = np.column_stack(coords_tuple)
logger.success(f"Generate coordinate 3D per {coords.shape[0]} voxel.")

# --- 4. Raggruppamento Spaziale con K-means ---
logger.info(f"Avvio del clustering K-means per raggruppare {M} voxel in {M_PRIME} super-voxel...")
# Il codice di KMeans rimane invariato. La patch di Intel lo accelera automaticamente.
kmeans = KMeans(n_clusters=M_PRIME, init='k-means++', max_iter=1000, n_init=3, random_state=42)
kmeans.fit(coords)
supervoxel_labels = kmeans.labels_
logger.success("Clustering K-means completato.")
joblib.dump(kmeans, output_kmeans_model_file)
logger.success(f"Modello K-means salvato in '{output_kmeans_model_file}'.")

# --- 5. Aggregazione del Segnale per Super-voxel ---
logger.info("Aggregazione delle intensità dei voxel in segnali di super-voxel (media)...")

def aggregate_to_supervoxels(X_data, labels, n_clusters):
    df = pd.DataFrame(X_data.T)
    df['cluster'] = labels
    supervoxel_df = df.groupby('cluster').mean()
    return supervoxel_df.reindex(range(n_clusters)).fillna(0).T.values

X_rem_sv = aggregate_to_supervoxels(X_rem_scaled, supervoxel_labels, M_PRIME).astype(np.float32)
X_hold_test_sv = aggregate_to_supervoxels(X_hold_test_scaled, supervoxel_labels, M_PRIME).astype(np.float32)

logger.success("Aggregazione e conversione a float32 completate.")
logger.info(f"Nuove dimensioni 'X_rem_sv': {X_rem_sv.shape} | Tipo Dati: {X_rem_sv.dtype}")
logger.info(f"Nuove dimensioni 'X_hold_test_sv': {X_hold_test_sv.shape} | Tipo Dati: {X_hold_test_sv.dtype}")

# --- 6. Salvataggio dei Dati Basati su Super-voxel ---
logger.info(f"Salvataggio dei dati aggregati in '{output_data_file}'...")
np.savez_compressed(output_data_file,
                    X_rem_sv=X_rem_sv,
                    y_rem=y_rem,
                    X_hold_test_sv=X_hold_test_sv,
                    y_hold_test=y_hold_test,
                    supervoxel_labels=supervoxel_labels)
logger.success("Script completato con successo!")


"""
# %% STEP 3: SUPER-VOXEL CLUSTERING (Accelerated with PyTorch for CUDA)
# Author: Gemini
# Date: 08/10/2025
# --------------------------------------------------------------------------
# Questo script raggruppa i voxel in super-voxel, aggrega i segnali
# di intensità e salva il risultato. Utilizza PyTorch per il clustering 
# K-means per accelerare drasticamente i calcoli tramite CUDA.

import os
import sys
import h5py
import joblib
import numpy as np
import pandas as pd
from loguru import logger
import torch

# --- 0. Configurazione del Logger ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Definizione dei Parametri e Percorsi ---
M_PRIME = 10000
mat_file_path = '../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat'
split_data_file = 'split_scaled_data.npz'
output_data_file = 'supervoxel_data.npz'
output_kmeans_model_file = 'kmeans_supervoxel_model.pt' # Salvataggio modello PyTorch

# --- 2. Configurazione del dispositivo PyTorch ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Utilizzo del dispositivo: {device}")

# --- 3. Funzione K-Means con PyTorch ---
def kmeans_pytorch(X, n_clusters, max_iter=300, tol=1e-4):
    ""
    Esegue il clustering K-Means usando PyTorch.
    ""
    # Inizializzazione casuale dei centroidi
    indices = torch.randperm(X.shape[0])[:n_clusters]
    centroids = X[indices]
    
    for i in range(max_iter):
        # Calcolo delle distanze e assegnazione ai cluster
        distances = torch.cdist(X, centroids)
        labels = torch.argmin(distances, dim=1)
        
        # Aggiornamento dei centroidi
        new_centroids = torch.zeros_like(centroids)
        for c in range(n_clusters):
            cluster_members = X[labels == c]
            if len(cluster_members) > 0:
                new_centroids[c] = cluster_members.mean(dim=0)
        
        # Criterio di convergenza
        centroid_shift = torch.norm(new_centroids - centroids)
        centroids = new_centroids
        
        logger.info(f"Iterazione K-means {i+1}/{max_iter} | Spostamento centroidi: {centroid_shift:.4f}")
        if centroid_shift < tol:
            logger.success("Convergenza K-means raggiunta.")
            break
            
    return labels, centroids

# --- 4. Caricamento dei Dati ---
logger.info("Caricamento dei dati di input...")
try:
    with np.load(split_data_file) as data:
        X_rem_scaled = data['X_rem']
        y_rem = data['y_rem']
        X_hold_test_scaled = data['X_hold_test']
        y_hold_test = data['y_hold_test']
    logger.success(f'Caricato "{split_data_file}" con successo.')

    with h5py.File(mat_file_path, 'r') as f:
        mask3D = f['mask'][()] 
        voxelIdx = f['voxelIdx'][()].ravel().astype(int)
    logger.success(f'Caricati metadati da "{mat_file_path}" con successo.')
except FileNotFoundError as e:
    logger.critical(f"File non trovato: {e}. Assicurarsi di aver eseguito gli script precedenti.")
    sys.exit(1)
except KeyError as e:
    logger.critical(f"Errore di chiave nel file .mat: {e}. Assicurarsi che il file contenga le variabili necessarie con i nomi corretti.")
    sys.exit(1)

# --- 5. Recupero Coordinate 3D dei Voxel ---
logger.info("Conversione degli indici lineari dei voxel in coordinate 3D...")
M = X_rem_scaled.shape[1]
coords_tuple = np.unravel_index(voxelIdx - 1, mask3D.shape, order='F')
coords_np = np.column_stack(coords_tuple).astype(np.float32)
logger.success(f"Generate coordinate 3D per {coords_np.shape[0]} voxel.")

# --- 6. Raggruppamento Spaziale con K-means (PyTorch) ---
logger.info(f"Avvio del clustering K-means (PyTorch) per raggruppare {M} voxel in {M_PRIME} super-voxel...")
coords_tensor = torch.from_numpy(coords_np).to(device)

# Esecuzione del clustering
supervoxel_labels_tensor, supervoxel_centroids_tensor = kmeans_pytorch(coords_tensor, n_clusters=M_PRIME)

# Riporta le etichette sulla CPU come array NumPy
supervoxel_labels = supervoxel_labels_tensor.cpu().numpy()

logger.success("Clustering K-means completato.")
torch.save(supervoxel_centroids_tensor, output_kmeans_model_file)
logger.success(f"Modello K-means (centroidi) salvato in '{output_kmeans_model_file}'.")

# --- 7. Aggregazione del Segnale per Super-voxel ---
logger.info("Aggregazione delle intensità dei voxel in segnali di super-voxel (media)...")

def aggregate_to_supervoxels(X_data, labels, n_clusters):
    df = pd.DataFrame(X_data.T)
    df['cluster'] = labels
    supervoxel_df = df.groupby('cluster').mean()
    return supervoxel_df.reindex(range(n_clusters)).fillna(0).T.values

X_rem_sv = aggregate_to_supervoxels(X_rem_scaled, supervoxel_labels, M_PRIME).astype(np.float32)
X_hold_test_sv = aggregate_to_supervoxels(X_hold_test_scaled, supervoxel_labels, M_PRIME).astype(np.float32)

logger.success("Aggregazione e conversione a float32 completate.")
logger.info(f"Nuove dimensioni 'X_rem_sv': {X_rem_sv.shape} | Tipo Dati: {X_rem_sv.dtype}")
logger.info(f"Nuove dimensioni 'X_hold_test_sv': {X_hold_test_sv.shape} | Tipo Dati: {X_hold_test_sv.dtype}")

# --- 8. Salvataggio dei Dati Basati su Super-voxel ---
logger.info(f"Salvataggio dei dati aggregati in '{output_data_file}'...")
np.savez_compressed(output_data_file,
                    X_rem_sv=X_rem_sv,
                    y_rem=y_rem,
                    X_hold_test_sv=X_hold_test_sv,
                    y_hold_test=y_hold_test,
                    supervoxel_labels=supervoxel_labels)
logger.success("Script completato con successo!")
"""