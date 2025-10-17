# %% STEP 2: SPLIT DATA AND INDICES (UNIVERSAL VERSION)
# Author: Gemini
# Date: 16/10/2025
# --------------------------------------------------------------------------
# Questo script funge da "master splitter" per l'intera pipeline.
# 1. Carica i dati grezzi (raw) dal file .mat generato da Preliminaries.
# 2. Suddivide i dati e gli indici dei soggetti in un "remaining set" (80%)
#    e un "hold-out test set" (20%) in modo stratificato.
# 3. Salva i dati NON SCALATI e gli INDICI, rendendoli utilizzabili sia
#    dalla pipeline SVM (che necessita dei dati grezzi) sia da quella
#    EfficientNet (che necessita degli indici).

import os
import sys
import h5py
import joblib
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

# --- 0. Configurazione del Logger ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Definizione dei Percorsi ---
mat_file_path = '../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat'
output_data_file = 'split_data_and_indices.npz'

# --- 2. Caricamento dei Dati Grezzi ---
logger.info(f'Caricamento dei dati grezzi da "{mat_file_path}"...')
try:
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f'File "{mat_file_path}" non trovato. Assicurati di aver eseguito lo script MATLAB Preliminaries con ID dei soggetti univoci.')

    with h5py.File(mat_file_path, 'r') as f:
        # Carica i dati e li traspone per avere il formato (soggetti, features)
        X_raw = f['X_raw'][()].T
        y_all = f['y_all'][()].T.ravel()  # .ravel() appiattisce in un array 1D

    logger.success(f'Dati caricati con successo. Trovati {X_raw.shape[0]} soggetti.')

except (FileNotFoundError, KeyError) as e:
    logger.critical(f"Errore nel caricamento del file .mat: {e}")
    sys.exit(1)

# --- 3. Creazione degli Indici dei Soggetti ---
# Crea un array di indici da 0 a N-1 che rappresenta ogni riga della matrice.
# Questo array verrà suddiviso insieme ai dati per sapere quali soggetti
# appartengono a ciascun set, cosa fondamentale per EfficientNet.
subject_indices = np.arange(X_raw.shape[0])

# --- 4. Partizionamento Stratificato 80% / 20% ---
logger.info('Esecuzione del partizionamento stratificato 80/20 su dati, etichette e indici...')

# Suddividiamo simultaneamente i dati (X), le etichette (y) e gli indici.
# 'stratify=y_all' assicura che la proporzione AD/CTRL sia mantenuta in entrambi i set.
# 'random_state=42' garantisce che la suddivisione sia sempre la stessa.
X_rem, X_hold_test, y_rem, y_hold_test, trainVal_indices, testHold_indices = train_test_split(
    X_raw,
    y_all,
    subject_indices,
    test_size=0.20,
    stratify=y_all,
    random_state=42
)

logger.success('Partizionamento completato.')
logger.info(f'Remaining Set: {X_rem.shape[0]} soggetti')
logger.info(f'Hold-Out Set: {X_hold_test.shape[0]} soggetti')

# --- 5. Salvataggio dei Dati Suddivisi (NON SCALATI) e degli Indici ---
logger.info(f'Salvataggio dei dati e degli indici in "{output_data_file}"...')

# Usiamo np.savez per evitare potenziali MemoryError con la compressione su dati grandi.
# Salviamo tutto ciò che serve alle pipeline successive.
np.savez_compressed(output_data_file,
         # Dati NON SCALATI per la pipeline SVM
         X_rem=X_rem,
         X_hold_test=X_hold_test,
         
         # Etichette per entrambe le pipeline
         y_rem=y_rem,
         y_hold_test=y_hold_test,
         
         # Indici per la pipeline EfficientNet
         trainVal_indices=trainVal_indices,
         testHold_indices=testHold_indices,

         # Vettore completo delle etichette, utile per referenziare
         y_all=y_all
        )

logger.success(f'Script completato. Dati salvati in "{output_data_file}".')