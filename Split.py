# %% STEP 2: SPLIT "HOLD-OUT TEST SET" vs "REMAINING" AND SCALE
# Author: Gemini
# Date: 07/10/2025
# --------------------------------------------------------------------------
# Questo script carica i dati preprocessati, li suddivide in un "remaining set"
# (80%) e un "hold-out test set" (20%), e infine applica la standardizzazione
# ai dati. Lo scaler viene addestrato solo sul remaining set per evitare
# data leakage.

from loguru import logger
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import joblib

# --- 0. Configurazione del Logger ---
# Rimuove il logger di default e ne aggiunge uno nuovo per un output pulito.
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Caricamento dei dati preprocessati ---
mat_file_path = '../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat'
logger.info(f'Caricamento dei dati da "{mat_file_path}"...')

try:
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f'File "{mat_file_path}" non trovato. Eseguire prima lo script MATLAB Preliminaries.')

    # Usa h5py per leggere i file .mat v7.3
    with h5py.File(mat_file_path, 'r') as f:
        # Trasponi (.T) gli array per ottenere il formato (samples, features)
        # atteso da scikit-learn, a causa della memorizzazione column-major di MATLAB.
        X_raw = f['X_raw'][()].T
        y_all = f['y_all'][()].T.ravel() # .ravel() per appiattire in un array 1D

    logger.success('Dati caricati con successo.')
    logger.info(f'Dimensioni X_raw: {X_raw.shape}')
    logger.info(f'Dimensioni y_all: {y_all.shape}')

except FileNotFoundError as e:
    logger.critical(e)
    sys.exit(1) # Esce dallo script se il file non viene trovato

# --- 2. Partizionamento Stratificato 80% / 20% ---
# Separa i dati in un "remaining set" per la cross-validation e un "hold-out set"
# per il test finale, mantenendo le proporzioni delle classi.

logger.info('Esecuzione del partizionamento stratificato 80/20...')
# 'test_size=0.20' -> 20% dei dati per il test set.
# 'stratify=y_all' -> assicura che la proporzione AD/CTRL sia mantenuta.
# 'random_state=42' -> garantisce la riproducibilità della suddivisione.
X_rem, X_hold_test, y_rem, y_hold_test = train_test_split(
    X_raw,
    y_all,
    test_size=0.20,
    stratify=y_all,
    random_state=42
)

logger.success('Partizionamento completato.')
logger.info(f'Dimensioni Remaining Set (X_rem): {X_rem.shape}')
logger.info(f'Dimensioni Hold-Out Set (X_hold_test): {X_hold_test.shape}')

# --- 3. Standardizzazione dei Dati ---
# Applica la standardizzazione (rimuove la media e scala a varianza unitaria).
# IMPORTANTE: Lo scaler viene addestrato (fit) SOLO sul remaining set (X_rem)
# per evitare di "inquinare" i dati di training con informazioni dal test set.

logger.info('Avvio della standardizzazione (fit su X_rem, transform su entrambi i set)...')
scaler = StandardScaler()

# Addestra lo scaler e trasforma X_rem
X_rem_scaled = scaler.fit_transform(X_rem)

# Usa lo scaler GIA' ADDESTRATO per trasformare X_hold_test
X_hold_test_scaled = scaler.transform(X_hold_test)

logger.success('Standardizzazione completata.')
# Le dimensioni non cambiano dopo la standardizzazione
logger.info(f'Dimensioni X_rem_scaled: {X_rem_scaled.shape}')
logger.info(f'Dimensioni X_hold_test_scaled: {X_hold_test_scaled.shape}')


# --- 4. Salvataggio dei dati suddivisi e dello Scaler ---
# Salva i set di dati scalati in un file .npz e lo scaler in un file .joblib.
output_data_file = 'split_scaled_data.npz'
output_scaler_file = 'scaler.joblib'

logger.info(f'Salvataggio dei dati suddivisi e scalati in "{output_data_file}"...')
np.savez_compressed(output_data_file,
                    X_rem=X_rem_scaled, y_rem=y_rem,
                    X_hold_test=X_hold_test_scaled, y_hold_test=y_hold_test)

logger.info(f'Salvataggio dell\'oggetto scaler in "{output_scaler_file}"...')
joblib.dump(scaler, output_scaler_file)

logger.success('Salvataggio completato con successo.')