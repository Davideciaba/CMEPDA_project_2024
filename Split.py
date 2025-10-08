# %% STEP 2: SPLIT "HOLD-OUT TEST SET" vs "REMAINING" (20% / 80%)
# Author: Gemini
# Date: 07/10/2025
# --------------------------------------------------------------------------
# Questo script carica i dati preprocessati da un file .mat e li suddivide
# in un "remaining set" (80%) per la validazione incrociata interna e un
# "hold-out test set" (20%) per la valutazione finale del modello.

from loguru import logger
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import os

# --- 1. Caricamento dei dati preprocessati ---
# Si assume che il file .mat sia stato generato dalla fase "Preliminaries".
mat_file_path = '../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat'
print(f'Caricamento dei dati da "{mat_file_path}"...')

if not os.path.exists(mat_file_path):
    raise FileNotFoundError(f'File "{mat_file_path}" non trovato. Eseguire prima lo script MATLAB Preliminaries.')

# Use h5py to read the v7.3 .mat file
with h5py.File(mat_file_path, 'r') as f:
    # Data is stored as datasets. Access them by name.
    # We must transpose (.T) the arrays to match the (samples, features) shape
    # expected by scikit-learn, due to MATLAB's column-major storage.
    X_raw = f['X_raw'][()].T 
    # Also transpose y_all and then flatten it to a 1D array
    y_all = f['y_all'][()].T.ravel()
print('Dati caricati con successo.')
print(f'Dimensioni X_raw: {X_raw.shape}')
print(f'Dimensioni y_all: {y_all.shape}')

# --- 2. Partizionamento Stratificato 80% / 20% ---
# Obiettivo: Separare i dati mantenendo le proporzioni delle classi.

print('Esecuzione del partizionamento stratificato 80/20...')
# 'test_size=0.2' -> 20% dei dati per il test set (hold-out).
# 'stratify=y_all' -> assicura che la proporzione di AD/CTRL sia la stessa nei due set.
# 'random_state' -> garantisce che la suddivisione sia sempre la stessa.
X_rem, X_hold_test, y_rem, y_hold_test = train_test_split(
    X_raw, 
    y_all, 
    test_size=0.20, 
    stratify=y_all, 
    random_state=42  # Per la riproducibilità
)

print('Partizionamento completato.')
print(f'Dimensioni Remaining Set (X_rem): {X_rem.shape}')
print(f'Dimensioni Hold-Out Set (X_hold_test): {X_hold_test.shape}')

# --- 3. Salvataggio dei dati suddivisi (Opzionale) ---
# Salva i nuovi set in un formato .npz, efficiente per gli array NumPy.
output_file = 'hold_out_split_data.npz'
print(f'Salvataggio dei dati suddivisi in "{output_file}"...')
np.savez_compressed(output_file, 
                    X_rem=X_rem, y_rem=y_rem, 
                    X_hold_test=X_hold_test, y_hold_test=y_hold_test)
print('Salvataggio completato.')