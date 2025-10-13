# %% STEP 2: SPLIT DATA (VERSIONE FINALE CON AUTO-DIAGNOSI)
# Author: Gemini
# Date: 13/10/2025
# --------------------------------------------------------------------------
# Questo script prima verifica la coerenza di TUTTI i nomi dei file .nii.
# Se tutti i nomi sono riconosciuti, procede. Altrimenti, si ferma
# e mostra una lista dei file problematici.

import sys
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger
from pathlib import Path
import re

# --- 0. Configurazione del Logger ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Definizione dei Percorsi ---
# Usa percorsi relativi come richiesto
DATA_ROOT = '../CMEPDA_project_2024/AD_CTRL' # Cartella che contiene AD_s3 e CTRL_s3
mat_file_path = '../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat'
output_data_file = 'split_data_and_indices.npz'

# --- 2. Scansione, Verifica e Diagnosi dei Nomi dei File ---
logger.info(f"Scansione della cartella '{DATA_ROOT}' per i file .nii...")
try:
    data_root_path = Path(DATA_ROOT).resolve(strict=True)
    all_nii_files = list(data_root_path.glob('**/*.nii')) # Cerca solo .nii
    
    if not all_nii_files:
        raise FileNotFoundError("Nessun file .nii trovato. Controlla il percorso in DATA_ROOT.")

    logger.info(f"Trovati {len(all_nii_files)} file .nii totali. Inizio verifica dei nomi...")

    available_subject_indices = []
    unmatched_files = []
    
    for f in all_nii_files:
        # Tenta di riconoscere i pattern noti in ordine di probabilità
        # Pattern 1: Cerca un trattino seguito da un numero (es. "smwc1AD-1.nii")
        match = re.search(r'-(\d+)', f.name)
        
        # Pattern 2: Se fallisce, cerca "subj" seguito da un numero (pattern originale)
        if not match:
            match = re.search(r'subj(\d+)', f.name)
        
        # (Se scopriamo un terzo pattern, andrà aggiunto qui)

        if match:
            subject_id = int(match.group(1))
            available_subject_indices.append(subject_id)
        else:
            unmatched_files.append(f.name)

    # --- FASE DI AUTO-DIAGNOSI ---
    if unmatched_files:
        logger.critical("ERRORE DI COERENZA NEI NOMI DEI FILE")
        logger.warning(f"Sono stati trovati {len(unmatched_files)} file con un nome non riconosciuto.")
        logger.info("La pipeline non può continuare in modo affidabile.")
        logger.info("Ecco i primi 15 esempi di file non riconosciuti:")
        for i, unmatched_name in enumerate(unmatched_files[:15]):
            print(f"  -> {unmatched_name}")
        logger.info("Per favore, mostra questo output per ricevere la correzione finale.")
        sys.exit(1) # Interrompe l'esecuzione

    logger.success(f"Verifica completata: tutti i {len(all_nii_files)} file hanno un nome valido e riconosciuto.")
    available_subject_indices = np.array(sorted(list(set(available_subject_indices))))

except FileNotFoundError as e:
    logger.critical(f"Errore nel percorso dei file: {e}")
    sys.exit(1)

# Se la diagnosi è stata superata, il resto dello script viene eseguito normalmente
# --- 3. Caricamento e Filtraggio Dati dal .mat ---
# ... (il resto del codice da qui in poi è identico e corretto) ...
logger.info(f'Caricamento dei dati da "{mat_file_path}"...')
try:
    with h5py.File(mat_file_path, 'r') as f:
        X_raw_all = f['X_raw'][()].T
        y_all_all = f['y_all'][()].T.ravel()
    available_array_indices = available_subject_indices - 1
    X_raw = X_raw_all[available_array_indices, :]
    y_all = y_all_all[available_array_indices]
    logger.success(f"Dati filtrati. Si procede con {len(y_all)} soggetti.")
except (FileNotFoundError, KeyError) as e:
    logger.critical(f"Errore nel caricamento del file .mat: {e}")
    sys.exit(1)
except IndexError as e:
    logger.critical(f"Errore di indicizzazione: {e}. Controlla che gli ID nei tuoi .nii corrispondano al .mat.")
    sys.exit(1)

# --- 4. Partizionamento Stratificato 80% / 20% ---
logger.info('Esecuzione del partizionamento stratificato 80/20...')
X_rem, X_hold_test, y_rem, y_hold_test, trainValList, testHoldList = train_test_split(
    X_raw, y_all, available_subject_indices,
    test_size=0.20, stratify=y_all, random_state=42
)
logger.success('Partizionamento completato.')

# --- 5. Salvataggio Finale ---
logger.info(f'Salvataggio di dati e indici in "{output_data_file}"...')
np.savez_compressed(output_data_file,
                    X_rem=X_rem, y_rem=y_rem,
                    X_hold_test=X_hold_test, y_hold_test=y_hold_test,
                    trainValList=trainValList, testHoldList=testHoldList,
                    y_all=y_all)
logger.success('Salvataggio completato con successo.')