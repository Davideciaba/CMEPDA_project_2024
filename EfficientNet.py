# %% STEP 8: EFFICIENTNET3D ANALYSIS (LOOCV, FINAL MODEL, AND MAPPING)
# Author: Gemini
# Date: 09/10/2025
# --------------------------------------------------------------------------
# Questo script implementa l'intera pipeline di analisi per l'EfficientNet3D
# come descritto nel documento di pipeline.
# SEZIONE 4: Esegue una LOOCV interna sul "remaining set" (80%) usando i file .nii.
# SEZIONE 5: Addestra il modello finale e lo valuta sul "hold-out set" (20%).
# SEZIONE 6: Aggrega le mappe GradCAM in "aspetti" e crea una mappa NIfTI.
# --------------------------------------------------------------------------

import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import re # Per estrarre gli indici dei soggetti dai nomi dei file

# Assicurati di aver installato le librerie necessarie:
# pip install torch torchvision torchaudio
# pip install monai nibabel scikit-learn pandas
try:
    import nibabel as nib
    from monai.networks.nets import EfficientNetBN
    from monai.visualize import GradCAM
except ImportError as e:
    logger.critical(f"Libreria mancante: {e}. Assicurati di aver installato tutti i pacchetti necessari.")
    sys.exit(1)

# --- 0. Configurazione del Logger ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Definizione dei Parametri e Percorsi ---
# Percorsi di Input
# NOTA: USARE PATH ASSOLUTI O CORRETTAMENTE RELATIVI È FONDAMENTALE.
# Path.resolve() aiuta a gestire i percorsi in modo robusto.
try:
    # Usa Path per gestire i percorsi in modo cross-platform
    data_root = Path('C:/Tancredi/Libri/Fisica/Magistrale/Computing_methods/Progetto/CMEPDA_project_2024_/CMEPDA_project_2024/AD_CTRL').resolve(strict=True)
    mat_file_path = Path('../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat').resolve(strict=True)
    split_indices_file = 'split_indices.npz' # !IMPORTANTE: File con gli indici di split
    clustering_results_file = 'hierarchical_clustering_results.npz'
except FileNotFoundError as e:
    logger.critical(f"Percorso non trovato: {e}. Verifica la correttezza dei percorsi di input.")
    sys.exit(1)

# Parametri di Training e Modello
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Dispositivo PyTorch in uso: {DEVICE}")
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
MAX_EPOCHS_LOOCV = 50
PATIENCE = 5

# --- 2. Caricamento Dati e Metadati ---
logger.info("Caricamento di indici, metadati e costruzione delle liste di file...")
try:
    with np.load(split_indices_file) as data:
        trainValList = data['trainValList']
        testHoldList = data['testHoldList']
        y_all = data['y_all']

    with np.load(clustering_results_file) as data:
        cluster_labels_voxel = data['cluster_labels_voxel']
        K = int(data['num_clusters'])

    with h5py.File(mat_file_path, 'r') as f:
        mask3D = f['mask'][()]
    
    n_rem = len(trainValList)
    M = np.sum(mask3D)
    
    # Costruzione del dizionario che mappa l'indice del soggetto (1-based) al suo file .nii
    all_nii_files = list(data_root.glob('**/*.nii'))
    subject_file_map = {}
    for f in all_nii_files:
        match = re.search(r'subj(\d+)', f.name)
        if match:
            subject_idx = int(match.group(1))
            subject_file_map[subject_idx] = f

    # Creazione delle liste di file per i due set
    trainValFiles = [subject_file_map[i] for i in trainValList]
    testHoldFiles = [subject_file_map[i] for i in testHoldList]
    y_rem = y_all[np.array(trainValList) - 1]
    y_hold_test = y_all[np.array(testHoldList) - 1]

    logger.success(f"Dati caricati: {len(trainValFiles)} file per LOOCV, {len(testHoldFiles)} per hold-out.")

except (FileNotFoundError, KeyError) as e:
    logger.critical(f"Errore nel caricamento dei file: {e}. Assicurati di aver generato tutti i file necessari.")
    sys.exit(1)

# --- Funzioni di Supporto ---
def load_and_resize(nii_path):
    """Carica un file NIfTI e lo restituisce come array numpy."""
    # Per ora non implementiamo il resize, assumiamo che i dati siano già pronti.
    # In una pipeline reale, qui useresti monai.transforms.
    img = nib.load(nii_path)
    return img.get_fdata().astype(np.float32)

class GM3DDataset(Dataset):
    """Dataset PyTorch per caricare e normalizzare i volumi 3D."""
    def __init__(self, file_paths, labels, mean=0., std=1.):
        self.file_paths = file_paths
        self.labels = labels
        self.mean = mean
        self.std = std
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        vol = load_and_resize(self.file_paths[idx])
        vol = (vol - self.mean) / self.std
        x = torch.tensor(vol[None, ...], dtype=torch.float32) # Aggiunge canale
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ==========================================================================
# SEZIONE 4: LOOCV INTERNO SUL REMAINING SET (80%)
# ==========================================================================
logger.info("INIZIO SEZIONE 4: Esecuzione del LOOCV interno per EfficientNet3D...")
G_all_rem = np.zeros((n_rem, M))
epochs_per_fold = [] # Per calcolare la mediana delle epoche

for k in tqdm(range(n_rem), desc="EfficientNet3D Internal LOOCV"):
    
    val_file = trainValFiles[k]
    train_files = np.delete(trainValFiles, k).tolist()
    y_val_int = y_rem[k]
    y_train_int = np.delete(y_rem, k)
    
    # Calcolo di media e std SOLO sui volumi di training interni
    train_vols_for_norm = np.stack([load_and_resize(f) for f in train_files])
    mean_int, std_int = train_vols_for_norm.mean(), train_vols_for_norm.std()
    
    train_dataset = GM3DDataset(train_files, y_train_int, mean_int, std_int)
    val_dataset = GM3DDataset([val_file], [y_val_int], mean_int, std_int)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    model = EfficientNetBN(spatial_dims=3, in_channels=1, num_classes=2, model_name="efficientnet-b0").to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    model_save_path = f"./best_model_fold_{k}.pth"
    
    for epoch in range(MAX_EPOCHS_LOOCV):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        current_val_loss = 0.0
        with torch.no_grad():
            inputs, labels = next(iter(val_loader))
            outputs = model(inputs.to(DEVICE))
            current_val_loss = criterion(outputs, labels.to(DEVICE)).item()

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                epochs_per_fold.append(epoch + 1)
                break
    
    # Calcolo GradCAM sul soggetto di validazione
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    val_tensor = next(iter(val_loader))[0].to(DEVICE)
    gradcam = GradCAM(nn_module=model, target_layers=model._conv_head)
    heatmap_3d = gradcam(x=val_tensor, class_idx=1)[0, 0, ...].cpu().numpy()
    
    G_all_rem[k, :] = heatmap_3d[mask3D]
    os.remove(model_save_path)

logger.success("SEZIONE 4 COMPLETATA: LOOCV interno terminato.")

# ==========================================================================
# SEZIONE 5 & 6: MODELLO FINALE, VALUTAZIONE E MAPPA
# ==========================================================================
logger.info("INIZIO SEZIONE 5: Addestramento del modello finale e valutazione.")

# 5.2.1 Standardizzazione su tutti i volumi del remaining set
all_rem_vols = np.stack([load_and_resize(f) for f in trainValFiles])
mean_rem, std_rem = all_rem_vols.mean(), all_rem_vols.std()
rem_dataset = GM3DDataset(trainValFiles, y_rem, mean_rem, std_rem)
rem_loader = DataLoader(rem_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 5.2.2 Training Finale
n_epochs_final = int(np.median(epochs_per_fold)) if epochs_per_fold else MAX_EPOCHS_LOOCV // 2
logger.info(f"Addestramento del modello finale per {n_epochs_final} epoche (mediana del LOOCV)...")

model_hold = EfficientNetBN(spatial_dims=3, in_channels=1, num_classes=2, model_name="efficientnet-b0").to(DEVICE)
optimizer = optim.Adam(model_hold.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(n_epochs_final):
    model_hold.train()
    for inputs, labels in rem_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model_hold(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 5.2.3 Predizione e GradCAM su Hold-Out set
logger.info(f"Valutazione e calcolo GradCAM su {len(testHoldFiles)} soggetti del hold-out set...")
model_hold.eval()
gradcam_final = GradCAM(nn_module=model_hold, target_layers=model_hold._conv_head)
G_hold = np.zeros((len(testHoldFiles), M))

hold_dataset = GM3DDataset(testHoldFiles, y_hold_test, mean_rem, std_rem)
hold_loader = DataLoader(hold_dataset, batch_size=1)

with torch.no_grad():
    for h, (inputs, _) in enumerate(hold_loader):
        heatmap_3d = gradcam_final(x=inputs.to(DEVICE), class_idx=1)[0, 0, ...].cpu().numpy()
        G_hold[h, :] = heatmap_3d[mask3D]

# 6.2 Aggregazione GradCAM e creazione mappa NIfTI
logger.info("INIZIO SEZIONE 6: Aggregazione dei risultati di GradCAM.")
G_per_aspect_hold = np.zeros((len(testHoldFiles), K))
for h in range(len(testHoldFiles)):
    for a in range(1, K + 1):
        idx = np.where(cluster_labels_voxel == a)[0]
        if len(idx) > 0:
            G_per_aspect_hold[h, a - 1] = G_hold[h, idx].mean()

idx_AD_hold = np.where(y_hold_test == 1)[0]
idx_CTRL_hold = np.where(y_hold_test == 0)[0]

G_AD_hold_mean = G_per_aspect_hold[idx_AD_hold, :].mean(axis=0)
G_CTRL_hold_mean = G_per_aspect_hold[idx_CTRL_hold, :].mean(axis=0)
Delta_G_hold = G_AD_hold_mean - G_CTRL_hold_mean

flat_map = np.zeros(M)
for a in range(1, K + 1):
    idx = np.where(cluster_labels_voxel == a)[0]
    flat_map[idx] = Delta_G_hold[a - 1]

G_map_hold_3D = np.zeros(mask3D.shape)
G_map_hold_3D[mask3D] = flat_map

ref_img = nib.load(all_nii_files[0])
nifti_image = nib.Nifti1Image(G_map_hold_3D, ref_img.affine, ref_img.header)
output_nii_file = 'GradCAM_map_aspects_holdout.nii'
nib.save(nifti_image, output_nii_file)

logger.success(f"SEZIONI 5 & 6 COMPLETATE: Mappa NIfTI salvata in '{output_nii_file}'.")