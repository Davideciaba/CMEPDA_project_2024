# %% STEP 8: EFFICIENTNET3D ANALYSIS (VERSIONE FINALE COMPLETA)
# Author: Gemini
# Date: 16/10/2025
# --------------------------------------------------------------------------
# QUESTA VERSIONE INTEGRA TUTTE LE CORREZIONI:
# - Mappatura file intelligente per ID non univoci.
# - Correzione MemoryError (mask booleana).
# - Correzione GradCAM (nome layer come stringa).
# - Correzione torch.load (weights_only=True).
# - Implementazione di Randomized Search per il learning rate.

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
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from scipy.stats import loguniform

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
DATA_ROOT_NII = '../CMEPDA_project_2024/AD_CTRL'
split_data_file = 'split_data_and_indices.npz'
mat_file_path = '../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat'
clustering_results_file = 'hierarchical_clustering_results.npz'

# Parametri di Training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Dispositivo PyTorch in uso: {DEVICE}")
BATCH_SIZE = 4
MAX_EPOCHS_LOOCV = 50
PATIENCE = 5

# ==========================================================================
# SEZIONE 2: CARICAMENTO DATI E MAPPATURA FILE INTELLIGENTE
# ==========================================================================
logger.info("Caricamento di indici e metadati...")
try:
    with np.load(split_data_file) as data:
        trainVal_indices = data['trainVal_indices']
        testHold_indices = data['testHold_indices']
        y_all = data['y_all']

    with np.load(clustering_results_file) as data:
        cluster_labels_voxel, K = data['cluster_labels_voxel'], int(data['num_clusters'])
    
    with h5py.File(mat_file_path, 'r') as f:
        mask3D = f['mask'][()]

    # --- FIX CRUCIALE PER MEMORYERROR ---
    # Converte la maschera in un tipo booleano per un'indicizzazione efficiente.
    mask3D = mask3D.astype(bool)
    logger.info("Maschera 3D convertita in tipo booleano per risolvere MemoryError.")
    # ------------------------------------

    logger.info(f"Scansione di '{DATA_ROOT_NII}' per i file .nii/.nii.gz...")
    data_root_path = Path(DATA_ROOT_NII)
    all_nii_files = list(data_root_path.glob('**/*.nii')) + list(data_root_path.glob('**/*.nii.gz'))

    ad_files = sorted([f for f in all_nii_files if 'AD' in f.name])
    ctrl_files = sorted([f for f in all_nii_files if 'CTRL' in f.name])
    all_files_ordered = ad_files + ctrl_files
    
    if len(all_files_ordered) != len(y_all):
        logger.critical(f"Discrepanza! Trovati {len(all_files_ordered)} file, ma il file .mat contiene {len(y_all)} soggetti.")
        sys.exit(1)
        
    logger.success(f"Mappatura completata: {len(all_files_ordered)} file .nii ordinati correttamente.")

    all_files_ordered = np.array(all_files_ordered)
    trainValFiles = all_files_ordered[trainVal_indices].tolist()
    testHoldFiles = all_files_ordered[testHold_indices].tolist()

    y_rem = y_all[trainVal_indices]
    y_hold_test = y_all[testHold_indices]
    
    n_rem, M = len(trainValFiles), np.sum(mask3D)
    logger.success(f"Procedura avviata con dati coerenti: {n_rem} soggetti per LOOCV, {len(testHoldFiles)} per hold-out.")

except (FileNotFoundError, KeyError, IndexError) as e:
    logger.critical(f"Errore critico durante il caricamento o la mappatura: {e}. Controlla i percorsi e assicurati che tutti gli script siano aggiornati.")
    sys.exit(1)

# --- Funzioni di Supporto ---
def load_and_resize(nii_path):
    img = nib.load(nii_path)
    return img.get_fdata().astype(np.float32)

class GM3DDataset(Dataset):
    def __init__(self, file_paths, labels, mean=0., std=1.):
        self.file_paths, self.labels, self.mean, self.std = file_paths, labels, mean, std
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        vol = (load_and_resize(self.file_paths[idx]) - self.mean) / self.std
        return torch.tensor(vol[None, ...], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================================================
# SEZIONE 4: LOOCV INTERNO CON RANDOMIZED SEARCH PER IL LEARNING RATE
# ==========================================================================
logger.info("INIZIO SEZIONE 4: Esecuzione del LOOCV interno per EfficientNet3D con Randomized Search...")
G_all_rem = np.zeros((n_rem, M))
epochs_per_fold = []
best_lr_per_fold = []

# Definiamo una distribuzione da cui campionare i learning rate.
lr_distribution = loguniform(1e-5, 1e-3)

for k in tqdm(range(n_rem), desc="EfficientNet3D Internal LOOCV"):
    val_file = trainValFiles[k]
    train_files = np.delete(trainValFiles, k).tolist()
    y_val_int = y_rem[k]
    y_train_int = np.delete(y_rem, k)
    
    train_vols_for_norm = np.stack([load_and_resize(f) for f in train_files])
    mean_int, std_int = train_vols_for_norm.mean(), train_vols_for_norm.std()
    
    train_dataset = GM3DDataset(train_files, y_train_int, mean_int, std_int)
    val_dataset = GM3DDataset([val_file], [y_val_int], mean_int, std_int)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    model = EfficientNetBN(spatial_dims=3, in_channels=1, num_classes=2, model_name="efficientnet-b0").to(DEVICE)
    
    current_learning_rate = lr_distribution.rvs()
    optimizer = optim.Adam(model.parameters(), lr=current_learning_rate)
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
                best_lr_per_fold.append(current_learning_rate)
                break
    else: # Se il loop finisce senza 'break'
        epochs_per_fold.append(MAX_EPOCHS_LOOCV)
        best_lr_per_fold.append(current_learning_rate)

    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()
    
    val_tensor = next(iter(val_loader))[0].to(DEVICE)
    gradcam = GradCAM(nn_module=model, target_layers="_conv_head")
    heatmap_3d = gradcam(x=val_tensor, class_idx=1)[0, 0, ...].cpu().numpy()
    
    G_all_rem[k, :] = heatmap_3d[mask3D] # Ora funziona grazie alla mask booleana
    os.remove(model_save_path)

logger.success("SEZIONE 4 COMPLETATA: LOOCV interno terminato.")

# ==========================================================================
# SEZIONE 5 & 6: MODELLO FINALE, VALUTAZIONE, GRAFICI E MAPPA
# ==========================================================================
logger.info("INIZIO SEZIONE 5: Addestramento del modello finale.")

FINAL_LEARNING_RATE = np.median(best_lr_per_fold) if best_lr_per_fold else 1e-4
logger.info(f"Learning rate finale scelto (mediana dei fold): {FINAL_LEARNING_RATE:.6f}")

all_rem_vols = np.stack([load_and_resize(f) for f in trainValFiles])
mean_rem, std_rem = all_rem_vols.mean(), all_rem_vols.std()
rem_dataset = GM3DDataset(trainValFiles, y_rem, mean_rem, std_rem)
rem_loader = DataLoader(rem_dataset, batch_size=BATCH_SIZE, shuffle=True)

n_epochs_final = int(np.median(epochs_per_fold)) if epochs_per_fold else MAX_EPOCHS_LOOCV // 2
logger.info(f"Addestramento del modello finale per {n_epochs_final} epoche...")

model_hold = EfficientNetBN(spatial_dims=3, in_channels=1, num_classes=2, model_name="efficientnet-b0").to(DEVICE)
optimizer = optim.Adam(model_hold.parameters(), lr=FINAL_LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
final_loss_history = []

for epoch in tqdm(range(n_epochs_final), desc="Final Model Training"):
    model_hold.train()
    epoch_loss = 0.0
    for inputs, labels in rem_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model_hold(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    final_loss_history.append(epoch_loss / len(rem_loader))

logger.info(f"Valutazione del modello finale su {len(testHoldFiles)} soggetti del hold-out set...")
model_hold.eval()
hold_dataset = GM3DDataset(testHoldFiles, y_hold_test, mean_rem, std_rem)
hold_loader = DataLoader(hold_dataset, batch_size=1)

y_prob_eff_hold = []
y_pred_eff_hold = []
with torch.no_grad():
    for inputs, _ in hold_loader:
        outputs = model_hold(inputs.to(DEVICE))
        probs = torch.softmax(outputs, dim=1)
        y_prob_eff_hold.append(probs[0, 1].item())
        y_pred_eff_hold.append(probs.argmax(dim=1).item())

accuracy_hold = accuracy_score(y_hold_test, y_pred_eff_hold)
auc_hold = roc_auc_score(y_hold_test, y_prob_eff_hold)
logger.success(f"Performance su Hold-Out Set: Accuracy = {accuracy_hold:.4f}, AUC = {auc_hold:.4f}")

logger.info("Creazione dei grafici di performance...")
plt.style.use('seaborn-v0_8-whitegrid')

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(final_loss_history)
ax1.set_title('EfficientNet Finale: Curva di Loss del Training')
ax1.set_xlabel('Epoca')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.grid(True)
plt.tight_layout()
fig1.savefig('efficientnet_final_training_loss.png', dpi=300)

fig2, ax2 = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_hold_test, y_pred_eff_hold)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=['CTRL', 'AD'], yticklabels=['CTRL', 'AD'])
ax2.set_title('Matrice di Confusione su Hold-Out Set')
ax2.set_xlabel('Predetto')
ax2.set_ylabel('Vero')
plt.tight_layout()
fig2.savefig('efficientnet_holdout_confusion_matrix.png', dpi=300)

fig3, ax3 = plt.subplots(figsize=(10, 6))
fpr, tpr, _ = roc_curve(y_hold_test, y_prob_eff_hold)
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_hold:.2f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax3.set_title('Curva ROC su Hold-Out Set')
ax3.set_xlabel('False Positive Rate'); ax3.set_ylabel('True Positive Rate')
ax3.legend(loc="lower right"); ax3.grid(True)
plt.tight_layout()
fig3.savefig('efficientnet_holdout_roc_curve.png', dpi=300)
logger.success("Grafici di performance salvati con successo.")
logger.success("SEZIONE 5 COMPLETATA.")

logger.info("INIZIO SEZIONE 6: Calcolo GradCAM e aggregazione dei risultati.")
gradcam_final = GradCAM(nn_module=model_hold, target_layers="_conv_head")
G_hold = np.zeros((len(testHoldFiles), M))

for h, (inputs, _) in tqdm(enumerate(hold_loader), total=len(hold_loader), desc="GradCAM on Hold-Out"):
    heatmap_3d = gradcam_final(x=inputs.to(DEVICE), class_idx=1)[0, 0, ...].cpu().numpy()
    G_hold[h, :] = heatmap_3d[mask3D]

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

ref_img = nib.load(trainValFiles[0])
nifti_image = nib.Nifti1Image(G_map_hold_3D, ref_img.affine, ref_img.header)
output_nii_file = 'GradCAM_map_aspects_holdout.nii'
nib.save(nifti_image, output_nii_file)

output_effnet_vectors_file = 'efficientnet_results.npz'
np.savez_compressed(output_effnet_vectors_file, Delta_G_hold=Delta_G_hold, G_hold=G_hold)
logger.success(f"Vettori GradCAM salvati in '{output_effnet_vectors_file}'")
logger.success(f"SEZIONE 6 COMPLETATA: Mappa NIfTI salvata in '{output_nii_file}'.")