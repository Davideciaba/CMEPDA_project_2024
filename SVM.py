# %% STEP 7: SVM ANALYSIS WITH PYTORCH (CON RANDOMIZED SEARCH)
# Author: Gemini
# Date: 16/10/2025
# --------------------------------------------------------------------------
# QUESTA VERSIONE SOSTITUISCE LA GRID SEARCH MANUALE PER IL PARAMETRO C
# CON UNA PIÙ EFFICIENTE RANDOMIZED SEARCH.

import sys
import h5py
import numpy as np
from scipy.stats import mode, loguniform # <-- Import aggiunto
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from loguru import logger
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import nibabel as nib
except ImportError:
    logger.critical("Libreria NiBabel non trovata. Installala con: pip install nibabel")
    sys.exit(1)

# --- 0. Configurazione del Logger e Dispositivo PyTorch ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Dispositivo PyTorch in uso: {DEVICE}")

# --- 1. Definizione dei Percorsi dei File di Input ---
split_data_file = 'split_data_and_indices.npz'
mat_file_path = '../CMEPDA_Project_2024/MATLAB_preliminaries/preliminaries_output.mat'
clustering_results_file = 'hierarchical_clustering_results.npz'
reference_nii_path = 'C:/spm12/canonical/avg152T1.nii' # ESEMPIO: Sostituisci con un percorso valido

# --- 2. Caricamento Dati ---
logger.info("Caricamento di tutti i dati necessari dai file di preprocessing...")
try:
    # Carica i dati NON SCALATI e gli indici dal nuovo Split.py
    with np.load(split_data_file) as data:
        X_rem, y_rem = data['X_rem'], data['y_rem']
        X_hold_test, y_hold_test = data['X_hold_test'], data['y_hold_test']
    
    with np.load(clustering_results_file) as data:
        cluster_labels_voxel = data['cluster_labels_voxel']
        K = int(data['num_clusters'])
    with h5py.File(mat_file_path, 'r') as f:
        mask3D = f['mask'][()].astype(bool) # Assicura che la maschera sia booleana
        voxelIdx = f['voxelIdx'][()].ravel().astype(int) - 1
    n_rem, M = X_rem.shape
    logger.success("Tutti i file sono stati caricati con successo.")
except FileNotFoundError as e:
    logger.critical(f"File di input non trovato: {e}. Assicurati di aver eseguito tutti gli script precedenti.")
    sys.exit(1)


# --- 3. Modello SVM e Funzione di Training con PyTorch ---
class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

def train_svm(model, X_train, y_train, c_value, lr=0.001, epochs=100):
    weight_decay = 1 / (c_value * len(y_train))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_train_hinge = torch.tensor(np.where(y_train == 0, -1, 1), dtype=torch.float32).to(DEVICE).unsqueeze(1)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    
    loss_history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = torch.mean(torch.clamp(1 - y_train_hinge * outputs, min=0))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return model, loss_history

# ==========================================================================
# SEZIONE 4: LOOCV INTERNO CON RANDOMIZED SEARCH PER IL TUNING DI C
# ==========================================================================
logger.info("INIZIO SEZIONE 4: Esecuzione del LOOCV interno con Randomized Search...")
W_all_rem = np.zeros((n_rem, M))
bestC_int_all = np.zeros(n_rem)

# --- INIZIO MODIFICHE PER RANDOMIZED SEARCH ---
# 1. Definiamo una distribuzione da cui campionare i valori di C.
# loguniform è ideale per parametri che variano su ordini di grandezza.
distribution = loguniform(0.01, 100)
# 2. Definiamo quanti candidati casuali testare per ogni fold.
n_random_candidates = 10
logger.info(f"Verranno testati {n_random_candidates} candidati 'C' casuali per fold da una distribuzione log-uniforme [0.01, 100].")
# --- FINE MODIFICHE ---


for k in tqdm(range(n_rem), desc="SVM (PyTorch) Internal LOOCV"):
    val_idx = k
    train_indices = np.delete(np.arange(n_rem), k)
    X_train_int_raw, y_train_int = X_rem[train_indices, :], y_rem[train_indices]
    X_val_int_raw, y_val_int = X_rem[val_idx, :].reshape(1, -1), y_rem[val_idx]

    # La standardizzazione viene fatta correttamente DENTRO il loop
    scaler_int = StandardScaler()
    X_train_int_std = scaler_int.fit_transform(X_train_int_raw)
    X_val_int_std = scaler_int.transform(X_val_int_raw)
    
    # Campiona N candidati casuali per questo specifico fold
    C_candidates = distribution.rvs(n_random_candidates)
    
    best_c_int, best_val_acc_int = -1, -1
    for c_value in C_candidates:
        svm_c = SVM(input_dim=M).to(DEVICE)
        svm_c, _ = train_svm(svm_c, X_train_int_std, y_train_int, c_value)
        with torch.no_grad():
            val_tensor = torch.tensor(X_val_int_std, dtype=torch.float32).to(DEVICE)
            output = svm_c(val_tensor).item()
            pred = 1 if output >= 0 else 0
            val_acc = 1 if pred == y_val_int else 0
        if val_acc > best_val_acc_int:
            best_val_acc_int = val_acc
            best_c_int = c_value
    bestC_int_all[k] = best_c_int
    
    X_trval_int_std = np.vstack([X_train_int_std, X_val_int_std])
    y_trval_int = np.append(y_train_int, y_val_int)
    
    svm_int_final = SVM(input_dim=M).to(DEVICE)
    svm_int_final, _ = train_svm(svm_int_final, X_trval_int_std, y_trval_int, best_c_int)
    W_all_rem[k, :] = svm_int_final.linear.weight.data.cpu().numpy().flatten()
logger.success("SEZIONE 4 COMPLETATA: LOOCV interno terminato.")

# ==========================================================================
# SEZIONE 5: MODELLO FINALE E VALUTAZIONE SULL'HOLD-OUT SET (20%)
# ==========================================================================
logger.info("INIZIO SEZIONE 5: Addestramento del modello finale e valutazione.")
C_final = mode(bestC_int_all).mode # .mode[0] non è più necessario
logger.info(f"Iperparametro finale scelto (moda dei fold): C = {C_final}")

scaler_final = StandardScaler()
X_rem_std = scaler_final.fit_transform(X_rem)
X_hold_std = scaler_final.transform(X_hold_test)

logger.info(f"Addestramento del modello SVM (PyTorch) finale su {n_rem} soggetti...")
svm_final_hold = SVM(input_dim=M).to(DEVICE)
svm_final_hold, loss_history = train_svm(svm_final_hold, X_rem_std, y_rem, C_final, epochs=200)
w_final_hold = svm_final_hold.linear.weight.data.cpu().numpy().flatten()

logger.info(f"Valutazione del modello finale su {len(y_hold_test)} soggetti del hold-out set...")
with torch.no_grad():
    X_hold_tensor = torch.tensor(X_hold_std, dtype=torch.float32).to(DEVICE)
    scores_svm_hold = svm_final_hold(X_hold_tensor).cpu().numpy().flatten()

y_pred_svm_hold = (scores_svm_hold >= 0).astype(int)
accuracy_hold = accuracy_score(y_hold_test, y_pred_svm_hold)
auc_hold = roc_auc_score(y_hold_test, scores_svm_hold)
logger.success(f"Performance su Hold-Out Set: Accuracy = {accuracy_hold:.4f}, AUC = {auc_hold:.4f}")

# --- SEZIONE GRAFICI (invariata) ---
logger.info("Creazione dei grafici di performance...")
plt.style.use('seaborn-v0_8-whitegrid')

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(loss_history)
ax1.set_title('SVM Finale: Curva di Loss del Training')
ax1.set_xlabel('Epoca')
ax1.set_ylabel('Hinge Loss')
ax1.grid(True)
plt.tight_layout()
fig1.savefig('svm_final_training_loss.png', dpi=300)
logger.success("Grafico della curva di loss salvato in 'svm_final_training_loss.png'")

fig2, ax2 = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_hold_test, y_pred_svm_hold)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Control (CTRL)', 'Alzheimer (AD)'],
            yticklabels=['Control (CTRL)', 'Alzheimer (AD)'])
ax2.set_title('Matrice di Confusione su Hold-Out Set')
ax2.set_xlabel('Predetto')
ax2.set_ylabel('Vero')
plt.tight_layout()
fig2.savefig('svm_holdout_confusion_matrix.png', dpi=300)
logger.success("Grafico della matrice di confusione salvato in 'svm_holdout_confusion_matrix.png'")

fig3, ax3 = plt.subplots(figsize=(10, 6))
fpr, tpr, _ = roc_curve(y_hold_test, scores_svm_hold)
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_hold:.2f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('Curva ROC su Hold-Out Set')
ax3.legend(loc="lower right")
ax3.grid(True)
plt.tight_layout()
fig3.savefig('svm_holdout_roc_curve.png', dpi=300)
logger.success("Grafico della curva ROC salvato in 'svm_holdout_roc_curve.png'")
logger.success("SEZIONE 5 COMPLETATA: Modello finale addestrato e valutato.")

# ==========================================================================
# SEZIONE 6: AGGREGAZIONE DEI PESI IN ASPETTI E MAPPA NIfTI
# ==========================================================================
logger.info("INIZIO SEZIONE 6: Aggregazione dei pesi e creazione della mappa NIfTI.")
W_bar_hold = np.zeros(K)
for a in range(1, K + 1):
    vox_indices_in_aspect = np.where(cluster_labels_voxel == a)[0]
    if len(vox_indices_in_aspect) > 0:
        W_bar_hold[a - 1] = np.sum(w_final_hold[vox_indices_in_aspect])

logger.info("Creazione della mappa 3D dei pesi aggregati...")
flat_map = W_bar_hold[cluster_labels_voxel - 1]
W_map_hold_3D = np.zeros(mask3D.shape, dtype=np.float32)
W_map_hold_3D[mask3D] = flat_map

try:
    ref_img = nib.load(reference_nii_path)
    nifti_image = nib.Nifti1Image(W_map_hold_3D, ref_img.affine, ref_img.header)
    output_nii_file = 'SVM_PyTorch_weight_map_aspects_holdout.nii'
    nib.save(nifti_image, output_nii_file)
    logger.success(f"Mappa NIfTI salvata con successo in: '{output_nii_file}'")
except FileNotFoundError:
    logger.error(f"File NIfTI di riferimento non trovato in '{reference_nii_path}'.")

output_svm_vectors_file = 'svm_results.npz'
np.savez_compressed(output_svm_vectors_file,
                    W_bar_hold=W_bar_hold,
                    w_final_hold=w_final_hold)
logger.success(f"Vettori di importanza SVM salvati in: '{output_svm_vectors_file}'")

logger.success("SEZIONE 6 COMPLETATA: Pipeline SVM (PyTorch) terminata.")