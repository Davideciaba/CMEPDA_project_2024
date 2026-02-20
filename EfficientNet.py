import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from monai.networks.nets import EfficientNetBN

# ==========================================
# 1. Definizione del Modello (dal codice precedente)
# ==========================================
def build_efficientnet_3d(model_name="efficientnet-b0", in_channels=1, num_classes=2):
    return EfficientNetBN(
        model_name=model_name,
        spatial_dims=3,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False # False per test locale veloce
    )

# ==========================================
# 2. Funzione di Training con Hold-out interno ed Early Stopping
# ==========================================
def train_and_evaluate_fold(X_train, y_train, X_test, y_test, device, batch_size=4, max_epochs=10, patience=3):
    # Split del training in train_inner e val_inner (Hold-out interno per Early Stopping)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    
    # Creazione DataLoader
    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Aggiungi drop_last=True almeno ai loader di training e validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Sul test non è necessario perché siamo in model.eval()
    
    # Inizializzazione modello, loss e ottimizzatore
    model = build_efficientnet_3d().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Variabili per Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("   Inizio training fold...")
    for epoch in range(max_epochs):
        # -- TRAINING --
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # -- VALIDATION (Early Stopping) --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        # Check Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Salvare i pesi del modello migliore qui (opzionale per dummy test)
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"   Early stopping all'epoca {epoch+1} (Val Loss: {val_loss:.4f})")
            break
            
    # -- TEST SUL FOLD ESTERNO --
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs) # Restituisce i logit
            probs = torch.softmax(logits, dim=1)[:, 1] # Probabilità classe AD (1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    # Calcolo Metriche
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
        'auc': roc_auc_score(all_targets, all_probs)
    }
    
    return metrics

# ==========================================
# 3. Dummy Test Executer
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione su: {device}")
    
    # Generazione dati dummy leggeri per portatile
    # 20 soggetti, 1 canale, volumi 3D di 32x32x32 voxel
    n_subj = 20
    # Passiamo da 32x32x32 a 64x64x64 per evitare il collasso spaziale nella rete
    X_dummy = torch.randn(n_subj, 1, 64, 64, 64) 
    y_dummy = torch.randint(0, 2, (n_subj,))
    
    # K-Fold CV Esterna (Es. 3-fold per fare veloce)
    n_splits_ext = 3
    skf = StratifiedKFold(n_splits=n_splits_ext, shuffle=True, random_state=42)
    
    fold_metrics = {'accuracy': [], 'balanced_accuracy': [], 'auc': []}
    
    print(f"\nAvvio Cross-Validation Esterna a {n_splits_ext} fold...")
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_dummy, y_dummy)):
        print(f"\n--- Fold Esterno {fold_idx + 1}/{n_splits_ext} ---")
        
        X_train, X_test = X_dummy[train_idx], X_dummy[test_idx]
        y_train, y_test = y_dummy[train_idx], y_dummy[test_idx]
        
        metrics = train_and_evaluate_fold(
            X_train, y_train, X_test, y_test, 
            device=device, 
            batch_size=2,   # Batch piccolo per RAM limitata
            max_epochs=5,   # Poche epoche per il test
            patience=2
        )
        
        print(f"   Risultati Test Fold {fold_idx + 1}:")
        print(f"   Accuracy: {metrics['accuracy']:.3f} | Bal. Acc: {metrics['balanced_accuracy']:.3f} | AUC: {metrics['auc']:.3f}")
        
        fold_metrics['accuracy'].append(metrics['accuracy'])
        fold_metrics['balanced_accuracy'].append(metrics['balanced_accuracy'])
        fold_metrics['auc'].append(metrics['auc'])

    print("\n=== RISULTATI FINALI MEDI ===")
    print(f"Accuracy Media: {np.mean(fold_metrics['accuracy']):.3f}")
    print(f"Balanced Accuracy Media: {np.mean(fold_metrics['balanced_accuracy']):.3f}")
    print(f"AUC ROC Media: {np.mean(fold_metrics['auc']):.3f}")