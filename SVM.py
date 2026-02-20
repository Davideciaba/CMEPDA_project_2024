import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

def pipeline_svm_cmepda(X, y, n_splits_ext=5, n_splits_int=5):
    """
    Pipeline SVM lineare con Nested CV e calcolo del pattern di Haufe.
    
    Riferimenti:
    - Pipeline CMEPDA [cite: 1, 12, 18]
    - Haufe et al., 2014 [cite: 112]
    
    Args:
        X (np.array): Array 2D (n_soggetti, n_voxel). Mappe GM flattate[cite: 20].
        y (np.array): Array 1D etichette (0=CTRL, 1=AD).
        n_splits_ext (int): Fold per CV esterna[cite: 13].
        n_splits_int (int): Fold per CV interna (tuning)[cite: 14].

    Returns:
        metrics (dict): Dizionario con liste di metriche per ogni fold.
        raw_weights (list): Lista dei vettori w (pesi grezzi).
        haufe_patterns (list): Lista delle pattern maps 'a'[cite: 28].
        best_c_values (list): Lista degli iperparametri C scelti[cite: 16].
    """
    
    # 1. Definizione dello schema di Nested Cross-Validation [cite: 13]
    cv_esterno = StratifiedKFold(n_splits=n_splits_ext, shuffle=True, random_state=42)
    cv_interno = StratifiedKFold(n_splits=n_splits_int, shuffle=True, random_state=42)
    
    # Griglia iperparametro C (linear SVM soft-margin) [cite: 21]
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    
    # Inizializzazione contenitori risultati
    metrics = {
        'accuracy': [], 
        'balanced_accuracy': [], 
        'auc': [],
        'sensitivity': [], # Richiesto da 
        'specificity': []  # Richiesto da 
    }
    haufe_patterns = []
    raw_weights = []
    best_c_values = []
    
    print(f"Avvio pipeline SVM su {X.shape[0]} soggetti e {X.shape[1]} voxel.")

    # Ciclo CV Esterna [cite: 13]
    for fold_idx, (train_idx, test_idx) in enumerate(cv_esterno.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 2. Tuning Iperparametro (CV Interna) [cite: 14]
        svm_base = SVC(kernel='linear', class_weight='balanced', probability=True) # [cite: 21]
        
        # Nota: n_jobs=-1 può causare memory overflow con MRI ad alta ris. 
        # Se capita, impostare n_jobs=1.
        grid_search = GridSearchCV(
            estimator=svm_base, 
            param_grid=param_grid, 
            cv=cv_interno, 
            scoring='balanced_accuracy', 
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_svm = grid_search.best_estimator_
        best_c_values.append(grid_search.best_params_['C'])
        
        # 3. Predizione e Metriche sul Test Set 
        y_pred = best_svm.predict(X_test)
        y_prob = best_svm.predict_proba(X_test)[:, 1] # Probabilità per classe 1 (AD)
        
        # Calcolo metriche base
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Calcolo Sensibilità e Specificità tramite matrice di confusione 
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics['accuracy'].append(acc)
        metrics['balanced_accuracy'].append(bal_acc)
        metrics['auc'].append(auc)
        metrics['sensitivity'].append(sensitivity)
        metrics['specificity'].append(specificity)
        
        # 4. XAI: Pesi Grezzi e Pattern di Haufe
        
        # 4a. Pesi grezzi w 
        # w shape: (1, n_voxel) -> convertiamo a (n_voxel,)
        w = best_svm.coef_.flatten()
        raw_weights.append(w)
        
        # 4b. Pattern di Haufe 
        # Formula teorica: a = Sigma_x * w
        # Implementazione efficiente: Cov(X, s) dove s è il latent factor (decision function)
        # Riferimento Haufe 2014, Eq. 7  e Appendix C[cite: 846].
        
        # Calcolo dei latent factors (s) sul training set
        s_train = best_svm.decision_function(X_train)
        
        # Centratura dei dati (essenziale per la covarianza)
        X_train_centered = X_train - np.mean(X_train, axis=0)
        s_train_centered = s_train - np.mean(s_train)
        
        # Correzione Bug: usare shape[0] per il numero di campioni
        n_samples_train = X_train.shape[0]
        
        # Calcolo Covarianza vettorizzata: (X^T * s) / (N-1)
        # Questo evita di costruire la matrice Sigma_x (N_voxel x N_voxel)
        haufe_pattern = np.dot(X_train_centered.T, s_train_centered) / (n_samples_train - 1)
        
        haufe_patterns.append(haufe_pattern)
        
        print(f"Fold {fold_idx+1}/{n_splits_ext} completato. Best C: {grid_search.best_params_['C']}")
        
    return metrics, raw_weights, haufe_patterns, best_c_values

# Esempio di utilizzo con dati dummy
if __name__ == "__main__":
    # Simulazione: 100 soggetti, 5000 voxel
    n_subj = 100
    n_vox = 5000
    X_dummy = np.random.randn(n_subj, n_vox)
    y_dummy = np.random.randint(0, 2, n_subj)

    met, pesi, patterns, C_opt = pipeline_svm_cmepda(X_dummy, y_dummy)
    
    print("\n--- Risultati Medi ---")
    print(f"Balanced Accuracy: {np.mean(met['balanced_accuracy']):.3f}")
    print(f"Sensitivity: {np.mean(met['sensitivity']):.3f}")
    print(f"Specificity: {np.mean(met['specificity']):.3f}")