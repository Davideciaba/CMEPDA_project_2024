import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import nibabel as nib
import os
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class NeuroimagingPreprocessor:
    """
    Classe per il preprocessing di dati di neuroimaging per analisi GM.
    
    Pipeline:
    1. Creazione maschera GM > 0
    2. Vettorializzazione 
    3. Normalizzazione con TIV
    4. Hold-out split
    5. Aggregazione in aspects tramite clustering gerarchico
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.gm_mask = None
        self.aspect_labels = None
        self.scaler = StandardScaler()
        self.n_aspects = None
        
    def create_gm_mask(self, gm_images: List[str], threshold: float = 0.0) -> np.ndarray:
        """
        Crea una maschera per voxels con intensità GM > threshold.
        
        Args:
            gm_images: Lista di percorsi alle immagini GM
            threshold: Soglia per intensità GM (default: 0.0)
            
        Returns:
            Maschera binaria 3D
        """
        print("Creazione maschera GM...")
        
        # Carica la prima immagine per ottenere le dimensioni
        first_img = nib.load(gm_images[0])
        img_shape = first_img.shape
        
        # Inizializza maschera
        mask_sum = np.zeros(img_shape)
        
        # Somma tutte le immagini GM
        for img_path in gm_images:
            img = nib.load(img_path)
            gm_data = img.get_fdata()
            mask_sum += (gm_data > threshold).astype(int)
        
        # Crea maschera finale (voxel presenti in almeno il 50% dei soggetti)
        self.gm_mask = mask_sum >= (len(gm_images) * 0.5)
        
        print(f"Maschera GM creata: {np.sum(self.gm_mask)} voxels selezionati")
        return self.gm_mask
    
    def vectorize_images(self, gm_images: List[str], mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Vettorializza le immagini GM applicando la maschera.
        
        Args:
            gm_images: Lista di percorsi alle immagini GM
            mask: Maschera da applicare (se None, usa self.gm_mask)
            
        Returns:
            Matrice (n_subjects x n_voxels)
        """
        if mask is None:
            mask = self.gm_mask
            
        if mask is None:
            raise ValueError("Nessuna maschera disponibile. Eseguire prima create_gm_mask()")
        
        print("Vettorializzazione immagini...")
        
        n_voxels = np.sum(mask)
        vectorized_data = np.zeros((len(gm_images), n_voxels))
        
        for i, img_path in enumerate(gm_images):
            img = nib.load(img_path)
            gm_data = img.get_fdata()
            vectorized_data[i] = gm_data[mask]
        
        print(f"Dati vettorializzati: {vectorized_data.shape}")
        return vectorized_data
    
    def normalize_with_tiv(self, vectorized_data: np.ndarray, tiv_values: List[float]) -> np.ndarray:
        """
        Normalizza i voxels con TIV_i / max(TIV).
        
        Args:
            vectorized_data: Dati vettorializzati
            tiv_values: Lista dei valori TIV per ogni soggetto
            
        Returns:
            Dati normalizzati
        """
        print("Normalizzazione con TIV...")
        
        tiv_array = np.array(tiv_values)
        max_tiv = np.max(tiv_array)
        
        # Normalizzazione: ogni soggetto viene moltiplicato per TIV_i / max(TIV)
        normalization_factors = tiv_array / max_tiv
        normalized_data = vectorized_data * normalization_factors.reshape(-1, 1)
        
        print(f"Normalizzazione completata. Range TIV: {np.min(tiv_array):.2f} - {np.max(tiv_array):.2f}")
        return normalized_data
    
    def hold_out_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide i dati in training+validation (80%) e test (20%) con stratificazione.
        
        Args:
            X: Features (dati normalizzati)
            y: Labels
            
        Returns:
            X_train_val, X_test, y_train_val, y_test
        """
        print("Hold-out split...")
        print(f"Dataset originale: {len(y)} soggetti")
        print(f"Distribuzione originale: {np.bincount(y)} (CTRL: {np.sum(y==0)}, AD: {np.sum(y==1)})")
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Training+Validation set: {X_train_val.shape[0]} soggetti")
        print(f"Test set: {X_test.shape[0]} soggetti")
        print(f"Distribuzione train+val: CTRL={np.sum(y_train_val==0)}, AD={np.sum(y_train_val==1)}")
        print(f"Distribuzione test: CTRL={np.sum(y_test==0)}, AD={np.sum(y_test==1)}")
        print(f"Proporzioni mantenute: {np.sum(y_train_val==0)/len(y_train_val):.3f} CTRL, {np.sum(y_train_val==1)/len(y_train_val):.3f} AD")
        
        return X_train_val, X_test, y_train_val, y_test
    
    def create_aspects_clustering(self, X_train_val: np.ndarray, n_aspects: int = 100, 
                                method: str = 'ward', criterion: str = 'maxclust') -> np.ndarray:
        """
        Crea aspects tramite correlazione di Spearman e clustering gerarchico.
        
        Args:
            X_train_val: Dati di training+validation
            n_aspects: Numero di aspects desiderato
            method: Metodo di linkage per clustering
            criterion: Criterio per determinare i cluster
            
        Returns:
            Array con label di aspect per ogni voxel
        """
        print("Creazione aspects tramite clustering gerarchico...")
        print(f"Calcolo correlazioni di Spearman per {X_train_val.shape[1]} voxels...")
        
        # Calcola matrice di correlazione di Spearman tra voxels
        n_voxels = X_train_val.shape[1]
        corr_matrix = np.zeros((n_voxels, n_voxels))
        
        # Calcola correlazioni in batch per efficienza
        batch_size = 1000
        for i in range(0, n_voxels, batch_size):
            end_i = min(i + batch_size, n_voxels)
            for j in range(i, n_voxels, batch_size):
                end_j = min(j + batch_size, n_voxels)
                
                # Calcola correlazioni per il batch corrente
                for vi in range(i, end_i):
                    for vj in range(max(j, vi), end_j):
                        if vi == vj:
                            corr_matrix[vi, vj] = 1.0
                        else:
                            corr, _ = spearmanr(X_train_val[:, vi], X_train_val[:, vj])
                            if np.isnan(corr):
                                corr = 0.0
                            corr_matrix[vi, vj] = corr
                            corr_matrix[vj, vi] = corr
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"Processati {end_i}/{n_voxels} voxels...")
        
        # Converti correlazioni in distanze
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Clustering gerarchico
        print("Esecuzione clustering gerarchico...")
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method=method)
        
        # Estrai cluster
        self.aspect_labels = fcluster(linkage_matrix, n_aspects, criterion=criterion)
        self.n_aspects = len(np.unique(self.aspect_labels))
        
        print(f"Creati {self.n_aspects} aspects")
        print(f"Distribuzione dimensioni aspects:")
        unique, counts = np.unique(self.aspect_labels, return_counts=True)
        print(f"Min: {np.min(counts)}, Max: {np.max(counts)}, Media: {np.mean(counts):.1f}")
        
        return self.aspect_labels
    
    def load_data_from_csv(self, csv_path: str, ctrl_folder: str, ad_folder: str, 
                          subject_column: int = 0, group_column: int = 1, tiv_column: int = 5,
                          file_extension: str = ".nii.gz") -> Tuple[List[str], List[int], List[float]]:
        """
        Carica dati dal CSV con associazione soggetti-gruppi-TIV.
        
        Args:
            csv_path: Percorso al file CSV
            ctrl_folder: Cartella con file CTRL
            ad_folder: Cartella con file AD  
            subject_column: Indice colonna con ID soggetto (default: 0)
            group_column: Indice colonna con gruppo CTRL/AD (default: 1)
            tiv_column: Indice colonna con valori TIV (default: 5)
            file_extension: Estensione file immagini (default: ".nii.gz")
            
        Returns:
            Tuple con (lista_percorsi_immagini, lista_labels, lista_tiv)
        """
        print(f"Caricamento dati da CSV: {csv_path}")
        
        try:
            # Carica CSV
            df = pd.read_csv(csv_path)
            print(f"CSV caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
            
            # Verifica colonne necessarie
            required_cols = max(subject_column, group_column, tiv_column) + 1
            if df.shape[1] < required_cols:
                raise ValueError(f"CSV ha solo {df.shape[1]} colonne, servono almeno {required_cols}")
            
            # Estrai dati
            subjects = df.iloc[:, subject_column].astype(str).tolist()
            groups = df.iloc[:, group_column].astype(str).tolist()
            tiv_values = df.iloc[:, tiv_column].astype(float).tolist()
            
            print(f"Soggetti estratti: {len(subjects)}")
            print(f"Gruppi unici: {set(groups)}")
            print(f"Range TIV: {min(tiv_values):.2f} - {max(tiv_values):.2f}")
            
            # Costruisci percorsi e labels
            gm_images = []
            labels = []
            final_tiv_values = []
            
            missing_files = []
            
            for i, (subject, group, tiv) in enumerate(zip(subjects, groups, tiv_values)):
                # Determina cartella e label
                if group.upper() in ['CTRL', 'CONTROL', 'CN', '0']:
                    folder = ctrl_folder
                    label = 0
                elif group.upper() in ['AD', 'ALZHEIMER', 'DEMENTIA', '1']:
                    folder = ad_folder  
                    label = 1
                else:
                    print(f"Attenzione: gruppo non riconosciuto '{group}' per soggetto {subject}")
                    continue
                
                # Costruisci percorso file
                filename = f"{subject}{file_extension}"
                filepath = os.path.join(folder, filename)
                
                # Verifica esistenza file
                if os.path.exists(filepath):
                    gm_images.append(filepath)
                    labels.append(label)
                    final_tiv_values.append(tiv)
                else:
                    missing_files.append(filepath)
            
            # Report finale
            print(f"\n=== REPORT CARICAMENTO ===")
            print(f"File trovati: {len(gm_images)}")
            print(f"CTRL: {labels.count(0)} soggetti")
            print(f"AD: {labels.count(1)} soggetti")
            print(f"File mancanti: {len(missing_files)}")
            
            if missing_files:
                print("\nPrimi 5 file mancanti:")
                for f in missing_files[:5]:
                    print(f"  - {f}")
                if len(missing_files) > 5:
                    print(f"  ... e altri {len(missing_files)-5}")
            
            print(f"TIV finale - Media: {np.mean(final_tiv_values):.2f} ± {np.std(final_tiv_values):.2f}")
            
            return gm_images, labels, final_tiv_values
            
        except Exception as e:
            print(f"Errore nel caricamento dal CSV: {e}")
            raise

    def load_tiv_from_csv(self, csv_path: str, tiv_column: int = 5) -> List[float]:
        """
        Carica i valori TIV dalla colonna specificata di un file CSV.
        DEPRECATA: Usa load_data_from_csv() per gestione completa.
        
        Args:
            csv_path: Percorso al file CSV
            tiv_column: Indice della colonna TIV (default: 5 per la sesta colonna)
            
        Returns:
            Lista dei valori TIV
        """
        print(f"Caricamento valori TIV da {csv_path}...")
        print("NOTA: Considera l'uso di load_data_from_csv() per gestione completa")
        
        try:
            # Carica CSV
            df = pd.read_csv(csv_path)
            print(f"CSV caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
            
            # Estrai colonna TIV (sesta colonna = indice 5)
            if df.shape[1] <= tiv_column:
                raise ValueError(f"Il CSV ha solo {df.shape[1]} colonne, impossibile accedere alla colonna {tiv_column+1}")
            
            tiv_values = df.iloc[:, tiv_column].tolist()
            
            print(f"Valori TIV estratti: {len(tiv_values)}")
            print(f"Range TIV: {min(tiv_values):.2f} - {max(tiv_values):.2f}")
            print(f"Media TIV: {np.mean(tiv_values):.2f} ± {np.std(tiv_values):.2f}")
            
            return tiv_values
            
        except Exception as e:
            print(f"Errore nel caricamento del CSV: {e}")
            raise
        """
        Restituisce informazioni sugli aspects creati.
        
        Returns:
            Dizionario con informazioni sugli aspects
        """
        if self.aspect_labels is None:
            return {}
        
        unique_aspects, counts = np.unique(self.aspect_labels, return_counts=True)
        
        return {
            'n_aspects': len(unique_aspects),
            'aspect_sizes': dict(zip(unique_aspects, counts)),
            'min_size': np.min(counts),
            'max_size': np.max(counts),
            'mean_size': np.mean(counts),
            'std_size': np.std(counts)
        }
    
    def save_preprocessing_results(self, output_dir: str):
        """
        Salva i risultati del preprocessing.
        
        Args:
            output_dir: Directory di output
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva maschera GM
        if self.gm_mask is not None:
            mask_img = nib.Nifti1Image(self.gm_mask.astype(np.uint8), affine=np.eye(4))
            nib.save(mask_img, os.path.join(output_dir, 'gm_mask.nii.gz'))
        
        # Salva aspect labels
        if self.aspect_labels is not None:
            np.save(os.path.join(output_dir, 'aspect_labels.npy'), self.aspect_labels)
            
            # Salva anche come immagine 3D
            if self.gm_mask is not None:
                aspect_img_3d = np.zeros(self.gm_mask.shape, dtype=np.int32)
                aspect_img_3d[self.gm_mask] = self.aspect_labels
                aspect_nii = nib.Nifti1Image(aspect_img_3d, affine=np.eye(4))
                nib.save(aspect_nii, os.path.join(output_dir, 'aspects_map.nii.gz'))
        
        # Salva informazioni
        info = self.get_aspect_info()
        pd.DataFrame([info]).to_csv(os.path.join(output_dir, 'preprocessing_info.csv'), index=False)
        
        print(f"Risultati salvati in: {output_dir}")

def run_preprocessing_pipeline(gm_images: List[str], labels: List[int], tiv_values: List[float],
                             n_aspects: int = 100, output_dir: Optional[str] = None) -> Dict:
    """
    Esegue la pipeline completa di preprocessing.
    
    Args:
        gm_images: Lista di percorsi alle immagini GM
        labels: Lista di label (0: CTRL, 1: AD)
        tiv_values: Lista dei valori TIV
        n_aspects: Numero di aspects da creare
        output_dir: Directory per salvare i risultati
        
    Returns:
        Dizionario con tutti i risultati del preprocessing
    """
    print("=== AVVIO PIPELINE PREPROCESSING ===")
    
    # Verifica coerenza dati
    if not (len(gm_images) == len(labels) == len(tiv_values)):
        raise ValueError(f"Lunghezze diverse: immagini={len(gm_images)}, labels={len(labels)}, TIV={len(tiv_values)}")
    
    # Inizializza preprocessor
    preprocessor = NeuroimagingPreprocessor()
    
    # 1. Crea maschera GM
    gm_mask = preprocessor.create_gm_mask(gm_images)
    
    # 2. Vettorializza immagini
    vectorized_data = preprocessor.vectorize_images(gm_images)
    
    # 3. Normalizza con TIV
    normalized_data = preprocessor.normalize_with_tiv(vectorized_data, tiv_values)
    
    # 4. Hold-out split
    y = np.array(labels)
    X_train_val, X_test, y_train_val, y_test = preprocessor.hold_out_split(normalized_data, y)
    
    # 5. Crea aspects tramite clustering
    aspect_labels = preprocessor.create_aspects_clustering(X_train_val, n_aspects)
    
    # Salva risultati se richiesto
    if output_dir:
        preprocessor.save_preprocessing_results(output_dir)
    
    # Prepara risultati
    results = {
        'preprocessor': preprocessor,
        'gm_mask': gm_mask,
        'X_train_val': X_train_val,
        'X_test': X_test,
        'y_train_val': y_train_val,
        'y_test': y_test,
        'aspect_labels': aspect_labels,
        'aspect_info': preprocessor.get_aspect_info()
    }
    
    print("=== PREPROCESSING COMPLETATO ===")
    return results

# Esempio di utilizzo con i tuoi dati reali
if __name__ == "__main__":
    # Configurazione per il tuo dataset
    print("=== CONFIGURAZIONE PER DATASET REALE ===")
    print("CTRL: 189 soggetti")
    print("AD: 144 soggetti") 
    print("Totale: 333 soggetti")
    print("Proporzione: 56.8% CTRL, 43.2% AD")
    
    # Esempio di come preparare i tuoi dati
    import glob
    
    # Percorsi alle cartelle (sostituisci con i tuoi percorsi effettivi)
    ctrl_folder = "./CMEPDA_project_2024/AD_CTRL/CTRL_s3"  # Cartella con 189 file CTRL
    ad_folder = "./CMEPDA_project_2024/AD_CTRL/AD_s3"      # Cartella con 144 file AD
    
    # Carica liste di file
    ctrl_files = sorted(glob.glob(os.path.join(ctrl_folder, "*.nii*")))
    ad_files = sorted(glob.glob(os.path.join(ad_folder, "*.nii*")))
    
    print(f"File CTRL trovati: {len(ctrl_files)}")
    print(f"File AD trovati: {len(ad_files)}")
    
    # Combina file e crea labels
    all_gm_images = ctrl_files + ad_files
    all_labels = [0] * len(ctrl_files) + [1] * len(ad_files)  # 0=CTRL, 1=AD
    
    # TIV values - carica dalla sesta colonna del CSV
    tiv_csv_path = "./CMEPDA_project_2024/covariateADCTRLsexAgeTIV.csv"  # Sostituisci con il percorso del tuo CSV
    
    # Inizializza preprocessor per usare la funzione di caricamento TIV
    temp_preprocessor = NeuroimagingPreprocessor()
    
    # DECOMMENTARE per caricare i valori TIV reali:
    all_tiv_values = temp_preprocessor.load_tiv_from_csv(tiv_csv_path, tiv_column=5)
    
    # Valori simulati per test (DA SOSTITUIRE con la riga sopra)
    #all_tiv_values = np.random.normal(1500, 200, len(all_gm_images)).tolist()
    
    print(f"Dataset preparato:")
    print(f"- Immagini GM: {len(all_gm_images)}")
    print(f"- Labels: {len(all_labels)} (CTRL: {all_labels.count(0)}, AD: {all_labels.count(1)})")
    print(f"- TIV values: {len(all_tiv_values)}")
    
    # Esegui preprocessing
    print("\n=== AVVIO PREPROCESSING ===")
    #DECOMMENTARE le righe seguenti per eseguire con dati reali
    results = run_preprocessing_pipeline(
                gm_images=all_gm_images,
                 labels=all_labels,
                 tiv_values=all_tiv_values,
                 n_aspects=100,
                 output_dir='preprocessing_results_333subjects'
                )
    
    # Lo split viene gestito automaticamente dalla pipeline!
    # La funzione run_preprocessing_pipeline() chiamerà hold_out_split()
    # che farà la divisione stratificata 80/20 mantenendo le proporzioni
    
    print("\nNOTA: Lo split training/test viene gestito automaticamente dalla pipeline")
    print("con stratificazione per mantenere le proporzioni CTRL/AD in entrambi i set.")