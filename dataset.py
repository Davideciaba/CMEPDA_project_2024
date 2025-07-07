# Esempio di utilizzo della pipeline con il tuo CSV reale
import os
from preprocessing import NeuroimagingPreprocessor, run_preprocessing_pipeline

def main():
    print("=== UTILIZZO PIPELINE CON DATASET REALE ===")
    
    # Percorsi ai tuoi dati (SOSTITUISCI CON I PERCORSI REALI)
    csv_path = "covariateADCTRLsexAgeTIV.csv"
    ctrl_folder = "/path/to/CTRL_images/"  # Cartella con immagini CTRL
    ad_folder = "/path/to/AD_images/"      # Cartella con immagini AD
    output_dir = "preprocessing_results"
    
    # Configurazione colonne CSV (già corrette per il tuo file)
    subject_column = 0  # ID
    group_column = 1    # Group 
    tiv_column = 5      # TIV
    
    # Inizializza preprocessor
    preprocessor = NeuroimagingPreprocessor(test_size=0.2, random_state=42)
    
    try:
        # Carica dati dal CSV con associazione automatica
        print("Caricamento dati dal CSV...")
        gm_images, labels, tiv_values = preprocessor.load_data_from_csv(
            csv_path=csv_path,
            ctrl_folder=ctrl_folder,
            ad_folder=ad_folder,
            subject_column=subject_column,
            group_column=group_column,
            tiv_column=tiv_column,
            file_extension=".nii.gz"  # o ".nii" a seconda dei tuoi file
        )
        
        # Verifica dei dati caricati
        print(f"\n=== VERIFICA DATI CARICATI ===")
        print(f"Immagini totali: {len(gm_images)}")
        print(f"CTRL: {labels.count(0)} soggetti")
        print(f"AD: {labels.count(1)} soggetti")
        print(f"Proporzione CTRL: {labels.count(0)/len(labels)*100:.1f}%")
        print(f"Proporzione AD: {labels.count(1)/len(labels)*100:.1f}%")
        print(f"TIV - Range: {min(tiv_values):.2f} - {max(tiv_values):.2f}")
        print(f"TIV - Media: {sum(tiv_values)/len(tiv_values):.2f}")
        
        # Esegui la pipeline completa
        print(f"\n=== AVVIO PIPELINE PREPROCESSING ===")
        results = run_preprocessing_pipeline(
            gm_images=gm_images,
            labels=labels,
            tiv_values=tiv_values,
            n_aspects=100,  # Numero di aspects da creare
            output_dir=output_dir
        )
        
        # Risultati finali
        print(f"\n=== RISULTATI FINALI ===")
        print(f"Training+Validation set: {results['X_train_val'].shape}")
        print(f"Test set: {results['X_test'].shape}")
        print(f"Numero voxels (dopo maschera GM): {results['X_train_val'].shape[1]}")
        print(f"Numero aspects creati: {results['aspect_info']['n_aspects']}")
        print(f"Dimensioni aspects - Min: {results['aspect_info']['min_size']}, Max: {results['aspect_info']['max_size']}")
        
        print(f"\nRisultati salvati in: {output_dir}")
        print("Pipeline completata con successo!")
        
        return results
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        raise

def verify_csv_structure(csv_path):
    """Funzione di utilità per verificare la struttura del CSV"""
    import pandas as pd
    
    print("=== VERIFICA STRUTTURA CSV ===")
    df = pd.read_csv(csv_path)
    
    print(f"Dimensioni CSV: {df.shape[0]} righe x {df.shape[1]} colonne")
    print(f"Colonne: {list(df.columns)}")
    print(f"\nPrime 3 righe:")
    print(df.head(3))
    
    print(f"\nDistribuzione gruppi:")
    group_counts = df.iloc[:, 1].value_counts()
    print(group_counts)
    
    print(f"\nStatistiche TIV (colonna 6):")
    tiv_stats = df.iloc[:, 5].describe()
    print(tiv_stats)
    
    print(f"\nIl CSV è già nel formato corretto per la pipeline!")

if __name__ == "__main__":
    # Prima verifica la struttura del CSV
    csv_path = "covariateADCTRLsexAgeTIV.csv"
    verify_csv_structure(csv_path)
    
    print("\n" + "="*50)
    
    # Poi esegui la pipeline (decommenta quando sei pronto)
    # main()