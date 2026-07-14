import sys
import pathlib
import numpy as np
import nibabel as nib
import torch
import pandas as pd

current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.append(str(project_root))

from monai.data import Dataset, DataLoader
from Python.utils.py_logger import CustomLogger
from Python.utils.cv_manager import CVManager
from Python.Models.efficientnet_classifier import EfficientNetClassifier
from Python.XAI.xai_efficientnet import EfficientNetExplainer
from Python.utils.model_renderer import ModelRenderer

def run_efficientnet_xai():
    CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_DIR = CURRENT_DIR.parent.parent
    results_dir = CURRENT_DIR / "Results"
    SETUP_DIR = PROJECT_DIR / "Python" / "Common_Setup"
    XAI_DIR = results_dir / "XAI_Maps"
    XAI_PLOTS_DIR = CURRENT_DIR / "Plots" / "XAI_Visualizations"
    mask_path = SETUP_DIR / "Mask" / "tpm_mask.nii"

    XAI_DIR.mkdir(parents=True, exist_ok=True)
    XAI_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log = CustomLogger(name="XAIPipeline")
    log.add_console_handler(level="DEBUG", use_colors=True)
    log.info("--- Booting EfficientNet Integrated Gradients Engine ---")

    # 1. Configurazione
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PARAM_GRID_MOCK = {'lr': [1e-4], 'wd': [1e-3], 'optimizer': ['adamw'], 'scheduler': ['none']}
    
    engine = EfficientNetClassifier(logger=log, device=device, param_grid=PARAM_GRID_MOCK)
    explainer = EfficientNetExplainer(logger=log, device=device)
    renderer = ModelRenderer(logger=log, output_dir=str(XAI_PLOTS_DIR))

    # Caricamento Dati
    registry_csv_path = SETUP_DIR / "python_registry.csv"
    subjects, data_dicts, y_full = engine.load_data(str(registry_csv_path))
    cv_splits = CVManager.load_from_json(str(SETUP_DIR / "cv_folds_registry.json"))

    registry_df = pd.read_csv(registry_csv_path)
    ctrl_candidates = registry_df[registry_df['subject_id'].str.contains("CTRL-117")]
    bg_path = str(ctrl_candidates.iloc[0]['file_path']) if not ctrl_candidates.empty else None

    # Estrazione dell'Affine da un file campione (necessario per salvare i NIfTI)
    sample_img = nib.load(data_dicts[0]['image'])
    reference_affine = sample_img.affine
    original_shape = sample_img.shape # Dovrebbe essere (121, 145, 121)
 

    slice_config = 3.0

    # --------------------------------------------------------------------------
    # 2. SELEZIONE DEI SOGGETTI LOCALI CO-OCCORRENTI (1 AD, 1 CTRL)
    # Per garantire che AD e CTRL siano nello stesso Training Set il maggior numero 
    # di volte possibile (4 su 5), li peschiamo dinamicamente tra i soggetti che 
    # sono finiti nello STESSO Test Set (qui usiamo il Test Set del Fold 1).
    # In questo modo, per i restanti 4 Folds, saranno entrambi nel Training Set.
    # --------------------------------------------------------------------------
    test_idx_fold_1 = cv_splits[0]['outer_test_idx']
    test_subjects_fold_1 = subjects[test_idx_fold_1]
    test_labels_fold_1 = y_full[test_idx_fold_1]
    
    # Prende il primo AD e il primo CTRL trovati in quel test set
    TARGET_AD_SUBJECT = test_subjects_fold_1[test_labels_fold_1 == 1][0]
    TARGET_CTRL_SUBJECT = test_subjects_fold_1[test_labels_fold_1 == 0][0]
    
    log.info(f"Target Local Subjects (Co-occurring in 4/5 Train Folds) -> AD: {TARGET_AD_SUBJECT} | CTRL: {TARGET_CTRL_SUBJECT}")

    # 3. CICLO SUI FOLD PER ESTRAZIONE IG
    for split in cv_splits:
        fold_idx = split['fold']
        log.info(f"--- Extracting Integrated Gradients for Outer Fold {fold_idx} ---")
        
        train_idx = split['outer_train_idx']
        
        # Creiamo un subset specifico per il train_set. 
        # IMPORTANTE: batch_size=1 e shuffle=False per sapere esattamente chi stiamo spiegando
        train_dicts = [data_dicts[i] for i in train_idx]
        train_dataset = Dataset(data=train_dicts, transform=engine._get_transforms())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        train_subjects = subjects[train_idx]
        
        # Inizializza la mappa globale vuota (con le dimensioni originali MNI)
        global_ig_map = np.zeros(original_shape, dtype=np.float32)

        # Carica il modello addestrato per questo fold (Simulato: devi assicurarti che il modello 
        # sia stato salvato su disco in execute_nested_cv o fornito qui. 
        # Esempio: model.load_state_dict(torch.load(f"Models/fold_{fold_idx}.pth")))
        model = engine._prepare_model()
        model_path = CURRENT_DIR / "Results" / f"EfficientNet_Fold_{fold_idx}.pth"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            log.warning(f"Model for Fold {fold_idx} not found at {model_path}. Using uninitialized weights for demonstration!")
        
        model.eval()

        for step, batch_data in enumerate(train_loader):
            curr_subject = train_subjects[step]
            curr_label = batch_data["label"].item() # 0 o 1
            input_tensor = batch_data["image"].to(device)

            # 1. Calcolo IG (Restituisce tensore paddato 160x160x160)
            ig_map_padded = explainer.compute_integrated_gradients(model, input_tensor, target_class=curr_label, steps=50)

            # 2. Inverse Crop -> Torna a 121x145x121
            ig_map = explainer.remove_symmetric_padding(ig_map_padded, original_shape)

            # 3. Aggiorna la Mappa Globale (Somma dei Valori Assoluti)
            global_ig_map += np.abs(ig_map)

            # 4. Estrazione Mappe Locali (Se è uno dei due soggetti target)
            if curr_subject in [TARGET_AD_SUBJECT, TARGET_CTRL_SUBJECT]:
                log.info(f"Target Subject {curr_subject} found in Fold {fold_idx} train set. Extracting Local Map...")
                
                # Nomenclatura chiara
                status = "AD" if curr_subject == TARGET_AD_SUBJECT else "CTRL"
                local_filename_top1 = f"Local_IG_Fold{fold_idx}_{status}_{curr_subject}_Top1.nii"
                local_nifti_path_top1 = XAI_DIR / local_filename_top1
                local_filename_top5 = f"Local_IG_Fold{fold_idx}_{status}_{curr_subject}_Top5.nii"
                local_nifti_path_top5 = XAI_DIR / local_filename_top5
                
                # Salva Mappa Locale mantenendo IL SEGNO ORIGINALE
                local_ig_map_top1 = np.where(np.abs(ig_map) >= np.percentile(np.abs(ig_map), 99), ig_map, 0)
                local_ig_map_top5 = np.where(np.abs(ig_map) >= np.percentile(np.abs(ig_map), 95), ig_map, 0)
                explainer.reconstruct_nifti(local_ig_map_top1, reference_affine, str(local_nifti_path_top1))
                explainer.reconstruct_nifti(local_ig_map_top5, reference_affine, str(local_nifti_path_top5))
                
                # Renderizza la Mappa Locale
                try:
                    renderer.plot_3d_activation_map(
                        bg_nifti_path=bg_path,
                        stats_nifti_path=str(local_nifti_path_top1),
                        mask_nifti_path=mask_path,
                        map_title=f"IG Map - {status} ({curr_subject}) - Fold {fold_idx} Top 1%",
                        export_filename=f"Render_Local_{status}_{curr_subject}_Fold{fold_idx}_Top1.png",
                        slice_config=slice_config
                    )
                    renderer.plot_3d_activation_map(
                        bg_nifti_path=bg_path,
                        stats_nifti_path=str(local_nifti_path_top5),
                        mask_nifti_path=mask_path,
                        map_title=f"IG Map - {status} ({curr_subject}) - Fold {fold_idx} Top 5%",
                        export_filename=f"Render_Local_{status}_{curr_subject}_Fold{fold_idx}_Top5.png",
                        slice_config=slice_config
                    )
                except Exception as e:
                    log.warning(f"Could not render Local Map for {curr_subject}: {e}")

        # 5. Salvataggio e Rendering della Mappa Globale del Fold
        # (Opzionale ma consigliato: media i valori assoluti per il numero di soggetti)
        global_ig_map /= len(train_subjects)
        
        global_filename_top1 = f"Global_IG_Fold{fold_idx}_AbsMean_Top1.nii"
        global_filename_top5 = f"Global_IG_Fold{fold_idx}_AbsMean_Top5.nii"
        global_nifti_path_top1 = XAI_DIR / global_filename_top1
        global_nifti_path_top5 = XAI_DIR / global_filename_top5
        global_ig_map_top1 = np.where(np.abs(global_ig_map) >= np.percentile(np.abs(global_ig_map), 99), global_ig_map, 0)
        global_ig_map_top5 = np.where(np.abs(global_ig_map) >= np.percentile(np.abs(global_ig_map), 95), global_ig_map, 0)
        explainer.reconstruct_nifti(global_ig_map_top1, reference_affine, str(global_nifti_path_top1))
        explainer.reconstruct_nifti(global_ig_map_top5, reference_affine, str(global_nifti_path_top5))
        
        try:
            renderer.plot_3d_activation_map(
                bg_nifti_path=bg_path, 
                stats_nifti_path=str(global_nifti_path_top1),
                mask_nifti_path=mask_path,
                map_title=f"Global IG (Abs Mean) - Fold {fold_idx} Top 1%",
                export_filename=f"Render_Global_IG_Fold{fold_idx}_Top1.png",
                slice_config=slice_config
            )
            renderer.plot_3d_activation_map(
                bg_nifti_path=bg_path, 
                stats_nifti_path=str(global_nifti_path_top5),
                mask_nifti_path=mask_path,
                map_title=f"Global IG (Abs Mean) - Fold {fold_idx} Top 5%",
                export_filename=f"Render_Global_IG_Fold{fold_idx}_Top5.png",
                slice_config=slice_config
            )
        except Exception as e:
            log.warning(f"Could not render Global Map for Fold {fold_idx}: {e}")

    log.success("--- XAI Extraction and Rendering Complete ---")

if __name__ == "__main__":
    run_efficientnet_xai()