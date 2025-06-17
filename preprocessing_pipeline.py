import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import train_test_split
import argparse


def load_subjects(data_dir, label, extensions=('nii', 'nii.gz')):
    """
    Load all NIfTI file paths from a directory and assign a label.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory non trovata: {data_dir}")
    files = []
    for ext in extensions:
        files.extend(sorted(glob.glob(os.path.join(data_dir, f'*.{ext}'))))
    if not files:
        print(f"Warning: nessun file NIfTI trovato in {data_dir}")
    return [(fp, label) for fp in files]


def compute_tiv_estimate(subject_files):
    tiv_map = {}
    for fp in subject_files:
        sid = os.path.basename(fp).split('.')[0]
        nii = nib.load(fp)
        data = nii.get_fdata()
        voxel_vol = np.prod(nii.header.get_zooms())
        tiv_map[sid] = np.count_nonzero(data) * voxel_vol
    if not tiv_map:
        raise ValueError("compute_tiv_estimate: nessun soggetto trovato per stima TIV")
    return tiv_map, max(tiv_map.values())


def create_gm_mask(example_nii):
    data = example_nii.get_fdata()
    mask = data > 0
    if not mask.any():
        raise ValueError("Maschera GM vuota: controlla l'immagine di esempio")
    return mask


def vectorize_subject(fp, mask):
    data = nib.load(fp).get_fdata()
    vec = data[mask]
    if vec.size == 0:
        raise ValueError(f"Voxel vector vuoto per {fp}")
    return vec


def normalize_vector(vec, tiv, max_tiv):
    if max_tiv <= 0:
        raise ValueError("max_tiv non positivo")
    return vec * (tiv / max_tiv)


def preprocess(data_dirs, labels, holdout_ratio=0.2, random_state=42):
    all_subjects = []
    for d, lab in zip(data_dirs, labels):
        all_subjects.extend(load_subjects(d, lab))
    if not all_subjects:
        raise FileNotFoundError("Nessun soggetto caricato: controlla data_dirs e contenuto delle cartelle")

    subj_files = [fp for fp, _ in all_subjects]
    subj_ids = [os.path.basename(fp).split('.')[0] for fp in subj_files]

    tiv_map, max_tiv = compute_tiv_estimate(subj_files)

    mask = create_gm_mask(nib.load(subj_files[0]))

    X_list, y, ids = [], [], []
    for fp, lab in all_subjects:
        sid = os.path.basename(fp).split('.')[0]
        vec = vectorize_subject(fp, mask)
        X_list.append(normalize_vector(vec, tiv_map[sid], max_tiv))
        y.append(lab)
        ids.append(sid)
    X = np.vstack(X_list)
    y = np.array(y)

    X_trval, X_test, y_trval, y_test, ids_trval, ids_test = \
        train_test_split(X, y, ids, test_size=holdout_ratio,
                         stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val, ids_train, ids_val = \
        train_test_split(X_trval, y_trval, ids_trval,
                         test_size=0.25, stratify=y_trval,
                         random_state=random_state)

    return {
        'mask': mask,
        'X_train': X_train, 'y_train': y_train, 'ids_train': ids_train,
        'X_val': X_val, 'y_val': y_val, 'ids_val': ids_val,
        'X_test': X_test, 'y_test': y_test, 'ids_test': ids_test
    }


def compute_aspects(X, threshold=0.7, method='average', criterion='distance'):
    corr, _ = spearmanr(X, axis=0)
    n = X.shape[1]
    corr = corr[:n, :n]
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    Z = linkage(dist[np.triu_indices(n,1)], method=method)
    labels = fcluster(Z, t=threshold, criterion=criterion)
    return labels, labels.max()


def preprocess_and_cluster(data_dirs, labels, holdout_ratio=0.2,
                            random_state=42, cluster_threshold=0.7):
    splits = preprocess(data_dirs, labels, holdout_ratio=holdout_ratio,
                        random_state=random_state)
    X_tv = np.vstack([splits['X_train'], splits['X_val']])
    aspect_labels, n_aspects = compute_aspects(X_tv, threshold=cluster_threshold)
    splits.update({'aspect_labels': aspect_labels, 'n_aspects': n_aspects})
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline preprocessing e generazione aspects per dati NIfTI.")
    parser.add_argument("--ad_dir", required=True,
                        help="Cartella contenente soggetti AD (.nii/.nii.gz)")
    parser.add_argument("--ctrl_dir", required=True,
                        help="Cartella contenente soggetti CTRL (.nii/.nii.gz)")
    parser.add_argument("--output", default="aspect_map.nii",
                        help="Percorso del file NIfTI di output per la mappa degli aspetti")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Soglia di distanza per clustering gerarchico")
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="Frazione di hold-out per test set (default 0.2)")
    args = parser.parse_args()

    data_dirs = [args.ad_dir, args.ctrl_dir]
    labels = [1, 0]
    splits = preprocess_and_cluster(data_dirs, labels,
                                    holdout_ratio=args.holdout,
                                    cluster_threshold=args.threshold)

    mask, aspects = splits['mask'], splits['aspect_labels']
    full_map = np.zeros(mask.shape, dtype=int)
    full_map[mask] = aspects

    # Recupera affine dal primo file AD
    aff_file = glob.glob(os.path.join(args.ad_dir, '*.nii*'))
    if not aff_file:
        raise FileNotFoundError(f"Nessun file NIfTI trovato in {args.ad_dir} per affine")
    affine = nib.load(aff_file[0]).affine
    nib.save(nib.Nifti1Image(full_map.astype(np.int16), affine), args.output)
    print(f"Mappa degli aspetti salvata in {args.output}")

if __name__ == '__main__':
    main()
