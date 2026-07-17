"""
Module: tpm_mask_generator.py

TPM Mask Generator Module.

PURPOSE:
    This module acts as the native Python replacement for the MATLAB-based 
    BrainMask generation pipeline. It guarantees spatial alignment between a 
    reference cohort and an external SPM Tissue Probability Map (TPM), 
    extracts the Gray Matter volume, and binarizes it.
"""
import pathlib
import pandas as pd
import numpy as np
import nibabel as nib
from typing import Tuple, Any

from Python.utils.py_logger import CustomLogger

# Constants
DEFAULT_THRESHOLD = 0.01
AFFINE_TOLERANCE = 1e-4
GRAY_MATTER_INDEX = 0

class TpmMaskGenerator:
    """
    Handles the spatial validation and computation of the TPM boolean mask.
    """

    def __init__(self, logger: CustomLogger):
        """
        Initializes the mask generator.
        
        Args:
            logger (CustomLogger): Centralized logging instance.
        """
        self.logger = logger

    def generate_mask(self, registry_csv_path: str, tpm_nifti_path: str, output_mask_path: str, threshold: float = DEFAULT_THRESHOLD) -> None:
        """
        Main orchestration method to compute and save the TPM Mask.
        
        PURPOSE:
            Loads the cohort space, verifies alignment with the SPM TPM, binarizes 
            the Gray Matter layer, and exports the NIfTI mask to disk.
            
        Args:
            registry_csv_path (str): Absolute path to the cohort CSV.
            tpm_nifti_path (str): Absolute path to the external SPM TPM.nii file.
            output_mask_path (str): Absolute path where the binary mask will be exported.
            threshold (float): Probability threshold for Grey Matter binarization.
        """
        self.logger.info("--- Starting TPM Preprocessing ---")
        
        # Extract reference data
        ref_shape, ref_affine, ref_header = self._get_reference_metadata(registry_csv_path)
        
        # Process TPM Volume
        tpm_data, tpm_affine = self._process_tpm_volume(tpm_nifti_path)
        
        # Validate alignment
        self._validate_spatial_alignment(ref_shape, ref_affine, tpm_data.shape, tpm_affine)
        
        # Binarization
        self.logger.info(f"Computing Mask (Grey Matter threshold = {threshold:.2f})...")
        binary_mask = (tpm_data >= threshold).astype(np.float32)
        
        # Exporting
        out_path = pathlib.Path(output_mask_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Exporting computed mask to disk...")
        try:
            nifti_img = nib.Nifti1Image(binary_mask, affine=ref_affine, header=ref_header)
            nib.save(nifti_img, str(out_path))
            self.logger.success(f"Common Preprocessing complete. Mask safely saved at: {out_path}")
        except Exception as e:
            self.logger.error(f"Failed to export NIfTI mask: {e}")
            raise

    def _get_reference_metadata(self, csv_path: str) -> Tuple[Tuple[int, ...], np.ndarray, Any]:
        """
        Extracts physical dimensions and the affine matrix from the cohort's first volume.
        
        Args:
            csv_path (str): Path to the cohort registry.
            
        Returns:
            Tuple[Tuple[int, ...], np.ndarray, Any]: 
                Contains the image shape, the 4x4 affine matrix, and the NIfTI header.
        """
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError("Registry CSV is empty.")
                
            first_volume_path = df.iloc[0]['file_path']
            
            img = nib.load(first_volume_path)
            self.logger.success("Reference spatial metadata successfully extracted.")
            return img.shape, img.affine, img.header
            
        except FileNotFoundError:
            self.logger.error(f"Registry CSV not found at: {csv_path}")
            raise
        except KeyError:
            self.logger.error(f"Invalid CSV structure. Missing 'file_path' column in {csv_path}")
            raise
        except nib.filebasedimages.ImageFileError:
            self.logger.error(f"Failed to read the reference NIfTI volume at: {first_volume_path}")
            raise

    def _process_tpm_volume(self, tpm_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the SPM Tissue Probability Map and extracts the Gray Matter volume.
        
        Args:
            tpm_path (str): Path to the SPM TPM.nii file.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: The extracted 3D Gray Matter tensor and its affine matrix.
        """
        self.logger.info(f"Attempting to load SPM Tissue Probability Map (TPM) from: {tpm_path}")
        
        try:
            tpm_img = nib.load(tpm_path)
            tpm_data = tpm_img.get_fdata(dtype=np.float32)
            
            # Handle 4D volumes (SPM TPM contains multiple tissue classes)
            if tpm_data.ndim > 3:
                self.logger.info(f"Detected a {tpm_data.ndim}D NIfTI file. Forcing extraction of Volume {GRAY_MATTER_INDEX + 1} (Gray Matter).")
                tpm_data = tpm_data[..., GRAY_MATTER_INDEX]
                
            # Clean up NaNs
            tpm_data = np.nan_to_num(tpm_data, copy=False, nan=0.0)
            
            return tpm_data, tpm_img.affine
            
        except FileNotFoundError:
            self.logger.error(f"TPM NIfTI file not found at: {tpm_path}")
            raise
        except nib.filebasedimages.ImageFileError:
            self.logger.error(f"Corrupted or invalid TPM NIfTI file at: {tpm_path}")
            raise

    def _validate_spatial_alignment(self, ref_shape: Tuple[int, ...], ref_affine: np.ndarray, 
                                    tpm_shape: Tuple[int, ...], tpm_affine: np.ndarray) -> None:
        """
        Guarantees the external map mathematically matches the cohort's spatial space.
        
        PURPOSE:
            Fails fast if the Affine transform deviates beyond a micro-tolerance 
            (AFFINE_TOLERANCE = 1e-4), blocking the creation of fundamentally 
            misaligned masks.
            
        Args:
            ref_shape (Tuple): Dimensionality of the reference cohort.
            ref_affine (np.ndarray): 4x4 affine matrix of the reference cohort.
            tpm_shape (Tuple): Dimensionality of the TPM.
            tpm_affine (np.ndarray): 4x4 affine matrix of the TPM..
        """
        # Dimension Check
        if ref_shape != tpm_shape:
            self.logger.error(f"Dimensions mismatch: Loaded {tpm_shape} vs Cohort {ref_shape}")
            raise ValueError("Spatial dimensions are incompatible.")
            
        # Affine Matrix Check
        max_deviation = np.max(np.abs(ref_affine - tpm_affine))
        if not np.allclose(ref_affine, tpm_affine, atol=AFFINE_TOLERANCE):
            self.logger.error(f"Affine matrices differ significantly. Max deviation: {max_deviation:.6f}")
            raise ValueError("The external mask and the cohort volumes are not aligned.")