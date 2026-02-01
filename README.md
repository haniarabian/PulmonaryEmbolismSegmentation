# Pulmonary Embolism Segmentation – Preprocessing and Data Conversion Pipeline:

This repository provides the preprocessing, data preparation, and conversion pipeline used in our study on automatic pulmonary embolism (PE) detection and segmentation from CTPA images. The implementation follows a hybrid MATLAB–Python workflow, reflecting the exact pipeline used in the manuscript. Data preprocessing, augmentation, and selected evaluation procedures were implemented in MATLAB, whereas model training, inference, and performance evaluation were conducted in Python. The core backbone network was adapted from an existing open-source implementation, which is properly cited in the manuscript. For these components, we provide explicit references and links to the original repositories rather than re-uploading unchanged source code, in accordance with best practices for open-source reuse. 


# Pipeline Overview:
The preprocessing pipeline consists of four main stages:
  1. CT volume standardization and normalization (MATLAB)
  2. Lung region extraction and spatial cropping (MATLAB)
  3. Slice-level dataset preparation and class balancing (MATLAB)
  4. Data augmentation
  5. Conversion from NIfTI to PNG format for deep learning training (Python)


# MATLAB Preprocessing Pipeline:
This repository contains the MATLAB-based preprocessing pipeline used in this study for pulmonary embolism (PE) analysis in CTPA images. The pipeline was applied to both public datasets (CAD-PE and FUMPE) and a private dataset, and consists of the following steps:

1. First_preprocess.m — Intensity normalization and resampling
  This script performs initial preprocessing on the raw CTPA volumes.
  Main steps:
    •	Reads raw CTPA images from the Raw Online Dataset directory.
    •	Resamples all volumes to an isotropic voxel size of 1×1×1 mm³.
    •	Clips outlier intensity values to predefined minimum and maximum limits.
    •	Normalizes all CT intensities to the range [0, 4095].
    •	Clips CT values above 2100 HU to enhance lung tissue contrast.
    •	Normalizes intensities to [0, 1].
    •	Converts embolism masks to binary format (embolus = 1, background = 0).
    •	Saves all images and masks in NIfTI (.nii) format.
    •	Outputs are stored in the Pre-processed dataset directory.

2. Second_preprocess.m — Lung localization and spatial cropping
  This script focuses on lung region extraction and spatial normalization.
  Main steps:
    •	Reads images from the Pre-processed dataset directory.
    •	Selects the central axial slice of each volume.
    •	Projects the selected slice onto the XY plane.
    •	Applies Otsu’s thresholding on the projected slice to separate lung regions.
    •	Converts the image into a binary mask.
    •	Uses morphological operations to refine lung masks.
    •	Computes lung extents along the X and Y axes.
    •	Crops the full image volume based on lung boundaries.
    •	Stores original image scale factors.
    •	Resizes cropped images to 256×256 using scaling.
    •	Outputs are stored in the Cropped dataset directory.

3. Third_preprocess.m — Slice filtering and dataset cleaning
  This script removes non-informative slices and volumes.
  Main steps:
    •	Reads images from the Cropped dataset directory.
    •	Removes volumes without emboli (9 negative cases: IDs 2, 31, 115, 123, 135, 139, 142, 154, 156; 2 single-slice positive cases: IDs 144 and 151).
    •	Removes non-embolus slices from the remaining volumes.
    •	Saves embolus-containing slices in the Prepared dataset/nifti.
    •	Stores indices of embolus and non-embolus slices.
    •	Final dataset composition (8,268 embolus-positive slices and 25,159 embolus-negative slices).

4. Forth_preprocess.m — Data augmentation
This script implements classical data augmentation techniques to improve model generalization and reduce overfitting, particularly given the limited size of embolus-positive samples.
Main steps:
  •	Reads preprocessed slices from the Prepared dataset directory.
  •	Applies intensity scaling to adjust contrast within a predefined range while preserving anatomical realism.
  •	Applies brightness variation by increasing or decreasing pixel intensities.
  •	Augmentations are applied identically to images and corresponding masks to preserve spatial consistency.


# NIfTI-to-PNG Conversion Scripts (Python / Jupyter):
To facilitate compatibility with 2D deep learning pipelines, NIfTI-formatted images and corresponding segmentation masks were converted into PNG slices using a modular Python-based workflow. Script overview:
  1. transfer_image_dataset.ipynb
    •	Converts volumetric CTPA images from NIfTI (.nii) format to 2D PNG slices.
    •	Requires user-defined input and output directory paths.
    •	Each axial slice is saved as a separate PNG image while preserving slice ordering.
    •	Intensity values are normalized to ensure consistent visualization across slices.
  2. transfer_annot_dataset.ipynb
    •	Converts corresponding segmentation masks from NIfTI format to PNG.
    •	Uses identical slice indexing to ensure pixel-wise alignment with the converted images.
    •	Binary masks are preserved without interpolation to avoid label corruption.
  3. transfer_to_png.py
    •	Serves as a wrapper script that orchestrates the full conversion pipeline.
    •	Imports and executes the core conversion functions defined in the two notebooks/scripts above.
    •	Ensures synchronized conversion of images and masks.
    •	Allows batch processing by running a single command after directory paths are specified.


# Upload Output Data:
To facilitate reproducibility without sharing raw clinical data, the outputs generated by Transfertopng.py for a representative anonymized sample are provided as a compressed archive:
  [Google Drive link – png.zip]
This archive contains:
  •	PNG images
  •	Corresponding segmentation masks



