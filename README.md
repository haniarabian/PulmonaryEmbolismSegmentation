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
    1)	Reads raw CTPA images from the Raw Online Dataset directory.
    2)	Resamples all volumes to an isotropic voxel size of 1×1×1 mm³.
    3)	Clips outlier intensity values to predefined minimum and maximum limits.
    4)	Normalizes all CT intensities to the range [0, 4095].
    5)	Clips CT values above 2100 HU to enhance lung tissue contrast.
    6)	Normalizes intensities to [0, 1].
    7)	Converts embolism masks to binary format (embolus = 1, background = 0).
    8)	Saves all images and masks in NIfTI (.nii) format.
    9)	Outputs are stored in the Pre-processed dataset directory.

2. Second_preprocess.m — Lung localization and spatial cropping
  This script focuses on lung region extraction and spatial normalization.
  Main steps:
    1)	Reads images from the Pre-processed dataset directory.
    2)	Selects the central axial slice of each volume.
    3)	Projects the selected slice onto the XY plane.
    4)	Applies Otsu’s thresholding on the projected slice to separate lung regions.
    5)	Converts the image into a binary mask.
    6)	Uses morphological operations to refine lung masks.
    7)	Computes lung extents along the X and Y axes.
    8)	Crops the full image volume based on lung boundaries.
    9)	Stores original image scale factors.
    10)	Resizes cropped images to 256×256 using scaling.
    11)	Outputs are stored in the Cropped dataset directory.

3. Third_preprocess.m — Slice filtering and dataset cleaning
  This script removes non-informative slices and volumes.
  Main steps:
    1)	Reads images from the Cropped dataset directory.
    2)	Removes volumes without emboli (9 negative cases: IDs 2, 31, 115, 123, 135, 139, 142, 154, 156; 2 single-slice positive cases: IDs 144 and 151).
    3)	Removes non-embolus slices from the remaining volumes.
    4)	Saves embolus-containing slices in the Prepared dataset/nifti.
    5)	Stores indices of embolus and non-embolus slices.
    6)	Final dataset composition (8,268 embolus-positive slices and 25,159 embolus-negative slices).

4. Forth_preprocess.m — Data augmentation
  This script implements classical data augmentation techniques to improve model generalization and reduce overfitting, particularly given the limited size of embolus-positive samples.
  Main steps:
    1)	Reads preprocessed slices from the Prepared dataset directory.
    2)	Applies intensity scaling to adjust contrast within a predefined range while preserving anatomical realism.
    3)	Applies brightness variation by increasing or decreasing pixel intensities.
    4)	Augmentations are applied identically to images and corresponding masks to preserve spatial consistency.


# NIfTI-to-PNG Conversion Scripts (Python / Jupyter):
To facilitate compatibility with 2D deep learning pipelines, NIfTI-formatted images and corresponding segmentation masks were converted into PNG slices using a modular Python-based workflow. Script overview:
  1. transfer_image_dataset.ipynb
    1)	Converts volumetric CTPA images from NIfTI (.nii) format to 2D PNG slices.
    2)	Requires user-defined input and output directory paths.
    3)	Each axial slice is saved as a separate PNG image while preserving slice ordering.
    4)	Intensity values are normalized to ensure consistent visualization across slices.
  2. transfer_annot_dataset.ipynb
    1)	Converts corresponding segmentation masks from NIfTI format to PNG.
    2)	Uses identical slice indexing to ensure pixel-wise alignment with the converted images.
    3)	Binary masks are preserved without interpolation to avoid label corruption.
  3. transfer_to_png.py
    1)	Serves as a wrapper script that orchestrates the full conversion pipeline.
    2)	Imports and executes the core conversion functions defined in the two notebooks/scripts above.
    3)	Ensures synchronized conversion of images and masks.
    4)	Allows batch processing by running a single command after directory paths are specified.


# Upload Output Data:
To facilitate reproducibility without sharing raw clinical data, the outputs generated by Transfertopng.py for a representative anonymized sample are provided as a compressed archive:
  [Google Drive link – png.zip]
This archive contains:
  1) PNG images
  2) Corresponding segmentation masks


# Instructions for Setup and Execution:
Follow the instructions below to set up the pipeline and run the model.

Step 1: Download the Original Network Code
  First, download the original network code from the following GitHub repository:https://github.com/Adversarian/attention-cnn-MS-segmentation
  This code contains the base framework for training the model.

Step 2: Replace Files in the Downloaded Code
  Once you have downloaded the original code, navigate to the respective folders and replace the files with the modified files you downloaded from this repository (attention-cnn-MS-segmentation).

Step 3: Upload to Google Drive
  After replacing the necessary files, upload the entire directory to your Google Drive.
  The code accesses these files via Google Drive, so it is important to ensure the folder structure is correct.

Step 4: Running the Model
  To run the model, execute the following script to run the model:
  Training_process.py
  

