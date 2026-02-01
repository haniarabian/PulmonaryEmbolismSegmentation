# PulmonaryEmbolismSegmentation

The implementation of the proposed framework was developed using a hybrid MATLAB–Python workflow. Data preprocessing, augmentation, and selected evaluation procedures were implemented in MATLAB, whereas model training, inference, and performance evaluation were conducted in Python. The core backbone network was adapted from an existing open-source implementation, which is properly cited in the manuscript. For these components, we provide explicit references and links to the original repositories rather than re-uploading unchanged source code, in accordance with best practices for open-source reuse. 


MATLAB Preprocessing Pipeline:
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
    •	Removes volumes without emboli
      (9 negative cases: IDs 2, 31, 115, 123, 135, 139, 142, 154, 156; 2 single-slice positive cases: IDs 144 and 151).
    •	Removes non-embolus slices from the remaining volumes.
    •	Saves embolus-containing slices in the Prepared dataset/nifti.
    •	Stores indices of embolus and non-embolus slices.
    •	Final dataset composition (8,268 embolus-positive slices and 25,159 embolus-negative slices).


Data Format to .png Conversion Pipeline:


