
import numpy as np
from PIL import Image
import nibabel as nib
import os


def make_image_dataset(vol_nii_path, save_dir, sub_number):
    if not os.path.exists(os.path.join(save_dir, f"P{sub_number}_T1")):
        os.makedirs(os.path.join(save_dir, f"P{sub_number}_T1"))

    vol = nib.load(vol_nii_path)
    img = vol.get_fdata()

    for slice_num in range(img.shape[2]):
        slice = img[:, :, slice_num]
        slice *= 255.0 / slice.max()
        slice = np.uint8(slice)
        im_save_path = os.path.join(
            save_dir,
            f"P{sub_number}_T1",
            "training"
            + str(sub_number // 10)
            + str(sub_number % 10)
            + "_01_flair_pp"
            + str((slice_num + 1) // 100)
            + str((slice_num + 1) // 10)
            + str((slice_num + 1) % 10)
            + ".png")

        im = Image.fromarray(slice)
        im.save(im_save_path)


for sub in range(1, 23, 1):
    if sub!=31 and sub!=115 and sub!=123:
        make_image_dataset(os.path.join(r"F:\PhD\Thesis\Datasets\Prepared dataset\Salehi\nifti\GE\\image", f"PE_{sub}.nii"), r"F:\PhD\Thesis\Datasets\Prepared dataset\Salehi\png\GE\\train", sub)
        print(f"sub_{sub}")
