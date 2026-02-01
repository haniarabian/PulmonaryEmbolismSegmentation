
import numpy as np
from PIL import Image
import nibabel as nib
import os


def convert_to_rgb(array):
    new_img = np.zeros((array.shape[0], array.shape[1], 3))
    new_img[:, :, 0] = array / 255
    new_img[:, :, 1] = array / 255
    new_img[:, :, 2] = array / 255
    return np.uint8(new_img * 255.0)


def make_annot(path_mask, save_dir, sub_number):
    if not os.path.exists(os.path.join(save_dir, f"P{sub_number}_T1")):
        os.makedirs(os.path.join(save_dir, f"P{sub_number}_T1"))

    vol_mask = nib.load(path_mask)
    mask = vol_mask.get_fdata()

    for slice_num in range(mask.shape[2]):
        slice = mask[:, :, slice_num]
        slice *= 255.0 / slice.max()
        slice = np.uint8(slice)
        slice = convert_to_rgb(slice)

        im_save_path = os.path.join(
            save_dir,
            f"P{sub_number}_T1",
            "training"
            + str(sub_number // 10)
            + str(sub_number % 10)
            + "_01_mask1"
            + str((slice_num + 1) // 100)
            + str((slice_num + 1) // 10)
            + str((slice_num + 1) % 10)
            + ".png"
        )

        im = Image.fromarray(slice)
        im.save(im_save_path)


for sub in range(1, 13, 1):
    if sub!=31 and sub!=115 and sub!=123:
        make_annot(os.path.join(r"F:\PhD\Thesis\Datasets\Prepared dataset\Salehi\nifti\GE\\ref", f"PE_{sub}.nii"), r"F:\PhD\Thesis\Datasets\Prepared dataset\Salehi\png\GE\\trainannot", sub)
        print(f"sub_{sub}")
