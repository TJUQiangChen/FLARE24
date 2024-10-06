import argparse
import os
import random

import numpy as np
import SimpleITK as sitk
import torch
from configs.config import get_config, get_val_config
from scipy.ndimage import zoom

def is_directory_only_symlinks(directory_path):
    try:
        # use os.scandir foreach read dir
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if not entry.is_symlink():
                    # if found one file without softlink，return False
                    return False
        # if all files are soft link, return True
        return True
    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")
        return False
    except PermissionError:
        print(f"Permission denied: {directory_path}")
        return False

def replace_symlink(input_path, output_path):
    try:
        os.symlink(input_path, output_path)
    except FileExistsError:
        os.remove(output_path)
        os.symlink(input_path, output_path)

def resume_itk_info(image_array, itk_info):
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_image.SetSpacing(itk_info["spacing"])
    sitk_image.SetOrigin(itk_info["origin"])
    sitk_image.SetDirection(itk_info["direction"])

    return sitk_image

def resize_by_spacing(array_image, raw_spacing, new_spacing, type="img"):
    is_resample = False
    raw_size = array_image.shape
    new_size = [
        int(round(raw_size[0] * (raw_spacing[0] / new_spacing[0]))),
        int(round(raw_size[1] * (raw_spacing[1] / new_spacing[1]))),
        int(round(raw_size[2] * (raw_spacing[2] / new_spacing[2]))),
    ]

    zoom_factors = [n / o for n, o in zip(new_size, raw_size)]

    if type == "img":
        array = zoom(array_image, zoom_factors, order=3)
    else:
        array = zoom(array_image, zoom_factors, order=0)

    current_shape = array.shape
    if (
        current_shape[0] != new_size[0]
        or current_shape[1] != new_size[1]
        or current_shape[2] != new_size[2]
    ):
        print("new_shape {}".format(new_size))
        print("current_shape {}".format(current_shape))
        is_resample = True

        exact_resized_array = np.zeros(new_size, dtype=array_image.dtype)

        slices_from = [slice(0, min(c, n)) for c, n in zip(current_shape, new_size)]
        slices_to = [slice(0, min(c, n)) for c, n in zip(new_size, current_shape)]

        exact_resized_array[slices_to[0], slices_to[1], slices_to[2]] = array[
            slices_from[0], slices_from[1], slices_from[2]
        ]

        array = exact_resized_array

    return array, new_size, is_resample


def crop_bbox_restore2raw(
    cropped_image, original_shape, crop_start, crop_end, original_start, original_end
):
    restored_image = np.zeros(original_shape)

    restored_image[
        crop_start[0] : crop_end[0],
        crop_start[1] : crop_end[1],
        crop_start[2] : crop_end[2],
    ] = cropped_image[
        original_start[0] : original_end[0],
        original_start[1] : original_end[1],
        original_start[2] : original_end[2],
    ]

    return restored_image


def crop_bbox_by_stand_spacing(
    image, original_spacing, stand_range=[[1.18, 320], [1.18, 320], [2.6, 76]]
):
    box_length = [
        int(np.ceil(stand_range[0][0] / original_spacing[0] * stand_range[0][1])),
        int(np.ceil(stand_range[1][0] / original_spacing[1] * stand_range[1][1])),
        int(np.ceil(stand_range[2][0] / original_spacing[2] * stand_range[2][1])),
    ]
    image_shape = image.shape

    crop_start = [0, 0, 0]
    crop_end = list(image_shape)
    original_start = [0, 0, 0]
    original_end = list(box_length)

    for i in range(3):
        if image_shape[i] > box_length[i]:
            crop_start[i] = int(np.floor(image_shape[i] / 2)) - int(
                np.floor(box_length[i] / 2)
            )
            crop_end[i] = crop_start[i] + box_length[i]
            original_start[i] = 0
            original_end[i] = box_length[i]
        elif image_shape[i] < box_length[i]:
            crop_start[i] = 0
            crop_end[i] = image_shape[i]
            original_start[i] = int(np.floor(box_length[i] / 2)) - int(
                np.floor(image_shape[i] / 2)
            )
            original_end[i] = original_start[i] + image_shape[i]
        else:
            crop_start[i] = 0
            crop_end[i] = image_shape[i]
            original_start[i] = 0
            original_end[i] = image_shape[i]

    cropped_image = np.zeros(box_length)
    cropped_image[
        original_start[0] : original_end[0],
        original_start[1] : original_end[1],
        original_start[2] : original_end[2],
    ] = image[
        crop_start[0] : crop_end[0],
        crop_start[1] : crop_end[1],
        crop_start[2] : crop_end[2],
    ]

    return cropped_image, crop_start, crop_end, original_start, original_end


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def to_cuda(data, non_blocking=True):
    if isinstance(data, list):
        data = [i.cuda(non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(non_blocking=non_blocking)
    return data


def load_checkpoint(checkpoint_path, checkpoint_name="final_checkpoint.pth"):
    checkpoint_file = checkpoint_name
    checkpoint = torch.load(
        os.path.join(checkpoint_path, checkpoint_file), map_location=torch.device("cpu")
    )
    return checkpoint


def parse_option(phase="train"):
    """
    phase: include 'train', 'val'
    """
    parser = argparse.ArgumentParser("FLARE2024")
    parser.add_argument("--cfg", type=str, metavar="FILE", help="path to config file")

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    args = parser.parse_args()
    if phase == "train":
        parser.add_argument("--tag", help="tag of experiment")
        parser.add_argument("-wm", "--wandb_mode", default="offline")
        parser.add_argument(
            "-bs", "--batch-size", type=int, help="batch size for single GPU"
        )
        parser.add_argument(
            "-wd",
            "--with_distributed",
            help="training without DDP",
            required=False,
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "-ws", "--world_size", type=int, help="process number for DDP"
        )
        args = parser.parse_args()
        config = get_config(args)
    elif phase == "other":
        config = get_val_config(args)

    return args, config
