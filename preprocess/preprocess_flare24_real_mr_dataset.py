import os
import re
from tqdm import tqdm
import json
from datetime import datetime
import SimpleITK as sitk
import numpy as np
import pandas as pd


def regis_label_postprocess(image_path, label_path):
    # Read the image and pseudo-label.
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # Convert to a numpy array.
    image_np = sitk.GetArrayFromImage(image)
    label_np = sitk.GetArrayFromImage(label)

    # Generate a mask for the image, assuming the background area has a pixel value of 0.
    mask = image_np > 0

    # Convert back to a SimpleITK image and save.
    label_processed = label_np * mask

    # Convert back to SimpleITK image and save.
    label_processed_sitk = sitk.GetImageFromArray(label_processed)
    label_processed_sitk.CopyInformation(label)
    return label_processed_sitk


def replace_symlink(input_path, output_path):
    try:
        os.symlink(input_path, output_path)
    except FileExistsError:
        os.remove(output_path)
        os.symlink(input_path, output_path)


def load_data(data_path):
    data_itk = sitk.ReadImage(data_path)
    data_npy = sitk.GetArrayFromImage(data_itk)[None].astype(np.float32)
    data_spacing = np.array(data_itk.GetSpacing())[[2, 1, 0]]
    direction = data_itk.GetDirection()
    direction = np.array((direction[8], direction[4], direction[0]))
    return data_npy[0], data_spacing, direction


def extract_image_id(filename):
    """
    Extract the image_id from the given filename.
    Filenames may take the following forms:
    1. MR129136_1_C+A_0000.nii.gz ，image_id是129136
    2. MR-391135_1_C+A_0000.nii.gz, image_id是3911135
    3. amos_7427_0000.nii.gz，image_id是7427
    """
    # 使用正则表达式匹配不同的格式并提取image_id
    patterns = [
        r"MR(\d+).*\.nii\.g",  # 匹配MR129136_1_C+A_0000.nii.gz
        r"MR-(\d+).*\.nii\.g",  # 匹配MR-391135_1_C+A_0000.nii.gz
        r"amos_(\d+)_\d+.nii.gz",  # 匹配amos_7427_0000.nii.gz
        r"Case_(\d+)_\d+.nii.gz",  # 匹配Case_7427_0000.nii.gz
        r"LLD_(\d+)_\d+.nii.gz",
        r"Case_(\d+).nii.gz",
        r"amos_(\d+).nii.gz",
        r"LLD_(\d+).nii.gz",
    ]

    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return match.group(1)

    # 如果没有匹配的格式，返回None或引发错误
    return None


def preprocess_amos_datset_plabel(
    origin_image_path,
    temp_output_label_path,
    output_label_path,
    output_image_path,
    filter_path=None,
):
    if filter_path is not None:
        df = pd.read_csv(filter_path)
        filter_file = [f.split(".")[0].split("_")[1] for f in df["file_name"].tolist()]
    else:
        filter_file = None

    images_files = os.listdir(origin_image_path)
    for images_file in tqdm(images_files):
        file_idx = extract_image_id(images_file)
        if filter_file is not None:
            if file_idx in filter_file:
                continue
        train_image_path = os.path.join(origin_image_path, images_file)
        output_image_file_name = os.path.basename(train_image_path)
        lable_path = os.path.join(
            temp_output_label_path, "amos_{:04d}.nii.gz".format(int(file_idx))
        )
        # img copy
        os.makedirs(output_image_path, exist_ok=True)
        replace_symlink(
            train_image_path, os.path.join(output_image_path, output_image_file_name)
        )
        # label copy
        replace_symlink(
            lable_path,
            os.path.join(
                output_label_path,
                "_".join(output_image_file_name.split("_")[:-1]) + ".nii.gz",
            ),
        )
        # If the softmax results exist, save them as well.
        if os.path.exists(lable_path.replace(".nii.gz", "_softmax.npy")):
            replace_symlink(
                lable_path.replace(".nii.gz", "_softmax.npy"),
                os.path.join(
                    output_label_path,
                    "_".join(output_image_file_name.split("_")[:-1]) + "_softmax.npy",
                ),
            )


def preprocess_lld_datset_plabel(
    origin_image_path,
    temp_output_label_path,
    output_label_path,
    output_image_path,
    pair_json_path="Dataset667_FLARE24v1-20240809.json",
    bad_image_file_path="test",
    filter_path=None,
    use_regis_dataset=False,
):
    bad_image_files = []
    with open(pair_json_path, "r") as f:
        pair_Dict = json.loads(f.read())

    if filter_path is not None:
        df = pd.read_csv(filter_path)
        filter_file = [
            f.split(".")[0].replace("LLD_", "") for f in df["file_name"].tolist()
        ]
    else:
        filter_file = None

    # Read the original LLD data.
    for file_idx in tqdm(range(len(pair_Dict["input_image_file_path"]))):
        train_image_path = pair_Dict["input_image_file_path"][file_idx]
        output_image_file_name = os.path.basename(
            pair_Dict["output_image_file_path"][file_idx]
        )
        image_id = extract_image_id(output_image_file_name)
        if filter_file is not None:
            if image_id in filter_file:
                print("{} has being filt!".format(image_id))
                continue
        if "LLD_" in os.listdir(temp_output_label_path)[0]:
            lable_path = os.path.join(
                temp_output_label_path, "LLD_{}.nii.gz".format(int(image_id))
            )
        else:
            lable_path = os.path.join(
                temp_output_label_path, "{}.nii.gz".format(int(image_id))
            )
        current_image_prefix = "_".join(
            [os.path.basename(train_image_path).split("_")[0], ""]
        )
        all_current_image_files = [
            f for f in os.listdir(origin_image_path) if current_image_prefix in f
        ]
        target_image, _, _ = load_data(train_image_path)
        for cureent_image_file in all_current_image_files:
            if use_regis_dataset:
                # Post-process is required for the registered labels.
                try:
                    label_processed_sitk = regis_label_postprocess(
                        os.path.join(origin_image_path, cureent_image_file), lable_path
                    )
                except ValueError as e:
                    print(
                        "{}:size bug! please check the label file!".format(lable_path)
                    )
                    continue
                except RuntimeError as e:
                    print(
                        "{}:file error! please check the label file!".format(
                            os.path.join(origin_image_path, cureent_image_file)
                        )
                    )
                    continue
                sitk.WriteImage(
                    label_processed_sitk,
                    os.path.join(
                        output_label_path,
                        "_".join(cureent_image_file.split("_")[:-1]) + ".nii.gz",
                    ),
                )
                os.makedirs(output_image_path, exist_ok=True)
                replace_symlink(
                    os.path.join(origin_image_path, cureent_image_file),
                    os.path.join(
                        output_image_path,
                        "_".join(cureent_image_file.split("_")[:-1]) + "_0000.nii.gz",
                    ),
                )
                # If the results of the softmax exist, they should also be saved.
                if os.path.exists(
                    lable_path.replace(".nii.gz", "_softmax.npy"),
                ):
                    replace_symlink(
                        lable_path.replace(".nii.gz", "_softmax.npy"),
                        os.path.join(
                            output_label_path,
                            "_".join(cureent_image_file.split("_")[:-1])
                            + "_softmax.npy",
                        ),
                    )

            else:
                origni_img, _, _ = load_data(
                    os.path.join(origin_image_path, cureent_image_file)
                )

                if not (
                    np.array(target_image.shape) == np.array(origni_img.shape)
                ).all():
                    print("{} bad".format(cureent_image_file))
                    bad_image_files.append(cureent_image_file)
                else:
                    os.makedirs(output_image_path, exist_ok=True)
                    replace_symlink(
                        os.path.join(origin_image_path, cureent_image_file),
                        os.path.join(
                            output_image_path,
                            "_".join(cureent_image_file.split("_")[:-1])
                            + "_0000.nii.gz",
                        ),
                    )
                    replace_symlink(
                        lable_path,
                        os.path.join(
                            output_label_path,
                            "_".join(cureent_image_file.split("_")[:-1]) + ".nii.gz",
                        ),
                    )
    with open(bad_image_file_path, "w") as f:
        f.write(json.dumps(bad_image_files, indent=4))


def process_lld_data_img(path_info, output_prefix="default"):
    """
    Extract data that is only T2.
    """
    num_of_dir = len(path_info["train_image_path"])
    image_file_outputID_pair = {
        "input_image_file_path": [],
        "output_image_file_path": [],
    }

    for i in range(num_of_dir):
        train_image_path = path_info["train_image_path"][i]
        output_image_path = path_info["output_image_path"]

        os.makedirs(output_image_path, exist_ok=True)

        # Retrieve the filenames from the `train_image` directory.
        image_files = sorted(
            [f for f in os.listdir(train_image_path) if f.endswith(".nii.gz")]
        )

        for file_id in tqdm(range(len(image_files))):
            image_file = image_files[file_id]
            image_id = extract_image_id(image_file)
            if image_id is None:
                print(f"{image_file} extract image_id failed!")
                continue
            # NOTE: only process  "C+Delay" modality dataloader，
            if "C+Delay" not in image_file:
                continue

            image_path = os.path.join(train_image_path, image_file)

            if os.path.exists(image_path):
                output_image = os.path.join(
                    output_image_path,
                    f"{output_prefix}_{image_id}_0000.nii.gz",
                )

                replace_symlink(image_path, output_image)
                image_file_outputID_pair["input_image_file_path"].append(image_path)
                image_file_outputID_pair["output_image_file_path"].append(output_image)
            else:
                print(f"{image_path} file not found!")

    return image_file_outputID_pair
