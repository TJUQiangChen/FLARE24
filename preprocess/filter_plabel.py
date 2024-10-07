import json
import os
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from tqdm import tqdm


def static_gt_each_labels_num(input_dir, output_csv):
    voxel_stats = []

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, file_name)

            image = nib.load(file_path)
            target_spacing = np.array([1, 1, 1])
            resized_image = resize_image(image, target_spacing)
            # Calculate voxel statistics, excluding voxel values of 0.
            stats = calculate_voxel_statistics(resized_image)

            # Record the statistical results for each file.
            stats['file_name'] = file_name
            voxel_stats.append(stats)

    df = pd.DataFrame(voxel_stats).fillna(0)

    # Calculate the mean and standard deviation for each category.
    summary = df.describe().T[['mean', 'std']]

    # Set the threshold for outliers (this can be adjusted according to specific requirements).
    summary['threshold'] =  3 * summary['std']

    summary.to_csv(output_csv)

    return summary

def label_onehot(inputs, num_segments):
    inputs[inputs == 255] = 0
    return torch.nn.functional.one_hot(inputs, num_segments).permute((0, 4, 1, 2, 3))

def resize_image(image, target_spacing):
    spacing = np.array(image.header.get_zooms())
    resize_factor = spacing / target_spacing
    image_data = image.get_fdata()

    resized_image = zoom(image_data, resize_factor, order=0)
    return resized_image

def calculate_voxel_statistics(image_data):
    '''Calculate voxel statistics, excluding voxel values of 0.'''
    unique, counts = np.unique(image_data[image_data > 0], return_counts=True)
    return dict(zip(unique, counts))


def check_voxel_counts(stats, reference_summary):
    anomalies = {}

    for label, count in stats.items():
        if label in reference_summary.index:
            mean = reference_summary.loc[label, "mean"]
            # threshold = reference_summary.loc[label, "std"]

            if (count > (mean * 1.8)) or (count < (mean * 0.2)):
                anomalies[label] = count

    return anomalies


# Process the pseudo-label files and save the results.
def process_fake_labels(input_dir, reference_summary, output_csv):
    results = []

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, file_name)

            image = nib.load(file_path)

            target_spacing = np.array([1, 1, 1])
            resized_image = resize_image(image, target_spacing)


            stats = calculate_voxel_statistics(resized_image)


            anomalies = check_voxel_counts(stats, reference_summary)


            if anomalies != {}:
                result = {"file_name": file_name, "anomalies": anomalies}
                results.append(result)


    anomaly_df = pd.DataFrame(results)

    anomaly_df.to_csv(output_csv, index=False)


def filter_fake_labels_by_entropy(softmax_file_dir, num_classes=13):
    # Calculate the entropy for each category within a sample.
    sample_bank = {}
    softmax_files = [
        f for f in os.listdir(softmax_file_dir) if f.endswith(".npy") and "softmax" in f
    ]
    for softmax_file in tqdm(softmax_files):
        # Calculate the entropy for each category in a sample.
        prob = (
            torch.tensor(np.load(os.path.join(softmax_file_dir, softmax_file)))
            .unsqueeze(0)
            .cuda()
        )
        _, pesudo_label = torch.max(prob, dim=1)
        entropy = -prob * torch.log(prob + 1e-10)
        one_hot_label = label_onehot(pesudo_label.long(), num_classes)

        class_entropy_bank = {}
        for i in range(num_classes):
            if one_hot_label[:, i : i + 1, ...].sum() == 0:
                continue
            class_i_entropy = (
                entropy[:, i : i + 1, ...] * one_hot_label[:, i : i + 1, ...]
            ).sum() / one_hot_label[:, i : i + 1, ...].sum()
            class_entropy_bank.setdefault(i, class_i_entropy)

        sample_bank.setdefault(
            os.path.basename(softmax_file).split("_")[0], class_entropy_bank
        )
        sample_bank[os.path.basename(softmax_file).split("_")[0]]["not_filter"] = 1
        re_class_num = [0 for i in range(num_classes)]
        thresholds_c = 0.3
        for c in range(num_classes):
            class_c_all = {
                key: value[c] for key, value in sample_bank.items() if c in value.keys()
            }  # Class C Entropy: {name: entropy value}
            for key, value in class_c_all.items():
                if value < thresholds_c:
                    sample_bank[key][c] = 1
                    re_class_num[c] += 1
                else:
                    sample_bank[key][c] = 0
                    sample_bank[os.path.basename(softmax_file).split("_")[0]][
                        "not_filter"
                    ] = 0

    sample_bank = sorted(
        sample_bank.items(),
        key=lambda kv: sum(list(kv[1].values())) / len(list(kv[1].values())),
        reverse=True,
    )

    return sample_bank
