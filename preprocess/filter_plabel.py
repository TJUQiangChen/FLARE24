import json
import os
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from tqdm import tqdm


# 处理文件并保存结果
def static_gt_each_labels_num(input_dir, output_csv):
    voxel_stats = []

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, file_name)

            # 读取nii.gz文件
            image = nib.load(file_path)

            # 将图像resize到目标spacing [1,1,1]
            target_spacing = np.array([1, 1, 1])
            resized_image = resize_image(image, target_spacing)

            # 计算体素统计，排除体素值为0
            stats = calculate_voxel_statistics(resized_image)

            # 记录每个文件的统计结果
            stats['file_name'] = file_name
            voxel_stats.append(stats)

    # 转换为DataFrame
    df = pd.DataFrame(voxel_stats).fillna(0)  # 填充NaN值为0

    # 计算每个类别的平均值和标准差
    summary = df.describe().T[['mean', 'std']]

    # 设置异常值阈值 (可以根据具体需求进行调整)
    summary['threshold'] =  3 * summary['std']
    # 保存为CSV文件
    summary.to_csv(output_csv)

    return summary

def label_onehot(inputs, num_segments):
    inputs[inputs == 255] = 0
    return torch.nn.functional.one_hot(inputs, num_segments).permute((0, 4, 1, 2, 3))

# 使用scipy的zoom方法进行resize
def resize_image(image, target_spacing):
    spacing = np.array(image.header.get_zooms())
    resize_factor = spacing / target_spacing
    image_data = image.get_fdata()

    # 进行resize
    resized_image = zoom(image_data, resize_factor, order=0)  # 使用最近邻插值
    return resized_image


# 计算类别体素数量，排除0
def calculate_voxel_statistics(image_data):
    unique, counts = np.unique(image_data[image_data > 0], return_counts=True)
    return dict(zip(unique, counts))


# 检查体素数量是否在标准阈值范围内
def check_voxel_counts(stats, reference_summary):
    anomalies = {}

    for label, count in stats.items():
        if label in reference_summary.index:
            mean = reference_summary.loc[label, "mean"]
            threshold = reference_summary.loc[label, "std"]
            # v1版本
            # if (count > (mean * 3)) or (count < (mean * 0.05)):
            # V2版本，更加严格
            if (count > (mean * 1.8)) or (count < (mean * 0.2)):
                anomalies[label] = count

    return anomalies


# 处理伪标签文件并保存结果
def process_fake_labels(input_dir, reference_summary, output_csv):
    results = []

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, file_name)

            # 读取nii.gz文件
            image = nib.load(file_path)

            # 将图像resize到目标spacing [1,1,1]
            target_spacing = np.array([1, 1, 1])
            resized_image = resize_image(image, target_spacing)

            # 计算体素统计，排除体素值为0
            stats = calculate_voxel_statistics(resized_image)

            # 检查体素数量是否超出标准阈值范围
            anomalies = check_voxel_counts(stats, reference_summary)

            # 如果有异常，则记录
            if anomalies != {}:
                result = {"file_name": file_name, "anomalies": anomalies}
                results.append(result)

    # 将结果转换为DataFrame
    anomaly_df = pd.DataFrame(results)

    # 保存为CSV文件
    anomaly_df.to_csv(output_csv, index=False)


def filter_fake_labels_by_entropy(softmax_file_dir, num_classes=13):
    # 计算一个样本中每个类别的熵
    sample_bank = {}
    softmax_files = [
        f for f in os.listdir(softmax_file_dir) if f.endswith(".npy") and "softmax" in f
    ]
    for softmax_file in tqdm(softmax_files):
        # 计算一个样本中每个类别的熵
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
            }  # 第c类的{name: 熵值}
            for key, value in class_c_all.items():
                if value < thresholds_c:
                    sample_bank[key][c] = 1
                    re_class_num[c] += 1
                else:
                    sample_bank[key][c] = 0
                    sample_bank[os.path.basename(softmax_file).split("_")[0]][
                        "not_filter"
                    ] = 0
            # if sample_bank[os.path.basename(softmax_file).split('_')[0]]['is_filter']:
            #     break
    sample_bank = sorted(
        sample_bank.items(),
        key=lambda kv: sum(list(kv[1].values())) / len(list(kv[1].values())),
        reverse=True,
    )  # 降序排列

    return sample_bank


if __name__ == "__main__":
    # 使用示例
    # V100-lld
    # inptu_img_dir  = '/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset/MR/Training/LLD-MMRI-3984-T2_tmp'
    # input_dir = '/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset/MR/Training/PLabel/LLD-MMRI-3984-T2_tmp_v3'  # 输入伪标签nii.gz文件的目录
    # reference_summary_csv = '/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset/static_info/label_class_static.csv'  # 之前计算的标准数量和阈值的CSV文件
    # output_csv = "/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset/static_info/filter_plabel_lld_v3.csv"
    # check_dir = '/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset/MR/Training/temp'
    # V100-amos
    inptu_img_dir = "/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset/MR/Training/AMOS_MR_good_spacing-833"
    input_dir = "/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset/MR/Training/PLabel/AMOS_MR_good_spacing-833_tmp_v3"  # 输入伪标签nii.gz文件的目录
    reference_summary_csv = "/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset/static_info/label_class_static.csv"  # 之前计算的标准数量和阈值的CSV文件
    output_csv = "/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset/static_info/filter_plabel_amos_v3.csv"
    check_dir = (
        "/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset/MR/Training/temp_amos"
    )
    # 3090 - amos
    # inptu_img_dir = ''
    # input_dir = '/mnt16t_ext/lijiaxi/MR/AMOS_MR_good_spacing-833_plabel'
    # reference_summary_csv = '/mnt16t_ext/lijiaxi/MR/label_class_static.csv'  # 之前计算的标准数量和阈值的CSV文件
    # output_csv = '/mnt16t_ext/lijiaxi/MR/filter_plabel_amos_voxel.csv'  # 输出异常检测结果的CSV文件路径
    # check_dir = ''

    filter_type = "voxel"
    if filter_type == "voxel":
        # 读取标准数量和阈值数据
        reference_summary = pd.read_csv(reference_summary_csv, index_col=0)

        # 处理伪标签文件
        process_fake_labels(input_dir, reference_summary, output_csv)

        # 拷贝伪标签
        # df = pd.read_csv(output_csv)
        # df = df.sort_values(by='anomalies', key=lambda x: x.apply(len), ascending=False)
        # check_data_files = df.head(15)['file_name'].tolist()
        # for check_data_file in tqdm(check_data_files):
        #         shutil.copy(
        #             os.path.join(input_dir, check_data_file),
        #             os.path.join(check_dir, check_data_file),
        #         )
        #
        #         shutil.copy(
        #             os.path.join(inptu_img_dir, 'LLD_'+check_data_file.split('.')[0]+'_0000.nii.gz'),
        #             os.path.join(check_dir, 'LLD_'+check_data_file.split('.')[0]+'_0000.nii.gz'),
        #         )
    elif filter_type == "entropy":
        sample_bank = filter_fake_labels_by_entropy(input_dir, 14)
        with open(output_csv.replace("csv", "json"), "w") as f:
            f.write(json.dumps(sample_bank, indent=4))

    elif filter_type == "check_entropy":
        bad_cases = []
        with open(output_csv, "r") as f:
            sample_bank = json.loads(f.read())
        for sample_key, filter_infos in tqdm(sample_bank):
            # filter_infos = sample_bank[sample_key]
            bad_num = 0
            for i in range(1, 14):
                if str(i) in filter_infos.keys() and filter_infos[str(i)] == 0:
                    bad_num = bad_num + 1
            if bad_num > 12:
                bad_cases.append(sample_key)

        for bad_case in tqdm(bad_cases):
            shutil.copy(
                os.path.join(input_dir, "{}.nii.gz".format(bad_case)),
                os.path.join(check_dir, "{}.nii.gz".format(bad_case)),
            )

            shutil.copy(
                os.path.join(inptu_img_dir, "LLD_" + bad_case + "_0000.nii.gz"),
                os.path.join(check_dir, "LLD_" + bad_case + "_0000.nii.gz"),
            )
