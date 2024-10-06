import os
import re
from tqdm import tqdm
import json
from datetime import datetime
import SimpleITK as sitk
import numpy as np
import pandas as pd

# MODALITY_PAIR = {
#     "default": "0000",
#     "C+A": "0001",
#     "C+Delay": "0002",
#     "C+V": "0003",
#     "C-Pre": "0004",
#     "DWI": "0005",
#     "InPhase": "0006",
#     "OutPhase": "0007",
#     "T2WI": "0008",
# }
# 从numpy数组创建SimpleITK图像

def regis_label_postprocess(image_path,label_path):
    # 读取图像和伪标签


    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # 转换为numpy数组
    image_np = sitk.GetArrayFromImage(image)
    label_np = sitk.GetArrayFromImage(label)

    # 生成图像的掩码，假设空白区域的像素值为0
    mask = image_np > 0

    # 应用掩码到伪标签
    label_processed = label_np * mask

    # 转换回SimpleITK图像并保存
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
    从给定的文件名中提取image_id。
    文件名可能是以下几种形式：
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
        origin_image_path ,
        temp_output_label_path,
        output_label_path,
        output_image_path,
        filter_path=None,
):

    if filter_path is not None:
        df = pd.read_csv(filter_path)
        filter_file = [ f.split('.')[0].split('_')[1] for f in df['file_name'].tolist()]
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
        #image_id = extract_image_id(output_image_file_name)
        lable_path = os.path.join(
            temp_output_label_path,
            'amos_{:04d}.nii.gz'.format(int(file_idx))
        )
        # img拷贝
        os.makedirs(output_image_path, exist_ok=True)
        replace_symlink(
            train_image_path,
            os.path.join(
                output_image_path,
                output_image_file_name
            )
        )
        # label拷贝
        replace_symlink(
            lable_path,
            os.path.join(
                output_label_path,
                '_'.join(output_image_file_name.split('_')[:-1]) + '.nii.gz'
            )
        )
        # 如果存在softmax的结果也保存上
        if os.path.exists(
                lable_path.replace('.nii.gz', '_softmax.npy')
        ):
            replace_symlink(
                lable_path.replace('.nii.gz', '_softmax.npy'),
                os.path.join(
                    output_label_path,
                    '_'.join(output_image_file_name.split('_')[:-1]) + '_softmax.npy'
                )
            )
def preprocess_lld_datset_plabel(
        origin_image_path ,
        temp_output_label_path,
        output_label_path,
        output_image_path,
        pair_json_path="Dataset667_FLARE24v1-20240809.json",
        bad_image_file_path='test',
        filter_path=None,
        use_regis_dataset=False,
):
    bad_image_files = []
    with open(pair_json_path, 'r') as f:
        pair_Dict = json.loads(f.read())

    if filter_path is not None:
        df = pd.read_csv(filter_path)
        filter_file = [ f.split('.')[0].replace('LLD_', '') for f in df['file_name'].tolist() ]
    else:
        filter_file = None

    # 读取lld的原始数据
    for file_idx in tqdm(range(len(pair_Dict['input_image_file_path']))):

        train_image_path = pair_Dict['input_image_file_path'][file_idx]
        output_image_file_name = os.path.basename(pair_Dict['output_image_file_path'][file_idx])
        image_id = extract_image_id(output_image_file_name)
        if filter_file is not None:
            if image_id in filter_file:
                print("{} has being filt!".format(image_id))
                continue
        if 'LLD_' in os.listdir(temp_output_label_path)[0]:
            lable_path = os.path.join(
                temp_output_label_path,
                'LLD_{}.nii.gz'.format(int(image_id))
            )
        else:
            lable_path = os.path.join(
                temp_output_label_path,
                '{}.nii.gz'.format(int(image_id))
            )
        # 代码bug 已修复
        current_image_prefix = '_'.join([os.path.basename(train_image_path).split('_')[0], ''])
        all_current_image_files = [f for f in os.listdir(origin_image_path) if current_image_prefix in f]
        target_image, _, _ = load_data(train_image_path)
        for cureent_image_file in all_current_image_files:

            if use_regis_dataset:
                # 处理配准标签需要进行后处理
                try:
                    label_processed_sitk = regis_label_postprocess(os.path.join(origin_image_path, cureent_image_file), lable_path)
                except ValueError as e:
                    print('{}:size bug! please check the label file!'.format(lable_path))
                    continue
                except RuntimeError as e:
                    print('{}:file error! please check the label file!'.format(os.path.join(origin_image_path, cureent_image_file)))
                    continue
                sitk.WriteImage(
                    label_processed_sitk,
                    os.path.join(
                        output_label_path,
                        '_'.join(cureent_image_file.split('_')[:-1]) + '.nii.gz'
                    )
                )
                # img软连接
                os.makedirs(output_image_path, exist_ok=True)
                replace_symlink(
                    os.path.join(origin_image_path, cureent_image_file),
                    os.path.join(
                        output_image_path,
                        '_'.join(cureent_image_file.split('_')[:-1]) + '_0000.nii.gz'
                    )
                )
                # 如果存在softmax的结果也保存上
                if os.path.exists(
                    lable_path.replace('.nii.gz', '_softmax.npy'),
                ):
                    replace_symlink(
                        lable_path.replace('.nii.gz', '_softmax.npy'),
                        os.path.join(
                            output_label_path,
                            '_'.join(cureent_image_file.split('_')[:-1]) + '_softmax.npy'
                        )
                    )

            else:
                origni_img, _, _ = load_data(os.path.join(origin_image_path, cureent_image_file))
                # 使用same size不需要比较
                if not (np.array(target_image.shape) == np.array(origni_img.shape)).all():
                    print("{} bad".format(cureent_image_file))
                    bad_image_files.append(cureent_image_file)
                else:
                    # img软连接
                    os.makedirs(output_image_path, exist_ok=True)
                    replace_symlink(
                        os.path.join(origin_image_path, cureent_image_file),
                        os.path.join(
                            output_image_path,
                            '_'.join(cureent_image_file.split('_')[:-1]) + '_0000.nii.gz'
                        )
                    )
                    # label软链接
                    replace_symlink(
                        lable_path,
                        os.path.join(
                            output_label_path,
                            '_'.join(cureent_image_file.split('_')[:-1]) + '.nii.gz'
                        )
                    )
    with open(bad_image_file_path, 'w') as f:
        f.write(json.dumps(bad_image_files, indent=4))



# def extract_modality_id(filename):
#     """
#     从给定的文件名中提取模态。
#     文件名可能是以下几种形式：
#     1. MR129136_1_C+A_0000.nii.gz ，模态是C+A
#     2. MR-391135_1_C+A_0000.nii.gz, 模态是C+A
#     3. amos_7427_0000.nii.gz，模态是default
#     """
#     # 检查文件名是否包含已知模态
#     for modality in MODALITY_PAIR.keys():
#         if modality in filename:
#             return MODALITY_PAIR['modality']
#
#     # 如果没有找到已知模态，则默认为"default"
#     return "0000"

def process_lld_data_img(path_info, output_prefix="default"):
    '''
    提取只有T2的数据,
    '''
    num_of_dir = len(path_info["train_image_path"])
    image_file_outputID_pair = {
        "input_image_file_path": [],
        "output_image_file_path": [],
    }

    for i in range(num_of_dir):
        train_image_path = path_info["train_image_path"][i]
        output_image_path = path_info["output_image_path"]

        os.makedirs(output_image_path, exist_ok=True)

        # 获取train_image中的文件名
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
                # output_image = os.path.join(
                #         output_image_path,
                #         f"{output_prefix}_{'_'.join(image_file.split('_')[-4:])}",
                #     )

                replace_symlink(image_path, output_image)  # 创建软链接
                image_file_outputID_pair["input_image_file_path"].append(image_path)
                image_file_outputID_pair["output_image_file_path"].append(output_image)
            else:
                print(f"{image_path} file not found!")


    return image_file_outputID_pair

if __name__ == "__main__":

    stage = 1
    data_root_path = '/ChenQiang/DataSet/Medical/MICCAI24_FLARE/Dataset'
    # lld
    # ## stage1 need
    origin_image_path = os.path.join(data_root_path, "MR/Training/LLD-MMRI-3984")
    #origin_image_path = os.path.join(data_root_path, "MR/Training/LLD-MMRI-3984_samesize")
    #origin_image_path = os.path.join(data_root_path, "MR/Training/regis_data/LLD-MMRI-3984-regis-samesize")
    #origin_image_path = os.path.join(data_root_path, "MR/Training/regis_data/LLD-MMRI-3984-regis-Syn-differentsize")
    img_temp_path = os.path.join(data_root_path, "MR/Training/PLabel_image/LLD-MMRI-3984-check_bad")
    #img_temp_path = os.path.join(data_root_path, "MR/Training/LLD-MMRI-3984-T2_tmp")
    data_pair_path = os.path.join(data_root_path, "MR/Training/preprocess_data_pair")
    # ## engines need
    use_regis_dataset = True  # 是否使用regis的数据
    # temp_output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-T2_tmp_v1")
    #temp_output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-T2_tmp_v2")
    temp_output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-T2_tmp_v3_softmax")
    #########################
    #### regis_same_v1
    # output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-regis_same_v1")
    # output_image_path = os.path.join(data_root_path, "MR/Training/PLabel_image/LLD-MMRI-3984-regis_same_v1")
    #### regis_same (old)
    # output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-regis_v2")
    # output_image_path = os.path.join(data_root_path, "MR/Training/PLabel_image/LLD-MMRI-3984-regis_v2")
    #### regis_same (new)
    # output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-regis_same_v2")
    # output_image_path = os.path.join(data_root_path, "MR/Training/PLabel_image/LLD-MMRI-3984-regis_same_v2")
    ### regis_same_v3
    output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-regis_same_v3_softmax")
    output_image_path = os.path.join(data_root_path, "MR/Training/PLabel_image/LLD-MMRI-3984-regis_same_v3_softmax")
    #### regis_diff_v1
    # output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-regis_diff_v1")
    # output_image_path = os.path.join(data_root_path, "MR/Training/PLabel_image/LLD-MMRI-3984-regis_diff_v1")
    #### regis_diff_v2
    # output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-regis_diff_v2")
    # output_image_path = os.path.join(data_root_path, "MR/Training/PLabel_image/LLD-MMRI-3984-regis_diff_v2")
    # #### regis_diff_v3
    # output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-regis_diff_v3_softmax")
    # output_image_path = os.path.join(data_root_path, "MR/Training/PLabel_image/LLD-MMRI-3984-regis_diff_v3_softmax")
    #### temp
    # output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/LLD-MMRI-3984-temp")
    # output_image_path = os.path.join(data_root_path, "MR/Training/PLabel_image/LLD-MMRI-3984-temp")
    #########################
    filter_path = os.path.join(data_root_path, 'static_info', 'filter_plabel_lld_v3.csv')
    pair_json_path = "Dataset667_FLARE24v1-20240815.json"
    bad_image_file_path = os.path.join(data_root_path, "MR/Training/preprocess_data_pair/LLD_v3.json")

    # ## amos
    # origin_image_path = os.path.join(data_root_path, "MR/Training/AMOS_MR_good_spacing-833")
    # temp_output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/AMOS_MR_good_spacing-833_tmp_v3_softmax")
    # output_label_path = os.path.join(data_root_path, "MR/Training/PLabel/AMOS_MR_good_spacing-833_v3_softmax")
    # output_image_path = os.path.join(data_root_path, "MR/Training/PLabel_image/AMOS_MR_good_spacing-833_v3_softmax")
    # filter_path = os.path.join(data_root_path, 'static_info', 'filter_plabel_amos_v3.csv')


    # 改造版本


    if stage == 1:
        path_info = {
            "train_image_path": [
                origin_image_path,
            ],
            "output_image_path": img_temp_path
        }
        image_file_outputID_pair = process_lld_data_img(path_info, "LLD")

        now = datetime.now()
        output_pair_name = "Dataset667_FLARE24v1"
        # 转换为字符串，指定格式
        date_str = now.strftime("%Y%m%d")
        with open(
            os.path.join(
                data_pair_path,
                "{}-{}.json".format(output_pair_name, date_str),
            ),
            "w",
        ) as f:
            f.write(json.dumps(image_file_outputID_pair))
    elif stage == 2:
        os.makedirs(output_label_path, exist_ok=True)
        if 'amos' in origin_image_path.lower():
            preprocess_amos_datset_plabel(
                origin_image_path,
                temp_output_label_path,
                output_label_path,
                output_image_path,
                filter_path,
            )
        else:
            preprocess_lld_datset_plabel(
                origin_image_path,
                temp_output_label_path,
                output_label_path,
                output_image_path,
                os.path.join(data_pair_path, pair_json_path),
                bad_image_file_path,
                filter_path,
                use_regis_dataset,
            )
