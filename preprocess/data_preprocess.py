import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import traceback
from collections import OrderedDict
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from filter_plabel import process_fake_labels, static_gt_each_labels_num
from preprocess_flare24_real_mr_dataset import (preprocess_amos_datset_plabel,
                                                preprocess_lld_datset_plabel,
                                                process_lld_data_img)
from regis_batch import regis_data
from skimage.transform import resize

from engines.common import Inference
from engines.dataloader.utils import (change_axes_of_image,
                                      clip_and_normalize_mean_std,
                                      create_two_class_mask,
                                      crop_image_according_to_mask, load_data,
                                      resize_segmentation)
from utils.common_function import (crop_bbox_by_stand_spacing, load_checkpoint,
                                   parse_option)


def run_prepare_data(config, is_multiprocessing=True):
    data_prepare = model_train_data_process(config)
    if is_multiprocessing:
        pool = Pool(int(cpu_count() * 0.3))
        for data in data_prepare.data_list:
            try:
                pool.apply_async(data_prepare.process, (data,))
            except Exception as err:
                traceback.print_exc()
                print(
                    "Create image/label throws exception %s, with series_id %s!"
                    % (err, data_prepare.data_info)
                )

        pool.close()
        pool.join()
    else:
        for data in data_prepare.data_list:
            data_prepare.process(data)


class model_train_data_process(object):
    def __init__(self, config):
        self.config = config
        self.train_type = self.config.TRAINING_TYPE
        print(self.train_type)
        self.coarse_size = self.config.DATASET.COARSE.PREPROCESS_SIZE
        self.fine_size = self.config.DATASET.FINE.PREPROCESS_SIZE
        self.nor_dir = self.config.DATASET.IS_NORMALIZATION_DIRECTION
        self.extend_size = self.config.DATASET.EXTEND_SIZE
 
        self.image_path = os.path.join(
            config.DATASET.BASE_DIR, config.DATASET.TRAIN_IMAGE_PATH
        )
        self.mask_path = os.path.join(
            config.DATASET.BASE_DIR, config.DATASET.TRAIN_MASK_PATH
        )
        self.preprocess_coarse_path = os.path.join(
            config.DATASET.BASE_DIR, config.DATASET.COARSE.PROPRECESS_PATH
        )
        self.preprocess_fine_path = os.path.join(
            config.DATASET.BASE_DIR, config.DATASET.FINE.PROPRECESS_PATH
        )
        self.data_list = os.listdir(self.image_path)
        os.makedirs(self.preprocess_coarse_path, exist_ok=True)
        os.makedirs(self.preprocess_fine_path, exist_ok=True)
        self.is_abdomen_crop = config.DATASET.IS_ABDOMEN_CROP

    def process(self, image_id):
        is_softmax_exists = False
        data_id = image_id.split("_0000.nii.gz")[0]
        image, image_spacing, image_direction, image_itk_info = load_data(
            join(self.image_path, data_id + "_0000.nii.gz"))
        mask, _, mask_direction, label_itk_info = load_data(
            join(self.mask_path, data_id + ".nii.gz")
        )
        # if softmax matrix exists
        if os.path.exists(os.path.join(self.mask_path, data_id + "_softmax.npy")):
            softmax_image = np.load(
                os.path.join(self.mask_path, data_id + "_softmax.npy")
            )
            is_softmax_exists = True
        assert image_direction.all() == mask_direction.all()
        if self.is_abdomen_crop:
            (
                image,
                crop_start_slice,
                crop_end_slice,
                original_start,
                original_end,
            ) = crop_bbox_by_stand_spacing(image, image_spacing)
            mask, _, _, _, _ = crop_bbox_by_stand_spacing(mask, image_spacing)
        #
        image = image.transpose(1, 2, 0)
        mask = mask.transpose(1, 2, 0)
        if is_softmax_exists:
            softmax_image = softmax_image.transpose(0, 2, 3, 1)

        if self.nor_dir:
            image = change_axes_of_image(image, image_direction)
            mask = change_axes_of_image(mask, mask_direction)
            if is_softmax_exists:
                for s_i in range(softmax_image.shape[0]):
                    softmax_image[s_i] = change_axes_of_image(
                        softmax_image[s_i], mask_direction
                    )
        if "coarse" in self.train_type:
            data_info = OrderedDict()
            data_info["raw_shape"] = image.shape
            data_info["raw_spacing"] = image_spacing
            resize_spacing = image_spacing * image.shape / self.coarse_size
            data_info["resize_spacing"] = resize_spacing
            data_info["image_direction"] = image_direction
            with open(
                os.path.join(self.preprocess_coarse_path, "%s_info.pkl" % data_id), "wb"
            ) as f:
                pickle.dump(data_info, f)

            image_resize = resize(
                image, self.coarse_size, order=3, mode="edge", anti_aliasing=False
            )

            mask_resize = resize_segmentation(mask, self.coarse_size, order=0)
            mask_binary = create_two_class_mask(mask_resize)

            image_normal = clip_and_normalize_mean_std(image_resize)
            np.savez_compressed(
                os.path.join(self.preprocess_coarse_path, "%s.npz" % data_id),
                data=image_normal[None, ...],
                seg=mask_binary[None, ...],
            )

            margin = [
                int(self.extend_size / image_spacing[0]),
                int(self.extend_size / image_spacing[1]),
                int(self.extend_size / image_spacing[2]),
            ]
            crop_image, crop_mask = crop_image_according_to_mask(
                image, np.array(mask, dtype=int), margin
            )
            if is_softmax_exists:
                new_crop_image = []
                for s_i in range(softmax_image.shape[0]):
                    crop_softmax_image_tmp, _ = crop_image_according_to_mask(
                        softmax_image[s_i], np.array(mask, dtype=int), margin
                    )
                    new_crop_image.append(crop_softmax_image_tmp)
                crop_softmax_image = np.array(new_crop_image)
        else:
            crop_image = image
            crop_mask = mask
            if is_softmax_exists:
                crop_softmax_image = softmax_image

        if "fine" in self.train_type:
            data_info_crop = OrderedDict()
            data_info_crop["raw_shape"] = image.shape
            data_info_crop["crop_shape"] = crop_image.shape
            data_info_crop["raw_spacing"] = image_spacing
            resize_crop_spacing = image_spacing * crop_image.shape / self.fine_size
            data_info_crop["resize_crop_spacing"] = resize_crop_spacing
            data_info_crop["image_direction"] = image_direction
            with open(
                os.path.join(self.preprocess_fine_path, "%s_info.pkl" % data_id), "wb"
            ) as f:
                pickle.dump(data_info_crop, f)

            crop_image_resize = resize(
                crop_image, self.fine_size, order=3, mode="edge", anti_aliasing=False
            )
            crop_mask_resize = resize_segmentation(crop_mask, self.fine_size, order=0)
            crop_image_normal = clip_and_normalize_mean_std(crop_image_resize)
            if is_softmax_exists:
                crop_softmax_resize = resize(
                    crop_softmax_image,
                    [crop_softmax_image.shape[0], *self.fine_size],
                    order=1,
                    mode="edge",
                    anti_aliasing=False,
                )
            else:
                if len(crop_image_resize.shape) == (len(self.fine_size) + 1):
                    crop_softmax_resize = np.zeros([*crop_image_resize.shape]).astype(
                        np.int8
                    )
                else:
                    crop_softmax_resize = np.zeros(
                        [1, *crop_image_resize.shape]
                    ).astype(np.int8)

            np.savez_compressed(
                os.path.join(self.preprocess_fine_path, "%s.npz" % data_id),
                data=crop_image_normal[None, ...],
                seg=crop_mask_resize[None, ...],
                softmax_iamge=crop_softmax_resize[None, ...],
            )
            print("End processing %s." % data_id)


if __name__ == "__main__":
    _, config = parse_option("other")

    data_root_path = config.MR_DATA_PREPROCESS.ROOT_PATH  # /datasets
    preprocess_stage = config.MR_DATA_PREPROCESS.STAGE
    if "LLD-MMRI" in config.MR_DATA_PREPROCESS.MR_RAW_DATA_PATH:
        data_set_name = "lld"
    else:
        data_set_name = "amos"
    if preprocess_stage > 1:

        ############################ stage1 ############################
        origin_image_path = os.path.join(
            data_root_path, config.MR_DATA_PREPROCESS.MR_RAW_DATA_PATH
        )
        data_pair_path = os.path.join(
            data_root_path, config.MR_DATA_PREPROCESS.DATA_PAIR_PATH
        )
        temp_output_label_path = os.path.join(
            data_root_path,
            config.MR_DATA_PREPROCESS.TEMP_PREPROCESSED_PSUEDO_LABEL_PATH,
        )
        os.makedirs(temp_output_label_path, exist_ok=True)
        # Pseudo-label Preprocessing:
        # Utilize inference.py to generate corresponding predicted labels for MR data
        # that these are intermediate prediction results since the pseudo-labels need to be filtered before final use, hence the output directory name should include "temp").
        # Preprocessing, Step 1: Process the LLD dataset (special) Copy the corresponding original files.
        # This includes only C+Delay.
        if data_set_name == "lld":
            img_temp_path = os.path.join(
                data_root_path, config.MR_DATA_PREPROCESS.TEMP_MR_RAW_DATA_PATH
            )

            print("current is LLD dataset.\n processing mr_preprocessed_stage1")
            path_info = {
                "train_image_path": [
                    origin_image_path,
                ],
                "output_image_path": img_temp_path,
            }
            image_file_outputID_pair = process_lld_data_img(path_info, "LLD")

            now = datetime.now()
            output_pair_name = "Dataset667_FLARE24v3"
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

        # # # 2 using infercence to inference_new_result
        # ############################ stage2 ############################

        config.defrost()
        config.DATASET.VAL_IMAGE_PATH = config.MR_DATA_PREPROCESS.TEMP_MR_RAW_DATA_PATH
        config.VAL_OUTPUT_PATH = temp_output_label_path
        old_trainning_type = config.TRAINING_TYPE
        old_coarse_size = config.DATASET.COARSE.SIZE
        old_fine_size = config.DATASET.FINE.SIZE
        # Refine the corresponding parameters based on the previously trained model.
        fine_model_checkpoint = load_checkpoint(
            config.FINE_MODEL_PATH, config.CHECK_POINT_NAME
        )
        config.DATASET.FINE.SIZE = fine_model_checkpoint["config"].DATASET.FINE.SIZE
        config.DATASET.COARSE.SIZE = fine_model_checkpoint["config"].DATASET.COARSE.SIZE
        config.TRAINING_TYPE = fine_model_checkpoint["config"].TRAINING_TYPE
        config.freeze()
        predict = Inference(config)
        predict.run()
        print("stage2:inference psuedo label")
        # 3. Registration (only needs to be done once)
        # ############################ stage3 ############################
        # Need a lots of time
        if data_set_name == "lld":
            regis_data_output_path = os.path.join(
                config.MR_DATA_PREPROCESS.ROOT_PATH,
                config.MR_DATA_PREPROCESS.REGIS.OUTPUT_PATH,
            )
            regis_data(
                origin_image_path,
                regis_data_output_path,
            )
            print("stage3:Registration lld data")
        # # # 4 filter plabel
        # # ############################ stage3 ############################
        config.defrost()
        config.TRAINING_TYPE = old_trainning_type
        config.DATASET.COARSE.SIZE = old_coarse_size
        config.DATASET.FINE.SIZE = old_fine_size
        config.freeze()
        ct_gt_file = os.path.join(data_root_path, config.MR_DATA_PREPROCESS.CT_GT_PATH)
        output_each_ct_labels_num_static_path = os.path.join(
            data_root_path, config.MR_DATA_PREPROCESS.OUTPUT_EACH_CT_LABELS_NUM_STATIC_PATH
        )
        filter_path = os.path.join(
            data_root_path, config.MR_DATA_PREPROCESS.FILTER_CASE_SAVE_PATH
        )
        print("stage4:filter Psuedo label")
        # Count the number of voxels for each organ category in the standard CT data.
        if not os.path.exists(output_each_ct_labels_num_static_path):
            _ = static_gt_each_labels_num(
                ct_gt_file,
                output_each_ct_labels_num_static_path,
            )
        # Filter based on voxels.
        reference_summary = pd.read_csv(output_each_ct_labels_num_static_path, index_col=0)
        # Get the filtered data and export the filter results to CSV.
        process_fake_labels(
            config.VAL_OUTPUT_PATH,
            reference_summary,
            filter_path,
        )
        output_label_path = os.path.join(data_root_path, config.DATASET.TRAIN_MASK_PATH)
        output_image_path = os.path.join(data_root_path, config.DATASET.TRAIN_IMAGE_PATH)

        bad_image_file_path = os.path.join(
            data_root_path, config.MR_DATA_PREPROCESS.BAD_CASE_PATH
        )
        use_regis_dataset = config.MR_DATA_PREPROCESS.IS_LLD_REGIS_DATA
        os.makedirs(output_label_path, exist_ok=True)
        os.makedirs(output_image_path, exist_ok=True)
        os.makedirs(os.path.dirname(bad_image_file_path), exist_ok=True)
        if data_set_name == "lld":
            if use_regis_dataset:
                # If you use registered data, you need to modify the 'origin_image_path'.
                origin_image_path = regis_data_output_path
            preprocess_lld_datset_plabel(
                origin_image_path,
                temp_output_label_path,
                output_label_path,
                output_image_path,
                os.path.join(
                    data_pair_path,
                    "{}-{}.json".format(output_pair_name, date_str),
                ),
                bad_image_file_path,
                filter_path,
                use_regis_dataset,
            )
        else:
            preprocess_amos_datset_plabel(
                origin_image_path,
                temp_output_label_path,
                output_label_path,
                output_image_path,
                filter_path,
            )
    # 6.Preprocess the data required for model generation
    run_prepare_data(config, True)

