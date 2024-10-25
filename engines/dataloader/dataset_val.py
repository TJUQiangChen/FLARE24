# 针对原始数据，构建符合spacing分布的新数据集
import os

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles

from torch.utils.data import Dataset

from .utils import change_axes_of_image, load_data
from utils.common_function import resize_by_spacing


class predict_dataset(Dataset):
    def __init__(self, config):
        super(predict_dataset, self).__init__()
        self.config = config
        self.data_path = os.path.join(config.DATASET.BASE_DIR, config.DATASET.VAL_IMAGE_PATH)
        print(config.DATASET.VAL_IMAGE_PATH)
        self.is_nor_dir = self.config.DATASET.IS_NORMALIZATION_DIRECTION
        # print(self.data_path) 
        self.series_ids = subfiles(self.data_path, join=False, suffix='gz')
    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        image_id = self.series_ids[idx].split("_")[1]
        raw_image, image_spacing, image_direction, itk_info= load_data(os.path.join(self.data_path,self.series_ids[idx]))
        if self.is_nor_dir:
            raw_image = change_axes_of_image(raw_image, image_direction)
        return {
                'image_path': self.series_ids[idx],
                'image_id': image_id,
                'raw_image': np.ascontiguousarray(raw_image),
                'raw_spacing': image_spacing,
                'image_direction': image_direction,
                'itk_info': itk_info,
                }




class valp_dataset(Dataset):
    def __init__(self, config):
        super(valp_dataset, self).__init__()
        self.config = config
        self.change_spacing_1 = config.DATASET.VAL_CHANGE_SPACING
        self.data_path = os.path.join(config.DATASET.BASE_DIR, config.DATASET.VAL_IMAGE_PATH)
     
        self.is_nor_dir = self.config.DATASET.IS_NORMALIZATION_DIRECTION

        self.series_ids_image = subfiles(os.path.join(self.data_path, 'imagesVal'), join=False, suffix='gz')
        self.series_ids_label = subfiles(os.path.join(self.data_path, 'labelsVal'), join=False, suffix='gz')



    def __len__(self):
        return len(self.series_ids_image)

    def __getitem__(self, idx):
        image_id = self.series_ids_image[idx].split("_")[1]
        image_file = self.series_ids_image[idx].split(".")[0]
        label_id = self.series_ids_label[idx].split("_")[1]

        labels, _, labels_direction,itk_info = load_data(os.path.join(os.path.join(self.data_path, 'labelsVal'),self.series_ids_label[idx]))
        raw_image, image_spacing, image_direction, _ = load_data(os.path.join(os.path.join(self.data_path, 'imagesVal'),self.series_ids_image[idx]))
        if self.is_nor_dir:
            raw_image = change_axes_of_image(raw_image, image_direction)
            labels = change_axes_of_image(labels, labels_direction)

        if self.change_spacing_1:
            raw_image, img_size, is_resample = resize_by_spacing(raw_image, image_spacing, [1, 1, 1], type='img')
            labels, _, is_resample = resize_by_spacing(labels, image_spacing, [1, 1, 1], type='label')
            image_spacing = np.array([1,1,1])

            if is_resample:
                print('current {} resampe different!'.format(image_id))

        return {
                'image_path': self.series_ids_image[idx],
                'image_id': image_id,
                'image_file': image_file,
                'raw_image': np.ascontiguousarray(raw_image),
                'raw_spacing': image_spacing,
                'image_direction': image_direction,
                'seg': np.ascontiguousarray(labels),
                'itk_info': itk_info,
                }
              
    