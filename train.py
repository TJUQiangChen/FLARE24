import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.common import Training, merge_tph_processed_data_img_and_label

from utils.common_function import parse_option, is_directory_only_symlinks


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10000"
    _, config = parse_option("train")

    # 如果是TPH processed的数据组合，那么在这里处理
    if config.TRAINING_TYPE == 'fine':
        split_preprocess_path = config.DATASET.FINE.SPLIT_PREPROCESS_PATH
        preprocess_PATH = config.DATASET.FINE.PROPRECESS_PATH
    elif config.TRAINING_TYPE == 'coarse':
        split_preprocess_path = config.DATASET.COARSE.SPLIT_PREPROCESS_PATH
        preprocess_PATH = config.DATASET.COARSE.PROPRECESS_PATH

    if len(split_preprocess_path) > 0:
        preprocessed_path = os.path.join(
                config.DATASET.BASE_DIR,
                preprocess_PATH,
            )
        if is_directory_only_symlinks(preprocessed_path):
            print('Del exists folder!')
            shutil.rmtree(preprocessed_path)
        path_info = {
            'train_data_path':[],
            'output_data_path':preprocessed_path
        }
        for tph_processed_path in split_preprocess_path:
            path_info['train_data_path'].append(
                os.path.join(
                    config.DATASET.BASE_DIR,
                    tph_processed_path
                )
            )
        merge_tph_processed_data_img_and_label(path_info)
        print('Finsh TPH Data Processed!')
        print('Begin Trainning...')
    # 否则就正常跑
    trainer = Training(config)
    trainer.train()
