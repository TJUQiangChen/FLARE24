import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from .dataloader import DataLoaderX, build_loader
from .dataloader.dataset_val import predict_dataset, valp_dataset
from .dataloader.utils import (
    change_axes_of_image,
    crop_image_according_to_bbox,
    extract_topk_largest_candidates,
    get_bbox_from_mask,
    input_downsample,
    output_upsample,
    to_one_hot,
)
from fvcore.nn import FlopCountAnalysis
from loguru import logger
from torch.cuda.amp import autocast
from torchinfo import summary
from tqdm import tqdm
from .losses import build_loss
from .lr_scheduler import build_scheduler
from .metrics import AverageMeter, run_evaluation
from .models import build_coarse_model, build_fine_model
from .optimizer import build_optimizer
from .trainer import Trainer
from utils.common_function import (
    crop_bbox_by_stand_spacing,
    crop_bbox_restore2raw,
    load_checkpoint,
    replace_symlink,
    resume_itk_info,
    seed_torch,
    to_cuda,
)

VAL_INFOS = {
    "image_file": [],
    "class1_dice": [],
    "class2_dice": [],
    "class3_dice": [],
    "class4_dice": [],
    "class5_dice": [],
    "class6_dice": [],
    "class7_dice": [],
    "class8_dice": [],
    "class9_dice": [],
    "class10_dice": [],
    "class11_dice": [],
    "class12_dice": [],
    "class13_dice": [],
    "global_avg_dice": [],
}

MODALITY_PAIR = {
    "default": "0000",
    "C+A": "0001",
    "C+Delay": "0002",
    "C+V": "0003",
    "C-Pre": "0004",
    "DWI": "0005",
    "InPhase": "0006",
    "OutPhase": "0007",
    "T2WI": "0008",
    "amos": "0009",
}


def extract_modality_id(filename):
    """
    Extract the modality from the given filenames.
    Filenames may take the following forms:

    MR129136_1_C+A_0000.nii.gz, the modality is C+A
    MR-391135_1_C+A_0000.nii.gz, the modality is C+A
    amos_7427_0000.nii.gz, the modality is default
    """
    # Check if the filename contains a known modality.
    for modality in MODALITY_PAIR.keys():
        if modality in filename:
            return MODALITY_PAIR[modality]

    # If no known modality is found, default to "default".
    return "0000"


def merge_tph_processed_data_img_and_label(path_info, output_prefix="default"):
    num_of_dir = len(path_info["train_data_path"])
    output_data_path = path_info["output_data_path"]

    image_file_outputID_pair = {
        "input_data_file_path": [],
        "output_data_file_path": [],
    }
    num_output_id = 1

    for i in range(num_of_dir):
        train_image_path = path_info["train_data_path"][i]

        os.makedirs(output_data_path, exist_ok=True)

        # get the filename which in 'train_image'
        image_files = sorted(
            [f for f in os.listdir(train_image_path) if f.endswith(".npz")]
        )

        for file_id in tqdm(range(len(image_files))):
            image_file = image_files[file_id]
            image_path = os.path.join(train_image_path, image_file)
            info_path = os.path.join(
                train_image_path, image_file.replace(".npz", "_info.pkl")
            )
            modal_id = extract_modality_id(image_file)

            if os.path.exists(image_path):
                output_image = os.path.join(
                    output_data_path,
                    f"{output_prefix}_{num_output_id:04d}_{modal_id}.npz",
                )
                output_info = os.path.join(
                    output_data_path,
                    f"{output_prefix}_{num_output_id:04d}_{modal_id}_info.pkl",
                )

                replace_symlink(info_path, output_info)
                replace_symlink(image_path, output_image)

                image_file_outputID_pair["input_data_file_path"].append(image_path)
                image_file_outputID_pair["output_data_file_path"].append(output_image)
                num_output_id += 1
            else:
                print(f"{image_path} file not found!")

    return image_file_outputID_pair


class Training(object):
    def __init__(self, config) -> None:
        # set dp
        self.config = config
        self.train_type = config.TRAINING_TYPE
        print("############## Train Type: {} ##############".format(self.train_type))

    def train(self):
        if self.config.DIS:
            mp.spawn(
                self._main_worker,
                args=(self.config,),
                nprocs=self.config.WORLD_SIZE,
            )
        else:
            self._main_worker(0, self.config)

    def _main_worker(self, local_rank, config):
        if local_rank == 0:
            config.defrost()
            config.EXPERIMENT_ID = f"{config.WANDB.TAG}_{config.DATASET.DATASET_NAME}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
            config.freeze()
            if self.train_type == "coarse":
                wandb.init(
                    project=config.WANDB.COARSE_PROJECT,
                    name=config.EXPERIMENT_ID,
                    config=config,
                    mode=config.WANDB.MODE,
                )
            elif self.train_type == "fine":
                wandb.init(
                    project=config.WANDB.FINE_PROJECT,
                    name=config.EXPERIMENT_ID,
                    config=config,
                    mode=config.WANDB.MODE,
                )
        np.set_printoptions(formatter={"float": "{: 0.4f}".format}, suppress=True)
        torch.cuda.set_device(local_rank)
        if config.DIS:
            dist.init_process_group(
                "nccl",
                init_method="env://",
                rank=local_rank,
                world_size=config.WORLD_SIZE,
            )
        seed = config.SEED + local_rank
        seed_torch(seed)
        cudnn.benchmark = True

        if self.train_type == "fine":
            train_loader, val_loader = build_loader(
                config,
                config.DATASET.FINE.SIZE,
                os.path.join(
                    config.DATASET.BASE_DIR, config.DATASET.FINE.PROPRECESS_PATH
                ),
                os.path.join(
                    config.DATASET.BASE_DIR, config.DATASET.FINE.PROPRECESS_UL_PATH
                ),
                config.MODEL.FINE.POOL_OP_KERNEL_SIZES,
                config.DATASET.FINE.NUM_EACH_EPOCH,
            )
            model = build_fine_model(config).cuda()

            losses = {
                "ct": build_loss(
                    config,
                    config.MODEL.DEEP_SUPERVISION,
                    config.MODEL.FINE.POOL_OP_KERNEL_SIZES,
                    config.LOSS.IGNORE_EDGE,
                ),
                "mr": build_loss(
                    config,
                    False,
                    config.MODEL.FINE.POOL_OP_KERNEL_SIZES,
                    config.LOSS.IGNORE_EDGE,
                ),
            }
            if self.config.TRAIN.MULTI_TASK.IS_OPEN:
                # TODO:需要改
                losses["center_class"] = build_loss(
                    config,
                    False,
                    config.MODEL.FINE.POOL_OP_KERNEL_SIZES,
                    config.LOSS.IGNORE_EDGE,
                    True,
                )
        elif self.train_type == "coarse":
            train_loader, val_loader = build_loader(
                config,
                config.DATASET.COARSE.SIZE,
                os.path.join(
                    config.DATASET.BASE_DIR, config.DATASET.COARSE.PROPRECESS_PATH
                ),
                os.path.join(
                    config.DATASET.BASE_DIR, config.DATASET.COARSE.PROPRECESS_UL_PATH
                ),
                config.MODEL.COARSE.POOL_OP_KERNEL_SIZES,
                config.DATASET.COARSE.NUM_EACH_EPOCH,
            )
            model = build_coarse_model(config).cuda()
            losses = {
                "ct": build_loss(
                    config,
                    config.MODEL.DEEP_SUPERVISION,
                    config.MODEL.COARSE.POOL_OP_KERNEL_SIZES,
                    config.LOSS.IGNORE_EDGE,
                ),
                "mr": build_loss(
                    config,
                    False,
                    config.MODEL.COARSE.POOL_OP_KERNEL_SIZES,
                    config.LOSS.IGNORE_EDGE,
                ),
            }
        if config.DIS:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )
        logger.info(f"\n{model}\n")

        optimizer = build_optimizer(config, model)
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
        trainer = Trainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            losses=losses,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        trainer.train()


class Inference(object):
    def __init__(self, config) -> None:
        self.config = config
        self.train_type = config.TRAINING_TYPE
        self.is_TTA = config.VAL.TTA
        print("############## Train Type: {} ##############".format(self.train_type))
        self.output_path = self.config.VAL_OUTPUT_PATH
        self.is_multi_task = self.config.TRAIN.MULTI_TASK.IS_OPEN
        os.makedirs(self.output_path, exist_ok=True)
        self.coarse_size = self.config.DATASET.COARSE.SIZE
        self.fine_size = self.config.DATASET.FINE.SIZE
        self.extend_size = self.config.DATASET.EXTEND_SIZE
        self.is_post_process = self.config.VAL.IS_POST_PROCESS
        self.is_nor_dir = self.config.DATASET.IS_NORMALIZATION_DIRECTION
        self.is_with_dataloader = self.config.VAL.IS_WITH_DATALOADER
        self.save_softmax = self.config.INFERENCE.SAVE_SOFTMAX
        self.is_overwrite_predict_mask = self.config.INFERENCE.IS_OVERWRITE_PREDICT_MASK
        if self.is_with_dataloader:
            val_dataset = predict_dataset(config)
            self.val_loader = DataLoaderX(
                val_dataset,
                batch_size=1,
                num_workers=0,
                pin_memory=config.DATALOADER.PIN_MEMORY,
                shuffle=False,
            )
            print(len(self.val_loader))
        else:
            self.val_loader = predict_dataset(config)
        cudnn.benchmark = True

    def run(self):
        torch.cuda.synchronize()
        t_start = time.time()
        tbar = tqdm(self.val_loader, ncols=150)
        with autocast():
            with torch.no_grad():
                for idx, image_dict in enumerate(tbar):
                    image_dict = (
                        image_dict[0] if type(image_dict) is list else image_dict
                    )
                    if self.is_with_dataloader:
                        image_path = image_dict["image_path"][0]
                        image_id = image_dict["image_id"][0]
                        raw_image = np.array(image_dict["raw_image"].squeeze(0))
                        raw_spacing = np.array(image_dict["raw_spacing"][0])
                        image_direction = np.array(image_dict["image_direction"][0])
                        itk_info = {
                            "spacing": tuple(
                                [float(v) for v in image_dict["itk_info"]["spacing"]]
                            ),
                            "direction": tuple(
                                [float(v) for v in image_dict["itk_info"]["direction"]]
                            ),
                            "origin": tuple(
                                [float(v) for v in image_dict["itk_info"]["origin"]]
                            ),
                        }
                    else:
                        image_path = image_dict["image_path"]
                        image_id = image_dict["image_id"]
                        raw_image = image_dict["raw_image"]
                        raw_spacing = image_dict["raw_spacing"]
                        image_direction = image_dict["image_direction"]
                        itk_info = image_dict["itk_info"]

                    if not self.is_overwrite_predict_mask:
                        if os.path.exists(
                            os.path.join(
                                self.output_path,
                                image_path.replace("_0000.nii.gz", ".nii.gz"),
                            )
                        ):
                            # If the prediction result is not forced to be overwritten. If the prediction file has already been generated, then skip it and do not regenerate.
                            continue

                    if self.train_type == "coarse-fine":
                        coarse_image = (
                            torch.from_numpy(raw_image)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .float()
                        )
                        raw_image_shape = raw_image.shape
                        coarse_resize_factor = np.array(raw_image.shape) / np.array(
                            self.coarse_size
                        )
                        coarse_image = input_downsample(coarse_image, self.coarse_size)
                        coarse_image = self.coarse_predict(
                            coarse_image,
                            self.config.COARSE_MODEL_PATH,
                            self.config.COARSE_CHECK_POINT_NAME,
                        )
                        coarse_pre = F.softmax(coarse_image, 1)
                        coarse_pre = coarse_pre.cpu().float()
                        torch.cuda.empty_cache()
                        coarse_mask = (
                            coarse_pre.argmax(1)
                            .squeeze(axis=0)
                            .numpy()
                            .astype(np.uint8)
                        )
                        lab_unique = np.unique(coarse_mask)
                        coarse_mask = to_one_hot(coarse_mask)
                        coarse_mask = extract_topk_largest_candidates(
                            coarse_mask, lab_unique, 1
                        )
                        coarse_bbox = get_bbox_from_mask(coarse_mask)
                        raw_bbox = [
                            [
                                int(coarse_bbox[0][0] * coarse_resize_factor[0]),
                                int(coarse_bbox[0][1] * coarse_resize_factor[0]),
                            ],
                            [
                                int(coarse_bbox[1][0] * coarse_resize_factor[1]),
                                int(coarse_bbox[1][1] * coarse_resize_factor[1]),
                            ],
                            [
                                int(coarse_bbox[2][0] * coarse_resize_factor[2]),
                                int(coarse_bbox[2][1] * coarse_resize_factor[2]),
                            ],
                        ]
                        margin = [self.extend_size / raw_spacing[i] for i in range(3)]
                        crop_image, crop_fine_bbox = crop_image_according_to_bbox(
                            raw_image, raw_bbox, margin
                        )
                    elif self.train_type == "fine":
                        crop_image = raw_image

                    abdomen_cropped_image = crop_image
                    crop_image = (
                        torch.from_numpy(abdomen_cropped_image)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    crop_image = input_downsample(crop_image, self.fine_size)
                    if self.is_TTA:
                        crop_image_pred = self.fine_predict(
                            crop_image,
                            self.config.FINE_MODEL_PATH,
                            self.config.CHECK_POINT_NAME,
                        )
                        # TTA
                        predict = crop_image_pred
                        predict += torch.flip(
                            self.fine_predict(
                                torch.flip(crop_image, (2, 3, 4)),
                                self.config.FINE_MODEL_PATH,
                                self.config.CHECK_POINT_NAME,
                            ),
                            (2, 3, 4),
                        )
                        crop_image = predict
                        torch.cuda.empty_cache()
                    else:
                        crop_image = self.fine_predict(
                            crop_image,
                            self.config.FINE_MODEL_PATH,
                            self.config.CHECK_POINT_NAME,
                        )
                    crop_image = output_upsample(
                        crop_image, abdomen_cropped_image.shape
                    )
                    crop_image = F.softmax(crop_image, 1)

                    fine_mask = (
                        crop_image.argmax(1).squeeze(axis=0).numpy().astype(np.uint8)
                    )
                    if self.is_post_process:
                        lab_unique = np.unique(fine_mask)
                        fine_mask = to_one_hot(fine_mask)
                        fine_mask = extract_topk_largest_candidates(
                            fine_mask, lab_unique, 1
                        )
                    if self.train_type == "coarse-fine":
                        out_mask = np.zeros(raw_image_shape, np.uint8)
                        out_mask[
                            crop_fine_bbox[0][0] : crop_fine_bbox[0][1],
                            crop_fine_bbox[1][0] : crop_fine_bbox[1][1],
                            crop_fine_bbox[2][0] : crop_fine_bbox[2][1],
                        ] = fine_mask
                        if self.save_softmax:
                            # 需要调整
                            soft_weight_image = np.zeros(raw_image_shape, np.uint8)
                            soft_weight_image[
                                crop_fine_bbox[0][0] : crop_fine_bbox[0][1],
                                crop_fine_bbox[1][0] : crop_fine_bbox[1][1],
                                crop_fine_bbox[2][0] : crop_fine_bbox[2][1],
                            ] = crop_image

                    elif self.train_type == "fine":
                        out_mask = fine_mask
                        if self.save_softmax:
                            soft_weight_image = crop_image[0]

                    out_mask = np.ascontiguousarray(out_mask)
                    if self.is_nor_dir:
                        out_mask = change_axes_of_image(out_mask, image_direction)
                    sitk_image = resume_itk_info(out_mask, itk_info)
                    sitk.WriteImage(
                        sitk_image,
                        os.path.join(
                            self.output_path,
                            image_path.replace("_0000.nii.gz", ".nii.gz"),
                        ),
                        True,
                    )
                    if self.save_softmax:
                        np.save(
                            os.path.join(
                                self.output_path,
                                image_path.replace("_0000.nii.gz", "_softmax.npy"),
                            ),
                            soft_weight_image.half().cpu().numpy(),
                        )
                    print(f"{image_id} Done")

        torch.cuda.synchronize()
        t_end = time.time()
        average_time_usage = (t_end - t_start) * 1.0 / len(self.val_loader)
        print("Average time usage: {} s".format(average_time_usage))

    def fine_predict(self, input, model_path, check_point_name="final_checkpoint.pth"):
        fine_model_checkpoint = load_checkpoint(model_path, check_point_name)
        fine_model = build_fine_model(fine_model_checkpoint["config"], True).eval()
        fine_model.load_state_dict(
            {
                k.replace("module.", ""): v
                for k, v in fine_model_checkpoint["state_dict"].items()
            }
        )
        self._set_requires_grad(fine_model, False)
        fine_model = fine_model.cuda().half()
        input = to_cuda(input).half()
        if self.is_multi_task:
            out, _ = fine_model(input, False)
        else:
            out = fine_model(input, False)
        del fine_model
        return out.cpu().float()

    def coarse_predict(
        self, input, model_path, check_point_name="final_checkpoint.pth"
    ):
        coarse_model_checkpoint = load_checkpoint(model_path, check_point_name)
        coarse_model = build_coarse_model(
            coarse_model_checkpoint["config"], True
        ).eval()
        coarse_model.load_state_dict(
            {
                k.replace("module.", ""): v
                for k, v in coarse_model_checkpoint["state_dict"].items()
            }
        )
        self._set_requires_grad(coarse_model, False)
        coarse_model = coarse_model.cuda().half()
        input = to_cuda(input).half()
        out = coarse_model(input, False)
        return out.cpu().float()

    @staticmethod
    def _set_requires_grad(model, requires_grad=False):
        for param in model.parameters():
            param.requires_grad = requires_grad


class Validation(object):
    def __init__(self, config) -> None:
        self.config = config
        self.train_type = config.TRAINING_TYPE
        print("############## Train Type: {} ##############".format(self.train_type))
        self.is_TTA = self.config.VAL.TTA
        self.is_post_process = self.config.VAL.TEST_NETWORK_PARAMETER
        self.is_multi_task = self.config.TRAIN.MULTI_TASK.IS_OPEN
        self.output_path = self.config.VAL_OUTPUT_PATH
        self.version = self.config.VERSION
        os.makedirs(config.VAL_OUTPUT_PATH, exist_ok=True)
        self.coarse_size = self.config.DATASET.COARSE.SIZE
        self.fine_size = self.config.DATASET.FINE.SIZE
        self.extend_size = self.config.DATASET.EXTEND_SIZE
        self.is_post_process = self.config.VAL.IS_POST_PROCESS
        self.is_nor_dir = self.config.DATASET.IS_NORMALIZATION_DIRECTION
        self.is_with_dataloader = self.config.VAL.IS_WITH_DATALOADER
        self.is_crop = self.config.VAL.IS_CROP
        if self.is_with_dataloader:
            val_dataset = valp_dataset(config)
            self.val_loader = DataLoaderX(
                val_dataset,
                batch_size=1,
                num_workers=0,
                pin_memory=config.DATALOADER.PIN_MEMORY,
                shuffle=False,
            )
        else:
            self.val_loader = valp_dataset(config)
        cudnn.benchmark = True

    def run(self):
        torch.cuda.synchronize()
        t_start = time.time()
        self.FINE_DICE = AverageMeter()
        self.class_dc = [AverageMeter() for i in range(13)]
        tbar = tqdm(self.val_loader, ncols=150)
        with autocast():
            with torch.no_grad():
                for idx, image_dict in enumerate(tbar):
                    # for image_dict in self.val_loader:
                    image_dict = (
                        image_dict[0] if type(image_dict) is list else image_dict
                    )
                    if self.is_with_dataloader:
                        image_path = image_dict["image_path"][0]
                        image_id = image_dict["image_id"][0]
                        image_file = image_dict["image_file"][0]
                        raw_image = np.array(image_dict["raw_image"].squeeze(0))
                        raw_spacing = np.array(image_dict["raw_spacing"][0])
                        image_direction = np.array(image_dict["image_direction"][0])
                        gt = image_dict["seg"][0]
                        itk_info = {
                            "spacing": tuple(
                                [float(v) for v in image_dict["itk_info"]["spacing"]]
                            ),
                            "direction": tuple(
                                [float(v) for v in image_dict["itk_info"]["direction"]]
                            ),
                            "origin": tuple(
                                [float(v) for v in image_dict["itk_info"]["origin"]]
                            ),
                        }

                    else:
                        image_path = image_dict["image_path"]
                        image_id = image_dict["image_id"]
                        image_file = image_dict["image_file"]
                        raw_image = image_dict["raw_image"]
                        raw_spacing = image_dict["raw_spacing"]
                        image_direction = image_dict["image_direction"]
                        gt = image_dict["seg"]
                        itk_info = image_dict["itk_info"]

                    VAL_INFOS["image_file"].append(image_file)
                    print("current_image_id:{}".format(image_id))
                    if self.train_type == "coarse-fine":
                        coarse_image = (
                            torch.from_numpy(raw_image)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .float()
                        )
                        raw_image_shape = raw_image.shape
                        coarse_resize_factor = np.array(raw_image.shape) / np.array(
                            self.coarse_size
                        )
                        coarse_image = input_downsample(coarse_image, self.coarse_size)
                        coarse_image = self.coarse_predict(
                            coarse_image,
                            self.config.COARSE_MODEL_PATH,
                            self.config.COARSE_CHECK_POINT_NAME,
                        )
                        coarse_pre = F.softmax(coarse_image, 1)
                        coarse_pre = coarse_pre.cpu().float()
                        torch.cuda.empty_cache()
                        coarse_mask = (
                            coarse_pre.argmax(1)
                            .squeeze(axis=0)
                            .numpy()
                            .astype(np.uint8)
                        )
                        lab_unique = np.unique(coarse_mask)
                        coarse_mask = to_one_hot(coarse_mask)
                        coarse_mask = extract_topk_largest_candidates(
                            coarse_mask, lab_unique, 1
                        )
                        coarse_bbox = get_bbox_from_mask(coarse_mask)
                        raw_bbox = [
                            [
                                int(coarse_bbox[0][0] * coarse_resize_factor[0]),
                                int(coarse_bbox[0][1] * coarse_resize_factor[0]),
                            ],
                            [
                                int(coarse_bbox[1][0] * coarse_resize_factor[1]),
                                int(coarse_bbox[1][1] * coarse_resize_factor[1]),
                            ],
                            [
                                int(coarse_bbox[2][0] * coarse_resize_factor[2]),
                                int(coarse_bbox[2][1] * coarse_resize_factor[2]),
                            ],
                        ]
                        margin = [self.extend_size / raw_spacing[i] for i in range(3)]

                        crop_image, crop_fine_bbox = crop_image_according_to_bbox(
                            raw_image, raw_bbox, margin
                        )

                    elif self.train_type == "fine":
                        crop_image = raw_image

                    if self.is_crop and not self.train_type == "coarse-fine":
                        (
                            abdomen_cropped_image,
                            crop_start_slice,
                            crop_end_slice,
                            original_start,
                            original_end,
                        ) = crop_bbox_by_stand_spacing(crop_image, raw_spacing)
                    else:
                        abdomen_cropped_image = crop_image

                    crop_image = (
                        torch.from_numpy(abdomen_cropped_image)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    crop_image = input_downsample(crop_image, self.fine_size)

                    if self.is_TTA:
                        crop_image_pred = self.fine_predict(
                            crop_image,
                            self.config.FINE_MODEL_PATH,
                            self.config.CHECK_POINT_NAME,
                        )
                        predict = crop_image_pred
                        predict += torch.flip(
                            self.fine_predict(
                                torch.flip(crop_image, (2, 3, 4)),
                                self.config.FINE_MODEL_PATH,
                                self.config.CHECK_POINT_NAME,
                            ),
                            (2, 3, 4),
                        )
                        crop_image = predict / 2.0
                        torch.cuda.empty_cache()
                    else:
                        crop_image = self.fine_predict(
                            crop_image,
                            self.config.FINE_MODEL_PATH,
                            self.config.CHECK_POINT_NAME,
                        )
                    crop_image = output_upsample(
                        crop_image, abdomen_cropped_image.shape
                    )
                    crop_image = F.softmax(crop_image, 1)
                    fine_mask = (
                        crop_image.argmax(1).squeeze(axis=0).numpy().astype(np.uint8)
                    )
                    if self.is_crop:
                        fine_mask = crop_bbox_restore2raw(
                            fine_mask,
                            raw_image.shape,
                            crop_start_slice,
                            crop_end_slice,
                            original_start,
                            original_end,
                        ).astype(np.uint8)
                    if self.is_post_process:
                        lab_unique = np.unique(fine_mask)
                        fine_mask = to_one_hot(fine_mask)
                        fine_mask = extract_topk_largest_candidates(
                            fine_mask, lab_unique, 1
                        )
                    if self.train_type == "coarse-fine":
                        out_mask = np.zeros(raw_image_shape, np.uint8)
                        out_mask[
                            crop_fine_bbox[0][0] : crop_fine_bbox[0][1],
                            crop_fine_bbox[1][0] : crop_fine_bbox[1][1],
                            crop_fine_bbox[2][0] : crop_fine_bbox[2][1],
                        ] = fine_mask
                    elif self.train_type == "fine":
                        out_mask = fine_mask

                    out_mask = torch.tensor(out_mask)
                    global_avg_dice, classes_avg_dice = run_evaluation(out_mask, gt)
                    self.FINE_DICE.update(global_avg_dice)
                    class_dice = classes_avg_dice
                    global_dice = 0
                    for idx, dc in enumerate(class_dice):
                        self.class_dc[idx].update(dc)
                        if np.isnan(dc):
                            VAL_INFOS["class{}_dice".format(idx + 1)].append(0)
                            global_dice += 0
                        else:
                            VAL_INFOS["class{}_dice".format(idx + 1)].append(dc)
                            global_dice += dc

                        print(
                            "Class {} Dice: {}".format(idx, self.class_dc[idx].average)
                        )

                    tbar.set_description(
                        "VAL | DICE {} ".format(self.FINE_DICE.average)
                    )
                    VAL_INFOS["global_avg_dice"].append(global_dice / 13)
                    out_mask = np.ascontiguousarray(fine_mask)
                    os.makedirs(
                        os.path.join(self.output_path, "predcit_mask", self.version),
                        exist_ok=True,
                    )
                    if self.is_nor_dir:
                        out_mask = change_axes_of_image(out_mask, image_direction)
                    sitk_image = resume_itk_info(out_mask, itk_info)
                    sitk.WriteImage(
                        sitk_image,
                        os.path.join(
                            self.output_path,
                            "predcit_mask",
                            self.version,
                            image_path.replace("_0000.nii.gz", ".nii.gz"),
                        ),
                        True,
                    )

                    print(f"{image_id} Done")

        torch.cuda.synchronize()
        t_end = time.time()
        average_time_usage = (t_end - t_start) * 1.0 / len(self.val_loader)
        print("Average time usage: {} s".format(average_time_usage))
        print("Dice:{}".format(self.FINE_DICE.average))
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        pd.DataFrame(VAL_INFOS).to_csv(
            os.path.join(
                self.config.VAL_OUTPUT_PATH,
                "val_infos_"
                + os.path.basename(self.config.DATASET.VAL_IMAGE_PATH)
                + "_"
                + date_str
                + ".csv",
            )
        )

    def coarse_predict(
        self, input, model_path, check_point_name="final_checkpoint.pth"
    ):
        coarse_model_checkpoint = load_checkpoint(model_path, check_point_name)
        coarse_model = build_coarse_model(
            coarse_model_checkpoint["config"], True
        ).eval()
        coarse_model.load_state_dict(
            {
                k.replace("module.", ""): v
                for k, v in coarse_model_checkpoint["state_dict"].items()
            }
        )
        self._set_requires_grad(coarse_model, False)
        coarse_model = coarse_model.cuda().half()
        input = to_cuda(input).half()
        out = coarse_model(input, False)
        coarse_model = coarse_model.cpu()
        return out.cpu().float()

    def fine_predict(self, input, model_path, check_point_name="final_checkpoint.pth"):
        fine_model_checkpoint = load_checkpoint(model_path, check_point_name)
        fine_model = build_fine_model(fine_model_checkpoint["config"], True).eval()
        fine_model.load_state_dict(
            {
                k.replace("module.", ""): v
                for k, v in fine_model_checkpoint["state_dict"].items()
            }
        )
        self._set_requires_grad(fine_model, False)
        fine_model = fine_model.cuda().half()
        input = to_cuda(input).half()
        if self.is_multi_task:
            out, _ = fine_model(input, False)
        else:
            out = fine_model(input, False)
        fine_model = fine_model.cpu()
        return out.cpu().float()

    @staticmethod
    def _set_requires_grad(model, requires_grad=False):
        for param in model.parameters():
            param.requires_grad = requires_grad


class TorchSummary(object):
    def __init__(self, config) -> None:
        self.config = config
        self.train_type = config.TRAINING_TYPE
        print("############## Train Type: {} ##############".format(self.train_type))
        cudnn.benchmark = True

    def run(self):
        torch.cuda.synchronize()
        with autocast():
            with torch.no_grad():
                if self.train_type == "coarse-fine":
                    coarse_model_checkpoint = load_checkpoint(
                        self.config.FINE_MODEL_PATH, self.config.CHECK_POINT_NAME
                    )
                    model = build_coarse_model(
                        coarse_model_checkpoint["config"], True
                    ).eval()
                    fine_model_checkpoint = load_checkpoint(
                        self.config.FINE_MODEL_PATH, self.config.CHECK_POINT_NAME
                    )
                    fine_model = build_fine_model(
                        fine_model_checkpoint["config"], True
                    ).eval()

                    summary(
                        model,
                        input_size=(
                            self.config.DATALOADER.BATCH_SIZE,
                            1,
                            *self.config.DATASET.COARSE.SIZE,
                        ),
                    )
                    print("====================== Flops Coarse======================")
                    flops = FlopCountAnalysis(
                        model,
                        torch.rand(
                            self.config.DATALOADER.BATCH_SIZE,
                            1,
                            *self.config.DATASET.COARSE.SIZE,
                        )
                        .half()
                        .cuda(),
                    )
                    print("Flops:{}G".format(flops.total() / 1000 / 1000 / 1000))
                    print("====================== Flops Fine======================")
                    flops = FlopCountAnalysis(
                        fine_model,
                        torch.rand(
                            self.config.DATALOADER.BATCH_SIZE,
                            1,
                            *self.config.DATASET.FINE.SIZE,
                        )
                        .half()
                        .cuda(),
                    )
                    print("Flops:{}G".format(flops.total() / 1000 / 1000 / 1000))

                elif self.train_type == "fine":
                    fine_model_checkpoint = load_checkpoint(
                        self.config.FINE_MODEL_PATH, self.config.CHECK_POINT_NAME
                    )
                    model = build_fine_model(
                        fine_model_checkpoint["config"], True
                    ).eval()
                    data_size = self.config.DATASET.FINE.SIZE

                    summary(
                        model,
                        input_size=(self.config.DATALOADER.BATCH_SIZE, 1, *data_size),
                    )
                    print("====================== Flops ======================")
                    flops = FlopCountAnalysis(
                        model,
                        torch.rand(self.config.DATALOADER.BATCH_SIZE, 1, *data_size)
                        .half()
                        .cuda(),
                    )
                    print("Flops:{}G".format(flops.total() / 1000 / 1000 / 1000))
