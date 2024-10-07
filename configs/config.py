import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# Base settings
# -----------------------------------------------------------------------------
_C.BASE = [""]

_C.DIS = False
_C.WORLD_SIZE = 1
_C.SEED = 1234
_C.AMP = True
_C.EXPERIMENT_ID = ""
_C.SAVE_DIR = "checkpoint_pth"
_C.VAL_OUTPUT_PATH = "TPH/val_output"
_C.COARSE_MODEL_PATH = ""
_C.FINE_MODEL_PATH = ""
_C.COARSE_CHECK_POINT_NAME = "final_checkpoint.pth"
_C.CHECK_POINT_NAME = "final_checkpoint.pth"
_C.VERSION = "v1"
_C.RESUME = False
_C.FINETUNE = False
_C.TRAINING_TYPE = "fine"  # include {'fine', 'coarse-fine', 'coarse'}

# -----------------------------------------------------------------------------
# Wandb settings
# -----------------------------------------------------------------------------
_C.WANDB = CN()
_C.WANDB.COARSE_PROJECT = "FLARE2024_COARSE"
_C.WANDB.FINE_PROJECT = "FLARE2024_Fine"
_C.WANDB.TAG = "Demo"
_C.WANDB.MODE = "offline"


# -----------------------------------------------------------------------------
# Real MR Data Preprocess(pseduo label)
# -----------------------------------------------------------------------------
_C.MR_DATA_PREPROCESS = CN()
#  preprocess
_C.MR_DATA_PREPROCESS.STAGE = 1
_C.MR_DATA_PREPROCESS.ROOT_PATH = ""
_C.MR_DATA_PREPROCESS.MR_RAW_DATA_PATH = ""
_C.MR_DATA_PREPROCESS.PSUEDO_LABEL_PATH = ""  # inference psuedo label path
_C.MR_DATA_PREPROCESS.TEMP_MR_RAW_DATA_PATH = ""  # filter C+Delay data
_C.MR_DATA_PREPROCESS.DATA_PAIR_PATH = ""  # using symlink and rename to save Hard Disk Space. it's recoder source filename and symlink name pair. it's json file path.
_C.MR_DATA_PREPROCESS.TEMP_PREPROCESSED_PSUEDO_LABEL_PATH = ""  # test

# filter psuedo label
_C.MR_DATA_PREPROCESS.CT_GT_PATH = ""  # save filtered case infos . it's csv file path.
_C.MR_DATA_PREPROCESS.OUTPUT_EACH_CT_LABELS_NUM_STATIC_PATH = ""
_C.MR_DATA_PREPROCESS.FILTER_CASE_SAVE_PATH = (
    ""  # save filtered case infos . it's csv file path.
)

# preprocess
_C.MR_DATA_PREPROCESS.BAD_CASE_PATH = (
    ""  # preprocessed fail case file. it's json file path.
)

# regis
_C.MR_DATA_PREPROCESS.REGIS = CN()
_C.MR_DATA_PREPROCESS.REGIS.SOURCE_DATA_PATH = ""
_C.MR_DATA_PREPROCESS.REGIS.TARGET_DATA_PATH = ""
_C.MR_DATA_PREPROCESS.REGIS.OUTPUT_PATH = ""

_C.MR_DATA_PREPROCESS.OUTPUT_MR_RAW_DATA_PATH = ""

_C.MR_DATA_PREPROCESS.IS_LLD_REGIS_DATA = False

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.WITH_VAL = False
_C.DATASET.BASE_DIR = ""
_C.DATASET.DATASET_NAME = "base"

_C.DATASET.TRAIN_UNLABELED_IMAGE_PATH = (
    ""  # base_id + TRAIN_UNLABELED_IMAGE_PATH = final_path
)
_C.DATASET.TRAIN_UNLABELED_MASK_PATH = ""
_C.DATASET.TRAIN_IMAGE_PATH = ""
_C.DATASET.TRAIN_MASK_PATH = ""
_C.DATASET.VAL_IMAGE_PATH = ""
_C.DATASET.VAL_MASK_PATH = ""
_C.DATASET.EXTEND_SIZE = 20
_C.DATASET.IS_NORMALIZATION_DIRECTION = True

_C.DATASET.COARSE = CN()
_C.DATASET.COARSE.SPLIT_PREPROCESS_PATH = []
_C.DATASET.COARSE.PROPRECESS_PATH = ""
_C.DATASET.COARSE.PROPRECESS_UL_PATH = ""
_C.DATASET.COARSE.NUM_EACH_EPOCH = 512
_C.DATASET.COARSE.SIZE = [64, 64, 64]
_C.DATASET.COARSE.PREPROCESS_SIZE = [64, 64, 64]
_C.DATASET.COARSE.LABEL_CLASSES = 2

_C.DATASET.FINE = CN()
_C.DATASET.FINE.SPLIT_PREPROCESS_PATH = []
_C.DATASET.FINE.PROPRECESS_PATH = ""
_C.DATASET.FINE.PROPRECESS_UL_PATH = ""
_C.DATASET.FINE.NUM_EACH_EPOCH = 512
_C.DATASET.FINE.SIZE = [96, 256, 256]  # for training
_C.DATASET.FINE.PREPROCESS_SIZE = [256, 256, 96]  # for preprocess

_C.DATASET.FINE.LABEL_CLASSES = 14

_C.DATASET.DA = CN()
_C.DATASET.DA.DO_2D_AUG = True
_C.DATASET.DA.DO_ELASTIC = True
_C.DATASET.DA.DO_SCALING = True
_C.DATASET.DA.DO_ROTATION = True
_C.DATASET.DA.RANDOM_CROP = False
_C.DATASET.DA.DO_GAMMA = True
_C.DATASET.DA.DO_MIRROR = False
_C.DATASET.DA.DO_ADDITIVE_BRIGHTNESS = True

# whether to crop during preprocessing
_C.DATASET.IS_ABDOMEN_CROP = False
# whether to change the spacing to 1,1,1 during the test phase
_C.DATASET.VAL_CHANGE_SPACING = False


# -----------------------------------------------------------------------------
# Dataloader settings
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 1
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEEP_SUPERVISION = True

_C.MODEL.COARSE = CN()
_C.MODEL.COARSE.TYPE = "phtrans"
_C.MODEL.COARSE.BASE_NUM_FEATURES = 16
_C.MODEL.COARSE.NUM_ONLY_CONV_STAGE = 2
_C.MODEL.COARSE.NUM_CONV_PER_STAGE = 2
_C.MODEL.COARSE.FEAT_MAP_MUL_ON_DOWNSCALE = 2
_C.MODEL.COARSE.POOL_OP_KERNEL_SIZES = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
_C.MODEL.COARSE.CONV_KERNEL_SIZES = [
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
]
_C.MODEL.COARSE.DROPOUT_P = 0.1

_C.MODEL.COARSE.MAX_NUM_FEATURES = 200
_C.MODEL.COARSE.DEPTHS = [2, 2, 2, 2]
_C.MODEL.COARSE.NUM_HEADS = [4, 4, 4, 4]
_C.MODEL.COARSE.WINDOW_SIZE = [4, 4, 4]
_C.MODEL.COARSE.MLP_RATIO = 1.0
_C.MODEL.COARSE.QKV_BIAS = True
_C.MODEL.COARSE.QK_SCALE = None
_C.MODEL.COARSE.DROP_RATE = 0.0
_C.MODEL.COARSE.DROP_PATH_RATE = 0.1

_C.MODEL.FINE = CN()
_C.MODEL.FINE.TYPE = "phtrans"
_C.MODEL.FINE.BASE_NUM_FEATURES = 24
_C.MODEL.FINE.NUM_ONLY_CONV_STAGE = 2
_C.MODEL.FINE.NUM_CONV_PER_STAGE = 2
_C.MODEL.FINE.FEAT_MAP_MUL_ON_DOWNSCALE = 2
_C.MODEL.FINE.POOL_OP_KERNEL_SIZES = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
_C.MODEL.FINE.CONV_KERNEL_SIZES = [
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
]
_C.MODEL.FINE.DROPOUT_P = 0.1

_C.MODEL.FINE.MAX_NUM_FEATURES = 360
_C.MODEL.FINE.DEPTHS = [2, 2, 2, 2]
_C.MODEL.FINE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.FINE.WINDOW_SIZE = [3, 4, 4]
_C.MODEL.FINE.MLP_RATIO = 4.0
_C.MODEL.FINE.QKV_BIAS = True
_C.MODEL.FINE.QK_SCALE = None
_C.MODEL.FINE.DROP_RATE = 0.0
_C.MODEL.FINE.DROP_PATH_RATE = 0.2

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.DO_BACKPROP = True
_C.TRAIN.VAL_NUM_EPOCHS = 1
_C.TRAIN.SAVE_PERIOD = 25

_C.TRAIN.EPOCHS = 500
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# multi task (add classfication)
_C.TRAIN.MULTI_TASK = CN()
_C.TRAIN.MULTI_TASK.IS_OPEN = False
_C.TRAIN.MULTI_TASK.CLASS_NUM = 2
# new loss filter import voxel
_C.TRAIN.SELECT_IMPORT_VOXEL = CN()
_C.TRAIN.SELECT_IMPORT_VOXEL.IS_OPEN = False
_C.TRAIN.SELECT_IMPORT_VOXEL.DROP_VOXEL = False
_C.TRAIN.SELECT_IMPORT_VOXEL.UPDATE_LOSS_WEIGHT = False
_C.TRAIN.SELECT_IMPORT_VOXEL.IGNORE_LABEL = None

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"

# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9


#
#  new loss: ignore edge
_C.LOSS = CN()
_C.LOSS.IGNORE_EDGE = False
_C.LOSS.IGNORE_EDGE_KERNEL_SIZE = 5
_C.LOSS.FLARE24_CHANGE_WEIGHT = False

# -----------------------------------------------------------------------------
# Validation settings
# -----------------------------------------------------------------------------
_C.VAL = CN()
_C.VAL.EVAL_OUTPUT_RESULT = "."
_C.VAL.IS_POST_PROCESS = True
_C.VAL.IS_WITH_DATALOADER = True
_C.VAL.IS_CROP = False
_C.VAL.TEST_NETWORK_PARAMETER = False
_C.VAL.TTA = False


# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
# # Whether to overwrite the predicted mask during the inference phase, if True it will be rerun each time, if False, the i that has been run will not be rerun
_C.INFERENCE.IS_OVERWRITE_PREDICT_MASK = (
    True
)
_C.INFERENCE.SAVE_SOFTMAX = False  # Whether to save the weight matrix, in order to new loss


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if args.cfg is not None:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    if args.batch_size:
        config.DATALOADER.BATCH_SIZE = args.batch_size
    if args.tag:
        config.WANDB.TAG = args.tag
    if args.wandb_mode == "online":
        config.WANDB.MODE = args.wandb_mode
    if args.world_size:
        config.WORLD_SIZE = args.world_size
    if args.with_distributed:
        config.DIS = True
    config.freeze()


def update_val_config(config, args):
    if args.cfg is not None:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    config.freeze()


def get_config(args=None):
    config = _C.clone()
    update_config(config, args)

    return config


def get_config_no_args():
    config = _C.clone()

    return config


def get_val_config(args=None):
    config = _C.clone()
    update_val_config(config, args)

    return config
