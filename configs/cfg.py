from yacs.config import CfgNode as CN
import os
from utils.helpers import makedir


_C = CN()

_C.EXPERIMENT_RUN = 0

# Model
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda" 
_C.MODEL.BACKBONE = "NA" 
_C.MODEL.PROTOTYPE_SHAPE = (0, 0, 0, 0) 
_C.MODEL.PROTOTYPE_ACTIVATION_FUNCTION = "" 
_C.MODEL.ADD_ON_LAYERS_TYPE = ""
_C.MODEL.USE_COSINE = False 

# Dataset
_C.DATASET = CN()
_C.DATASET.NAME = "NA"
_C.DATASET.NUM_CLASSES = 0
_C.DATASET.IMAGE_SIZE = 0
_C.DATASET.DATA_PATH = "NA"
_C.DATASET.TRAIN_DIR = "NA "
_C.DATASET.TEST_DIR = "NA"
_C.DATASET.TRAIN_PUSH_DIR = "NA"
_C.DATASET.TRAIN_BATCH_SIZE = 80
_C.DATASET.TEST_BATCH_SIZE = 100
_C.DATASET.TRAIN_PUSH_BATCH_SIZE = 75
_C.DATASET.TRANFORM_MEAN = ()
_C.DATASET.TRANFORM_STD = ()

_C.DATASET.BIOSCAN = CN() 
_C.DATASET.BIOSCAN.TAXONOMY_NAME = "NA" 
_C.DATASET.BIOSCAN.ORDER_NAME = "NA"
_C.DATASET.BIOSCAN.DIPTERA = "NA" 
_C.DATASET.BIOSCAN.CHOP_LENGTH = 0


# Training
_C.OPTIM = CN()

# Joint optimizer
_C.OPTIM.JOINT_OPTIMIZER_LAYERS = CN() 
_C.OPTIM.JOINT_OPTIMIZER_LAYERS.FEATURES = 1e-4 
_C.OPTIM.JOINT_OPTIMIZER_LAYERS.ADD_ON_LAYERS = 3e-3
_C.OPTIM.JOINT_OPTIMIZER_LAYERS.PROTOTYPE_VECTORS = 3e-3
_C.OPTIM.JOINT_OPTIMIZER_LAYERS.LR_STEP_SIZE = 5
_C.OPTIM.JOINT_OPTIMIZER_LAYERS.WEIGHT_DECAY = 1e-3

# Warm optimizer
_C.OPTIM.WARM_OPTIMIZER_LAYERS = CN()
_C.OPTIM.WARM_OPTIMIZER_LAYERS.ADD_ON_LAYERS = 3e-3
_C.OPTIM.WARM_OPTIMIZER_LAYERS.PROTOTYPE_VECTORS = 3e-3
_C.OPTIM.WARM_OPTIMIZER_LAYERS.WEIGHT_DECAY= 1e-3

# Last layer optimizer
_C.OPTIM.LAST_LAYER_OPTIMIZER_LAYERS = CN()
_C.OPTIM.LAST_LAYER_OPTIMIZER_LAYERS.LR = 1e-4

# Coefficients
_C.OPTIM.COEFS = CN()
_C.OPTIM.COEFS.CRS_ENT = 1 
_C.OPTIM.COEFS.CLST = 0.8 
_C.OPTIM.COEFS.SEP = -0.08 
_C.OPTIM.COEFS.L1 = 1e-4 

_C.OPTIM.NUM_TRAIN_EPOCHS = 1000
_C.OPTIM.NUM_WARM_EPOCHS = 5

_C.OPTIM.PUSH_START = 10
_C.OPTIM.PUSH_EPOCHS = [i for i in range(_C.OPTIM.NUM_TRAIN_EPOCHS) if i % 10 == 0]


# Output 
_C.OUTPUT = CN()
_C.OUTPUT.MODEL_DIR = "NA"
_C.OUTPUT.IMG_DIR = "NA" 
_C.OUTPUT.WEIGHT_MATRIX_FILENAME = "NA" 
_C.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX = "NA" 
_C.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX = "NA" 
_C.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX = "NA" 


def get_cfg_defaults(): 
    return _C.clone()



def update_cfg(cfg, args): 
    if cfg.MODEL.DEVICE == "cuda": 
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
        print(f"Using GPU : {os.environ['CUDA_VISIBLE_DEVICES']}")

    # first update model 
    cfg.MODEL.BACKBONE = args.backbone
    cfg.MODEL.PROTOTYPE_ACTIVATION_FUNCTION = "log" 
    cfg.MODEL.USE_COSINE = False 

    if args.dataset == "cub": 

        cfg.MODEL.PROTOTYPE_SHAPE = (2000, 128, 1, 1) 
        cfg.MODEL.ADD_ON_LAYERS_TYPE = "regular"
        
        cfg.DATASET.NAME = "cub"
        cfg.DATASET.NUM_CLASSES = 200 
        cfg.DATASET.IMAGE_SIZE = 224
        cfg.DATASET.DATA_PATH = os.path.join("data", "CUB_200_2011", "cub200_cropped")
        cfg.DATASET.TRAIN_DIR = os.path.join(cfg.DATASET.DATA_PATH, "train_cropped_augmented")
        cfg.DATASET.TEST_DIR = os.path.join(cfg.DATASET.DATA_PATH, "test_cropped")
        cfg.DATASET.TRAIN_PUSH_DIR = os.path.join(cfg.DATASET.DATA_PATH, "train_cropped")
        cfg.DATASET.TRAIN_BATCH_SIZE = 80
        cfg.DATASET.TRANSFORM_MEAN = (0.485, 0.456, 0.406) 
        cfg.DATASET.TRANSFORM_STD = (0.229, 0.224, 0.225)

    elif args.dataset == "bioscan":
        cfg.MODEL.PROTOTYPE_SHAPE = (40 * 40, 128, 1, 1) 
        cfg.MODEL.ADD_ON_LAYERS_TYPE = None 
        
        cfg.DATASET.NAME = "bioscan"
        cfg.DATASET.BIOSCAN.TAXONOMY_NAME = "family"
        cfg.DATASET.BIOSCAN.ORDER_NAME = "Diptera"
        cfg.DATASET.BIOSCAN.CHOP_LENGTH = 720 
        cfg.DATASET.NUM_CLASSES = 40
        cfg.DATASET.IMAGE_SIZE = (4, 1, cfg.DATASET.BIOSCAN.CHOP_LENGTH)
        # cfg.DATASET.DATA_PATH = os.path.join("data", "CUB_200_2011", "cub200_cropped")
        # cfg.DATASET.TRAIN_DIR = os.path.join(cfg.DATASET.DATA_PATH, "train_cropped_augmented")
        # cfg.DATASET.TEST_DIR = os.path.join(cfg.DATASET.DATA_PATH, "test_cropped")
        # cfg.DATASET.TRAIN_PUSH_DIR = os.path.join(cfg.DATASET.DATA_PATH, "train_cropped")
        cfg.DATASET.TRAIN_BATCH_SIZE = 80

    else: 
        raise Exception("Invalid Dataset")

    model_dir = os.path.join("saved_models", f"{cfg.DATASET.NAME}_ppnet", str(cfg.EXPERIMENT_RUN).zfill(3))
    while os.path.isdir(model_dir): 
        cfg.EXPERIMENT_RUN += 1
        model_dir = os.path.join("saved_models", f"{cfg.DATASET.NAME}_ppnet", str(cfg.EXPERIMENT_RUN).zfill(3))
    makedir(model_dir)

    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    print(f"Model Dir: {model_dir}")

    cfg.OUTPUT.MODEL_DIR = model_dir
    cfg.OUTPUT.IMG_DIR = img_dir 
    cfg.OUTPUT.WEIGHT_MATRIX_FILENAME = weight_matrix_filename 
    cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX = prototype_img_filename_prefix 
    cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX = prototype_self_act_filename_prefix 
    cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX = proto_bound_boxes_filename_prefix 
