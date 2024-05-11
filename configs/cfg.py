from yacs.config import CfgNode as CN

_C = CN()

_C.EXPERIMENT_RUN = 0

# Model
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda" 
_C.MODEL.BACKBONE = 'resnet50'
_C.MODEL.PROTOTYPE_DISTANCE_FUNCTION = 'l2'
_C.MODEL.PROTOTYPE_ACTIVATION_FUNCTION = 'log'
_C.MODEL.GENETIC_MODE = False


# Dataset
_C.DATASET = CN()
_C.DATASET.NAME = "NA"
_C.DATASET.NUM_CLASSES = 0

_C.DATASET.TRAIN_BATCH_SIZE = 80
_C.DATASET.TEST_BATCH_SIZE = 100
_C.DATASET.TRAIN_PUSH_BATCH_SIZE = 75


# Image Dataset
_C.DATASET.IMAGE = CN()

_C.DATASET.IMAGE.SIZE = 0
_C.DATASET.IMAGE.PROTOTYPE_SHAPE = (0, 0, 0, 0) 

_C.DATASET.IMAGE.MODEL_PATH = "NA"
_C.DATASET.IMAGE.TRAIN_DIR = "NA"
_C.DATASET.IMAGE.TEST_DIR = "NA"
_C.DATASET.IMAGE.TRAIN_PUSH_DIR = "NA"
_C.DATASET.IMAGE.TRAIN_BATCH_SIZE = 0
_C.DATASET.IMAGE.TRANSFORM_MEAN = ()
_C.DATASET.IMAGE.TRANSFORM_STD = ()


# Genetic Dataset 
_C.DATASET.GENETIC = CN()
_C.DATASET.GENETIC.TAXONOMY_NAME = "NA"
_C.DATASET.GENETIC.ORDER_NAME = "NA"
_C.DATASET.GENETIC.SIZE = 0
_C.DATASET.GENETIC.PROTOTYPE_SHAPE = (0, 0, 0, 0)

_C.DATASET.GENETIC.TRANSFORM = 'onehot'

_C.DATASET.GENETIC.MODEL_PATH = "NA"
_C.DATASET.GENETIC.TRAIN_PATH = "NA"
_C.DATASET.GENETIC.VALIDATION_PATH = "NA"
_C.DATASET.GENETIC.TRAIN_PUSH_DIR = "NA"
_C.DATASET.GENETIC.FIX_PROTOTYPES = True


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

_C.OPTIM.NUM_TRAIN_EPOCHS = 100
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
_C.OUTPUT.NO_SAVE = False
_C.OUTPUT.PREPROCESS_INPUT_FUNCTION = None


def get_cfg_defaults(): 
    return _C.clone()
