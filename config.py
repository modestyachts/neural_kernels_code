# my_project/config.py
from yacs.config import CfgNode as CN
import json
import hashlib

_C = CN()

_C.TRAIN_EVAL_ID = "" 
_C.TEST_EVAL_ID = "" 

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 0
_C.SYSTEM.BATCH_SIZE = 16
_C.SYSTEM.USE_AWS_BATCH = False
_C.SYSTEM.AWS_BATCH_CORES = 0
_C.SYSTEM.CACHE_PATH = "tc_cache"
_C.SYSTEM.FLOAT_32= True
_C.SYSTEM.VERBOSE = False
_C.SYSTEM.USE_TQDM = True
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 1

_C.DATASET = CN()
_C.DATASET.NAME = "cifar-10-zca"
_C.DATASET.RAW_TRAIN_NAME = "cifar-10-train"
_C.DATASET.RAW_TEST_NAME = "cifar-10-test"
_C.DATASET.TRAIN_SUBSET_START = 0
_C.DATASET.TRAIN_FULL_SIZE = 0
_C.DATASET.TEST_FULL_SIZE = 0
_C.DATASET.TRAIN_SUBSET = 32
_C.DATASET.TEST_SUBSET_START = 0
_C.DATASET.TEST_SUBSET = 64
_C.DATASET.RAND_SEED =  0
_C.DATASET.NUM_AUGMENTATIONS = 0
_C.DATASET.AUGMENTATIONS = [("FlipLR", 1)]
_C.DATASET.RAND_AUGMENT_AUGS = ["FlipLR", "Solarize", "Color", "Brightness", "Contrast", "Sharpness", "Posterize", "Equalize", "Identity"]
_C.DATASET.RAND_AUGMENT_N_MAX = 2
_C.DATASET.RAND_AUGMENT_RAND_N = False
_C.DATASET.RAND_AUGMENT_REPLACE = False
_C.DATASET.AUGMENTATION_PROB = 0.5
_C.DATASET.ZCA  = True
_C.DATASET.ZCA_EXTRA_AUGMENT  = False
_C.DATASET.STANDARD_PREPROCESS = True
_C.DATASET.NUM_ZCA_EXTRA_AUGMENT = 5
_C.DATASET.ZCA_BIAS = 1e-5
_C.DATASET.CORNERS = False
_C.DATASET.ZCA_FULL = False 

_C.DATASET.SHARD_X_IDX = 0
_C.DATASET.SHARD_Y_IDX = 0
_C.DATASET.SHARD_X_SIZE = 0
_C.DATASET.SHARD_Y_SIZE = 0
_C.DATASET.KERNEL_UUID = "DEADBEEF"

# Hyperparameters for line search
_C.LINE_SEARCH = True
_C.ALPHA = (1 + 5 ** 0.5) / 2 # The golden ratio
_C.BETA = 2 / (1 + 5 ** 0.5)
_C.EPSILON = 0.002

# if RANDOM is false then the subset will be first K examples
_C.DATASET.RANDOM_TRAIN_SUBSET = False
_C.DATASET.RANDOM_TEST_SUBSET = False

_C.KERNEL = CN()

_C.KERNEL.COATESNG = CN()
_C.KERNEL.COATESNG.ON = False
_C.KERNEL.COATESNG.PATCH_SIZE = 5
_C.KERNEL.COATESNG.POOL_SIZE = 1
_C.KERNEL.COATESNG.BIAS = 1
_C.KERNEL.COATESNG.POOL_STRIDE = 1
_C.KERNEL.COATESNG.NUM_FILTERS = 1024
_C.KERNEL.COATESNG.FILTER_SCALE = 1e-4
_C.KERNEL.COATESNG.FILTER_BATCH_SIZE = 2048
_C.KERNEL.COATESNG.PATCH_DISTRIBUTION = "empirical"
_C.KERNEL.COATESNG.NORMALIZE_PATCHES = False
_C.KERNEL.COATESNG.FLIP_PATCHES = True
_C.KERNEL.COATESNG.SEED = 0

_C.KERNEL.ARCH = [
  {"layer": "conv3zp"},
  {"layer": "relu"},
  {"layer": "conv3zp"},
  {"layer": "relu"},
  {"layer": "conv3zp"},
  {"layer": "relu"},
  {"layer": "pool2"},
  {"layer": "conv3zp"},
  {"layer": "relu"},
  {"layer": "pool2"},
  {"layer": "conv3zp"},
  {"layer": "relu"},
  {"layer": "pool8"}
]

_C.KERNEL.ARCH_DEFAULTS = [("relu", {"store_norm": False}), ("relu_project", {"store_norm": False}), ("quartic", {"store_norm": False}), ("quartic_project", {"store_norm": False}), ("pool2", {"store_norm": False}), ("pool8", {"store_norm": False}), ("quadratic", {"store_norm": False}), ("quadratic_project", {"store_norm": False}), ("exponential", {"store_norm": False}), ("exponential_shifted", {"store_norm": False, "gamma": 1.0})]

_C.KERNEL.CACHE_DIR = "./.kernel_cache"


_C.SOLVE = CN()
_C.SOLVE.LOO_TILT  = False
# Regularization
_C.SOLVE.REGS = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

def hash_arch(x):
   return hashlib.md5(json.dumps(x, indent=True, sort_keys=True).encode()).hexdigest()

short_name_layer_map =\
{
  'input': 'inp',
  'conv3': 'c3',
  'conv5': 'c5',
  'conv3zp': 'cz3',
  'conv3zpinorm': 'cz3',
  'conv5zp': 'cz5',
  'pool2': 'p2',
  'pool7': 'p7',
  'pool8': 'p8',
  'pool4': 'p4',
  'relu': 'r',
  'relu_project': 'rp',
  'quartic': 'qrtc',
  'quartic_project': 'qrtcp',
  'quadratic': 'qdrt',
  'quadratic_project': 'qdrtp',
  'exponential': 'expo',
  'exponential_shifted': 'expshft',
  'exponential_shifted_project': 'expshft_project',
  'group_norm_32': 'gn32',
  'group_norm_16': 'gn16',
  'group_norm_8': 'gn8',
  'group_norm_4': 'gn4',
  'conv3zp_all_chan': 'cza3'
}

def arch_short_name(x):
  out_str = ""
  for elem in x:
     out_str += short_name_layer_map[elem['layer']]
     out_str += "_"
  return out_str[:-1]
