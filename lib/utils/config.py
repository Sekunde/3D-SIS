from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import yaml
import ipdb
from easydict import EasyDict as edict

__C = edict()
cfg = __C


#---------------------------------------------
# Optimizer
#---------------------------------------------
# Initial learning rate
__C.LEARNING_RATE = 0.001
# Momentum
__C.MOMENTUM = 0.9
# Weight decay, for regularization
__C.WEIGHT_DECAY = 0.0005
# Factor for reducing the learning rate
__C.GAMMA = 0.1
# Step size for reducing the learning rate, currently only support one step
__C.STEPSIZE = [30000]
# Iteration intervals for showing the loss during training, on command line interface
__C.DISPLAY = 10
# Whether to double the learning rate for bias
__C.DOUBLE_BIAS = True
# Whether to initialize the weights with truncated normal distribution
__C.INIT_TRUNCATED = True
# Whether to have weight decay on bias as well
__C.BIAS_DECAY = False

#---------------------------------------------
# RPN
#---------------------------------------------
# IOU >= thresh: positive example RoI
__C.RPN_POSITIVE_OVERLAP = 0.6
# IOU < thresh: negative example RoI
__C.RPN_NEGATIVE_OVERLAP = 0.2
# Max number of foreground examples
__C.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.RPN_BATCHSIZE = 256
# Anchor border
__C.ALLOW_BORDER = 0
# ONLY FIRST TIME GENERATE ANCHOR FOR TRAINING
__C.FIRST_TIME_ANCHORS = True
# How many channels in RPN
__C.RPN_CHANNELS = 256

#--------------------------------------------
# NMS
#--------------------------------------------
__C.TRAIN = edict()
# NMS threshold used on RPN proposals in train
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

__C.TEST = edict()
## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.35
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

#-----------------------------------------------
# Classification
#-----------------------------------------------
# Minibatch size (number of regions of interest [ROIs])
__C.CLASS_BATCH_SIZE = 128
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.FG_FRACTION = 0.25
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.FG_THRESH = 0.3
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI)) for classification
__C.BG_THRESH = 0.2
__C.BG_THRESH_LO = 0.2
# Whether to add ground truth boxes to the pool when sampling regions
__C.USE_GT = True
__C.NORMALIZE_WEIGHTS = 1.0
# Pool size
__C.CLASS_POOLING_SIZE = 2

#---------------------------------------------
# Mask
#---------------------------------------------
# Overlap threshold for selecting bbox as mask candidate
__C.MASK_FG_THRESH = 0.5
# If use image in mask head
__C.MASK_USE_IMAGES = False
# If only use image in mask head
__C.MASK_ONLY_IMAGES = False

#----------------------------------------------
# Checkpoints
#----------------------------------------------
__C.VAL_TIME = 2.0
__C.SNAPSHOT_KEPT = 3
__C.VAL_NUM = 20
__C.VAL_SAVE_DIR = ''
__C.TEST_SAVE_DIR = ''


#----------------------------------
# Evaluation
#----------------------------------
__C.MAP_THRESH = 0.25
__C.ROI_THRESH = 0.9
__C.CLASS_THRESH = 0.9
__C.MASK_THRESH = 0.5

__C.MODE = ''
__C.MAX_IMAGE = 400
__C.MAX_VOLUME = 2000000

#--------------------------------
# Dataloader
#--------------------------------
__C.NUM_CLASSES = 0
__C.BATCH_SIZE = 1
__C.TRAIN_FILELIST = ''
__C.VAL_FILELIST = ''
__C.TEST_FILELIST = ''
__C.TRAINVAL_FILELIST = ''
# the box/mask that are more than this threshold will be kept, 0 means keep all.
__C.KEEP_THRESH = 0.0
__C.LABEL_MAP = 'datagen/fileLists/nyu40labels.csv'
__C.VOXEL_SIZE = 0.09375
__C.TRUNCATED = 3.0
__C.FLIP_TSDF = False
__C.LOG_TSDF = False

#-----------------------------
# Anchors
#----------------------------
__C.NUM_ANCHORS_LEVEL1 = 9
__C.NUM_ANCHORS_LEVEL2 = 9
__C.NUM_ANCHORS_LEVEL3 = 9
__C.ANCHORS_TYPE_LEVEL1 = 'suncg'
__C.ANCHORS_TYPE_LEVEL2 = 'suncg'
__C.ANCHORS_TYPE_LEVEL3 = 'suncg'
__C.FILTER_ANCHOR_LEVEL1 = ''
__C.FILTER_ANCHOR_LEVEL2 = ''
__C.FILTER_ANCHOR_LEVEL3 = ''

#----------------------------
# Nets
#---------------------------
# backbone
__C.LOAD_BACKBONE = False
__C.USE_BACKBONE = False
__C.FIX_BACKBONE = False

#RPN
__C.LOAD_RPN = False
__C.USE_RPN = False
__C.FIX_RPN = False

#Classification
__C.LOAD_CLASS = False
__C.USE_CLASS = False
__C.FIX_CLASS = False

# mask
__C.USE_MASK = True

#Enet
__C.FIX_ENET= True
__C.NET = 'overfitting_net'
__C.MASK_BACKBONE = ''

#---------------------------------
# Color Pipeline
#---------------------------------
__C.USE_IMAGES = False
__C.ONLY_IMAGES = False
__C.USE_IMAGES_GT = True
__C.NUM_2D_CLASSES = 41
__C.NUM_IMAGES = 1
__C.RANDOM_NUM_IMAGES = False
__C.BASE_IMAGE_PATH = '/mnt/local_datasets/SUNCG/suncg_frames'
__C.PRETRAINED_ENET_PATH = ''
__C.IMAGE_SHAPE = [328, 256]
__C.PROJ_DEPTH_MIN = 0.1
__C.PROJ_DEPTH_MAX = 4.0

#----------------------------
# SUNCG
#---------------------------
#__C.IMAGE_TYPE = 'label'
#__C.IMAGE_EXT = '.png'
#__C.DEPTH_SHAPE = [328, 256]
#__C.NUM_IMAGE_CHANNELS = 1
#__C.INTRINSIC = [[284.0561, 0, 163.5, 0],
#                 [0, 295.60319, 127.5, 0],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 1]]

__C.IMAGE_TYPE = 'color2'
__C.IMAGE_EXT = '.jpg'
__C.DEPTH_SHAPE = [41, 32]
__C.NUM_IMAGE_CHANNELS = 128
__C.INTRINSIC = [[35.5070229, 0, 20, 0],
                 [0, 36.9504013, 15.5, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]

__C.COLOR_MEAN = [0.47083, 0.44685, 0.40733]
__C.COLOR_STD = [0.27861, 0.27409, 0.28844]
#--------------------------------
# ScanNet
#--------------------------------
#__C.IMAGE_TYPE = 'label-eroded'
#__C.IMAGE_EXT = '.png'
#__C.DEPTH_SHAPE = [328, 256]
#__C.NUM_IMAGE_CHANNELS = 1
#__C.INTRINSIC = [[296.15868, 0, 163.5, 0],
#                 [0, 308.197656, 127.5, 0],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 1]]

#__C.IMAGE_TYPE = 'color'
#__C.IMAGE_EXT = '.jpg'
#__C.DEPTH_SHAPE = [41, 32]
#__C.NUM_IMAGE_CHANNELS = 128
#__C.INTRINSIC = [[37.01983, 0, 20, 0],
#                 [0, 38.52470, 15.5, 0],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 1]]

#__C.COLOR_MEAN = [0.496342, 0.466664, 0.440796]
#__C.COLOR_STD = [0.277856, 0.28623, 0.291129]

#---------------------------------
# NYUv2
#----------------------------------
__C.NYUV2_FINETUNE = False


def _merge_a_into_b(a, b):
    """
    merge the a into b, if the items already in a, the same item in b will be clobbered.

    :param a:
    :param b:
    :return:
    """

    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
           if isinstance(b[k], np.ndarray):
               v = np.array(v, dtype=b[k].dtype)
           else:
               raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.

    :param filename:
    :return:
    """
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_to_file(filename):
    """

    :param filename:
    :return:
    """
    with open(filename, 'w') as f:
        yaml.dump(cfg, f)

