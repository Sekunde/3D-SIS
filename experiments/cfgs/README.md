
### Configuration File


```yaml
#----------- Filelists--------------
# filelist of training chunks
TRAIN_FILELIST: experiments/filelists/ScanNet/v2/train.txt
# filelist of validation chunks
VAL_FILELIST: experiments/filelists/ScanNet/v2/val_chunk.txt
# filelist of subset of training chunks, used to check for overfitting degree
TRAINVAL_FILELIST: experiments/filelists/ScanNet/v2/trainval.txt
# filelist of test scenes
TEST_FILELIST: experiments/filelists/ScanNet/v2/test.txt

#----------- Result folder -----------
# where to store the validation results (chunks)
VAL_SAVE_DIR: ../results/ScanNet/benchmark/val
# where to store the test results (scenes)
TEST_SAVE_DIR: ../results/ScanNet/benchmark/test

# ----------- Backbone -------------
# load checkpoint for backbone
LOAD_BACKBONE: True
# use the backbone
USE_BACKBONE: True
# fix the backbone weights or not
FIX_BACKBONE: False
# load checkpoint for RPN
LOAD_RPN: True
# use RPN, if false, use groundtruth bbox
USE_RPN: True
# fix weights of RPN network
FIX_RPN: False
# load checkpoint for classification network
LOAD_CLASS: True
# use classification network, if not, use the groundtruth class labels
USE_CLASS: True
# fix the classification network weights or not
FIX_CLASS: False
# use the second backbone for mask or not
USE_MASK: True

#-------------Enet---------------------
# use color images or not
USE_IMAGES: True
# where is the image folder
BASE_IMAGE_PATH: '/mnt/local_datasets/ScanNet/frames_square'
# where is the enet pretrained network
PRETRAINED_ENET_PATH: /mnt/local_datasets/ScanNet/scannetv2_enet.pth

#------------Extra--------------------
# in the training, every 2 hours, validate inference on chunks (from VAL_FILELIST)
VAL_TIME: 2.0
# reduce learning rate after 200k and 300k steps
STEPSIZE: [200000, 300000]
# keep the latest N checkpoint, 0 means keep all
SNAPSHOT_KEPT: 0
# backbone defined in /lib/net/backbone.py
NET: ScanNet_Backbone
# second backbone defined in /lib/net/backbone.py
MASK_BACKBONE: MaskBackbone
```
