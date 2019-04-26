import math
import ipdb

import torch
import torch.nn as nn

import torch.nn.functional as F
from lib.nets.network import Network
from lib.utils.config import cfg
from lib.utils.timer import Timer
from torch.autograd import Variable
from lib.layer_utils.coord_conv3d_random import CoordConv3d

#-------------------------------------------------------------
# helper function
#------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(planes, inplanes, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += residual
        out = self.relu(out)
        return out


class Base_Backbone(Network):
    def __init__(self, obbox=True):
        super(Base_Backbone, self).__init__()
        self._feat_stride = [4, 4, 4]
        self._fc7_channels = 128 # number of depth of last layer of self.classifier

        if cfg.ONLY_IMAGES:
            self._net_conv_level1_channels = 128 # number of channels of last layer of self.feature
            self._net_conv_level2_channels = 128 # number of channels of last layer of self.feature
            self._net_conv_level3_channels = 128 # number of channels of last layer of self.feature
        else:
            self._net_conv_level1_channels = 128 # number of channels of last layer of self.feature
            self._net_conv_level2_channels = 128 # number of channels of last layer of self.feature
            self._net_conv_level3_channels = 128 # number of channels of last layer of self.feature

    # feature extractor
    def _init_backbone_classifier(self):
        #-------------------------
        # Geometry pipeline
        #-------------------------
        if not cfg.ONLY_IMAGES:
            self.geometry1 = nn.Sequential()

        #-------------------------
        # Color pipeline
        #-------------------------
        if cfg.USE_IMAGES:
            self.color = nn.Sequential()

        #------------------------
        # Combine pipeline
        #------------------------
        # only image
        if cfg.USE_IMAGES and cfg.ONLY_IMAGES:
            input_channels = 64
        # image + geometry
        elif cfg.USE_IMAGES:
            input_channels = 64 + 64
        # only geometry
        else:
            input_channels = 128
        self.geometry2 = nn.Sequential()

        #-------------------------
        # classfier
        #--------------------------
        self.classifier = nn.Sequential()

    # after fixing the roi size, do the classification
    def _classifier(self, pool5):
        #flat everything
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.classifier(pool5_flat)
        return fc7

    def _backbone(self):
        if cfg.USE_IMAGES and cfg.ONLY_IMAGES:
            net_conv_level1= self.color(self._imageft)
            net_conv_level2 = self.geometry2(net_conv_level1)
        elif cfg.USE_IMAGES:
            net_conv_color = self.color(self._imageft)
            net_conv_geometry = self.geometry1(self._scene)
            #net_conv_color = torch.zeros_like(net_conv_color).cuda()
            #net_conv_geometry = torch.zeros_like(net_conv_geometry).cuda()
            net_conv_level1 = torch.cat([net_conv_color, net_conv_geometry], 1) # NOTE: this assumes batch mode for the concat..
            net_conv_level2 = self.geometry2(net_conv_level1)
        else:
            net_conv_level1= self.geometry1(self._scene)
            net_conv_level2 = self.geometry2(net_conv_level1)

        return net_conv_level1, net_conv_level2, None

#--------------------------------------------------------------------------------------
# First backbone
#-------------------------------------------------------------------------------------
class SUNCG_Backbone(Base_Backbone):
    # feature extractor
    def _init_backbone_classifier(self):
        #-------------------------
        # Geometry pipeline
        #-------------------------
        if not cfg.ONLY_IMAGES or not cfg.USE_IMAGES:
            self.geometry1 = nn.Sequential(
                    nn.Conv3d(2, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                    nn.ReLU(True),
                    Bottleneck(64, 32, stride=1),
                    nn.Conv3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                    nn.ReLU(True),
                    Bottleneck(64, 32, stride=1))

        #-------------------------
        # Color pipeline
        #-------------------------
        if cfg.USE_IMAGES:
            self.color = nn.Sequential(
                    nn.Conv3d(cfg.NUM_IMAGE_CHANNELS, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                    nn.ReLU(True),
                    Bottleneck(64, 32, stride=1),
                    nn.Conv3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                    nn.ReLU(True),
                    Bottleneck(64, 32, stride=1))

        #-------------------------
        # Combine pipeline
        #-------------------------
        if cfg.USE_IMAGES and cfg.ONLY_IMAGES:
            input_channels = 64
        elif cfg.USE_IMAGES:
            input_channels = 64 + 64
        else:
            input_channels = 64

        self.geometry2 = nn.Sequential(
                nn.Conv3d(input_channels, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                nn.ReLU(True),
                Bottleneck(128, 64, stride=1))

        #-----------------------
        # Classifier pipeline
        #-----------------------
        self.classifier = nn.Sequential(
                nn.Linear(self._net_conv_level1_channels * cfg.CLASS_POOLING_SIZE * cfg.CLASS_POOLING_SIZE * cfg.CLASS_POOLING_SIZE, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Linear(256, 128),
                nn.ReLU(True))

class ScanNet_Backbone(Base_Backbone):
    # feature extractor
    def _init_backbone_classifier(self):
        if cfg.ONLY_IMAGES:
            geometry_channels = 0
            color_channels = 128
        elif cfg.USE_IMAGES:
            geometry_channels = 64
            color_channels = 64
        else:
            geometry_channels = 128
            color_channels = 0
        #---------------------------
        # Geometry pipeline
        #---------------------------
        if not cfg.ONLY_IMAGES or not cfg.USE_IMAGES:
            self.geometry1 = nn.Sequential(
                    nn.Conv3d(2, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                    nn.ReLU(True),
                    Bottleneck(32, 32, stride=1),
                    Bottleneck(32, 32, stride=1),

                    nn.Conv3d(32, geometry_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                    nn.ReLU(True),
                    Bottleneck(geometry_channels, 32, stride=1),
                    Bottleneck(geometry_channels, 32, stride=1))

        #--------------------------
        # Color pipeline
        #--------------------------
        if cfg.USE_IMAGES:
            self.color = nn.Sequential(
                    nn.Conv3d(cfg.NUM_IMAGE_CHANNELS, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                    nn.ReLU(True),
                    Bottleneck(64, 32, stride=1),
                    nn.MaxPool3d(3, 1, 1),
                    nn.Conv3d(64, color_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                    nn.ReLU(True),
                    Bottleneck(color_channels, 32, stride=1),
                    nn.MaxPool3d(3, 1, 1))

        #--------------------------
        # Combine pipeline
        #--------------------------
        self.geometry2 = nn.Sequential(
                nn.Conv3d(geometry_channels + color_channels, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                nn.ReLU(True),
                Bottleneck(128, 64, stride=1),
                Bottleneck(128, 64, stride=1),
                nn.MaxPool3d(3, 1, 1))

        #-----------------------
        # Classifier pipeline
        #-----------------------
        self.classifier = nn.Sequential(
                nn.Linear(self._net_conv_level1_channels * cfg.CLASS_POOLING_SIZE * cfg.CLASS_POOLING_SIZE * cfg.CLASS_POOLING_SIZE, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Linear(256, 128),
                nn.ReLU(True))

#-----------------------------------------------------------
# second backbone
#-----------------------------------------------------------
class MaskBackbone(nn.Module):
    def __init__(self):
        super(MaskBackbone, self).__init__()
        # Geometry pipeline
        self.geometry = nn.Sequential(
                nn.Conv3d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv3d(64, 64 if cfg.MASK_USE_IMAGES else cfg.NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=False)
                )

        if cfg.MASK_USE_IMAGES:
            self.color = nn.Sequential(
                    nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(True),
                    nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(True),
                    nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(True),
                    nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(True),
                    nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(True),
                    nn.Conv3d(64, cfg.NUM_CLASSES if cfg.MASK_ONLY_IMAGES else 64, kernel_size=1, stride=1, padding=0, bias=False)
                    )
            self.combine = nn.Sequential(
                    nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(True),
                    nn.Conv3d(128, cfg.NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=False)
                    )

    def forward(self, scene, imageft):
        if cfg.MASK_ONLY_IMAGES:
            net_conv = self.color(imageft)
        else:
            net_conv = self.geometry(scene)

            if cfg.MASK_USE_IMAGES:
                net_color = self.color(imageft)
                net_conv = torch.cat([net_conv, net_color], 1) # NOTE: this assumes batch mode for the concat..
                net_conv = self.combine(net_conv)

        if not self.training:
            net_conv = F.sigmoid(net_conv)
        return net_conv


