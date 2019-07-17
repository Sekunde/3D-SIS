import os
import sys
import argparse
import math
import h5py
import numpy as np 
import ipdb
from scipy import misc
from PIL import Image
import torch
from torch.utils.data import Dataset as Pytorch_Dataset
import torchvision.transforms as transforms
from lib.utils.config import cfg
from lib.datasets.BinaryReader import BinaryReader
import csv
from numpy import copy


class Dataset(Pytorch_Dataset):
    """
    SUNCG/ScanNet DATASET
    """
    def __init__(self, data_location, mode):
        super(Dataset, self).__init__()
        # mode = {chunk, scene, benchmark}
        #              max height     image      filter box 
        # chunk:          48            5          yes
        # scene           48           all         no
        # benchmark:      480          all         no
        self.mode = mode 

        if os.path.isdir(data_location):
            datalist = [os.path.join(data_location, x) for x in os.listdir(data_location) if os.path.isfile(os.path.join(data_location, x))]
            self.scenes = datalist
        elif os.path.isfile(data_location):
            datalist = open(data_location, 'r')
            self.scenes = [x.strip() for x in datalist.readlines()]

        if cfg.LABEL_MAP !='':
            self.mapping, self.weights  = Dataset.load_mapping(cfg.LABEL_MAP)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):

        #---------------------------
        # read sdf
        #---------------------------
        reader = BinaryReader(self.scenes[idx])
        dimX, dimY, dimZ = reader.read('UINT64', 3)
        data = reader.read('float', dimX * dimY * dimZ)
        data = np.expand_dims(np.reshape(data, (dimX, dimY, dimZ), order='F'), 0).astype(np.float32)

        if cfg.FLIP_TSDF:
            trunc_data = np.clip(data, -cfg.TRUNCATED, cfg.TRUNCATED)
            trunc_abs_data = np.abs(trunc_data)
            trunc_abs_data_flip = cfg.TRUNCATED - trunc_abs_data
            data = np.concatenate([trunc_abs_data_flip, np.greater(data, -1)], 0)
        elif cfg.LOG_TSDF:
            trunc_data = np.clip(data, -cfg.TRUNCATED, cfg.TRUNCATED)
            trunc_abs_data = np.abs(trunc_data)
            trunc_abs_data_log = np.log(trunc_abs_data)
            data = np.concatenate([trunc_abs_data_log, np.greater(data, -1)], 0)
        else:
            trunc_data = np.clip(data, -cfg.TRUNCATED, cfg.TRUNCATED)
            trunc_abs_data = np.abs(trunc_data)
            data = np.concatenate([trunc_abs_data, np.greater(data, -1)], 0)
        
        #---------------------------
        # read bbox
        #---------------------------
        (num_box,) = reader.read('uint32')

        # prepare retuan placeholder
        gt_box = np.zeros((num_box, 7), dtype=np.float32)
        gt_mask = []
        nearest_images = {}
        image_files = []

        for i in range(num_box):
            # gt_box
            minx, miny, minz, maxx, maxy, maxz = reader.read('float', 6)
            (labelid,) = reader.read('uint32')
            if cfg.LABEL_MAP != '':
                labelid = self.mapping[labelid]
            gt_box[i] = [math.floor(minx), math.floor(miny), math.floor(minz), math.ceil(maxx), math.ceil(maxy), math.ceil(maxz), labelid]

        #---------------------------
        # read mask
        #---------------------------
        if cfg.USE_MASK or cfg.KEEP_THRESH or cfg.USE_IMAGES:
            (num_mask,) = reader.read('uint32')

            for i in range(num_mask):
                (labelid,) = reader.read('uint32')
                dimX, dimY, dimZ = reader.read('UINT64', 3)
                mask_data = reader.read('uint16', dimX * dimY * dimZ)
                mask_data = np.reshape(mask_data, (dimX, dimY, dimZ), order='F').astype(np.uint8)
                mask_data[mask_data > 1] = 0
                gt_mask.append(mask_data)

        
        #---------------------------
        # read thresh
        #---------------------------
        if cfg.KEEP_THRESH or cfg.USE_IMAGES: # images data has portions of bbox inside volume information
            (num_box,) = reader.read('uint32')
            box_stats = []
            for i in range(num_box):
                (part_in_vol,) = reader.read('float')
                # sometimes file is bad, part_in_vol is always 1.0, so compute it on the fly
                if self.mode == 'chunk':
                    part_in_vol = self.outbbox_thresh(gt_box[i])
                box_stats.append(part_in_vol)

            # implement a filter here for the boxes/masks
            keep_inds = []
            for i in range(num_box):
                if box_stats[i] >= cfg.KEEP_THRESH:
                    keep_flag = True
                    # ignore 0 weight class
                    if self.weights[int(gt_box[i, 6])] == 0: 
                        keep_flag = False
                    if keep_flag:
                        keep_inds.append(i)
                
            gt_box = gt_box[keep_inds]
            if cfg.USE_MASK:
                gt_mask = [gt_mask[ind] for ind in keep_inds]

        #----------------------------
        # read images
        #----------------------------
        if cfg.USE_IMAGES:
            # read image, depth image, pose, world2grid
            depths = []
            images = []
            poses = []
            frameids = []
            world2grid = np.linalg.inv(np.transpose(np.reshape(reader.read('float', 16), (4, 4), order='F')).astype(np.float32))
            (num_images,) = reader.read('uint32')

            if cfg.BASE_IMAGE_PATH.endswith('augmented') or cfg.BASE_IMAGE_PATH.endswith('augmented/'):
                scene_name = os.path.basename(self.scenes[idx]).rsplit('_', 1)[0] if self.mode=='chunk' else os.path.basename(self.scenes[idx]).split('.')[0]
            elif cfg.BASE_IMAGE_PATH.endswith('square') or cfg.BASE_IMAGE_PATH.endswith('square/'):
                scene_name = os.path.basename(self.scenes[idx]).split('__')[0]
            else:
                raise NotImplementedError

            if self.mode != 'chunk':
                num_images = os.listdir(os.path.join(cfg.BASE_IMAGE_PATH, scene_name, 'depth'))
                # reload correct world2grid for scene
                world2grid = self.load_pose(os.path.join(cfg.BASE_IMAGE_PATH, scene_name, 'world2grid.txt'))
                # padding substraction
                world2grid[0][3] = world2grid[0][3] - 10
                world2grid[1][3] = world2grid[1][3] - 16
                world2grid[2][3] = world2grid[2][3] - 10
            else:
                num_images = range(num_images)

            for i in num_images:
                if self.mode != 'chunk':
                    frameid = i.split('.')[0]
                else:
                    (frameid,) = reader.read('uint32')  

                depth_file = os.path.join(cfg.BASE_IMAGE_PATH, scene_name, 'depth', str(frameid) + '.png')
                image_file = os.path.join(cfg.BASE_IMAGE_PATH, scene_name, cfg.IMAGE_TYPE, str(frameid) + cfg.IMAGE_EXT)
                pose_file = os.path.join(cfg.BASE_IMAGE_PATH, scene_name, 'pose', str(frameid) + '.txt')
                poses.append(self.load_pose(pose_file))
                depths.append(self.load_depth(depth_file, cfg.DEPTH_SHAPE))
                im_pre = self.load_image(image_file, cfg.IMAGE_SHAPE)

                if cfg.USE_IMAGES_GT and cfg.LABEL_MAP !='':
                    im_pre = np.where(im_pre <= 40, im_pre, 0)
                    im_post = copy(im_pre)
                    for k, v in self.mapping.items(): 
                        if self.weights[v] == 0:
                            v = 0
                        im_pre[im_post==k] = v

                images.append(im_pre)
                frameids.append(frameid)
                image_files.append(image_file)

            nearest_images = {'depths': depths, 'images': images, 'poses': poses, 'world2grid': world2grid, 'frameids': frameids}

        #---------------------------
        # crop max height
        #---------------------------
        if self.mode == 'benchmark':
            maxHeight = 480
        else:
            maxHeight = 48

        gt_box_list = []
        gt_mask_list = []
        for gt_box_ind, gt_box_single in enumerate(gt_box):
            if gt_box_single[1] <= maxHeight and gt_box_single[4] <= maxHeight:
                gt_box_list.append(gt_box_single)
                gt_mask_list.append(gt_mask[gt_box_ind])
        gt_box = np.array(gt_box_list)
        gt_mask = gt_mask_list
        data = data[:,:,:maxHeight,:]

        # dict return
        dict_return = {
            'id': self.scenes[idx],
            'data': data,
            'gt_box': gt_box,
            'gt_mask': gt_mask,
            'nearest_images': nearest_images,
            'image_files': image_files
        }
        reader.close()

        return dict_return

    def outbbox_thresh(self, gt_box):
        overall = (gt_box[3] - gt_box[0]) * (gt_box[4] - gt_box[1]) * (gt_box[5] - gt_box[2])
        minx = min(max(gt_box[0], 0), 96)
        miny = min(max(gt_box[1], 0), 48)
        minz = min(max(gt_box[2], 0), 96)
        maxx = min(max(gt_box[3], 0), 96)
        maxy = min(max(gt_box[4], 0), 48)
        maxz = min(max(gt_box[5], 0), 96)
        part_in = (maxx - minx) * (maxy - miny) * (maxz - minz)
        return part_in / overall

    def load_pose(self, filename):
        pose = np.zeros((4, 4))
        lines = open(filename).read().splitlines()
        assert len(lines) == 4
        lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
        return np.asarray(lines).astype(np.float32)

    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        return image

    def load_depth(self, file, image_dims):
        depth_image = misc.imread(file)
        # preprocess
        depth_image = self.resize_crop_image(depth_image, image_dims)
        depth_image = depth_image.astype(np.float32) / 1000.0
        return depth_image

    def load_image(self, file, image_dims):
        image = misc.imread(file)
        # preprocess
        image = self.resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=cfg.COLOR_MEAN, std=cfg.COLOR_STD)(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image

    @staticmethod
    def load_mapping(label_file):
        mapping = dict()
        weights_pre= dict()
        # first weight for background 
        weights = [0.3280746813009404]
        csvfile = open(label_file) 
        spamreader = csv.DictReader(csvfile, delimiter=',')
        for row in spamreader:
            mapping[int(row['nyu40id'])] = int(row['mappedIdConsecutive'])
            weights_pre[int(row['mappedIdConsecutive'])] = float(row['weight'])
        for key in sorted(weights_pre.keys()):
            weights.append(weights_pre[key])
        csvfile.close()

        return mapping, weights

