import importlib
import glob
import os
import pickle
import matplotlib
from tqdm import tqdm
from reprint import output
os.environ['QT_QPA_PLATFORM']='offscreen'

import numpy as np
import torch
from torchnet.meter.confusionmeter import ConfusionMeter

from lib.datasets.dataloader import get_dataloader
from lib.datasets.dataset import Dataset

from lib.layer_utils.projection import ProjectionHelper
from lib.utils.evaluation import DetectionMAP as Evaluate_metric

from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.utils.config import cfg
from lib.utils.logger import Logger
from lib.utils.timer import Timer
from lib.utils.evaluation import DetectionMAP as Evaluate_metric
from torch.autograd import Variable


def train(args):
    # prepare data
    dataset_train = Dataset(cfg.TRAIN_FILELIST, mode='chunk')
    dataset_val = Dataset(cfg.VAL_FILELIST, mode='chunk')
    dataset_trainval = Dataset(cfg.TRAINVAL_FILELIST, mode='chunk')

    dataloader_train = get_dataloader(dataset_train, batch_size=cfg.BATCH_SIZE, num_workers=1)
    #dataloader_train = dataset_train
    dataloader_val = get_dataloader(dataset=dataset_val, batch_size=1, num_workers=1)
    dataloader_trainval = get_dataloader(dataset=dataset_trainval, batch_size=1, num_workers=1)

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    log_dir = os.path.join(args.output_dir, 'logs')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    #load network
    net_module = importlib.import_module("lib.nets.backbones")
    net = getattr(net_module, cfg.NET)()

    sw = SolverWrapper(net, dataloader_train, dataloader_val, dataloader_trainval, args.output_dir, checkpoint_dir, log_dir)
    sw.train_model(args.epochs)

def benchmark(args):
    dataset = Dataset(cfg.TEST_FILELIST, args.mode)
    dataloader = get_dataloader(dataset, batch_size=1)
    #dataloader = dataset

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    saved_model = os.path.join(checkpoint_dir, 'step_' + args.step + '.pth')

    log_dir = os.path.join(args.output_dir, 'logs')
    logger = Logger(os.path.join(log_dir, 'test'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    net_module = importlib.import_module("lib.nets.backbones")
    net = getattr(net_module, cfg.NET)()
    net.init_modules()
    net.load_state_dict(torch.load(saved_model))
    print('loaded mode from {}'.format(saved_model))
    SolverWrapper.benchmark(net, dataloader, logger)

def test(args):
    dataset = Dataset(cfg.TEST_FILELIST, args.mode)
    dataloader = get_dataloader(dataset, batch_size=1)
    #dataloader = dataset

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    saved_model = os.path.join(checkpoint_dir, 'step_' + args.step + '.pth')

    log_dir = os.path.join(args.output_dir, 'logs')
    logger = Logger(os.path.join(log_dir, 'test'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    net_module = importlib.import_module("lib.nets.backbones")
    net = getattr(net_module, cfg.NET)()
    net.init_modules()
    net.load_state_dict(torch.load(saved_model))
    print('loaded mode from {}'.format(saved_model))
    SolverWrapper.test(net, dataloader, logger)


class SolverWrapper(object):
    """
    A wrapper class for the training process

    """

    def __init__(self, network, dataloader_train, dataloader_val, dataloader_trainval, output_dir, checkpoint_dir, log_dir):
        self.net = network

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.dataloader_trainval = dataloader_trainval

        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir

        self.logger_train = Logger(os.path.join(log_dir, 'train'))
        self.logger_val = Logger(os.path.join(log_dir, 'val'))
        self.logger_trainval = Logger(os.path.join(log_dir, 'trainval'))

    def snapshot(self, index):
        net = self.net

        # Store the model snapshot
        filename = 'step_{:d}'.format(index) + '.pth'
        filename = os.path.join(self.checkpoint_dir, filename)
        torch.save(net.state_dict(), filename)
        print('Write snapshot to {:s}'.format(filename))

        # also store some meta info, random state etd.
        nfilename = 'step_{:d}'.format(index) + '.pkl'
        nfilename = os.path.join(self.checkpoint_dir, nfilename)

        # current state of numpy random
        with open(nfilename, 'wb') as fid:
            pickle.dump(index, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def from_snapshot(self, sfile, nfile):
        print('Restoring mode snapshots from {:s}'.format(sfile))
        pretrained_dict = torch.load(str(sfile))
        model_dict = self.net.state_dict()

        if cfg.LOAD_BACKBONE:
            pretrained_dict_backbone1 = {k: v for k, v in pretrained_dict.items() if ('geometry' in k or 'combine'in k)}
            model_dict.update(pretrained_dict_backbone1)

        if cfg.LOAD_RPN:
            pretrained_dict_rpn = {k: v for k, v in pretrained_dict.items() if 'rpn' in k}
            model_dict.update(pretrained_dict_rpn)

        if cfg.LOAD_CLASS:
            if cfg.NYUV2_FINETUNE:
                pretrained_dict_class = {k: v for k, v in pretrained_dict.items() if ('classifier' in k and 'classifier_cls' not in k and 'classifier_bbox' not in k)}
            else:
                pretrained_dict_class = {k: v for k, v in pretrained_dict.items() if 'classifier' in k}
            model_dict.update(pretrained_dict_class)

        # enet is loaded already in creat_architecture
        if cfg.USE_IMAGES:
            pretrained_dict_image = {k: v for k, v in pretrained_dict.items() if 'color' in k}
            model_dict.update(pretrained_dict_image)


        self.net.load_state_dict(model_dict)
        
        print('Restored')

        with open(nfile, 'rb') as fid:
            last_iter = pickle.load(fid)

        if isinstance(last_iter, list):
            current_snapshot_epoch = last_iter[0]
            iter = last_iter[1]
            last_iter = len(self.dataloader_train) * current_snapshot_epoch + iter

        return last_iter


    def construct_optimizer(self, lr):
        # Optimizer
        params = []
        total_weights = 0
        for key, value in dict(self.net.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value], 'lr':lr*(cfg.DOUBLE_BIAS + 1), 'weight_decay': cfg.BIAS_DECAY and cfg.WEIGHT_DECAY}]
                else:
                    params += [{'params':[value], 'lr':lr, 'weight_decay': cfg.WEIGHT_DECAY}]
                total_weights += value.numel()
        print("total weights: {}".format(total_weights))
        self.optimizer = torch.optim.SGD(params, momentum=cfg.MOMENTUM)

        #set up lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def find_previous(self):
        sfiles = os.path.join(self.checkpoint_dir, 'step_*.pth')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)

        # Get the snapshot name in pytorch
        redfiles = []
        for stepsize in cfg.STEPSIZE:
            redfiles.append(os.path.join(self.checkpoint_dir, 'step_{:d}.pth'.format(stepsize + 1)))
        sfiles = [ss for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.checkpoint_dir, 'step_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.pth', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

    def initialize(self):
        """

        :return:
        """
        np_paths = []
        ss_paths = []

        lr = cfg.LEARNING_RATE
        stepsizes = list(cfg.STEPSIZE)

        return lr, 0, stepsizes, np_paths, ss_paths

    def restore(self, sfile, nfile):
        np_paths = [nfile]
        ss_paths = [sfile]

        last_iter = self.from_snapshot(sfile, nfile)

        lr_scale = 1
        stepsizes = []
        
        for stepsize in cfg.STEPSIZE:
            if last_iter > stepsize:
                lr_scale *= cfg.GAMMA
            else:
                stepsizes.append(stepsize)

        lr = cfg.LEARNING_RATE * lr_scale
        return lr, last_iter, stepsizes, np_paths, ss_paths

    def remove_snapshot(self):
        to_remove = len(self.np_paths) - cfg.SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = self.np_paths[0]
            os.remove(str(nfile))
            self.np_paths.remove(nfile)

        to_remove = len(self.ss_paths) - cfg.SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = self.ss_paths[0]
            os.remove(str(sfile))
            self.ss_paths.remove(sfile)

    def scale_lr(self, optimizer, lr):
        """
        Scale the learning rate of the optimizer

        :param optimizer:
        :param scale:
        :return:
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    def fix_eval_parts(self):
        # FIX PART
        for name, var in self.net.named_parameters():
            if cfg.FIX_BACKBONE and ('geometry' in name or 'color' in name or 'combine' in name) and (not 'mask_backbone' in name):
                var.requires_grad = False
            elif cfg.FIX_RPN and 'rpn' in name:
                var.requires_grad = False
            elif cfg.FIX_CLASS and 'classifier' in name:
                var.requires_grad = False
            elif cfg.FIX_ENET and 'enet' in name:
                var.requires_grad = False

            if cfg.NYUV2_FINETUNE and ('classfier_cls' in name or 'classifier_bbox' in name or 'classifier.4' in name):
                var.requires_grad = True

    def train_model(self, epochs):
        #1. construct the computation graph
        self.net.init_modules()

        #save net structure to data folder
        net_f = open(os.path.join(self.output_dir, 'nn.txt'), 'w')
        net_f.write(str(self.net))
        net_f.close()

        #find previous snapshot 
        lsf, nfiles, sfiles = self.find_previous()

        #2. restore weights
        if lsf == 0:
            lr, last_iter, stepsizes, self.np_paths, self.ss_paths = self.initialize()
        else:
            lr, last_iter, stepsizes, self.np_paths, self.ss_paths = self.restore(str(sfiles[-1]),
                                                                                 str(nfiles[-1]))
        #3. fix weights and eval mode
        self.fix_eval_parts()

        # construct optimizer
        self.construct_optimizer(lr)

        if len(stepsizes) != 0:
            next_stepsize = stepsizes.pop(0)
        else:
            next_stepsize = -1

        train_timer = Timer()
        current_snapshot_epoch = int(last_iter / len(self.dataloader_train))
        for epoch in range(current_snapshot_epoch, epochs):
            print("start epoch {}".format(epoch))
            with output(initial_len=9, interval=0) as content:
                for iter, blobs in enumerate(tqdm(self.dataloader_train)):
                    last_iter += 1
                    # adjust learning rate
                    if last_iter == next_stepsize:
                        lr *= cfg.GAMMA
                        self.scale_lr(self.optimizer, lr)
                        if len(stepsizes) != 0:
                            next_stepsize = stepsizes.pop(0)

                    batch_size = blobs['data'].shape[0]
                    if len(blobs['gt_box']) < batch_size: #invalid sample
                        continue
                    train_timer.tic()
                    # IMAGE PART
                    if cfg.USE_IMAGES:
                        grid_shape = blobs['data'].shape[-3:]
                        projection_helper = ProjectionHelper(cfg.INTRINSIC, cfg.PROJ_DEPTH_MIN, cfg.PROJ_DEPTH_MAX, cfg.DEPTH_SHAPE, grid_shape, cfg.VOXEL_SIZE)
                        proj_mapping = [[projection_helper.compute_projection(d.cuda(), c.cuda(), t.cuda()) for d, c, t in zip(blobs['nearest_images']['depths'][i], blobs['nearest_images']['poses'][i], blobs['nearest_images']['world2grid'][i])] for i in range(batch_size)]

                        jump_flag = False
                        for i in range(batch_size):
                            if None in proj_mapping[i]: #invalid sample
                                jump_flag = True
                                break
                        if jump_flag:
                            continue
                        
                        blobs['proj_ind_3d'] = []
                        blobs['proj_ind_2d'] = []
                        for i in range(batch_size):
                            proj_mapping0, proj_mapping1 = zip(*proj_mapping[i])
                            blobs['proj_ind_3d'].append(torch.stack(proj_mapping0))
                            blobs['proj_ind_2d'].append(torch.stack(proj_mapping1))

                        
                    self.net.forward(blobs)
                    self.optimizer.zero_grad()
                    self.net._losses["total_loss"].backward()
                    self.optimizer.step()

                    train_timer.toc()

                    # Display training information
                    if iter % (cfg.DISPLAY) == 0:
                        self.log_print(epoch*len(self.dataloader_train)+iter, lr, content, train_timer.average_time())
                    self.net.delete_intermediate_states()

                    # validate if satisfying the time criterion
                    if train_timer.total_time() / 3600 >= cfg.VAL_TIME:
                        print('------------------------VALIDATION------------------------------')
                        self.validation(last_iter, 'val')
                        print('------------------------TRAINVAL--------------------------------')
                        self.validation(last_iter, 'trainval')

                        # snapshot
                        if cfg.VAL_TIME > 0.0:
                            ss_path, np_path = self.snapshot(last_iter)
                            self.np_paths.append(np_path)
                            self.ss_paths.append(ss_path)

                            #remove old snapshots if too many
                            if len(self.np_paths) > cfg.SNAPSHOT_KEPT and cfg.SNAPSHOT_KEPT:
                                self.remove_snapshot()

                        train_timer.clean_total_time()


    def log_print(self, index, lr, content, average_time):
        total_loss = 0.0
        content[0] = 'tqdm eats it'
        if cfg.USE_RPN:
            if cfg.NUM_ANCHORS_LEVEL1 != 0:
                content[1] = '>>> rpn_loss_cls_level1: {:.9f}, rpn_loss_box_level1: {:.9f}, rpn_loss_level1: {:.9f}'. \
                        format(self.net._losses['rpn_cross_entropy_level1'].item(), self.net._losses['rpn_loss_box_level1'].item(), 
                            self.net._losses['rpn_cross_entropy_level1'].item() + self.net._losses['rpn_loss_box_level1'].item())
                #log
                self.logger_train.scalar_summary('rpn_loss_cls_level1', self.net._losses['rpn_cross_entropy_level1'].item(), index)
                self.logger_train.scalar_summary('rpn_loss_box_level1', self.net._losses['rpn_loss_box_level1'].item(), index)
                self.logger_train.scalar_summary('rpn_loss_level1', self.net._losses['rpn_cross_entropy_level1'].item() + self.net._losses['rpn_loss_box_level1'].item(), index)

            if cfg.NUM_ANCHORS_LEVEL2 != 0:
                content[2] = '>>> rpn_loss_cls_level2: {:.9f}, rpn_loss_box_level2: {:.9f}, rpn_loss_level2: {:.9f}'. \
                        format(self.net._losses['rpn_cross_entropy_level2'].item(), self.net._losses['rpn_loss_box_level2'].item(), 
                            self.net._losses['rpn_cross_entropy_level2'].item() + self.net._losses['rpn_loss_box_level2'].item())

                self.logger_train.scalar_summary('rpn_loss_cls_level2', self.net._losses['rpn_cross_entropy_level2'].item(), index)
                self.logger_train.scalar_summary('rpn_loss_box_level2', self.net._losses['rpn_loss_box_level2'].item(), index)
                self.logger_train.scalar_summary('rpn_loss_level2', self.net._losses['rpn_cross_entropy_level2'].item() + self.net._losses['rpn_loss_box_level2'].item(), index)

            if cfg.NUM_ANCHORS_LEVEL3 != 0:
                content[3] = '>>> rpn_loss_cls_level3: {:.9f}, rpn_loss_box_level3: {:.9f}, rpn_loss_level3: {:.9f}'. \
                        format(self.net._losses['rpn_cross_entropy_level3'].item(), self.net._losses['rpn_loss_box_level3'].item(), 
                            self.net._losses['rpn_cross_entropy_level3'].item() + self.net._losses['rpn_loss_box_level3'].item())

                self.logger_train.scalar_summary('rpn_loss_cls_level3', self.net._losses['rpn_cross_entropy_level3'].item(), index)
                self.logger_train.scalar_summary('rpn_loss_box_level3', self.net._losses['rpn_loss_box_level3'].item(), index)
                self.logger_train.scalar_summary('rpn_loss_level3', self.net._losses['rpn_cross_entropy_level3'].item() + self.net._losses['rpn_loss_box_level3'].item(), index)

        if cfg.USE_CLASS:
            content[4] = '>>> loss_cls: {:.9f}, loss_box: {:.9f}, classification_loss: {:.9f}'. \
                    format(self.net._losses['cross_entropy'].item(), self.net._losses['loss_box'].item(), 
                           self.net._losses['cross_entropy'].item() + self.net._losses['loss_box'].item())

            self.logger_train.scalar_summary('classification_loss_cls', self.net._losses['cross_entropy'].item(), index)
            self.logger_train.scalar_summary('classification_loss_box', self.net._losses['loss_box'].item(), index)
            self.logger_train.scalar_summary('classification_loss', self.net._losses['cross_entropy'].item() + self.net._losses['loss_box'].item(), index)
        if cfg.USE_MASK:
            content[5] = '>>> mask_loss: {:.9f}'.format(self.net._losses['loss_mask'].item())
            self.logger_train.scalar_summary('mask_loss', self.net._losses['loss_mask'].item(), index)

        content[6] = '>>> total_loss: {:.9f}, lr: {:.6f}, iteration time: {:.3f}s / iter'.format(self.net._losses['total_loss'].item(), lr, average_time)
        self.logger_train.scalar_summary('total_loss', self.net._losses['total_loss'], index)


    def validation(self, index, mode):
        #####################################
        # Preparation
        #####################################
        #-------------------------------
        # metric
        #-------------------------------
        mAP_RPN = Evaluate_metric(1, overlap_threshold=cfg.MAP_THRESH)
        mAP_CLASSIFICATION = Evaluate_metric(cfg.NUM_CLASSES, ignore_class=[0], overlap_threshold=cfg.MAP_THRESH)
        mAP_MASK = Evaluate_metric(cfg.NUM_CLASSES, ignore_class=[0], overlap_threshold=cfg.MAP_THRESH)
        if mode == 'val':
            data_loader = self.dataloader_val
            data_logger = self.logger_val
        elif mode == 'trainval':
            data_loader = self.dataloader_trainval
            data_logger = self.logger_trainval

        ####################################
        # Accumulate data
        ####################################
        timer = Timer()
        timer.tic()
        print('starting validation....')
        for iter, blobs in enumerate(tqdm(data_loader)):
            # if no box: skip
            if len(blobs['gt_box']) == 0:
                continue

            if cfg.USE_IMAGES:
                grid_shape = blobs['data'].shape[-3:]
                projection_helper = ProjectionHelper(cfg.INTRINSIC, cfg.PROJ_DEPTH_MIN, cfg.PROJ_DEPTH_MAX, cfg.DEPTH_SHAPE, grid_shape, cfg.VOXEL_SIZE)
                proj_mapping = [projection_helper.compute_projection(d.cuda(), c.cuda(), t.cuda()) for d, c, t in zip(blobs['nearest_images']['depths'][0], blobs['nearest_images']['poses'][0], blobs['nearest_images']['world2grid'][0])]

                if None in proj_mapping: #invalid sample
                    continue
                
                blobs['proj_ind_3d'] = []
                blobs['proj_ind_2d'] = []
                proj_mapping0, proj_mapping1 = zip(*proj_mapping)
                blobs['proj_ind_3d'].append(torch.stack(proj_mapping0))
                blobs['proj_ind_2d'].append(torch.stack(proj_mapping1))

            self.net.forward(blobs, 'TEST', [])
            #--------------------------------------
            # RPN: loss, metric 
            #--------------------------------------
            if cfg.USE_RPN:
                # (n, 6)
                gt_box = blobs['gt_box'][0].numpy()[:, 0:6]
                gt_box_label = np.zeros(gt_box.shape[0])

                try:
                    pred_box_num = (self.net._predictions['roi_scores'][0][:, 0] > cfg.ROI_THRESH).nonzero().size(0)
                    pred_box = self.net._predictions['rois'][0].cpu().numpy()[:pred_box_num]
                    pred_box_label = np.zeros(pred_box_num) 
                    pred_box_score = self.net._predictions['roi_scores'][0].cpu().numpy()[:pred_box_num, 0]
                except:
                    pred_box = self.net._predictions['rois'][0].cpu().numpy()[:1]
                    pred_box_label = np.zeros(1)
                    pred_box_score = self.net._predictions['roi_scores'][0].cpu().numpy()[:1, 0]

                #evaluation metric 
                mAP_RPN.evaluate(pred_box,
                                 pred_box_label,
                                 pred_box_score,
                                 gt_box,
                                 gt_box_label)

            #--------------------------------------
            # Classification: loss, metric 
            #--------------------------------------
            if cfg.USE_CLASS:
                # groundtruth
                gt_box = blobs['gt_box'][0].numpy()[:, 0:6]
                gt_class = blobs['gt_box'][0][:, 6].numpy()

                # predictions
                pred_class = self.net._predictions['cls_pred'].data.cpu().numpy()

                # only predictions['rois'] is list and is Tensor / others are no list and Variable
                rois = self.net._predictions['rois'][0].cpu()
                box_reg_pre = self.net._predictions["bbox_pred"].data.cpu().numpy()
                box_reg = np.zeros((box_reg_pre.shape[0], 6))
                pred_conf_pre = self.net._predictions['cls_prob'].data.cpu().numpy()
                pred_conf = np.zeros((pred_conf_pre.shape[0]))


                for pred_ind in range(pred_class.shape[0]):
                    box_reg[pred_ind, :] = box_reg_pre[pred_ind, pred_class[pred_ind]*6:(pred_class[pred_ind]+1)*6]
                    pred_conf[pred_ind] = pred_conf_pre[pred_ind, pred_class[pred_ind]]

                pred_box = bbox_transform_inv(rois, torch.from_numpy(box_reg).float())
                pred_box = clip_boxes(pred_box, self.net._scene_info[:3]).numpy()

                # pickup
                sort_index = []
                for conf_index in range(pred_conf.shape[0]):
                    if pred_conf[conf_index] > cfg.CLASS_THRESH:
                        sort_index.append(True)
                    else:
                        sort_index.append(False)

                # eliminate bad box
                for idx, box in enumerate(pred_box):
                    if round(box[0]) >= round(box[3]) or round(box[1]) >= round(box[4]) or round(box[2]) >= round(box[5]):
                        sort_index[idx] = False

                if len(pred_box[sort_index]) == 0:
                    print('no pred box')

                if iter < cfg.VAL_NUM:
                    os.makedirs('{}/{}'.format(cfg.VAL_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), exist_ok=True)
                    np.save('{}/{}/pred_class'.format(cfg.VAL_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), pred_class)
                    np.save('{}/{}/pred_conf'.format(cfg.VAL_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), pred_conf)
                    np.save('{}/{}/pred_box'.format(cfg.VAL_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), pred_box)
                    np.save('{}/{}/scene'.format(cfg.VAL_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), np.where(blobs['data'][0,0].numpy() <= 1, 1, 0))
                    np.save('{}/{}/gt_class'.format(cfg.VAL_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), gt_class)
                    np.save('{}/{}/gt_box'.format(cfg.VAL_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), gt_box)

                mAP_CLASSIFICATION.evaluate(
                        pred_box[sort_index],
                        pred_class[sort_index],
                        pred_conf[sort_index],
                        gt_box,
                        gt_class)

            #--------------------------------------
            # MASK: loss, metric 
            #--------------------------------------
            if cfg.USE_MASK:
                # gt data
                gt_box = blobs['gt_box'][0].numpy()[:, 0:6]
                gt_class = blobs['gt_box'][0][:, 6].numpy()
                gt_mask = blobs['gt_mask'][0]

                pred_class = self.net._predictions['cls_pred'].data.cpu().numpy()
                pred_conf = np.zeros((pred_class.shape[0]))
                for pred_ind in range(pred_class.shape[0]):
                    pred_conf[pred_ind] = self.net._predictions['cls_prob'].data.cpu().numpy()[pred_ind, pred_class.data[pred_ind]]

                # pickup
                sort_index = pred_conf > cfg.CLASS_THRESH

                # eliminate bad box
                for idx, box in enumerate(pred_box):
                    if round(box[0]) >= round(box[3]) or round(box[1]) >= round(box[4]) or round(box[2]) >= round(box[5]):
                        sort_index[idx] = False

                pred_mask = []
                mask_ind = 0
                for ind, cls in enumerate(pred_class):
                    if sort_index[ind]:
                        mask = self.net._predictions['mask_pred'][0][mask_ind][0][cls].data.cpu().numpy()
                        mask = np.where(mask >=cfg.MASK_THRESH, 1, 0).astype(np.float32)
                        pred_mask.append(mask)
                        mask_ind += 1

                if iter < cfg.VAL_NUM: 
                    pickle.dump(pred_mask, open('{}/{}/pred_mask'.format(cfg.VAL_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), 'wb'))
                    pickle.dump(sort_index, open('{}/{}/pred_mask_index'.format(cfg.VAL_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), 'wb'))
                    pickle.dump(gt_mask, open('{}/{}/gt_mask'.format(cfg.VAL_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), 'wb'))

                mAP_MASK.evaluate_mask(
                        pred_box[sort_index],
                        pred_class[sort_index],
                        pred_conf[sort_index],
                        pred_mask,
                        gt_box,
                        gt_class, 
                        gt_mask, 
                        self.net._scene_info)

            self.net.delete_intermediate_states()
        timer.toc()
        print('It took {:.3f}s for Validation on chunks'.format(timer.total_time()))

        ###################################
        # Summary
        ###################################
        if cfg.USE_RPN:
            mAP_RPN.finalize()
            print('AP of RPN: {}'.format(mAP_RPN.mAP()))
            data_logger.scalar_summary('AP_ROI', mAP_RPN.mAP(), index)

        if cfg.USE_CLASS:
            mAP_CLASSIFICATION.finalize()
            print('mAP of CLASSIFICATION: {}'.format(mAP_CLASSIFICATION.mAP()))
            for class_ind in range(cfg.NUM_CLASSES):
                if class_ind not in mAP_CLASSIFICATION.ignore_class:
                    print('class {}: {}'.format(class_ind, mAP_CLASSIFICATION.AP(class_ind)))
            data_logger.scalar_summary('mAP_CLASSIFICATION', mAP_CLASSIFICATION.mAP(), index)

        if cfg.USE_MASK:
            mAP_MASK.finalize()
            print('mAP of mask: {}'.format(mAP_MASK.mAP()))
            for class_ind in range(cfg.NUM_CLASSES):
                if class_ind not in mAP_MASK.ignore_class:
                    print('class {}: {}'.format(class_ind, mAP_MASK.AP(class_ind)))
            data_logger.scalar_summary('mAP_MASK', mAP_MASK.mAP(), index)

    @staticmethod
    def benchmark(net, data_loader, data_logger):
        #####################################
        # Preparation
        #####################################
        os.makedirs(cfg.TEST_SAVE_DIR, exist_ok=True)

        ####################################
        # Accumulate data
        ####################################
        timer = Timer()
        timer.tic()
        print('starting test on whole scan....')
        for iter, blobs in enumerate(tqdm(data_loader)):
            FLAG_EXIST_CLASS = False
            if os.path.isfile('{}/{}/pred_box.npy'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12])):
                pred_class = np.load('{}/{}/pred_class.npy'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]))
                pred_conf = np.load('{}/{}/pred_conf.npy'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]))
                pred_box = np.load('{}/{}/pred_box.npy'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]))
                FLAG_EXIST_CLASS = True

            #--------------------------------------
            # Classification: loss, metric 
            #--------------------------------------
            if not FLAG_EXIST_CLASS:
                # color proj
                killing_inds = None
                if cfg.USE_IMAGES:
                    grid_shape = blobs['data'].shape[-3:]
                    projection_helper = ProjectionHelper(cfg.INTRINSIC, cfg.PROJ_DEPTH_MIN, cfg.PROJ_DEPTH_MAX, cfg.DEPTH_SHAPE, grid_shape, cfg.VOXEL_SIZE)
                    if grid_shape[0]*grid_shape[1]*grid_shape[2] > cfg.MAX_VOLUME or blobs['nearest_images']['depths'][0].shape[0] > cfg.MAX_IMAGE:
                        proj_mapping = [projection_helper.compute_projection(d, c, t) for d, c, t in zip(blobs['nearest_images']['depths'][0], blobs['nearest_images']['poses'][0], blobs['nearest_images']['world2grid'][0])]
                    else:
                        proj_mapping = [projection_helper.compute_projection(d.cuda(), c.cuda(), t.cuda()) for d, c, t in zip(blobs['nearest_images']['depths'][0], blobs['nearest_images']['poses'][0], blobs['nearest_images']['world2grid'][0])]
                        
                    killing_inds = []
                    real_proj_mapping = []
                    if None in proj_mapping: #invalid sample
                        for killing_ind, killing_item in enumerate(proj_mapping):
                            if killing_item == None:
                                killing_inds.append(killing_ind)
                            else:
                                real_proj_mapping.append(killing_item)
                        print('{}: (invalid sample: no valid projection)'.format(blobs['id']))
                    else:
                        real_proj_mapping = proj_mapping
                    blobs['proj_ind_3d'] = []
                    blobs['proj_ind_2d'] = []
                    proj_mapping0, proj_mapping1 = zip(*real_proj_mapping)
                    blobs['proj_ind_3d'].append(torch.stack(proj_mapping0))
                    blobs['proj_ind_2d'].append(torch.stack(proj_mapping1))

                net.forward(blobs, 'TEST', killing_inds)

                # test with detection pipeline
                pred_class = net._predictions['cls_pred'].data.cpu().numpy()
                rois = net._predictions['rois'][0].cpu()
                box_reg_pre = net._predictions["bbox_pred"].data.cpu().numpy()
                box_reg = np.zeros((box_reg_pre.shape[0], 6))
                pred_conf_pre = net._predictions['cls_prob'].data.cpu().numpy()
                pred_conf = np.zeros((pred_conf_pre.shape[0]))

                for pred_ind in range(pred_class.shape[0]):
                    box_reg[pred_ind, :] = box_reg_pre[pred_ind, pred_class[pred_ind]*6:(pred_class[pred_ind]+1)*6]
                    pred_conf[pred_ind] = pred_conf_pre[pred_ind, pred_class[pred_ind]]

                pred_box = bbox_transform_inv(rois, torch.from_numpy(box_reg).float())
                pred_box = clip_boxes(pred_box, net._scene_info[:3]).numpy()

                # pickup
                sort_index = []
                for conf_index in range(pred_conf.shape[0]):
                    if pred_conf[conf_index] > cfg.CLASS_THRESH:
                        sort_index.append(True)
                    else:
                        sort_index.append(False)

                # eliminate bad box
                for idx, box in enumerate(pred_box):
                    if round(box[0]) >= round(box[3]) or round(box[1]) >= round(box[4]) or round(box[2]) >= round(box[5]):
                        sort_index[idx] = False

                os.makedirs('{}/{}'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), exist_ok=True)
                np.save('{}/{}/pred_class'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), pred_class)
                np.save('{}/{}/pred_conf'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), pred_conf)
                np.save('{}/{}/pred_box'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), pred_box)
                np.save('{}/{}/scene'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), np.where(blobs['data'][0,0].numpy() <= 1, 1, 0))

            if cfg.USE_MASK:
                # pickup
                sort_index = []
                for conf_index in range(pred_conf.shape[0]):
                    if pred_conf[conf_index] > cfg.CLASS_THRESH:
                        sort_index.append(True)
                    else:
                        sort_index.append(False)

                # eliminate bad box
                for idx, box in enumerate(pred_box):
                    if round(box[0]) >= round(box[3]) or round(box[1]) >= round(box[4]) or round(box[2]) >= round(box[5]):
                        sort_index[idx] = False

                # test with mask pipeline
                net.mask_backbone.eval()
                net.mask_backbone.cuda()
                mask_pred_batch = []
                for net_i in range(1):
                    mask_pred = []
                    for pred_box_ind, pred_box_item in enumerate(pred_box):
                        if sort_index[pred_box_ind]:
                            mask_pred.append(net.mask_backbone(Variable(blobs['data'].cuda())[net_i:net_i+1, :, 
                                                                            int(round(pred_box_item[0])):int(round(pred_box_item[3])),
                                                                            int(round(pred_box_item[1])):int(round(pred_box_item[4])), 
                                                                            int(round(pred_box_item[2])):int(round(pred_box_item[5]))
                                                                            ], [] if cfg.USE_IMAGES else None))

                    mask_pred_batch.append(mask_pred)
                net._predictions['mask_pred'] = mask_pred_batch

                # save test result
                pred_mask = []
                mask_ind = 0
                for ind, cls in enumerate(pred_class):
                    if sort_index[ind]:
                        mask = net._predictions['mask_pred'][0][mask_ind][0][cls].data.cpu().numpy()
                        mask = np.where(mask >=cfg.MASK_THRESH, 1, 0).astype(np.float32)
                        pred_mask.append(mask)
                        mask_ind += 1

                pickle.dump(pred_mask, open('{}/{}/pred_mask'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), 'wb'))
                pickle.dump(sort_index, open('{}/{}/pred_mask_index'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), 'wb'))

        timer.toc()
        print('It took {:.3f}s for test on whole scenes'.format(timer.total_time()))

    @staticmethod
    def test(net, data_loader, data_logger):
        #####################################
        # Preparation
        #####################################
        os.makedirs(cfg.TEST_SAVE_DIR, exist_ok=True)
        mAP_CLASSIFICATION = Evaluate_metric(cfg.NUM_CLASSES, ignore_class=[0], overlap_threshold=cfg.MAP_THRESH)
        mAP_MASK = Evaluate_metric(cfg.NUM_CLASSES, ignore_class=[0], overlap_threshold=cfg.MAP_THRESH)

        ####################################
        # Accumulate data
        ####################################
        pred_all = {}
        gt_all = {}

        timer = Timer()
        timer.tic()
        print('starting test on whole scan....')
        for iter, blobs in enumerate(tqdm(data_loader)):

            try:
                gt_box = blobs['gt_box'][0].numpy()[:, 0:6]
                gt_class = blobs['gt_box'][0][:, 6].numpy()
            except:
                continue

            # color proj
            killing_inds = None
            if cfg.USE_IMAGES:
                grid_shape = blobs['data'].shape[-3:]
                projection_helper = ProjectionHelper(cfg.INTRINSIC, cfg.PROJ_DEPTH_MIN, cfg.PROJ_DEPTH_MAX, cfg.DEPTH_SHAPE, grid_shape, cfg.VOXEL_SIZE)
                if grid_shape[0]*grid_shape[1]*grid_shape[2] > cfg.MAX_VOLUME or blobs['nearest_images']['depths'][0].shape[0] > cfg.MAX_IMAGE:
                    proj_mapping = [projection_helper.compute_projection(d, c, t) for d, c, t in zip(blobs['nearest_images']['depths'][0], blobs['nearest_images']['poses'][0], blobs['nearest_images']['world2grid'][0])]
                else:
                    proj_mapping = [projection_helper.compute_projection(d.cuda(), c.cuda(), t.cuda()) for d, c, t in zip(blobs['nearest_images']['depths'][0], blobs['nearest_images']['poses'][0], blobs['nearest_images']['world2grid'][0])]
                    
                killing_inds = []
                real_proj_mapping = []
                if None in proj_mapping: #invalid sample
                    for killing_ind, killing_item in enumerate(proj_mapping):
                        if killing_item == None:
                            killing_inds.append(killing_ind)
                        else:
                            real_proj_mapping.append(killing_item)
                    print('{}: (invalid sample: no valid projection)'.format(blobs['id']))
                else:
                    real_proj_mapping = proj_mapping
                blobs['proj_ind_3d'] = []
                blobs['proj_ind_2d'] = []
                proj_mapping0, proj_mapping1 = zip(*real_proj_mapping)
                blobs['proj_ind_3d'].append(torch.stack(proj_mapping0))
                blobs['proj_ind_2d'].append(torch.stack(proj_mapping1))

            net.forward(blobs, 'TEST', killing_inds)

            # test with detection pipeline
            pred_class = net._predictions['cls_pred'].data.cpu().numpy()
            rois = net._predictions['rois'][0].cpu()
            box_reg_pre = net._predictions["bbox_pred"].data.cpu().numpy()
            box_reg = np.zeros((box_reg_pre.shape[0], 6))
            pred_conf_pre = net._predictions['cls_prob'].data.cpu().numpy()
            pred_conf = np.zeros((pred_conf_pre.shape[0]))

            for pred_ind in range(pred_class.shape[0]):
                box_reg[pred_ind, :] = box_reg_pre[pred_ind, pred_class[pred_ind]*6:(pred_class[pred_ind]+1)*6]
                pred_conf[pred_ind] = pred_conf_pre[pred_ind, pred_class[pred_ind]]

            pred_box = bbox_transform_inv(rois, torch.from_numpy(box_reg).float())
            pred_box = clip_boxes(pred_box, net._scene_info[:3]).numpy()

            os.makedirs('{}/{}'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), exist_ok=True)
            np.save('{}/{}/pred_class'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), pred_class)
            np.save('{}/{}/pred_conf'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), pred_conf)
            np.save('{}/{}/pred_box'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), pred_box)
            np.save('{}/{}/scene'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), np.where(blobs['data'][0,0].numpy() <= 1, 1, 0))
            np.save('{}/{}/gt_class'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), gt_class)
            np.save('{}/{}/gt_box'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), gt_box)

            # pickup
            sort_index = []
            for conf_index in range(pred_conf.shape[0]):
                if pred_conf[conf_index] > cfg.CLASS_THRESH:
                    sort_index.append(True)
                else:
                    sort_index.append(False)

            # eliminate bad box
            for idx, box in enumerate(pred_box):
                if round(box[0]) >= round(box[3]) or round(box[1]) >= round(box[4]) or round(box[2]) >= round(box[5]):
                    sort_index[idx] = False

            mAP_CLASSIFICATION.evaluate(
                    pred_box[sort_index],
                    pred_class[sort_index],
                    pred_conf[sort_index],
                    gt_box,
                    gt_class)

            if cfg.USE_MASK:
                gt_mask = blobs['gt_mask'][0]
                # pickup
                sort_index = []
                for conf_index in range(pred_conf.shape[0]):
                    if pred_conf[conf_index] > cfg.CLASS_THRESH:
                        sort_index.append(True)
                    else:
                        sort_index.append(False)

                # eliminate bad box
                for idx, box in enumerate(pred_box):
                    if round(box[0]) >= round(box[3]) or round(box[1]) >= round(box[4]) or round(box[2]) >= round(box[5]):
                        sort_index[idx] = False

                # test with mask pipeline
                net.mask_backbone.eval()
                net.mask_backbone.cuda()
                mask_pred_batch = []
                for net_i in range(1):
                    mask_pred = []
                    for pred_box_ind, pred_box_item in enumerate(pred_box):
                        if sort_index[pred_box_ind]:
                            mask_pred.append(net.mask_backbone(Variable(blobs['data'].cuda())[net_i:net_i+1, :, 
                                                                            int(round(pred_box_item[0])):int(round(pred_box_item[3])),
                                                                            int(round(pred_box_item[1])):int(round(pred_box_item[4])), 
                                                                            int(round(pred_box_item[2])):int(round(pred_box_item[5]))
                                                                            ], [] if cfg.USE_IMAGES else None))

                    mask_pred_batch.append(mask_pred)
                net._predictions['mask_pred'] = mask_pred_batch

                # save test result
                pred_mask = []
                mask_ind = 0
                for ind, cls in enumerate(pred_class):
                    if sort_index[ind]:
                        mask = net._predictions['mask_pred'][0][mask_ind][0][cls].data.cpu().numpy()
                        mask = np.where(mask >=cfg.MASK_THRESH, 1, 0).astype(np.float32)
                        pred_mask.append(mask)
                        mask_ind += 1

                pickle.dump(pred_mask, open('{}/{}/pred_mask'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), 'wb'))
                pickle.dump(sort_index, open('{}/{}/pred_mask_index'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), 'wb'))
                pickle.dump(gt_mask, open('{}/{}/gt_mask'.format(cfg.TEST_SAVE_DIR, blobs['id'][0].split('/')[-1][:12]), 'wb'))

                mAP_MASK.evaluate_mask(
                        pred_box[sort_index],
                        pred_class[sort_index],
                        pred_conf[sort_index],
                        pred_mask,
                        gt_box,
                        gt_class, 
                        gt_mask, 
                        net._scene_info)

        timer.toc()
        print('It took {:.3f}s for test on whole scenes'.format(timer.total_time()))

        ###################################
        # Summary
        ###################################
        if cfg.USE_CLASS:
            mAP_CLASSIFICATION.finalize()
            print('mAP of CLASSIFICATION: {}'.format(mAP_CLASSIFICATION.mAP()))
            for class_ind in range(cfg.NUM_CLASSES):
                if class_ind not in mAP_CLASSIFICATION.ignore_class:
                    print('class {}: {}'.format(class_ind, mAP_CLASSIFICATION.AP(class_ind)))

        if cfg.USE_MASK:
            mAP_MASK.finalize()
            print('mAP of mask: {}'.format(mAP_MASK.mAP()))
            for class_ind in range(cfg.NUM_CLASSES):
                if class_ind not in mAP_MASK.ignore_class:
                    print('class {}: {}'.format(class_ind, mAP_MASK.AP(class_ind)))
