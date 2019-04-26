import struct

import numpy as np
import torch
import os
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.layer_utils.anchor_target_layer import anchor_target_layer
from lib.layer_utils.proposal_layer import proposal_layer
from lib.layer_utils.proposal_target_layer import proposal_target_layer
from lib.layer_utils.mask_target_layer import mask_target_layer
from lib.layer_utils.roi_pooling.roi_pool import RoIPoolFunction
from lib.layer_utils.projection import Projection
from lib.layer_utils.generate_anchors import generate_anchors

from lib.utils.config import cfg
from lib.utils.timer import Timer
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes

from lib.nets import backbones, enet

from tools.visualization import write_bbox, write_mask

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self._predictions = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._mask_targets = {}
        self._losses = {}

    def init_modules(self):
        self._init_backbone_classifier()
        #rpn
        if cfg.USE_RPN:
            if cfg.NUM_ANCHORS_LEVEL1 != 0:
                self.rpn_net_level1 = nn.Conv3d(self._net_conv_level1_channels, cfg.RPN_CHANNELS, [3, 3, 3], padding=1)
                self.rpn_cls_score_net_level1 = nn.Sequential(nn.Conv3d(cfg.RPN_CHANNELS, cfg.NUM_ANCHORS_LEVEL1 * 2, [1, 1, 1]))
                self.rpn_bbox_pred_net_level1 = nn.Conv3d(cfg.RPN_CHANNELS, cfg.NUM_ANCHORS_LEVEL1 * 6, [1, 1, 1])

            if cfg.NUM_ANCHORS_LEVEL2 != 0:
                self.rpn_net_level2 = nn.Conv3d(self._net_conv_level2_channels, cfg.RPN_CHANNELS, [3, 3, 3], padding=1)
                self.rpn_cls_score_net_level2 = nn.Sequential(nn.Conv3d(cfg.RPN_CHANNELS, cfg.NUM_ANCHORS_LEVEL2 * 2, [1, 1, 1]))
                self.rpn_bbox_pred_net_level2 = nn.Conv3d(cfg.RPN_CHANNELS, cfg.NUM_ANCHORS_LEVEL2 * 6, [1, 1, 1])

            if cfg.NUM_ANCHORS_LEVEL3 != 0:
                self.rpn_net_level3 = nn.Conv3d(self._net_conv_level3_channels, cfg.RPN_CHANNELS, [3, 3, 3], padding=1)
                self.rpn_cls_score_net_level3 = nn.Sequential(nn.Conv3d(cfg.RPN_CHANNELS, cfg.NUM_ANCHORS_LEVEL3 * 2, [1, 1, 1]))
                self.rpn_bbox_pred_net_level3 = nn.Conv3d(cfg.RPN_CHANNELS, cfg.NUM_ANCHORS_LEVEL3 * 6, [1, 1, 1])

        # classifier after ROI layer
        if cfg.USE_CLASS:
            self.classifier_cls_score_net = nn.Linear(self._fc7_channels, cfg.NUM_CLASSES)
            self.classifier_bbox_pred_net = nn.Linear(self._fc7_channels, cfg.NUM_CLASSES * 6)

        if cfg.USE_MASK:
            method_to_call = getattr(backbones, cfg.MASK_BACKBONE)
            self.mask_backbone = method_to_call()

        if cfg.USE_IMAGES and not cfg.USE_IMAGES_GT:
            self.image_enet_fixed, self.image_enet_trainable, self.image_enet_classification = enet.create_enet_for_3d(cfg.NUM_2D_CLASSES, cfg.PRETRAINED_ENET_PATH, cfg.NUM_CLASSES)

    def delete_intermediate_states(self):
        # Delete intermediate result to save memory
        for d in [self._losses, self._predictions, self._anchor_targets, self._proposal_targets, self._mask_targets]:
            for k in list(d):
                del d[k]

    def forward(self, blobs, mode='TRAIN', killing_inds=None):
        self._scene_info = blobs['data'].shape[2:]
        self._id = blobs['id'][0]
        self.cuda() 
        self.batch_size = blobs['data'].shape[0]

        if mode == 'TRAIN':
            self.train()
            if cfg.USE_IMAGES and not cfg.USE_IMAGES_GT:
                # eval of enet
                self.image_enet_fixed.eval()
                self.image_enet_trainable.eval()
            self._mode = 'TRAIN'
            self._scene = Variable(blobs['data'].cuda())
            self._gt_bbox = blobs['gt_box']
            self._gt_mask = blobs['gt_mask'] if cfg.USE_MASK else None


            if cfg.USE_IMAGES:
                grid_shape = blobs['data'].shape[-3:]
                self._imageft = []
                for i in range(self.batch_size):
                    num_images = blobs['nearest_images']['images'][i].shape[0]
                    if cfg.USE_IMAGES_GT:
                        imageft = Variable(blobs['nearest_images']['images'][i].cuda())
                        #imageft = imageft.expand(imageft.shape[0], 128, imageft.shape[2], imageft.shape[3]).contiguous()

                    else:
                        imageft = self.image_enet_fixed(Variable(blobs['nearest_images']['images'][i].cuda()))
                        imageft = self.image_enet_trainable(imageft)

                    proj3d = Variable(blobs['proj_ind_3d'][i].cuda())
                    proj2d = Variable(blobs['proj_ind_2d'][i].cuda())

                    # project 2d to 3d
                    imageft = [Projection.apply(ft, ind3d, ind2d, grid_shape) for ft, ind3d, ind2d in zip(imageft, proj3d, proj2d)]
                    imageft = torch.stack(imageft, dim=4)
                    # reshape to max pool over features
                    sz = imageft.shape
                    imageft = imageft.view(sz[0], -1, num_images)
                    imageft = torch.nn.MaxPool1d(kernel_size=num_images)(imageft)
                    imageft = imageft.view(sz[0], sz[1], sz[2], sz[3], 1)
                    self._imageft.append(imageft.permute(4, 0, 3, 2, 1))
                self._imageft = torch.cat(self._imageft, 0)


            #--------------------------
            # visualization snippets
            #-------------------------
            #import ipdb
            #ipdb.set_trace()
            #data = np.where(self._scene[0,0].data.cpu().numpy() <=1.0, 1, 0)
            #data = self._imageft[0]
            #write_mask(data, 'data.ply')
            #data = blobs['gt_box'][0].numpy()
            #write_bbox(data, 'bbox.ply')

            if cfg.USE_BACKBONE:
                net_conv_level1, net_conv_level2, net_conv_level3 = self._backbone()

            if cfg.USE_RPN:
                # build the anchors for the scene
                if cfg.FIRST_TIME_ANCHORS:
                    cfg.FIRST_TIME_ANCHORS = False
                    # build the anchors for the scene
                    if cfg.NUM_ANCHORS_LEVEL1 != 0:
                        size_level1 = [net_conv_level1.size(2), net_conv_level1.size(3), net_conv_level1.size(4)]
                    if cfg.NUM_ANCHORS_LEVEL2 != 0:
                        size_level2 = [net_conv_level2.size(2), net_conv_level2.size(3), net_conv_level2.size(4)]
                    if cfg.NUM_ANCHORS_LEVEL3 != 0:
                        size_level3 = [net_conv_level3.size(2), net_conv_level3.size(3), net_conv_level3.size(4)]

                    self._anchor_component(size_level1 if cfg.NUM_ANCHORS_LEVEL1 !=0 else [],
                                           size_level2 if cfg.NUM_ANCHORS_LEVEL2 !=0 else [],
                                           size_level3 if cfg.NUM_ANCHORS_LEVEL3 !=0 else [])

                self._region_proposal(net_conv_level1, net_conv_level2, net_conv_level3)
            else:
                # only predictions['rois']/['roi_scores']/['mask_pred'] batch is a list, since not even number/dim in each sample
                self._predictions['rois'] = [self._gt_bbox[i][:,:6].cuda() for i in range(self.batch_size)]
                self._predictions['roi_scores'] = [torch.ones(self._gt_bbox[i].size(0), 1).cuda() for i in range(self.batch_size)]

            if cfg.USE_CLASS:
                self._proposal_target_layer(self._predictions['rois'], self._predictions['roi_scores'], self._predictions['level_inds'])
                pool5 = self._roi_pool_layer(net_conv_level1, net_conv_level2, net_conv_level3, 
                                             self._proposal_targets['rois'], self._proposal_targets['levelInds'],
                                             self._feat_stride, cfg.CLASS_POOLING_SIZE)
                fc7 = self._classifier(pool5)
                self._region_classification(fc7)
            else:
                self._predictions["cls_pred"] = Variable(self._gt_bbox[0][:,6].long())
                self._predictions["cls_prob"] = Variable(torch.zeros((self._predictions['cls_pred'].shape[0], cfg.NUM_CLASSES)))
                self._predictions["bbox_pred"] = Variable(torch.zeros((self._predictions['cls_pred'].shape[0], cfg.NUM_CLASSES*6)))
                for ind_sample in range(self._predictions['cls_pred'].shape[0]):
                    self._predictions['cls_prob'][ind_sample, self._predictions['cls_pred'].data[ind_sample]] = 1.0

            if cfg.USE_MASK:
                self._mask_target_layer(self._predictions['rois'])
                mask_pred_batch = []
                for i in range(self.batch_size):
                    mask_pred = []
                    for roi in self._mask_targets['rois'][i]:
                        mask_pred.append(self.mask_backbone(self._scene[i:i+1, :, 
                                                                        int(round(roi[0].item())):int(round(roi[3].item())),
                                                                        int(round(roi[1].item())):int(round(roi[4].item())), 
                                                                        int(round(roi[2].item())):int(round(roi[5].item()))
                                                                        ], self._imageft[i:i+1, :, 
                                                                                         int(round(roi[0].item())):int(round(roi[3].item())),
                                                                                         int(round(roi[1].item())):int(round(roi[4].item())), 
                                                                                         int(round(roi[2].item())):int(round(roi[5].item()))] if cfg.USE_IMAGES else None))

                    mask_pred_batch.append(mask_pred)
                self._predictions['mask_pred'] = mask_pred_batch
            self._add_losses()

        elif mode == 'TEST':
            with torch.no_grad():
                self.eval()
                self._mode = 'TEST'
                self._scene = blobs['data'].cuda()
                self._gt_bbox = blobs['gt_box']
                self._gt_mask = blobs['gt_mask'] if cfg.USE_MASK else None
                if cfg.USE_IMAGES:
                    grid_shape = blobs['data'].shape[-3:]
                    self._imageft = []
                    for i in range(self.batch_size):
                        num_images = blobs['nearest_images']['images'][i].shape[0]
                        if cfg.USE_IMAGES_GT:
                            with torch.no_grad():
                                imageft = Variable(blobs['nearest_images']['images'][i].cuda())
                        else:
                            with torch.no_grad():
                                imageft = self.image_enet_fixed(Variable(blobs['nearest_images']['images'][i].cuda()))
                            imageft = self.image_enet_trainable(imageft)

                        proj3d = Variable(blobs['proj_ind_3d'][i])
                        proj2d = Variable(blobs['proj_ind_2d'][i])

                        if blobs['data'].shape[2]*blobs['data'].shape[3]*blobs['data'].shape[4] > cfg.MAX_VOLUME or len(proj3d) > cfg.MAX_IMAGE:
                            print('on cpu')
                            imageft = imageft.cpu()
                            proj3d = proj3d.cpu()
                            proj2d = proj2d.cpu()

                        # project 2d to 3d
                        counter = 0
                        init = True

                        for ft, ind3d, ind2d in zip(imageft, proj3d, proj2d):
                            counter += 1
                            if counter-1 in killing_inds:
                                continue
                            imageft_temp = Projection.apply(ft, ind3d, ind2d, grid_shape)[:, :,:, :].contiguous()
                            sz = imageft_temp.shape
                            if init:
                                imageft = imageft_temp.view(sz[0], sz[1], sz[2], sz[3])
                                init = False
                                continue

                            imageft = torch.stack([imageft, imageft_temp], dim=4)
                            # reshape to max pool over features
                            imageft = imageft.view(sz[0], -1, 2)
                            imageft = torch.nn.MaxPool1d(kernel_size=2)(imageft)
                            imageft = imageft.view(sz[0], sz[1], sz[2], sz[3])

                        imageft = imageft.view(sz[0], sz[1], sz[2], sz[3], self.batch_size)
                        self._imageft = imageft.permute(4, 0, 3, 2, 1)
                        self._imageft = self._imageft.cuda()
                        del proj3d
                        del proj2d
                        torch.cuda.empty_cache()

                if cfg.USE_BACKBONE:
                    net_conv_level1, net_conv_level2, net_conv_level3 = self._backbone()

                if cfg.USE_RPN:
                    # build the anchors for the scene
                    if cfg.NUM_ANCHORS_LEVEL1 != 0:
                        size_level1 = [net_conv_level1.size(2), net_conv_level1.size(3), net_conv_level1.size(4)]
                    if cfg.NUM_ANCHORS_LEVEL2 != 0:
                        size_level2 = [net_conv_level2.size(2), net_conv_level2.size(3), net_conv_level2.size(4)]
                    if cfg.NUM_ANCHORS_LEVEL3 != 0:
                        size_level3 = [net_conv_level3.size(2), net_conv_level3.size(3), net_conv_level3.size(4)]

                    self._anchor_component(size_level1 if cfg.NUM_ANCHORS_LEVEL1 !=0 else [],
                                           size_level2 if cfg.NUM_ANCHORS_LEVEL2 !=0 else [],
                                           size_level3 if cfg.NUM_ANCHORS_LEVEL3 !=0 else [])

                    self._region_proposal(net_conv_level1, net_conv_level2, net_conv_level3)

                else:
                    # only predictions['rois']/['roi_scores'] batch is a list, since not even number in each sample
                    self._predictions['rois'] = [self._gt_bbox[i][:,:6].cuda() for i in range(self.batch_size)]
                    self._predictions['roi_scores'] = [torch.ones(self._gt_bbox[i].size(0), 1).cuda() for i in range(self.batch_size)]

                # especially for validation, since we don't want to resample in val for mAP

                if cfg.USE_CLASS:
                    pool5 = self._roi_pool_layer(net_conv_level1, net_conv_level2, net_conv_level3, 
                                                 Variable(torch.cat(self._predictions['rois'], 0)), 
                                                 Variable(torch.cat(self._predictions['level_inds'], 0)),
                                                 self._feat_stride, cfg.CLASS_POOLING_SIZE)
                    fc7 = self._classifier(pool5)
                    self._region_classification(fc7)
                else:
                    self._predictions["cls_pred"] = Variable(self._gt_bbox[0][:,6].long())
                    self._predictions["cls_prob"] = Variable(torch.zeros((self._predictions['cls_pred'].shape[0], cfg.NUM_CLASSES)))
                    self._predictions["bbox_pred"] = Variable(torch.zeros((self._predictions['cls_pred'].shape[0], cfg.NUM_CLASSES*6)))
                    for ind_sample in range(self._predictions['cls_pred'].shape[0]):
                        self._predictions['cls_prob'][ind_sample, self._predictions['cls_pred'].data[ind_sample]] = 1.0

                if cfg.USE_MASK:
                    mask_pred_batch = []
                    rois = self._predictions['rois'][0].cpu()
                    box_reg_pre = self._predictions["bbox_pred"].data.cpu().numpy()
                    box_reg = np.zeros((box_reg_pre.shape[0], 6))
                    pred_class = self._predictions['cls_pred'].data.cpu().numpy()
                    pred_conf = np.zeros((pred_class.shape[0]))
                    for pred_ind in range(pred_class.shape[0]):
                        box_reg[pred_ind, :] = box_reg_pre[pred_ind, pred_class[pred_ind]*6:(pred_class[pred_ind]+1)*6]
                        pred_conf[pred_ind] = self._predictions['cls_prob'].data.cpu().numpy()[pred_ind, pred_class.data[pred_ind]]
                    pred_box = bbox_transform_inv(rois, torch.from_numpy(box_reg).float())
                    pred_box = clip_boxes(pred_box, self._scene_info[:3]).numpy()

                    sort_index = pred_conf > cfg.CLASS_THRESH

                    # eliminate bad box
                    for idx, box in enumerate(pred_box):
                        if round(box[0]) >= round(box[3]) or round(box[1]) >= round(box[4]) or round(box[2]) >= round(box[5]):
                            sort_index[idx] = False
                    
                    for i in range(self.batch_size):
                        mask_pred = []
                        for ind, roi in enumerate(pred_box):
                            if sort_index[ind]:
                                mask_pred.append(self.mask_backbone(self._scene[i:i+1, :, 
                                                                                int(round(roi[0])):int(round(roi[3])),
                                                                                int(round(roi[1])):int(round(roi[4])), 
                                                                                int(round(roi[2])):int(round(roi[5]))
                                                                                ], self._imageft[i:i+1, :, 
                                                                                                 int(round(roi[0])):int(round(roi[3])),
                                                                                                 int(round(roi[1])):int(round(roi[4])), 
                                                                                                 int(round(roi[2])):int(round(roi[5]))] if cfg.USE_IMAGES else None))

                        mask_pred_batch.append(mask_pred)
                    self._predictions['mask_pred'] = mask_pred_batch

    def _add_losses(self, sigma_rpn=3.0):

        loss = Variable(torch.zeros(1).cuda())
#        for name, var in self.named_parameters():
#            print(name, var.requires_grad)

        if not cfg.FIX_RPN:
            if cfg.NUM_ANCHORS_LEVEL1 != 0:
                #---------------------
                # level 1
                #---------------------
                # RPN, class loss
                rpn_cls_score_level1 = self._predictions['rpn_cls_score_level1'] #torch.Size([1, 2, 10, 5, 10, 9])
                rpn_label_level1 = self._anchor_targets['rpn_labels_level1'] #torch.Size([1, 10, 5, 10, 9])
                rpn_select_level1 = (rpn_label_level1.data != -1).nonzero()

                if rpn_select_level1.numel() != 0:
                    #TODO advanced indexing
                    rpn_cls_score_reshape_level1 = []
                    rpn_label_reshape_level1 = []
                    for i in rpn_select_level1:
                        rpn_cls_score_reshape_level1.append(rpn_cls_score_level1[i[0], :, i[1], i[2], i[3], i[4]])
                        rpn_label_reshape_level1.append(rpn_label_level1[i[0], i[1], i[2], i[3], i[4]])

                    rpn_cls_score_reshape_level1 = torch.stack(rpn_cls_score_reshape_level1, 0)
                    rpn_label_reshape_level1 = torch.stack(rpn_label_reshape_level1, 0)
                    rpn_cross_entropy_level1 = F.cross_entropy(rpn_cls_score_reshape_level1, rpn_label_reshape_level1)
                    self._losses['rpn_cross_entropy_level1'] = rpn_cross_entropy_level1

                    #RPN, bbox loss
                    rpn_bbox_pred_level1 = self._predictions['rpn_bbox_pred_level1']
                    rpn_bbox_targets_level1 = self._anchor_targets['rpn_bbox_targets_level1']
                    rpn_bbox_inside_weights_level1 = self._anchor_targets['rpn_bbox_inside_weights_level1']
                    rpn_bbox_outside_weights_level1 = self._anchor_targets['rpn_bbox_outside_weights_level1']
                    rpn_loss_box_level1 = self._smooth_l1_loss(rpn_bbox_pred_level1, rpn_bbox_targets_level1, 
                                                        rpn_bbox_inside_weights_level1, rpn_bbox_outside_weights_level1, 
                                                        sigma=2.0, dim=[1,2,3,4])
                    self._losses['rpn_loss_box_level1'] = rpn_loss_box_level1
                    loss += rpn_cross_entropy_level1 + rpn_loss_box_level1
                else:
                    self._losses['rpn_cross_entropy_level1'] = Variable(torch.FloatTensor([0.0]))
                    self._losses['rpn_loss_box_level1'] = Variable(torch.FloatTensor([0.0]))

            if cfg.NUM_ANCHORS_LEVEL2 != 0:
                #---------------------
                # level 2
                #---------------------
                # RPN, class loss
                rpn_cls_score_level2 = self._predictions['rpn_cls_score_level2'] #torch.Size([1, 2, 10, 5, 10, 9])
                rpn_label_level2 = self._anchor_targets['rpn_labels_level2'] #torch.Size([1, 10, 5, 10, 9])
                rpn_select_level2 = (rpn_label_level2.data != -1).nonzero()
                if rpn_select_level2.numel() != 0:
                    #TODO advanced indexing
                    rpn_cls_score_reshape_level2 = []
                    rpn_label_reshape_level2 = []
                    for i in rpn_select_level2:
                        rpn_cls_score_reshape_level2.append(rpn_cls_score_level2[i[0], :, i[1], i[2], i[3], i[4]])
                        rpn_label_reshape_level2.append(rpn_label_level2[i[0], i[1], i[2], i[3], i[4]])

                    rpn_cls_score_reshape_level2 = torch.stack(rpn_cls_score_reshape_level2, 0)
                    rpn_label_reshape_level2 = torch.stack(rpn_label_reshape_level2, 0)
                    rpn_cross_entropy_level2 = F.cross_entropy(rpn_cls_score_reshape_level2, rpn_label_reshape_level2)
                    self._losses['rpn_cross_entropy_level2'] = rpn_cross_entropy_level2

                    rpn_bbox_pred_level2 = self._predictions['rpn_bbox_pred_level2']
                    rpn_bbox_targets_level2 = self._anchor_targets['rpn_bbox_targets_level2']
                    rpn_bbox_inside_weights_level2 = self._anchor_targets['rpn_bbox_inside_weights_level2']
                    rpn_bbox_outside_weights_level2 = self._anchor_targets['rpn_bbox_outside_weights_level2']
                    rpn_loss_box_level2 = self._smooth_l1_loss(rpn_bbox_pred_level2, rpn_bbox_targets_level2, 
                                                        rpn_bbox_inside_weights_level2, rpn_bbox_outside_weights_level2, 
                                                        sigma=2.0, dim=[1,2,3,4])
                    self._losses['rpn_loss_box_level2'] = rpn_loss_box_level2
                    loss += rpn_cross_entropy_level2 + rpn_loss_box_level2
                else:
                    self._losses['rpn_cross_entropy_level2'] = Variable(torch.FloatTensor([0.0]))
                    self._losses['rpn_loss_box_level2'] = Variable(torch.FloatTensor([0.0]))

            if cfg.NUM_ANCHORS_LEVEL3 != 0:
                #---------------------
                # level 3
                #---------------------
                # RPN, class loss
                rpn_cls_score_level3 = self._predictions['rpn_cls_score_level3'] #torch.Size([1, 2, 10, 5, 10, 9])
                rpn_label_level3 = self._anchor_targets['rpn_labels_level3'] #torch.Size([1, 10, 5, 10, 9])
                rpn_select_level3 = (rpn_label_level3.data != -1).nonzero()
                if rpn_select_level3.numel() != 0:
                    #TODO advanced indexing
                    rpn_cls_score_reshape_level3 = []
                    rpn_label_reshape_level3 = []
                    for i in rpn_select_level3:
                        rpn_cls_score_reshape_level3.append(rpn_cls_score_level3[i[0], :, i[1], i[2], i[3], i[4]])
                        rpn_label_reshape_level3.append(rpn_label_level3[i[0], i[1], i[2], i[3], i[4]])

                    rpn_cls_score_reshape_level3 = torch.stack(rpn_cls_score_reshape_level3, 0)
                    rpn_label_reshape_level3 = torch.cat(rpn_label_reshape_level3, 0)
                    rpn_cross_entropy_level3 = F.cross_entropy(rpn_cls_score_reshape_level3, rpn_label_reshape_level3)
                    self._losses['rpn_cross_entropy_level3'] = rpn_cross_entropy_level3

                    rpn_bbox_pred_level3 = self._predictions['rpn_bbox_pred_level3']
                    rpn_bbox_targets_level3 = self._anchor_targets['rpn_bbox_targets_level3']
                    rpn_bbox_inside_weights_level3 = self._anchor_targets['rpn_bbox_inside_weights_level3']
                    rpn_bbox_outside_weights_level3 = self._anchor_targets['rpn_bbox_outside_weights_level3']
                    rpn_loss_box_level3 = self._smooth_l1_loss(rpn_bbox_pred_level3, rpn_bbox_targets_level3, 
                                                        rpn_bbox_inside_weights_level3, rpn_bbox_outside_weights_level3, 
                                                        sigma=2.0, dim=[1,2,3,4])
                    self._losses['rpn_loss_box_level3'] = rpn_loss_box_level3
                    loss += rpn_cross_entropy_level3 + rpn_loss_box_level3
                else:
                    self._losses['rpn_cross_entropy_level3'] = Variable(torch.FloatTensor([0.0]))
                    self._losses['rpn_loss_box_level3'] = Variable(torch.FloatTensor([0.0]))

        else:
            self._losses['rpn_loss_box_level1'] = Variable(torch.FloatTensor([0.0]))
            self._losses['rpn_cross_entropy_level1'] = Variable(torch.FloatTensor([0.0]))
            self._losses['rpn_loss_box_level2'] = Variable(torch.FloatTensor([0.0]))
            self._losses['rpn_cross_entropy_level2'] = Variable(torch.FloatTensor([0.0]))
            self._losses['rpn_loss_box_level3'] = Variable(torch.FloatTensor([0.0]))
            self._losses['rpn_cross_entropy_level3'] = Variable(torch.FloatTensor([0.0]))

        if not cfg.FIX_CLASS or cfg.NYUV2_FINETUNE:
            #RCNN, class loss
            cls_score = self._predictions['cls_score']
            label = self._proposal_targets['labels'].view(-1)
            normalize_weights = torch.FloatTensor(cfg.NORMALIZE_WEIGHTS).cuda() 
            cross_entropy = F.cross_entropy(cls_score, label, weight=normalize_weights, size_average=True, reduce=True)
            self._losses['cross_entropy'] = cross_entropy
            loss += cross_entropy

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, 
                                            bbox_outside_weights, sigma=1.0, dim=[1])
            self._losses['loss_box'] = loss_box
            loss += loss_box

        else:
            self._losses['loss_box'] = Variable(torch.FloatTensor([0.0]))
            self._losses['cross_entropy'] = Variable(torch.FloatTensor([0.0]))

        if cfg.USE_MASK:
            normalize_weights = torch.FloatTensor(cfg.NORMALIZE_WEIGHTS).cuda() 
            normalize_weights[0] = 0.0

            loss_mask = Variable(torch.zeros(1).cuda())
            counter = 0

            mask_preds = self._predictions['mask_pred']
            mask_targets = self._mask_targets['masks']
            mask_labels = self._mask_targets['labels']

            for i in range(self.batch_size):
                for mask_pred, mask_target, mask_label in zip(mask_preds[i], mask_targets[i], mask_labels[i]):
                    loss_mask += F.binary_cross_entropy_with_logits(mask_pred[0, mask_label], Variable(mask_target.float().cuda())) * normalize_weights[mask_label]
                    counter += normalize_weights[mask_label] != 0.0

            if counter != 0:
                self._losses['loss_mask'] = loss_mask / counter.item()
                loss += loss_mask / counter.item()
            else:
                self._losses['loss_mask'] = loss_mask

        self._losses['total_loss'] = loss

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights,
                        bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff

        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
        in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign\
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = out_loss_box
        for i in sorted(dim, reverse=True):
            loss_box = loss_box.sum(i)
        loss_box = loss_box.mean()

        return loss_box

    def _roi_pool_layer(self, bottom1, bottom2, bottom3, rois, levelInds, feat_stride=None, pool_size=None):
        batch_size = int(rois.shape[0] / self.batch_size)
        return_pool5 = Variable(torch.zeros(rois.shape[0], bottom1.shape[1], pool_size, pool_size, pool_size)).cuda()
        for i in range(self.batch_size):
            if cfg.NUM_ANCHORS_LEVEL1 != 0:
                inds_level1 = (levelInds[i*batch_size:(i+1)*batch_size] == 1).nonzero()
                if inds_level1.numel() != 0:
                    rois_level1 = rois[i*batch_size:(i+1)*batch_size][inds_level1[:,0]]
                    rois_level1 = RoIPoolFunction(pool_size, pool_size, pool_size, 1.0 / feat_stride[0])(bottom1[i:i+1], rois_level1)
                    #TODO batch size > 1
                    for counter, ind in enumerate(inds_level1):
                        return_pool5[ind.data[0]] = rois_level1[counter]

            if cfg.NUM_ANCHORS_LEVEL2 != 0:
                inds_level2 = (levelInds[i*batch_size:(i+1)*batch_size] == 2).nonzero()
                if inds_level2.numel() != 0:
                    rois_level2 = rois[i*batch_size:(i+1)*batch_size][inds_level2[:,0]]
                    rois_level2 = RoIPoolFunction(pool_size, pool_size, pool_size, 1.0 / feat_stride[1])(bottom2[i:i+1], rois_level2)
                    #TODO batch size > 1
                    for counter, ind in enumerate(inds_level2):
                        return_pool5[ind.data[0]] = rois_level2[counter]

            if cfg.NUM_ANCHORS_LEVEL3 != 0:
                inds_level3 = (levelInds[i*batch_size:(i+1)*batch_size] == 3).nonzero()
                if inds_level3.numel() != 0:
                    rois_level3 = rois[i*batch_size:(i+1)*batch_size][inds_level3[:,0]]
                    rois_level3 = RoIPoolFunction(pool_size, pool_size, pool_size, 1.0 / feat_stride[2])(bottom3[i:i+1], rois_level3)
                    #TODO batch size > 1
                    for counter, ind in enumerate(inds_level3):
                        return_pool5[ind.data[0]] = rois_level3[counter]

        return return_pool5


    def _region_proposal(self, net_conv_level1, net_conv_level2, net_conv_level3):
        if cfg.NUM_ANCHORS_LEVEL1 != 0:
            rpn_level1 = F.relu(self.rpn_net_level1(net_conv_level1))
            # batch x w x h x l x (num_anchors x 6)
            rpn_bbox_pred_level1 = self.rpn_bbox_pred_net_level1(rpn_level1).permute(0, 2, 3, 4, 1).contiguous()
            # batch x 2 x w x h x l x num_anchors
            rpn_cls_score_level1 = self.rpn_cls_score_net_level1(rpn_level1).view(self.batch_size, 2, cfg.NUM_ANCHORS_LEVEL1, rpn_bbox_pred_level1.size(1), rpn_bbox_pred_level1.size(2), rpn_bbox_pred_level1.size(3)).permute(0, 1, 3, 4, 5, 2).contiguous()

            # batch x 2 x w x h x l x num_anchors
            rpn_cls_prob_level1 = F.softmax(rpn_cls_score_level1) 
            self._predictions["rpn_cls_score_level1"] = rpn_cls_score_level1
            self._predictions["rpn_cls_prob_level1"] = rpn_cls_prob_level1
            self._predictions["rpn_bbox_pred_level1"] = rpn_bbox_pred_level1

        if cfg.NUM_ANCHORS_LEVEL2 != 0:
            rpn_level2 = F.relu(self.rpn_net_level2(net_conv_level2))
            # batch x w x h x l x (num_anchors x 6)
            rpn_bbox_pred_level2 = self.rpn_bbox_pred_net_level2(rpn_level2).permute(0, 2, 3, 4, 1).contiguous()
            # batch x 2 x w x h x l x num_anchors
            rpn_cls_score_level2 = self.rpn_cls_score_net_level2(rpn_level2).view(self.batch_size, 2, cfg.NUM_ANCHORS_LEVEL2, rpn_bbox_pred_level2.size(1), rpn_bbox_pred_level2.size(2), rpn_bbox_pred_level2.size(3)).permute(0, 1, 3, 4, 5, 2).contiguous()

            # batch x 2 x w x h x l x num_anchors
            rpn_cls_prob_level2 = F.softmax(rpn_cls_score_level2) 
            self._predictions["rpn_cls_score_level2"] = rpn_cls_score_level2
            self._predictions["rpn_cls_prob_level2"] = rpn_cls_prob_level2
            self._predictions["rpn_bbox_pred_level2"] = rpn_bbox_pred_level2

        if cfg.NUM_ANCHORS_LEVEL3 != 0:
            rpn_level3 = F.relu(self.rpn_net_level3(net_conv_level3))
            # batch x w x h x l x (num_anchors x 6)
            rpn_bbox_pred_level3 = self.rpn_bbox_pred_net_level3(rpn_level3).permute(0, 2, 3, 4, 1).contiguous()
            # batch x 2 x w x h x l x num_anchors
            rpn_cls_score_level3 = self.rpn_cls_score_net_level3(rpn_level3).view(self.batch_size, 2, cfg.NUM_ANCHORS_LEVEL3, rpn_bbox_pred_level3.size(1), rpn_bbox_pred_level3.size(2), rpn_bbox_pred_level3.size(3)).permute(0, 1, 3, 4, 5, 2).contiguous()
            # batch x 2 x w x h x l x num_anchors
            rpn_cls_prob_level3 = F.softmax(rpn_cls_score_level3) 
            self._predictions["rpn_cls_score_level3"] = rpn_cls_score_level3
            self._predictions["rpn_cls_prob_level3"] = rpn_cls_prob_level3
            self._predictions["rpn_bbox_pred_level3"] = rpn_bbox_pred_level3

        if self._mode == 'TRAIN':
            self._anchor_target_layer(
                    [*rpn_cls_score_level1.shape[2:5]] if cfg.NUM_ANCHORS_LEVEL1 != 0 else None, 
                    [*rpn_cls_score_level2.shape[2:5]] if cfg.NUM_ANCHORS_LEVEL2 != 0 else None,
                    [*rpn_cls_score_level3.shape[2:5]] if cfg.NUM_ANCHORS_LEVEL3 != 0 else None)

        self._proposal_layer(rpn_cls_prob_level1 if cfg.NUM_ANCHORS_LEVEL1 != 0 else None,
                             rpn_bbox_pred_level1 if cfg.NUM_ANCHORS_LEVEL1 !=0 else None,
                             rpn_cls_prob_level2 if cfg.NUM_ANCHORS_LEVEL2 !=0 else None,
                             rpn_bbox_pred_level2 if cfg.NUM_ANCHORS_LEVEL2 !=0 else None,
                             rpn_cls_prob_level3 if cfg.NUM_ANCHORS_LEVEL3 !=0 else None,
                             rpn_bbox_pred_level3 if cfg.NUM_ANCHORS_LEVEL3 !=0 else None)

    def _region_classification(self, fc7):
        """

        :param fc7:
        :return:
        """
        cls_score = self.classifier_cls_score_net(fc7)
        cls_pred = torch.max(cls_score, 1)[1]
        cls_prob = F.softmax(cls_score)

        bbox_pred = self.classifier_bbox_pred_net(fc7)

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"]  = cls_pred
        self._predictions["cls_prob"]  = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

    def _anchor_target_layer(self, feat_size_level1, feat_size_level2, feat_size_level3):
        rpn_labels_level1, rpn_bbox_targets_level1, rpn_bbox_inside_weights_level1, rpn_bbox_outsise_weights_level1, \
        rpn_labels_level2, rpn_bbox_targets_level2, rpn_bbox_inside_weights_level2, rpn_bbox_outsise_weights_level2, \
        rpn_labels_level3, rpn_bbox_targets_level3, rpn_bbox_inside_weights_level3, rpn_bbox_outsise_weights_level3, \
        = anchor_target_layer(self._anchors_level1 if cfg.NUM_ANCHORS_LEVEL1 != 0 else None, feat_size_level1, 
                              self._anchors_level2 if cfg.NUM_ANCHORS_LEVEL2 != 0 else None, feat_size_level2, 
                              self._anchors_level3 if cfg.NUM_ANCHORS_LEVEL3 != 0 else None, feat_size_level3, 
                              self._gt_bbox, self._scene_info, 
                              self._anchors_filter_level1[self._id] if cfg.FILTER_ANCHOR_LEVEL1 else None,
                              self._anchors_filter_level2[self._id] if cfg.FILTER_ANCHOR_LEVEL2 else None,
                              self._anchors_filter_level3[self._id] if cfg.FILTER_ANCHOR_LEVEL3 else None,
                              )

        if cfg.NUM_ANCHORS_LEVEL1 != 0:
            self._anchor_targets['rpn_labels_level1']               = Variable(rpn_labels_level1.long().cuda())
            self._anchor_targets['rpn_bbox_targets_level1']         = Variable(rpn_bbox_targets_level1.float().cuda())
            self._anchor_targets['rpn_bbox_inside_weights_level1']  = Variable(rpn_bbox_inside_weights_level1.float().cuda())
            self._anchor_targets['rpn_bbox_outside_weights_level1'] = Variable(rpn_bbox_outsise_weights_level1.float().cuda())

        if cfg.NUM_ANCHORS_LEVEL2 != 0:
            self._anchor_targets['rpn_labels_level2']               = Variable(rpn_labels_level2.long().cuda())
            self._anchor_targets['rpn_bbox_targets_level2']         = Variable(rpn_bbox_targets_level2.float().cuda())
            self._anchor_targets['rpn_bbox_inside_weights_level2']  = Variable(rpn_bbox_inside_weights_level2.float().cuda())
            self._anchor_targets['rpn_bbox_outside_weights_level2'] = Variable(rpn_bbox_outsise_weights_level2.float().cuda())

        if cfg.NUM_ANCHORS_LEVEL3 != 0:
            self._anchor_targets['rpn_labels_level3']               = Variable(rpn_labels_level3.long().cuda())
            self._anchor_targets['rpn_bbox_targets_level3']         = Variable(rpn_bbox_targets_level3.float().cuda())
            self._anchor_targets['rpn_bbox_inside_weights_level3']  = Variable(rpn_bbox_inside_weights_level3.float().cuda())
            self._anchor_targets['rpn_bbox_outside_weights_level3'] = Variable(rpn_bbox_outsise_weights_level3.float().cuda())


    def _proposal_target_layer(self, rois, roi_scores, levelInds):
        rois, roi_scores, labels, levelInds, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
        proposal_target_layer([roi.cpu() for roi in rois], [roi_score.cpu() for roi_score in roi_scores], [levelInd.cpu() for levelInd in levelInds],
                              self._gt_bbox, self._mode)

        self._proposal_targets['rois']                 = Variable(rois.cuda())
        self._proposal_targets['labels']               = Variable(labels.cuda())
        self._proposal_targets['levelInds']            = Variable(levelInds.cuda())
        self._proposal_targets['bbox_targets']         = Variable(bbox_targets.cuda())
        self._proposal_targets['bbox_inside_weights']  = Variable(bbox_inside_weights.cuda())
        self._proposal_targets['bbox_outside_weights'] = Variable(bbox_outside_weights.cuda())

    def _mask_target_layer(self, rois):
        mask_rois, mask_targets, mask_labels = mask_target_layer(
            rois, self._gt_bbox, gt_mask=self._gt_mask, scene_info=self._scene_info, scene=self._scene)
        self._mask_targets['rois']   = mask_rois
        self._mask_targets['masks']  = mask_targets #will make it Variable in _add_loss
        self._mask_targets['labels'] = mask_labels

    def _proposal_layer(self, rpn_cls_prob_level1, rpn_bbox_pred_level1, 
                              rpn_cls_prob_level2, rpn_bbox_pred_level2,
                              rpn_cls_prob_level3, rpn_bbox_pred_level3):
        """

        :param rpn_cls_prob:
        :param rpn_bbox_pred:
        :return:
        """
        rois, roi_scores, levelInds = proposal_layer(rpn_cls_prob_level1.data if cfg.NUM_ANCHORS_LEVEL1 != 0 else None,
                                          rpn_bbox_pred_level1.data if cfg.NUM_ANCHORS_LEVEL1 != 0 else None,
                                          self._anchors_level1.cuda() if cfg.NUM_ANCHORS_LEVEL1 != 0 else None,
                                          rpn_cls_prob_level2.data if cfg.NUM_ANCHORS_LEVEL2 != 0 else None,
                                          rpn_bbox_pred_level2.data if cfg.NUM_ANCHORS_LEVEL2 != 0 else None, 
                                          self._anchors_level2.cuda() if cfg.NUM_ANCHORS_LEVEL2 != 0 else None, 
                                          rpn_cls_prob_level3.data if cfg.NUM_ANCHORS_LEVEL3 != 0 else None,
                                          rpn_bbox_pred_level3.data if cfg.NUM_ANCHORS_LEVEL3 != 0 else None, 
                                          self._anchors_level3.cuda() if cfg.NUM_ANCHORS_LEVEL3 != 0 else None, 
                                          self._scene_info, self._mode, 
                                          self._anchors_filter_level1[self._id] if cfg.FILTER_ANCHOR_LEVEL1 else None, 
                                          self._anchors_filter_level2[self._id] if cfg.FILTER_ANCHOR_LEVEL2 else None,
                                          self._anchors_filter_level3[self._id] if cfg.FILTER_ANCHOR_LEVEL3 else None
                                          )

        self._predictions['rois'] = rois
        self._predictions['roi_scores'] = roi_scores
        self._predictions['level_inds'] = levelInds

    def _anchor_component(self, size_level1, size_level2, size_level3):
        anchors_level1, anchors_level2, anchors_level3 = generate_anchors(size_level1, size_level2, size_level3, self._feat_stride)

        if cfg.NUM_ANCHORS_LEVEL1 != 0:
            self._anchors_level1 = torch.from_numpy(anchors_level1)

        if cfg.NUM_ANCHORS_LEVEL2 != 0:
            self._anchors_level2 = torch.from_numpy(anchors_level2)

        if cfg.NUM_ANCHORS_LEVEL3 != 0:
            self._anchors_level3 = torch.from_numpy(anchors_level3)

    def _init_backbone_classifier(self):
        raise NotImplementedError
  
    def _classifer(self, pool5):
        raise NotImplementedError

    def _backbone(self):
        raise NotImplementedError


