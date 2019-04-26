import numpy as np
import skimage.transform as sktf
import torch
import math
from torch.autograd import Variable

from lib.utils.overlap import bbox_overlap
from lib.utils.bbox_transform import bbox_transform
from lib.utils.config import cfg
import lib


def proposal_target_layer(rpn_rois, rpn_scores, rpn_levelInds, gt_bbox, mode='TRAIN'):
    """
    Assign object detection proposals to ground-truth targets. Produce proposal
    classification labels and bounding box regression targets

    :param rpn_rois:
    :param rpn_scores:
    :param gt_bbox: (num, 7)
    :return:
    """
    rois_batch = []
    scores_batch = []
    labels_batch = []
    levelInds_batch = []
    bbox_targets_batch = []
    bbox_inside_weights_batch = []
    bbox_outside_weights_batch = []
    
    for i in range(len(rpn_rois)):
        rois = rpn_rois[i]
        scores = rpn_scores[i]
        levelInds = rpn_levelInds[i]
        

        #some one said it is an important trick
        if cfg.USE_GT:
            #if include the groundtruth into the candidate rois
            rois = torch.cat([rois, gt_bbox[i][:, :6]], 0)
            scores = torch.cat([scores, torch.ones(gt_bbox[i].shape[0], 1)], 0)
            levelInds_gt = torch.ones(gt_bbox[i].shape[0]) * 2
            for box_ind, box in enumerate(gt_bbox[i]):
                x_dim = box[3] - box[0]
                y_dim = box[4] - box[1]
                z_dim = box[5] - box[2]
                if x_dim <= 20 and y_dim <= 20 and z_dim <= 20:
                    levelInds_gt[box_ind] = 1
            levelInds = torch.cat([levelInds, levelInds_gt], 0)
                

        num_fg = int(round(cfg.FG_FRACTION * cfg.CLASS_BATCH_SIZE))

        # sample rois with classification labels and bounding box regression
        rois, scores, labels, levelInds, bbox_targets, bbox_inside_weights, bbox_outside_weights = _sample_rois(rois, scores, levelInds, gt_bbox[i], num_fg)

        rois_batch.append(rois)
        scores_batch.append(scores)
        labels_batch.append(labels.long())
        levelInds_batch.append(levelInds)
        bbox_targets_batch.append(bbox_targets)
        bbox_inside_weights_batch.append(bbox_inside_weights)
        bbox_outside_weights_batch.append(bbox_outside_weights)


    return torch.cat(rois_batch, 0), torch.cat(scores_batch, 0), torch.cat(labels_batch, 0), torch.cat(levelInds_batch, 0),\
           torch.cat(bbox_targets_batch, 0), torch.cat(bbox_inside_weights_batch, 0), torch.cat(bbox_outside_weights_batch, 0)


def _sample_rois(rois, scores, levelInds, gt_bbox, num_fg):
    """

    :param rois:
    :param scores:
    :param gt_bbox <torch.Tensor>: [n,6]
    :param num_fg:
    :return:
    """

    overlaps = bbox_overlap(rois, gt_bbox[:,:6])
    max_overlaps, gt_assignment = overlaps.max(1)

    #gt_assignment: give each roi a gt
    labels = gt_bbox[gt_assignment, [6]]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = (max_overlaps >= cfg.FG_THRESH).nonzero().view(-1)

    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI]
    #bg_inds = (max_overlaps < cfg.BG_THRESH).nonzero().view(-1)
    bg_inds = ((max_overlaps < cfg.BG_THRESH) + (max_overlaps >= cfg.BG_THRESH_LO) == 2).nonzero().view(-1)

    # padding with negative! there must be fg_inds, since we use gt_box as fg_inds, it might be there is no bg if we use []
    if fg_inds.numel() > 0 and bg_inds.numel() > 0:
        num_fg = min(num_fg, fg_inds.numel())
        fg_inds_choose = torch.from_numpy(np.random.choice(np.arange(0, fg_inds.numel()), size=int(num_fg), replace=False)).long()
        fg_inds = fg_inds[fg_inds_choose]

        num_bg = cfg.CLASS_BATCH_SIZE - num_fg
        bg_inds_choose = torch.from_numpy(np.random.choice(np.arange(0, bg_inds.numel()), size=int(num_bg), replace=bg_inds.numel() < num_bg)).long()
        bg_inds = bg_inds[bg_inds_choose]

    elif fg_inds.numel() > 0:
        num_fg = cfg.CLASS_BATCH_SIZE
        fg_inds_choose = torch.from_numpy(np.random.choice(np.arange(0, fg_inds.numel()), size=int(num_fg), replace=fg_inds.numel() < cfg.CLASS_BATCH_SIZE)).long()
        fg_inds = fg_inds[fg_inds_choose]

        num_bg = 0

    elif bg_inds.numel() > 0:
        num_fg = 0

        num_bg = cfg.CLASS_BATCH_SIZE
        bg_inds_choose = torch.from_numpy(np.random.choice(np.arange(0, bg_inds.numel()), size=int(num_bg), replace=bg_inds.numel() < cfg.CLASS_BATCH_SIZE)).long()
        bg_inds = bg_inds[bg_inds_choose]

    else:
        num_fg = 0

        bg_inds = (max_overlaps < cfg.BG_THRESH).nonzero().view(-1)
        num_bg = cfg.CLASS_BATCH_SIZE
        bg_inds_choose = torch.from_numpy(np.random.choice(np.arange(0, bg_inds.numel()), size=int(num_bg), replace=bg_inds.numel() < cfg.CLASS_BATCH_SIZE)).long()
        bg_inds = bg_inds[bg_inds_choose]

    if num_fg != 0 and num_fg != 0:
        keep_inds = torch.cat([fg_inds, bg_inds], 0)
    elif num_fg != 0:
        keep_inds = fg_inds
    elif num_bg != 0:
        keep_inds = bg_inds

    labels = labels[keep_inds].contiguous()

    if num_bg != 0:
        labels[int(num_fg):] = 0

    #with open('class_stats.txt', 'a') as f:
    #    for label in labels:
    #        f.write('{} '.format(label))
    #    f.write('\n')

    rois = rois[keep_inds].contiguous()
    scores = scores[keep_inds].contiguous()
    levelInds = levelInds[keep_inds].contiguous()

    bbox_target_data = _compute_targets(rois, gt_bbox[gt_assignment[keep_inds]][:,:6], labels)
    bbox_targets, bbox_inside_weights, bbox_outside_weights = _get_bbox_regression_labels(bbox_target_data)

    return rois, scores, labels, levelInds, bbox_targets, bbox_inside_weights, bbox_outside_weights

def _get_bbox_regression_labels(bbox_target_data):
    cls = bbox_target_data[:, 6]
    bbox_targets = cls.new(cls.numel(), 6*cfg.NUM_CLASSES).zero_()
    bbox_inside_weights = cls.new(bbox_targets.shape).zero_()
    fg_inds = (cls > 0).nonzero().view(-1)

    if fg_inds.numel() > 0:
        cls = cls[fg_inds].contiguous().view(-1, 1)
        inds_dim1 = fg_inds.unsqueeze(1).expand(fg_inds.size(0), 6)
        inds_dim2 = torch.cat([6*cls + 0, 6*cls + 1, 6*cls + 2, 
                               6*cls + 3, 6*cls + 4, 6*cls + 5], 1).long()

        bbox_targets[inds_dim1, inds_dim2] = bbox_target_data[fg_inds][:, :6]
        # inside weights are 1 where nonzeros in bbox_targets
        bbox_inside_weights[inds_dim1, inds_dim2] = \
            bbox_targets.new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).view(-1, 6).expand_as(inds_dim1)

    bbox_outside_weights = (bbox_inside_weights > 0).float()

    return bbox_targets, bbox_inside_weights, bbox_outside_weights



def _compute_targets(ex_rois, gt_rois, labels):
    """
    Compute bounding box regression targets for an image
    Inputs are tensor

    :param ex_rois:
    :param gt_rois:
    :param labels:
    :return:
    """
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 6
    assert gt_rois.shape[1] == 6 or gt_rois.shape[1] == 12

    targets = bbox_transform(ex_rois, gt_rois)

    return torch.cat([targets, labels.unsqueeze(1)], 1)





