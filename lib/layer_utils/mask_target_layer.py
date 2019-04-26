import math
import numpy as np

import torch
from torch.autograd import Variable
from lib.utils.overlap import bbox_overlap, mask_overlap
from lib.utils.config import cfg


def mask_target_layer(all_rois, gt_bbox, gt_mask, scene_info=None, scene=None):
    """
    Assign object detection proposals to ground-truth targets. Produce proposal
    classification labels and bounding box regression targets

    :param rpn_rois:
    :param gt_boxes: (num, 7)
    :return:
    """

    masks_batch = []
    rois_batch = []
    labels_batch= []
    for i in range(len(all_rois)):
        # prepare gt
        combined_rois = torch.cat([all_rois[i], gt_bbox[i][:,:6].cuda()], dim=0)
        gt_scene = torch.zeros(scene_info)

        for j, gt_box in enumerate(gt_bbox[i]):
            gt_scene[int(gt_box[0]):int(gt_box[3]), int(gt_box[1]):int(gt_box[4]), int(gt_box[2]):int(gt_box[5])] = gt_mask[i][j].float()

        #gt_assignment: give each roi a gt
        overlaps = bbox_overlap(combined_rois.cpu(), gt_bbox[i].cpu(), )
        max_overlaps, gt_assignment = overlaps.max(1)
        fg_inds = (max_overlaps >= cfg.MASK_FG_THRESH).nonzero().view(-1)

        masks = []
        rois = []
        labels = []
        label_gt = gt_bbox[i][gt_assignment, [6]]
        for fg_ind in fg_inds:
            roi = combined_rois[fg_ind]
            rois.append(roi)
            masks.append(gt_scene[int(round(roi[0].item())):int(round(roi[3].item())), int(round(roi[1].item())):int(round(roi[4].item())), int(round(roi[2].item())):int(round(roi[5].item()))])
            labels.append(label_gt[fg_ind])

        rois_batch.append(torch.stack(rois, 0))
        masks_batch.append(masks)
        labels_batch.append(torch.Tensor(labels).long())

    return rois_batch, masks_batch, labels_batch

