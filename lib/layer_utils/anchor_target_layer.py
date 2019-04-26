import numpy as np
import torch

from lib.utils.overlap import bbox_overlap
from lib.utils.bbox_transform import bbox_transform
from lib.utils.config import cfg
from collections import deque


def anchor_target_layer(all_anchors_level1, feat_size_level1, 
                        all_anchors_level2, feat_size_level2,
                        all_anchors_level3, feat_size_level3,
                        gt_bbox, scene_info,
                        anchors_filter_level1, anchors_filter_level2, anchors_filter_level3):
    """
    Same as the anchor target layer in original Faster/er RCNN
    :param all_anchors <torch.Tensor>: [n, 6]
    :param gt_bbox <list of torch.Tensor>: [[n, 6]]
    :param scene_info: [64, 32, 64]
    :param feat_size: [width, height, length]
    :return:

    """
    _allowed_border = cfg.ALLOW_BORDER

    if cfg.NUM_ANCHORS_LEVEL1 != 0:
        total_anchors_level1 = all_anchors_level1.size(0)
        (width_level1, height_level1, length_level1) = feat_size_level1
        # only keep anchors inside the image
        inds_inside_level1 = np.where(
            (all_anchors_level1[:, 0] >= -_allowed_border) &
            (all_anchors_level1[:, 1] >= -_allowed_border) &
            (all_anchors_level1[:, 2] >= -_allowed_border) &
            (all_anchors_level1[:, 3] < scene_info[0] + _allowed_border) &  #width
            (all_anchors_level1[:, 4] < scene_info[1] + _allowed_border) &  #height
            (all_anchors_level1[:, 5] < scene_info[2] + _allowed_border)   #length
        )[0]
        if anchors_filter_level1 is not None:
            if len(anchors_filter_level1) == 0:
                anchors_filter_level1 = [0]
            inds_inside_level1 = inds_inside_level1[anchors_filter_level1]
        anchors_level1 = all_anchors_level1[inds_inside_level1, :]

        labels_batch_level1 = []
        bbox_targets_batch_level1 = []
        bbox_inside_weights_batch_level1 = []
        bbox_outside_weights_batch_level1 = []

    if cfg.NUM_ANCHORS_LEVEL2 != 0:
        total_anchors_level2 = all_anchors_level2.size(0)
        # map of shape (..., W, H, L)
        (width_level2, height_level2, length_level2) = feat_size_level2
        inds_inside_level2 = np.where(
            (all_anchors_level2[:, 0] >= -_allowed_border) &
            (all_anchors_level2[:, 1] >= -_allowed_border) &
            (all_anchors_level2[:, 2] >= -_allowed_border) &
            (all_anchors_level2[:, 3] < scene_info[0] + _allowed_border) &  #width
            (all_anchors_level2[:, 4] < scene_info[1] + _allowed_border) &  #height
            (all_anchors_level2[:, 5] < scene_info[2] + _allowed_border)   #length
        )[0]
        if anchors_filter_level2 is not None:
            if len(anchors_filter_level2) == 0:
                anchors_filter_level2 = [0]
            inds_inside_level2 = inds_inside_level2[anchors_filter_level2]
        anchors_level2 = all_anchors_level2[inds_inside_level2, :]

        labels_batch_level2 = []
        bbox_targets_batch_level2 = []
        bbox_inside_weights_batch_level2 = []
        bbox_outside_weights_batch_level2 = []


    if cfg.NUM_ANCHORS_LEVEL3 != 0:
        total_anchors_level3 = all_anchors_level3.size(0)
        # map of shape (..., W, H, L)
        (width_level3, height_level3, length_level3) = feat_size_level3
        inds_inside_level3 = np.where(
            (all_anchors_level3[:, 0] >= -_allowed_border) &
            (all_anchors_level3[:, 1] >= -_allowed_border) &
            (all_anchors_level3[:, 2] >= -_allowed_border) &
            (all_anchors_level3[:, 3] < scene_info[0] + _allowed_border) &  #width
            (all_anchors_level3[:, 4] < scene_info[1] + _allowed_border) &  #height
            (all_anchors_level3[:, 5] < scene_info[2] + _allowed_border)   #length
        )[0]
        if anchors_filter_level3 is not None:
            if len(anchors_filter_level3) == 0:
                anchors_filter_level3 = [0]
            inds_inside_level3 = inds_inside_level3[anchors_filter_level3]
        anchors_level3 = all_anchors_level3[inds_inside_level3, :]

        labels_batch_level3 = []
        bbox_targets_batch_level3 = []
        bbox_inside_weights_batch_level3 = []
        bbox_outside_weights_batch_level3 = []

    for i in range(len(gt_bbox)):
        labels_level_list = []
        if cfg.NUM_ANCHORS_LEVEL1 != 0:
            # label:  1 is positive, 0 is negative, -1 is no care
            labels_level1 = np.empty((len(inds_inside_level1),), dtype=np.float32)
            labels_level1.fill(-1)
            # overlaps between the anchors and the gt_boxes
            overlaps_level1 = bbox_overlap(anchors_level1, gt_bbox[i]).numpy()

            #box= np.stack([np.concatenate([anchors_level1[9], np.ones(1)], 0),
            #               np.concatenate([anchors_level1[46], np.ones(1)], 0),
            #               np.concatenate([anchors_level1[239], np.ones(1)], 0),
            #               np.concatenate([anchors_level1[241], np.ones(1)], 0)], 0)
            #visualize('./vis', 'anchor_positive', data=None, bbox=box)

            #box= np.stack([np.concatenate([anchors_level1[5222], np.ones(1)], 0),
            #               np.concatenate([anchors_level1[5223], np.ones(1)], 0),
            #               np.concatenate([anchors_level1[5224], np.ones(1)], 0),
            #               np.concatenate([anchors_level1[5225], np.ones(1)], 0),
            #               np.concatenate([anchors_level1[5226], np.ones(1)], 0),
            #               np.concatenate([anchors_level1[5227], np.ones(1)], 0)], 0)
            #visualize('./vis', 'anchor_negtive', data=None, bbox=box)
            #----------------------------------
            # anchor-wise: give every anchor a label
            #----------------------------------
            # give every anchor a gt_bbox ind [n_anchor, ]
            argmax_overlaps_level1 = overlaps_level1.argmax(axis=1)
            # give every anchor a 'max overlap ratio' [n_anchor, ] 
            max_overlaps_level1 = overlaps_level1[np.arange(len(inds_inside_level1)), argmax_overlaps_level1]
            # first set the negatives
            labels_level1[max_overlaps_level1 >= cfg.RPN_POSITIVE_OVERLAP] = 1
            labels_level1[max_overlaps_level1 < cfg.RPN_NEGATIVE_OVERLAP] = 0

            '''
            #----------------------------------
            # gtbbox-wise: give anchors label 1 that 
            # has the max overlap with gt_bbox
            #----------------------------------
            # give every gt_bbox an anchor ind [n_gtbbox, ]
            gt_argmax_overlaps_level1 = overlaps_level1.argmax(axis=0)
            # give every gt_bbox a 'max overlap ratio' [n_gtbbox, ] 
            gt_argmax_overlaps_level1 = overlaps_level1[gt_argmax_overlaps_level1, np.arange(overlaps_level1.shape[1])]
            # avoid 0 overlapping gt_box (too many positives)
            for gt_max_overlap_ind in range(gt_argmax_overlaps_level1.shape[0]):
                if gt_argmax_overlaps_level1[gt_max_overlap_ind] == 0.0:
                    gt_argmax_overlaps_level1[gt_max_overlap_ind] = -1
            gt_argmax_overlaps_level1 = np.where(overlaps_level1 == gt_argmax_overlaps_level1)[0]
            # fg label: for gt, anchor with highest overlap
            labels_level1[gt_argmax_overlaps_level1] = 1
            '''
            labels_level_list.append(labels_level1)


        if cfg.NUM_ANCHORS_LEVEL2 !=0:
            labels_level2 = np.empty((len(inds_inside_level2),), dtype=np.float32)
            labels_level2.fill(-1)
            overlaps_level2 = bbox_overlap(anchors_level2, gt_bbox[i]).numpy()
            argmax_overlaps_level2 = overlaps_level2.argmax(axis=1)
            max_overlaps_level2 = overlaps_level2[np.arange(len(inds_inside_level2)), argmax_overlaps_level2]
            labels_level2[max_overlaps_level2 >= cfg.RPN_POSITIVE_OVERLAP] = 1
            labels_level2[max_overlaps_level2 < cfg.RPN_NEGATIVE_OVERLAP] = 0

            '''
            #----------------------------------
            # gtbbox-wise: give anchors label 1 that 
            # has the max overlap with gt_bbox
            #----------------------------------
            gt_argmax_overlaps_level2 = overlaps_level2.argmax(axis=0)
            gt_argmax_overlaps_level2 = overlaps_level2[gt_argmax_overlaps_level2, np.arange(overlaps_level2.shape[1])]
            for gt_max_overlap_ind in range(gt_argmax_overlaps_level2.shape[0]):
                if gt_argmax_overlaps_level2[gt_max_overlap_ind] == 0.0:
                    gt_argmax_overlaps_level2[gt_max_overlap_ind] = -1
            gt_argmax_overlaps_level2 = np.where(overlaps_level2 == gt_argmax_overlaps_level2)[0]
            labels_level2[gt_argmax_overlaps_level2] = 1

            '''
            labels_level_list.append(labels_level2)


        if cfg.NUM_ANCHORS_LEVEL3 !=0:
            #--------------------------------
            # change the cls9 box by adding 4 voxels on thin dim
            #--------------------------------
            dims_changed = []
            for ind, box in enumerate(gt_bbox[i]):
                # change box size of class 9
                if box[6] in [6, 7, 9, 12, 18]:
                    x_size = box[3] - box[0]
                    y_size = box[4] - box[1]
                    z_size = box[5] - box[2]
                    if x_size <= y_size and x_size <= z_size:
                        gt_bbox[i][ind][0] = gt_bbox[i][ind][0] - 2
                        gt_bbox[i][ind][3] = gt_bbox[i][ind][3] + 2
                        dims_changed.append('x')

                    elif y_size <= x_size and y_size <= z_size:
                        gt_bbox[i][ind][1] = gt_bbox[i][ind][1] - 2
                        gt_bbox[i][ind][4] = gt_bbox[i][ind][4] + 2
                        dims_changed.append('y')

                    elif z_size <= y_size and z_size <= x_size:
                        gt_bbox[i][ind][2] = gt_bbox[i][ind][2] - 2
                        gt_bbox[i][ind][5] = gt_bbox[i][ind][5] + 2
                        dims_changed.append('z')
                else:
                    gt_bbox[i][ind][1] = gt_bbox[i][ind][1] + 100
                    gt_bbox[i][ind][4] = gt_bbox[i][ind][4] - 100

            dims_changed = deque(dims_changed)

            labels_level3 = np.empty((len(inds_inside_level3),), dtype=np.float32)
            labels_level3.fill(-1)
            overlaps_level3 = bbox_overlap(anchors_level3, gt_bbox[i]).numpy()
            argmax_overlaps_level3 = overlaps_level3.argmax(axis=1)
            max_overlaps_level3 = overlaps_level3[np.arange(len(inds_inside_level3)), argmax_overlaps_level3]
            labels_level3[max_overlaps_level3 >= cfg.RPN_POSITIVE_OVERLAP] = 1
            labels_level3[max_overlaps_level3 < cfg.RPN_NEGATIVE_OVERLAP] = 0

            '''
            #----------------------------------
            # gtbbox-wise: give anchors label 1 that 
            # has the max overlap with gt_bbox
            #----------------------------------
            gt_argmax_overlaps_level2 = overlaps_level2.argmax(axis=0)
            gt_argmax_overlaps_level2 = overlaps_level2[gt_argmax_overlaps_level2, np.arange(overlaps_level2.shape[1])]
            for gt_max_overlap_ind in range(gt_argmax_overlaps_level2.shape[0]):
                if gt_argmax_overlaps_level2[gt_max_overlap_ind] == 0.0:
                    gt_argmax_overlaps_level2[gt_max_overlap_ind] = -1
            gt_argmax_overlaps_level2 = np.where(overlaps_level2 == gt_argmax_overlaps_level2)[0]
            labels_level2[gt_argmax_overlaps_level2] = 1

            '''
            labels_level_list.append(labels_level3)

            #--------------------------------
            # change the cls9 box back
            #--------------------------------
            for ind, box in enumerate(gt_bbox[i]):
                # change box size of class 9
                if box[6] in [6, 7, 9, 12, 18]:
                    dim_changed = dims_changed.popleft()
                    if dim_changed == 'x':
                        gt_bbox[i][ind][0] = gt_bbox[i][ind][0] + 2
                        gt_bbox[i][ind][3] = gt_bbox[i][ind][3] - 2

                    elif dim_changed == 'y':
                        gt_bbox[i][ind][1] = gt_bbox[i][ind][1] + 2
                        gt_bbox[i][ind][4] = gt_bbox[i][ind][4] - 2

                    elif dim_changed == 'z':
                        gt_bbox[i][ind][2] = gt_bbox[i][ind][2] + 2
                        gt_bbox[i][ind][5] = gt_bbox[i][ind][5] - 2
                else:
                    gt_bbox[i][ind][1] = gt_bbox[i][ind][1] - 100
                    gt_bbox[i][ind][4] = gt_bbox[i][ind][4] + 100

        labels = np.concatenate(labels_level_list, 0)

        #----------------------------------
        # sampling
        #----------------------------------
        # subsample positive labels if we have too many
        num_fg = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]

        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        '''
        #----------------------------------
        # dynamic sampling
        #----------------------------------
        # subsample negative labels if we have too many
        num_bg = np.sum(labels == 1)
        if num_bg == 0:
            num_bg = 1
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
        '''

        if cfg.NUM_ANCHORS_LEVEL1 != 0:
            labels_level1 = labels[:labels_level1.shape[0]]
            #-----------------------------------
            # bbox reg
            #-----------------------------------
            # each anchor has a gtbbox target to regress
            bbox_targets_level1 = _compute_targets(anchors_level1, gt_bbox[i][argmax_overlaps_level1, :])
            # only the positive ones have regression targets
            # inside weights: only regress those anchors whose label is 1
            bbox_inside_weights_level1 = np.zeros((len(inds_inside_level1), 6), dtype=np.float32)
            bbox_inside_weights_level1[labels_level1 == 1, :] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            # reweight anchors regression
            bbox_outside_weights_level1 = np.zeros((len(inds_inside_level1), 6), dtype=np.float32)
            bbox_outside_weights_level1[labels_level1 == 1, :] = np.ones((1, 6)) / np.sum(labels_level1 == 1)

            # map up to original set of anchors
            labels_level1 = _unmap(labels_level1, total_anchors_level1, inds_inside_level1, fill=-1)
            bbox_targets_level1 = _unmap(bbox_targets_level1, total_anchors_level1, inds_inside_level1, fill=0)
            bbox_inside_weights_level1 = _unmap(bbox_inside_weights_level1, total_anchors_level1, inds_inside_level1, fill=0)
            bbox_outside_weights_level1 = _unmap(bbox_outside_weights_level1, total_anchors_level1, inds_inside_level1, fill=0)

            # labels reshape for corresponding to rpn_cls_score_reshape
            labels_level1 = labels_level1.reshape((1, width_level1, height_level1, length_level1, cfg.NUM_ANCHORS_LEVEL1))
            bbox_targets_level1 = bbox_targets_level1.reshape((1, width_level1, height_level1, length_level1, cfg.NUM_ANCHORS_LEVEL1 * 6))
            bbox_inside_weights_level1 = bbox_inside_weights_level1.reshape((1, width_level1, height_level1, length_level1, cfg.NUM_ANCHORS_LEVEL1 * 6))
            bbox_outside_weights_level1 = bbox_outside_weights_level1.reshape((1, width_level1, height_level1, length_level1, cfg.NUM_ANCHORS_LEVEL1 * 6))
            # into batch
            labels_batch_level1.append(torch.from_numpy(labels_level1))
            bbox_targets_batch_level1.append(torch.from_numpy(bbox_targets_level1))
            bbox_inside_weights_batch_level1.append(torch.from_numpy(bbox_inside_weights_level1))
            bbox_outside_weights_batch_level1.append(torch.from_numpy(bbox_outside_weights_level1))

        if cfg.NUM_ANCHORS_LEVEL2 != 0:
            if cfg.NUM_ANCHORS_LEVEL3 != 0:
                labels_level2 = labels[-(labels_level2.shape[0]+labels_level3.shape[0]):-labels_level3.shape[0]]
            else:
                labels_level2 = labels[-labels_level2.shape[0]:]

            bbox_targets_level2 = _compute_targets(anchors_level2, gt_bbox[i][argmax_overlaps_level2, :])
            bbox_inside_weights_level2 = np.zeros((len(inds_inside_level2), 6), dtype=np.float32)
            bbox_inside_weights_level2[labels_level2 == 1, :] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            bbox_outside_weights_level2 = np.zeros((len(inds_inside_level2), 6), dtype=np.float32)
            bbox_outside_weights_level2[labels_level2 == 1, :] = np.ones((1, 6)) / np.sum(labels_level2 == 1)

            labels_level2 = _unmap(labels_level2, total_anchors_level2, inds_inside_level2, fill=-1)
            bbox_targets_level2 = _unmap(bbox_targets_level2, total_anchors_level2, inds_inside_level2, fill=0)
            bbox_inside_weights_level2 = _unmap(bbox_inside_weights_level2, total_anchors_level2, inds_inside_level2, fill=0)
            bbox_outside_weights_level2 = _unmap(bbox_outside_weights_level2, total_anchors_level2, inds_inside_level2, fill=0)

            labels_level2 = labels_level2.reshape((1, width_level2, height_level2, length_level2, cfg.NUM_ANCHORS_LEVEL2))
            bbox_targets_level2 = bbox_targets_level2.reshape((1, width_level2, height_level2, length_level2, cfg.NUM_ANCHORS_LEVEL2 * 6))
            bbox_inside_weights_level2 = bbox_inside_weights_level2.reshape((1, width_level2, height_level2, length_level2, cfg.NUM_ANCHORS_LEVEL2 * 6))
            bbox_outside_weights_level2 = bbox_outside_weights_level2.reshape((1, width_level2, height_level2, length_level2, cfg.NUM_ANCHORS_LEVEL2 * 6))

            labels_batch_level2.append(torch.from_numpy(labels_level2))
            bbox_targets_batch_level2.append(torch.from_numpy(bbox_targets_level2))
            bbox_inside_weights_batch_level2.append(torch.from_numpy(bbox_inside_weights_level2))
            bbox_outside_weights_batch_level2.append(torch.from_numpy(bbox_outside_weights_level2))

        if cfg.NUM_ANCHORS_LEVEL3 != 0:
            labels_level3 = labels[-labels_level3.shape[0]:]
            bbox_targets_level3 = _compute_targets(anchors_level3, gt_bbox[i][argmax_overlaps_level3, :])
            bbox_inside_weights_level3 = np.zeros((len(inds_inside_level3), 6), dtype=np.float32)
            bbox_inside_weights_level3[labels_level3 == 1, :] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            bbox_outside_weights_level3 = np.zeros((len(inds_inside_level3), 6), dtype=np.float32)
            bbox_outside_weights_level3[labels_level3 == 1, :] = np.ones((1, 6)) / np.sum(labels_level3 == 1)

            labels_level3 = _unmap(labels_level3, total_anchors_level3, inds_inside_level3, fill=-1)
            bbox_targets_level3 = _unmap(bbox_targets_level3, total_anchors_level3, inds_inside_level3, fill=0)
            bbox_inside_weights_level3 = _unmap(bbox_inside_weights_level3, total_anchors_level3, inds_inside_level3, fill=0)
            bbox_outside_weights_level3 = _unmap(bbox_outside_weights_level3, total_anchors_level3, inds_inside_level3, fill=0)

            labels_level3 = labels_level3.reshape((1, width_level3, height_level3, length_level3, cfg.NUM_ANCHORS_LEVEL3))
            bbox_targets_level3 = bbox_targets_level3.reshape((1, width_level3, height_level3, length_level3, cfg.NUM_ANCHORS_LEVEL3 * 6))
            bbox_inside_weights_level3 = bbox_inside_weights_level3.reshape((1, width_level3, height_level3, length_level3, cfg.NUM_ANCHORS_LEVEL3 * 6))
            bbox_outside_weights_level3 = bbox_outside_weights_level3.reshape((1, width_level3, height_level3, length_level3, cfg.NUM_ANCHORS_LEVEL3 * 6))

            labels_batch_level3.append(torch.from_numpy(labels_level3))
            bbox_targets_batch_level3.append(torch.from_numpy(bbox_targets_level3))
            bbox_inside_weights_batch_level3.append(torch.from_numpy(bbox_inside_weights_level3))
            bbox_outside_weights_batch_level3.append(torch.from_numpy(bbox_outside_weights_level3))


    return torch.cat(labels_batch_level1, 0) if cfg.NUM_ANCHORS_LEVEL1 != 0 else None, \
           torch.cat(bbox_targets_batch_level1, 0) if cfg.NUM_ANCHORS_LEVEL1 !=0 else None,\
           torch.cat(bbox_inside_weights_batch_level1, 0) if cfg.NUM_ANCHORS_LEVEL1 != 0 else None,\
           torch.cat(bbox_outside_weights_batch_level1, 0) if cfg.NUM_ANCHORS_LEVEL1 !=0 else None,\
           torch.cat(labels_batch_level2, 0) if cfg.NUM_ANCHORS_LEVEL2 !=0 else None, \
           torch.cat(bbox_targets_batch_level2, 0) if cfg.NUM_ANCHORS_LEVEL2 !=0 else None, \
           torch.cat(bbox_inside_weights_batch_level2, 0) if cfg.NUM_ANCHORS_LEVEL2 != 0 else None, \
           torch.cat(bbox_outside_weights_batch_level2, 0) if cfg.NUM_ANCHORS_LEVEL2 !=0 else None,\
           torch.cat(labels_batch_level3, 0) if cfg.NUM_ANCHORS_LEVEL3 !=0 else None, \
           torch.cat(bbox_targets_batch_level3, 0) if cfg.NUM_ANCHORS_LEVEL3 !=0 else None, \
           torch.cat(bbox_inside_weights_batch_level3, 0) if cfg.NUM_ANCHORS_LEVEL3 != 0 else None, \
           torch.cat(bbox_outside_weights_batch_level3, 0) if cfg.NUM_ANCHORS_LEVEL3 !=0 else None
    


def _unmap(data, count, inds, fill=0):
    """
    Unmap a subset of item (data) back to the original set of items (of size count)
    :param data:
    :param count:
    :param inds:
    :param fill:
    :return:
    """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(anchor_rois, gt_rois):
    """
    Compute bounding box regression targets for an image
    :param anchor_rois <torch.Tensor>:
    :param gt_rois:
    :return:
    """

    assert anchor_rois.shape[0] == gt_rois.shape[0]
    assert anchor_rois.shape[1] == 6
    assert gt_rois.shape[1] == 7

    return bbox_transform(anchor_rois, gt_rois[:,:6])
