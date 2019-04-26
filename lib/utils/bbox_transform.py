import torch
import numpy as np

def clip_boxes(boxes, scene_shape):
    """
    clip boxes to scene boundaries

    :param boxes: (K*A,6)
    :param scene_shape: [width, height, length]
    :return:
    """

    boxes = torch.stack([boxes[:,0].clamp(0, scene_shape[0]),
                         boxes[:,1].clamp(0, scene_shape[1]),
                         boxes[:,2].clamp(0, scene_shape[2]),
                         boxes[:,3].clamp(0, scene_shape[0]),
                         boxes[:,4].clamp(0, scene_shape[1]),
                         boxes[:,5].clamp(0, scene_shape[2])], 1)


    return boxes


def bbox_transform(anchor_rois, gt_rois):
    """

    :param anchor_rois <torch.Tensor>:  
    :param gt_rois <torch.Tensor>:
    :return:
    """
    anchor_widths  = anchor_rois[:, 3] - anchor_rois[:, 0]
    anchor_heights = anchor_rois[:, 4] - anchor_rois[:, 1]
    anchor_lengths = anchor_rois[:, 5] - anchor_rois[:, 2]

    anchor_ctr_x = anchor_rois[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchor_rois[:, 1] + 0.5 * anchor_heights
    anchor_ctr_z = anchor_rois[:, 2] + 0.5 * anchor_lengths

    gt_widths  = gt_rois[:, 3] - gt_rois[:, 0]
    gt_heights = gt_rois[:, 4] - gt_rois[:, 1]
    gt_lengths = gt_rois[:, 5] - gt_rois[:, 2]

    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    gt_ctr_z = gt_rois[:, 2] + 0.5 * gt_lengths

    targets_dx = (gt_ctr_x - anchor_ctr_x) / (anchor_widths + 1e-14)
    targets_dy = (gt_ctr_y - anchor_ctr_y) / (anchor_heights + 1e-14)
    targets_dz = (gt_ctr_z - anchor_ctr_z) / (anchor_lengths + 1e-14)

    targets_dw = torch.log(gt_widths  / (anchor_widths  + 1e-14) + 1e-14)
    targets_dh = torch.log(gt_heights / (anchor_heights + 1e-14) + 1e-14)
    targets_dl = torch.log(gt_lengths / (anchor_lengths + 1e-14) + 1e-14)

    targets = torch.stack([targets_dx, targets_dy, targets_dz, targets_dw, targets_dh, targets_dl], 1)

    return targets

def bbox_transform_inv(boxes, deltas):
    """

    :param boxes <torch.Tensor>: (K*A, 6) anchors in scene coord.
    :param deltas <torch.Tensor>: (K*A, 6) preds
    :return:
    """
    if len(boxes) == 0:
        return deltas.detach() * 0

    widths = boxes[:, 3] - boxes[:,0]
    heights = boxes[:, 4] - boxes[:,1]
    lengths = boxes[:, 5] - boxes[:,2]

    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    ctr_z = boxes[:, 2] + 0.5 * lengths

    dx = deltas[:, 0::6]
    dy = deltas[:, 1::6]
    dz = deltas[:, 2::6]
    dw = deltas[:, 3::6]
    dh = deltas[:, 4::6]
    dl = deltas[:, 5::6]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_ctr_z = dz * lengths.unsqueeze(1) + ctr_z.unsqueeze(1)

    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)
    pred_l = torch.exp(dl) * lengths.unsqueeze(1)
    pred_boxes = torch.cat([pred_ctr_x - 0.5 * pred_w,
                            pred_ctr_y - 0.5 * pred_h,
                            pred_ctr_z - 0.5 * pred_l,
                            pred_ctr_x + 0.5 * pred_w,
                            pred_ctr_y + 0.5 * pred_h,
                            pred_ctr_z + 0.5 * pred_l], 1)


    return pred_boxes



