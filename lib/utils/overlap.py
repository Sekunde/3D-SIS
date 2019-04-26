import torch
import numpy as np
import math

def bbox_overlap(boxes, query_boxes):
    """

    :param boxes <torch.Tensor>: (N, 6)
    :param query_boxes <torch.Tensor>: (K, 6) 
    :return: overlaps (N, K) overlap between boxes and query boxes
    """

    out_fn = lambda x: x
    box_ares = (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])
    query_ares = (query_boxes[:, 3] - query_boxes[:, 0]) * (query_boxes[:, 4] - query_boxes[:, 1]) *\
                 (query_boxes[:, 5] - query_boxes[:, 2])

    iw = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t())).clamp(min=0)
    ih = (torch.min(boxes[:, 4:5], query_boxes[:, 4:5].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t())).clamp(min=0)
    il = (torch.min(boxes[:, 5:6], query_boxes[:, 5:6].t()) - torch.max(boxes[:, 2:3], query_boxes[:, 2:3].t())).clamp(min=0)

    ua = box_ares.view(-1, 1) + query_ares.view(1, -1) - iw*ih*il
    overlaps = iw*ih*il / ua

    return out_fn(overlaps)


def mask_overlap(gt_mask, gt_bbox, rois, scene_info):
    """
    gt_mask: torch Tensor(1, 7)
    roi_box: torch Tensor(1, 7)
    gt_box: torch Tensor(1, 7)
    scene_info: torch size(3)

    """
    warp_map = torch.ByteTensor(scene_info).zero_().cuda() if rois.is_cuda() else torch.ByteTensor(scene_info).zero_() 
    minx = int(gt_bbox[0])
    miny = int(gt_bbox[1])
    minz = int(gt_bbox[2])
    maxx = int(gt_bbox[3])
    maxy = int(gt_bbox[4])
    maxz = int(gt_bbox[5])
    try:
        warp_map[minx:maxx, miny:maxy, minz:maxz] = gt_mask
    except:
        import ipdb
        ipdb.set_trace()
        print('error')
    minx = min(max(round(rois[1]), 0), scene_info[0])
    miny = min(max(round(rois[2]), 0), scene_info[1])
    minz = min(max(round(rois[3]), 0), scene_info[2])
    maxx = min(max(round(rois[4]), 0), scene_info[0])
    maxy = min(max(round(rois[5]), 0), scene_info[1])
    maxz = min(max(round(rois[6]), 0), scene_info[2])
    return_warp = warp_map[minx:maxx, miny:maxy, minz:maxz]
    return return_warp
