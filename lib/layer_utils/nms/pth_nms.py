import numpy as np
import torch

from ._ext import nms


def cpu_nms(dets, thresh):
    """
    cpu python nms: numpy
    :param dets:
    :param thresh:
    :return:
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    z1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    z2 = dets[:, 5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
    order = np.arange(0, dets.shape[0])

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        zz1 = np.maximum(z1[i], z1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        zz2 = np.minimum(z2[i], z2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        l = np.maximum(0.0, zz2 - zz1 + 1)
        inter = w * h * l
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def pth_nms(dets, thresh):
    """

    :param dets:
    :param thresh:
    :return: the index of bboxes kept
    """


    order = torch.from_numpy(np.arange(0, dets.size(0))).long().cuda()
    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    nms.gpu_nms(keep, num_out, dets, thresh)
    #keep takes "num_out" boxes indices

    return order[keep[:num_out[0]].cuda()].contiguous()

