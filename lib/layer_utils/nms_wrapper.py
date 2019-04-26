import torch

from lib.layer_utils.nms.pth_nms import cpu_nms
from lib.layer_utils.nms.pth_nms import pth_nms


def nms(dets, thresh):
    """
    Dispatch to either CPU (numpy) or GPU NMS implementations
    accept dets as tensor

    :param dets:
    :param thresh:
    :return:
    """
    return pth_nms(dets, thresh) if dets.is_cuda else torch.from_numpy(cpu_nms(dets.numpy(), thresh))
