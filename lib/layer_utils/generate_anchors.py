import numpy as np
from lib.utils.config import cfg

def generate_anchors_for_single_voxel():
    """
    Generate anchors (reference) cube by enumerating aspect ratios x scales
    wrt a reference (0, 0, 0, 15, 15, 15) cube

    :param base_size:
    :param ratios:
    :param scales:
    :return: anchors.shape = [num_anchors, 6(minx, miny, minz, maxx, maxy, maxz)]
    """
    anchors_level1 = None
    anchors_level2 = None
    anchors_level3 = None

    if cfg.NUM_ANCHORS_LEVEL1 != 0:
        f = open('experiments/anchors/{}'.format(cfg.ANCHORS_TYPE_LEVEL1))
        anchors_read = f.readlines()
        f.close()
        anchors_level1 = np.zeros((len(anchors_read), 6))
        for idx, anchor in enumerate(anchors_read):
            anchors_level1[idx][0] = -float(anchor.strip().split(',')[0]) / 2
            anchors_level1[idx][1] = -float(anchor.strip().split(',')[1]) / 2
            anchors_level1[idx][2] = -float(anchor.strip().split(',')[2]) / 2
            anchors_level1[idx][3] =  float(anchor.strip().split(',')[0]) / 2
            anchors_level1[idx][4] =  float(anchor.strip().split(',')[1]) / 2
            anchors_level1[idx][5] =  float(anchor.strip().split(',')[2]) / 2

    if cfg.NUM_ANCHORS_LEVEL2 != 0:
        f = open('experiments/anchors/{}'.format(cfg.ANCHORS_TYPE_LEVEL2))
        anchors_read = f.readlines()
        f.close()
        anchors_level2 = np.zeros((len(anchors_read), 6))
        for idx, anchor in enumerate(anchors_read):
            anchors_level2[idx][0] = -float(anchor.strip().split(',')[0]) / 2
            anchors_level2[idx][1] = -float(anchor.strip().split(',')[1]) / 2
            anchors_level2[idx][2] = -float(anchor.strip().split(',')[2]) / 2
            anchors_level2[idx][3] =  float(anchor.strip().split(',')[0]) / 2
            anchors_level2[idx][4] =  float(anchor.strip().split(',')[1]) / 2
            anchors_level2[idx][5] =  float(anchor.strip().split(',')[2]) / 2

    if cfg.NUM_ANCHORS_LEVEL3 != 0:
        f = open('experiments/anchors/{}'.format(cfg.ANCHORS_TYPE_LEVEL3))
        anchors_read = f.readlines()
        f.close()
        anchors_level3 = np.zeros((len(anchors_read), 6))
        for idx, anchor in enumerate(anchors_read):
            anchors_level3[idx][0] = -float(anchor.strip().split(',')[0]) / 2
            anchors_level3[idx][1] = -float(anchor.strip().split(',')[1]) / 2
            anchors_level3[idx][2] = -float(anchor.strip().split(',')[2]) / 2
            anchors_level3[idx][3] =  float(anchor.strip().split(',')[0]) / 2
            anchors_level3[idx][4] =  float(anchor.strip().split(',')[1]) / 2
            anchors_level3[idx][5] =  float(anchor.strip().split(',')[2]) / 2
    return anchors_level1, anchors_level2, anchors_level3

def generate_anchors(size_level1, size_level2, size_level3, feat_stride):
    """
    A wrapper function generate anchors given different scales

    :param x:
    :param y:
    :param z:
    :param feat_stride:
    :param anchor_scales:
    :param anchor_ratios:
    :return:
    """


    anchors_level1, anchors_level2, anchors_level3 = generate_anchors_for_single_voxel()

    if cfg.NUM_ANCHORS_LEVEL1 != 0:
        # level 1
        shift_x = np.arange(0, size_level1[0]) * feat_stride[0]
        shift_y = np.arange(0, size_level1[1]) * feat_stride[0]
        shift_z = np.arange(0, size_level1[2]) * feat_stride[0]
        # meshgrid return the matrix wise representation, the input could be 'ij' or 'xy'
        shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z, indexing='ij')
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_z.ravel(), shift_x.ravel(), shift_y.ravel(), shift_z.ravel())).transpose()

        A = anchors_level1.shape[0]
        K = shifts.shape[0]

        anchors_level1 = anchors_level1.reshape((1, A, 6)) + shifts.reshape((1, K, 6)).transpose((1,0,2))
        anchors_level1 = anchors_level1.reshape((K*A, 6)).astype(np.float32, copy=False)

    if cfg.NUM_ANCHORS_LEVEL2 != 0:
        # level 2
        shift_x = np.arange(0, size_level2[0]) * feat_stride[1]
        shift_y = np.arange(0, size_level2[1]) * feat_stride[1]
        shift_z = np.arange(0, size_level2[2]) * feat_stride[1]
        # meshgrid return the matrix wise representation, the input could be 'ij' or 'xy'
        shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z, indexing='ij')
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_z.ravel(), shift_x.ravel(), shift_y.ravel(), shift_z.ravel())).transpose()

        A = anchors_level2.shape[0]
        K = shifts.shape[0]

        anchors_level2 = anchors_level2.reshape((1, A, 6)) + shifts.reshape((1, K, 6)).transpose((1,0,2))
        anchors_level2 = anchors_level2.reshape((K*A, 6)).astype(np.float32, copy=False)

    if cfg.NUM_ANCHORS_LEVEL3 != 0:
        # level 2
        shift_x = np.arange(0, size_level3[0]) * feat_stride[2]
        shift_y = np.arange(0, size_level3[1]) * feat_stride[2]
        shift_z = np.arange(0, size_level3[2]) * feat_stride[2]
        # meshgrid return the matrix wise representation, the input could be 'ij' or 'xy'
        shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z, indexing='ij')
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_z.ravel(), shift_x.ravel(), shift_y.ravel(), shift_z.ravel())).transpose()

        A = anchors_level3.shape[0]
        K = shifts.shape[0]

        anchors_level3 = anchors_level3.reshape((1, A, 6)) + shifts.reshape((1, K, 6)).transpose((1,0,2))
        anchors_level3 = anchors_level3.reshape((K*A, 6)).astype(np.float32, copy=False)

    return anchors_level1, anchors_level2, anchors_level3


