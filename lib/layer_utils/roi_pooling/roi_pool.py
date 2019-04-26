from torch.autograd import Function
from ._ext import roi_pooling
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np

class RoIPoolFunction(Function):
    def __init__(self, pooled_width, pooled_height, pooled_length, spatial_scale):
        super(RoIPoolFunction, self).__init__()
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.pooled_length = int(pooled_length)
        self.spatial_scale = float(spatial_scale)
        self.argmax = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        _, num_channels, _, _, _ = features.size()
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, num_channels, self.pooled_width, self.pooled_height, self.pooled_length)
        argmax = torch.IntTensor(num_rois, num_channels, self.pooled_width, self.pooled_height, self.pooled_length).zero_()

        output = output.cuda() if features.is_cuda else output
        argmax = argmax.cuda() if features.is_cuda else argmax
        if features.is_cuda:
            roi_pooling.roi_pooling_forward_cuda(self.pooled_width, self.pooled_height, self.pooled_length, self.spatial_scale,
                                            features, rois, output, argmax)
        else:
            roi_pooling.roi_pooling_forward(self.pooled_width, self.pooled_height, self.pooled_length, self.spatial_scale,
                                            features, rois, output)
        self.argmax = argmax
        self.rois = rois
        self.feature_size = features.size()

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_width, data_height, data_length = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_width, data_height, data_length).cuda()
        roi_pooling.roi_pooling_backward_cuda(self.pooled_width, self.pooled_height, self.pooled_length, self.spatial_scale,
                                         grad_output, self.rois, grad_input, self.argmax)

        return grad_input, torch.zeros_like(self.rois)



class RoIPool(Function):
    def __init__(self, pooled_width, pooled_height, pooled_length, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.pooled_length = int(pooled_length)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        self.batch_size, self.num_channels, self.data_width, self.data_height, self.data_length = features.size()
        self.num_rois = rois.size()[0]
        self.remember_for_backward = torch.zeros(self.num_rois, self.num_channels, self.pooled_width, self.pooled_height, self.pooled_length, 3) - 1
        outputs = torch.zeros(self.num_rois, self.num_channels, self.pooled_width, self.pooled_height, self.pooled_length)

        for roi_ind, roi in enumerate(rois):
            
            roi_start_w, roi_start_h, roi_start_l, roi_end_w, roi_end_h, roi_end_l = roi.cpu().numpy() * self.spatial_scale

            roi_start_w = int(math.floor(roi_start_w))
            roi_start_h = int(math.floor(roi_start_h))
            roi_start_l = int(math.floor(roi_start_l))

            roi_end_w = int(math.ceil(roi_end_w))
            roi_end_h = int(math.ceil(roi_end_h))
            roi_end_l = int(math.ceil(roi_end_l))

            roi_width = max(roi_end_w - roi_start_w, 1)
            roi_height = max(roi_end_h - roi_start_h, 1)
            roi_length = max(roi_end_l - roi_start_l, 1)
            #roi_width = roi_end_w - roi_start_w
            #roi_height = roi_end_h - roi_start_h
            #roi_length = roi_end_l - roi_start_l
            #if roi_width < 1 or roi_height < 1 or roi_length < 1:
            #    continue

            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)
            bin_size_l = float(roi_length) / float(self.pooled_length)

            for pw in range(self.pooled_width):
                for ph in range(self.pooled_height):
                    for pl in range(self.pooled_length):
                        wstart = int(np.floor(pw * bin_size_w))
                        hstart = int(np.floor(ph * bin_size_h))
                        lstart = int(np.floor(pl * bin_size_l))

                        wend = int(np.ceil((pw + 1) * bin_size_w))
                        hend = int(np.ceil((ph + 1) * bin_size_h))
                        lend = int(np.ceil((pl + 1) * bin_size_l))

                        wstart = min(self.data_width, max(0, wstart + roi_start_w))
                        hstart = min(self.data_height, max(0, hstart + roi_start_h))
                        lstart = min(self.data_length, max(0, lstart + roi_start_l))

                        wend = min(self.data_width, max(0, wend + roi_start_w))
                        hend = min(self.data_height, max(0, hend + roi_start_h))
                        lend = min(self.data_length, max(0, lend + roi_start_l))

                        is_empty = (hend <= hstart) or(wend <= wstart) or (lend <= lstart)
                        if is_empty:
                            outputs[roi_ind, :, pw, ph, pl] = 0
                        else:
                            data = features[0]
                            outputs[roi_ind, :, pw, ph, pl] = torch.max(torch.max(torch.max(data[:, wstart:wend, hstart:hend, lstart:lend], 1)[0], 1)[0], 1)[0].view(-1)
                            for c in range(self.num_channels):
                                ind_w, ind_h, ind_l = np.unravel_index(data[c, wstart:wend, hstart:hend, lstart:lend].numpy().argmax(), data[c, wstart:wend, hstart:hend, lstart:lend].numpy().shape)
                                self.remember_for_backward[roi_ind, c, pw, ph, pl] = torch.from_numpy(np.array([ind_w+wstart, ind_h+hstart, ind_l+lstart])).float()
        return outputs
    '''
    def forward(self, features, rois):
        self.batch_size, self.num_channels, self.data_width, self.data_height, self.data_length = features.size()
        self.num_rois = rois.size()[0]
        self.remember_for_backward = torch.zeros(self.num_rois, self.num_channels, self.pooled_width, self.pooled_height, self.pooled_length, 3) - 1
        outputs = torch.zeros(self.num_rois, self.num_channels, self.pooled_width, self.pooled_height, self.pooled_length)

        for roi_ind, roi in enumerate(rois):
            #print('[roi]: %s', str(roi[1:].cpu().numpy()))
            
            batch_ind = int(roi[0])
            roi_start_w, roi_start_h, roi_start_l, roi_end_w, roi_end_h, roi_end_l = roi[1:].cpu().numpy() * self.spatial_scale
            #print('\tstart: (%f, %f, %f) -> end (%f, %f, %f)' % (roi_start_w, roi_start_h, roi_start_l, roi_end_w, roi_end_h, roi_end_l))
            #visualization.create_bbox_mesh(np.array([roi_start_w, roi_start_h, roi_start_l, roi_end_w, roi_end_h, roi_end_l]), 'check_roi_box.obj', radius=0.02, scale=np.array([1.0/self.spatial_scale, 1.0/self.spatial_scale, 1.0/self.spatial_scale]))

            roi_width = max(roi_end_w - roi_start_w, 1)
            roi_height = max(roi_end_h - roi_start_h, 1)
            roi_length = max(roi_end_l - roi_start_l, 1)
            #print('\tstart: (%.3f, %.3f, %.3f) -> end (%.3f, %.3f, %.3f) ==> extent (%.3f, %.3f, %.3f)' % (roi_start_w, roi_start_h, roi_start_l, roi_end_w, roi_end_h, roi_end_l, roi_width, roi_height, roi_length))

            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)
            bin_size_l = float(roi_length) / float(self.pooled_length)
            #print('\tbin size: (%.3f, %.3f, %.3f)' % (bin_size_w, bin_size_h, bin_size_l))
            #debug_start = np.array([0, 0, 0], dtype=np.float32)
            #debug_end = np.array([self.pooled_width, self.pooled_height, self.pooled_length], dtype=np.float32)
            #debug_scale = np.array([roi_width, roi_height, roi_length], dtype=np.float32) / np.array([self.pooled_width, self.pooled_height, self.pooled_length], dtype=np.float32) / self.spatial_scale
            #print('scale: roi_dim (%f, %f, %f) / pool_dim (%f, %f, %f) / spatial_scale (%f) ==> (%f, %f, %f)' % (roi_width, roi_height, roi_length, self.pooled_width, self.pooled_height, self.pooled_length, self.spatial_scale, debug_scale[0], debug_scale[1], debug_scale[2]))
            #print('debug start (%f, %f, %f) -> debug end (%f, %f, %f)' % (debug_start[0], debug_start[1], debug_start[2], debug_end[0], debug_end[1], debug_end[2]))
            #debug_start *= debug_scale
            #debug_end *= debug_scale
            #print('(scaled) debug start (%f, %f, %f) -> debug end (%f, %f, %f)' % (debug_start[0], debug_start[1], debug_start[2], debug_end[0], debug_end[1], debug_end[2]))
            #debug_start += np.array([roi[1], roi[2], roi[3]])
            #debug_end += np.array([roi[1], roi[2], roi[3]])
            #print('(scaled) debug start (%f, %f, %f) -> debug end (%f, %f, %f)' % (debug_start[0], debug_start[1], debug_start[2], debug_end[0], debug_end[1], debug_end[2]))
            #visualization.create_bbox_mesh(np.array([0, 0, 0, self.pooled_width, self.pooled_height, self.pooled_length]), 'check_roi_pool_box.obj', radius=0.02, offset=np.array([roi_start_w, roi_start_h, roi_start_l])/self.spatial_scale, scale=debug_scale)
            #d0 = np.array([0, 0, 0]) * debug_scale
            #d1 = np.array([self.pooled_width, self.pooled_height, self.pooled_length]) * debug_scale
            #print('[CHECK_ROI_POOL_BOX] (%f, %f, %f) -> (%f, %f, %f) ==> (scale) (%f, %f, %f) -> (%f, %f, %f) ==> (offset) (%f, %f, %f) -> (%f, %f, %f)' % (0, 0, 0, self.pooled_width, self.pooled_height, self.pooled_length, d0[0], d0[1], d0[2], d1[0], d1[1], d1[2], d0[0]+roi_start_w/self.spatial_scale, d0[1]+roi_start_h/self.spatial_scale, d0[2]+roi_start_l/self.spatial_scale, d1[0]+roi_start_w/self.spatial_scale, d1[1]+roi_start_h/self.spatial_scale, d1[2]+roi_start_l/self.spatial_scale))
            #print(np.where(self.remember_for_backward[0,0] != -1))

            for pw in range(self.pooled_width):
                for ph in range(self.pooled_height):
                    for pl in range(self.pooled_length):
                        #print('[PW, PH, PL]: %d, %d, %d' % (pw, ph, pl))
                        hstart = ph * bin_size_h
                        lstart = pl * bin_size_l
                        wend = (pw + 1) * bin_size_w
                        hend = (ph + 1) * bin_size_h
                        lend = (pl + 1) * bin_size_l
                        wstart = min(self.data_width, max(0,  int(np.floor(wstart + roi_start_w))))
                        hstart = min(self.data_height, max(0, int(np.floor(hstart + roi_start_h))))
                        lstart = min(self.data_length, max(0, int(np.floor(lstart + roi_start_l))))                        
                        wend = min(self.data_width, max(0,  int(np.ceil(wend + roi_start_w))))
                        hend = min(self.data_height, max(0, int(np.ceil(hend + roi_start_h))))
                        lend = min(self.data_length, max(0, int(np.ceil(lend + roi_start_l))))
                        
                        is_empty = (hend <= hstart) or(wend <= wstart) or (lend <= lstart)
                        if is_empty:
                            outputs[roi_ind, :, pw, ph, pl] = 0
                        else:
                            data = features[batch_ind]
                            outputs[roi_ind, :, pw, ph, pl] = torch.max(torch.max(torch.max(data[:, wstart:wend, hstart:hend, lstart:lend], 1)[0], 1)[0], 1)[0].view(-1)
                            for c in range(self.num_channels):
                                ind_w, ind_h, ind_l = np.unravel_index(data[c, wstart:wend, hstart:hend, lstart:lend].numpy().argmax(), data[c, wstart:wend, hstart:hend, lstart:lend].numpy().shape)
                                self.remember_for_backward[roi_ind, c, pw, ph, pl] = torch.from_numpy(np.array([ind_w+wstart, ind_h+hstart, ind_l+lstart])).float()
        return outputs
    '''
    def backward(self, grad_output):
        grad_input = torch.zeros(1, self.num_channels, self.data_width, self.data_height, self.data_length)
        for roi_ind in range(self.num_rois):
            for c in range(self.num_channels):
                for pw in range(self.pooled_width):
                    for ph in range(self.pooled_height):
                        for pl in range(self.pooled_length):
                            w, h, l = self.remember_for_backward[roi_ind, c, pw, ph, pl]
                            if w != -1:
                                grad_input[0, int(c), int(w), int(h), int(l)] += grad_output[roi_ind, c, pw, ph, pl]
        return grad_input

