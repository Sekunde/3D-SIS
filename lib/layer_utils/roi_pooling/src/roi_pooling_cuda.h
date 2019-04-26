int roi_pooling_forward_cuda(int pooled_width, int pooled_height, int pooled_length, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output, THCudaIntTensor * argmax);

int roi_pooling_backward_cuda(int pooled_width, int pooled_height, int pooled_length, float spatial_scale,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * argmax);