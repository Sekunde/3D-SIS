int roi_pooling_forward(int pooled_width, int pooled_height, int pooled_length, float spatial_scale,
                        THFloatTensor* features, THFloatTensor* rois, THFloatTensor* output);