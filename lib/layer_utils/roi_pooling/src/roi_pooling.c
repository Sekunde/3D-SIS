#include <TH/TH.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

int roi_pooling_forward(int pooled_width, int pooled_height, int pooled_length, float spatial_scale,
                        THFloatTensor* features, THFloatTensor* rois, THFloatTensor* output)
{
    // grab the input tensor
    float * data_flat = THFloatTensor_data(features);
    float * rois_flat = THFloatTensor_data(rois);
    float * output_flat = THFloatTensor_data(output);

    // number of ROIs
    int num_rois = THFloatTensor_size(rois, 0);
    int size_rois = THFloatTensor_size(rois, 1);
    if (size_rois != 6)
        return 0;

    //printf("im here");
    // batch size
    int batch_size = THFloatTensor_size(features, 0);

    if (batch_size != 1)
        return 0;
    // Number of channels
    int num_channels = THFloatTensor_size(features, 1);
    //printf("num_channels: %d \n", num_channels);
    //data_width
    int data_width = THFloatTensor_size(features, 2);
    //data_height
    int data_height = THFloatTensor_size(features, 3);
    //data_length
    int data_length = THFloatTensor_size(features, 4);


    int index_roi = 0;
    //printf("num_rois: %d \n", num_rois);
    for(int n = 0; n < num_rois; n++)
    {
       int roi_batch_ind = 0;
       int roi_start_w = floor(rois_flat[index_roi + 0] * spatial_scale);
       int roi_start_h = floor(rois_flat[index_roi + 1] * spatial_scale);
       int roi_start_l = floor(rois_flat[index_roi + 2] * spatial_scale);

       int roi_end_w = ceil(rois_flat[index_roi + 3] * spatial_scale);
       int roi_end_h = ceil(rois_flat[index_roi + 4] * spatial_scale);
       int roi_end_l = ceil(rois_flat[index_roi + 5] * spatial_scale);
       index_roi += size_rois;

       int roi_width = fmaxf(roi_end_w - roi_start_w, 1);
       int roi_height = fmaxf(roi_end_h - roi_start_h, 1);
       int roi_length = fmaxf(roi_end_l - roi_start_l, 1);

       float bin_size_w = (float)(roi_width) / (float)(pooled_width);
       float bin_size_h = (float)(roi_height) / (float)(pooled_height);
       float bin_size_l = (float)(roi_length) / (float)(pooled_length);


        for (int c = 0; c < num_channels; c++)
        {
           for(int pw = 0; pw < pooled_width; pw++)
           {
               for(int ph = 0; ph < pooled_height;  ph++)
               {
                    for(int pl = 0; pl < pooled_length; pl++)
                    {
                        int wstart = (int)(floor((float)(pw) * bin_size_w));
                        int hstart = (int)(floor((float)(ph) * bin_size_h));
                        int lstart = (int)(floor((float)(pl) * bin_size_l));

                        int wend = (int)(ceil((float)(pw + 1) * bin_size_w));
                        int hend = (int)(ceil((float)(ph + 1) * bin_size_h));
                        int lend = (int)(ceil((float)(pl + 1) * bin_size_l));

                        // Add roi offsets and clip to input boundaries
                        wstart = fminf(fmaxf(wstart + roi_start_w, 0), data_width);
                        hstart = fminf(fmaxf(hstart + roi_start_h, 0), data_height);
                        lstart = fminf(fmaxf(lstart + roi_start_l, 0), data_length);

                        wend = fminf(fmaxf(wend + roi_start_w, 0), data_width);
                        hend = fminf(fmaxf(hend + roi_start_h, 0), data_height);
                        lend = fminf(fmaxf(lend + roi_start_l, 0), data_length);

                        int pooled_index = (((n * num_channels + c) * pooled_width + pw) * pooled_height + ph) * pooled_length + pl;

                        bool is_empty = (hend <= hstart) || (wend <= wstart) || (lend <= lstart);
                         // Define an empty pooling region to be zero
                        float maxval = is_empty ? 0 : -FLT_MAX;
                        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd

                        //max pooling https://blog.deepsense.ai/region-of-interest-pooling-explained/
                        //printf("wstart: %d \n", wstart);
                        //printf("hstart: %d \n", hstart);
                        //printf("lstart: %d \n", lstart);

                        //printf("wend: %d \n", wend);
                        //printf("hend: %d \n", hend);
                        //printf("lend: %d \n", lend);
                        for (int w = wstart; w < wend; ++w)
                        {
                            for (int h = hstart; h < hend; ++h)
                            {
                               for (int l = lstart; l < lend; ++l)
                               {
                                  int bottom_index = (c * data_width + w) * data_height * data_length + h * data_length + l;
                                  if (data_flat[bottom_index] > maxval)
                                  {
                                    maxval = data_flat[bottom_index];
                                  }
                                  //printf("data in %d is %f \n", bottom_index, data_flat[bottom_index]);

                               }
                            }
                        }
                        output_flat[pooled_index] = maxval;
                        //printf("output in %d is %f \n", pooled_index, maxval);
                    }
               }
            }
        }
    }
    return 1;
}
