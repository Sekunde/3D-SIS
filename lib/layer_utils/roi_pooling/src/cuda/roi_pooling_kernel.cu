#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "roi_pooling_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


__global__ void ROIPoolForward(const int nthreads, const float* bottom_data,
    const float spatial_scale, const int width, const int height, const int length,
    const int channels, const int pooled_width, const int pooled_height, const int pooled_length,
    const float* bottom_rois, float* top_data, int* argmax_data)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, pw, ph, pl) is an element in the pooled output
        int n = index;
        int pl = n % pooled_length;
        n /= pooled_length;
        int ph = n % pooled_height;
        n /= pooled_height;
        int pw = n % pooled_width;
        n /= pooled_width;
        int c = n % channels;
        n /= channels;
        //printf("index: %d, c: %d, pw: %d, ph: %d, pl: %d, pooled_length: %d \n", index, c, pw, ph, pl, pooled_length);

        bottom_rois += n * 6;

        //from scene coord to feature map coord
        int roi_batch_ind = 0;
        int roi_start_w = floor(bottom_rois[0] * spatial_scale);
        int roi_start_h = floor(bottom_rois[1] * spatial_scale);
        int roi_start_l = floor(bottom_rois[2] * spatial_scale);

        int roi_end_w = ceil(bottom_rois[3] * spatial_scale);
        int roi_end_h = ceil(bottom_rois[4] * spatial_scale);
        int roi_end_l = ceil(bottom_rois[5] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = fmaxf(roi_end_w - roi_start_w, 1);
        int roi_height = fmaxf(roi_end_h - roi_start_h, 1);
        int roi_length = fmaxf(roi_end_l - roi_start_l, 1);
        //int roi_width = roi_end_w - roi_start_w;
        //int roi_height = roi_end_h - roi_start_h;
        //int roi_length = roi_end_l - roi_start_l;
	//if (roi_width < 1 or roi_height < 1 or roi_length < 1)
	//	continue;

        // from feature map coord to AFTER ROI,  separate region in feature map into regions, on
        // each region do max pooling
        // voxels in each subvolumn
        float bin_size_w = (float)(roi_width) / (float)(pooled_width);
        float bin_size_h = (float)(roi_height) / (float)(pooled_height);
        float bin_size_l = (float)(roi_length) / (float)(pooled_length);

        int wstart = (int)(floor((float)(pw) * bin_size_w));
        int hstart = (int)(floor((float)(ph) * bin_size_h));
        int lstart = (int)(floor((float)(pl) * bin_size_l));

        int wend = (int)(ceil((float)(pw + 1) * bin_size_w));
        int hend = (int)(ceil((float)(ph + 1) * bin_size_h));
        int lend = (int)(ceil((float)(pl + 1) * bin_size_l));


        // Add roi offsets and clip to input boundaries
        wstart = fminf(fmaxf(wstart + roi_start_w, 0), width);
        hstart = fminf(fmaxf(hstart + roi_start_h, 0), height);
        lstart = fminf(fmaxf(lstart + roi_start_l, 0), length);

        wend = fminf(fmaxf(wend + roi_start_w, 0), width);
        hend = fminf(fmaxf(hend + roi_start_h, 0), height);
        lend = fminf(fmaxf(lend + roi_start_l, 0), length);

        bool is_empty = (hend <= hstart) || (wend <= wstart) || (lend <= lstart);

        // Define an empty pooling region to be zero
        float maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;
        bottom_data += roi_batch_ind * channels * width * height * length;

        //max pooling https://blog.deepsense.ai/region-of-interest-pooling-explained/
        for (int w = wstart; w < wend; ++w)
        {
            for (int h = hstart; h < hend; ++h)
             {
                for (int l = lstart; l < lend; ++l)
                {
                    int bottom_index = (c * width + w) * height * length + h * length + l;
                    if (bottom_data[bottom_index] > maxval)
                    {
                        maxval = bottom_data[bottom_index];
                        maxidx = bottom_index;
                    }
                }
            }
        }
        top_data[index] = maxval;
        if (argmax_data != NULL)
            argmax_data[index] = maxidx;
    }
}


int ROIPoolForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int width,
    const int height, const int length, const int channels, const int pooled_width,
    const int pooled_height, const int pooled_length, const float* bottom_rois,
    float* top_data, int* argmax_data, cudaStream_t stream)
{
    //every block has 1024 threads
    const int kThreadsPerBlock = 1024;
    const int output_size = num_rois * channels * pooled_width * pooled_height * pooled_length;
    cudaError_t err;

    ROIPoolForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, bottom_data, spatial_scale, width, height, length, channels, pooled_width,
      pooled_height, pooled_length, bottom_rois, top_data, argmax_data);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
    return 1;
}


__global__ void ROIPoolBackward(const int nthreads, const float* top_diff,
    const int* argmax_data, const int num_rois, const float spatial_scale,
    const int width, const int height, const int length, const int channels,
    const int pooled_width, const int pooled_height, const int pooled_length,
    float* bottom_diff, const float* bottom_rois)
    {
        CUDA_1D_KERNEL_LOOP(index, nthreads)
        {

            // (n, c, w, h, l) is an element in the input
            int n = index;
            int l = n % length;
            n /= length;
            int h = n % height;
            n /= height;
            int w = n % width;
            n /= width;
            int c = n % channels;
            n /= channels;

            float gradient = 0;
            // Accumulate gradient over all ROIs that pooled this element
            for (int roi_n = 0; roi_n < num_rois; roi_n++)
            {
                const float* offset_bottom_rois = bottom_rois + roi_n * 6;
                int roi_batch_ind = 0;
                // Skip if ROI's batch index doesn't match n
                if (n != roi_batch_ind) {
                    continue;
                }
                int offset = roi_n * pooled_width * pooled_height * pooled_length * channels;
                const float* offset_top_diff = top_diff + offset;
                const int* offset_argmax_data = argmax_data + offset;

		/*
                int roi_start_w = floor(offset_bottom_rois[1] * spatial_scale);
                int roi_start_h = floor(offset_bottom_rois[2] * spatial_scale);
                int roi_start_l = floor(offset_bottom_rois[3] * spatial_scale);

                int roi_end_w = ceil(offset_bottom_rois[4] * spatial_scale);
                int roi_end_h = ceil(offset_bottom_rois[5] * spatial_scale);
                int roi_end_l = ceil(offset_bottom_rois[6] * spatial_scale);

                // Skip if ROI doesn't include (w, h, l)
                const bool in_roi = (w >= roi_start_w && w < roi_end_w &&
                                   h >= roi_start_h &&  h < roi_end_h &&
                                   l >= roi_start_l && l < roi_end_l);
                if (!in_roi) {
                    continue;
                }

                // Compute feasible set of pooled units that could have pooled
                // this bottom unit

                // Force malformed ROIs to be 1x1
                int roi_width = fmaxf(roi_end_w - roi_start_w, 1);
                int roi_height = fmaxf(roi_end_h - roi_start_h, 1);
                int roi_length = fmaxf(roi_end_l - roi_start_l, 1);
                float bin_size_w = (float)(roi_width) / (float)(pooled_width);
                float bin_size_h = (float)(roi_height) / (float)(pooled_height);
                float bin_size_l = (float)(roi_length) / (float)(pooled_length);


                int pwstart = floor((float)(w - roi_start_w) / bin_size_w);
                int pwend = ceil((float)(w - roi_start_w + 1) / bin_size_w);

                int phstart = floor((float)(h - roi_start_h) / bin_size_h);
                int phend = ceil((float)(h - roi_start_h + 1) / bin_size_h);

                int plstart = floor((float)(l - roi_start_l) / bin_size_l);
                int plend = ceil((float)(l - roi_start_l + 1) / bin_size_l);

                pwstart = fminf(fmaxf(pwstart, 0), pooled_width);
                pwend = fminf(fmaxf(pwend, 0), pooled_width);
                phstart = fminf(fmaxf(phstart, 0), pooled_height);
                phend = fminf(fmaxf(phend, 0), pooled_height);
                plstart = fminf(fmaxf(plstart, 0), pooled_length);
                plend = fminf(fmaxf(plend, 0), pooled_length);

              	for (int pw = pwstart; pw < pwend; ++pw)
                {
                     for (int ph = phstart; ph < phend; ++ph)
                     {
                        for (int pl = plstart; pl < plend; ++pl)
                        {
                            if (offset_argmax_data[(c * pooled_width + pw) * pooled_height * pooled_length + ph * pooled_length + pl] == index)
                            {
                                gradient += offset_top_diff[(c * pooled_width + pw) * pooled_height * pooled_length + ph * pooled_length + pl];
                            }
                        }
                    }
                }
		*/

              	for (int pw = 0; pw < pooled_width; ++pw)
                {
                     for (int ph = 0; ph < pooled_height; ++ph)
                     {
                        for (int pl = 0; pl < pooled_length; ++pl)
                        {
                            if (offset_argmax_data[(c * pooled_width + pw) * pooled_height * pooled_length + ph * pooled_length + pl] == index)
                            {
                                gradient += offset_top_diff[(c * pooled_width + pw) * pooled_height * pooled_length + ph * pooled_length + pl];
                            }
                        }
                    }
                }

            }
            bottom_diff[index] = gradient;
      }
}

int ROIPoolBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int length, const int channels, const int pooled_height,
    const int pooled_width, const int pooled_length, const float* bottom_rois,
    float* bottom_diff, const int* argmax_data, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = batch_size * width * height * length * channels;
    cudaError_t err;

    ROIPoolBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, top_diff, argmax_data, num_rois, spatial_scale, width, height, length, channels, pooled_width,
      pooled_height, pooled_length, bottom_diff, bottom_rois);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


#ifdef __cplusplus
}
#endif
