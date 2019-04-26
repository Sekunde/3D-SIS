#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdio.h>
#include <float.h>
#include "nms_kernel.h"


__device__ inline float devIoU(float const * const a, float const * const b)
{
   float left = fmaxf(a[0], b[0]);
   float top = fmaxf(a[1], b[1]);
   float left_top = fmaxf(a[2], b[2]);

   float right =  fminf(a[3], b[3]);
   float bottom = fminf(a[4], b[4]);
   float right_bottom = fminf(a[5], b[5]);

   float width = fmaxf(right - left + 1, 0.f);
   float height = fmaxf(bottom - top + 1, 0.f);
   float length = fmaxf(right_bottom - left_top + 1, 0.f);

   float interS = width * height * length;
   float Sa = (a[3] - a[0] + 1) * (a[4] - a[1] + 1) * (a[5] - a[2] + 1);
   float Sb = (b[3] - b[0] + 1) * (b[4] - b[1] + 1) * (b[5] - b[2] + 1);

   return interS / (Sa + Sb - interS);

}


__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float* dev_boxes, unsigned long long * dev_mask)
{
    //printf("n_boxes: %d \n", n_boxes);
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    //for the sake of last block does not have all threads
    const int row_size = fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size = fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock*6];
    if (threadIdx.x < col_size)
    {
        block_boxes[threadIdx.x*6 + 0] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x)*6+0];
        block_boxes[threadIdx.x*6 + 1] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x)*6+1];
        block_boxes[threadIdx.x*6 + 2] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x)*6+2];
        block_boxes[threadIdx.x*6 + 3] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x)*6+3];
        block_boxes[threadIdx.x*6 + 4] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x)*6+4];
        block_boxes[threadIdx.x*6 + 5] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x)*6+5];

    }
    __syncthreads();

    if(threadIdx.x < row_size)
    {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float *cur_box = dev_boxes + cur_box_idx * 6;
        unsigned long long t = 0;
        int start = 0;
        if(row_start == col_start)
        {
            start = threadIdx.x + 1;
        }
        for(int i=start; i < col_size; i++)
        {
            if(devIoU(cur_box, block_boxes + i * 6) > nms_overlap_thresh)
            {
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }

}

void _nms(int boxes_num, float* boxes_dev, unsigned long long * mask_dev,
          float nms_overlap_thresh)
{
    //printf("boxes num: %d \n", boxes_num);
    //threadsPerBlock = 64
    //DIVUP == ceil
    dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
                DIVUP(boxes_num, threadsPerBlock));
    dim3 threads(threadsPerBlock);

    nms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh,
                                    boxes_dev, mask_dev);

}

#ifdef __cplusplus
}
#endif
