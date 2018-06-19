#ifndef EACONV_SRC_EACONV2D_KERNEL_H_
#define EACONV_SRC_EACONV2D_KERNEL_H_

#include <cudnn.h>

#include "THC/THC.h"

#ifdef __cplusplus
    extern "C" {
#endif

void EAConv2d_cudnn_forward_bias(cudnnHandle_t cudnn,
                                 float *bias, float* output,
                                 int output_batch_size,
                                 int output_channels,
                                 int output_h,
                                 int output_w);

void EAConv2d_cudnn_backward_bias(cudnnHandle_t cudnn,
                                  float *grad_bias, float* gradOutput,
                                  int output_batch_size,
                                  int output_channels,
                                  int output_h,
                                  int output_w);

void EAConv2d_cudnn_forward(THCState* state,
                            cudnnHandle_t cudnn,
                            float* input,
                            float* weight,
                            float* output,
                            int stride_x,
                            int stride_y,
                            int padding_x,
                            int padding_y,
                            int dilation_x,
                            int dilation_y,
                            int groups,
                            int input_batch_size,
                            int input_channels,
                            int input_h,
                            int input_w,
                            int kernel_out,
                            int kernel_in,
                            int kernel_h,
                            int kernel_w,
                            int output_batch_size,
                            int output_channels,
                            int output_h,
                            int output_w);

void EAConv2d_cudnn_backward(THCState* state,
                             cudnnHandle_t cudnn,
                             float* grad_input,
                             float* grad_weight,
                             float* gradOutput,
                             float* input,
                             float* weight,
                             int stride_x,
                             int stride_y,
                             int padding_x,
                             int padding_y,
                             int dilation_x,
                             int dilation_y,
                             int groups,
                             int input_batch_size,
                             int input_channels,
                             int input_h,
                             int input_w,
                             int kernel_out,
                             int kernel_in,
                             int kernel_h,
                             int kernel_w,
                             int output_batch_size,
                             int output_channels,
                             int output_h,
                             int output_w);

#ifdef __cplusplus
    }
#endif

#endif  // EACONV_SRC_EACONV2D_KERNEL_H_
