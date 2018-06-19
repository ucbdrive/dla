#include "EAConv2d_kernel.h"
#include "handle.h"

#include <cuda.h>

#include <THC.h>

extern THCState* state;

void EAConv2d_cuda_forward_bias(THCudaTensor* bias,
                                THCudaTensor* output) {
  cudnnHandle_t cudnn = getCudnnHandle();
  cudnnSetStream(cudnn, THCState_getCurrentStream(state));
  EAConv2d_cudnn_forward_bias(cudnn,
                              THCudaTensor_data(state, bias),
                              THCudaTensor_data(state, output),
                              output->size[0],
                              output->size[1],
                              output->size[2],
                              output->size[3]);
}

void EAConv2d_cuda_backward_bias(THCudaTensor* grad_bias,
                                 THCudaTensor* gradOutput) {
  cudnnHandle_t cudnn = getCudnnHandle();
  cudnnSetStream(cudnn, THCState_getCurrentStream(state));
  EAConv2d_cudnn_backward_bias(cudnn,
                               THCudaTensor_data(state, grad_bias),
                               THCudaTensor_data(state, gradOutput),
                               gradOutput->size[0],
                               gradOutput->size[1],
                               gradOutput->size[2],
                               gradOutput->size[3]);
}

int EAConv2d_cuda_forward(THCudaTensor* input,
                          THCudaTensor* weight,
                          THCudaTensor* output,
                          int stride_x,
                          int stride_y,
                          int padding_x,
                          int padding_y,
                          int dilation_x,
                          int dilation_y,
                          int groups
  ) {
    cudnnHandle_t cudnn = getCudnnHandle();
    cudnnSetStream(cudnn, THCState_getCurrentStream(state));
    EAConv2d_cudnn_forward(state,
                           cudnn,
                           THCudaTensor_data(state, input),
                           THCudaTensor_data(state, weight),
                           THCudaTensor_data(state, output),
                           stride_x,
                           stride_y,
                           padding_x,
                           padding_y,
                           dilation_x,
                           dilation_y,
                           groups,
                           input->size[0],
                           input->size[1],
                           input->size[2],
                           input->size[3],
                           weight->size[0],
                           weight->size[1],
                           weight->size[2],
                           weight->size[3],
                           output->size[0],
                           output->size[1],
                           output->size[2],
                           output->size[3]);
    return 1;
}

int EAConv2d_cuda_backward(THCudaTensor* grad_input,
                           THCudaTensor* grad_weight,
                           THCudaTensor* gradOutput,
                           THCudaTensor* input,
                           THCudaTensor* weight,
                           int stride_x,
                           int stride_y,
                           int padding_x,
                           int padding_y,
                           int dilation_x,
                           int dilation_y,
                           int groups
  ) {
    cudnnHandle_t cudnn = getCudnnHandle();
    cudnnSetStream(cudnn, THCState_getCurrentStream(state));
    EAConv2d_cudnn_backward(state,
                            cudnn,
                            THCudaTensor_data(state, grad_input),
                            THCudaTensor_data(state, grad_weight),
                            THCudaTensor_data(state, gradOutput),
                            THCudaTensor_data(state, input),
                            THCudaTensor_data(state, weight),
                            stride_x,
                            stride_y,
                            padding_x,
                            padding_y,
                            dilation_x,
                            dilation_y,
                            groups,
                            grad_input->size[0],
                            grad_input->size[1],
                            grad_input->size[2],
                            grad_input->size[3],
                            grad_weight->size[0],
                            grad_weight->size[1],
                            grad_weight->size[2],
                            grad_weight->size[3],
                            gradOutput->size[0],
                            gradOutput->size[1],
                            gradOutput->size[2],
                            gradOutput->size[3]);
    return 1;
}
