#include <iostream>

#include "eaconv/src/EAConv2d_kernel.h"
#include "eaconv/src/cuda_check.h"
#include "eaconv/src/handle.h"
#include "eaconv/src/conv_params.h"

struct Workspace {
  Workspace(THCState* state, size_t size) :
      state(state), size(size), data(NULL) {
    checkCUDA(THCudaMalloc(state, &data, size));
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  ~Workspace() {
    if (data) {
      THCudaFree(state, data);
    }
  }
  THCState* state;
  size_t size;
  void* data;
};

#ifdef __cplusplus
    extern "C" {
#endif

void EAConv2d_cudnn_forward_bias(cudnnHandle_t cudnn,
                                 float *bias, float* output,
                                 int output_batch_size,
                                 int output_channels,
                                 int output_h,
                                 int output_w) {
  Convolution_Params params(1, 1,
                            0, 0,
                            1, 1,
                            3, 3, 3, 3,
                            3, 3, 3, 3,
                            output_batch_size,
                            output_channels,
                            output_h,
                            output_w);
  const float alpha = 1;
  checkCUDNN(cudnnAddTensor(cudnn, &alpha,
                            params.bias_desc, bias,
                            &alpha,
                            params.output_desc, output));
}

void EAConv2d_cudnn_backward_bias(cudnnHandle_t cudnn,
                                  float *grad_bias, float* gradOutput,
                                  int output_batch_size,
                                  int output_channels,
                                  int output_h,
                                  int output_w) {
  Convolution_Params params(1, 1,
                            0, 0,
                            1, 1,
                            3, 3, 3, 3,
                            3, 3, 3, 3,
                            output_batch_size,
                            output_channels,
                            output_h,
                            output_w);
  const float alpha = 1;
  checkCUDNN(cudnnConvolutionBackwardBias(cudnn, &alpha,
                                          params.output_desc, gradOutput,
                                          &alpha,
                                          params.bias_desc, grad_bias));
}

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
                            int output_w) {
  Convolution_Params params(stride_x,
                                 stride_y,
                                 padding_x,
                                 padding_y,
                                 dilation_x,
                                 dilation_y,
                                 input_batch_size,
                                 input_channels,
                                 input_h,
                                 input_w,
                                 kernel_out,
                                 kernel_in,
                                 kernel_h,
                                 kernel_w,
                                 output_batch_size,
                                 output_channels,
                                 output_h,
                                 output_w);
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(
      cudnnGetConvolutionForwardAlgorithm(cudnn,
                                          params.input_desc,
                                          params.kernel_desc,
                                          params.conv_desc,
                                          params.output_desc,
                                          // CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                          /*memoryLimitInBytes=*/0,
                                          &convolution_algorithm));
  size_t workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     params.input_desc,
                                                     params.kernel_desc,
                                                     params.conv_desc,
                                                     params.output_desc,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
  Workspace cur_ws(state, workspace_bytes);

  const float alpha = 1;
  checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     params.input_desc,
                                     input,
                                     params.kernel_desc,
                                     weight,
                                     params.conv_desc,
                                     convolution_algorithm,
                                     cur_ws.data,
                                     cur_ws.size,
                                     &alpha,
                                     params.output_desc,
                                     output));
}

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
                             int output_w) {
  Convolution_Params params(stride_x,
                            stride_y,
                            padding_x,
                            padding_y,
                            dilation_x,
                            dilation_y,
                            input_batch_size,
                            input_channels,
                            input_h,
                            input_w,
                            kernel_out,
                            kernel_in,
                            kernel_h,
                            kernel_w,
                            output_batch_size,
                            output_channels,
                            output_h,
                            output_w);
  // backward filter
  cudnnConvolutionBwdFilterAlgo_t convolution_filter_algorithm;
  checkCUDNN(
    cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn,
      params.input_desc,
      params.output_desc,
      params.conv_desc,
      params.kernel_desc,
      // CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
      CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
      /*memoryLimitInBytes=*/0,
      &convolution_filter_algorithm));
  size_t filter_workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
             cudnn,
             params.input_desc,
             params.output_desc,
             params.conv_desc,
             params.kernel_desc,
             convolution_filter_algorithm,
             &filter_workspace_bytes));

  // backward data
  cudnnConvolutionBwdDataAlgo_t convolution_data_algorithm;
  checkCUDNN(
    cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn,
      params.kernel_desc,
      params.output_desc,
      params.conv_desc,
      params.input_desc,
      // CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
      /*memoryLimitInBytes=*/0,
      &convolution_data_algorithm));
  size_t data_workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnn,
    params.kernel_desc,
    params.output_desc,
    params.conv_desc,
    params.input_desc,
    convolution_data_algorithm,
    &data_workspace_bytes));

  Workspace filter_ws(state, filter_workspace_bytes);
  Workspace data_ws(state, data_workspace_bytes);

  const float alpha = 1;
  checkCUDNN(cudnnConvolutionBackwardFilter(cudnn,
                                            &alpha,
                                            params.input_desc,
                                            input,
                                            params.output_desc,
                                            gradOutput,
                                            params.conv_desc,
                                            convolution_filter_algorithm,
                                            filter_ws.data,
                                            filter_ws.size,
                                            &alpha,
                                            params.kernel_desc,
                                            grad_weight));
  checkCUDNN(cudnnConvolutionBackwardData(cudnn,
                                          &alpha,
                                          params.kernel_desc,
                                          weight,
                                          params.output_desc,
                                          gradOutput,
                                          params.conv_desc,
                                          convolution_data_algorithm,
                                          data_ws.data,
                                          data_ws.size,
                                          &alpha,
                                          params.input_desc,
                                          grad_input));
}

#ifdef __cplusplus
    }
#endif
