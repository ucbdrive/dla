#include "eaconv/src/cuda_check.h"
#include "eaconv/src/conv_params.h"

Convolution_Params::Convolution_Params(int stride_x,
                                       int stride_y,
                                       int padding_x,
                                       int padding_y,
                                       int dilation_x,
                                       int dilation_y,
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
  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/input_batch_size,
                                        /*channels=*/input_channels,
                                        /*image_height=*/input_h,
                                        /*image_width=*/input_w));
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/output_batch_size,
                                        /*channels=*/output_channels,
                                        /*image_height=*/output_h,
                                        /*image_width=*/output_w));
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_desc));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_desc,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/kernel_out,
                                        /*in_channels=*/kernel_in,
                                        /*kernel_height=*/kernel_h,
                                        /*kernel_width=*/kernel_w));
  checkCUDNN(cudnnCreateTensorDescriptor(&bias_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(bias_desc,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        1,
                                        /*channels=*/output_channels,
                                        1,
                                        1));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  checkCUDNN(cudnnSetConvolution2dDescriptor(
    conv_desc,
    /*pad_height=*/padding_x,
    /*pad_width=*/padding_y,
    /*vertical_stride=*/stride_x,
    /*horizontal_stride=*/stride_y,
    /*dilation_height=*/dilation_x,
    /*dilation_width=*/dilation_y,
    /*mode=*/CUDNN_CROSS_CORRELATION,
    /*computeType=*/CUDNN_DATA_FLOAT));
  checkCUDNN(cudnnSetConvolutionMathType(conv_desc,
                                         CUDNN_TENSOR_OP_MATH));
}

Convolution_Params::~Convolution_Params() {
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(kernel_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}
