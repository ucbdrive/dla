#ifndef EACONV_SRC_CONV_PARAMS_H_
#define EACONV_SRC_CONV_PARAMS_H_

#include <cudnn.h>

class Convolution_Params {
 public:
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnFilterDescriptor_t kernel_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  Convolution_Params(int stride_x,
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
                     int output_w);
  ~Convolution_Params();
};

#endif  // EACONV_SRC_CONV_PARAMS_H_
