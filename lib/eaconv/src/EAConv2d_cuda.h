void EAConv2d_cuda_forward_bias(THCudaTensor* bias,
                                THCudaTensor* output);

void EAConv2d_cuda_backward_bias(THCudaTensor* grad_bias,
                                 THCudaTensor* gradOutput);

int EAConv2d_cuda_forward(THCudaTensor* input,
                          THCudaTensor* weight,
                          THCudaTensor* output,
                          int stride_x,
                          int stride_y,
                          int padding_x,
                          int padding_y,
                          int dilation_x,
                          int dilation_y,
                          int groups);

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
                           int groups);
