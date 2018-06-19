import torch
from torch.autograd import Function
from eaconv._ext import eaconv2d


def _conv_output_dim(inputDim, pad, filterDim, dilation, stride):
    tmp = 1 + (inputDim + 2 * pad - (((filterDim - 1) * dilation) + 1))
    if tmp % stride == 0\
            and pad >= 0\
            and stride >= 1\
            and dilation >= 1\
            and tmp > 0:
        return tmp // stride
    else:
        raise ValueError('Parameters of the kernel must be compatible '
                         'with the dimensions of the input')


class EAConv2dFunction(Function):

    def __init__(self, stride, padding, dilation, groups):
        super(EAConv2dFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, *bias_inp_and_weight):
        num_inp = len(bias_inp_and_weight) // 2
        inp = bias_inp_and_weight[:num_inp]
        if len(bias_inp_and_weight) % 2 == 1:
            weight = bias_inp_and_weight[num_inp:-1]
            bias = bias_inp_and_weight[-1]
        else:
            weight = bias_inp_and_weight[num_inp:]
            bias = None

        self.saved_for_later = inp, weight, bias
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        groups = self.groups

        h = _conv_output_dim(inp[0].size(2), padding[0],
                             weight[0].size(2), dilation[0], stride[0])
        w = _conv_output_dim(inp[0].size(3), padding[1],
                             weight[0].size(3), dilation[1], stride[1])
        output = inp[0].new(inp[0].size(0), weight[0].size(0), h, w).zero_()
        if bias is not None:
            eaconv2d.EAConv2d_cuda_forward_bias(bias, output)
        for _inp, _weight in zip(inp, weight):
            if not isinstance(_inp, torch.cuda.FloatTensor):
                raise NotImplementedError
            eaconv2d.EAConv2d_cuda_forward(_inp, _weight, output,
                                           stride[0], stride[1],
                                           padding[0], padding[1],
                                           dilation[0], dilation[1],
                                           groups)
        return output

    def backward(self, gradOutput):
        gradOutput = gradOutput.contiguous()

        inp, weight, bias = self.saved_for_later

        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        groups = self.groups

        if bias is not None:
            grad_bias = bias.new(*bias.size()).zero_()
        else:
            grad_bias = None

        grad_inp = []
        grad_weight = []
        for _inp in inp:
            grad_inp.append(_inp.new(*_inp.size()).zero_())
        for _weight in weight:
            grad_weight.append(_weight.new(*_weight.size()).zero_())

        if bias is not None:
            eaconv2d.EAConv2d_cuda_backward_bias(grad_bias, gradOutput)
        for _inp, _weight, _grad_inp, _grad_weight in zip(inp, weight,
                                                          grad_inp,
                                                          grad_weight):
            eaconv2d.EAConv2d_cuda_backward(_grad_inp, _grad_weight,
                                            gradOutput,
                                            _inp, _weight,
                                            stride[0], stride[1],
                                            padding[0], padding[1],
                                            dilation[0], dilation[1],
                                            groups)
        return_val = tuple(grad_inp) + tuple(grad_weight)
        if bias is not None:
            return_val = return_val + (grad_bias,)
        return return_val
