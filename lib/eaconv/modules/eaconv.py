import math
import collections

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from .util import _pair
from eaconv.functions import EAConv2dFunction


def _check_input_dimensions(input_list):
    for i in range(1, len(input_list)):
        if input_list[i - 1].size(0) != input_list[i].size(0)\
                or input_list[i - 1].size(2) != input_list[i].size(2)\
                or input_list[i - 1].size(3) != input_list[i].size(3):
            return False
    return True


class _EAConvNd(Module):
    def __init__(self, in_channels_list, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_EAConvNd, self).__init__()
        in_channels = sum(in_channels_list)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            raise NotImplementedError
        else:
            self.weight = torch.nn.ParameterList(
                [Parameter(torch.cuda.FloatTensor(
                    out_channels, c // groups, *kernel_size))
                    for c in in_channels_list])
            # for i, param in enumerate(self.weight):
            #     setattr(self, 'weight' + str(i), param)
        if bias:
            self.bias = Parameter(torch.cuda.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        for i in range(len(self.weight)):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class EAConv2d(_EAConvNd):
    '''A 2-dimensional convolution layer with efficient aggregation

    Overall, the APIs are the same as torch.nn.Conv2d with a few exceptions:
    1. The in_channels argument is replaced with in_channels_list,
    which accepts either a single int or a list of ints.
    2. This module accepts a variable number of inputs. The number of inputs
    must match the length of in_channels_list.
    '''

    def __init__(self, in_channels_list, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        if groups != 1:
            raise NotImplementedError
        if not isinstance(in_channels_list, collections.Iterable):
            in_channels_list = (in_channels_list,)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(EAConv2d, self).__init__(
            in_channels_list, out_channels, kernel_size,
            stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, *inp):
        if not _check_input_dimensions(inp):
            raise ValueError('all except the channel dimensions '
                             'of input tesors must coincide')
        eaconv2dfunc = EAConv2dFunction(self.stride, self.padding,
                                        self.dilation, self.groups)
        if self.bias is None:
            return eaconv2dfunc(*inp, *self.weight)
        else:
            return eaconv2dfunc(*inp, *self.weight, self.bias)
