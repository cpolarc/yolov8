import torch.nn as nn
import numpy as np
import torch

import math

class Conv2d(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,act=True):
        super(Conv2d, self).__init__()
        self.in_channels = c1
        self.out_channels = c2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.act = act
        # Initialize weights and bias
        self.cv2 = nn.Conv2d(c1, c2, 1, stride, autopad(1, padding, dilation), groups=groups, dilation=dilation, bias=False)  # add 1x1 conv
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, autopad(kernel_size, padding, dilation), groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        self.weight = nn.Parameter(torch.Tensor(c2, c1 // groups, *kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(c2))
        else:
            self.register_parameter('bias_param', None)

        # Initialize weights and bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        return self.act(self.bn(self.conv(x)))
