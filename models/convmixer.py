# https://openreview.net/forum?id=TVHS5Y4dNvM

import torch.nn as nn
from functools import reduce
from operator import __add__

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    kernel_sizes = (kernel_size, kernel_size)
    conv_padding = reduce(__add__, 
    [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_sizes[::-1]])

    return nn.Sequential(
           nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
           nn.GELU(),
           nn.BatchNorm2d(dim),
           *[nn.Sequential(
           Residual(nn.Sequential(
           nn.ZeroPad2d(conv_padding),
           nn.Conv2d(dim, dim, kernel_size, groups=dim),
           nn.GELU(),
           nn.BatchNorm2d(dim)
           )),
           nn.Conv2d(dim, dim, kernel_size=1),
           nn.GELU(),
           nn.BatchNorm2d(dim)
           ) for i in range(depth)],
           nn.AdaptiveAvgPool2d((1,1)),
           nn.Flatten(),
           nn.Linear(dim, n_classes)
           )