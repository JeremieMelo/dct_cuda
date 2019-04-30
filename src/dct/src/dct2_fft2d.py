#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-29 21:07:35
@LastEditTime: 2019-04-29 21:09:15
'''

import numpy as np
import torch
from torch.autograd import Function
from torch import nn
import pdb

import dct2_fft2d_cpp as dct_cpp
import dct2_fft2d_cuda as dct_cuda

def dct2(x, expk0, expk1, buf, out):
    """compute 2D discrete cosine transformation
    """
    if x.is_cuda:
        dct_cuda.dct2(x, expk0, expk1, buf, out)
    else:
        dct_cpp.dct2(x, expk0, expk1, buf, out)
    return out

class DCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1, buf, out):
        return dct2(x, expk0, expk1, buf, out)

class DCT2(nn.Module):
    def __init__(self, expk0=None, expk1=None):
        super(DCT2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
        self.buf = None
        self.out = None
    def forward(self, x): 
        if self.expk0 is None or self.expk0.size(-1) != x.size(-2):
            if x.is_cuda: 
                self.expk0 = dct_cuda.precompute_dct_cos(x.size(-2))
            else:
                self.expk0 = dct_cpp.precompute_dct_cos(x.size(-2))
        if self.expk1 is None or self.expk1.size(-1) != x.size(-1):
            if x.is_cuda: 
                self.expk1 = dct_cuda.precompute_dct_cos(x.size(-1))
            else:
                self.expk1 = dct_cpp.precompute_dct_cos(x.size(-1))
        if self.out is None or self.out.size() != x.size():
            self.buf = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return DCT2Function.apply(x, self.expk0, self.expk1, self.buf, self.out)