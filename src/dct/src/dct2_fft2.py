#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-29 21:07:35
@LastEditTime: 2019-04-30 14:16:09
'''

import numpy as np
import torch
from torch.autograd import Function
from torch import nn
import pdb

import dct2_fft2_cuda as dct_cuda


def precompute_dct2_fft2_expk(N, dtype, device):
    """ Compute 2*exp(-1j*pi*u/(2N)), but not exactly the same.
    The actual return is 2*cos(pi*u/(2N)), 2*sin(pi*u/(2N)).
    This will make later multiplication easier.
    """
    pik_by_2N = torch.arange(N, dtype=dtype, device=device)
    pik_by_2N.mul_(np.pi/(2*N))
    # cos, -sin
    # I use -sin because the real part requires subtraction
    # this will be easier for multiplication
    expk = torch.stack([pik_by_2N.cos(), -1 * pik_by_2N.sin()], dim=-1)

    return expk.contiguous()
    
def dct2(x, expkM, expkN):
    """compute 2D discrete cosine transformation
    """
    if x.is_cuda:
        out = dct_cuda.dct2_fft2(x, expkM, expkN)
    else:
        assert 0, "No CPU Implementation"
    return out

class DCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN):
        return dct2(x, expkM, expkN)

class DCT2(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(DCT2, self).__init__()
        self.expkM = expkM
        self.expkN = expkN
    def forward(self, x): 
        assert self.expkM is not None and self.expkN is not None, "expkM and expkN must be input"
        # if self.expkM is None or self.expkM.size(-1) != x.size(-2):
        #     if x.is_cuda: 
        #         self.expkM = dct_cuda.precompute_dct2_fft2_expk(x.size(-2))
        #     else:
        #         assert 0, "No CPU Implementation"
        # if self.expkN is None or self.expkN.size(-1) != x.size(-1):
        #     if x.is_cuda: 
        #         self.expkN = dct_cuda.precompute_dct2_fft2_expk(x.size(-1))
        #     else:
        #         assert 0, "No CPU Implementation"
        return DCT2Function.apply(x, self.expkM, self.expkN)
        