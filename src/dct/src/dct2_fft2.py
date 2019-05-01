#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-29 21:07:35
@LastEditTime: 2019-04-30 19:38:48
'''

import numpy as np
import torch
from torch.autograd import Function
from torch import nn
import pdb

import dct2_fft2_cuda
import idct_idxst_cuda
import idxst_idct_cuda
import idct2_fft2_cuda


def precompute_expk(N, dtype, device):
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


class DCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2_cuda.dct2_fft2(x, expkM, expkN, out, buf)
            return out
        else:
            assert 0, "No CPU Implementation"

class DCT2(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(DCT2, self).__init__()
        self.expkM = expkM
        self.expkN = expkN
        self.out = None
        self.buf = None

    def forward(self, x, M, N):
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
        if self.out is None or self.out.size() != x.size():
            self.out = torch.empty_like(x)
            self.buf = torch.empty(M, N / 2 + 1, 2, device=x.device)
        return DCT2Function.apply(x, self.expkM, self.expkN, self.out, self.buf)


def idct_idxst(x, expkM, expkN):
    """compute 2D discrete cosine transformation
    """
    if x.is_cuda:
        out = idct_idxst_cuda.idct_idxst(x, expkM, expkN)
    else:
        assert 0, "No CPU Implementation"
    return out


class IDCT_IDXSTFunction(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN):
        return idct_idxst(x, expkM, expkN)


class IDCT_IDXST(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(IDCT_IDXST, self).__init__()
        self.expkM = expkM
        self.expkN = expkN

    def forward(self, x):
        assert self.expkM is not None and self.expkN is not None, "expkM and expkN must be input"
        return IDCT_IDXSTFunction.apply(x, self.expkM, self.expkN)

# idxst_idct


def idxst_idct(x, expkM, expkN):
    """compute 2D discrete cosine transformation
    """
    if x.is_cuda:
        out = idxst_idct_cuda.idxst_idct(x, expkM, expkN)
    else:
        assert 0, "No CPU Implementation"
    return out


class IDXST_IDCTFunction(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN):
        return idxst_idct(x, expkM, expkN)


class IDXST_IDCT(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(IDXST_IDCT, self).__init__()
        self.expkM = expkM
        self.expkN = expkN

    def forward(self, x):
        assert self.expkM is not None and self.expkN is not None, "expkM and expkN must be input"
        return IDXST_IDCTFunction.apply(x, self.expkM, self.expkN)

# idct2_fft2


def idct2(x, expkM, expkN):
    """compute 2D discrete cosine transformation
    """
    if x.is_cuda:
        out = idct2_fft2_cuda.idct2_fft2(x, expkM, expkN)
    else:
        assert 0, "No CPU Implementation"
    return out


class IDCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN):
        return idct2(x, expkM, expkN)


class IDCT2(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(IDCT2, self).__init__()
        self.expkM = expkM
        self.expkN = expkN

    def forward(self, x):
        assert self.expkM is not None and self.expkN is not None, "expkM and expkN must be input"
        return IDCT2Function.apply(x, self.expkM, self.expkN)
