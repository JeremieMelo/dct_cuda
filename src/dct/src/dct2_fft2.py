#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-29 21:07:35
@LastEditTime: 2019-05-01 12:07:25
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
    def __init__(self, M, N, dtype=torch.float64, device=torch.cuda, expkM=None, expkN=None):
        super(DCT2, self).__init__()

        if expkM is None or expkM.size(-1) != M or expkM.dtype != dtype:
            self.expkM = precompute_expk(M, dtype=dtype, device=device)
        else:
            self.expkM = expkM.to(device)

        if expkN is None or expkN.size(-1) != N or expkN.dtype != dtype:
            self.expkN = precompute_expk(N, dtype=dtype, device=device)
        else:
            self.expkN = expkN.to(device)

        self.out = torch.empty(M, N, dtype=dtype, device=device)
        self.buf = torch.empty(M, N / 2 + 1, 2, dtype=dtype, device=device)

    def forward(self, x):
        return DCT2Function.apply(x, self.expkM, self.expkN, self.out, self.buf)


class IDCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            idct2_fft2_cuda.idct2_fft2(x, expkM, expkN, out, buf)
            return out
        else:
            assert 0, "No CPU Implementation"


class IDCT2(nn.Module):
    def __init__(self, M, N, dtype=torch.float64, device=torch.cuda, expkM=None, expkN=None):
        super(IDCT2, self).__init__()

        if expkM is None or expkM.size(-1) != M or expkM.dtype != dtype:
            self.expkM = precompute_expk(M, dtype=dtype, device=device)
        else:
            self.expkM = expkM.to(device)

        if expkN is None or expkN.size(-1) != N or expkN.dtype != dtype:
            self.expkN = precompute_expk(N, dtype=dtype, device=device)
        else:
            self.expkN = expkN.to(device)

        self.out = torch.empty(M, N, dtype=dtype, device=device)
        self.buf = torch.empty(M, N / 2 + 1, 2, dtype=dtype, device=device)

    def forward(self, x):
        return IDCT2Function.apply(x, self.expkM, self.expkN, self.out, self.buf)


class IDCT_IDXSTFunction(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            idct_idxst_cuda.idct_idxst(x, expkM, expkN, out, buf)
            return out
        else:
            assert 0, "No CPU Implementation"


class IDCT_IDXST(nn.Module):
    def __init__(self, M, N, dtype=torch.float64, device=torch.cuda, expkM=None, expkN=None):
        super(IDCT_IDXST, self).__init__()

        if expkM is None or expkM.size(-1) != M or expkM.dtype != dtype:
            self.expkM = precompute_expk(M, dtype=dtype, device=device)
        else:
            self.expkM = expkM.to(device)

        if expkN is None or expkN.size(-1) != N or expkN.dtype != dtype:
            self.expkN = precompute_expk(N, dtype=dtype, device=device)
        else:
            self.expkN = expkN.to(device)

        self.out = torch.empty(M, N, dtype=dtype, device=device)
        self.buf = torch.empty(M, N / 2 + 1, 2, dtype=dtype, device=device)

    def forward(self, x):
        return IDCT_IDXSTFunction.apply(x, self.expkM, self.expkN, self.out, self.buf)


class IDXST_IDCTFunction(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            idxst_idct_cuda.idxst_idct(x, expkM, expkN, out, buf)
            return out
        else:
            assert 0, "No CPU Implementation"


class IDXST_IDCT(nn.Module):
    def __init__(self, M, N, dtype=torch.float64, device=torch.cuda, expkM=None, expkN=None):
        super(IDXST_IDCT, self).__init__()

        if expkM is None or expkM.size(-1) != M or expkM.dtype != dtype:
            self.expkM = precompute_expk(M, dtype=dtype, device=device)
        else:
            self.expkM = expkM.to(device)

        if expkN is None or expkN.size(-1) != N or expkN.dtype != dtype:
            self.expkN = precompute_expk(N, dtype=dtype, device=device)
        else:
            self.expkN = expkN.to(device)

        self.out = torch.empty(M, N, dtype=dtype, device=device)
        self.buf = torch.empty(M, N / 2 + 1, 2, dtype=dtype, device=device)

    def forward(self, x):
        return IDXST_IDCTFunction.apply(x, self.expkM, self.expkN, self.out, self.buf)
