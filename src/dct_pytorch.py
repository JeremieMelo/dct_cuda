#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-15 19:19:32
@LastEditTime: 2019-04-25 16:26:53
'''
import torch
from torch.autograd import Function, Variable
import time
import scipy
from scipy import fftpack
import numpy as np
import random


def gen_testcase_1d(N=512**2, dim=1):
    if(dim == 1):
        x = torch.empty(N, dtype=torch.float64).uniform_(0, 10.0)
        with open("test_1d.dat", "w") as f:
            f.write("{}\n".format(N))
            for i in range(N):
                f.write("{}\n".format(x[i]))


def gen_testcase_2d(M=512, N=512, dim=1):
    if(dim == 1):
        x = torch.empty(M, N, dtype=torch.float64).uniform_(0, 10.0)
        # x = torch.Tensor(np.array([[random.randint(0,M-1) for j in range(N)] for i in range(M)])).to(torch.float64)
        x = x.view([M*N])
        with open("test_2d.dat", "w") as f:
            f.write("{}\n".format(M))
            f.write("{}\n".format(N))
            for i in range(M*N):
                f.write("{}\n".format(x[i]))


def dct_1d(test_case="test_1d.dat"):
    runs = 2
    with open(test_case, "r") as f:
        lines = f.readlines()
        N = int(lines[0].strip())
        x = np.array([float(i) for i in lines[1:]]).astype(np.float64)

    tt = time.time()
    for i in range(runs):
        y = fftpack.dct(x.T, norm=None)/N
    print("CPU: scipy takes %.3f ms" % ((time.time()-tt)/runs*1000))

    with open("result_1d.dat", "w") as f:
        f.write("{}\n".format(N))
        for i in range(N):
            f.write("{}\n".format(y[i]))


def dct_2d(test_case="test_2d.dat"):
    runs = 2
    with open(test_case, "r") as f:
        lines = f.readlines()
        M = int(lines[0].strip())
        N = int(lines[1].strip())
        x = np.resize(np.array([float(i)
                                for i in lines[2:]]).astype(np.float64), [M, N])

    tt = time.time()
    for i in range(runs):
        y = fftpack.dct(fftpack.dct(x.T, norm=None).T/N, norm=None)/M
    print("CPU: scipy takes %.3f ms" % ((time.time()-tt)/runs*1000))

    y = np.resize(y, [M*N])
    with open("result_2d.dat", "w") as f:
        f.write("{}\n".format(M))
        f.write("{}\n".format(N))
        for i in range(M*N):
            f.write("{}\n".format(y[i]))


def fft_2d(test_case="test_2d_fft.dat"):
    runs = 2
    with open(test_case, "r") as f:
        lines = f.readlines()
        M = int(lines[0].strip())
        N = int(lines[1].strip())
        x = np.resize(np.array([float(i)
                                for i in lines[2:]]).astype(np.float64), [M, N])

    x_r = np.zeros_like(x)
    x_r[0:M//2, 0:N//2] = x[0:M:2, 0:N:2]
    x_r[M//2:, 0:N//2] = x[M:0:-2, 0:N:2]
    x_r[0:M//2, N//2:] = x[0:M:2, N:0:-2]
    x_r[M//2:, N//2:] = x[M:0:-2, N:0:-2]
    print(x)
    print(x_r)
    tt = time.time()
    for i in range(runs):
        y = fftpack.fft2(x_r)
    print("CPU: scipy fft2 takes %.3f ms" % ((time.time()-tt)/runs*1000))
    print(y)
    y = np.resize(y, [M * N])
    with open("result_2d_fft.dat", "w") as f:
        f.write("{}\n".format(M))
        f.write("{}\n".format(N))
        for i in range(M*N):
            f.write("{}\n".format(y[i].real))
            f.write("{}\n".format(y[i].imag))


if __name__ == "__main__":
    gen_testcase_2d(M=1024, N=1024)
    dct_2d()
