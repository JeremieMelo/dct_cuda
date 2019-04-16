#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-15 19:19:32
@LastEditTime: 2019-04-16 17:03:30
'''
import torch
from torch.autograd import Function, Variable
import time 
import scipy
from scipy import fftpack
import numpy as np

def gen_testcase(N = 256**2, dim=1):
    if(dim == 1):
        x = torch.empty(N, dtype=torch.float64).uniform_(0, 10.0)
        with open("test_1d.dat", "w") as f:
            f.write(f"{N}\n")
            for i in range(N):
                f.write(f"{x[i]:.10f}\n")

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
        f.write(f"{N}\n")
        for i in range(N):
            f.write(f"{y[i]:.10f}\n")    
                
def dct_2d():
    N = 512
    runs = 10
    # x = torch.empty(10, N, N, dtype=torch.float64).uniform_(0, 10.0).cuda()

    x_numpy = x.data.cpu().numpy()
    tt = time.time()
    for i in range(runs): 
       y = fftpack.dct(fftpack.dct(x_numpy[0].T, norm=None).T/N, norm=None)/N
    print("CPU: scipy takes %.3f ms" % ((time.time()-tt)/runs*1000))

if __name__ == "__main__":
    gen_testcase()
    dct_1d()
    