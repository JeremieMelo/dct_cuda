#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-05-01 12:10:51
@LastEditTime: 2019-05-01 12:35:40
'''
#!/usr/bin/env python
# coding=UTF-8
import sys
sys.path.append('..')

from dct.src import discrete_spectral_transform
import random
import numpy as np
from scipy import fftpack
import scipy
import time
from torch.autograd import Function, Variable
import torch

def gen_input_1d(N=512**2, dim=1):
    if(dim == 1):
        x = torch.empty(N, dtype=torch.float64).uniform_(0, 10.0)
        with open("test_1d.dat", "w") as f:
            f.write("{}\n".format(N))
            for i in range(N):
                f.write("{}\n".format(x[i]))


def gen_input_2d(M=512, N=512, dim=1):
    if(dim == 1):
        x = torch.empty(M, N, dtype=torch.float64).uniform_(0, 10.0)
        # x = torch.Tensor(np.array([[random.randint(0,M-1) for j in range(N)] for i in range(M)])).to(torch.float64)
        x = x.view([M*N])
        with open("test_2d.dat", "w") as f:
            f.write("{}\n".format(M))
            f.write("{}\n".format(N))
            for i in range(M*N):
                f.write("{}\n".format(x[i]))


def gen_output_2d(test_case="test_2d.dat"):
    with open(test_case, "r") as f:
        lines = f.readlines()
        M = int(lines[0].strip())
        N = int(lines[1].strip())
        x = np.resize(np.array([float(i) for i in lines[2:]]).astype(np.float64), [M, N])

    first_row = x[0, :]
    first_column = x[:, 0]
    x = torch.Tensor(x).to(torch.float64)
    first_row = torch.Tensor(first_row).to(torch.float64)
    first_column = torch.Tensor(first_column).to(torch.float64)

    dct_2d(x, M, N)
    # idct_2d(x, M, N)
    # idct2d_idcct2(x, M, N, first_row, first_column)
    # idcct2(x, M, N)
    # idcst2(x, M, N)
    # idsct2(x, M, N)
    idxst_idct(x, M, N)
    # idct_idxst(x, M, N)


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


def dct_2d(x, M, N):
    y = discrete_spectral_transform.dct2_2N(x).numpy()
    y = np.resize(y, [M*N])
    with open("result_2d.dat", "w") as f:
        f.write("{}\n".format(M))
        f.write("{}\n".format(N))
        y = '\n'.join(map(lambda x: str(x), y.tolist()))
        f.write(y)


def idct_2d(x, M, N):
    y = discrete_spectral_transform.idct2_N(x).numpy()
    y = np.resize(y, [M*N])
    with open("idct_2d_result.dat", "w") as f:
        f.write("{}\n".format(M))
        f.write("{}\n".format(N))
        for i in range(M*N):
            f.write("{}\n".format(y[i]))


# use idct_2d to calculate idcct2
def idct2d_idcct2(x, M, N, first_row, first_column):
    y = discrete_spectral_transform.idct2_N(x).numpy()
    y = np.resize(y, [M*N])
    row = discrete_spectral_transform.idct_N(first_row).numpy()
    column = discrete_spectral_transform.idct_N(first_column).numpy()
    with open("idct2d_idcct2.dat", "w") as f:
        f.write("{}\n".format(M))
        f.write("{}\n".format(N))
        for i in range(M*N):
            row_id = i % N
            column_id = i // N
            result = (y[i] + row[row_id] + column[column_id] + x[0][0]) / 4
            f.write("{}\n".format(result))


def idcct2(x, M, N):
    y = discrete_spectral_transform.idcct2(x).numpy()
    y = np.resize(y, [M*N])
    with open("idcct2_result.dat", "w") as f:
        f.write("{}\n".format(M))
        f.write("{}\n".format(N))
        for i in range(M*N):
            f.write("{}\n".format(y[i]))


def idcst2(x, M, N):
    y = discrete_spectral_transform.idcst2(x).numpy()
    y = np.resize(y, [M*N])
    with open("idcst2_result.dat", "w") as f:
        f.write("{}\n".format(M))
        f.write("{}\n".format(N))
        for i in range(M*N):
            f.write("{}\n".format(y[i]))


def idsct2(x, M, N):
    y = discrete_spectral_transform.idsct2(x).numpy()
    y = np.resize(y, [M*N])
    with open("idsct2_result.dat", "w") as f:
        f.write("{}\n".format(M))
        f.write("{}\n".format(N))
        for i in range(M*N):
            f.write("{}\n".format(y[i]))


def idxst_idct(x, M, N):
    y = discrete_spectral_transform.idxst_idct(x).numpy()/M/N
    y = np.resize(y, [M*N])
    with open("idxst_idct.dat", "w") as f:
        f.write("{}\n".format(M))
        f.write("{}\n".format(N))
        for i in range(M*N):
            f.write("{}\n".format(y[i]))


def idct_idxst(x, M, N):
    y = discrete_spectral_transform.idct_idxst(x).numpy()
    y = np.resize(y, [M*N])
    with open("idct_idxst.dat", "w") as f:
        f.write("{}\n".format(M))
        f.write("{}\n".format(N))
        for i in range(M*N):
            f.write("{}\n".format(y[i]))


if __name__ == "__main__":
    gen_input_2d(M=2048, N=4096)
    gen_output_2d()
    # gen_input_1d(1024)
    # dct_1d()
