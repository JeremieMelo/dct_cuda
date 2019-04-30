#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-13 19:59:17
@LastEditTime: 2019-04-29 21:05:00
'''
##
# @file   dct_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os
import sys
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable
import time
import scipy
from scipy import fftpack
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import dct
from src import dct_lee
#from src import dct_lee as dct
from src import discrete_spectral_transform
sys.path.pop()
import pdb

dtype = torch.float32

class DCTOpTest(unittest.TestCase):
    def test_dctRandom(self):
        N = 4
        x = torch.empty(N, N, dtype=dtype).uniform_(0, 10.0)
        #x = Variable(torch.tensor([[1, 2, 7, 9, 20, 31], [4, 5, 9, 2, 1, 6]], dtype=dtype))

        golden_value = discrete_spectral_transform.dct_2N(x).data.numpy()
        print("golden_value")
        print(golden_value)

        # test cpu using N-FFT
        #pdb.set_trace()
        custom = dct.DCT(algorithm='N')
        dct_value = custom.forward(x)
        print("dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using 2N-FFT
        #pdb.set_trace()
        custom = dct.DCT(algorithm='2N')
        dct_value = custom.forward(x)
        print("dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test gpu
        custom = dct.DCT(algorithm='N')
        dct_value = custom.forward(x.cuda()).cpu()
        print("dct_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test gpu
        custom = dct.DCT(algorithm='2N')
        dct_value = custom.forward(x.cuda()).cpu()
        print("dct_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        #golden_value = discrete_spectral_transform.dct2_2N(x).data.numpy()
        #print("2D golden_value")
        #print(golden_value)

        #custom = dct.DCT()
        #dct2_value = custom.forward(dct_value.cuda().t().contiguous()).cpu()
        #dct2_value = dct2_value.t().contiguous()
        #print("dct2_value cuda")
        #print(dct2_value.data.numpy())

        #np.testing.assert_allclose(dct2_value.data.numpy(), golden_value)

    def test_idctRandom(self):
        N = 4
        x = torch.empty(N, N, dtype=dtype).uniform_(0, 10.0)
        #x = Variable(torch.tensor([[1, 2, 7, 9, 20, 31], [4, 5, 9, 2, 1, 6]], dtype=dtype))
        print("x")
        print(x)

        y = discrete_spectral_transform.dct_N(x)
        print("y")
        print(y.data.numpy())

        golden_value = discrete_spectral_transform.idct_2N(y).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu use N-FFT
        #pdb.set_trace()
        custom = dct.IDCT(algorithm='N')
        dct_value = custom.forward(y)
        print("idct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-5)

        # test cpu use 2N-FFT
        #pdb.set_trace()
        custom = dct.IDCT(algorithm='2N')
        dct_value = custom.forward(y)
        print("idct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-5)

        # test gpu
        custom = dct.IDCT(algorithm='N')
        dct_value = custom.forward(y.cuda()).cpu()
        print("idct_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-5)

        # test gpu
        custom = dct.IDCT(algorithm='2N')
        dct_value = custom.forward(y.cuda()).cpu()
        print("idct_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-5)

    def test_dct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=dtype).uniform_(0, 10.0)

        golden_value = discrete_spectral_transform.dct2_N(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu using N-FFT
        #pdb.set_trace()
        custom = dct.DCT2(algorithm='N')
        dct_value = custom.forward(x)
        print("2D dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using 2N-FFT
        #pdb.set_trace()
        custom = dct.DCT2(algorithm='2N')
        dct_value = custom.forward(x)
        print("2D dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test gpu
        custom = dct.DCT2(algorithm='N')
        dct_value = custom.forward(x.cuda()).cpu()
        print("2D dct_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test gpu
        custom = dct.DCT2(algorithm='2N')
        dct_value = custom.forward(x.cuda()).cpu()
        print("2D dct_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

    def test_idct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.tensor(torch.empty(M, N, dtype=torch.int32).random_(0, 10), dtype=dtype)
        print("2D x")
        print(x)

        y = discrete_spectral_transform.dct2_2N(x)

        golden_value = discrete_spectral_transform.idct2_2N(y).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu using N-FFT
        #pdb.set_trace()
        custom = dct.IDCT2(algorithm='N')
        dct_value = custom.forward(y)
        print("2D dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using 2N-FFT
        #pdb.set_trace()
        custom = dct.IDCT2(algorithm='2N')
        dct_value = custom.forward(y)
        print("2D dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test gpu
        custom = dct.IDCT2(algorithm='N')
        dct_value = custom.forward(y.cuda()).cpu()
        print("2D dct_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test gpu
        custom = dct.IDCT2(algorithm='2N')
        dct_value = custom.forward(y.cuda()).cpu()
        print("2D dct_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

    def test_idxct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        golden_value = discrete_spectral_transform.idxt(x, 0).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        #pdb.set_trace()
        custom = dct.IDXCT()
        dct_value = custom.forward(x)
        print("dxt_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, atol=1e-14)

        # test gpu
        custom = dct.IDXCT()
        dct_value = custom.forward(x.cuda()).cpu()
        print("dxt_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, atol=1e-14)

class DSTOpTest(unittest.TestCase):
    def test_dstRandom(self):
        N = 4
        x = torch.empty(N, N, dtype=dtype).uniform_(0, 10.0)
        #x = Variable(torch.tensor([[1, 2, 7, 9, 20, 31], [4, 5, 9, 2, 1, 6]], dtype=dtype))
        import scipy
        from scipy import fftpack

        #golden_value = discrete_spectral_transform.dst(x).data.numpy()
        golden_value = torch.from_numpy(fftpack.dst(x.data.numpy())).data.numpy() / N
        print("golden_value")
        print(golden_value)

        # test cpu
        #pdb.set_trace()
        custom = dct.DST()
        dst_value = custom.forward(x)
        print("dst_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

        # test gpu
        custom = dct.DST()
        dst_value = custom.forward(x.cuda()).cpu()
        print("dst_value cuda")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

    def test_idstRandom(self):
        N = 4
        x = torch.empty(N, N, dtype=dtype).uniform_(0, 10.0)
        #x = Variable(torch.tensor([[1, 2, 7, 9, 20, 31], [4, 5, 9, 2, 1, 6]], dtype=dtype))
        print("x")
        print(x)
        import scipy
        from scipy import fftpack

        #y = discrete_spectral_transform.dst(x)
        y = torch.from_numpy(fftpack.dst(x.data.numpy()))
        print("y")
        print(y.data.numpy())

        #golden_value = discrete_spectral_transform.idst(y).data.numpy()
        golden_value = torch.from_numpy(fftpack.idst(y.data.numpy())).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        #pdb.set_trace()
        custom = dct.IDST()
        dst_value = custom.forward(y)
        print("idst_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

        # test gpu
        custom = dct.IDST()
        dst_value = custom.forward(y.cuda()).cpu()
        print("idst_value cuda")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

    def test_idxst2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        golden_value = discrete_spectral_transform.idxt(x, 1).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        #pdb.set_trace()
        custom = dct.IDXST()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        # test gpu
        custom = dct.IDXST()
        dst_value = custom.forward(x.cuda()).cpu()
        print("dxt_value cuda")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

class DXTOpTest(unittest.TestCase):
    def test_idcct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        golden_value = discrete_spectral_transform.idcct2(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        #pdb.set_trace()
        custom = dct.IDCCT2()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        # test gpu
        custom = dct.IDCCT2()
        dst_value = custom.forward(x.cuda()).cpu()
        print("dxt_value cuda")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

    def test_idcst2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        golden_value = discrete_spectral_transform.idcst2(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        #pdb.set_trace()
        custom = dct.IDCST2()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        # test gpu
        custom = dct.IDCST2()
        dst_value = custom.forward(x.cuda()).cpu()
        print("dxt_value cuda")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

    def test_idsct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        golden_value = discrete_spectral_transform.idsct2(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        #pdb.set_trace()
        custom = dct.IDSCT2()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        # test gpu
        custom = dct.IDSCT2()
        dst_value = custom.forward(x.cuda()).cpu()
        print("dxt_value cuda")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

def eval_runtime():
    # x = torch.tensor([1, 2, 7, 9, 20, 31], dtype=torch.float64)
    # print(dct_N(x))

    N = 4096
    runs = 100
    # x = torch.empty(10, N, N, dtype=torch.float64).uniform_(0, 10.0).cuda()
    with open("../result_2d.dat", "r") as f:
        lines = f.readlines()
        M = int(lines[0].strip())
        N = int(lines[1].strip())
        x = np.resize(np.array([float(i)
                                for i in lines[2:]]).astype(np.float64), [M, N])
        x = torch.Tensor(x).to(torch.float64).cuda()
    expk0 = discrete_spectral_transform.get_expk(M, dtype=x.dtype, device=x.device)
    expk1 = discrete_spectral_transform.get_expk(N, dtype=x.dtype, device=x.device)

    print("M = {}, N = {}".format(M, N))
    '''
    x_numpy = x.data.cpu().numpy()
    tt = time.time()
    for i in range(runs):
       y = fftpack.dct(fftpack.dct(x_numpy.T, norm=None).T/N, norm=None)/M
    print("CPU: scipy.fftpack.dct2d takes %f ms" % ((time.time()-tt)/runs*1000))

    # 9s for 200 iterations 1024x1024 on GTX 1080
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
       y_2N = discrete_spectral_transform.dct2_2N(x, expk0=expk0, expk1=expk1)
    torch.cuda.synchronize()
    #print(prof)
    print("Pytorch: dct2d_2N takes %.5f ms" % ((time.time()-tt)/runs*1000))

    # 11s for 200 iterations 1024x1024 on GTX 1080
    perm0 = discrete_spectral_transform.get_perm(M, dtype=torch.int64, device=x.device)
    perm1 = discrete_spectral_transform.get_perm(N, dtype=torch.int64, device=x.device)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
       y_N = discrete_spectral_transform.dct2_N(x, perm0=perm0, expk0=expk0, perm1=perm1, expk1=expk1)
    torch.cuda.synchronize()
    #print(prof)
    print("Pytorch: dct2d_N takes %.5f ms" % ((time.time()-tt)/runs*1000))

    dct2func = dct.DCT2(expk0, expk1, algorithm='2N')
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
       y_2N = dct2func.forward(x)
    torch.cuda.synchronize()
    #print(prof)
    print("CUDA: DCT2d_2N Function takes %.5f ms" % ((time.time()-tt)/runs*1000))

    dct2func = dct.DCT2(expk0, expk1, algorithm='N')
    y_N = dct2func.forward(x)
    torch.cuda.synchronize()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    tt = time.time()
    for i in range(runs):
        y_N = dct2func.forward(x)
    torch.cuda.synchronize()
    #print(prof)
    print("CUDA: DCT2d_N Function takes %.5f ms" % ((time.time()-tt)/runs*1000))
    exit()

    dct2func = dct_lee.DCT2(expk0, expk1)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
       y_N = dct2func.forward(x)
    torch.cuda.synchronize()
    #print(prof)
    print("CUDA: DCT2d_Lee Function takes %.5f ms" % ((time.time()-tt)/runs*1000))
    exit()
    '''

    y_N = discrete_spectral_transform.idct2_2N(x, expk0=expk0, expk1=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
       y_N = discrete_spectral_transform.idct2_2N(x, expk0=expk0, expk1=expk1)
    torch.cuda.synchronize()
    #print(prof)
    print("idct2_2N takes %.5f ms" % ((time.time()-tt)/runs*1000))

    idct2func = dct.IDCT2(expk0, expk1, algorithm='2N')
    y_N = idct2func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = idct2func.forward(x)
    torch.cuda.synchronize()
    #print(prof)
    print("IDCT2_2N Function takes %.5f ms" % ((time.time()-tt)/runs*1000))

    idct2func = dct.IDCT2(expk0, expk1, algorithm='N')
    y_N = idct2func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = idct2func.forward(x)
    torch.cuda.synchronize()
    #print(prof)
    print("IDCT2_N Function takes %.5f ms" % ((time.time()-tt)/runs*1000))
    
    y_N = discrete_spectral_transform.idxt(x, 1, expk=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
       y_N = discrete_spectral_transform.idxt(x, 1, expk=expk1)
    torch.cuda.synchronize()
    #print(prof)
    print("idxt takes %.5f ms" % ((time.time()-tt)/runs*1000))

    idxct_func = dct.IDXST(expk1)
    y_N = idxct_func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
       y_N = idxct_func.forward(x)
    torch.cuda.synchronize()
    #print(prof)
    print("IDXCT Function takes %.5f ms" % ((time.time()-tt)/runs*1000))

    # torch.cuda.synchronize()
    # tt = time.time()
    # #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    # for i in range(runs):
    #     y_N = torch.rfft(x[i%10].view([1, N, N]), signal_ndim=2, onesided=False)
    # torch.cuda.synchronize()
    # #print(prof)
    # print("torch.rfft2d takes %.5f ms" % ((time.time()-tt)/runs*1000))
    
    y_N = discrete_spectral_transform.idcct2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = discrete_spectral_transform.idcct2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    #print(prof)
    print("idcct2 takes %.5f ms" % ((time.time()-tt)/runs*1000))

    func = dct.IDCCT2(expk0, expk1)
    y_N = func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = func.forward(x)
    torch.cuda.synchronize()
    #print(prof)
    print("IDCCT2 Function takes %.5f ms" % ((time.time()-tt)/runs*1000))

    y_N = discrete_spectral_transform.idcst2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = discrete_spectral_transform.idcst2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    #print(prof)
    print("idcst2 takes %.5f ms" % ((time.time()-tt)/runs*1000))

    func = dct.IDCST2(expk0, expk1)
    y_N = func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = func.forward(x)
    torch.cuda.synchronize()
    #print(prof)
    print("IDCST2 Function takes %.5f ms" % ((time.time()-tt)/runs*1000))

    y_N = discrete_spectral_transform.idsct2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = discrete_spectral_transform.idsct2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    #print(prof)
    print("idsct2 takes %.5f ms" % ((time.time()-tt)/runs*1000))

    func = dct.IDSCT2(expk0, expk1)
    y_N = func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = func.forward(x)
    torch.cuda.synchronize()
    #print(prof)
    print("IDSCT2 Function takes %.5f ms" % ((time.time()-tt)/runs*1000))


if __name__ == '__main__':
    # torch.manual_seed(10)
    # np.random.seed(10)
    # unittest.main()
    eval_runtime()
