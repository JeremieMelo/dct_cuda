#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-30 16:06:13
@LastEditTime: 2019-05-01 11:00:08
'''
##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
        name='dct',
        ext_modules=[
            CppExtension('dct_cpp', 
                [
                    'dct.cpp',
                    'dst.cpp',
                    'dxt.cpp', 
                    'dct_2N.cpp'
                    ]),
            CUDAExtension('dct_cuda', 
                [
                    'dct_cuda.cpp',
                    'dct_cuda_kernel.cu',
                    'dst_cuda.cpp',
                    'dst_cuda_kernel.cu',
                    'dxt_cuda.cpp', 
                    'dct_2N_cuda.cpp'
                    ]),
            CppExtension('dct_lee_cpp', 
                [
                    'dct_lee.cpp'
                    ]),
            CUDAExtension('dct_lee_cuda', 
                [
                    'dct_lee_cuda.cpp',
                    'dct_lee_cuda_kernel.cu', 
                    'dct_cuda_kernel.cu',
                    'dst_cuda_kernel.cu'
                    ]),
            CUDAExtension('dct2_fft2_cuda', 
                [
                    'dct2_fft2_cuda.cpp',
                    'dct_cuda_kernel.cu'
                    ]),
            CUDAExtension('idct2_fft2_cuda', 
                [
                    'idct2_fft2_cuda.cpp',
                    'dct_cuda_kernel.cu'
                    ]),
            CUDAExtension('idct_idxst_cuda', 
                [
                    'idct_idxst_cuda.cpp',
                    'dct_cuda_kernel.cu'
                    ]),
            CUDAExtension('idxst_idct_cuda', 
                [
                    'idxst_idct_cuda.cpp',
                    'dct_cuda_kernel.cu'
                    ]),
            CUDAExtension('dct2_lee_cuda', 
                [
                    'dct2_lee_cuda.cpp',
                    'dct2_lee_cuda_kernel.cu'
                    ])
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
