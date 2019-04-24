/*
 * @Author: Jake Gu
 * @Date: 2019-04-02 16:34:45
 * @LastEditTime: 2019-04-23 15:46:29
 */
/**
 * @file   dct_2N_cuda.cpp
 * @author Yibo Lin
 * @date   Nov 2018
 */
#include "dct_cuda.h"

at::Tensor dct_2N_forward(
        at::Tensor x,
        at::Tensor expk) 
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(expk);
    CHECK_CONTIGUOUS(expk);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    auto x_pad = at::zeros({M, 2*N}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct_2N_forward", [&] {
            computePadCudaLauncher<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    x_pad.data<scalar_t>()
                    );

            auto y = at::rfft(x_pad, 1, false, true);

            // re-use x_pad as output 
            x_pad.resize_({M, N});
            computeMulExpk_2N_CudaLauncher(
                    y.data<scalar_t>(), 
                    expk.data<scalar_t>(), 
                    M, 
                    N, 
                    x_pad.data<scalar_t>()
                    );
            x_pad.mul_(1.0/N);
            });

    return x_pad; 
}

at::Tensor idct_2N_forward(
        at::Tensor x,
        at::Tensor expk) 
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(expk);
    CHECK_CONTIGUOUS(expk);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    auto x_pad = at::zeros({M, 2*N, 2}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct_2N_forward", [&] {
            computeMulExpkAndPad_2N_CudaLauncher<scalar_t>(
                    x.data<scalar_t>(), 
                    expk.data<scalar_t>(), 
                    M, 
                    N, 
                    x_pad.data<scalar_t>()
                    );

            // y is real now 
            auto y = at::irfft(x_pad, 1, false, false, {2*N});

            // reuse x_pad 
            x_pad.resize_({M, N});
            computeTruncationCudaLauncher(
                    y.data<scalar_t>(), 
                    M, 
                    N, 
                    x_pad.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";

            // this is to match python implementation 
            // normal way should be multiply by 0.25*N
            x_pad.mul_(N); 
    });

    return x_pad; 
}

at::Tensor dct2_2N_forward(
        at::Tensor x,
        at::Tensor expk0, 
        at::Tensor expk1) 
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(expk0);
    CHECK_CONTIGUOUS(expk0);
    CHECK_GPU(expk1);
    CHECK_CONTIGUOUS(expk1);

    // 1D DCT to columns

    //std::cout << "x\n" << x << "\n";
    auto N = x.size(-1);
    auto M = x.numel()/N; 
    auto x_pad = at::zeros({M, 2*N}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct2_2N_forward", [&] {
            computePadCudaLauncher<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    x_pad.data<scalar_t>()
                    );

            auto y = at::rfft(x_pad, 1, false, true);

            // re-use x_pad as output 
            x_pad.resize_({M, N});
            computeMulExpk_2N_CudaLauncher(
                    y.data<scalar_t>(), 
                    expk1.data<scalar_t>(), 
                    M, 
                    N, 
                    x_pad.data<scalar_t>()
                    );
            //x_pad.mul_(1.0/N);

            // 1D DCT to rows 
            auto xt = x_pad.transpose(-2, -1).contiguous();
            // I do not want to allocate memory another time 
            // must zero-out x_pad 
            x_pad.resize_({N, 2*M}).zero_();
            computePadCudaLauncher<scalar_t>(
                    xt.data<scalar_t>(), 
                    N, 
                    M, 
                    x_pad.data<scalar_t>()
                    );

            y = at::rfft(x_pad, 1, false, true);
            //y.mul_(1.0/M);

            // re-use x_reorder as output 
            x_pad.resize_({N, M});
            computeMulExpk_2N_CudaLauncher(
                    y.data<scalar_t>(), 
                    expk0.data<scalar_t>(), 
                    N, 
                    M, 
                    x_pad.data<scalar_t>()
                    );

            x_pad.mul_(1.0/(M*N));
            x_pad.transpose_(-2, -1);
    });

    return x_pad.contiguous(); 
}

at::Tensor idct2_2N_forward(
        at::Tensor x,
        at::Tensor expk0, 
        at::Tensor expk1) 
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(expk0);
    CHECK_CONTIGUOUS(expk0);
    CHECK_GPU(expk1);
    CHECK_CONTIGUOUS(expk1);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    // 1D DCT to columns

    auto x_pad = at::zeros({M, 2*N, 2}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct2_2N_forward", [&] {
            computeMulExpkAndPad_2N_CudaLauncher<scalar_t>(
                    x.data<scalar_t>(), 
                    expk1.data<scalar_t>(), 
                    M, 
                    N, 
                    x_pad.data<scalar_t>()
                    );

            // y is real now 
            auto y = at::irfft(x_pad, 1, false, false, {2*N});

            // reuse x_pad 
            x_pad.resize_({M, N});
            computeTruncationCudaLauncher(
                    y.data<scalar_t>(), 
                    M, 
                    N, 
                    x_pad.data<scalar_t>()
                    );

            // this is to match python implementation 
            // normal way should be multiply by 0.25*N
            //x_pad.mul_(N); 

            // 1D DCT to rows 
            auto xt = x_pad.transpose(-2, -1).contiguous();
            x_pad.resize_({N, 2*M, 2}).zero_();
            computeMulExpkAndPad_2N_CudaLauncher<scalar_t>(
                    xt.data<scalar_t>(), 
                    expk0.data<scalar_t>(), 
                    N, 
                    M, 
                    x_pad.data<scalar_t>()
                    );

            // y is real now 
            y = at::irfft(x_pad, 1, false, false, {2*M});

            // reuse x_pad 
            x_pad.resize_({N, M});
            computeTruncationCudaLauncher(
                    y.data<scalar_t>(), 
                    N, 
                    M, 
                    x_pad.data<scalar_t>()
                    );

            // this is to match python implementation 
            // normal way should be multiply by 0.25*0.25*M*N
            x_pad.mul_(M*N);
            x_pad.transpose_(-2, -1);
    });

    return x_pad.contiguous(); 
}
