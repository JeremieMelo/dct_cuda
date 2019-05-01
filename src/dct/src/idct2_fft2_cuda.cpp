/*
 * @Author: Jake Gu
 * @Date: 2019-04-02 16:34:45
 * @LastEditTime: 2019-04-30 20:17:18
 */

#include "dct_cuda.h"


at::Tensor idct2_fft2_forward(
        at::Tensor x,
        at::Tensor expkM, 
        at::Tensor expkN
        ) 
{
    CHECK_GPU(x);
    CHECK_GPU(expkM);
    CHECK_GPU(expkN);
    // CHECK_GPU(buf);
    // CHECK_GPU(out);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(expkM);
    CHECK_CONTIGUOUS(expkN);
    // CHECK_CONTIGUOUS(buf);
    // CHECK_CONTIGUOUS(out);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    auto buf = at::empty({M, N+2}, x.options());
    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct2_fft2_forward", [&] {

                idct2_fft2PreprocessCudaLauncher<scalar_t>(
                        x.data<scalar_t>(),
                        buf.data<scalar_t>(),
                        M,
                        N,
                        expkM.data<scalar_t>(),
                        expkN.data<scalar_t>()
                        );
                
                buf.resize_({M, N/2+1, 2});
                auto y = at::irfft(buf, 2, false, true, {M,N});
                
                idct2_fft2PostprocessCudaLauncher<scalar_t>(
                        y.data<scalar_t>(),
                        buf.data<scalar_t>(),
                        M, 
                        N
                );
                });
        buf.resize_({M,N+2});
        return buf.contiguous().slice(1,0,N,1);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("idct2_fft2", &idct2_fft2_forward, "IDCT IDXST FFT2D");
}
