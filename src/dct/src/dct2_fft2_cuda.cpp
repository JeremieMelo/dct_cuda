/*
 * @Author: Jake Gu
 * @Date: 2019-04-02 16:34:45
 * @LastEditTime: 2019-04-30 17:21:04
 */

#include "dct_cuda.h"


at::Tensor dct2_fft2_forward(
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

    auto out = at::empty({M, N}, x.options());
    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct2_fft2_forward", [&] {

                dct2dPreprocessCudaLauncher<scalar_t>(
                        x.data<scalar_t>(),
                        out.data<scalar_t>(),
                        M,
                        N
                        );
                        
                auto buf = at::rfft(out, 2, false, true);
                
                dct2dPostprocessCudaLauncher<scalar_t>(
                        buf.data<scalar_t>(),
                        out.data<scalar_t>(),
                        M, 
                        N,
                        expkM.data<scalar_t>(),
                        expkN.data<scalar_t>()
                );
                });
        return out.contiguous();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dct2_fft2", &dct2_fft2_forward, "DCT2 FFT2D");
}
