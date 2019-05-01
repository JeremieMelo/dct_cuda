/*
 * @Author: Jake Gu
 * @Date: 2019-04-02 16:34:45
 * @LastEditTime: 2019-04-30 22:34:26
 */

#include "dct_cuda.h"

void idct2_fft2_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf)
{
    CHECK_GPU(x);
    CHECK_GPU(expkM);
    CHECK_GPU(expkN);
    CHECK_GPU(out);
    CHECK_GPU(buf);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(expkM);
    CHECK_CONTIGUOUS(expkN);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct2_fft2_forward", [&] {
        idct2_fft2PreprocessCudaLauncher<scalar_t>(
            x.data<scalar_t>(),
            buf.data<scalar_t>(),
            M,
            N,
            expkM.data<scalar_t>(),
            expkN.data<scalar_t>());

        auto y = at::irfft(buf, 2, false, true, {M, N});

        idct2_fft2PostprocessCudaLauncher<scalar_t>(
            y.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("idct2_fft2", &idct2_fft2_forward, "IDCT2 FFT2D");
}
