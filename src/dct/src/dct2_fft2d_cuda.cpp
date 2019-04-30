/*
 * @Author: Jake Gu
 * @Date: 2019-04-02 16:34:45
 * @LastEditTime: 2019-04-29 21:20:51
 */

#include "dct_cuda.h"
#include "dct_lee_cuda.h"

at::Tensor dct2_fft2d_precompute_dct_expk(int N)
{
    typedef double T; 

    auto out = at::empty(N, torch::CUDA(at::kDouble));

    lee::precompute_dct_cos(out.data<T>(), N);

    return out; 
}


void dct2_fft2d_forward(
        at::Tensor x,
        at::Tensor expkM, 
        at::Tensor expkN,
        at::Tensor expkMN_1,
        at::Tensor expkMN_2, 
        at::Tensor buf, 
        at::Tensor out
        ) 
{
    //CHECK_GPU(x);
    //CHECK_CONTIGUOUS(x);
    //CHECK_GPU(cos0);
    //CHECK_CONTIGUOUS(cos0);
    //CHECK_GPU(cos1);
    //CHECK_CONTIGUOUS(cos1);

    // 1D DCT to columns

    //std::cout << "x\n" << x << "\n";
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct2_fft2d_forward", [&] {
                buf.copy_(x);
                negateOddEntriesCudaLauncher<scalar_t>(
                        buf.data<scalar_t>(), 
                        M, 
                        N
                        );

                dct_lee_forward(buf, expk, buf, out);
                //std::cout << "y\n" << y << "\n";

                computeFlipCudaLauncher<scalar_t>(
                        out.data<scalar_t>(), 
                        M, 
                        N, 
                        buf.data<scalar_t>()
                        );
                });

    out.copy_(buf);

    out.resize_({M, N});
    buf.resize_({M, N});

    dct_lee_forward(x, cos1, out, buf); 

    // 1D DCT to rows 
    out.resize_({N, M}); 
    out.copy_(buf.transpose(-2, -1)); 
    buf.resize_({N, M}); 

    dct_lee_forward(out, cos0, out, buf); 

    out.resize_({M, N}); 
    out.copy_(buf.transpose(-2, -1)); 
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("precompute_dct_expk", &dct2_fft2d_precompute_dct_expk, "Precompute DCT expk");
  m.def("dct2", &dct2_fft2d_forward, "DCT2 forward");
}
