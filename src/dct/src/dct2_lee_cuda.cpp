/*
 * @Author: Jake Gu
 * @Date: 2019-05-01 10:33:16
 * @LastEditTime: 2019-05-01 10:50:12
 */
#include "dct_cuda.h"
#include "dct2_lee_cuda.h"

inline void dct2_lee_forward(
        at::Tensor x,
        at::Tensor cos0,
        at::Tensor cos1, 
        at::Tensor buf,
        at::Tensor out 
        ) 
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(cos0);
    CHECK_CONTIGUOUS(cos0);
    CHECK_GPU(cos1);
    CHECK_CONTIGUOUS(cos1);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct2_lee_forward", [&] {
            lee::dct2(
                    x.data<scalar_t>(), 
                    out.data<scalar_t>(), 
                    buf.data<scalar_t>(), 
                    cos0.data<scalar_t>(),
                    cos1.data<scalar_t>(), 
                    M, 
                    N
                    );
            });

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dct2_lee", &dct2_lee_forward, "DCT2 LEE forward");
}