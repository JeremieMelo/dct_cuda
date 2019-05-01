/*
 * @Author: Jake Gu
 * @Date: 2019-05-01 10:47:31
 * @LastEditTime: 2019-05-01 11:01:01
 */
#ifndef GPUPLACE_DCT2_LEE_CUDA_H
#define GPUPLACE_DCT2_LEE_CUDA_H

namespace lee
{

template <typename TValue>
void dct2(const TValue *x, TValue *y, TValue* scratch, const TValue *cos0, const TValue *cos1, const int M, const int N);

} // End of namespace lee

#endif