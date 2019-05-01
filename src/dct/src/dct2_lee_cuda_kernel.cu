#include <cassert>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_utils.cuh"
#include "dct2_lee_cuda.h"

namespace lee
{

#define TPB 512

template <typename T, typename TIndex>
__global__ __launch_bounds__(1024, 10) void dct_transpose_kernel(const T *__restrict__ vec, T *out, const T *cos, const int M, const int N)
{
    extern __shared__ T sdata[];
    T *curr_ptr = sdata;
    T *next_ptr = curr_ptr + N;

    for (TIndex i = threadIdx.x; i < N; i += blockDim.x)
    {
        curr_ptr[i] = vec[blockIdx.y * N + i];
    }
    __syncthreads();

    // Current bufferfly length and half length
    int len = N;
    int halfLen = len / 2;
    // Iteratively bi-partition sequences into sub-sequences
    int cosOffset = 0;

    const TIndex halfN = halfLen;
    while (halfLen)
    {
        #pragma unroll 2
        for (TIndex thread_id = threadIdx.x; thread_id < halfN; thread_id += blockDim.x)
        {
            TIndex rest = thread_id & (halfN - 1);
            TIndex i = rest & (halfLen - 1);
            TIndex offset = (thread_id - i) * 2;
            T *next = next_ptr + offset + i;
            T *curr = curr_ptr + offset;

            next[0] = curr[i] + curr[len - i - 1];
            next[halfLen] = (curr[i] - curr[len - i - 1]) * cos[cosOffset + i];
        }
        cosOffset += halfLen;
        len = halfLen;
        halfLen /= 2;
        __syncthreads();
        swap(curr_ptr, next_ptr);
    }

    // Bottom-up form the final DCT solution
    // Note that the case len = 2 will do nothing, so we start from len = 4
    len = 4;
    halfLen = 2;
    while (len < N)
    {
        #pragma unroll 2
        for (TIndex thread_id = threadIdx.x; thread_id < halfN; thread_id += blockDim.x)
        {
            TIndex rest = thread_id & (halfN - 1);
            TIndex i = rest & (halfLen - 1);
            TIndex offset = (thread_id - i) * 2;
            T *next = next_ptr + offset + i * 2;
            T *curr = curr_ptr + offset;

            T tmp1 = curr[i];
            T tmp2 = (i + 1 == halfLen) ? curr[len - 1] : curr[halfLen + i] + curr[halfLen + i + 1];

            *(double2 *)next = make_double2(tmp1, tmp2);
            // next[0] = curr[i];
            // next[1] = (i + 1 == halfLen) ? curr[len - 1] : curr[halfLen + i] + curr[halfLen + i + 1];
        }
        halfLen = len;
        len *= 2;
        __syncthreads();
        swap(curr_ptr, next_ptr);
    }
    #pragma unroll 2
    for (TIndex thread_id = threadIdx.x; thread_id < halfN; thread_id += blockDim.x)
    {
        TIndex rest = thread_id & (halfN - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2;
        T *next = out + blockIdx.y + (offset + i * 2) * M;
        T *curr = curr_ptr + offset;

        next[0] = curr[i];
        next[M] = (i + 1 == halfLen) ? curr[len - 1] : curr[halfLen + i] + curr[halfLen + i + 1];
    }
    __syncthreads();
}

template <typename T>
void dct_transpose(const T *vec, T *out, const T *cos, int M, int N)
{
    dim3 gridSize(1, M, 1);
    dim3 blockSize(std::min(TPB, N >> 1), 1, 1);
    size_t shared_memory_size = 2 * N * sizeof(T);
    dct_transpose_kernel<T, int><<<gridSize, blockSize, shared_memory_size>>>(vec, out, cos, M, N);
}

template <typename T>
void dct2(const T *x, T *y, T* scratch, const T* cos0, const T* cos1, const int M, const int N)
{
    dct_transpose<T>(x, scratch, cos0, M, N);
    dct_transpose<T>(scratch, y, cos1, N, M);
    cudaDeviceSynchronize();
}

} // End of namespace lee

#define REGISTER_DCT2_KERNEL_LAUNCHER(type) \
    void instantiateDct2CudaLauncher(\
        const type* x, \
        type* y, \
        type* scratch, \
        const type* cos0, \
        const type* cos1, \
        const int M, \
        const int N \
        ) \
    { \
        lee::dct2<type>( \
                x, \
                y, \
                scratch, \
                cos0, \
                cos1, \
                M, \
                N \
                ); \
    }

// REGISTER_DCT2_KERNEL_LAUNCHER(float);
REGISTER_DCT2_KERNEL_LAUNCHER(double);
