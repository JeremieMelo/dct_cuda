#include <cuda.h>
#include "cuda_runtime.h"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>
#include "../utils/cuda_utils.cuh"

#define TPB (32)
#define epsilon (1e-2)
#define NUM_RUNS (5)
#define DEBUG

/// Precompute cosine values needed for N-point dct
/// @param  cos  size N - 1 buffer on GPU, contains the result after function call
/// @param  N    the length of target dct, must be power of 2
template <typename TValue>
__global__ void precompute_dct_cos_kernel_backup(TValue *d_cos, int N)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N - 1)
    {
        int sum = N / 2;
        int halfLen = N / 2;
        while (tid >= sum) {
            halfLen = halfLen / 2;
            sum += halfLen;
        }
        TValue phase = (0.5 + tid - (sum - halfLen)) * PI / (halfLen << 1);
        d_cos[tid] = 0.5 / cos(phase);
    }
    else if (tid == N - 1)
    {
        d_cos[tid] = 0;
    }
}

/// Precompute cosine values needed for N-point dct
/// @param  cos  size N - 1 buffer on GPU, contains the result after function call
/// @param  N    the length of target dct, must be power of 2
template <typename TValue>
__global__ void precompute_dct_cos_kernel(TValue *d_cos, int N, int log_N)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_height = log_N;
    if (tid < N - 1)
    {
        int k = N - tid - 1;
        // int total_height = LogBase2(N);
        int height = LogBase2(k);
        // int len = N / (1 << (total_height - height - 1));
        int len = 1 << (height + 1);
        int i = len - k - 1;
        
        TValue phase = (0.5 + i) * PI / len;
        d_cos[tid] = 0.5 / cos(phase);
    }
    else if (tid == N - 1)
    {
        d_cos[tid] = 0;
    }
}

/// Precompute cosine values needed for N-point dct
/// @param  cos  size N - 1 buffer on GPU, contains the result after function call
/// @param  N    the length of target dct, must be power of 2
template <typename TValue>
void precompute_dct_cos(TValue *cos, int N)
{
    // The input length must be power of 2
    if (! isPowerOf2<int>(N))
    {
        printf("Input length is not power of 2.\n");
        assert(0); 
    }
  

    // create the array on host 
    TValue* cos_host = new TValue [N]; 

    int offset = 0;
    int halfLen = N / 2;
    while (halfLen)
    {
        TValue phaseStep = PI / (halfLen << 1);
        // TValue phase_start = 0.5 * phaseStep;
        // #pragma omp parallel for
        for (int i = 0; i < halfLen; ++i)
        {
            TValue phase = (0.5 + i) * phaseStep;
            cos_host[offset + i] = 0.5 / std::cos(phase);
        }
        offset += halfLen;
        halfLen >>= 1;
    }
    // printf("last cos: %f\n", cos_host[N-1]);
    // while (halfLen)
    // {
    //     TValue phaseStep = 0.5 * PI / halfLen;
    //     TValue phase = 0.5 * phaseStep;
    //     for (int i = 0; i < halfLen; ++i)
    //     {
    //         cos_host[offset + i] = 0.5 / std::cos(phase);
    //         phase += phaseStep;
    //     }
    //     offset += halfLen;
    //     halfLen /= 2;
    // }

    // copy to GPU 
    cudaMemcpy(cos, cos_host, N*sizeof(TValue), cudaMemcpyHostToDevice);
     

    delete [] cos_host; 
}

template <typename TValue, typename TIndex>
__global__ void computeDctForward(const TValue *curr, TValue *next, const TValue *cos, TIndex N, TIndex len, TIndex halfLen, TIndex cosOffset)
{
    TIndex halfN = (N >> 1);
    TIndex stride = blockDim.x * gridDim.x;
    for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id < halfN; thread_id += stride)
    {
        TIndex rest = thread_id & (halfN - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2;

        next[offset + i] = curr[offset + i] + curr[offset + len - i - 1];
        // next[offset + i + halfLen] = (curr[offset + i] - curr[offset + len - i - 1]) * cos[cosOffset + i];
    }
    // for (TIndex thread_id = halfMN_by_gridDim*blockIdx.x + threadIdx.x; thread_id < halfMN_by_gridDim*(blockIdx.x+1); thread_id += blockDim.x)
    for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id < halfN; thread_id += stride)
    {
        TIndex rest = thread_id & (halfN - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2;

        //next[offset + i] = curr[offset + i] + curr[offset + len - i - 1];
        next[offset + i + halfLen] = (curr[offset + i] - curr[offset + len - i - 1]) * cos[cosOffset + i];
    }
}

template <typename TValue, typename TIndex>
__global__ void computeDctBackward(const TValue *curr, TValue *next, TIndex N, TIndex len, TIndex halfLen)
{
    
    TIndex halfN = (N >> 1);
    // TIndex halfMN = M * halfN;
    //TIndex halfMN_by_gridDim = halfMN/gridDim.x;
    //for (TIndex thread_id = halfMN_by_gridDim*blockIdx.x + threadIdx.x; thread_id < halfMN_by_gridDim*(blockIdx.x+1); thread_id += blockDim.x)
    for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id < halfN; thread_id += blockDim.x * gridDim.x)
    {
        TIndex rest = thread_id & (halfN - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2;

        next[offset + i * 2] = curr[offset + i];
        next[offset + i * 2 + 1] = (i + 1 == halfLen) ? curr[offset + len - 1] : curr[offset + halfLen + i] + curr[offset + halfLen + i + 1];
    }
}

template <typename T>
__global__ void normalize(T* x, const T* y, const int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        x[tid] = y[tid] / N * 2;
    }
}

/// The implementation of fast Discrete Cosine Transform (DCT) algorithm and its inverse (IDCT) are Lee's algorithms
/// Algorithm reference: A New Algorithm to Compute the Discrete Cosine Transform, by Byeong Gi Lee, 1984
///
/// Lee's algorithm has a recursive structure in nature.
/// Here is a sample recursive implementation: https://www.nayuki.io/page/fast-discrete-cosine-transform-algorithms
///   
/// My implementation here is iterative, which is more efficient than the recursive version.
/// Here is a sample iterative implementation: https://www.codeproject.com/Articles/151043/Iterative-Fast-1D-Forvard-DCT

/// Compute y[k] = sum_n=0..N-1 (x[n] * cos((n + 0.5) * k * PI / N)), for k = 0..N-1
/// 
/// @param  vec   length M * N sequence to be transformed in last dimension
/// @param  out   length M * N helping buffer, which is also the output
/// @param  buf   length M * N helping buffer
/// @param  cos   length N - 1, stores cosine values precomputed by function 'precompute_dct_cos'
/// @param  M     length of dimension 0 of vec  
/// @param  N     length of dimension 1 of vec, must be power of 2
template <typename TValue>
void dct_ref(const TValue *vec, TValue *out, TValue* buf, const TValue *cos, int N)
{
    int block_count = (N + TPB - 1) / TPB; 
    int thread_count = TPB; 

    // The input length must be power of 2
    if (! isPowerOf2<int>(N))
    {
        printf("Input length is not power of 2.\n");
        assert(0); 
    }

    // Pointers point to the beginning indices of two adjacent iterations
    TValue *curr = buf; 
    TValue *next = out; 

    // 'temp' used to store date of two adjacent iterations
    // Copy 'vec' to the first N element in 'temp'
    cudaMemcpy(curr, vec, N * sizeof(TValue), cudaMemcpyDeviceToDevice);

    // Current bufferfly length and half length
    int len = N;
    int halfLen = len / 2;

    // Iteratively bi-partition sequences into sub-sequences
    int cosOffset = 0;
    while (halfLen)
    {
        computeDctForward<<<block_count, thread_count>>>(curr, next, cos, N, len, halfLen, cosOffset);
        cosOffset += halfLen;
        len = halfLen;
        halfLen /= 2;
        cudaDeviceSynchronize();
        swap(curr, next);
    }

    // Bottom-up form the final DCT solution
    // Note that the case len = 2 will do nothing, so we start from len = 4
    len = 4;
    halfLen = 2;
    while (halfLen < N)
    {
        computeDctBackward<<<block_count, thread_count>>>(curr, next, N, len, halfLen);
        halfLen = len;
        len *= 2;
        cudaDeviceSynchronize();
        swap(curr, next);
        
    }

    // Populate the final results into 'out'
    normalize<TValue><<<block_count, thread_count>>>(out, curr, N);
    
}

template <typename T>
__global__ void dct_1d_lee_kernel(const T* x, T* y, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < N)
    {
        for (int i = 0; i < N; i++)
        {
            y[tid] += x[i] * __cosf(PI / N * (i + 0.5) * tid);       
        }
        
        y[tid] = y[tid] / N * 2;
    }

}

CpuTimer Timer;

template <typename T>
void dct_1d_lee(
        const T *h_x,
        T *h_y,
        const int N
        )
{
    T *d_x;
    T *d_y;
    T *scratch;
    T *d_cos;
    dim3 gridSize((N + TPB - 1) / TPB, 1, 1);
    dim3 blockSize(TPB, 1, 1);
    size_t size = N * sizeof(T);

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&scratch, size);
    cudaMalloc((void **)&d_cos, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    
    
    // precompute_dct_cos<T>(d_cos, N);
    precompute_dct_cos_kernel<T><<<gridSize, blockSize>>>(d_cos, N, (int)log2(N));
    cudaDeviceSynchronize();
    Timer.Start();
    dct_ref<T>(d_x, d_y, scratch, d_cos, N);
    cudaDeviceSynchronize();
    Timer.Stop();
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}

template <typename T>
int validate(T* result_cuda, T* result_gt, const int N)
{
    for(int i = 0; i < N; ++i)
    {
        int flag = (std::abs(result_cuda[i] - result_gt[i]) / std::abs(result_gt[i])) < epsilon;
        if(flag == 0)
        {
            printf("%d:, cuda_res: %f, gt_res: %f\n", i, result_cuda[i], result_gt[i]);
            // return 0;
        }
    }
    return 1;
}

template <typename T>
int load_data(T* &data, T* &result)
{
    std::ifstream input_file("test_1d.dat", std::ios_base::in);

    int i = 0;
    T val;
    int N;
    input_file >> N;
    printf("N: %d\n", N);
    data = new T[N];
    while(input_file >> val)
    {
        data[i] = val;
        i++;
    }

    std::ifstream input_file2("result_1d.dat", std::ios_base::in);

    i = 0;
    input_file2 >> N;
    result = new T[N];
    while(input_file2 >> val)
    {
        result[i] = val;
        i++;
    }
    printf("[I] data load done.\n");
    return N;
}

typedef double dtype;
int main()
{
    dtype *h_x;
    dtype *h_y;
    dtype *h_gt;

    int N = load_data<dtype>(h_x, h_gt);
    h_y = new dtype[N];
    
    double total_time = 0;
    for (int i = 0; i < NUM_RUNS; ++i)
    {
        dct_1d_lee<dtype>(h_x, h_y, N);
        int flag = validate<dtype>(h_y, h_gt, N);
        if (!flag)
        {
            printf("[I] Error! Results are incorrect.\n", flag);
            for (int i = 0; i < 64; ++i)
            {
                printf("index: %d, result: %f, GT: %f, scale: %f\n", i, h_y[i], h_gt[i], h_y[i] / h_gt[i]);
            }
        }
        printf("[D] dct1d_lee takes %g ms\n", Timer.ElapsedMillis());
        total_time += i > 0 ? Timer.ElapsedMillis() : 0;
    }

    printf("[D] dct1d_lee (%d) takes average %g ms\n", N, total_time / (NUM_RUNS - 1));

    delete [] h_x;
    delete [] h_y;
    delete [] h_gt;

    return 0;
}
