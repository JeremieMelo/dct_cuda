#include <cuda.h>
#include "cuda_runtime.h"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>

#define PI (3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481)
#define TPB (32)
#define epsilon (1e-2)
#define NUM_RUNS (5)
#define DEBUG

#define checkCUDA(status) \
{\
	if (status != cudaSuccess) { \
		printf("CUDA Runtime Error: %s\n", \
			cudaGetErrorString(status)); \
		assert(status == cudaSuccess); \
	} \
}

typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

inline hr_clock_rep get_globaltime(void)
{
	using namespace std::chrono;
	return high_resolution_clock::now().time_since_epoch().count();
}

// Returns the period in miliseconds
inline double get_timer_period(void)
{
	using namespace std::chrono;
	return 1000.0 * high_resolution_clock::period::num / high_resolution_clock::period::den;
}

hr_clock_rep timer_start, timer_stop;

/// Return true if a number is power of 2
template <typename T = unsigned>
inline bool isPowerOf2(T val)
{
    return val && (val & (val - 1)) == 0;
}

template <typename T>
inline void swap(T& x, T& y)
{
    T tmp = x; 
    x = y; 
    y = tmp; 
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
    // while (halfLen)
    // {
    //     TValue phaseStep = PI / (halfLen << 1);
    //     TValue phase_start = 0.5 * phaseStep;
    //     #pragma omp parallel for
    //     for (int i = 0; i < halfLen; ++i)
    //     {
    //         TValue phase = phase_start + i * phaseStep;
    //         cos_host[offset + i] = 0.5 / std::cos(phase);
    //     }
    //     offset += halfLen;
    //     halfLen >>= 1;
    // }
    while (halfLen)
    {
        TValue phaseStep = 0.5 * PI / halfLen;
        TValue phase = 0.5 * phaseStep;
        for (int i = 0; i < halfLen; ++i)
        {
            cos_host[offset + i] = 0.5 / std::cos(phase);
            phase += phaseStep;
        }
        offset += halfLen;
        halfLen /= 2;
    }

    // copy to GPU 
    cudaMemcpy(cos, cos_host, N*sizeof(TValue), cudaMemcpyHostToDevice); 

    delete [] cos_host; 
}

template <typename TValue, typename TIndex>
__global__ void computeDctForward(const TValue *curr, TValue *next, const TValue *cos, TIndex N, TIndex len, TIndex halfLen, TIndex cosOffset)
{
    TIndex halfN = (N >> 1);
    // TIndex halfMN = M * halfN;
    //for (TIndex thread_id = halfMN_by_gridDim*blockIdx.x + threadIdx.x; thread_id < halfMN_by_gridDim*(blockIdx.x+1); thread_id += blockDim.x)
    for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id < halfN; thread_id += blockDim.x * gridDim.x)
    {
        TIndex rest = thread_id & (halfN - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2;

        next[offset + i] = curr[offset + i] + curr[offset + len - i - 1];
        //next[offset + i + halfLen] = (curr[offset + i] - curr[offset + len - i - 1]) * cos[cosOffset + i];
    }
    //for (TIndex thread_id = halfMN_by_gridDim*blockIdx.x + threadIdx.x; thread_id < halfMN_by_gridDim*(blockIdx.x+1); thread_id += blockDim.x)
    for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id < halfN; thread_id += blockDim.x * gridDim.x)
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
        cudaDeviceSynchronize();
        swap(curr, next);
        cosOffset += halfLen;
        len = halfLen;
        halfLen /= 2;
    }

    // Bottom-up form the final DCT solution
    // Note that the case len = 2 will do nothing, so we start from len = 4
    len = 4;
    halfLen = 2;
    while (halfLen < N)
    {
        computeDctBackward<<<block_count, thread_count>>>(curr, next, N, len, halfLen);
        cudaDeviceSynchronize();
        swap(curr, next);
        halfLen = len;
        len *= 2;
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
    
    precompute_dct_cos<T>(d_cos, N);
    dct_ref<T>(d_x, d_y, scratch, d_cos, N);
    cudaDeviceSynchronize();
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
    

    for(int i = 0;i<10;++i)
    {
        printf("%d: %f\n", i, h_x[i]);
    }
    double total_time = 0;
    for(int i=0; i<NUM_RUNS; ++i)
    {
        timer_start = get_globaltime();
        dct_1d_lee<dtype>(h_x, h_y, N);
        timer_stop = get_globaltime();
        int flag = validate<dtype>(h_y, h_gt, N);
        printf("[I] validation: %d\n", flag);
        printf("[D] dct 1D takes %g ms\n", (timer_stop-timer_start)*get_timer_period());
        total_time += (timer_stop-timer_start)*get_timer_period();
    }
        
    // int flag = validate<dtype>(h_y, h_gt, N);
    // printf("[D] dct 1D takes %g ms\n", (timer_stop-timer_start)*get_timer_period());
    printf("[D] dct 1D takes average %g ms\n", total_time/NUM_RUNS);
    // printf("[I] validation: %d\n", flag);
    
    for(int i = 0;i<10;++i)
    {
        printf("%d: %f\n", i, h_y[i]);
    }

    delete [] h_x;
    delete [] h_y;
    delete [] h_gt;

    return 0;
}
