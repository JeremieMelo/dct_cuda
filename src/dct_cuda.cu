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
#define epsilon (1e-4)
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
__global__ void dct_1d_naive_kernel(const T* x, T* y, const int N)
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
void dct_1d_naive(
        const T *h_x,
        T *h_y,
        const int N
        )
{
    T *d_x;
    T *d_y;
    dim3 gridSize((N + TPB - 1) / TPB, 1, 1);
    dim3 blockSize(TPB, 1, 1);
    size_t size = N * sizeof(T);

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    
    dct_1d_naive_kernel<T><<<gridSize, blockSize>>>(d_x, d_y, N);
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
        int flag = std::abs(result_cuda[i] - result_gt[i]) < epsilon;
        if(flag == 0)
        {
            return 0;
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

int main()
{
    float *h_x;
    float *h_y;
    float *h_gt;

    int N = load_data<float>(h_x, h_gt);
    h_y = new float[N];
    

    for(int i = 0;i<10;++i)
    {
        printf("%d: %f\n", i, h_x[i]);
    }
    double total_time = 0;
    for(int i=0; i<NUM_RUNS; ++i)
    {
        timer_start = get_globaltime();
        dct_1d_naive<float>(h_x, h_y, N);
        timer_stop = get_globaltime();
        int flag = validate<float>(h_y, h_gt, N);
        printf("[I] validation: %d\n", flag);
        printf("[D] dct 1D takes %g ms\n", (timer_stop-timer_start)*get_timer_period());
        total_time += (timer_stop-timer_start)*get_timer_period();
    }
        
    // int flag = validate<float>(h_y, h_gt, N);
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
