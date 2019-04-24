#include <cuda.h>
#include "cuda_runtime.h"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>
#include <cublas_v2.h>
#include <cufft.h>

#define PI (3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481)
#define TPB (32)
#define epsilon (1e-2) //relative error
#define NUM_RUNS (5)

#define checkCUDA(status)                       \
    {                                           \
        if (status != cudaSuccess)              \
        {                                       \
            printf("CUDA Runtime Error: %s\n",  \
                   cudaGetErrorString(status)); \
            assert(status == cudaSuccess);      \
        }                                       \
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

inline __device__ int INDEX(const int hid, const int wid, const int N)
{
    return (hid * N + wid);
}

template <typename T>
__global__ void reorderInput(const T *x, T *y, const int M, const int N)
{
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < M && wid < N)
    {
        int cond = ((hid < M / 2) << 1) | (wid < N / 2);
        int index;
        switch (cond)
        {
        case 0:
            index = INDEX(((M - hid) << 1) - 1, ((N - wid) << 1) - 1, N);
            break;
        case 1:
            index = INDEX(((M - hid) << 1) - 1, wid << 1, N);
            break;
        case 2:
            index = INDEX(hid << 1, ((N - wid) << 1) - 1, N);
            break;
        case 3:
            index = INDEX(hid << 1, wid << 1, N);
            break;
        default:
            assert(0);
            break;
        }
        y[INDEX(hid, wid, N)] = x[index];
    }
}

inline __device__ cufftDoubleComplex complexMul(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
    cufftDoubleComplex res;
    res.x = x.x * y.x - x.y * y.y;
    res.y = x.x * y.y + x.y * y.x;
    return res;
}

inline __device__ cufftDoubleReal RealPartOfMul(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
    return x.x * y.x - x.y * y.y;
}

inline __device__ cufftDoubleComplex complexAdd(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
    cufftDoubleComplex res;
    res.x = x.x + y.x;
    res.y = x.y + y.y;
    return res;
}

inline __device__ cufftDoubleComplex complexAverage(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
    cufftDoubleComplex res;
    res.x = (x.x + y.x) / 2.;
    res.y = (x.y + y.y) / 2.;
    return res;
}

inline __device__ cufftDoubleComplex complexConj(const cufftDoubleComplex &x)
{
    cufftDoubleComplex res;
    res.x = x.x;
    res.y = -1 * x.y;
    return res;
}

__global__ void precomputeExpk(cufftDoubleComplex *expkM, cufftDoubleComplex *expkN, cufftDoubleComplex *expkMconj, const int M, const int N)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < M)
    {
        int hid = tid;
        cufftDoubleComplex W_h_4M = make_double2(cos(PI * hid / (2 * M)), -1 * sin(PI * hid / (M * 2)));
        cufftDoubleComplex W_h_4M_conj = make_double2(W_h_4M.x, -1 * W_h_4M.y);
        expkM[hid] = W_h_4M;
        expkMconj[hid] = W_h_4M_conj;
    }
    if (tid < N)
    {
        int wid = tid;
        cufftDoubleComplex W_w_4N = make_double2(cos(PI * wid / (2 * N)), -1 * sin(PI * wid / (N * 2)));
        // cufftDoubleComplex W_w_4N_conj = make_double2(W_w_4N.x, -1 * W_w_4N.y);
        expkN[wid] = W_w_4N;
        // expkNconj[wid] = W_w_4N_conj;
    }
}

template <typename T>
__global__ void computeMulExpk(const cufftDoubleComplex *V, T *y, const int M, const int N,
                               const cufftDoubleComplex *__restrict__ expkM,
                               const cufftDoubleComplex *__restrict__ expkN,
                               const cufftDoubleComplex *__restrict__ expkMconj)
{
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < M && wid < N)
    {
        if (hid == 0)
        {
            cufftDoubleComplex tmp;
            if (wid <= N / 2)
            {
                tmp = V[wid];
            }
            else
            {
                tmp = complexConj(V[N - wid]);
            }
            // tmp = complexMul(expkM[0], tmp); // expkM[0] = 1
            y[wid] = RealPartOfMul(expkN[wid], tmp) * 4. / (M * N);
        }
        else
        {
            cufftDoubleComplex tmp1, tmp2;
            if (wid <= N / 2)
            {
                tmp1 = V[INDEX(hid, wid, N / 2 + 1)];
                tmp2 = V[INDEX(M - hid, wid, N / 2 + 1)];
            }
            else
            {
                tmp1 = complexConj(V[INDEX(M - hid, N - wid, N / 2 + 1)]);
                tmp2 = complexConj(V[INDEX(hid, N - wid, N / 2 + 1)]);
            }

            tmp1 = complexMul(expkM[hid], tmp1);
            tmp2 = complexMul(expkMconj[hid], tmp2);
            tmp1 = complexAverage(tmp1, tmp2);
            y[INDEX(hid, wid, N)] = RealPartOfMul(expkN[wid], tmp1) * 4. / (M * N);
        }
    }
}

template <typename T>
void dct_1d_z2z(cufftDoubleComplex *d_x,
                cufftDoubleComplex *d_y,
                const int M,
                const int N)
{
    cufftHandle plan;
    int n[1] = {N};
    int BATCH = M / 2;

    cufftPlanMany(&plan, 1, n,
                  NULL, 1, N,
                  NULL, 1, N,
                  CUFFT_Z2Z, BATCH);
    cufftExecZ2Z(plan, d_x, d_y, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    cufftDestroy(plan);
}

template <typename T>
void dct_2d_d2z(cufftDoubleReal *d_x,
                cufftDoubleComplex *d_y,
                const int M,
                const int N)
{
    cufftHandle plan;
    cufftPlan2d(&plan, M, N, CUFFT_D2Z);
    cufftExecD2Z(plan, d_x, d_y);
    cudaDeviceSynchronize();
    cufftDestroy(plan);
}

template <typename T>
void dct_2d_fft(const T *h_x, T *h_y, const int M, const int N)
{
    cufftDoubleReal *d_x;
    T *d_y;
    cufftDoubleComplex *scratch;
    cufftDoubleComplex *expkM, *expkN, *expkMconj;

    if (!isPowerOf2<int>(N) || !isPowerOf2<int>(M))
    {
        printf("Input length is not power of 2.\n");
        assert(0);
    }

    size_t size = M * N * sizeof(T);
    checkCUDA(cudaMalloc((void **)&d_x, size));
    checkCUDA(cudaMalloc((void **)&expkM, M * sizeof(cufftDoubleComplex)));
    checkCUDA(cudaMalloc((void **)&expkN, N * sizeof(cufftDoubleComplex)));
    checkCUDA(cudaMalloc((void **)&expkMconj, M * sizeof(cufftDoubleComplex)));

    checkCUDA(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));
    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    precomputeExpk<<<(std::max(M, N) + 1023) / 1024, 1024>>>(expkM, expkN, expkMconj, M, N);
    cudaDeviceSynchronize();

    timer_start = get_globaltime();
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&scratch, size * 2);

    reorderInput<T><<<gridSize, blockSize>>>(d_x, d_y, M, N);
    cudaDeviceSynchronize();

    dct_2d_d2z<T>(d_y, scratch, M, N);

    computeMulExpk<T><<<gridSize, blockSize>>>(scratch, d_y, M, N, expkM, expkN, expkMconj);
    cudaDeviceSynchronize();
    timer_stop = get_globaltime();

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(scratch);
    cudaFree(expkM);
    cudaFree(expkN);
    cudaFree(expkMconj);
}

template <typename T>
int validate_fft(T *result_cuda, T *result_gt, const int M, const int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int flag = (std::abs(result_cuda[(i * N + j) << 1] - result_gt[(i * N + j) << 1]) / std::abs(result_gt[(i * N + j) << 1])) < epsilon;
            if (flag == 0)
            {
                // printf("cuda_res[%d][%d]: %f, gt_res[%d][%d]: %f\n", i, j, result_cuda[i*N+j], i, j, result_gt[i*N+j]);
                return 0;
            }
        }
    }
    return 1;
}

template <typename T>
int validate2D(T *result_cuda, T *result_gt, const int M, const int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int flag = (std::abs(result_cuda[i * N + j] - result_gt[i * N + j]) / std::abs(result_gt[i * N + j])) < epsilon;
            if (flag == 0)
            {
                // printf("cuda_res[%d][%d]: %f, gt_res[%d][%d]: %f\n", i, j, result_cuda[i*N+j], i, j, result_gt[i*N+j]);
                return 0;
            }
        }
    }
    return 1;
}

template <typename T>
T **allocateMatrix(int M, int N)
{
    T **data;
    data = new T *[M];
    for (int i = 0; i < M; i++)
    {
        data[i] = new T[N];
    }
    return data;
}

template <typename T>
void destroyMatrix(T **&data, int M)
{
    for (int i = 0; i < M; i++)
    {
        delete[] data[i];
    }
    delete[] data;
}

template <typename T>
void load_data(T *&data, T *&result, int &M, int &N)
{
    std::ifstream input_file("test_2d.dat", std::ios_base::in);

    int i = 0;
    T val;
    input_file >> M;
    input_file >> N;
    printf("M: %d\n", M);
    printf("N: %d\n", N);
    data = new T[M * N];
    while (input_file >> val)
    {
        data[i] = val;
        i++;
    }

    std::ifstream input_file2("result_2d.dat", std::ios_base::in);

    i = 0;
    input_file2 >> M;
    input_file2 >> N;
    result = new T[M * N];
    while (input_file2 >> val)
    {
        result[i] = val;
        i++;
    }
    printf("[I] data load done.\n");
}

template <typename T>
void load_data_fft(T *&data, T *&result, int &M, int &N)
{
    std::ifstream input_file("test_2d_fft.dat", std::ios_base::in);

    int i = 0;
    T val, imag;
    input_file >> M;
    input_file >> N;
    printf("M: %d\n", M);
    printf("N: %d\n", N);
    data = new T[M * N];
    while (input_file >> val)
    {
        data[i] = val;
        i++;
    }

    std::ifstream input_file2("result_2d_fft.dat", std::ios_base::in);

    i = 0;
    input_file2 >> M;
    input_file2 >> N;
    result = new T[M * N * 2];
    while (input_file2 >> val >> imag)
    {
        result[i] = val;
        result[i + 1] = imag;
        i += 2;
    }
    printf("[I] data load done.\n");
}

typedef double dtype;
int main()
{
    dtype *h_x;
    dtype *h_y;
    dtype *h_gt;

    int M, N;
    load_data<dtype>(h_x, h_gt, M, N);
    // load_data_fft<dtype>(h_x, h_gt, M, N);
    h_y = new dtype[M * N];
    // h_y = new dtype[M * N * 2];

    double total_time = 0;
    for (int i = 0; i < NUM_RUNS; ++i)
    {
        dct_2d_fft<dtype>(h_x, h_y, M, N);
        int flag = validate2D<dtype>(h_y, h_gt, M, N);
        // int flag = validate_fft<dtype>(h_y, h_gt, M ,N);
        if (!flag)
        {
            printf("[I] Error! Results are incorrect.\n", flag);
            for (int i = 0; i < 4; ++i)
            {
                printf("index: %d, result: %f, GT: %f, scale: %f\n", i, h_y[i], h_gt[i], h_y[i] / h_gt[i]);
            }
        }
        printf("[D] dct 2D takes %g ms\n", (timer_stop - timer_start) * get_timer_period());
        total_time += i > 0 ? (timer_stop - timer_start) * get_timer_period() : 0;
    }

    printf("[D] dct 2D (%d * %d) takes average %g ms\n", M, N, total_time / (NUM_RUNS - 1));

    delete[] h_x;
    delete[] h_y;
    delete[] h_gt;

    return 0;
}
