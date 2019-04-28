#include <cuda.h>
#include "cuda_runtime.h"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>
#include <cufft.h>

#define PI (3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481)
#define TPB (16)
#define NUM_RUNS (101)

#if 0
typedef float dtype;
typedef cufftReal dtypeReal;
typedef cufftComplex dtypeComplex;
#define epsilon (5e-1) //relative error
#else
typedef double dtype;
typedef cufftDoubleReal dtypeReal;
typedef cufftDoubleComplex dtypeComplex;
#define epsilon (1e-2) //relative error
#endif

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
__global__ void idct2d_postprocess_backup(const T *x, T *y, const int M, const int N, const int halfN)
{
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < M && wid < N)
    {
        int index;
        int cond = (((hid & 1) == 0) << 1) | ((wid & 1) == 0);
        switch (cond)
        {
        case 0:
            index = INDEX(2 * M - (hid + 1), N - (wid + 1) / 2, halfN);
            break;
        case 1:
            index = INDEX(2 * M - (hid + 1), wid / 2, halfN);
            break;
        case 2:
            index = INDEX(hid, N - (wid + 1) / 2, halfN);
            break;
        case 3:
            index = INDEX(hid, wid / 2, halfN);
            break;
        default:
            break;
        }
        y[INDEX(hid, wid, N)] = x[index] / 4;
    }
}

template <typename T>
__global__ void idct2d_postprocess(const T *x, T *y, const int M, const int N, const int halfN)
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
        y[index] = x[INDEX(hid, wid, N)] / 4;
    }
}

inline __device__ cufftDoubleComplex complexMul(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
    cufftDoubleComplex res;
    res.x = x.x * y.x - x.y * y.y;
    res.y = x.x * y.y + x.y * y.x;
    return res;
}

inline __device__ cufftComplex complexMul(const cufftComplex &x, const cufftComplex &y)
{
    cufftComplex res;
    res.x = x.x * y.x - x.y * y.y;
    res.y = x.x * y.y + x.y * y.x;
    return res;
}

inline __device__ cufftDoubleReal RealPartOfMul(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
    return x.x * y.x - x.y * y.y;
}

inline __device__ cufftReal RealPartOfMul(const cufftComplex &x, const cufftComplex &y)
{
    return x.x * y.x - x.y * y.y;
}

inline __device__ cufftDoubleReal ImaginaryPartOfMul(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
    return x.x * y.y + x.y * y.x;
}

inline __device__ cufftReal ImaginaryPartOfMul(const cufftComplex &x, const cufftComplex &y)
{
    return x.x * y.y + x.y * y.x;
}

inline __device__ cufftDoubleComplex complexAdd(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
    cufftDoubleComplex res;
    res.x = x.x + y.x;
    res.y = x.y + y.y;
    return res;
}

inline __device__ cufftComplex complexAdd(const cufftComplex &x, const cufftComplex &y)
{
    cufftComplex res;
    res.x = x.x + y.x;
    res.y = x.y + y.y;
    return res;
}

inline __device__ cufftDoubleComplex complexSubtract(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
    cufftDoubleComplex res;
    res.x = x.x - y.x;
    res.y = x.y - y.y;
    return res;
}

inline __device__ cufftComplex complexSubtract(const cufftComplex &x, const cufftComplex &y)
{
    cufftComplex res;
    res.x = x.x - y.x;
    res.y = x.y - y.y;
    return res;
}

inline __device__ cufftDoubleComplex complexConj(const cufftDoubleComplex &x)
{
    cufftDoubleComplex res;
    res.x = x.x;
    res.y = -1 * x.y;
    return res;
}

inline __device__ cufftComplex complexConj(const cufftComplex &x)
{
    cufftComplex res;
    res.x = x.x;
    res.y = -1 * x.y;
    return res;
}

__global__ void precomputeExpk(cufftDoubleComplex *expkM, cufftDoubleComplex *expkN, const int M, const int N)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < M)
    {
        int hid = tid;
        cufftDoubleComplex W_h_4M = make_double2(cos(PI * hid / (2 * M)), -1 * sin(PI * hid / (M * 2)));
        expkM[hid] = W_h_4M;
    }
    if (tid <= N / 2)
    {
        int wid = tid;
        cufftDoubleComplex W_w_4N = make_double2(cos(PI * wid / (2 * N)), -1 * sin(PI * wid / (N * 2)));
        expkN[wid] = W_w_4N;
    }
}

__global__ void precomputeExpk(cufftComplex *expkM, cufftComplex *expkN, const int M, const int N)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < M)
    {
        int hid = tid;
        cufftComplex W_h_4M = make_float2(__cosf((float)PI * hid / (2 * M)), -1 * __sinf((float)PI * hid / (M * 2)));
        expkM[hid] = W_h_4M;
    }
    if (tid <= N / 2)
    {
        int wid = tid;
        cufftComplex W_w_4N = make_float2(__cosf((float)PI * wid / (2 * N)), -1 * __sinf((float)PI * wid / (N * 2)));
        expkN[wid] = W_w_4N;
    }
}

template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void idct2d_preprocess(const T *input, TComplex *output, const int M, const int N,
                                                                  const int halfM, const int halfN,
                                                                  const TComplex *__restrict__ expkM, const TComplex *__restrict__ expkN)
{
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < halfM && wid < halfN)
    {
        int cond = ((hid != 0) << 1) | (wid != 0);
        switch (cond)
        {
        case 0:
        {
            T tmp1;
            TComplex tmp_up;

            output[0].x = input[0];
            output[0].y = 0;

            tmp1 = input[halfN];
            tmp_up.x = tmp1;
            tmp_up.y = tmp1;
            output[halfN] = complexConj(complexMul(expkN[halfN], tmp_up));

            tmp1 = input[INDEX(halfM, 0, N)];
            tmp_up.x = tmp1;
            tmp_up.y = tmp1;
            output[INDEX(halfM, 0, halfN + 1)] = complexConj(complexMul(expkM[halfM], tmp_up));

            tmp1 = input[INDEX(halfM, halfN, N)];
            tmp_up.x = 0;
            tmp_up.y = 2 * tmp1;
            output[INDEX(halfM, halfN, halfN + 1)] = complexConj(complexMul(complexMul(expkM[halfM], expkN[halfN]), tmp_up));
            break;
        }

        case 1:
        {
            TComplex tmp_up;
            tmp_up.x = input[wid];
            tmp_up.y = input[N - wid];
            output[wid] = complexConj(complexMul(expkN[wid], tmp_up));

            T tmp1 = input[INDEX(halfM, wid, N)];
            T tmp2 = input[INDEX(halfM, N - wid, N)];
            tmp_up.x = tmp1 - tmp2;
            tmp_up.y = tmp1 + tmp2;
            output[INDEX(halfM, wid, halfN + 1)] = complexConj(complexMul(complexMul(expkM[halfM], expkN[wid]), tmp_up));
            break;
        }

        case 2:
        {
            T tmp1, tmp3;
            TComplex tmp_up, tmp_down;

            tmp1 = input[INDEX(hid, 0, N)];
            tmp3 = input[INDEX(M - hid, 0, N)];
            tmp_up.x = tmp1;
            tmp_up.y = tmp3;
            tmp_down.x = tmp3;
            tmp_down.y = tmp1;

            output[INDEX(hid, 0, halfN + 1)] = complexConj(complexMul(expkM[hid], tmp_up));
            output[INDEX(M - hid, 0, halfN + 1)] = complexConj(complexMul(expkM[M - hid], tmp_down));

            tmp1 = input[INDEX(hid, halfN, N)];
            tmp3 = input[INDEX(M - hid, halfN, N)];
            tmp_up.x = tmp1 - tmp3;
            tmp_up.y = tmp3 + tmp1;
            tmp_down.x = tmp3 - tmp1;
            tmp_down.y = tmp1 + tmp3;

            output[INDEX(hid, halfN, halfN + 1)] = complexConj(complexMul(complexMul(expkM[hid], expkN[halfN]), tmp_up));
            output[INDEX(M - hid, halfN, halfN + 1)] = complexConj(complexMul(complexMul(expkM[M - hid], expkN[halfN]), tmp_down));
            break;
        }

        case 3:
        {
            T tmp1 = input[INDEX(hid, wid, N)];
            T tmp2 = input[INDEX(hid, N - wid, N)];
            T tmp3 = input[INDEX(M - hid, wid, N)];
            T tmp4 = input[INDEX(M - hid, N - wid, N)];
            TComplex tmp_up, tmp_down;
            tmp_up.x = tmp1 - tmp4;
            tmp_up.y = tmp3 + tmp2;
            tmp_down.x = tmp3 - tmp2;
            tmp_down.y = tmp1 + tmp4;

            output[INDEX(hid, wid, halfN + 1)] = complexConj(complexMul(complexMul(expkM[hid], expkN[wid]), tmp_up));
            output[INDEX(M - hid, wid, halfN + 1)] = complexConj(complexMul(complexMul(expkM[M - hid], expkN[wid]), tmp_down));
            break;
        }

        default:
            assert(0);
            break;
        }
    }
}

template <typename T>
void makeCufftPlan(const int M, const int N, cufftHandle *plan) {}

template <>
void makeCufftPlan<cufftComplex>(const int M, const int N, cufftHandle *plan)
{
    cufftPlan2d(plan, M, N, CUFFT_C2R);
}

template <>
void makeCufftPlan<cufftDoubleComplex>(const int M, const int N, cufftHandle *plan)
{
    cufftPlan2d(plan, M, N, CUFFT_Z2D);
}

void ifft2D(cufftDoubleComplex *d_x, cufftDoubleReal *d_y, const int M, const int N, cufftHandle &plan)
{
    cufftExecZ2D(plan, d_x, d_y);
    cudaDeviceSynchronize();
}

void ifft2D(cufftComplex *d_x, cufftReal *d_y, const int M, const int N, cufftHandle &plan)
{
    cufftExecC2R(plan, d_x, d_y);
    cudaDeviceSynchronize();
}

template <typename T, typename TReal = cufftDoubleReal, typename TComplex = cufftDoubleComplex>
void dct_2d_fft(const T *h_x, T *h_y, const int M, const int N)
{
    T *d_x;
    T *d_y;
    TComplex *scratch;
    TComplex *expkM, *expkN;

    if (!isPowerOf2<int>(N) || !isPowerOf2<int>(M))
    {
        printf("Input length is not power of 2.\n");
        assert(0);
    }

    size_t size = M * N * sizeof(T);
    checkCUDA(cudaMalloc((void **)&d_x, size));
    checkCUDA(cudaMalloc((void **)&d_y, size));
    checkCUDA(cudaMalloc((void **)&expkM, M * sizeof(TComplex)));
    checkCUDA(cudaMalloc((void **)&expkN, (N / 2 + 1) * sizeof(TComplex)));
    checkCUDA(cudaMalloc((void **)&scratch, M * (N / 2 + 1) * sizeof(TComplex)));
    checkCUDA(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

    cufftHandle plan;
    makeCufftPlan<TComplex>(M, N, &plan);

    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 gridSize2((N / 2 + TPB - 1) / TPB, (M / 2 + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    precomputeExpk<<<(std::max(M, N) + 1023) / 1024, 1024>>>(expkM, expkN, M, N);
    cudaDeviceSynchronize();

    timer_start = get_globaltime();
    idct2d_preprocess<T, TComplex><<<gridSize2, blockSize>>>(d_x, scratch, M, N, M / 2, N / 2, expkM, expkN);
    cudaDeviceSynchronize();

    ifft2D(scratch, d_x, M, N, plan);

    idct2d_postprocess<T><<<gridSize, blockSize>>>(d_x, d_y, M, N, N / 2);
    cudaDeviceSynchronize();
    timer_stop = get_globaltime();

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(scratch);
    cudaFree(expkM);
    cudaFree(expkN);
    cufftDestroy(plan);
}

template <typename T>
int validate2D(T *result_cuda, T *result_gt, const int M, const int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int flag;
            if (std::abs(result_gt[i * N + j]) < 1e-6)
            {
                flag = (std::abs(result_cuda[i * N + j] - result_gt[i * N + j])) < epsilon / 100.;
            }
            else
            {
                flag = (std::abs(result_cuda[i * N + j] - result_gt[i * N + j]) / std::abs(result_gt[i * N + j])) < epsilon;
            }
            if (flag == 0)
            {
                printf("cuda_res[%d][%d]: %f, gt_res[%d][%d]: %f\n", i, j, result_cuda[i * N + j], i, j, result_gt[i * N + j]);
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
    std::ifstream input_file("result_2d.dat", std::ios_base::in);

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

    std::ifstream input_file2("test_2d.dat", std::ios_base::in);

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

int main()
{
    dtype *h_x;
    dtype *h_y;
    dtype *h_gt;

    int M, N;
    load_data<dtype>(h_x, h_gt, M, N);
    h_y = new dtype[M * N];

    double total_time = 0;
    for (int i = 0; i < NUM_RUNS; ++i)
    {
        dct_2d_fft<dtype, dtypeReal, dtypeComplex>(h_x, h_y, M, N);
        int flag = validate2D<dtype>(h_y, h_gt, M, N);
        if (!flag)
        {
            printf("[I] Error! Results are incorrect.\n", flag);
            for (int i = 0; i < 64; ++i)
            {
                printf("index: %d, result: %f, GT: %f, scale: %f\n", i, h_y[i], h_gt[i], h_y[i] / h_gt[i]);
            }
        }
        printf("[D] idct 2D takes %g ms\n", (timer_stop - timer_start) * get_timer_period());
        total_time += i > 0 ? (timer_stop - timer_start) * get_timer_period() : 0;
    }

    printf("[D] idct 2D (%d * %d) takes average %g ms\n", M, N, total_time / (NUM_RUNS - 1));

    delete[] h_x;
    delete[] h_y;
    delete[] h_gt;

    return 0;
}
