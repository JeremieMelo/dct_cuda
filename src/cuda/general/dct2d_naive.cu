#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "global.cuh"

#define TPB (32)

// convert a linear index to a linear index in the transpose
struct transpose_index : public thrust::unary_function<size_t, size_t>
{
    size_t m, n;

    __host__ __device__
    transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

    __host__ __device__
        size_t
        operator()(size_t linear_index)
    {
        size_t i = linear_index / n;
        size_t j = linear_index % n;

        return m * j + i;
    }
};

// transpose an M-by-N array
template <typename T>
void transpose(T *&src_ptr, T *&dst_ptr, size_t M, size_t N)
{
    thrust::device_ptr<T> src_thrust_ptr(src_ptr);
    thrust::device_ptr<T> dst_thrust_ptr(dst_ptr);

    thrust::device_vector<T> src(src_thrust_ptr, src_thrust_ptr + M * N);
    thrust::device_vector<T> dst(dst_thrust_ptr, dst_thrust_ptr + M * N);

    thrust::counting_iterator<size_t> indices(0);

    thrust::gather(thrust::make_transform_iterator(indices, transpose_index(N, M)),
                   thrust::make_transform_iterator(indices, transpose_index(N, M)) + dst.size(),
                   src.begin(), dst.begin());
    dst_ptr = thrust::raw_pointer_cast(dst.data());
}

template <typename T>
__global__ void dct_1d_naive_kernel(const T *x_ptr, T *y_ptr, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int rid = blockIdx.y;
    const T *x = x_ptr + N * rid;
    T *y = y_ptr + N * rid;
    if (tid < N)
    {
        for (int i = 0; i < N; i++)
        {
            y[tid] += x[i] * cos(PI * (i + 0.5) / N * tid);
        }
        y[tid] = y[tid] * 2 / N;
    }
}

template <typename T>
__global__ __launch_bounds__(1024, 2) void dct_2d_kernel_1(T *x, T *y, const int M, const int N)
{
    const int xtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ytid = blockIdx.y * blockDim.y + threadIdx.y;

    if (xtid < N && ytid < M)
    {
        for (int j = 0; j < N; ++j)
        {
            y[ytid * N + xtid] += x[ytid * N + j] * cos(PI * (j + 0.5) / N * xtid);
        }
    }
}

template <typename T>
__global__ __launch_bounds__(1024, 2) void dct_2d_kernel_2(T *x, T *y, const int M, const int N)
{
    const int xtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ytid = blockIdx.y * blockDim.y + threadIdx.y;

    if (xtid < N && ytid < M)
    {
        T tmp = 0;
        for (int j = 0; j < M; ++j)
        {
            tmp += y[j * N + xtid] * cos(PI * (j + 0.5) / M * ytid);
        }
        x[ytid * N + xtid] = tmp / (M * N) * 4;
    }
}

template <typename T>
__global__ __launch_bounds__(1024, 2) void dct_2d_kernel_transpose_1(T *x, T *y, const int M, const int N)
{
    const int xtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ytid = blockIdx.y * blockDim.y + threadIdx.y;

    if (xtid < N && ytid < M)
    {
        for (int j = 0; j < N; ++j)
        {
            y[xtid * M + ytid] += x[ytid * N + j] * cos(PI * (j + 0.5) / N * xtid);
        }
    }
}

template <typename T>
__global__ __launch_bounds__(1024, 2) void dct_2d_kernel_transpose_2(T *x, T *y, const int M, const int N)
{
    const int xtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ytid = blockIdx.y * blockDim.y + threadIdx.y;

    if (xtid < M && ytid < N)
    {
        T tmp = 0;
        for (int j = 0; j < M; ++j)
        {
            tmp += y[ytid * M + j] * cos(PI * (j + 0.5) / M * xtid);
        }
        x[xtid * N + ytid] = tmp / (M * N) * 4;
    }
}

template <typename T>
__global__ void dct_2d_naive_kernel(const T *x, T *y, const int M, const int N)
{
    const int xtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ytid = blockIdx.y * blockDim.y + threadIdx.y;

    if (xtid < N && ytid < M)
    {
        for (int i = 0; i < M; ++i)
        {
            T tmp = 0;
            // #pragma unroll 8
            for (int j = 0; j < N; ++j)
            {
                tmp += x[i * N + j] * cos(PI * (j + 0.5) / N * xtid);
            }
            y[ytid * N + xtid] += tmp * cos(PI * (i + 0.5) / M * ytid);
        }
        y[ytid * N + xtid] = y[ytid * N + xtid] / (M * N) * 4;
    }
}

CpuTimer Timer;
template <typename T>
void dct_2d_naive(const T *h_x, T *h_y, int M, int N)
{
    T *d_x;
    T *d_y;
    size_t size = M * N * sizeof(T);

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMemset(d_y, 0, size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    dim3 gridSize1((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 gridSize2((M + TPB - 1) / TPB, (N + TPB - 1) / TPB, 1);

    cudaDeviceSynchronize();
    Timer.Start();

    #if 1
    dct_2d_kernel_transpose_1<<<gridSize1, blockSize>>>(d_x, d_y, M, N);
    dct_2d_kernel_transpose_2<<<gridSize2, blockSize>>>(d_x, d_y, M, N);
    #elif 1
    dct_2d_kernel_1<<<gridSize, blockSize>>>(d_x, d_y, M, N);
    dct_2d_kernel_2<<<gridSize, blockSize>>>(d_x, d_y, M, N);
    #elif 1
    dct_2d_naive_kernel<<<gridSize, blockSize>>>(d_x, d_y, M, N);
    #endif

    cudaDeviceSynchronize();
    Timer.Stop();

    cudaMemcpy(h_y, d_x, size, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

template <typename T>
int validate(T *result_cuda, T *result_gt, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        int flag = (std::abs(result_cuda[i] - result_gt[i]) / std::abs(result_gt[i])) < epsilon;
        if (flag == 0)
        {
            printf("%d:, cuda_res: %f, gt_res: %f\n", i, result_cuda[i], result_gt[i]);
            // return 0;
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
void load_data(T *&data, T *&result, int &M, int &N)
{
    std::ifstream input_file("test_2d.dat", std::ios_base::in);

    int i = 0;
    T val;
    // int N;
    // int M;
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
        dct_2d_naive<dtype>(h_x, h_y, M, N);
        int flag = validate2D<dtype>(h_y, h_gt, M, N);
        if (!flag)
        {
            printf("[I] Error! Results are incorrect.\n", flag);
            for (int i = 0; i < 64; ++i)
            {
                printf("index: %d, result: %f, GT: %f, scale: %f\n", i, h_y[i], h_gt[i], h_y[i] / h_gt[i]);
            }
        }
        printf("[D] dct 2D takes %g ms\n", Timer.ElapsedMillis());
        total_time += i > 0 ? Timer.ElapsedMillis() : 0;
    }
    printf("[D] dct 2D (%d * %d) takes average %g ms\n", M, N, total_time / (NUM_RUNS - 1));

    delete[] h_x;
    delete[] h_y;
    delete[] h_gt;

    return 0;
}
