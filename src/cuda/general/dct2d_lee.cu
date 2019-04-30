#include "global.cuh"

#define TPB (1024)

/// Precompute cosine values needed for N-point dct
/// @param  cos  size N - 1 buffer on GPU, contains the result after function call
/// @param  N    the length of target dct, must be power of 2
template <typename T>
__global__ void precompute_dct_cos_kernel_backup(T *d_cos, T *scratch, int N)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N - 1)
    {
        int sum = N / 2;
        int halfLen = N / 2;
        while (tid >= sum)
        {
            halfLen = halfLen / 2;
            sum += halfLen;
        }
        T phase = (0.5 + tid - (sum - halfLen)) * PI / (halfLen << 1);
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
template <typename T>
__global__ void precompute_dct_cos_kernel(T *d_cos, int N, int log_N)
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

        T phase = (0.5 + i) * PI / len;
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
template <typename T>
void precompute_dct_cos(T *cos, int N)
{
    // The input length must be power of 2
    if (!isPowerOf2<int>(N))
    {
        printf("Input length is not power of 2.\n");
        assert(0);
    }

    // create the array on host
    T *cos_host = new T[N];

    int offset = 0;
    int halfLen = N / 2;
    while (halfLen)
    {
        T phaseStep = PI / (halfLen << 1);
        // T phase_start = 0.5 * phaseStep;
        // #pragma omp parallel for
        for (int i = 0; i < halfLen; ++i)
        {
            T phase = (0.5 + i) * phaseStep;
            cos_host[offset + i] = 0.5 / std::cos(phase);
        }
        offset += halfLen;
        halfLen >>= 1;
    }
    // printf("last cos: %f\n", cos_host[N-1]);
    // while (halfLen)
    // {
    //     T phaseStep = 0.5 * PI / halfLen;
    //     T phase = 0.5 * phaseStep;
    //     for (int i = 0; i < halfLen; ++i)
    //     {
    //         cos_host[offset + i] = 0.5 / std::cos(phase);
    //         phase += phaseStep;
    //     }
    //     offset += halfLen;
    //     halfLen /= 2;
    // }

    // copy to GPU
    cudaMemcpy(cos, cos_host, N * sizeof(T), cudaMemcpyHostToDevice);

    delete[] cos_host;
}

template <typename T, typename TIndex>
__global__ void computeDctForward_1(const T *__restrict__ curr_ptr, T *next_ptr, const T *cos, TIndex N, TIndex len, TIndex halfLen, TIndex cosOffset)
{
    TIndex halfN = (N >> 1);
    TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < halfN)
    {
        TIndex rest = thread_id & (halfN - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2 + blockIdx.y * N;
        T *next = next_ptr + offset + i;
        const T *__restrict__ curr = curr_ptr + offset;

        next[0] = curr[i] + curr[len - i - 1];
        next[halfLen] = (curr[i] - curr[len - i - 1]) * cos[cosOffset + i];
    }
}

template <typename T, typename TIndex>
__global__ void computeDctBackward_1(const T *__restrict__ curr_ptr, T *next_ptr, TIndex N, TIndex len, TIndex halfLen)
{
    TIndex halfN = (N >> 1);
    TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < halfN)
    {
        TIndex rest = thread_id & (halfN - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2 + blockIdx.y * N;
        T *next = next_ptr + offset + i * 2;
        const T *__restrict__ curr = curr_ptr + offset;

        next[0] = curr[i];
        next[1] = (i + 1 == halfLen) ? curr[len - 1] : curr[halfLen + i] + curr[halfLen + i + 1];
    }
}

template <typename T, typename TIndex>
__global__ void computeDctBackward_lasttime_1(const T *__restrict__ curr_ptr, T *next_ptr, TIndex M, TIndex N, TIndex len, TIndex halfLen)
{
    TIndex halfN = (N >> 1);
    TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < halfN)
    {
        TIndex rest = thread_id & (halfN - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2;
        T *next = next_ptr + blockIdx.y + (offset + i * 2) * M;
        const T *__restrict__ curr = curr_ptr + offset + blockIdx.y * N;

        next[0] = curr[i];
        next[M] = (i + 1 == halfLen) ? curr[len - 1] : curr[halfLen + i] + curr[halfLen + i + 1];
    }
}

#define ROW2COL(IDX, COL, N) ((IDX) * (N) + (COL))

template <typename T, typename TIndex>
__global__ void computeDctForward_2(const T *__restrict__ curr, T *next, const T *cos, TIndex M, TIndex N, TIndex len, TIndex halfLen, TIndex cosOffset)
{
    TIndex halfM = (M >> 1);
    TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < halfM)
    {
        TIndex col = blockIdx.y;
        TIndex rest = thread_id & (halfM - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2;

        next[ROW2COL(offset + i, col, N)] = curr[ROW2COL(offset + i, col, N)] + curr[ROW2COL(offset + len - i - 1, col, N)];
        next[ROW2COL(offset + i + halfLen, col, N)] = (curr[ROW2COL(offset + i, col, N)] - curr[ROW2COL(offset + len - i - 1, col, N)]) * cos[cosOffset + i];
    }
}

template <typename T, typename TIndex>
__global__ void computeDctBackward_2(const T *__restrict__ curr, T *next, TIndex M, TIndex N, TIndex len, TIndex halfLen)
{
    TIndex halfM = (M >> 1);
    TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < halfM)
    {
        TIndex col = blockIdx.y;
        TIndex rest = thread_id & (halfM - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2;

        next[ROW2COL(offset + i * 2, col, N)] = curr[ROW2COL(offset + i, col, N)];
        next[ROW2COL(offset + i * 2 + 1, col, N)] = (i + 1 == halfLen) ? curr[ROW2COL(offset + len - 1, col, N)] : curr[ROW2COL(offset + halfLen + i, col, N)] + curr[ROW2COL(offset + halfLen + i + 1, col, N)];
    }
}

template <typename T>
__global__ void normalize(T *x, const T *__restrict__ y, const int M, const int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < M * N)
    {
        x[tid] = y[tid] / (M * N) * 4;
    }
}

template <typename T>
__global__ void normalize4(T *x, const T *__restrict__ y, const int size, T factor)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    double4 *tmp_x = (double4 *)x;
    const double4 *__restrict__ tmp_y = (const double4 *)y;
    tmp_x[tid] = make_double4(tmp_y[tid].x * factor, tmp_y[tid].y * factor, tmp_y[tid].z * factor, tmp_y[tid].w * factor);
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
template <typename T>
void dct_ref_1(const T *vec, T *out, T *buf, const T *cos, int M, int N)
{
    dim3 gridSize((N / 2 + TPB - 1) / TPB, M, 1);
    dim3 blockSize(TPB, 1, 1);

    // Pointers point to the beginning indices of two adjacent iterations
    T *curr = buf;
    T *next = out;

    // 'temp' used to store date of two adjacent iterations
    // Copy 'vec' to the first N element in 'temp'
    cudaMemcpy(curr, vec, M * N * sizeof(T), cudaMemcpyDeviceToDevice);

    // Current bufferfly length and half length
    int len = N;
    int halfLen = len / 2;

    // Iteratively bi-partition sequences into sub-sequences
    int cosOffset = 0;
    while (halfLen)
    {
        computeDctForward_1<<<gridSize, blockSize>>>(curr, next, cos, N, len, halfLen, cosOffset);
        cosOffset += halfLen;
        len = halfLen;
        halfLen /= 2;
        swap(curr, next);
    }

    // Bottom-up form the final DCT solution
    // Note that the case len = 2 will do nothing, so we start from len = 4
    len = 4;
    halfLen = 2;
    while (halfLen < N)
    {
        computeDctBackward_1<<<gridSize, blockSize>>>(curr, next, N, len, halfLen);
        halfLen = len;
        len *= 2;
        swap(curr, next);
    }

    // Populate the final results into 'out'
    if (curr != out)
    {
        swap(out, buf);
    }
}

template <typename T>
void dct_ref_2(const T *vec, T *out, T *buf, const T *cos, int M, int N)
{
    dim3 gridSize((M / 2 + TPB - 1) / TPB, N, 1);
    dim3 blockSize(TPB, 1, 1);
    // int block_count = (N + TPB - 1) / TPB;
    // int thread_count = TPB;

    // Pointers point to the beginning indices of two adjacent iterations
    T *curr = buf;
    T *next = out;

    // 'temp' used to store date of two adjacent iterations
    // Copy 'vec' to the first N element in 'temp'
    cudaMemcpy(curr, vec, M * N * sizeof(T), cudaMemcpyDeviceToDevice);

    // Current bufferfly length and half length
    int len = M;
    int halfLen = len / 2;

    // Iteratively bi-partition sequences into sub-sequences
    int cosOffset = 0;
    while (halfLen)
    {
        computeDctForward_2<<<gridSize, blockSize>>>(curr, next, cos, M, N, len, halfLen, cosOffset);
        cosOffset += halfLen;
        len = halfLen;
        halfLen /= 2;
        swap(curr, next);
    }

    // Bottom-up form the final DCT solution
    // Note that the case len = 2 will do nothing, so we start from len = 4
    len = 4;
    halfLen = 2;
    while (halfLen < M)
    {
        computeDctBackward_2<<<gridSize, blockSize>>>(curr, next, M, N, len, halfLen);
        halfLen = len;
        len *= 2;
        swap(curr, next);
    }

    // Populate the final results into 'out'
    normalize<T><<<(N * M + TPB - 1) / TPB, TPB>>>(out, curr, M, N);
}

template <typename T, typename TIndex>
__global__ __launch_bounds__(1024, 10) void dct_1d_kernel(const T *__restrict__ vec, T *out, const T *cos, const int N, const T factor)
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
        T *next = out + offset + i * 2;
        T *curr = curr_ptr + offset;

        T tmp1 = curr[i] * factor;
        T tmp2 = ((i + 1 == halfLen) ? curr[len - 1] : curr[halfLen + i] + curr[halfLen + i + 1]) * factor;
        *(double2 *)next = make_double2(tmp1, tmp2);
        // next[0] = curr[i];
        // next[1] = (i + 1 == halfLen) ? curr[len - 1] : curr[halfLen + i] + curr[halfLen + i + 1];
    }
    __syncthreads();
}

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

template <typename T, typename TIndex>
__global__ void dct_transpose_normalize_kernel(const T *__restrict__ vec, T *out, const T *cos, const int M, const int N)
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
        for (TIndex thread_id = threadIdx.x; thread_id < halfN; thread_id += blockDim.x)
        {
            TIndex rest = thread_id & (halfN - 1);
            TIndex i = rest & (halfLen - 1);
            TIndex offset = (thread_id - i) * 2;
            T *next = next_ptr + offset + i * 2;
            T *curr = curr_ptr + offset;

            next[0] = curr[i];
            next[1] = (i + 1 == halfLen) ? curr[len - 1] : curr[halfLen + i] + curr[halfLen + i + 1];
        }
        halfLen = len;
        len *= 2;
        __syncthreads();
        swap(curr_ptr, next_ptr);
    }

    for (TIndex thread_id = threadIdx.x; thread_id < halfN; thread_id += blockDim.x)
    {
        TIndex rest = thread_id & (halfN - 1);
        TIndex i = rest & (halfLen - 1);
        TIndex offset = (thread_id - i) * 2;
        T *next = out + blockIdx.y + (offset + i * 2) * M;
        T *curr = curr_ptr + offset;

        next[0] = curr[i] / (M * N) * 4;
        next[M] = ((i + 1 == halfLen) ? curr[len - 1] : curr[halfLen + i] + curr[halfLen + i + 1]) / (M * N) * 4;
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
void dct_transpose_normalize(const T *vec, T *out, const T *cos, int M, int N)
{
    dim3 gridSize(1, M, 1);
    dim3 blockSize(std::min(TPB, N >> 1), 1, 1);
    size_t shared_memory_size = 2 * N * sizeof(T);
    dct_transpose_normalize_kernel<T, int><<<gridSize, blockSize, shared_memory_size>>>(vec, out, cos, M, N);
}

CpuTimer Timer;

template <typename T>
void dct_1d_lee(const T *h_x, T *h_y, const int N)
{
    T *d_x;
    T *d_y;
    T *d_cos0;

    if (!isPowerOf2<int>(N))
    {
        printf("Input length is not power of 2.\n");
        assert(0);
    }

    size_t size = N * sizeof(T);
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&d_cos0, N * sizeof(T)); // row

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    precompute_dct_cos_kernel<T><<<(N + TPB - 1) / TPB, TPB, 0, streams[0]>>>(d_cos0, N, (int)log2(N));
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(std::min(TPB, N >> 1), 1, 1);
    size_t shared_memory_size = 2 * N * sizeof(T);

    cudaDeviceSynchronize();
    Timer.Start();

    dct_1d_kernel<T, int><<<gridSize, blockSize, shared_memory_size>>>(d_x, d_y, d_cos0, N, 2.0 / N);

    cudaDeviceSynchronize();
    Timer.Stop();

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_cos0);
}

template <typename T>
void dct_2d_lee(const T *h_x, T *h_y, const int M, const int N)
{
    T *d_x;
    T *d_y;
    T *scratch;
    T *d_cos0;
    T *d_cos1;

    if (!isPowerOf2<int>(N) || !isPowerOf2<int>(M))
    {
        printf("Input length is not power of 2.\n");
        assert(0);
    }

    size_t size = M * N * sizeof(T);
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&scratch, size);
    cudaMalloc((void **)&d_cos0, N * sizeof(T)); // row
    cudaMalloc((void **)&d_cos1, M * sizeof(T)); // column

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    precompute_dct_cos_kernel<T><<<(N + TPB - 1) / TPB, TPB, 0, streams[0]>>>(d_cos0, N, (int)log2(N));
    precompute_dct_cos_kernel<T><<<(M + TPB - 1) / TPB, TPB, 0, streams[1]>>>(d_cos1, M, (int)log2(M));
    cudaDeviceSynchronize();

    Timer.Start();

    #if 1
    dct_transpose<T>(d_x, scratch, d_cos0, M, N);
    dct_transpose<T>(scratch, d_y, d_cos1, N, M);
    // normalize<T><<<(N * M + TPB - 1) / TPB, TPB>>>(d_y, d_y, M, N);
    normalize4<T><<<(N * M / 4 + TPB - 1) / TPB, TPB>>>(d_y, d_y, M * N / 4, 4. / (M * N));
    #elif 1
    dct_transpose<T>(d_x, scratch, d_cos0, M, N);
    dct_transpose_normalize<T>(scratch, d_y, d_cos1, N, M);
    #elif 0
    dct_ref_1<T>(d_x, d_y, scratch, d_cos0, M, N);
    transpose<T>(d_y, scratch, M, N);
    dct_ref_1<T>(scratch, d_y, scratch, d_cos1, N, M);
    transpose<T>(d_y, scratch, N, M);
    normalize<T><<<(N * M + TPB - 1) / TPB, TPB>>>(d_y, scratch, M, N);
    #elif 1
    dct_ref_1<T>(d_x, d_y, scratch, d_cos0, M, N);
    dct_ref_2<T>(d_y, d_y, scratch, d_cos1, M, N);
    #endif

    cudaDeviceSynchronize();
    Timer.Stop();

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(scratch);
    cudaFree(d_cos0);
    cudaFree(d_cos1);
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
        if (M == 1)
        {
            dct_1d_lee<dtype>(h_x, h_y, N);
        }
        else
        {
            dct_2d_lee<dtype>(h_x, h_y, M, N);
        }
        int flag = validate2D<dtype>(h_y, h_gt, M, N);
        if (!flag)
        {
            printf("[I] Error! Results are incorrect.\n", flag);
            for (int i = 0; i < 5; ++i)
            {
                printf("index: %d, result: %f, GT: %f\n", i, h_y[i], h_gt[i]);
            }
        }
        printf("[D] dct 2D takes %g ms\n", Timer.ElapsedMillis());
        total_time += i > 0 ? Timer.ElapsedMillis() : 0;
    }

    printf("[D] dct 2D (%d * %d) takes average %g ms\n", M, N, total_time / NUM_RUNS);

    delete[] h_x;
    delete[] h_y;
    delete[] h_gt;

    return 0;
}
