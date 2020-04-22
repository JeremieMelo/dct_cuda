#include "global.cuh"

#define TPB (16)

template <typename T>
__global__ void dct2d_preprocess_backup(const T *x, T *y, const int M, const int N, const int halfM, const int halfN)
{
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < M && wid < N)
    {
        int cond = ((hid < halfM) << 1) | (wid < halfN);
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

template <typename T>
__global__ void dct2d_preprocess(const T *x, T *y, const int M, const int N, const int halfN)
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
            index = INDEX((M<<1) - (hid + 1), N - ((wid + 1) >> 1), halfN);
            break;
        case 1:
            index = INDEX((M<<1) - (hid + 1), (wid >>1), halfN);
            break;
        case 2:
            index = INDEX(hid, N - ((wid + 1) >> 1), halfN);
            break;
        case 3:
            index = INDEX(hid, (wid >> 1), halfN);
            break;
        default:
            break;
        }
        y[index] = x[INDEX(hid, wid, N)];
    }
}

__global__ void precomputeExpk(cufftDoubleComplex *expkM, cufftDoubleComplex *expkN, const int M, const int N)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid <= M / 2)
    {
        int hid = tid;
        cufftDoubleComplex W_h_4M = make_double2(cos(PI * hid / (2 * M)), -sin(PI * hid / (M * 2)));
        expkM[hid] = W_h_4M;
    }
    if (tid <= N / 2)
    {
        int wid = tid;
        cufftDoubleComplex W_w_4N = make_double2(cos(PI * wid / (2 * N)), -sin(PI * wid / (N * 2)));
        expkN[wid] = W_w_4N;
    }
}

__global__ void precomputeExpk(cufftComplex *expkM, cufftComplex *expkN, const int M, const int N)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid <= M / 2)
    {
        int hid = tid;
        cufftComplex W_h_4M = make_float2(__cosf((float)PI * hid / (2 * M)), -__sinf((float)PI * hid / (M * 2)));
        expkM[hid] = W_h_4M;
    }
    if (tid <= N / 2)
    {
        int wid = tid;
        cufftComplex W_w_4N = make_float2(__cosf((float)PI * wid / (2 * N)), -__sinf((float)PI * wid / (N * 2)));
        expkN[wid] = W_w_4N;
    }
}

template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void dct2d_postprocess_backup(const TComplex *V, T *y, const int M, const int N,
                                                                         const int halfN, const T two_over_MN, const T four_over_MN,
                                                                         const TComplex *__restrict__ expkM, const TComplex *__restrict__ expkN)
{
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < M && wid < N)
    {
        if (hid == 0)
        {
            TComplex tmp;
            if (wid <= halfN)
            {
                tmp = V[wid];
            }
            else
            {
                tmp = complexConj(V[N - wid]);
            }
            y[wid] = RealPartOfMul(expkN[wid], tmp) * four_over_MN;
        }
        else
        {
            TComplex tmp1, tmp2, tmp;
            if (wid <= halfN)
            {
                tmp1 = V[INDEX(hid, wid, halfN + 1)];
                tmp2 = V[INDEX(M - hid, wid, halfN + 1)];
                tmp.x = expkM[hid].x * (tmp1.x + tmp2.x) + expkM[hid].y * (tmp2.y - tmp1.y);
                tmp.y = expkM[hid].x * (tmp1.y + tmp2.y) + expkM[hid].y * (tmp1.x - tmp2.x);
            }
            else
            {
                tmp1 = V[INDEX(M - hid, N - wid, halfN + 1)];
                tmp2 = V[INDEX(hid, N - wid, halfN + 1)];
                tmp.x = expkM[hid].x * (tmp1.x + tmp2.x) + expkM[hid].y * (tmp1.y - tmp2.y);
                tmp.y = expkM[hid].y * (tmp1.x - tmp2.x) - expkM[hid].x * (tmp1.y + tmp2.y);
            }
            y[INDEX(hid, wid, N)] = RealPartOfMul(expkN[wid], tmp) * two_over_MN;
        }
    }
}

template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void dct2d_postprocess_backup2(const TComplex *V, T *y, const int M, const int N,
                                                                          const int halfM, const int halfN, const T two_over_MN, const T four_over_MN,
                                                                          const TComplex *__restrict__ expkM, const TComplex *__restrict__ expkN)
{
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < M / 2 && wid < N)
    {
        if (hid == 0)
        {
            // 0th row
            TComplex tmp;
            if (wid <= halfN)
            {
                tmp = V[wid];
            }
            else
            {
                tmp = complexConj(V[N - wid]);
            }
            y[wid] = RealPartOfMul(expkN[wid], tmp) * four_over_MN;

            // M/2 th row
            TComplex tmp1;
            if (wid <= halfN)
            {
                tmp1 = V[INDEX(M / 2, wid, halfN + 1)];
                tmp.x = expkM[M / 2].x * tmp1.x;
                tmp.y = expkM[M / 2].x * tmp1.y;
            }
            else
            {
                tmp1 = V[INDEX(M / 2, N - wid, halfN + 1)];
                tmp.x = expkM[M / 2].x * tmp1.x;
                tmp.y = -expkM[M / 2].x * tmp1.y;
            }
            y[INDEX(M / 2, wid, N)] = RealPartOfMul(expkN[wid], tmp) * four_over_MN;
        }
        else
        {
            TComplex tmp1, tmp2, tmp_up, tmp_down;
            if (wid <= halfN)
            {
                tmp1 = V[INDEX(hid, wid, halfN + 1)];
                tmp2 = V[INDEX(M - hid, wid, halfN + 1)];
                tmp_up.x = expkM[hid].x * (tmp1.x + tmp2.x) + expkM[hid].y * (tmp2.y - tmp1.y);
                tmp_up.y = expkM[hid].x * (tmp1.y + tmp2.y) + expkM[hid].y * (tmp1.x - tmp2.x);
                tmp_down.x = -expkM[hid].y * (tmp1.x + tmp2.x) + expkM[hid].x * (tmp2.y - tmp1.y);
                tmp_down.y = -expkM[hid].y * (tmp1.y + tmp2.y) + expkM[hid].x * (tmp1.x - tmp2.x);
            }
            else
            {
                tmp1 = V[INDEX(M - hid, N - wid, halfN + 1)];
                tmp2 = V[INDEX(hid, N - wid, halfN + 1)];
                tmp_up.x = expkM[hid].x * (tmp1.x + tmp2.x) + expkM[hid].y * (tmp1.y - tmp2.y);
                tmp_up.y = expkM[hid].y * (tmp1.x - tmp2.x) - expkM[hid].x * (tmp1.y + tmp2.y);
                tmp_down.x = -expkM[hid].y * (tmp1.x + tmp2.x) + expkM[hid].x * (tmp1.y - tmp2.y);
                tmp_down.y = expkM[hid].x * (tmp1.x - tmp2.x) + expkM[hid].y * (tmp1.y + tmp2.y);
            }
            y[INDEX(hid, wid, N)] = RealPartOfMul(expkN[wid], tmp_up) * two_over_MN;
            y[INDEX(M - hid, wid, N)] = RealPartOfMul(expkN[wid], tmp_down) * two_over_MN;
        }
    }
}

template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void dct2d_postprocess(const TComplex *V, T *y, const int M, const int N,
                                                                  const int halfM, const int halfN, const T two_over_MN, const T four_over_MN,
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
            y[0] = V[0].x * four_over_MN;
            y[halfN] = RealPartOfMul(expkN[halfN], V[halfN]) * four_over_MN;
            y[INDEX(halfM, 0, N)] = expkM[halfM].x * V[INDEX(halfM, 0, halfN + 1)].x * four_over_MN;
            y[INDEX(halfM, halfN, N)] = expkM[halfM].x * RealPartOfMul(expkN[halfN], V[INDEX(halfM, halfN, halfN + 1)]) * four_over_MN;
            break;
        }

        case 1:
        {
            TComplex tmp;

            tmp = V[wid];
            y[wid] = RealPartOfMul(expkN[wid], tmp) * four_over_MN;
            y[N - wid] = -ImaginaryPartOfMul(expkN[wid], tmp) * four_over_MN;

            tmp = V[INDEX(halfM, wid, halfN + 1)];
            y[INDEX(halfM, wid, N)] = expkM[halfM].x * RealPartOfMul(expkN[wid], tmp) * four_over_MN;
            y[INDEX(halfM, N - wid, N)] = -expkM[halfM].x * ImaginaryPartOfMul(expkN[wid], tmp) * four_over_MN;
            break;
        }

        case 2:
        {
            TComplex tmp1, tmp2, tmp_up, tmp_down;
            tmp1 = V[INDEX(hid, 0, halfN + 1)];
            tmp2 = V[INDEX(M - hid, 0, halfN + 1)];
            tmp_up.x = expkM[hid].x * (tmp1.x + tmp2.x) + expkM[hid].y * (tmp2.y - tmp1.y);
            tmp_down.x = -expkM[hid].y * (tmp1.x + tmp2.x) + expkM[hid].x * (tmp2.y - tmp1.y);
            y[INDEX(hid, 0, N)] = tmp_up.x * two_over_MN;
            y[INDEX(M - hid, 0, N)] = tmp_down.x * two_over_MN;

            tmp1 = complexAdd(V[INDEX(hid, halfN, halfN + 1)], V[INDEX(M - hid, halfN, halfN + 1)]);
            tmp2 = complexSubtract(V[INDEX(hid, halfN, halfN + 1)], V[INDEX(M - hid, halfN, halfN + 1)]);
            tmp_up.x = expkM[hid].x * tmp1.x - expkM[hid].y * tmp2.y;
            tmp_up.y = expkM[hid].x * tmp1.y + expkM[hid].y * tmp2.x;
            tmp_down.x = -expkM[hid].y * tmp1.x - expkM[hid].x * tmp2.y;
            tmp_down.y = -expkM[hid].y * tmp1.y + expkM[hid].x * tmp2.x;
            y[INDEX(hid, halfN, N)] = RealPartOfMul(expkN[halfN], tmp_up) * two_over_MN;
            y[INDEX(M - hid, halfN, N)] = RealPartOfMul(expkN[halfN], tmp_down) * two_over_MN;
            break;
        }

        case 3:
        {
            TComplex tmp1, tmp2, tmp_up, tmp_down;
            tmp1 = complexAdd(V[INDEX(hid, wid, halfN + 1)], V[INDEX(M - hid, wid, halfN + 1)]);
            tmp2 = complexSubtract(V[INDEX(hid, wid, halfN + 1)], V[INDEX(M - hid, wid, halfN + 1)]);
            tmp_up.x = expkM[hid].x * tmp1.x - expkM[hid].y * tmp2.y;
            tmp_up.y = expkM[hid].x * tmp1.y + expkM[hid].y * tmp2.x;
            tmp_down.x = -expkM[hid].y * tmp1.x - expkM[hid].x * tmp2.y;
            tmp_down.y = -expkM[hid].y * tmp1.y + expkM[hid].x * tmp2.x;
            y[INDEX(hid, wid, N)] = RealPartOfMul(expkN[wid], tmp_up) * two_over_MN;
            y[INDEX(M - hid, wid, N)] = RealPartOfMul(expkN[wid], tmp_down) * two_over_MN;
            y[INDEX(hid, N - wid, N)] = -ImaginaryPartOfMul(expkN[wid], tmp_up) * two_over_MN;
            y[INDEX(M - hid, N - wid, N)] = -ImaginaryPartOfMul(expkN[wid], tmp_down) * two_over_MN;
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
    cufftPlan2d(plan, M, N, CUFFT_R2C);
}

template <>
void makeCufftPlan<cufftDoubleComplex>(const int M, const int N, cufftHandle *plan)
{
    cufftPlan2d(plan, M, N, CUFFT_D2Z);
}

void fft2D(cufftDoubleReal *d_x, cufftDoubleComplex *d_y, cufftHandle &plan)
{
    cufftExecD2Z(plan, d_x, d_y);
    cudaDeviceSynchronize();
}

void fft2D(cufftReal *d_x, cufftComplex *d_y, cufftHandle &plan)
{
    cufftExecR2C(plan, d_x, d_y);
    cudaDeviceSynchronize();
}

CpuTimer Timer;

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
    cudaSafeCall(cudaMalloc((void **)&d_x, size));
    cudaSafeCall(cudaMalloc((void **)&d_y, size));
    cudaSafeCall(cudaMalloc((void **)&expkM, (M / 2 + 1) * sizeof(TComplex)));
    cudaSafeCall(cudaMalloc((void **)&expkN, (N / 2 + 1) * sizeof(TComplex)));
    cudaSafeCall(cudaMalloc((void **)&scratch, M * (N / 2 + 1) * sizeof(TComplex)));
    cudaSafeCall(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

    cufftHandle plan;
    makeCufftPlan<TComplex>(M, N, &plan);

    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 gridSize2((N / 2 + TPB - 1) / TPB, (M / 2 + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    precomputeExpk<<<(std::max(M, N) + 1023) / 1024, 1024>>>(expkM, expkN, M, N);
    cudaDeviceSynchronize();

    Timer.Start();
    dct2d_preprocess<T><<<gridSize, blockSize>>>(d_x, d_y, M, N, N / 2);
    // dct2d_preprocess_backup<T><<<gridSize, blockSize>>>(d_x, d_y, M, N, M / 2, N / 2);
    cudaDeviceSynchronize();
    Timer.Stop();
    fft2D(d_y, scratch, plan);

    dct2d_postprocess<T, TComplex><<<gridSize2, blockSize>>>(scratch, d_y, M, N, M / 2, N / 2, 2. / (M * N), 4. / (M * N), expkM, expkN);
    cudaDeviceSynchronize();


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
        printf("[D] dct 2D takes %g ms\n", Timer.ElapsedMillis());
        total_time += i > 0 ? Timer.ElapsedMillis() : 0;
    }

    printf("[D] dct 2D (%d * %d) takes average %g ms\n", M, N, total_time / (NUM_RUNS - 1));

    delete[] h_x;
    delete[] h_y;
    delete[] h_gt;

    return 0;
}
