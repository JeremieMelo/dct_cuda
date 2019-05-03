// idct(idxst(x)) is similar to the idct2d(x),
// except tiny modification on preprocessing and postprocessing
#include "global.cuh"

#define TPB (16)

// Adpated from idct2d_postprocess() with changes on sign and scale
// if (wid % 2 == 1)
//     new_output[hid][wid] = -output[hid][wid];
// else
//     new_output[hid][wid] = output[hid][wid];
template <typename T>
__global__ void idct_idxst_postprocess_backup(const T *x, T *y, const int M, const int N, const int halfN)
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
            y[INDEX(hid, wid, N)] = -x[index];
            break;
        case 1:
            index = INDEX(2 * M - (hid + 1), wid / 2, halfN);
            y[INDEX(hid, wid, N)] = x[index];
            break;
        case 2:
            index = INDEX(hid, N - (wid + 1) / 2, halfN);
            y[INDEX(hid, wid, N)] = -x[index];
            break;
        case 3:
            index = INDEX(hid, wid / 2, halfN);
            y[INDEX(hid, wid, N)] = x[index];
            break;
        default:
            assert(0);
            break;
        }
    }
}

// Adpated from idct2d_postprocess() with changes on sign and scale
// if (wid % 2 == 1)
//     new_output[hid][wid] = -output[hid][wid];
// else
//     new_output[hid][wid] = output[hid][wid];
template <typename T>
__global__ void idct_idxst_postprocess(const T *x, T *y, const int M, const int N, const int halfN)
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
            y[index] = -x[INDEX(hid, wid, N)];
            break;
        case 1:
            index = INDEX(((M - hid) << 1) - 1, wid << 1, N);
            y[index] = x[INDEX(hid, wid, N)];
            break;
        case 2:
            index = INDEX(hid << 1, ((N - wid) << 1) - 1, N);
            y[index] = -x[INDEX(hid, wid, N)];
            break;
        case 3:
            index = INDEX(hid << 1, wid << 1, N);
            y[index] = x[INDEX(hid, wid, N)];
            break;
        default:
            assert(0);
            break;
        }
    }
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

// Adpated from idct2d_preprocess(). The only change is the reordered input
// if (wid != 0)
//     new_input[hid][wid] = input[hid][N - wid];
// else
//     new_input[hid][0] = 0
template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void idct_idxst_preprocess(const T *input, TComplex *output, const int M, const int N,
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

            output[0].x = 0;
            output[0].y = 0;

            tmp1 = input[halfN];
            tmp_up.x = tmp1;
            tmp_up.y = tmp1;
            output[halfN] = complexConj(complexMul(expkN[halfN], tmp_up));

            output[INDEX(halfM, 0, halfN + 1)].x = 0;
            output[INDEX(halfM, 0, halfN + 1)].y = 0;

            tmp1 = input[INDEX(halfM, halfN, N)];
            tmp_up.x = 0;
            tmp_up.y = 2 * tmp1;
            output[INDEX(halfM, halfN, halfN + 1)] = complexConj(complexMul(complexMul(expkM[halfM], expkN[halfN]), tmp_up));
            break;
        }

        case 1:
        {
            TComplex tmp_up;
            tmp_up.x = input[N - wid];
            tmp_up.y = input[wid];
            output[wid] = complexConj(complexMul(expkN[wid], tmp_up));

            T tmp1 = input[INDEX(halfM, N - wid, N)];
            T tmp2 = input[INDEX(halfM, wid, N)];
            tmp_up.x = tmp1 - tmp2;
            tmp_up.y = tmp1 + tmp2;
            output[INDEX(halfM, wid, halfN + 1)] = complexConj(complexMul(complexMul(expkM[halfM], expkN[wid]), tmp_up));
            break;
        }

        case 2:
        {
            T tmp1, tmp3;
            TComplex tmp_up, tmp_down;

            output[INDEX(hid, 0, halfN + 1)].x = 0;
            output[INDEX(hid, 0, halfN + 1)].y = 0;
            output[INDEX(M - hid, 0, halfN + 1)].x = 0;
            output[INDEX(M - hid, 0, halfN + 1)].y = 0;

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
            T tmp1 = input[INDEX(hid, N - wid, N)];
            T tmp2 = input[INDEX(hid, wid, N)];
            T tmp3 = input[INDEX(M - hid, N - wid, N)];
            T tmp4 = input[INDEX(M - hid, wid, N)];
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

void ifft2D(cufftDoubleComplex *d_x, cufftDoubleReal *d_y, cufftHandle &plan)
{
    cufftExecZ2D(plan, d_x, d_y);
    cudaDeviceSynchronize();
}

void ifft2D(cufftComplex *d_x, cufftReal *d_y, cufftHandle &plan)
{
    cufftExecC2R(plan, d_x, d_y);
    cudaDeviceSynchronize();
}

CpuTimer Timer;
template <typename T, typename TReal = cufftDoubleReal, typename TComplex = cufftDoubleComplex>
void idct_idxst(const T *h_x, T *h_y, const int M, const int N)
{
    T *d_x;
    T *d_y;
    T *ifft_result;
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
    cudaSafeCall(cudaMalloc((void **)&ifft_result, size));
    cudaSafeCall(cudaMalloc((void **)&expkM, M * sizeof(TComplex)));
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
    idct_idxst_preprocess<T, TComplex><<<gridSize2, blockSize>>>(d_x, scratch, M, N, M / 2, N / 2, expkM, expkN);
    cudaDeviceSynchronize();

    ifft2D(scratch, ifft_result, plan);

    idct_idxst_postprocess<T><<<gridSize, blockSize>>>(ifft_result, d_y, M, N, N / 2);
    cudaDeviceSynchronize();
    Timer.Stop();

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(ifft_result);
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

    std::ifstream input_file2("idct_idxst.dat", std::ios_base::in);

    i = 0;
    input_file2 >> M;
    input_file2 >> N;
    result = new T[M * N];
    while (input_file2 >> val)
    {
        result[i] = val * 2; // scale factor
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
        idct_idxst<dtype, dtypeReal, dtypeComplex>(h_x, h_y, M, N);
        int flag = validate2D<dtype>(h_y, h_gt, M, N);
        if (!flag)
        {
            printf("[I] Error! Results are incorrect.\n", flag);
            for (int i = 0; i < 64; ++i)
            {
                printf("index: %d, result: %f, GT: %f, scale: %f\n", i, h_y[i], h_gt[i], h_y[i] / h_gt[i]);
            }
        }
        printf("[D] idct_idxst takes %g ms\n", Timer.ElapsedMillis());
        total_time += i > 0 ? Timer.ElapsedMillis() : 0;
    }

    printf("[D] idct_idxst (%d * %d) takes average %g ms\n", M, N, total_time / (NUM_RUNS - 1));

    delete[] h_x;
    delete[] h_y;
    delete[] h_gt;

    return 0;
}
