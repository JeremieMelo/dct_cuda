#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "cuda_utils.cuh"

#define TPB (16)

template <typename T>
__global__ void computeMulExpk(
        const T* x, 
        const T* expk, 
        const int M, 
        const int N, 
        T* z
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int col_2x = (col<<1);
        int fft_onesided_size = (N>>1)+1;
        int fft_onesided_size_2x = fft_onesided_size<<1;

        if (col_2x <= N)
        {
            int j = row*fft_onesided_size_2x + col_2x;
            //printf("x[%d]*expk[%d] + x[%d]*expk[%d] = z[%d]\n", j, col_2x, j+1, col_2x+1, i);
            z[i] = x[j]*expk[col_2x] + x[j+1]*expk[col_2x+1];
        }
        else 
        {
            int j = row*fft_onesided_size_2x + (N<<1) - col_2x;
            //printf("x[%d]*expk[%d] + x[%d]*expk[%d] = z[%d]\n", j, col_2x, j+1, col_2x+1, i);
            z[i] = x[j]*expk[col_2x] - x[j+1]*expk[col_2x+1];
        }
    }
}

template <typename T>
void computeMulExpkCudaLauncher(
        const T* x, 
        const T* expk, 
        const int M, 
        const int N, 
        T* z
        )
{
    const int thread_count = 1024; 
    const int block_count = 32; 

    computeMulExpk<<<block_count, thread_count>>>(
            x, 
            expk, 
            M, 
            N, 
            z
            );
}

template <typename T>
__global__ void computeReorder(
        const T* x, 
        const int M, 
        const int N, 
        T* y
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int ii = i%N; 

        if (ii < (N>>1))
        {
            // i*2
            //printf("x[%d] = y[%d]\n", i+ii, i);
            y[i] = x[i+ii];
        }
        else 
        {
            // (N-i)*2-1
            //printf("x[%d] = y[%d]\n", i+N*2-ii*3-1, i);
            y[i] = x[i+N*2-ii*3-1];
        }
    }
}

template <typename T>
void computeReorderCudaLauncher(
        const T* x, 
        const int M, 
        const int N, 
        T* y
        )
{
    const int thread_count = 1024; 
    const int block_count = 32; 

    computeReorder<<<block_count, thread_count>>>(
            x, 
            M, 
            N, 
            y
            );
}

template <typename T>
__global__ void computeVk(
        const T* x, 
        const T* expk, 
        const int M, 
        const int N, 
        T* v
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*(N/2+1); i += blockDim.x * gridDim.x) 
    {
        int ncol = N/2+1; 
        int row = i/ncol; // row
        int col = i-row*ncol; // column
        int col_2x = (col<<1);

        // real 
        T real = x[row*N+col];
        T imag = (col == 0)? 0 : -x[row*N+N-col];

        v[2*i] = real*expk[col_2x] - imag*expk[col_2x+1];
        // imag, x[N-i]
        v[2*i+1] = real*expk[col_2x+1] + imag*expk[col_2x]; 
    }

}

template <typename T>
void computeVkCudaLauncher(
        const T* x, 
        const T* expk, 
        const int M, 
        const int N, 
        T* v
        )
{
    const int thread_count = 1024; 
    const int block_count = 32; 

    computeVk<<<block_count, thread_count>>>(
            x, 
            expk, 
            M, 
            N, 
            v
            );
}


template <typename T>
__global__ void computeReorderReverse(
        const T* y, 
        const int M, 
        const int N, 
        T* z
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int row = i/N; // row
        int col = i-row*N; // column

        //printf("z[%d] = y[%d]\n", i, (col&1)? (i-col*3/2+N-1) : (i-col/2));
        //z[i] = (col&1)? y[(i-col*3/2+N-1)] : y[(i-col/2)];
        // according to the paper, it should be N - (col+1)/2 for col is odd 
        // but it seems previous implementation accidentally matches this as well 
        z[i] = (col&1)? y[(i-col) + N - (col+1)/2] : y[(i-col/2)];
    }
}

template <typename T>
void computeReorderReverseCudaLauncher(
        const T* y, 
        const int M, 
        const int N, 
        T* z
        )
{
    const int thread_count = 1024; 
    const int block_count = 32; 

    computeReorderReverse<<<block_count, thread_count>>>(
            y, 
            M, 
            N, 
            z
            );
}

template <typename T>
__global__ void addX0AndScale(
        const T* x,
        const int M, 
        const int N, 
        T* y
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int i0 = int(i/N)*N; 
        y[i] = (y[i]+x[i0])*0.5;
    }
}

template <typename T>
void addX0AndScaleCudaLauncher(
        const T* x,
        const int M, 
        const int N, 
        T* y
        )
{
    addX0AndScale<<<32, 1024>>>(
            x, 
            M, 
            N, 
            y
            );
}

/// extends from addX0AndScale to merge scaling 
template <typename T>
__global__ void addX0AndScaleN(
        const T* x,
        const int M, 
        const int N, 
        T* y
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int i0 = int(i/N)*N; 
        // this is to match python implementation 
        // normal way should be multiply by 0.25*N
        y[i] = y[i]*0.25*N+x[i0]*0.5; 
    }
}

template <typename T>
void addX0AndScaleNCudaLauncher(
        const T* x,
        const int M, 
        const int N, 
        T* y
        )
{
    addX0AndScaleN<<<32, 1024>>>(
            x, 
            M, 
            N, 
            y
            );
}

template <typename T>
__global__ void computePad(
        const T* x, // M*N
        const int M, 
        const int N, 
        T* z // M*2N
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int j = row*(N<<1) + col; 
        z[j] = x[i]; 
    }
}

template <typename T>
void computePadCudaLauncher(
        const T* x, // M*N
        const int M, 
        const int N, 
        T* z // M*2N
        )
{
    computePad<<<32, 1024>>>(
            x, 
            M, 
            N, 
            z
            );
}

template <typename T>
__global__ void computeMulExpk_2N(
        const T* x, // M*(N+1)*2
        const T* expk, 
        const int M, 
        const int N, 
        T* z // M*N
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int col_2x = (col<<1);
        int j = row*((N+1)<<1) + col_2x; 
        z[i] = x[j]*expk[col_2x] + x[j+1]*expk[col_2x+1];
    }
}

template <typename T>
void computeMulExpk_2N_CudaLauncher(
        const T* x, // M*(N+1)*2
        const T* expk, 
        const int M, 
        const int N, 
        T* z // M*N
        )
{
    computeMulExpk_2N<<<32, 1024>>>(
            x, 
            expk, 
            M, 
            N, 
            z
            );
}

template <typename T>
__global__ void computeMulExpkAndPad_2N(
        const T* x, // M*N
        const T* expk, 
        const int M, 
        const int N, 
        T* z // M*2N*2
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int col_2x = (col<<1);
        int j = row*(N<<2) + col_2x; 
        z[j] = x[i]*expk[col_2x]; 
        z[j+1] = x[i]*expk[col_2x+1];
    }
}

template <typename T>
void computeMulExpkAndPad_2N_CudaLauncher(
        const T* x, // M*N
        const T* expk, 
        const int M, 
        const int N, 
        T* z // M*2N*2
        )
{
    computeMulExpkAndPad_2N<<<32, 1024>>>(
            x, 
            expk, 
            M, 
            N, 
            z
            );
}

/// remove last N entries in each column 
template <typename T>
__global__ void computeTruncation(
        const T* x, // M*2N
        const int M, 
        const int N, 
        T* z // M*N
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int j = row*(N<<1) + col; 
        z[i] = x[j]; 
    }
}

template <typename T>
void computeTruncationCudaLauncher(
        const T* x, // M*2N
        const int M, 
        const int N, 
        T* z // M*N
        )
{
    computeTruncation<<<32, 1024>>>(
            x, 
            M, 
            N, 
            z
            );
}

// dct2_fft2
template <typename T>
__global__ void dct2dPreprocess(const T *x, T *y, const int M, const int N, const int halfN)
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
        y[index] = x[INDEX(hid, wid, N)];
    }
}

template <typename T>
void dct2dPreprocessCudaLauncher(
                const T* x,
                T* y,
                const int M, 
                const int N
                )
{
    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    dct2dPreprocess<T><<<gridSize, blockSize>>>(x, y, M, N, N/2);
}

template <typename T>
__global__ __launch_bounds__(1024, 10) void dct2dPostprocess(const T *x_raw, T *y, const int M, const int N,
                                                               const int halfM, const int halfN, const T two_over_MN, const T four_over_MN,
                                                               const T *__restrict__ expkM_raw, const T *__restrict__ expkN_raw)
{
    const ComplexType<T> *V = (ComplexType<T>*)x_raw;
    const ComplexType<T> *__restrict__ expkM = (ComplexType<T>*) expkM_raw;
    const ComplexType<T> *__restrict__ expkN = (ComplexType<T>*) expkN_raw;
    
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
            ComplexType<T> tmp;

            tmp = V[wid];
            y[wid] = RealPartOfMul(expkN[wid], tmp) * four_over_MN;
            y[N - wid] = -1 * ImaginaryPartOfMul(expkN[wid], tmp) * four_over_MN;

            tmp = V[INDEX(halfM, wid, halfN + 1)];
            y[INDEX(halfM, wid, N)] = expkM[halfM].x * RealPartOfMul(expkN[wid], tmp) * four_over_MN;
            y[INDEX(halfM, N - wid, N)] = -1 * expkM[halfM].x * ImaginaryPartOfMul(expkN[wid], tmp) * four_over_MN;
            break;
        }

        case 2:
        {
            ComplexType<T> tmp1, tmp2, tmp_up, tmp_down;
            tmp1 = V[INDEX(hid, 0, halfN + 1)];
            tmp2 = V[INDEX(M - hid, 0, halfN + 1)];
            tmp_up.x = expkM[hid].x * (tmp1.x + tmp2.x) + expkM[hid].y * (tmp2.y - tmp1.y);
            tmp_down.x = -1 * expkM[hid].y * (tmp1.x + tmp2.x) + expkM[hid].x * (tmp2.y - tmp1.y);
            y[INDEX(hid, 0, N)] = tmp_up.x * two_over_MN;
            y[INDEX(M - hid, 0, N)] = tmp_down.x * two_over_MN;

            tmp1 = complexAdd(V[INDEX(hid, halfN, halfN + 1)], V[INDEX(M - hid, halfN, halfN + 1)]);
            tmp2 = complexSubtract(V[INDEX(hid, halfN, halfN + 1)], V[INDEX(M - hid, halfN, halfN + 1)]);
            tmp_up.x = expkM[hid].x * tmp1.x - expkM[hid].y * tmp2.y;
            tmp_up.y = expkM[hid].x * tmp1.y + expkM[hid].y * tmp2.x;
            tmp_down.x = -1 * expkM[hid].y * tmp1.x - expkM[hid].x * tmp2.y;
            tmp_down.y = -1 * expkM[hid].y * tmp1.y + expkM[hid].x * tmp2.x;
            y[INDEX(hid, halfN, N)] = RealPartOfMul(expkN[halfN], tmp_up) * two_over_MN;
            y[INDEX(M - hid, halfN, N)] = RealPartOfMul(expkN[halfN], tmp_down) * two_over_MN;
            break;
        }

        case 3:
        {
            ComplexType<T> tmp1, tmp2, tmp_up, tmp_down;
            tmp1 = complexAdd(V[INDEX(hid, wid, halfN + 1)], V[INDEX(M - hid, wid, halfN + 1)]);
            tmp2 = complexSubtract(V[INDEX(hid, wid, halfN + 1)], V[INDEX(M - hid, wid, halfN + 1)]);
            tmp_up.x = expkM[hid].x * tmp1.x - expkM[hid].y * tmp2.y;
            tmp_up.y = expkM[hid].x * tmp1.y + expkM[hid].y * tmp2.x;
            tmp_down.x = -1 * expkM[hid].y * tmp1.x - expkM[hid].x * tmp2.y;
            tmp_down.y = -1 * expkM[hid].y * tmp1.y + expkM[hid].x * tmp2.x;
            y[INDEX(hid, wid, N)] = RealPartOfMul(expkN[wid], tmp_up) * two_over_MN;
            y[INDEX(M - hid, wid, N)] = RealPartOfMul(expkN[wid], tmp_down) * two_over_MN;
            y[INDEX(hid, N - wid, N)] = -1 * ImaginaryPartOfMul(expkN[wid], tmp_up) * two_over_MN;
            y[INDEX(M - hid, N - wid, N)] = -1 * ImaginaryPartOfMul(expkN[wid], tmp_down) * two_over_MN;
            break;
        }

        default:
            assert(0);
            break;
        }
    }
}

template <typename T>
void dct2dPostprocessCudaLauncher(
                const T *x, 
                T *y, 
                const int M, 
                const int N,
                const T *__restrict__ expkM, 
                const T *__restrict__ expkN
                )
{
    dim3 gridSize((N/2 + TPB - 1) / TPB, (M/2 + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    dct2dPostprocess<T><<<gridSize, blockSize>>>(x, y, M, N, M/2, N/2, (T)(2./(M*N)), (T)(4./(M*N)), expkM, expkN);
}

//idct_idxst

template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void idct_idxstPreprocess(const T *input, TComplex *output, const int M, const int N,
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
void idct_idxstPreprocessCudaLauncher(
                const T* x,
                T* y,
                const int M, 
                const int N,
                const T *__restrict__ expkM, 
                const T *__restrict__ expkN
                )
{
    dim3 gridSize((N/2 + TPB - 1) / TPB, (M/2 + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    idct_idxstPreprocess<T, ComplexType<T>><<<gridSize, blockSize>>>(x, (ComplexType<T>*)y, M, N, M/2, N/2, (ComplexType<T>*)expkM, (ComplexType<T>*)expkN);
}

template <typename T>
__global__ void idct_idxstPostprocess(const T *x, T *y, const int M, const int N, const int halfN)
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
            y[index] = -0.5 * x[INDEX(hid, wid, N)];
            break;
        case 1:
            index = INDEX(((M - hid) << 1) - 1, wid << 1, N);
            y[index] = 0.5 * x[INDEX(hid, wid, N)];
            break;
        case 2:
            index = INDEX(hid << 1, ((N - wid) << 1) - 1, N);
            y[index] = -0.5 * x[INDEX(hid, wid, N)];
            break;
        case 3:
            index = INDEX(hid << 1, wid << 1, N);
            y[index] = 0.5 * x[INDEX(hid, wid, N)];
            break;
        default:
            assert(0);
            break;
        }
    }
}

template <typename T>
void idct_idxstPostprocessCudaLauncher(
                const T *x, 
                T *y, 
                const int M, 
                const int N
                )
{
    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    idct_idxstPostprocess<T><<<gridSize, blockSize>>>(x, y, M, N, N/2);
}

//idxst_idct

template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void idxst_idctPreprocess(const T *input, TComplex *output, const int M, const int N,
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

            output[halfN].x = 0;
            output[halfN].y = 0;

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
            output[wid].x = 0;
            output[wid].y = 0;

            TComplex tmp_up;
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

            tmp1 = input[INDEX(M - hid, 0, N)];
            tmp3 = input[INDEX(hid, 0, N)];
            tmp_up.x = tmp1;
            tmp_up.y = tmp3;
            tmp_down.x = tmp3;
            tmp_down.y = tmp1;

            output[INDEX(hid, 0, halfN + 1)] = complexConj(complexMul(expkM[hid], tmp_up));
            output[INDEX(M - hid, 0, halfN + 1)] = complexConj(complexMul(expkM[M - hid], tmp_down));

            tmp1 = input[INDEX(M - hid, halfN, N)];
            tmp3 = input[INDEX(hid, halfN, N)];
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
            T tmp1 = input[INDEX(M - hid, wid, N)];
            T tmp2 = input[INDEX(M - hid, N - wid, N)];
            T tmp3 = input[INDEX(hid, wid, N)];
            T tmp4 = input[INDEX(hid, N - wid, N)];
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
void idxst_idctPreprocessCudaLauncher(
                const T* x,
                T* y,
                const int M, 
                const int N,
                const T *__restrict__ expkM, 
                const T *__restrict__ expkN
                )
{
    dim3 gridSize((N/2 + TPB - 1) / TPB, (M/2 + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    idxst_idctPreprocess<T, ComplexType<T>><<<gridSize, blockSize>>>(x, (ComplexType<T>*)y, M, N, M/2, N/2, (ComplexType<T>*)expkM, (ComplexType<T>*)expkN);
}

template <typename T>
__global__ void idxst_idctPostprocess(const T *x, T *y, const int M, const int N, const int halfN)
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
            y[index] = -0.5 * x[INDEX(hid, wid, N)];
            break;
        case 1:
            index = INDEX(((M - hid) << 1) - 1, wid << 1, N);
            y[index] = -0.5 * x[INDEX(hid, wid, N)];
            break;
        case 2:
            index = INDEX(hid << 1, ((N - wid) << 1) - 1, N);
            y[index] = 0.5 * x[INDEX(hid, wid, N)];
            break;
        case 3:
            index = INDEX(hid << 1, wid << 1, N);
            y[index] = 0.5 * x[INDEX(hid, wid, N)];
            break;
        default:
            assert(0);
            break;
        }
    }
}

template <typename T>
void idxst_idctPostprocessCudaLauncher(
                const T *x, 
                T *y, 
                const int M, 
                const int N
                )
{
    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    idxst_idctPostprocess<T><<<gridSize, blockSize>>>(x, y, M, N, N/2);
}

// idct2_fft2

template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void idct2_fft2Preprocess(const T *input, TComplex *output, const int M, const int N,
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
void idct2_fft2PreprocessCudaLauncher(
                const T* x,
                T* y,
                const int M, 
                const int N,
                const T *__restrict__ expkM, 
                const T *__restrict__ expkN
                )
{
    dim3 gridSize((N/2 + TPB - 1) / TPB, (M/2 + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    idct2_fft2Preprocess<T, ComplexType<T>><<<gridSize, blockSize>>>(x, (ComplexType<T>*)y, M, N, M/2, N/2, (ComplexType<T>*)expkM, (ComplexType<T>*)expkN);
}

template <typename T>
__global__ void idct2_fft2Postprocess(const T *x, T *y, const int M, const int N, const int halfN)
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

template <typename T>
void idct2_fft2PostprocessCudaLauncher(
                const T *x, 
                T *y, 
                const int M, 
                const int N
                )
{
    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    idct2_fft2Postprocess<T><<<gridSize, blockSize>>>(x, y, M, N, N/2);
}

// manually instantiate the template function 
#define REGISTER_MULPEXPK_KERNEL_LAUNCHER(type) \
    void instantiateComputeMulExpkLauncher(\
        const type* x, \
        const type* expk, \
        const int M, \
        const int N, \
        type* z \
        ) \
    { \
        return computeMulExpkCudaLauncher<type>( \
                x, \
                expk, \
                M, \
                N, \
                z \
                ); \
    }

REGISTER_MULPEXPK_KERNEL_LAUNCHER(float);
REGISTER_MULPEXPK_KERNEL_LAUNCHER(double);

#define REGISTER_REORDER_KERNEL_LAUNCHER(type) \
    void instantiateComputeReorderLauncher(\
        const type* x, \
        const int M, \
        const int N, \
        type* y \
        ) \
    { \
        return computeReorderCudaLauncher<type>( \
                x, \
                M, \
                N, \
                y \
                ); \
    }

REGISTER_REORDER_KERNEL_LAUNCHER(float);
REGISTER_REORDER_KERNEL_LAUNCHER(double);

#define REGISTER_VK_KERNEL_LAUNCHER(type) \
    void instantiateComputeVkLauncher(\
        const type* x, \
        const type* expk, \
        const int M, \
        const int N, \
        type* v \
        ) \
    { \
        return computeVkCudaLauncher<type>( \
                x, \
                expk, \
                M, \
                N, \
                v \
                ); \
    }

REGISTER_VK_KERNEL_LAUNCHER(float);
REGISTER_VK_KERNEL_LAUNCHER(double);

#define REGISTER_REORDERREVERSE_KERNEL_LAUNCHER(type) \
    void instantiateComputeReorderReverseLauncher(\
        const type* y, \
        const int M, \
        const int N, \
        type* z \
        ) \
    { \
        return computeReorderReverseCudaLauncher<type>( \
                y, \
                M, \
                N, \
                z \
                ); \
    }

REGISTER_REORDERREVERSE_KERNEL_LAUNCHER(float);
REGISTER_REORDERREVERSE_KERNEL_LAUNCHER(double);

#define REGISTER_ADDX0ANDSCALE_KERNEL_LAUNCHER(type) \
    void instantiateAddX0AndScaleLauncher(\
        const type* x, \
        const int M, \
        const int N, \
        type* y \
        ) \
    { \
        return addX0AndScaleCudaLauncher<type>( \
                x, \
                M, \
                N, \
                y \
                ); \
    }

REGISTER_ADDX0ANDSCALE_KERNEL_LAUNCHER(float);
REGISTER_ADDX0ANDSCALE_KERNEL_LAUNCHER(double);

#define REGISTER_ADDX0ANDSCALEN_KERNEL_LAUNCHER(type) \
    void instantiateAddX0AndScaleNLauncher(\
        const type* x, \
        const int M, \
        const int N, \
        type* y \
        ) \
    { \
        return addX0AndScaleNCudaLauncher<type>( \
                x, \
                M, \
                N, \
                y \
                ); \
    }

REGISTER_ADDX0ANDSCALEN_KERNEL_LAUNCHER(float);
REGISTER_ADDX0ANDSCALEN_KERNEL_LAUNCHER(double);

#define REGISTER_COMPUTEPAD_KERNEL_LAUNCHER(type) \
    void instantiateComputePadCudaLauncher(\
            const type* x, \
            const int M, \
            const int N, \
            type* z \
        ) \
    { \
        return computePadCudaLauncher<type>( \
                x, \
                M, \
                N, \
                z \
                ); \
    }

REGISTER_COMPUTEPAD_KERNEL_LAUNCHER(float);
REGISTER_COMPUTEPAD_KERNEL_LAUNCHER(double);

#define REGISTER_COMPUTEMULEXPK_2N_KERNEL_LAUNCHER(type) \
    void instantiateComputeMulExpk_2N_CudaLauncher(\
            const type* x, \
            const type* expk, \
            const int M, \
            const int N, \
            type* z \
        ) \
    { \
        return computeMulExpk_2N_CudaLauncher<type>( \
                x, \
                expk, \
                M, \
                N, \
                z \
                ); \
    }

REGISTER_COMPUTEMULEXPK_2N_KERNEL_LAUNCHER(float);
REGISTER_COMPUTEMULEXPK_2N_KERNEL_LAUNCHER(double);

#define REGISTER_COMPUTEMULEXPKANDPAD_2N_KERNEL_LAUNCHER(type) \
    void instantiateComputeMulExpkAndPad_2N_CudaLauncher(\
            const type* x, \
            const type* expk, \
            const int M, \
            const int N, \
            type* z \
        ) \
    { \
        return computeMulExpkAndPad_2N_CudaLauncher<type>( \
                x, \
                expk, \
                M, \
                N, \
                z \
                ); \
    }

REGISTER_COMPUTEMULEXPKANDPAD_2N_KERNEL_LAUNCHER(float);
REGISTER_COMPUTEMULEXPKANDPAD_2N_KERNEL_LAUNCHER(double);

#define REGISTER_COMPUTETRUNCATION_KERNEL_LAUNCHER(type) \
    void instantiateComputeTruncationCudaLauncher(\
            const type* x, \
            const int M, \
            const int N, \
            type* z \
        ) \
    { \
        return computeTruncationCudaLauncher<type>( \
                x, \
                M, \
                N, \
                z \
                ); \
    }

REGISTER_COMPUTETRUNCATION_KERNEL_LAUNCHER(float);
REGISTER_COMPUTETRUNCATION_KERNEL_LAUNCHER(double);

// dct2_fft2
#define REGISTER_DCT2DPREPROCESS_KERNEL_LAUNCHER(type) \
    void instantiatedct2dPreprocessCudaLauncher(\
        const type* x,\
        type* y, \
        const int M, \
        const int N \
        ) \
    { \
        return dct2dPreprocessCudaLauncher<type>( \
                x, \
                y, \
                M, \
                N \
                ); \
    }

REGISTER_DCT2DPREPROCESS_KERNEL_LAUNCHER(float);
REGISTER_DCT2DPREPROCESS_KERNEL_LAUNCHER(double);

#define REGISTER_DCT2DPOSTPROCESS_KERNEL_LAUNCHER(type) \
    void instantiatedct2dPostprocessCudaLauncher(\
                const type *x, \
                type *y, \
                const int M, \
                const int N, \
                const type * __restrict__ expkM, \
                const type * __restrict__ expkN \
        ) \
    { \
        return dct2dPostprocessCudaLauncher<type>( \
                x, \
                y, \
                M, \
                N, \
                expkM, \
                expkN \
                ); \
    }

REGISTER_DCT2DPOSTPROCESS_KERNEL_LAUNCHER(float);
REGISTER_DCT2DPOSTPROCESS_KERNEL_LAUNCHER(double);

//idct_idxst
#define REGISTER_IDCT_IDXSTPREPROCESS_KERNEL_LAUNCHER(type) \
    void instantiateidct_idxstPreprocessCudaLauncher(\
        const type *x, \
        type *y, \
        const int M, \
        const int N, \
        const type * __restrict__ expkM, \
        const type * __restrict__ expkN \
    ) \
    { \
    return idct_idxstPreprocessCudaLauncher<type>( \
        x, \
        y, \
        M, \
        N, \
        expkM, \
        expkN \
        ); \
    }

REGISTER_IDCT_IDXSTPREPROCESS_KERNEL_LAUNCHER(float);
REGISTER_IDCT_IDXSTPREPROCESS_KERNEL_LAUNCHER(double);

#define REGISTER_IDCT_IDXSTPOSTPROCESS_KERNEL_LAUNCHER(type) \
    void instantiateidct_idxstPostprocessCudaLauncher(\
        const type* x,\
        type* y, \
        const int M, \
        const int N \
        ) \
    { \
        return idct_idxstPostprocessCudaLauncher<type>( \
                x, \
                y, \
                M, \
                N \
                ); \
    }
    

REGISTER_IDCT_IDXSTPOSTPROCESS_KERNEL_LAUNCHER(float);
REGISTER_IDCT_IDXSTPOSTPROCESS_KERNEL_LAUNCHER(double);

//idxst_idct
#define REGISTER_IDXST_IDCTPREPROCESS_KERNEL_LAUNCHER(type) \
    void instantiateidxst_idctPreprocessCudaLauncher(\
        const type *x, \
        type *y, \
        const int M, \
        const int N, \
        const type * __restrict__ expkM, \
        const type * __restrict__ expkN \
    ) \
    { \
    return idxst_idctPreprocessCudaLauncher<type>( \
        x, \
        y, \
        M, \
        N, \
        expkM, \
        expkN \
        ); \
    }

REGISTER_IDXST_IDCTPREPROCESS_KERNEL_LAUNCHER(float);
REGISTER_IDXST_IDCTPREPROCESS_KERNEL_LAUNCHER(double);

#define REGISTER_IDXST_IDCTPOSTPROCESS_KERNEL_LAUNCHER(type) \
    void instantiateidxst_idctPostprocessCudaLauncher(\
        const type* x,\
        type* y, \
        const int M, \
        const int N \
        ) \
    { \
        return idxst_idctPostprocessCudaLauncher<type>( \
                x, \
                y, \
                M, \
                N \
                ); \
    }
    

REGISTER_IDXST_IDCTPOSTPROCESS_KERNEL_LAUNCHER(float);
REGISTER_IDXST_IDCTPOSTPROCESS_KERNEL_LAUNCHER(double);

//idct2_fft2
#define REGISTER_IDCT2_FFT2PREPROCESS_KERNEL_LAUNCHER(type) \
    void instantiateidct2_fft2PreprocessCudaLauncher(\
        const type *x, \
        type *y, \
        const int M, \
        const int N, \
        const type * __restrict__ expkM, \
        const type * __restrict__ expkN \
    ) \
    { \
    return idct2_fft2PreprocessCudaLauncher<type>( \
        x, \
        y, \
        M, \
        N, \
        expkM, \
        expkN \
        ); \
    }

REGISTER_IDCT2_FFT2PREPROCESS_KERNEL_LAUNCHER(float);
REGISTER_IDCT2_FFT2PREPROCESS_KERNEL_LAUNCHER(double);

#define REGISTER_IDCT2_FFT2POSTPROCESS_KERNEL_LAUNCHER(type) \
    void instantiateidct2_fft2PostprocessCudaLauncher(\
        const type* x,\
        type* y, \
        const int M, \
        const int N \
        ) \
    { \
        return idct2_fft2PostprocessCudaLauncher<type>( \
                x, \
                y, \
                M, \
                N \
                ); \
    }
    

REGISTER_IDCT2_FFT2POSTPROCESS_KERNEL_LAUNCHER(float);
REGISTER_IDCT2_FFT2POSTPROCESS_KERNEL_LAUNCHER(double);
