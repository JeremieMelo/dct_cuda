/*
 * @Author: Jake Gu
 * @Date: 2019-04-29 13:07:55
 * @LastEditTime: 2019-04-29 13:07:55
 */
#ifndef __EPLACE_FFT_H__
#define __EPLACE_FFT_H__

#include <vector>
#include <cmath>
#include <stdexcept>
#include "global/global.h"

PROJECT_NAMESPACE_BEGIN

namespace eplace
{
namespace fft
{

constexpr double PI = 3.14159265358979323846;

/// Return true if a number is power of 2
template <typename T = unsigned>
inline bool isPowerOf2(T val)
{
    return val && (val & (val - 1)) == 0;
}

/// Transpose a column-major matrix with M rows and N columns using block transpose method
template <typename TValue, typename TIndex = unsigned>
inline void transpose(const TValue *in, TValue *out, TIndex M, TIndex N, TIndex blockSize = 16)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (TIndex i = 0; i < N; i += blockSize)
    {
        for (TIndex j = 0; j < M; j += blockSize)
        {
            // Transpose the block beginning at [i, j]
            TIndex xend = std::min(N, i + blockSize);
            TIndex yend = std::min(M, j + blockSize);
            for (TIndex x = i; x < xend; ++x)
            {
                for (TIndex y = j; y < yend; ++y)
                {
                    out[x + y * N] = in[y + x * M];
                }
            }
        }
    }
}

/// Negate values in odd position of a vector
template <typename TValue, typename TIndex = unsigned>
inline void negateOddEntries(TValue *vec, TIndex N)
{
    for (TIndex i = 1; i < N; i += 2)
    {
        vec[i] = -vec[i];
    }
}

/// Precompute cosine values needed for N-point dct
/// @param  cos  size N - 1 buffer, contains the result after function call
/// @param  N    the length of target dct, must be power of 2
template <typename TValue, typename TIndex = unsigned>
void precompute_dct_cos(TValue *cos, TIndex N)
{
    // The input length must be power of 2
    if (! isPowerOf2<TIndex>(N))
    {
        throw std::domain_error("Input length is not power of 2.");
    }

    TIndex offset = 0;
    TIndex halfLen = N / 2;
    while (halfLen)
    {
        TValue phaseStep = 0.5 * PI / halfLen;
        TValue phase = 0.5 * phaseStep;
        for (TIndex i = 0; i < halfLen; ++i)
        {
            cos[offset + i] = 0.5 / std::cos(phase);
            phase += phaseStep;
        }
        offset += halfLen;
        halfLen /= 2;
    }
}

/// Precompute cosine values needed for N-point idct
/// @param  cos  size N - 1 buffer, contains the result after function call
/// @param  N    the length of target idct, must be power of 2
template <typename TValue, typename TIndex = unsigned>
void precompute_idct_cos(TValue *cos, TIndex N)
{
    // The input length must be power of 2
    if (! isPowerOf2<TIndex>(N))
    {
        throw std::domain_error("Input length is not power of 2.");
    }

    TIndex offset = 0;
    TIndex halfLen = 1;
    while(halfLen < N)
    {
        TValue phaseStep = 0.5 * PI / halfLen;
        TValue phase = 0.5 * phaseStep;
        for (TIndex i = 0; i < halfLen; ++i)
        {
            cos[offset + i] = 0.5 / std::cos(phase);
            phase += phaseStep;
        }
        offset += halfLen;
        halfLen *= 2;
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
/// @param  vec   length N sequence to be transformed
/// @param  temp  length 2 * N helping buffer
/// @param  cos   length N - 1, stores cosine values precomputed by function 'precompute_dct_cos'
/// @param  N     length of vec, must be power of 2
template <typename TValue, typename TIndex = unsigned>
inline void dct(TValue *vec, TValue *temp, const TValue *cos, TIndex N)
{
    // The input length must be power of 2
    if (! isPowerOf2<TIndex>(N))
    {
        throw std::domain_error("Input length is not power of 2.");
    }

    // 'temp' is used to store data of two adjacent iterations
    // Copy 'vec' to the first N element in 'temp'
    std::copy(vec, vec + N, temp);

    // Pointers point to the beginning indices of two adjacent iterations
    TValue *curr = temp;
    TValue *next = temp + N;

    // Current bufferfly length and half length
    TIndex len = N;
    TIndex halfLen = len / 2;

    // Iteratively bi-partition sequences into sub-sequences
    TIndex cosOffset = 0;
    while (halfLen)
    {
        TIndex offset = 0;
        TIndex steps = N / len;
        for (TIndex k = 0; k < steps; ++k)
        {
            for (TIndex i = 0; i < halfLen; ++i)
            {
                next[offset + i] = curr[offset + i] + curr[offset + len - i - 1];
                next[offset + halfLen + i] = (curr[offset + i] - curr[offset + len -i - 1]) * cos[cosOffset + i];
            }
            offset += len;
        }
        std::swap(curr, next);
        cosOffset += halfLen;
        len = halfLen;
        halfLen /= 2;
    }

    // Bottom-up form the final DCT solution
    // Note that the case len = 2 will do nothing, so we start from len = 4
    len = 4;
    halfLen = 2;
    while(halfLen < N)
    {
        TIndex offset = 0;
        TIndex steps = N / len;
        for(TIndex k = 0; k < steps; ++k)
        {
            for(TIndex i = 0; i < halfLen - 1; ++i)
            {
                next[offset + i * 2] = curr[offset + i];
                next[offset + i * 2 + 1] = curr[offset + halfLen + i] + curr[offset + halfLen + i + 1];
            }
            next[offset + len - 2] = curr[offset + halfLen - 1];
            next[offset + len - 1] = curr[offset + len - 1];
            offset += len;
        }
        std::swap(curr, next);
        halfLen = len;
        len *= 2;
    }

    // Populate the final results into 'vec'
    std::copy(curr, curr + N, vec);
}

/// Compute y[k] = 0.5 * x[0] + sum_n=1..N-1 (x[n] * cos(n * (k + 0.5) * PI / N)), for k = 0..N-1
/// @param  vec   length N sequence to be transformed
/// @param  temp  length 2 * N helping buffer
/// @param  cos   length N - 1, stores cosine values precomputed by function 'precompute_idct_cos'
/// @param  N     length of vec, must be power of 2
template <typename TValue, typename TIndex = unsigned>
inline void idct(TValue *vec, TValue *temp, const TValue *cos, TIndex N)
{
    // The input length must be power of 2
    if (! isPowerOf2<TIndex>(N))
    {
        throw std::domain_error("Input length is not power of 2.");
    }

    // This array is used to store data of two adjacent iterations
    // Copy 'vec' to the first N element in 'temp'
    std::copy(vec, vec + N, temp);
    temp[0] /= 2;

    // Pointers point to the beginning indices of two adjacent iterations
    TValue *curr = temp;
    TValue *next = temp + N;

    // Current bufferfly length and half length
    TIndex len = N;
    TIndex halfLen = len / 2;

    // Iteratively bi-partition sequences into sub-sequences
    while (halfLen)
    {
        TIndex offset = 0;
        TIndex steps = N / len;
        for (TIndex k = 0; k < steps; ++k)
        {
            next[offset] = curr[offset];
            next[offset + halfLen] = curr[offset + 1];
            for (TIndex i = 1; i < halfLen; ++i)
            {
                next[offset + i] = curr[offset + i * 2];
                next[offset + halfLen + i] = curr[offset + i * 2 - 1] + curr[offset + i * 2 + 1];
            }
            offset += len;
        }
        std::swap(curr, next);
        len = halfLen;
        halfLen /= 2;
    }

    // Bottom-up form the final IDCT solution
    len = 2;
    halfLen = 1;
    TIndex cosOffset = 0;
    while(halfLen < N)
    {
        TIndex offset = 0;
        TIndex steps = N / len;
        for(TIndex k = 0; k < steps; ++k)
        {
            for(TIndex i = 0; i < halfLen; ++i)
            {
                TValue g = curr[offset + i];
                TValue h = curr[offset + halfLen + i] * cos[cosOffset + i];
                next[offset + i] = g + h;
                next[offset + len - 1 - i] = g - h;
            }
            offset += len;
        }
        std::swap(curr, next);
        cosOffset += halfLen;
        halfLen = len;
        len *= 2;
    }

    // Populate the final results into 'vec'
    std::copy(curr, curr + N, vec);
}

/// Compute y[k] = sum_n=0..N-1 (x[n] * sin((n + 0.5) * (k + 1) * PI / N)), for k = 0..N-1
/// @param  vec   length N sequence to be transformed
/// @param  temp  length 2 * N helping buffer
/// @param  cos   length N - 1, stores cosine values precomputed by function 'precompute_dct_cos'
/// @param  N     length of vec, must be power of 2
template <typename TValue, typename TIndex = unsigned>
inline void dst(TValue *vec, TValue *temp, const TValue *cos, TIndex N)
{
    negateOddEntries(vec, N);
    dct<TValue, TIndex>(vec, temp, cos, N);
    std::reverse(vec, vec + N);
}

/// Compute y[k] = 0.5 * (-1)^k * x[N - 1] + sum_n=1..N-2 (x[n] * sin((n + 1) * (k + 0.5) * PI / N)), for k = 0..N-1
/// @param  vec   length N sequence to be transformed
/// @param  temp  length 2 * N helping buffer
/// @param  cos   length N - 1, stores cosine values precomputed by function 'precompute_idct_cos'
/// @param  N     length of vec, must be power of 2
template <typename TValue, typename TIndex = unsigned>
inline void idst(TValue *vec, TValue *temp, const TValue *cos, TIndex N)
{
    std::reverse(vec, vec + N);
    idct<TValue, TIndex>(vec, temp, cos, N);
    negateOddEntries(vec, N);
}

/// Compute y[k] = sum_n=0..N-1 (x[n] * sin(n * (k + 0.5) * PI / N)), for k = 0..N-1
/// This is a bit different from the standard IDST
/// @param  vec   length N sequence to be transformed
/// @param  temp  length 2 * N helping buffer
/// @param  cos   length N - 1, stores cosine values precomputed by function 'precompute_idct_cos'
/// @param  N     length of vec, must be power of 2
template <typename TValue, typename TIndex = unsigned>
inline void idxst(TValue *vec, TValue *temp, const TValue *cos, TIndex N)
{
    // Left shift vec by 1 and pad 0 at vec[N - 1]
    // Then reverse vec
    vec[0] = 0;
    for (TIndex i = 1, halfN = N / 2; i < halfN; ++i)
    {
        std::swap(vec[i], vec[N - i]);
    }
    idct<TValue, TIndex>(vec, temp, cos, N);
    negateOddEntries(vec, N);
}

/// Compute dct(dct(mtx)^T)^T
/// @param  mtx   size M * N column-major matrix to be transformed
/// @param  temp  length 3 * M * N helping buffer, first 2 * M * N is for dct, the last M * N is for matrix transpose
/// @param  cosM  length M - 1, stores cosine values precomputed by function 'precompute_dct_cos' for M-point dct
/// @param  cosN  length N - 1, stores cosine values precomputed by function 'precompute_dct_cos' for N-point dct
/// @param  M     number of rows
/// @param  N     number of columns
template <typename TValue, typename TIndex = unsigned>
inline void dcct2(TValue *mtx, TValue *temp, const TValue *cosM, const TValue *cosN, TIndex M, TIndex N)
{
    #pragma omp parallel for schedule(static)
    for (TIndex i = 0; i < N; ++i)
    {
        dct<TValue, TIndex>(mtx + i * M, temp + 2 * i * M, cosM, M);
    }
    transpose<TValue, TIndex>(mtx, temp + 2 * M * N, M, N);

    #pragma omp parallel for schedule(static)
    for (TIndex i = 0; i < M; ++i)
    {
        dct<TValue, TIndex>(temp + 2 * M * N + i * N, temp + 2 * i * N, cosN, N);
    }
    transpose<TValue, TIndex>(temp + 2 * M * N, mtx, N, M);
}

/// Compute idct(idct(mtx)^T)^T
/// @param  mtx    size M * N column-major matrix to be transformed
/// @param  temp   length 3 * M * N helping buffer, first 2 * M * N is for idct, the last M * N is for matrix transpose
/// @param  cosM   length M - 1, stores cosine values precomputed by function 'precompute_idct_cos' for M-point dct
/// @param  cosN   length N - 1, stores cosine values precomputed by function 'precompute_idct_cos' for N-point dct
/// @param  M      number of rows
/// @param  N      number of columns
template <typename TValue, typename TIndex = unsigned>
inline void idcct2(TValue *mtx, TValue *temp, const TValue *cosM, const TValue *cosN, TIndex M, TIndex N)
{
    #pragma omp parallel for schedule(static)
    for (TIndex i = 0; i < N; ++i)
    {
        idct<TValue, TIndex>(mtx + i * M, temp + 2 * i * M, cosM, M);
    }
    transpose<TValue, TIndex>(mtx, temp + 2 * M * N, M, N);

    #pragma omp parallel for schedule(static)
    for (TIndex i = 0; i < M; ++i)
    {
        idct<TValue, TIndex>(temp + 2 * M * N + i * N, temp + 2 * i * N, cosN, N);
    }
    transpose<TValue, TIndex>(temp + 2 * M * N, mtx, N, M);
}

/// Compute idxst(idct(mtx)^T)^T
/// @param  mtx   size M * N column-major matrix to be transformed
/// @param  temp  length 3 * M * N helping buffer, first 2 * M * N is for idct, the last M * N is for matrix transpose
/// @param  cosM  length M - 1, stores cosine values precomputed by function 'precompute_idct_cos' for M-point dct
/// @param  cosN  length N - 1, stores cosine values precomputed by function 'precompute_idct_cos' for N-point dct
/// @param  M     number of rows
/// @param  N     number of columns
template <typename TValue, typename TIndex = unsigned>
inline void idsct2(TValue *mtx, TValue *temp, const TValue *cosM, const TValue *cosN, TIndex M, TIndex N)
{
    #pragma omp parallel for schedule(static)
    for (TIndex i = 0; i < N; ++i)
    {
        idct<TValue, TIndex>(mtx + i * M, temp + 2 * i * M, cosM, M);
    }
    transpose<TValue, TIndex>(mtx, temp + 2 * M * N, M, N);

    #pragma omp parallel for schedule(static)
    for (TIndex i = 0; i < M; ++i)
    {
        idxst<TValue, TIndex>(temp + 2 * M * N + i * N, temp + 2 * i * N, cosN, N);
    }
    transpose<TValue, TIndex>(temp + 2 * M * N, mtx, N, M);
}

/// Compute idct(idxst(mtx)^T)^T
/// @param  mtx   size M * N column-major matrix to be transformed
/// @param  temp  length 3 * M * N helping buffer, first 2 * M * N is for idct, the last M * N is for matrix transpose
/// @param  cosM  length M - 1, stores cosine values precomputed by function 'precompute_idct_cos' for M-point dct
/// @param  cosN  length N - 1, stores cosine values precomputed by function 'precompute_idct_cos' for N-point dct
/// @param  M     number of rows
/// @param  N     number of columns
template <typename TValue, typename TIndex = unsigned>
inline void idcst2(TValue *mtx, TValue *temp, const TValue *cosM, const TValue *cosN, TIndex M, TIndex N)
{
    #pragma omp parallel for schedule(static)
    for (TIndex i = 0; i < N; ++i)
    {
        idxst<TValue, TIndex>(mtx + i * M, temp + 2 * i * M, cosM, M);
    }
    transpose<TValue, TIndex>(mtx, temp + 2 * M * N, M, N);

    #pragma omp parallel for schedule(static)
    for (TIndex i = 0; i < M; ++i)
    {
        idct<TValue, TIndex>(temp + 2 * M * N + i * N, temp + 2 * i * N, cosN, N);
    }
    transpose<TValue, TIndex>(temp + 2 * M * N, mtx, N, M);
}

} // End of namespace eplace::fft
} // End of namespace eplace

PROJECT_NAMESPACE_END

#endif // __EPLACE_FFT_H__