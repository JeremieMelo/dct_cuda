/*
 * @Author: Jake Gu
 * @Date: 2019-04-29 21:21:31
 * @LastEditTime: 2019-04-29 21:22:09
 */
template <typename T>
void precomputeExpkCudaLauncher(
        const T* x, 
        const int M, 
        const int N, 
        T* y
        );