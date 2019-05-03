/*
 * @Author: Jake Gu
 * @Date: 2019-04-21 14:50:47
 * @LastEditTime: 2019-05-01 11:04:24
 */
 #ifndef __CUDA_UTILS_H__
 #define __CUDA_UTILS_H__

 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 #include <chrono>
 #include <cufft.h>
 #include <assert.h>
 #include <cmath>
 #include <cstdlib>
 #include <iostream>
 #include <string>
 #include <fstream>

 #ifdef CUBLAS
 #include <cublas_v2.h>
 #endif

 /*******************/
 /* Constant Values */
 /*******************/
 #define PI (3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481)

 /**********************/
 /* CUDA ERROR CHECK */
 /**********************/
 #ifndef cudaSafeCall
 #define cudaSafeCall(status) __cudaSafeCall(status, __FILE__, __LINE__)
 #endif

 inline void __cudaSafeCall(cudaError_t status, const char *file, const int line)
 {
     if (status != cudaSuccess)
     {
         fprintf(stderr, "CUDA error in file '%s', line %d\n \nerror %d \nterminating!\n%s\n", __FILE__, __LINE__, status, cudaGetErrorString(status));
         getchar();
         cudaDeviceReset();
         assert(status == cudaSuccess);
     }
 }

 /**********************/
 /* cuBLAS ERROR CHECK */
 /**********************/
 #ifdef CUBLAS
 #ifndef cublasSafeCall
 #define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
 #endif

 inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
 {
     if (CUBLAS_STATUS_SUCCESS != err)
     {
         fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n", __FILE__, __LINE__, err);
         getchar();
         cudaDeviceReset();
         assert(CUBLAS_STATUS_SUCCESS == err);
     }
 }
 #endif

 /*****************************/
 /* High Resolution CPU Clock */
 /*****************************/
 typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

 struct CpuTimer
 {
     std::chrono::high_resolution_clock::rep start;
     std::chrono::high_resolution_clock::rep stop;

     CpuTimer() {}

     ~CpuTimer() {}

     void Start()
     {
         using namespace std::chrono;
         start = high_resolution_clock::now().time_since_epoch().count();
     }

     void Stop()
     {
         using namespace std::chrono;
         stop = high_resolution_clock::now().time_since_epoch().count();
     }

     double ElapsedMillis()
     {
         using namespace std::chrono;
         double elapsed;
         elapsed = (stop - start) * 1000.0 * high_resolution_clock::period::num / high_resolution_clock::period::den;
         return elapsed;
     }
 };

 /*****************************/
 /* High Resolution GPU Clock */
 /*****************************/
 struct GpuTimer
 {
     cudaEvent_t start;
     cudaEvent_t stop;

     GpuTimer()
     {
         cudaEventCreate(&start);
         cudaEventCreate(&stop);
     }

     ~GpuTimer()
     {
         cudaEventDestroy(start);
         cudaEventDestroy(stop);
     }

     void Start()
     {
         cudaEventRecord(start, 0);
     }

     void Stop()
     {
         cudaEventRecord(stop, 0);
     }

     float ElapsedMillis()
     {
         float elapsed;
         cudaEventSynchronize(stop);
         cudaEventElapsedTime(&elapsed, start, stop);
         return elapsed;
     }
 };

 /**************/
 /* Arithmetic */
 /**************/

 /// Return true if a number is power of 2
 template <typename T = unsigned>
 inline __device__ __host__ bool isPowerOf2(T val)
 {
     return val && (val & (val - 1)) == 0;
 }

 template <typename T>
 inline __device__ __host__ void swap(T &x, T &y)
 {
     T tmp = x;
     x = y;
     y = tmp;
 }

 inline __device__ int INDEX(const int hid, const int wid, const int N)
 {
     return (hid * N + wid);
 }

 inline __device__ __host__ int LogBase2(uint64_t n)
 {
     static const int table[64] = {
         0, 58, 1, 59, 47, 53, 2, 60, 39, 48, 27, 54, 33, 42, 3, 61,
         51, 37, 40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4, 62,
         57, 46, 52, 38, 26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21, 56,
         45, 25, 31, 35, 16, 9, 12, 44, 24, 15, 8, 23, 7, 6, 5, 63};

     n |= n >> 1;
     n |= n >> 2;
     n |= n >> 4;
     n |= n >> 8;
     n |= n >> 16;
     n |= n >> 32;

     return table[(n * 0x03f6eaf2cd271461) >> 58];
 }

 template <typename T>
 struct  ComplexType
 {
     T x;
     T y;
     __host__ __device__ ComplexType()
     {
         x = 0;
         y = 0;
     }

     __host__ __device__ ComplexType(T real, T imag)
     {
         x = real;
         y = imag;
     }

     __host__ __device__ ~ComplexType() {}
 };

 template <typename T>
 inline __device__ ComplexType<T> complexMul(const ComplexType<T> &x, const ComplexType<T> &y)
 {
     ComplexType<T> res;
     res.x = x.x * y.x - x.y * y.y;
     res.y = x.x * y.y + x.y * y.x;
     return res;
 }

 template <typename T>
 inline __device__ T RealPartOfMul(const ComplexType<T> &x, const ComplexType<T> &y)
 {
     return x.x * y.x - x.y * y.y;
 }

 template <typename T>
 inline __device__ T ImaginaryPartOfMul(const ComplexType<T> &x, const ComplexType<T> &y)
 {
     return x.x * y.y + x.y * y.x;
 }

 template <typename T>
 inline __device__ ComplexType<T> complexAdd(const ComplexType<T> &x, const ComplexType<T> &y)
 {
     ComplexType<T> res;
     res.x = x.x + y.x;
     res.y = x.y + y.y;
     return res;
 }

 template <typename T>
 inline __device__ ComplexType<T> complexSubtract(const ComplexType<T> &x, const ComplexType<T> &y)
 {
     ComplexType<T> res;
     res.x = x.x - y.x;
     res.y = x.y - y.y;
     return res;
 }

 template <typename T>
 inline __device__ ComplexType<T> complexConj(const ComplexType<T> &x)
 {
     ComplexType<T> res;
     res.x = x.x;
     res.y = -x.y;
     return res;
 }

 template <typename T>
 inline __device__ ComplexType<T> complexMulConj(const ComplexType<T> &x, const ComplexType<T> &y)
 {
     ComplexType<T> res;
     res.x = x.x * y.x - x.y * y.y;
     res.y = -(x.x * y.y + x.y * y.x);
     return res;
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

 #ifdef CUBLAS
 inline void transposeCUBLAS(double *&src_ptr, double *&dst, const int M, const int N)
 {
     double alpha = 1.;
     double beta = 0.;
     const double *src = (const double *)src_ptr;
     cublasHandle_t handle;
     cublasSafeCall(cublasCreate(&handle));
     cublasSafeCall(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, &alpha, src, N, &beta, src, N, dst, M));
     cublasDestroy(handle);
 }

 inline void transposeCUBLAS(float *&src_ptr, float *&dst, const int M, const int N)
 {
     float alpha = 1.;
     float beta = 0.;
     const float *src = (const float *)src_ptr;
     cublasHandle_t handle;
     cublasSafeCall(cublasCreate(&handle));
     cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, &alpha, src, N, &beta, src, N, dst, M));
     cublasDestroy(handle);
 }

 #endif

 /*******************/
 /* Print CUDA Array*/
 /*******************/

 template <typename T>
 void printCUDAArray(const T *x, const int n, const char *str)
 {
     printf("%s[%d] = ", str, n);
     T *host_x = (T *)malloc(n * sizeof(T));
     if (host_x == NULL)
     {
         printf("failed to allocate memory on CPU\n");
         return;
     }
     cudaMemcpy(host_x, x, n * sizeof(T), cudaMemcpyDeviceToHost);
     for (int i = 0; i < n; ++i)
     {
         printf("%g ", double(host_x[i]));
     }
     printf("\n");

     free(host_x);
 }

 template <typename T>
 void printCUDAScalar(const T &x, const char *str)
 {
     printf("%s = ", str);
     T *host_x = (T *)malloc(sizeof(T));
     if (host_x == NULL)
     {
         printf("failed to allocate memory on CPU\n");
         return;
     }
     cudaMemcpy(host_x, &x, sizeof(T), cudaMemcpyDeviceToHost);
     printf("%g\n", double(*host_x));

     free(host_x);
 }

 template <typename T>
 void printCUDA2DArray(const T *x, const int m, const int n, const char *str)
 {
     printf("%s[%dx%d] = \n", str, m, n);
     T *host_x = (T *)malloc(m * n * sizeof(T));
     if (host_x == NULL)
     {
         printf("failed to allocate memory on CPU\n");
         return;
     }
     cudaMemcpy(host_x, x, m * n * sizeof(T), cudaMemcpyDeviceToHost);
     for (int i = 0; i < m * n; ++i)
     {
         if (i && (i % n) == 0)
         {
             printf("\n");
         }
         printf("%g ", double(host_x[i]));
     }
     printf("\n");

     free(host_x);
 }

 #endif // __CUDA_UTILS_H__
