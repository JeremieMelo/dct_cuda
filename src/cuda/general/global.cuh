#include "../utils/cuda_utils.cuh"

#if 1
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

#define NUM_RUNS (101)
