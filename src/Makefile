ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60

OPTIONS=-O3  -rdc=true -use_fast_math -lcurand  -Xcompiler -Wall -lineinfo -Xcompiler -fopenmp --default-stream per-thread #-maxrregcount 37

all: dct_cuda dct_cuda_lee

dct_cuda: dct_cuda.cu
	mkdir -p bin
	nvcc -w $(ARCH) $(OPTIONS) -o bin/dct_cuda dct_cuda.cu

dct_cuda_lee: dct_cuda_lee.cu
	mkdir -p bin
	nvcc -w $(ARCH) $(OPTIONS) -o bin/dct_cuda_lee dct_cuda_lee.cu

clean:
	rm -rf bin lib