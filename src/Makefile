ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60

OPTIONS=-O3 -rdc=true -use_fast_math -lcurand -Xcompiler -Wall -lineinfo -Xcompiler -fopenmp --default-stream per-thread -lcublas -lcufft #-maxrregcount 37

all: dct_cuda dct_cuda_lee dct2d_cuda dct2d_cuda_lee dct2d_cuda_fft

dct_cuda: dct_cuda.cu
	mkdir -p bin
	nvcc -w $(ARCH) $(OPTIONS) -o bin/dct_cuda dct_cuda.cu

dct_cuda_lee: dct_cuda_lee.cu
	mkdir -p bin
	nvcc -w $(ARCH) $(OPTIONS) -o bin/dct_cuda_lee dct_cuda_lee.cu

dct2d_cuda: dct2d_cuda.cu
	mkdir -p bin
	nvcc -w $(ARCH) $(OPTIONS) -maxrregcount 20 -o bin/dct2d_cuda dct2d_cuda.cu

dct2d_cuda_lee: dct2d_cuda_lee.cu
	mkdir -p bin
	nvcc -w $(ARCH) $(OPTIONS) -o bin/dct2d_cuda_lee dct2d_cuda_lee.cu

dct2d_cuda_fft: dct2d_cuda_fft.cu
	mkdir -p bin
	nvcc -w $(ARCH) $(OPTIONS) -o bin/dct2d_cuda_fft dct2d_cuda_fft.cu

dct2d_cuda_fft_noexp: dct2d_cuda_fft_noexpansion.cu
	mkdir -p bin
	nvcc -w $(ARCH) $(OPTIONS) -prec-div=false -ftz=true -o bin/dct2d_cuda_fft_noexp dct2d_cuda_fft_noexpansion.cu

clean:
	rm -rf bin lib
