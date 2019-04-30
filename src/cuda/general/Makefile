BIN_DIR := $(abspath ../../test/bin)

ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60

OPTIONS=-O3 -use_fast_math -lcurand -Xcompiler -Wall -lcublas -lcufft -prec-div=false -ftz=true -lineinfo # -rdc=true -Xcompiler -fopenmp --default-stream per-thread  #-maxrregcount 37

all: dct1d_lee dct2d_naive dct2d_lee dct2d_fft idct2d_fft idct_idxst idxst_idct

dct1d_lee: dct1d_lee.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

dct2d_naive: dct2d_naive.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

dct2d_lee: dct2d_lee.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

dct2d_fft: dct2d_fft.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

idct2d_fft: idct2d_fft.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

idct_idxst: idct_idxst.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

idxst_idct: idxst_idct.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

clean:
	rm -rf $(BIN_DIR)
