BIN_DIR := $(abspath ../../test/bin)

ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60

OPTIONS=-O3 -use_fast_math -lcurand -Xcompiler -Wall -lcublas -lcufft -prec-div=false -ftz=true -lineinfo # -rdc=true -Xcompiler -fopenmp --default-stream per-thread  #-maxrregcount 37

all: dct2_naive dct2_lee dct2_fft2 idct2_fft2 idct_idxst idxst_idct

dct2_naive: dct2_naive.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

dct2_lee: dct2_lee.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

dct2_fft2: dct2_fft2.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

idct2_fft2: idct2_fft2.cu
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
