# CUDA implementation of discrete cosine transform (DCT)
src/dct are imported from DREAMPlace
https://github.com/limbo018/DREAMPlace.
Other files are developed by Jiaqi Gu and Zixuan Jiang.
This implmentation is integrated into the DREAMPlace.

# Transforms
* 1d/2d DCT
* 1d/2d IDCT
* IDCT_IDXST
* IDXST_IDCT

# Algorithms
* Naive methods. Directly compute the DCT according to the definition.
* Lee’s Algorithm. The proposed algorithm in the paper is recursive. We have implemented the iterative version, which is more friendly to GPU.
```
Byeong Lee, "A new algorithm to compute the discrete cosine Transform," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 32, no. 6, pp. 1243-1245, December 1984.

The iterative version is demonstrated at
https://www.codeproject.com/Articles/151043/Iterative-Fast-1D-Forvard-DCT
```
* DCT with 4N, 2N and N point FFT. (1) Reorder the input matrix, (2) FFT, and (3) Postprocessing.
```
J. Makhoul, "A fast cosine transform in one and two dimensions," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 28, no. 1, pp. 27-34, February 1980.

https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
```

# Results
We take the implementation in DREAMPlace as baseline.

# Acknowledgement
We acknowledge the support from Yibo Lin and Wuxi Li.
