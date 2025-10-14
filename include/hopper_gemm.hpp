#ifndef HOPPER_GEMM_HPP
#define HOPPER_GEMM_HPP

#include <cuda_runtime.h>

void hopper_gemm_fp32(cudaStream_t stream, int m, int n, int k, float alpha,
                      const float* A, int lda, const float* B, int ldb,
                      float beta, float* C, int ldc);

#endif // HOPPER_GEMM_HPP