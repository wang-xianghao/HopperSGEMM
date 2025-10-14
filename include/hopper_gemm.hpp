#ifndef HOPPER_GEMM_HPP
#define HOPPER_GEMM_HPP

void hopper_gemm_fp32(int m, int n, int k, float alpha, const float* A, int lda,
                      const float* B, int ldb, float beta, float* C, int ldc);

#endif // HOPPER_GEMM_HPP