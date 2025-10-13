#ifndef HOPPER_SGEMM_HPP
#define HOPPER_SGEMM_HPP

void hopper_sgemm(int m, int n, int k, float alpha, const float* A, int lda,
                  const float* B, int ldb, float beta, float* C, int ldc);

#endif // HOPPER_SGEMM_HPP