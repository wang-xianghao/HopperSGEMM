#include <hopper_gemm.hpp>
#include <hopper_gemm_utils.hpp>

__global__ void hopper_gemm_fp32_kernel(int m, int n, int k, float alpha,
                                        const float* A, int lda, const float* B,
                                        int ldb, float beta, float* C, int ldc)
{
}

void hopper_gemm_fp32(int m, int n, int k, float alpha, const float* A, int lda,
                      const float* B, int ldb, float beta, float* C, int ldc)
{
    dim3 blockDim(32U * 32U, 1U);
    dim3 gridDim(128U, 128U, 1U);
    hopper_gemm_fp32_kernel<<<gridDim, blockDim>>>(m, n, k, alpha, A, lda, B,
                                                   ldb, beta, C, ldc);
}