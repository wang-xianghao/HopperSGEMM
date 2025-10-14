#include <hopper_gemm.hpp>
#include <hopper_gemm_utils.hpp>

__global__ void hopper_gemm_fp32_kernel(int m, int n, int k, float alpha,
                                        const float* A, int lda, const float* B,
                                        int ldb, float beta, float* C, int ldc)
{
}

void hopper_gemm_fp32(cudaStream_t stream, int m, int n, int k, float alpha,
                      const float* A, int lda, const float* B, int ldb,
                      float beta, float* C, int ldc)
{
    dim3 blockDim(32 * 32, 1);
    dim3 gridDim(128, 128, 1);
    hopper_gemm_fp32_kernel<<<gridDim, blockDim, 0, stream>>>(
        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}