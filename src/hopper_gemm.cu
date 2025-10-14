#include <hopper_gemm.hpp>
#include <hopper_gemm_utils.hpp>

template <int BLOCK_SIZE>
__global__ void hopper_gemm_fp32_kernel(int m, int n, int k, float alpha,
                                        const float* A, int lda, const float* B,
                                        int ldb, float beta, float* C, int ldc)
{
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    const int col = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;

    if (row < m && col < n)
    {
        float tmp = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            tmp += A[row * lda + i] * B[i * ldb + col];
        }
        C[row * ldc + col] = alpha * tmp + beta * C[row * ldc + col];
    }
}

void hopper_gemm_fp32(cudaStream_t stream, int m, int n, int k, float alpha,
                      const float* A, int lda, const float* B, int ldb,
                      float beta, float* C, int ldc)
{
    dim3 blockDim(32 * 32, 1);
    dim3 gridDim(CEIL_DIV(m, 32), CEIL_DIV(n, 32), 1);
    hopper_gemm_fp32_kernel<32><<<gridDim, blockDim, 0, stream>>>(
        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}