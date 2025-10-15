#include <hopper_gemm.hpp>
#include <hopper_gemm_utils.hpp>

template <int BLOCK_SIZE>
__global__ void hopper_gemm_fp32_kernel(int m, int n, int k, float alpha,
                                        const float* A, int lda, const float* B,
                                        int ldb, float beta, float* C, int ldc)
{
    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;
    const int threadRow = threadIdx.x / BLOCK_SIZE;
    const int threadCol = threadIdx.x % BLOCK_SIZE;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    A += blockRow * BLOCK_SIZE * lda;
    B += blockCol * BLOCK_SIZE;
    C += blockRow * BLOCK_SIZE * ldc + blockCol * BLOCK_SIZE;

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx + BLOCK_SIZE <= k; bkIdx += BLOCK_SIZE)
    {
        // Load input tiles
        As[threadRow][threadCol] = A[threadRow * lda + threadCol];
        Bs[threadRow][threadCol] = B[threadRow * ldb + threadCol];
        __syncthreads();
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * ldb;

        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx)
        {
            tmp += As[threadRow][dotIdx] * Bs[dotIdx][threadCol];
        }
        __syncthreads();
    }

    C[threadRow * ldc + threadCol] =
        alpha * tmp + beta * C[threadRow * ldc + threadCol];
}

void hopper_gemm_fp32(cudaStream_t stream, int m, int n, int k, float alpha,
                      const float* A, int lda, const float* B, int ldb,
                      float beta, float* C, int ldc)
{
    constexpr int BLOCK_SIZE{32};
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE, 1);
    dim3 gridDim(CEIL_DIV(m, BLOCK_SIZE), CEIL_DIV(n, BLOCK_SIZE), 1);
    hopper_gemm_fp32_kernel<BLOCK_SIZE><<<gridDim, blockDim, 0, stream>>>(
        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}