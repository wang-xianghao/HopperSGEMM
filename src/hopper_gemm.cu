#include <hopper_gemm.hpp>
#include <hopper_gemm_utils.hpp>

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_THREADS>
__global__ void hopper_gemm_fp32_kernel(int m, int n, int k, float alpha,
                                        const float* A, int lda, const float* B,
                                        int ldb, float beta, float* C, int ldc)
{
    const int blockRowC = blockIdx.x;
    const int blockColC = blockIdx.y;
    // A tile
    const int rowA = threadIdx.x / BLOCK_K;
    const int colA = threadIdx.x % BLOCK_K;
    constexpr int rowStrideA = NUM_THREADS / BLOCK_K;
    // B tile
    const int rowB = threadIdx.x / BLOCK_N;
    const int colB = threadIdx.x % BLOCK_N;
    constexpr int rowStrideB = NUM_THREADS / BLOCK_N;
    // C tile
    const int rowC = threadIdx.x / BLOCK_N;
    const int colC = threadIdx.x % BLOCK_N;
    constexpr int rowStrideC = NUM_THREADS / BLOCK_N;

    constexpr int NITER = (BLOCK_M * BLOCK_N) / NUM_THREADS;

    float threadResults[NITER] = {0.0f};
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];

    A += blockRowC * BLOCK_M * lda;
    B += blockColC * BLOCK_N;
    C += blockRowC * BLOCK_M * ldc + blockColC * BLOCK_N;
    for (int bkIdx = 0; bkIdx + BLOCK_K <= k; bkIdx += BLOCK_K)
    {
        // Load input tiles to shared memory
        for (int offset = 0; offset + rowStrideA <= BLOCK_M;
             offset += rowStrideA)
        {
            As[rowA + offset][colA] = A[(rowA + offset) * lda + colA];
        }
        for (int offset = 0; offset + rowStrideB <= BLOCK_K;
             offset += rowStrideB)
        {
            Bs[rowB + offset][colB] = B[(rowB + offset) * ldb + colB];
        }

        __syncthreads();
        A += BLOCK_K;
        B += BLOCK_K * ldb;

        // Compute output tile
        for (int offset = 0, iter = 0; offset + rowStrideC <= BLOCK_M;
             offset += rowStrideC, iter += 1)
        {
            for (int dotIdx = 0; dotIdx < BLOCK_K; ++dotIdx)
            {
                threadResults[iter] +=
                    As[rowC + offset][dotIdx] * Bs[dotIdx][colB];
            }
        }

        __syncthreads();
    }

    for (int offset = 0, iter = 0; offset + rowStrideC <= BLOCK_M;
         offset += rowStrideC, iter += 1)
    {
        C[(rowC + offset) * ldc + colC] =
            alpha * threadResults[iter] +
            beta * C[(rowC + offset) * ldc + colC];
    }
}

void hopper_gemm_fp32(cudaStream_t stream, int m, int n, int k, float alpha,
                      const float* A, int lda, const float* B, int ldb,
                      float beta, float* C, int ldc)
{
    constexpr int BLOCK_M{32};
    constexpr int BLOCK_N{32};
    constexpr int BLOCK_K{16};
    constexpr int NUM_THREADS{128};

    // Ensure alignment for loading A tile
    static_assert(NUM_THREADS % BLOCK_K == 0);
    static_assert((BLOCK_M * BLOCK_K) % NUM_THREADS == 0);
    // Ensure alignment for loading B tile
    static_assert(NUM_THREADS % BLOCK_N == 0);
    static_assert((BLOCK_K * BLOCK_N) % NUM_THREADS == 0);
    // Ensure alignment for computing C tile
    static_assert((BLOCK_M * BLOCK_N) % NUM_THREADS == 0);

    dim3 blockDim(NUM_THREADS, 1);
    dim3 gridDim(CEIL_DIV(m, BLOCK_M), CEIL_DIV(n, BLOCK_N), 1);
    hopper_gemm_fp32_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS>
        <<<gridDim, blockDim, 0, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta,
                                           C, ldc);
    CHECK_LAST_CUDA_ERROR();
}