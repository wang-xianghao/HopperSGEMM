#include <hopper_sgemm.hpp>
#include <hopper_sgemm_utils.hpp>

__global__ void hopper_sgemm_kernel(int m, int n, int k, float alpha,
                                    const float* A, int lda, const float* B,
                                    int ldb, float* C, int ldc)
{
}

void hopper_sgemm(int m, int n, int k, float alpha, const float* A, int lda,
                  const float* B, int ldb, float* C, int ldc)
{
}