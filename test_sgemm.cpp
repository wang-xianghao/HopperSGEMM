#include <cublas_v2.h>
#include <iostream>

#include "hopper_sgemm.hpp"
#include "hopper_sgemm_utils.hpp"

#define CHECK_CUBLASS_ERROR(val) check_cublass((val), #val, __FILE__, __LINE__)
void check_cublass(cublasStatus_t err, const char* const func,
                   const char* const file, const int line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Error at: " << file << ":" << line << std::endl;
        std::cerr << cublasGetStatusString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void run_cublas_gemm_fp32(cublasHandle_t handle, int m, int n, int k,
                          float alpha, const float* A, int lda, const float* B,
                          int ldb, float beta, float* C, int ldc)
{
    constexpr cudaDataType_t data_type{CUDA_R_32F};
    CHECK_CUBLASS_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                     &alpha, B, data_type, ldb, A, data_type,
                                     lda, &beta, C, data_type, ldc, data_type,
                                     CUBLAS_GEMM_DEFAULT));
}

int main()
{
    // Configure matrices
    const int m{4096};
    const int n{4096};
    const int k{4096};
    const float alpha{0.5f};
    const float beta{0.5f};

    // Allocate and initalize matrices

    // Configure cublas
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // Test and verify results
    // Run cublas

    // Run hopper sgemm
    hopper_sgemm(0, 0, 0, 0, nullptr, 0, nullptr, 0, 0, nullptr, 0);

    return 0;
}