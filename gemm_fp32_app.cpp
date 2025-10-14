#include <cublas_v2.h>
#include <iostream>

#include "app_utils.hpp"
#include "hopper_gemm.hpp"
#include "hopper_gemm_utils.hpp"

int main()
{
    // Configure matrices
    constexpr int align_bytes{16};
    constexpr int m{4096};
    constexpr int n{4096};
    constexpr int k{4096};
    constexpr int lda{CEIL_DIV(k, align_bytes) * align_bytes};
    constexpr int ldb{CEIL_DIV(n, align_bytes) * align_bytes};
    constexpr int ldc{CEIL_DIV(n, align_bytes) * align_bytes};
    constexpr float alpha{0.5f};
    constexpr float beta{0.5f};

    // Allocate and initalize matrices on host
    float *A_host{nullptr}, *B_host{nullptr}, *C_host{nullptr},
        *C_ref_host{nullptr};
    CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m * lda * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B_host, k * ldb * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host, m * ldc * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_ref_host, m * ldc * sizeof(float)));
    random_initialize(A_host, m * lda);
    random_initialize(B_host, k * ldb);
    random_initialize(C_host, m * ldc);

    // Allocate and copy matrices on device
    float *A_device{nullptr}, *B_device{nullptr}, *C_device{nullptr},
        *C_ref_device{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&A_device, m * lda * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_device, k * ldb * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_device, m * ldc * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_ref_device, m * ldc * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m * lda * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_device, B_host, k * ldb * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m * ldc * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_ref_device, C_host, m * ldc * sizeof(float),
                                cudaMemcpyHostToDevice));

    // Configure stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Configure cublas
    cublasHandle_t cublas_handle;
    CHECK_CUBLASS_ERROR(cublasCreate(&cublas_handle));
    CHECK_CUBLASS_ERROR(cublasSetStream(cublas_handle, stream));

    // Test and verify results
    // Test cublas
    run_cublas_gemm(cublas_handle, m, n, k, alpha, A_device, lda, B_device, ldb,
                    beta, C_ref_device, ldc);
    // Test hopper gemm
    hopper_gemm_fp32(stream, m, n, k, alpha, A_device, lda, B_device, ldb, beta,
                     C_device, ldc);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(C_ref_host, C_ref_device,
                                m * ldc * sizeof(float),
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(C_host, C_device, m * ldc * sizeof(float),
                                cudaMemcpyDeviceToHost));
    // Verify results
    constexpr float abs_tol{1.0e-3f};
    constexpr double rel_tol{0.0e-4f};
    bool is_close = all_close(C_host, C_ref_host, m, n, ldc, abs_tol, rel_tol);

    if (!is_close)
    {
        std::cout << "Terminated due to wrong results." << std::endl;
    }

    // Run hopper gemm
    // hopper_gemm_fp32(stream, 0, 0, 0, 0, nullptr, 0, nullptr, 0, 0, nullptr,
    // 0);

    return 0;
}