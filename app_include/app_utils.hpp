#ifndef APP_UTILS_HPP
#define APP_UTILS_HPP

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <functional>
#include <random>

#include "hopper_gemm_utils.hpp"

std::default_random_engine generator(69);

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

template <typename T>
constexpr cudaDataType_t cuda_data_type_trait()
{
    if (std::is_same<T, float>::value)
    {
        return CUDA_R_32F;
    }

    throw std::runtime_error("Unsupported data type.");
}

template <typename T>
void run_cublas_gemm(cublasHandle_t handle, int m, int n, int k, T alpha,
                     const T* A, int lda, const T* B, int ldb, T beta, T* C,
                     int ldc)
{
    constexpr cudaDataType_t data_type{cuda_data_type_trait<T>()};

    CHECK_CUBLASS_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                     &alpha, B, data_type, ldb, A, data_type,
                                     lda, &beta, C, data_type, ldc, data_type,
                                     CUBLAS_GEMM_DEFAULT));
}

float measure_latency(std::function<void(cudaStream_t)> bound_function,
                      cudaStream_t stream, int num_repeats)
{
    float latency;

    // Create events
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Measure latency
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i = 0; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&latency, start, stop));

    // Destroy events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return latency / num_repeats;
}

template <typename T>
void random_initialize(T* A, int m)
{
    std::normal_distribution<T> distribution(0, 1);
    for (int i = 0; i < m; ++i)
    {
        A[i] = distribution(generator);
    }
}

template <typename T>
bool all_close(T const* C, T const* C_ref, size_t m, size_t n, size_t ldc,
               T abs_tol, double rel_tol)
{
    bool status{true};
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            double const C_val{static_cast<double>(C[i * ldc + j])};
            double const C_ref_val{static_cast<double>(C_ref[i * ldc + j])};
            double const diff{C_val - C_ref_val};
            double const diff_val{std::abs(diff)};
            if (diff_val >
                std::max(static_cast<double>(abs_tol),
                         static_cast<double>(std::abs(C_ref_val)) * rel_tol))
            {
                std::cout << "C[" << i << ", " << j << "] = " << C_val
                          << " C_ref[" << i << ", " << j << "] = " << C_ref_val
                          << " Abs Diff: " << diff_val
                          << " Abs Diff Threshold: "
                          << static_cast<double>(abs_tol)
                          << " Rel->Abs Diff Threshold: "
                          << static_cast<double>(
                                 static_cast<double>(std::abs(C_ref_val)) *
                                 rel_tol)
                          << std::endl;
                status = false;
                return status;
            }
        }
    }
    return status;
}

template <typename T>
float compute_effective_tflops(int m, int n, int k, float latency)
{
    float latency_seconds{latency * 1e-3f};
    float num_flops{2.0f * m * n * k};
    return num_flops / latency_seconds * 1e-12f;
}

template <typename T>
float compute_effective_bandwidth(int m, int n, int k, float latency)
{
    float latency_seconds{latency * 1e-3f};
    float num_bytes{(m * k + k * n + m * n) * sizeof(T) + .0f};
    return num_bytes / latency * 1e-9f;
}

template <typename T>
void print_performance(int m, int n, int k, float latency)
{
    float tflops = compute_effective_tflops<T>(m, n, k, latency);
    float bandwidth = compute_effective_bandwidth<T>(m, n, k, latency);
    std::cout << "Latency: " << latency << " ms" << std::endl;
    std::cout << "Effective TFLOPS: " << tflops << " TFLOPS" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;
}

#endif // APP_UTILS_HPP