#ifndef APP_UTILS_HPP
#define APP_UTILS_HPP

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

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

void print_device_info()
{
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    std::cout << "Device Name: " << device_prop.name << std::endl;
    // Memory specs
    int memoryClockRate{0};
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate,
                           device_id);
    const float memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    float const peak_bandwidth{static_cast<float>(
        2.0f * memoryClockRate * (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
    std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << "Bus Width: " << device_prop.memoryBusWidth << " Bit"
              << std::endl;
    // SM specs
    std::cout << "SM Count: " << device_prop.multiProcessorCount << std::endl;
    std::cout << "Maximum Blocks per SM: "
              << device_prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Maximum Threads per SM: "
              << device_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Registers per SM: " << device_prop.regsPerMultiprocessor
              << std::endl;
    std::cout << "Shared Memory per SM: "
              << device_prop.sharedMemPerMultiprocessor << " Bytes"
              << std::endl;

    std::cout << std::endl;
}

void parse_command_arguments(int* m, int* n, int* k, int* num_repeats, int argc,
                             char* argv[])
{
    std::string exec(argv[0]);
    std::vector<std::string> args;
    args.assign(argv + 1, argv + argc);
    argc -= 1;

    // Set default values
    *m = 4096;
    *n = 4096;
    *k = 4096;
    *num_repeats = 8;

    for (int i = 0; i < argc - 1; ++i)
    {
        if (args[i] == "-m")
        {
            i += 1;
            *m = std::stoi(args[i]);
        }
        else if (args[i] == "-n")
        {
            i += 1;
            *n = std::stoi(args[i]);
        }
        else if (args[i] == "-k")
        {
            i += 1;
            *k = std::stoi(args[i]);
        }
        else if (args[i] == "-r")
        {
            i += 1;
            *num_repeats = std::stoi(args[i]);
        }
        else
        {
            fprintf(stderr, "%s [-m <m>] [-n <n>] [-k <k>] [-r <r>]\n");
            std::exit(1);
        }
    }
}

#endif // APP_UTILS_HPP