cmake_minimum_required(VERSION 3.28)

project(HopperSGEMM-Application VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Install the library
add_subdirectory(src)

# Build testing application
add_executable(gemm_fp32_app gemm_fp32_app.cpp)
target_link_libraries(gemm_fp32_app hopper_sgemm CUDA::cublas)