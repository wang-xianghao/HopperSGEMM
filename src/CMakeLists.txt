cmake_minimum_required(VERSION 3.28)

project(HopperGEMM VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Provide information for Nsight Compute
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Find include directory
find_path(HOPPER_GEMM_INCLUDE_DIRS hopper_gemm.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
file(GLOB HOPPER_GEMM_HEADERS ${CMAKE_SOURCE_DIR}/include/*.hpp ${CMAKE_SOURCE_DIR}/include/*.cuh)

# Add all the source files in the current directory to build the library
add_library(
    hopper_gemm
    SHARED
    hopper_gemm.cu
    hopper_gemm_utils.cu
)

# Add the include directory of the library to the include directories of the project
target_include_directories(hopper_gemm PUBLIC ${HOPPER_GEMM_INCLUDE_DIRS})

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set_target_properties(hopper_gemm PROPERTIES CUDA_ARCHITECTURES native)
install(TARGETS hopper_gemm DESTINATION lib)
install(FILES ${HOPPER_GEMM_HEADERS} DESTINATION include)
