cmake_minimum_required(VERSION 3.28)

project(HopperSGEMM-Application VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Install the library
add_subdirectory(src)

# Build testing application
add_executable(test_sgemm test_sgemm.cpp)
target_link_libraries(test_sgemm hopper_sgemm)