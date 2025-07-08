cmake_minimum_required(VERSION 3.20)
project(v1_tile LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES  86)
add_executable(v1_tile main.cu ../common/check.cpp ../common/matrix.cpp)
target_compile_features(v1_tile PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(v1_tile PRIVATE CUDA::cublas)