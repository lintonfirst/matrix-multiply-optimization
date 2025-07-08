cmake_minimum_required(VERSION 3.20)
project(v0_basic LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES  86)
add_executable(v0_basic main.cu ../common/check.cpp ../common/matrix.cpp)
target_compile_features(v0_basic PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(v0_basic PRIVATE CUDA::cublas)