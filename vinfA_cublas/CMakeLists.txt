cmake_minimum_required(VERSION 3.20)
project(vinfA_cublas LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES  86)
add_executable(vinfA_cublas main.cu ../common/check.cpp ../common/matrix.cpp)
target_compile_features(vinfA_cublas PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(vinfA_cublas PRIVATE CUDA::cublas)