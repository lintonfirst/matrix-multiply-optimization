cmake_minimum_required(VERSION 3.20)
project(vinfB_cutlass LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES  86)
add_executable(vinfB_cutlass main.cu)
target_compile_features(vinfB_cutlass PRIVATE cxx_std_17 cuda_std_17)