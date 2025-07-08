cmake_minimum_required(VERSION 3.20)
project(vinfB_cutlass LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (MSVC)
  # MSVC by default does not apply the correct __cplusplus version as specified by the C++ standard
  # because MSVC is not a completely compliant implementation. This option forces MSVC to use the
  # appropriate value given the requested --std option. This fixes a compilation issue mismatch
  # between GCC/Clang and MSVC.
  #
  # error : a constexpr function cannot have a nonliteral return type "dim3"
  #
  # See https://developercommunity.visualstudio.com/t/msvc-incorrectly-defines-cplusplus/139261
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler  /Zc:__cplusplus")
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES  86)
include_directories(cutlass/include)
add_executable(vinfB_cutlass main.cu ../common/check.cpp ../common/matrix.cpp)
target_compile_features(vinfB_cutlass PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(vinfB_cutlass PRIVATE CUDA::cublas)