cmake_minimum_required(VERSION 3.18)
project(SGEMM_CUTE LANGUAGES CXX CUDA)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找 CUDA 和 cuBLAS
find_package(CUDA REQUIRED)
find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
if(NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "cuBLAS not found!")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include)

add_executable(sgemm_cute sgemm_cute.cu)

target_link_libraries(sgemm_cute ${CUBLAS_LIBRARY} ${CUDA_CUDA_LIBRARY})

# === 关键：定义宏，激活代码中的 cuBLAS 分支 ===
target_compile_definitions(sgemm_cute PRIVATE CUTLASS_ENABLE_CUBLAS=1)