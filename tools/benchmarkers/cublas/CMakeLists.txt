find_package(CUDAToolkit)

if(TARGET CUDA::cublas)
    add_executable(cublas_gemm main.cpp)
    target_link_libraries(cublas_gemm CUDA::cudart CUDA::cublas)
endif()
