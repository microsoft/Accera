find_package(CUDAToolkit) # requires cmake >= 3.17

if(TARGET CUDA::cublas)
    add_executable(cublas_gemm main.cpp)
    target_link_libraries(cublas_gemm CUDA::cudart CUDA::cublas)
endif()
