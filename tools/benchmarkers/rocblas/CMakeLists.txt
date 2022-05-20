add_executable(rocblas_gemm EXCLUDE_FROM_ALL main.cpp)
target_include_directories(rocblas_gemm PUBLIC /opt/rocm/include)
target_link_directories(rocblas_gemm PUBLIC /opt/rocm/lib)
target_link_libraries(rocblas_gemm rocblas amdhip64)
