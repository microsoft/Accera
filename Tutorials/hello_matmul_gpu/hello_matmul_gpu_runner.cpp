#include <stdio.h>
#include <algorithm>

// Include the HAT file that declares GPU initialization/uninitialization functions
#include "AcceraGPUUtilities.hat"

// Include the HAT file that declares our MatMul function
#include "hello_matmul_gpu.hat"

#define M 1024
#define N 512
#define K 256

int main(int argc, const char** argv)
{
    // Prepare our matrices (using the heap for large matrices)
    float* A = new float[M*K];
    float* B = new float[K*N];
    float* C = new float[M*N];

    // Fill with data
    std::fill_n(A, M*K, 2.0f);
    std::fill_n(B, K*N, 3.0f);
    std::fill_n(C, M*N, 0.42f);

    // Initialize the GPU
    AcceraGPUInitialize();

    printf("Calling MatMul M=%d, K=%d, N=%d\n", M, K, N);
    hello_matmul_gpu(A, B, C);

    printf("Result (first 10 elements): ");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f ", C[i]);
    }
    printf("\n");

    // Uninitialize the GPU
    AcceraGPUDeInitialize();

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
