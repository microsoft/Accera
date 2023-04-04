#include <stdio.h>
#include <algorithm>

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

    float* dev_A;
    float* dev_B;
    float* dev_C;
    hipMalloc(&dev_A, M * K * sizeof(float));
    hipMalloc(&dev_B, K * N * sizeof(float));
    hipMalloc(&dev_C, M * N * sizeof(float));
    hipMemcpy(dev_A, A, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dev_B, B, K * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dev_C, C, M * N * sizeof(float), hipMemcpyHostToDevice);

    printf("Calling MatMul M=%d, K=%d, N=%d\n", M, K, N);
    hello_matmul_gpu(dev_A, dev_B, dev_C);

    hipDeviceSynchronize();
    hipMemcpy(C, dev_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

    printf("Result (first 10 elements): ");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f ", C[i]);
    }
    printf("\n");

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(dev_A);
    hipFree(dev_B);
    hipFree(dev_C);
    return 0;
}
