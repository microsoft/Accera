#include <stdio.h>
#include <algorithm>

// Include the HAT file that declares our MatMul function
#include "optimized_matmul.hat"

#define M 784
#define N 512
#define K 128

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

    printf("Calling MatMul M=%d, K=%d, N=%d\n", M, K, N);
    optimized_matmul_py(A, B, C);

    printf("Result (first 10 elements): ");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f ", C[i]);
    }
    printf("\n");

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
