#include <stdio.h>
#include <algorithm>

// Include the HAT file that declares our MatMul function
#include "hello_matmul.hat"

#define M 128
#define N 256
#define K 256

int main(int argc, const char** argv)
{
    // Prepare our matrices
    float A[M*K];
    float B[K*N];
    float C[M*N];

    // Fill with data
    std::fill_n(A, M*K, 2.0f);
    std::fill_n(B, K*N, 3.0f);
    std::fill_n(C, M*N, 0.42f);

    printf("Calling MatMul M=%d, K=%d, N=%d\n", M, K, N);
    hello_matmul_py(A, B, C);

    printf("Result (first few elements): ");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f ", C[i]);
    }
    printf("\n");
    return 0;
}
