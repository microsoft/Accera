#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>

#define CUDA_CALL(E)                                                              \
    do                                                                            \
    {                                                                             \
        cudaError_t e = (E);                                                      \
        if (e != cudaSuccess)                                                     \
        {                                                                         \
            printf("line %d: CUDA error: %s\n", __LINE__, cudaGetErrorString(e)); \
            exit(-2);                                                             \
        }                                                                         \
    } while (false)

#define CUBLAS_CALL(S)                                          \
    do                                                          \
    {                                                           \
        cublasStatus_t s = (S);                                 \
        if (s != CUBLAS_STATUS_SUCCESS)                         \
        {                                                       \
            printf("line %d: CUBLAS error: %d\n", __LINE__, s); \
            exit(-3);                                           \
        }                                                       \
    } while (false)

#define ASSERT_NON_ZERO(X)                             \
    do                                                 \
    {                                                  \
        if ((X) <= 0)                                  \
        {                                              \
            printf("error: " #X " = %d <= 0!\n", (X)); \
            exit(-1);                                  \
        }                                              \
    } while (false)

#define ASSERT_LTE(X, Y)                                             \
    do                                                               \
    {                                                                \
        if ((X) > (Y))                                               \
        {                                                            \
            printf("error: " #X " = %d > %d = " #Y "!\n", (X), (Y)); \
            exit(-1);                                                \
        }                                                            \
    } while (false)

// cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)

cublasOperation_t trans(int t)
{
    return t ? CUBLAS_OP_T : CUBLAS_OP_N;
}

template <typename T>
void gemm(cublasHandle_t handle, int m, int n, int k, int transA, int transB, T alpha, T beta, const T* a, int lda, const T* b, int ldb, T* c, int ldc);

template <>
void gemm<float>(cublasHandle_t handle, int m, int n, int k, int transA, int transB, float alpha, float beta, const float* a, int lda, const float* b, int ldb, float* c, int ldc)
{
    CUBLAS_CALL(cublasSgemm_v2(handle, trans(transA), trans(transB), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm<double>(cublasHandle_t handle, int m, int n, int k, int transA, int transB, double alpha, double beta, const double* a, int lda, const double* b, int ldb, double* c, int ldc)
{
    CUBLAS_CALL(cublasDgemm_v2(handle, trans(transA), trans(transB), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm<__half>(cublasHandle_t handle, int m, int n, int k, int transA, int transB, __half alpha, __half beta, const __half* a, int lda, const __half* b, int ldb, __half* c, int ldc)
{
    CUBLAS_CALL(cublasHgemm(handle, trans(transA), trans(transB), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

double calc_tflops(int m, int n, int k, float ms)
{
    return 1.0e-9 * 2.0 * k * m * n / ms;
}

template <typename T>
const char* type_to_str()
{
    if constexpr (std::is_same<T, float>())
    {
        return "s";
    }

    if constexpr (std::is_same<T, double>())
    {
        return "d";
    }

    if constexpr (std::is_same<T, __half>())
    {
        return "h";
    }
}

template <typename T>
void benchmark(int m, int n, int k, int transA, int transB, T alpha, T beta, int lda, int ldb, int ldc, int gpu_id)
{
    CUDA_CALL(cudaSetDevice(gpu_id));

    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    T *a, *b, *c;
    auto Asize = (transA ? m : k) * lda * sizeof(T);
    auto Bsize = (transB ? k : n) * ldb * sizeof(T);
    auto Csize = n * ldc * sizeof(T);
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&a),
                       Asize));
    CUDA_CALL(cudaMemset(a, 0xAB, Asize));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&b),
                       Bsize));
    CUDA_CALL(cudaMemset(b, 0xBC, Bsize));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&c),
                       Csize));
    CUDA_CALL(cudaMemset(c, 0xCD, Csize));

    cudaEvent_t start, end;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&end));

    CUDA_CALL(cudaEventRecord(start, nullptr));
    gemm(handle, m, n, k, transA, transB, alpha, beta, a, lda, b, ldb, c, ldc);
    CUDA_CALL(cudaEventRecord(end, nullptr));
    CUDA_CALL(cudaEventSynchronize(end));
    float ms;
    CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
    double tflops = calc_tflops(m, n, k, ms);

    CUDA_CALL(cudaEventDestroy(end));
    CUDA_CALL(cudaEventDestroy(start));
    CUBLAS_CALL(cublasDestroy(handle));

    printf("%s,%d,%d,%d,%d,%d,%f,%f,%d,%d,%d,%f,%f\n", type_to_str<T>(), m, n, k, transA, transB, float(alpha), float(beta), lda, ldb, ldc, ms, tflops);
}

void print_help(const char* progName)
{
    printf("usage: %s typename([hsd]) m n k transA transB alpha beta lda ldb ldc\n", progName);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        print_help(argv[0]);
        return 0;
    }

    std::ptrdiff_t iarg = 0;

    if (argc < 12)
    {
        print_help(argv[0]);
        return 0;
    }

    char* type = argv[++iarg];

    int m = atoi(argv[++iarg]);
    int n = atoi(argv[++iarg]);
    int k = atoi(argv[++iarg]);

    int transA = atoi(argv[++iarg]);
    int transB = atoi(argv[++iarg]);

    float alpha = atof(argv[++iarg]);
    float beta = atof(argv[++iarg]);

    int lda = atoi(argv[++iarg]);
    int ldb = atoi(argv[++iarg]);
    int ldc = atoi(argv[++iarg]);

    int gpu_id = atoi(argv[++iarg]);

    ASSERT_NON_ZERO(m);
    ASSERT_NON_ZERO(n);
    ASSERT_NON_ZERO(k);

    ASSERT_NON_ZERO(lda);
    ASSERT_NON_ZERO(ldb);
    ASSERT_NON_ZERO(ldc);

    switch (type[0])
    {
    case 'H':
    case 'h':
        benchmark<__half>(m, n, k, transA, transB, __half(alpha), __half(beta), lda, ldb, ldc, gpu_id);
        break;
    case 'S':
    case 's':
        benchmark<float>(m, n, k, transA, transB, alpha, beta, lda, ldb, ldc, gpu_id);
        break;
    case 'D':
    case 'd':
        benchmark<double>(m, n, k, transA, transB, alpha, beta, lda, ldb, ldc, gpu_id);
        break;
    default:
        printf("invaild typename: %s\n", type);
        return -1;
    }

    return 0;
}
