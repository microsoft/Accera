#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#define __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <rocblas.h>

#define HIP_CALL(E)                                                             \
    do                                                                          \
    {                                                                           \
        hipError_t e = (E);                                                     \
        if (e != hipSuccess)                                                    \
        {                                                                       \
            printf("line %d: HIP error: %s\n", __LINE__, hipGetErrorString(e)); \
            exit(-2);                                                           \
        }                                                                       \
    } while (false)

#define ROCBLAS_CALL(S)                                          \
    do                                                           \
    {                                                            \
        rocblas_status s = (S);                                  \
        if (s != rocblas_status_success)                         \
        {                                                        \
            printf("line %d: rocBLAS error: %d\n", __LINE__, s); \
            exit(-3);                                            \
        }                                                        \
    } while (false)

#define ASSERT_NON_ZERO(X)                              \
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

rocblas_operation trans(int t) {
    return t ? rocblas_operation_transpose : rocblas_operation_none;
}

template <typename T>
void gemm(rocblas_handle handle, int m, int n, int k, int transA, int transB, double alpha, double beta, const T* a, int lda, const T* b, int ldb, T* c, int ldc);

template <>
void gemm<float>(rocblas_handle handle, int m, int n, int k, int transA, int transB, double alpha, double beta, const float* a, int lda, const float* b, int ldb, float* c, int ldc)
{
    auto alpha_ = (float)alpha;
    auto beta_ = (float)beta;
    ROCBLAS_CALL(rocblas_sgemm(handle, trans(transA), trans(transB), m, n, k, &alpha_, a, lda, b, ldb, &beta_, c, ldc));
}

template <>
void gemm<double>(rocblas_handle handle, int m, int n, int k, int transA, int transB, double alpha, double beta, const double* a, int lda, const double* b, int ldb, double* c, int ldc)
{
    ROCBLAS_CALL(rocblas_dgemm(handle, trans(transA), trans(transB), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

auto fp32_to_fp16(float f)
{
    uint32_t x = *((uint32_t*)&f);
    uint16_t h = ((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
    return rocblas_half{h};
}

template <>
void gemm<rocblas_half>(rocblas_handle handle, int m, int n, int k, int transA, int transB, double alpha, double beta, const rocblas_half* a, int lda, const rocblas_half* b, int ldb, rocblas_half* c, int ldc)
{
    auto alpha_ = fp32_to_fp16(alpha);
    auto beta_ = fp32_to_fp16(beta);
    ROCBLAS_CALL(rocblas_hgemm(handle, trans(transA), trans(transB), m, n, k, &alpha_, a, lda, b, ldb, &beta_, c, ldc));
}

double calc_gflops(int m, int n, int k, float ms)
{
    auto flopms = (2.0 * k + 2) * m * n / ms;
    auto flops = flopms * 1e3;
    auto gflops = flops * 1e-9;
    return gflops;
}

template <typename T>
const char* type_to_str() {
    if constexpr (std::is_same<T, float>()) {
        return "s";
    }

    if constexpr (std::is_same<T, double>()) {
        return "d";
    }

    if constexpr (std::is_same<T, rocblas_half>()) {
        return "h";
    }
}


template <typename T>
void benchmark(int m, int n, int k, int transA, int transB, double alpha, double beta, int lda, int ldb, int ldc)
{
    rocblas_handle handle;

    ROCBLAS_CALL(rocblas_create_handle(&handle));

    T *a, *b, *c;
    auto Asize = (transA ? m : k) * lda * sizeof(T);
    auto Bsize = (transB ? k : n) * ldb * sizeof(T);
    auto Csize = n * ldc * sizeof(T);
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&a),
                       Asize));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&b),
                       Bsize));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&c),
                       Csize));
    HIP_CALL(hipMemset(a, 0xAB, Asize));
    HIP_CALL(hipMemset(b, 0xBC, Bsize));
    HIP_CALL(hipMemset(c, 0xCD, Csize));

    // warmup the library
    gemm(handle, m, n, k, transA, transB, alpha, beta, a, lda, b, ldb, c, ldc);

    HIP_CALL(hipMemset(a, 0xBC, Asize));
    HIP_CALL(hipMemset(b, 0xAB, Bsize));
    HIP_CALL(hipMemset(c, 0xDE, Csize));

    hipEvent_t start, end;
    float ms{};

    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&end));

    HIP_CALL(hipEventRecord(start, nullptr));
    HIP_CALL(hipEventSynchronize(start));
    gemm(handle, m, n, k, transA, transB, alpha, beta, a, lda, b, ldb, c, ldc);
    HIP_CALL(hipEventRecord(end, nullptr));
    HIP_CALL(hipEventSynchronize(end));
    HIP_CALL(hipEventElapsedTime(&ms, start, end));
    double gflops = calc_gflops(m, n, k, ms);

    HIP_CALL(hipEventDestroy(end));
    HIP_CALL(hipEventDestroy(start));
    ROCBLAS_CALL(rocblas_destroy_handle(handle));

    printf("%s,%d,%d,%d,%d,%d,%f,%f,%d,%d,%d,%f,%f\n", type_to_str<T>(), m, n, k, transA, transB, alpha, beta, lda, ldb, ldc, ms, gflops);
}

void print_help(const char* progName)
{
    printf("usage: %s [--headers] typename([hsd]) m n k transA transB alpha beta lda ldb ldc\n", progName);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        print_help(argv[0]);
        return 0;
    }

    ptrdiff_t iarg = 0;

    char* typeOrHeaders = argv[++iarg];
    if (strcmp(typeOrHeaders, "--headers") == 0)
    {
        printf("type,m,n,k,transA,transB,alpha,beta,lda,ldb,ldc,time_ms,gflops\n");
        return 0;
    }

    if (argc < 12)
    {
        print_help(argv[0]);
        return 0;
    }

    char* type = typeOrHeaders;

    int m = atoi(argv[++iarg]);
    int n = atoi(argv[++iarg]);
    int k = atoi(argv[++iarg]);

    int transA = atoi(argv[++iarg]);
    int transB = atoi(argv[++iarg]);

    double alpha = atof(argv[++iarg]);
    double beta = atof(argv[++iarg]);

    int lda = atoi(argv[++iarg]);
    int ldb = atoi(argv[++iarg]);
    int ldc = atoi(argv[++iarg]);

    ASSERT_NON_ZERO(m);
    ASSERT_NON_ZERO(n);
    ASSERT_NON_ZERO(k);

    ASSERT_NON_ZERO(lda);
    ASSERT_NON_ZERO(ldb);
    ASSERT_NON_ZERO(ldc);

    // rocblas_initialize();

    switch (type[0])
    {
    case 'H':
    case 'h':
        benchmark<rocblas_half>(m, n, k, transA, transB, alpha, beta, lda, ldb, ldc);
        break;
    case 'S':
    case 's':
        benchmark<float>(m, n, k, transA, transB, alpha, beta, lda, ldb, ldc);
        break;
    case 'D':
    case 'd':
        benchmark<double>(m, n, k, transA, transB, alpha, beta, lda, ldb, ldc);
        break;
    default:
        printf("invaild typename: %s\n", type);
        return -1;
    }

    return 0;
}
