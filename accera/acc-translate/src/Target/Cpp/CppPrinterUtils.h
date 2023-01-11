////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Captain Jack Sparrow
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CPP_PRINTER_UTILS_H_
#define CPP_PRINTER_UTILS_H_

#include "CppPrinter.h"
#include <ir/include/value/ValueMMAOp.h>

namespace vir = accera::ir::value;

namespace mlir
{
namespace cpp_printer
{
    bool isPrivateOrWorkgroupMemSpace(unsigned memspace);

    // Return the round-up number of bits that are valid for integer types, e.g.
    // 8, 16, 32, and 64
    int getIntTypeBitCount(int width);

    LogicalResult printMMAMatrixOp(PrinterState& state, CppPrinter* printer, Type elementType, const std::tuple<int, int, int>& matrixShape, Value dest, vir::MMAOperandType operandType, int totalBlocks, int blocks, bool rowMajor);
    LogicalResult printConstantMatrixOp(PrinterState& state, CppPrinter* printer, Value dest, Value value);
    LogicalResult printLoadMatrixOp(PrinterState& state, CppPrinter* printer, Value src, Value dest, vir::MMAOperandType operandType, mlir::Operation::operand_range indices, bool rowMajor, Value blockTid = {}, bool useStaticOffsets = {}, vir::MMAFragmentOp mmaPrologueOp = vir::MMAFragmentOp::None, Value mmaPrologueArg = {});
    LogicalResult printComputeMatrixOp(PrinterState& state, CppPrinter* printer, Value A, Value B, Value C, Value D, int cbsz = 0, int abid = 0, int blgp = 0);
    LogicalResult printStoreMatrixOp(PrinterState& state, CppPrinter* printer, Value src, Value dest, mlir::Operation::operand_range indices, Value blockTid = {}, bool useStaticOffsets = {}, vir::MMAFragmentOp mmaEpilogueOp = vir::MMAFragmentOp::None, Value mmaEpilogueArg = {});

    constexpr auto HipIncludesAndTypes = R"ROCM(
#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#endif

using int8_t = unsigned char;
using int16_t = short;
using uint8_t = unsigned char;
using uint16_t = unsigned short;
namespace std {
    using ::uint8_t;
    using ::uint16_t;
    using ::int8_t;
    using ::int16_t;
}

using float16_t = _Float16;
using bfloat16_t = uint16_t;
)ROCM";

    constexpr auto CudaIncludesAndTypes = R"CUDA(
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

using float16_t = __half;
using bfloat16_t = __nv_bfloat16;
using uint32_t = unsigned int;
using int32_t = int;
using uint8_t = unsigned char;
using int8_t = signed char;
)CUDA";

    constexpr auto ForceinlineMacro = R"ROCM(
#ifndef __forceinline__
#define __forceinline__ __inline__ __attribute__((always_inline))
#endif // __forceinline__
)ROCM";

    constexpr auto VectorTypes = R"ROCM(
using vhalfx2_t = float16_t __attribute__((ext_vector_type(2)));
using vhalfx4_t = float16_t __attribute__((ext_vector_type(4)));
using vhalfx8_t = float16_t __attribute__((ext_vector_type(8)));
using vhalfx16_t = float16_t __attribute__((ext_vector_type(16)));
using vhalfx32_t = float16_t __attribute__((ext_vector_type(32)));
using vhalfx64_t = float16_t __attribute__((ext_vector_type(64)));
using vbfloat16x2_t = bfloat16_t __attribute__((ext_vector_type(2)));
using vbfloat16x4_t = bfloat16_t __attribute__((ext_vector_type(4)));
using vbfloat16x8_t = bfloat16_t __attribute__((ext_vector_type(8)));
using vbfloat16x16_t = bfloat16_t __attribute__((ext_vector_type(16)));
using vbfloat16x32_t = bfloat16_t __attribute__((ext_vector_type(32)));
using vbfloat16x64_t = bfloat16_t __attribute__((ext_vector_type(64)));
using vfloatx2_t = float __attribute__((ext_vector_type(2)));
using vfloatx3_t = float __attribute__((ext_vector_type(3)));
using vfloatx4_t = float __attribute__((ext_vector_type(4)));
using vfloatx8_t = float __attribute__((ext_vector_type(8)));
using vfloatx16_t = float __attribute__((ext_vector_type(16)));
using vfloatx32_t = float __attribute__((ext_vector_type(32)));
using vfloatx64_t = float __attribute__((ext_vector_type(64)));
using vint32x4_t = int __attribute__((ext_vector_type(4)));
using vint32x16_t = int __attribute__((ext_vector_type(16)));
using vint32x32_t = int __attribute__((ext_vector_type(32)));
    )ROCM";

    constexpr auto BFloatCast = R"ROCM(
__device__ __forceinline__ float cast(const bfloat16_t val)
{
    // Copied from /opt/rocm/include/hip/hip_bfloat16.h
    union
    {
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(val) << 16};
    return u.fp32;
}

__device__ __forceinline__ bfloat16_t cast(const float f)
{
    // Copied from /opt/rocm/include/hip/hip_bfloat16.h
    // This does trucation instead of proper rounding (which is slower)
    union
    {
        float    fp32;
        uint32_t int32;
    } u = {f};
    return uint16_t(u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff));
}
    )ROCM";

    constexpr auto RocWmma = R"ROCM(
namespace rocwmma {
    enum fragment_type { matrix_a, matrix_b, accumulator };
    enum class layout_t { mem_row_major, mem_col_major };
    enum layout2_t { row_major, col_major };

    constexpr auto WARP_SIZE = 64;

    struct Idx {
        unsigned int m;
        unsigned int k;
    };

    template <int M>
    __device__ __forceinline__ Idx getThreadIdx(const unsigned int block_tid)
    {
        const auto warp_tid = block_tid % WARP_SIZE;
        return Idx {warp_tid % M, warp_tid / M};
    }

    template<typename _Ty, int B, int S>
    class frag_base
    {
        using _VTy = _Ty __attribute__((ext_vector_type(S)));
        _VTy data[B];

    public:
        __device__ __forceinline__ _VTy& operator()(const int b) { return data[b]; }
        __device__ __forceinline__ _VTy operator()(const int b) const { return data[b]; }
        __device__ __forceinline__ _Ty& operator[](const int i) { return ((_Ty*)data)[i]; }
        __device__ __forceinline__ _Ty operator[](const int i) const { return ((_Ty*)data)[i]; }
    };

    // Primary template
    template<fragment_type fragType, int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS, typename _Ty, layout2_t layout = row_major>
    struct fragment : public frag_base<_Ty, BLOCKS, M * K / WARP_SIZE> {
        static_assert(M == N, "M must be equal to N");
        static_assert(BLOCKS == 1, "Only accumulator can have BLOCKS > 1");
    };

    // Accumulator specialization
    template<int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS, typename _Ty>
    struct fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, _Ty> : public frag_base<_Ty, BLOCKS, M * N / WARP_SIZE / TOTAL_BLOCKS> {};

    template<fragment_type fragType, layout2_t layout, unsigned int LD>
    __device__ __forceinline__ int getIdx(const int row, const int col)
    {
        if constexpr (((fragType == matrix_a || fragType == accumulator) && layout == row_major) || (fragType == matrix_b && layout == col_major))
            return row * LD + col;

        return col * LD + row;
    }

    __constant__ uint8_t threadGroupOffsets_32[16][2] = {
        {0, 4},   {1, 5},   {2, 6},   {3, 7},   {8, 12},  {9, 13},  {10, 14}, {11, 15},
        {16, 20}, {17, 21}, {18, 22}, {19, 23}, {24, 28}, {25, 29}, {26, 30}, {27, 31}};

    __constant__ uint8_t threadGroupOffsets_16[4][4] = {
        {0, 4, 8, 12}, {1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}};

    template<int M> __device__ __forceinline__ int getOffset(const int r, const int c);
    template<> __device__ __forceinline__ int getOffset<32>(const int r, const int c) { return threadGroupOffsets_32[r][c]; }
    template<> __device__ __forceinline__ int getOffset<16>(const int r, const int c) { return threadGroupOffsets_16[r][c]; }

    template<int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS, typename _Ty>
    __device__ __forceinline__ void fill_fragment(fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, _Ty>& dest, const _Ty val)
    {
        constexpr auto elems = sizeof(dest) / sizeof(_Ty);
        _Ty* dest_data = reinterpret_cast<_Ty*>(&dest);
        for (int i = 0; i < elems; ++i)
        {
            dest_data[i] = val;
        }
    }

    template<bool STATIC_OFFSETS, int M, int N, int TOTAL_BLOCKS, int BLOCKS, int FragSize, layout_t layout, unsigned int LD, typename _Ty, typename Action>
    __device__ __forceinline__ void load_store_acc(const unsigned int block_tid, Action&& do_fn)
    {
        constexpr auto SubGroupSize = 4;
        constexpr auto ValsPerThreadPerBlock = FragSize / BLOCKS / sizeof(_Ty);
        constexpr auto MFMA_M = M / TOTAL_BLOCKS;
        constexpr auto Layout2 = layout == layout_t::mem_row_major ? row_major : col_major;
        const auto tid = getThreadIdx<MFMA_M>(block_tid);
        const auto threadGroupOffset = tid.k * SubGroupSize;

        for (int b = 0, blockRowOffset = 0; b < BLOCKS; ++b, blockRowOffset += MFMA_M)
        {
            if constexpr (STATIC_OFFSETS)
            {
                for (int i = 0; i < ValsPerThreadPerBlock; ++i)
                {
                    int row = blockRowOffset;
                    int col = tid.m;
                    if constexpr (TOTAL_BLOCKS == 1)
                    {
                        row += getOffset<MFMA_M>(i, tid.k);
                    }
                    else
                    {
                        constexpr auto offsetWidth = MFMA_M / TOTAL_BLOCKS;
                        row += getOffset<MFMA_M>(i % offsetWidth, tid.k);
                        col += (i / offsetWidth) * MFMA_M;
                    }
                    const auto idx = getIdx<accumulator, Layout2, LD>(row, col);
                    do_fn(idx, b, i);
                }
            }
            else
            {
                constexpr auto ValsPerThread = ValsPerThreadPerBlock * TOTAL_BLOCKS;
                constexpr auto ThreadsPerBlock = M * N / ValsPerThread;
                constexpr auto RowsPerSet = ThreadsPerBlock / MFMA_M * SubGroupSize;
                constexpr auto GroupsPerCol = MFMA_M / RowsPerSet;
                for (int itemGroup = 0, i = 0; itemGroup < ValsPerThreadPerBlock / SubGroupSize; ++itemGroup)
                {
                    const auto col = tid.m + (itemGroup / GroupsPerCol) * MFMA_M;
                    const auto itemGroupOffset = (itemGroup % GroupsPerCol) * RowsPerSet;
                    const auto rowOffset = threadGroupOffset + blockRowOffset + itemGroupOffset;
                    for (int itemOffset = 0, row = rowOffset; itemOffset < SubGroupSize; ++itemOffset, ++row)
                    {
                        const auto idx = getIdx<accumulator, Layout2, LD>(row, col);
                        do_fn(idx, b, i++);
                    }
                }
            }
        }
    }

    template <unsigned int LD, fragment_type fragType, int M, int N, int K, int TOTAL_BLOCKS, typename _Ty, layout2_t layout>
    __device__ __forceinline__ void load_matrix_sync(const unsigned int block_tid, fragment<fragType, M, N, K, TOTAL_BLOCKS, 1, _Ty, layout>& dest, const _Ty* __restrict__ src)
    {
        static_assert(fragType != accumulator, "this is only for matrix_a and matrix_b");

        constexpr auto Stride = WARP_SIZE / M;
        constexpr auto ValsPerThreadPerBlock = sizeof(dest) / sizeof(_Ty);
        const auto idx = getThreadIdx<M>(block_tid);
        for (int i = 0; i < ValsPerThreadPerBlock; ++i)
        {
            dest(0)[i] = src[getIdx<fragType, layout, LD>(idx.m, idx.k + i * Stride)];
        }
    }

    template <unsigned int LD, fragment_type fragType, int M, int N, int K, int TOTAL_BLOCKS, layout2_t layout>
    __device__ __forceinline__ void load_matrix_sync(const unsigned int block_tid, fragment<fragType, M, N, K, TOTAL_BLOCKS, 1, bfloat16_t, layout>& dest, const bfloat16_t* __restrict__ src)
    {
        static_assert(fragType != accumulator, "this is only for matrix_a and matrix_b");

        const auto idx = getThreadIdx<M>(block_tid);
        for (int i = 0; i < 2; ++i)
        {
            dest(0)[i] = src[getIdx<fragType, layout, LD>(idx.m, 2 * idx.k + i)];
        }
    }

    template <bool STATIC_OFFSETS, layout_t layout, unsigned int LD, int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS, typename _Ty, typename _Uy>
    __device__ __forceinline__ void load_matrix_sync(const unsigned int block_tid, fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, _Ty>& dest, const _Uy* __restrict__ src)
    {
        load_store_acc<STATIC_OFFSETS, M, N, TOTAL_BLOCKS, BLOCKS, sizeof(dest), layout, LD, _Ty>(block_tid, [&](const int idx, const int b, const int i) { dest(b)[i] = src[idx]; });
    }

    template <bool STATIC_OFFSETS, layout_t layout, unsigned int LD, int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS>
    __device__ __forceinline__ void load_matrix_sync(const unsigned int block_tid, fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, float>& dest, const bfloat16_t* __restrict__ src)
    {
        load_store_acc<STATIC_OFFSETS, M, N, TOTAL_BLOCKS, BLOCKS, sizeof(dest), layout, LD, float>(block_tid, [&](const int idx, const int b, const int i) { dest(b)[i] = cast(src[idx]); });
    }

    template <bool STATIC_OFFSETS, layout_t layout, unsigned int LD, int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS, typename _Ty, typename _Uy>
    __device__ __forceinline__ void store_matrix_sync(const unsigned int block_tid, _Uy* __restrict__ dest, const fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, _Ty>& src)
    {
        load_store_acc<STATIC_OFFSETS, M, N, TOTAL_BLOCKS, BLOCKS, sizeof(src), layout, LD, _Ty>(block_tid, [&](const int idx, const int b, const int i) { dest[idx] = src(b)[i]; });
    }

    template <bool STATIC_OFFSETS, layout_t layout, unsigned int LD, int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS>
    __device__ __forceinline__ void store_matrix_sync(const unsigned int block_tid, bfloat16_t* __restrict__ dest, const fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, float>& src)
    {
        load_store_acc<STATIC_OFFSETS, M, N, TOTAL_BLOCKS, BLOCKS, sizeof(src), layout, LD, float>(block_tid, [&](const int idx, const int b, const int i) { dest[idx] = cast(src(b)[i]); });
    }

    template <int CBSZ, int ABID, int BLGP, int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS, layout2_t layoutA, layout2_t layoutB>
    __device__ __forceinline__ void mma_sync(fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, float>& D, const fragment<matrix_a, M, N, K, TOTAL_BLOCKS, 1, float, layoutA>& A, const fragment<matrix_b, M, N, K, TOTAL_BLOCKS, 1, float, layoutB>& B, const fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, float>& C)
    {
        static_assert(sizeof(A(0)) / sizeof(float) == 1, "Input matrices should have 1 float per fragment");

        constexpr auto BlockId = BLOCKS == 1 ? 0 : ABID;
        if constexpr (TOTAL_BLOCKS == 4)
            D(BlockId) = __builtin_amdgcn_mfma_f32_16x16x1f32(A(0)[0], B(0)[0], C(BlockId), CBSZ, ABID, BLGP);
        else if constexpr (TOTAL_BLOCKS == 2)
            D(BlockId) = __builtin_amdgcn_mfma_f32_32x32x1f32(A(0)[0], B(0)[0], C(BlockId), CBSZ, ABID, BLGP);
        else if constexpr (M == 32)
            D(BlockId) = __builtin_amdgcn_mfma_f32_32x32x2f32(A(0)[0], B(0)[0], C(BlockId), CBSZ, ABID, BLGP);
        else if constexpr (M == 16)
            D(BlockId) = __builtin_amdgcn_mfma_f32_16x16x4f32(A(0)[0], B(0)[0], C(BlockId), CBSZ, ABID, BLGP);
    }

    template <int CBSZ, int ABID, int BLGP, int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS, layout2_t layoutA, layout2_t layoutB>
    __device__ __forceinline__ void mma_sync(fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, float>& D, const fragment<matrix_a, M, N, K, TOTAL_BLOCKS, 1, float16_t, layoutA>& A, const fragment<matrix_b, M, N, K, TOTAL_BLOCKS, 1, float16_t, layoutB>& B, const fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, float>& C)
    {
        constexpr auto BlockId = BLOCKS == 1 ? 0 : ABID;
        if constexpr (TOTAL_BLOCKS == 4)
            D(BlockId) = __builtin_amdgcn_mfma_f32_16x16x4f16(A(0), B(0), C(BlockId), CBSZ, ABID, BLGP);
        else if constexpr (TOTAL_BLOCKS == 2)
            D(BlockId) = __builtin_amdgcn_mfma_f32_32x32x4f16(A(0), B(0), C(BlockId), CBSZ, ABID, BLGP);
        else if constexpr (M == 32)
            D(BlockId) = __builtin_amdgcn_mfma_f32_32x32x8f16(A(0), B(0), C(BlockId), CBSZ, ABID, BLGP);
        else if constexpr (M == 16)
            D(BlockId) = __builtin_amdgcn_mfma_f32_16x16x16f16(A(0), B(0), C(BlockId), CBSZ, ABID, BLGP);
    }

    template <int CBSZ, int ABID, int BLGP, int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS, layout2_t layoutA, layout2_t layoutB>
    __device__ __forceinline__ void mma_sync(fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, float>& D, const fragment<matrix_a, M, N, K, TOTAL_BLOCKS, 1, bfloat16_t, layoutA>& A, const fragment<matrix_b, M, N, K, TOTAL_BLOCKS, 1, bfloat16_t, layoutB>& B, const fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, float>& C)
    {
        constexpr auto BlockId = BLOCKS == 1 ? 0 : ABID;
        if constexpr (TOTAL_BLOCKS == 4)
            D(BlockId) = __builtin_amdgcn_mfma_f32_16x16x2bf16(A(0), B(0), C(BlockId), CBSZ, ABID, BLGP);
        else if constexpr (TOTAL_BLOCKS == 2)
            D(BlockId) = __builtin_amdgcn_mfma_f32_32x32x2bf16(A(0), B(0), C(BlockId), CBSZ, ABID, BLGP);
        else if constexpr (M == 32)
            D(BlockId) = __builtin_amdgcn_mfma_f32_32x32x4bf16(A(0), B(0), C(BlockId), CBSZ, ABID, BLGP);
        else if constexpr (M == 16)
            D(BlockId) = __builtin_amdgcn_mfma_f32_16x16x8bf16(A(0), B(0), C(BlockId), CBSZ, ABID, BLGP);
    }

    template <int CBSZ, int ABID, int BLGP, int M, int N, int K, int TOTAL_BLOCKS, int BLOCKS, layout2_t layoutA, layout2_t layoutB>
    __device__ __forceinline__ void mma_sync(fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, int>& D, const fragment<matrix_a, M, N, K, TOTAL_BLOCKS, 1, int8_t, layoutA>& A, const fragment<matrix_b, M, N, K, TOTAL_BLOCKS, 1, int8_t, layoutB>& B, const fragment<accumulator, M, N, K, TOTAL_BLOCKS, BLOCKS, int>& C)
    {
        static_assert(sizeof(A(0)) / sizeof(int8_t) == 4, "Input matrices should have 4 ints per fragment");

        constexpr auto BlockId = BLOCKS == 1 ? 0 : ABID;
        D(BlockId) = C(BlockId);
        for (int i = 0; i < 4; ++i)
        {
            if constexpr (TOTAL_BLOCKS == 4)
                D(BlockId) = __builtin_amdgcn_mfma_i32_16x16x4i8(A(0)[i], B(0)[i], D(BlockId), CBSZ, ABID, BLGP);
            else if constexpr (TOTAL_BLOCKS == 2)
                D(BlockId) = __builtin_amdgcn_mfma_i32_32x32x4i8(A(0)[i], B(0)[i], D(BlockId), CBSZ, ABID, BLGP);
            else if constexpr (M == 32)
                D(BlockId) = __builtin_amdgcn_mfma_i32_32x32x8i8(A(0)[i], B(0)[i], D(BlockId), CBSZ, ABID, BLGP);
            else if constexpr (M == 16)
                D(BlockId) = __builtin_amdgcn_mfma_i32_16x16x16i8(A(0)[i], B(0)[i], D(BlockId), CBSZ, ABID, BLGP);
        }
    }
} // namespace rocwmma
)ROCM";

    constexpr auto BlockCacheCopy = R"CUDA(
enum class CopyMode
{
    Blocked,    // thread0 copies a contiguous chunk followed by thread1, etc.
    Striped     // thread0 copies element 0, thread1 copies element 1, etc.
};

enum class MemSpace
{
    None = 0,
    Shared = 3,
    Private = 5
};

template<bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, bool FORWARD, int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper{};

template<int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper<true, true, true, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(_Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, const int idx, int, int, AccessMap&& srcAccessMap)
    {
        const auto srcOffset = srcAccessMap(src_r, src_c);
        *reinterpret_cast<_vTy*>(&dst[idx]) = *reinterpret_cast<_vTy*>(&src[srcOffset]);
    }
};

template<int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper<true, true, false, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(_Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, const int idx, int, int, AccessMap&& srcAccessMap)
    {
        const auto srcOffset = srcAccessMap(src_r, src_c);
        *reinterpret_cast<_vTy*>(&src[srcOffset]) = *reinterpret_cast<_vTy*>(&dst[idx]);
    }
};

template<int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper<false, true, true, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(_Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, const int idx, int, int, AccessMap&& srcAccessMap)
    {
        _vTy val;
        const auto pVal = reinterpret_cast<_Ty*>(&val);
        for (int i = 0; i < STRIDE; ++i)
        {
            const auto srcOffset = srcAccessMap(src_r, src_c + i);
            pVal[i] = src[srcOffset];
        }
        *reinterpret_cast<_vTy*>(&dst[idx]) = val;
    }
};

template<int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper<false, true, false, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(_Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, const int idx, int, int, AccessMap&& srcAccessMap)
    {
        _vTy val = *reinterpret_cast<_vTy*>(&dst[idx]);
        const auto pVal = reinterpret_cast<_Ty*>(&val);
        for (int i = 0; i < STRIDE; ++i)
        {
            const auto srcOffset = srcAccessMap(src_r, src_c + i);
            src[srcOffset] = pVal[i];
        }
    }
};

template<int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper<true, false, true, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(_Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, int, const int dst_r, const int dst_c, AccessMap&& srcAccessMap)
    {
        const auto srcOffset = srcAccessMap(src_r, src_c);
        const auto val = *reinterpret_cast<const _vTy*>(&src[srcOffset]);
        const auto pVal = reinterpret_cast<const _Ty*>(&val);
        for (int i = 0, p = dst_c * TILE_R + dst_r; i < STRIDE; ++i, p += TILE_R)
        {
            dst[p] = pVal[i];
        }
    }
};

template<int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper<true, false, false, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(_Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, int, const int dst_r, const int dst_c, AccessMap&& srcAccessMap)
    {
        const auto srcOffset = srcAccessMap(src_r, src_c);
        _vTy val;
        const auto pVal = reinterpret_cast<_Ty*>(&val);
        for (int i = 0, p = dst_c * TILE_R + dst_r; i < STRIDE; ++i, p += TILE_R)
        {
            pVal[i] = dst[p];
        }
        *reinterpret_cast<_vTy*>(&src[srcOffset]) = val;
    }
};

template<int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper<false, false, true, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(_Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, int, const int dst_r, const int dst_c, AccessMap&& srcAccessMap)
    {
        for (int i = 0, p = dst_c * TILE_R + dst_r; i < STRIDE; ++i, p += TILE_R)
        {
            const auto srcOffset = srcAccessMap(src_r, src_c + i);
            dst[p] = src[srcOffset];
        }
    }
};

template<int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper<false, false, false, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(_Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, int, const int dst_r, const int dst_c, AccessMap&& srcAccessMap)
    {
        for (int i = 0, p = dst_c * TILE_R + dst_r; i < STRIDE; ++i, p += TILE_R)
        {
            const auto srcOffset = srcAccessMap(src_r, src_c + i);
            src[srcOffset] = dst[p];
        }
    }
};

template<CopyMode MODE, int BLOCK_SIZE, int WPT, int STRIDE, int TILE_C>
struct CopyModeHelper {};

template<int BLOCK_SIZE, int WPT, int STRIDE, int TILE_C>
struct CopyModeHelper<CopyMode::Blocked, BLOCK_SIZE, WPT, STRIDE, TILE_C>
{
    // thread0 copies a contiguous chunk followed by thread1, etc.
    // +-------------------------------------------------------------------+
    // |       t0       |       t1       |       t2       |       t3       |
    // +-------------------------------------------------------------------+
    // |       t4       |                                                  |
    // +-------------------------------------------------------------------+
    // |                                                                   |
    // +-------------------------------------------------------------------+
    // |                                                  |       t15      |
    // +-------------------------------------------------------------------+

    template <typename StrideCopyHelper>
    static __device__ __forceinline__ void copy(const int blockTid, StrideCopyHelper&& copyHelper)
    {
        const auto start = blockTid * WPT;
        if constexpr (TILE_C % WPT == 0)            // all elements are on the same row
        {
            const auto r = start / TILE_C;
            for (int i = 0, c = start % TILE_C, p = start; i < WPT; i += STRIDE, c += STRIDE, p += STRIDE)
            {
                copyHelper(i, p, r, c);
            }
        }
        else
        {
            for (int i = 0, p = start; i < WPT; i += STRIDE, p += STRIDE)
            {
                const auto r = p / TILE_C;
                const auto c = p % TILE_C;
                copyHelper(i, p, r, c);
            }
        }
    }
};

template<int BLOCK_SIZE, int WPT, int STRIDE, int TILE_C>
struct CopyModeHelper<CopyMode::Striped, BLOCK_SIZE, WPT, STRIDE, TILE_C>
{
    // thread0 copies element 0, thread1 copies element 1, etc.
    // +-----------------------------------------------------------------------+
    // |   t0   |   t1   |   t2   |   t3   |   t4   |   t5   |   t6   |   t7   |
    // +-----------------------------------------------------------------------+
    // |   t8   |                                                     |   t15  |
    // +-----------------------------------------------------------------------+
    // |   t0   |   t1   |                                                     |
    // +-----------------------------------------------------------------------+
    // |                                                              |   t15  |
    // +-----------------------------------------------------------------------+

    template <typename StrideCopyHelper>
    static __device__ __forceinline__ void copy(const int blockTid, StrideCopyHelper&& copyHelper)
    {
        const auto start = blockTid * STRIDE;
        for (int i = 0, p = start; i < WPT; i += STRIDE, p += STRIDE * BLOCK_SIZE)
        {
            const auto r = p / TILE_C;
            const auto c = p % TILE_C;
            copyHelper(i, p, r, c);
        }
    }
};

template<MemSpace SRC_MEMSPACE, MemSpace DST_MEMSPACE, CopyMode MODE, int BLOCK_SIZE, bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, int STRIDE, int WPT, int TILE_R, int TILE_C, typename _vTy, typename _Ty>
struct ThreadCopyHelper {};

template<CopyMode MODE, int BLOCK_SIZE, bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, int STRIDE, int WPT, int TILE_R, int TILE_C, typename _vTy, typename _Ty>
struct ThreadCopyHelper<MemSpace::Private, MemSpace::Shared, MODE, BLOCK_SIZE, SRC_ROW_MAJOR, DST_ROW_MAJOR, STRIDE, WPT, TILE_R, TILE_C, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(const int blockTid, _Ty* __restrict__ src, int, int, _Ty* __restrict__ dst, AccessMap&&)
    {
        CopyModeHelper<MODE, BLOCK_SIZE, WPT, STRIDE, TILE_C>::copy(blockTid, [=](const int i, const int p, const int r, const int c){
            StrideCopyHelper<SRC_ROW_MAJOR, /*DST_ROW_MAJOR*/ true, /*FORWARD*/ false, STRIDE, TILE_R, _vTy, _Ty>::copy(src, r, c, dst, i, -1, -1, [=](int y, int x){ return SRC_ROW_MAJOR ? p : x * TILE_R + y; });
        });
    }
};

template<CopyMode MODE, int BLOCK_SIZE, bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, int STRIDE, int WPT, int TILE_R, int TILE_C, typename _vTy, typename _Ty>
struct ThreadCopyHelper<MemSpace::None, MemSpace::Private, MODE, BLOCK_SIZE, SRC_ROW_MAJOR, DST_ROW_MAJOR, STRIDE, WPT, TILE_R, TILE_C, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(const int blockTid, _Ty* __restrict__ src, const int srcOffsetRows, const int srcOffsetCols, _Ty* __restrict__ dst, AccessMap&& srcAccessMap)
    {
        CopyModeHelper<MODE, BLOCK_SIZE, WPT, STRIDE, TILE_C>::copy(blockTid, [=](const int i, const int p, const int r, const int c){
            StrideCopyHelper<SRC_ROW_MAJOR, /*DST_ROW_MAJOR*/ true, /*FORWARD*/ true, STRIDE, TILE_R, _vTy, _Ty>::copy(src, srcOffsetRows + r, srcOffsetCols + c, dst, i, -1, -1, srcAccessMap);
        });
    }
};

template<CopyMode MODE, int BLOCK_SIZE, bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, int STRIDE, int WPT, int TILE_R, int TILE_C, typename _vTy, typename _Ty>
struct ThreadCopyHelper<MemSpace::Private, MemSpace::None, MODE, BLOCK_SIZE, SRC_ROW_MAJOR, DST_ROW_MAJOR, STRIDE, WPT, TILE_R, TILE_C, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(const int blockTid, _Ty* __restrict__ src, const int srcOffsetRows, const int srcOffsetCols, _Ty* __restrict__ dst, AccessMap&& srcAccessMap)
    {
        CopyModeHelper<MODE, BLOCK_SIZE, WPT, STRIDE, TILE_C>::copy(blockTid, [=](const int i, const int p, const int r, const int c){
            StrideCopyHelper<SRC_ROW_MAJOR, /*DST_ROW_MAJOR*/ true, /*FORWARD*/ false, STRIDE, TILE_R, _vTy, _Ty>::copy(src, srcOffsetRows + r, srcOffsetCols + c, dst, i, -1, -1, srcAccessMap);
        });
    }
};

template<CopyMode MODE, int BLOCK_SIZE, bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, int STRIDE, int WPT, int TILE_R, int TILE_C, typename _vTy, typename _Ty>
struct ThreadCopyHelper<MemSpace::None, MemSpace::Shared, MODE, BLOCK_SIZE, SRC_ROW_MAJOR, DST_ROW_MAJOR, STRIDE, WPT, TILE_R, TILE_C, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(const int blockTid, _Ty* __restrict__ src, const int srcOffsetRows, const int srcOffsetCols, _Ty* __restrict__ dst, AccessMap&& srcAccessMap)
    {
        CopyModeHelper<MODE, BLOCK_SIZE, WPT, STRIDE, TILE_C>::copy(blockTid, [=](const int i, const int p, const int r, const int c){
            StrideCopyHelper<SRC_ROW_MAJOR, DST_ROW_MAJOR, /*FORWARD*/ true, STRIDE, TILE_R, _vTy, _Ty>::copy(src, srcOffsetRows + r, srcOffsetCols + c, dst, p, r, c, srcAccessMap);
        });
    }
};

template<CopyMode MODE, bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, int STRIDE, int WPT, int TILE_R, int TILE_C, int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z, MemSpace SRC_MEMSPACE, MemSpace DST_MEMSPACE, typename _vTy, typename _Ty, typename AccessMap>
__device__ __forceinline__ void block_copy(const int blockThreadId, _Ty* __restrict__ src, const int srcOffsetRows, const int srcOffsetCols, AccessMap&& srcAccessMap, _Ty* __restrict__ dst)
{
    constexpr auto BLOCK_SIZE = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    constexpr auto TOTAL_WORK = WPT * BLOCK_SIZE;
    static_assert(sizeof(_vTy) == STRIDE * sizeof(_Ty), "_vTy == _Ty[STRIDE]");
    static_assert(DST_MEMSPACE != MemSpace::Shared || TOTAL_WORK == TILE_R * TILE_C, "DST_MEMSPACE == Shared --> TOTAL_WORK == TILE_R * TILE_C");
    static_assert(SRC_MEMSPACE != MemSpace::Shared, "Source should not be Shared memory.");

    ThreadCopyHelper<SRC_MEMSPACE, DST_MEMSPACE, MODE, BLOCK_SIZE, SRC_ROW_MAJOR, DST_ROW_MAJOR, STRIDE, WPT, TILE_R, TILE_C, _vTy, _Ty>::copy(blockThreadId, src, srcOffsetRows, srcOffsetCols, dst, srcAccessMap);
}

)CUDA";

    constexpr auto FragmentHelpers = R"CUDA(
template<typename _Ty>
__device__ __forceinline__ void relu(_Ty& val)
{
    val = val > _Ty{} ? val : _Ty{};
}

template<typename _Ty>
__device__ __forceinline__ void relu_no_conditional(_Ty& val)
{
    val *= bool{val > _Ty{}};
}

template<typename _Ty>
__device__ __forceinline__ void set(_Ty& val, const _Ty arg)
{
    val = arg;
}

template<typename _Ty>
__device__ __forceinline__ void scale(_Ty& val, const _Ty arg)
{
    val *= arg;
}

)CUDA";

} // namespace cpp_printer
} // namespace mlir

#endif
