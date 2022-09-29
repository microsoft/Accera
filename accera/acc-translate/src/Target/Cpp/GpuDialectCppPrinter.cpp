////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Kern Handa, Captain Jack Sparrow
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GpuDialectCppPrinter.h"
#include "CppPrinterUtils.h"
#include <llvm/ADT/None.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LogicalResult.h>

#include <functional>

#include <ir/include/IRUtil.h>

using namespace mlir::gpu;

namespace ir = accera::ir;
namespace utilir = accera::ir::util;
namespace vir = accera::ir::value;

namespace mlir
{
namespace cpp_printer
{

    LogicalResult GpuDialectCppPrinter::printOp(BarrierOp barrierOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return barrierOp.emitError("non-cuda version is not supported yet");
        }

        os << "__syncthreads()";
        return success();
    }

    static Optional<uint64_t> getGridDim(Operation* op, gpu::Dimension dim)
    {

        if (auto fn = op->getParentOfType<FuncOp>())
        {
            if (!fn->hasAttrOfType<ArrayAttr>("gridSize"))
            {
                return llvm::None;
            }
            auto arrayAttr = utilir::ArrayAttrToVector<mlir::IntegerAttr>(fn->getAttrOfType<ArrayAttr>("gridSize"));
            return arrayAttr[static_cast<uint32_t>(dim)].getInt();
        }
        return llvm::None;
    }

    static Optional<uint64_t> getBlockDim(Operation* op, gpu::Dimension dim)
    {
        if (auto fn = op->getParentOfType<FuncOp>())
        {
            if (!fn->hasAttrOfType<ArrayAttr>("blockSize"))
            {
                return llvm::None;
            }
            auto arrayAttr = utilir::ArrayAttrToVector<mlir::IntegerAttr>(fn->getAttrOfType<ArrayAttr>("blockSize"));
            return arrayAttr[static_cast<uint32_t>(dim)].getInt();
        }
        return llvm::None;
    }

    LogicalResult GpuDialectCppPrinter::printOp(GridDimOp gridDimOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return gridDimOp.emitError("non-cuda version is not supported yet");
        }

        auto dimStr = gpu::stringifyDimension(gridDimOp.dimension()).str();
        const std::string varPrefix = std::string("gridDim_") + dimStr + "_";
        auto idx = state.nameState.getOrCreateName(
            gridDimOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        RETURN_IF_FAILED(printGPUIndexType());

        os << " " << idx << " = ";

        if (auto c = getGridDim(gridDimOp, gridDimOp.dimension()); c)
        {
            os << c.getValue();
        }
        else
        {
            os << "gridDim." << dimStr;
        }
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(BlockDimOp blockDimOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return blockDimOp.emitError("non-cuda version is not supported yet");
        }

        auto dimStr = gpu::stringifyDimension(blockDimOp.dimension()).str();
        const std::string varPrefix = std::string("blockDim_") + dimStr + "_";
        auto idx = state.nameState.getOrCreateName(
            blockDimOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        RETURN_IF_FAILED(printGPUIndexType());

        os << " " << idx << " = ";

        if (auto c = getBlockDim(blockDimOp, blockDimOp.dimension()); c)
        {
            os << c.getValue();
        }
        else
        {
            os << "blockDim." << dimStr;
        }
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(BlockIdOp bidOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return bidOp.emitError("non-cuda version is not supported yet");
        }

        auto dimStr = gpu::stringifyDimension(bidOp.dimension()).str();
        const std::string varPrefix = std::string("blockIdx_") + dimStr + "_";
        auto idx = state.nameState.getOrCreateName(
            bidOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        RETURN_IF_FAILED(printGPUIndexType());

        os << " " << idx << " = ";

        if (auto c = getGridDim(bidOp, bidOp.dimension()); c)
        {
            os << "(blockIdx." << dimStr << "%" << c.getValue() << ")";
        }
        else
        {

            os << "blockIdx." << dimStr;
        }
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(ThreadIdOp tidOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return tidOp.emitError("non-cuda version is not supported yet");
        }

        auto dimStr = gpu::stringifyDimension(tidOp.dimension()).str();
        const std::string varPrefix = std::string("threadIdx_") + dimStr + "_";
        auto idx = state.nameState.getOrCreateName(
            tidOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        RETURN_IF_FAILED(printGPUIndexType());

        os << " " << idx << " = ";

        if (auto c = getBlockDim(tidOp, tidOp.dimension()); c)
        {
            os << "(threadIdx." << dimStr << "%" << c.getValue() << ")";
        }
        else
        {
            os << "threadIdx." << dimStr;
        }
        return success();
    }

    vir::MMAOperandType convertToOperandType(const StringRef& operand)
    {
        if (operand == "AOp")
            return vir::MMAOperandType::A;
        if (operand == "BOp")
            return vir::MMAOperandType::B;
        if (operand == "COp")
            return vir::MMAOperandType::Acc;
        llvm_unreachable("Unknown mma operand");
    }

    int inferM(int64_t K, int64_t N)
    {
        // M16xN16xK16_B1
        if (N == 16 && K == 16)
            return 16;

        // M32xN8xK16_B1
        if (N == 8 && K == 16)
            return 32;

        // M8xN32xK16_B1
        if (N == 32 && K == 16)
            return 8;

        return {};
    }

    int inferN(int64_t M, int64_t K)
    {
        // M16xN16xK16_B1
        if (M == 16 && K == 16)
            return 16;

        // M32xN8xK16_B1
        if (M == 32 && K == 16)
            return 8;

        // M8xN32xK16_B1
        if (M == 8 && K == 16)
            return 32;

        return {};
    }

    int inferK(int64_t M, int64_t N)
    {
        // M16xN16xK16_B1
        if (M == 16 && N == 16)
            return 16;

        // M32xN8xK16_B1
        if (M == 32 && N == 8)
            return 16;

        // M8xN32xK16_B1
        if (M == 8 && N == 32)
            return 16;

        return {};
    }

    LogicalResult GpuDialectCppPrinter::printOp(gpu::SubgroupMmaConstantMatrixOp constantMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return constantMatrixOp.emitError("non-cuda version is not supported.");
        }

        auto mmaMatrix = constantMatrixOp.res().getType().cast<MMAMatrixType>();
        auto&& shape = mmaMatrix.getShape();
        const auto mmaShape = std::make_tuple(shape[0], shape[1], inferK(shape[0], shape[1]));
        return printConstantMatrixOp(state, printer, mmaShape, constantMatrixOp.res(), constantMatrixOp.value());
    }

    LogicalResult GpuDialectCppPrinter::printOp(SubgroupMmaLoadMatrixOp loadMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return loadMatrixOp.emitError("non-cuda version is not supported.");
        }

        const auto mmaMatrix = loadMatrixOp.res().getType().cast<MMAMatrixType>();
        const auto operandType = convertToOperandType(mmaMatrix.getOperand());
        auto&& shape = mmaMatrix.getShape();
        std::tuple<int, int, int> mmaShape;
        switch (operandType)
        {
        case vir::MMAOperandType::A:
            mmaShape = std::make_tuple(/*M*/ shape[0], inferN(shape[0], shape[1]), /*K*/ shape[1]);
            break;
        case vir::MMAOperandType::B:
            mmaShape = std::make_tuple(inferM(shape[0], shape[1]), /*N*/ shape[1], /*K*/ shape[0]);
            break;
        case vir::MMAOperandType::Acc:
            mmaShape = std::make_tuple(/*M*/ shape[0], /*N*/ shape[1], inferK(shape[0], shape[1]));
            break;
        default:
            return failure("Unsupported matrix used for MMA.");
        }

        int64_t offset;
        SmallVector<int64_t, 2> strides;
        RETURN_IF_FAILED(mlir::getStridesAndOffset(loadMatrixOp.srcMemref().getType().cast<MemRefType>(), strides, offset));
        const bool row_major = strides[1] == 1;

        return printLoadMatrixOp(state, printer, mmaShape, loadMatrixOp.srcMemref(), loadMatrixOp.res(), operandType, loadMatrixOp.indices(), row_major);
    }

    LogicalResult GpuDialectCppPrinter::printOp(gpu::SubgroupMmaComputeOp computeMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return computeMatrixOp.emitError("non-cuda version is not supported.");
        }

        auto mmaMatrix = computeMatrixOp.res().getType().cast<MMAMatrixType>();
        auto&& shape = mmaMatrix.getShape();
        const auto mmaShape = std::make_tuple(shape[0], shape[1], inferK(shape[0], shape[1]));
        return printComputeMatrixOp(state, printer, mmaShape, computeMatrixOp.opA(), computeMatrixOp.opB(), computeMatrixOp.opC(), computeMatrixOp.res());
    }

    LogicalResult GpuDialectCppPrinter::printOp(gpu::SubgroupMmaStoreMatrixOp storeMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return storeMatrixOp.emitError("non-cuda version is not supported.");
        }

        return printStoreMatrixOp(state, printer, storeMatrixOp.src(), storeMatrixOp.dstMemref(), storeMatrixOp.indices());
    }

    LogicalResult GpuDialectCppPrinter::printDialectOperation(Operation* op,
                                                              bool* /*skipped*/,
                                                              bool* consumed)
    {
        auto handler = [&, this](auto op_) {
            RETURN_IF_FAILED(printOp(op_));
            *consumed = true;
            return success();
        };

        return TypeSwitch<Operation*, LogicalResult>(op)
            // KEEP THIS SORTED
            .Case<BarrierOp>(handler)
            .Case<BlockDimOp>(handler)
            .Case<BlockIdOp>(handler)
            .Case<GPUFuncOp>(handler)
            .Case<GPUModuleOp>(handler)
            .Case<gpu::ReturnOp>(handler)
            .Case<GridDimOp>(handler)
            .Case<LaunchFuncOp>(handler)
            .Case<ModuleEndOp>(handler)
            .Case<SubgroupMmaComputeOp>(handler)
            .Case<SubgroupMmaConstantMatrixOp>(handler)
            .Case<SubgroupMmaLoadMatrixOp>(handler)
            .Case<SubgroupMmaStoreMatrixOp>(handler)
            .Case<ThreadIdOp>(handler)
            .Default([&](Operation*) { *consumed = false; return success(); });
    }

    LogicalResult GpuDialectCppPrinter::printGpuFPVectorType(VectorType vecType,
                                                             StringRef vecVar)
    {
        if (vecType.getNumDynamicDims())
        {
            os << "<<VectorType with dynamic dims is not supported yet>>";
            return failure();
        }

        auto rank = vecType.getRank();
        if (rank == 0)
        {
            os << "<<zero-ranked Vectortype is not supported yet>>";
            return failure();
        }

        auto shape = vecType.getShape();
        if (shape[rank - 1] % 2)
        {
            os << "<<can't be represented by " << printer->floatVecT<32>(shape[rank - 1]) << " as it is not a multiple of 2>>";
            return failure();
        }

        RETURN_IF_FAILED(printer->printType(VectorType::get({ shape[rank - 1] }, vecType.getElementType())));
        os << " " << vecVar;

        for (int i = 0; i < rank - 1; i++)
        {
            os << "[" << shape[i] << "]";
        }
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printVectorTypeArrayDecl(VectorType vecType,
                                                                 StringRef vecVar)
    {
        assert(state.hasRuntime(Runtime::CUDA) && "not for cuda?");

        auto elemType = vecType.getElementType();
        // TODO: support more vector types
        if (elemType.isa<Float32Type>() || elemType.isa<Float16Type>())
        {
            return printGpuFPVectorType(vecType, vecVar);
        }
        else
        {
            os << "<<only support fp32 and fp16 vec type>>";
            return failure();
        }
    }

    LogicalResult GpuDialectCppPrinter::runPrePrintingPasses(Operation* op)
    {
        if (auto moduleOp = dyn_cast<mlir::ModuleOp>(op))
        {
            auto& moduleRegion = moduleOp.getRegion();

            auto potentialGpuOps = moduleRegion.getOps<gpu::GPUModuleOp>();

            if (!potentialGpuOps.empty())
            {
                llvm::errs() << "GPU module detected, enabling CUDA runtime\n";
                _gpuModuleOps = llvm::to_vector<4>(potentialGpuOps);
            }
        }

        for (auto gpuOp : _gpuModuleOps)
        {
            auto execRuntime = accera::ir::util::ResolveExecutionRuntime(gpuOp);
            switch (execRuntime)
            {
            case vir::ExecutionRuntime::ROCM:
                state.setRuntime(Runtime::ROCM);
                // TODO: Make ROCM not a subset of CUDA
                [[fallthrough]]; // and also
            case vir::ExecutionRuntime::CUDA:
                state.setRuntime(Runtime::CUDA);
                break;
            case vir::ExecutionRuntime::NONE:
                [[fallthrough]];
            case vir::ExecutionRuntime::OPENMP:
                [[fallthrough]];
            case vir::ExecutionRuntime::VULKAN:
                [[fallthrough]];
            case vir::ExecutionRuntime::DEFAULT:
                [[fallthrough]];
            default:
                llvm::errs() << "Device functions runtime is unsupported\n";
                return failure();
            }
        }

        return success();
    }

    LogicalResult GpuDialectCppPrinter::printHeaderFiles()
    {
        if (state.hasRuntime(Runtime::ROCM))
        {
            os << R"CUDA(
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

#ifndef __forceinline__
#define __forceinline__ __inline__ __attribute__((always_inline))
#endif // __forceinline__

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

)CUDA";
        }
        else if (state.hasRuntime(Runtime::CUDA))
        {
            os << R"CUDA(
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
        }

        if (state.hasRuntime(Runtime::CUDA))
        {
            // Common to both CUDA and ROCM
            os << R"CUDA(
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

template<bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper{};

template<int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper<true, true, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(const _Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, const int idx, int, int, AccessMap&& srcAccessMap)
    {
        const auto srcOffset = srcAccessMap(src_r, src_c);
        *reinterpret_cast<_vTy*>(&dst[idx]) = *reinterpret_cast<const _vTy*>(&src[srcOffset]);
    }
};

template<int STRIDE, int TILE_R, typename _vTy, typename _Ty>
struct StrideCopyHelper<false, true, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(const _Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, const int idx, int, int, AccessMap&& srcAccessMap)
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
struct StrideCopyHelper<true, false, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(const _Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, int, const int dst_r, const int dst_c, AccessMap&& srcAccessMap)
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
struct StrideCopyHelper<false, false, STRIDE, TILE_R, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(const _Ty* __restrict__ src, const int src_r, const int src_c, _Ty* __restrict__ dst, int, const int dst_r, const int dst_c, AccessMap&& srcAccessMap)
    {
        for (int i = 0, p = dst_c * TILE_R + dst_r; i < STRIDE; ++i, p += TILE_R)
        {
            const auto srcOffset = srcAccessMap(src_r, src_c + i);
            dst[p] = src[srcOffset];
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
    static __device__ __forceinline__ void copy(const int blockTid, const _Ty* __restrict__ src, const int srcOffsetRows, const int srcOffsetCols, _Ty* __restrict__ dst, AccessMap&&)
    {
        CopyModeHelper<MODE, BLOCK_SIZE, WPT, STRIDE, TILE_C>::copy(blockTid, [=](const int i, const int p, const int r, const int c){
            StrideCopyHelper</*SRC_ROW_MAJOR*/ true, DST_ROW_MAJOR, STRIDE, TILE_R, _vTy, _Ty>::copy(src, -1, -1, dst, p, r, c, [=](int, int){ return i; });
        });
    }
};

template<CopyMode MODE, int BLOCK_SIZE, bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, int STRIDE, int WPT, int TILE_R, int TILE_C, typename _vTy, typename _Ty>
struct ThreadCopyHelper<MemSpace::None, MemSpace::Private, MODE, BLOCK_SIZE, SRC_ROW_MAJOR, DST_ROW_MAJOR, STRIDE, WPT, TILE_R, TILE_C, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(const int blockTid, const _Ty* __restrict__ src, const int srcOffsetRows, const int srcOffsetCols, _Ty* __restrict__ dst, AccessMap&& srcAccessMap)
    {
        CopyModeHelper<MODE, BLOCK_SIZE, WPT, STRIDE, TILE_C>::copy(blockTid, [=](const int i, const int p, const int r, const int c){
            StrideCopyHelper<SRC_ROW_MAJOR, /*DST_ROW_MAJOR*/ true, STRIDE, TILE_R, _vTy, _Ty>::copy(src, srcOffsetRows + r, srcOffsetCols + c, dst, i, -1, -1, srcAccessMap);
        });
    }
};

template<CopyMode MODE, int BLOCK_SIZE, bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, int STRIDE, int WPT, int TILE_R, int TILE_C, typename _vTy, typename _Ty>
struct ThreadCopyHelper<MemSpace::None, MemSpace::Shared, MODE, BLOCK_SIZE, SRC_ROW_MAJOR, DST_ROW_MAJOR, STRIDE, WPT, TILE_R, TILE_C, _vTy, _Ty>
{
    template<typename AccessMap>
    static __device__ __forceinline__ void copy(const int blockTid, const _Ty* __restrict__ src, const int srcOffsetRows, const int srcOffsetCols, _Ty* __restrict__ dst, AccessMap&& srcAccessMap)
    {
        CopyModeHelper<MODE, BLOCK_SIZE, WPT, STRIDE, TILE_C>::copy(blockTid, [=](const int i, const int p, const int r, const int c){
            StrideCopyHelper<SRC_ROW_MAJOR, DST_ROW_MAJOR, STRIDE, TILE_R, _vTy, _Ty>::copy(src, srcOffsetRows + r, srcOffsetCols + c, dst, p, r, c, srcAccessMap);
        });
    }
};

template<CopyMode MODE, bool SRC_ROW_MAJOR, bool DST_ROW_MAJOR, int STRIDE, int WPT, int TILE_R, int TILE_C, int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z, MemSpace SRC_MEMSPACE, MemSpace DST_MEMSPACE, typename _vTy, typename _Ty, typename AccessMap>
__device__ __forceinline__ void block_copy(const int blockThreadId, const _Ty* __restrict__ src, const int srcOffsetRows, const int srcOffsetCols, AccessMap&& srcAccessMap, _Ty* __restrict__ dst)
{
    constexpr auto BLOCK_SIZE = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    constexpr auto TOTAL_WORK = WPT * BLOCK_SIZE;
    static_assert(sizeof(_vTy) == STRIDE * sizeof(_Ty), "_vTy == _Ty[STRIDE]");
    static_assert(DST_MEMSPACE != MemSpace::Shared || TOTAL_WORK == TILE_R * TILE_C, "DST_MEMSPACE == Shared --> TOTAL_WORK == TILE_R * TILE_C");
    static_assert(SRC_MEMSPACE != MemSpace::Shared, "Source should not be Shared memory.");

    ThreadCopyHelper<SRC_MEMSPACE, DST_MEMSPACE, MODE, BLOCK_SIZE, SRC_ROW_MAJOR, DST_ROW_MAJOR, STRIDE, WPT, TILE_R, TILE_C, _vTy, _Ty>::copy(blockThreadId, src, srcOffsetRows, srcOffsetCols, dst, srcAccessMap);
}

)CUDA";
        }

        return success();
    }

    // TODO: Dedupe with CppPrinter's version
    LogicalResult GpuDialectCppPrinter::printFunctionDeclaration(
        gpu::GPUFuncOp funcOp,
        bool trailingSemiColon)
    {
        auto execRuntime = utilir::ResolveExecutionRuntime(funcOp, /* exact */ false);
        if (execRuntime != vir::ExecutionRuntime::CUDA &&
            execRuntime != vir::ExecutionRuntime::ROCM &&
            // TODO: ugh. remove
            execRuntime != vir::ExecutionRuntime::DEFAULT)
        {
            return funcOp.emitError("Expected either CUDA or ROCm runtimes on GPU function");
        }

        if (funcOp->hasAttr(ir::HeaderDeclAttrName) && funcOp->hasAttr(ir::RawPointerAPIAttrName))
        {
            os << "extern \"C\" ";
        }

        // TODO: We treat all functions to be CUDA global functions.
        // Need to add support for device functions
        os << "__global__ ";

        if (state.hasRuntime(Runtime::CUDA) && funcOp->hasAttrOfType<mlir::ArrayAttr>("blockSize"))
        {
            auto arrayAttr = utilir::ArrayAttrToVector<mlir::IntegerAttr>(funcOp->getAttrOfType<mlir::ArrayAttr>("blockSize"));
            auto blockSizeX = arrayAttr[0].getInt();
            auto blockSizeY = arrayAttr[1].getInt();
            auto blockSizeZ = arrayAttr[2].getInt();
            os << " __launch_bounds__(" << blockSizeX * blockSizeY * blockSizeZ << ") ";
        }

        auto resultType = funcOp.getType().getResults();
        if (state.hasRuntime(Runtime::CUDA) && !resultType.empty())
        {
            return funcOp.emitOpError() << "<<CUDA kernel must return void>>";
        }

        if (failed(printer->printTypes(funcOp.getType().getResults())))
        {
            return funcOp.emitOpError() << "<<Unable to print return type>>";
        }

        os << " " << funcOp.getName();

        os << "(";
        // external function
        if (funcOp.getBlocks().size() == 0)
        {
            (void)interleaveCommaWithError(
                funcOp.getType().getInputs(), os, [&](Type tp) -> LogicalResult {
                    if (auto memRefType = tp.dyn_cast<MemRefType>())
                    {
                        return printer->printDecayedArrayDeclaration(memRefType, /*arrayName*/ "");
                    }
                    else
                    {
                        return printer->printType(tp);
                    }
                });
        }
        else
        {
            SSANameState::Scope scope(state.nameState);
            auto usedNamesScope = state.nameState.createUsedNamesScope();

            (void)interleaveCommaWithError(funcOp.getArguments(), os, [&](BlockArgument arg) -> LogicalResult {
                return printer->printBlockArgument(arg);
            });
        }
        os << ") ";

        if (trailingSemiColon)
            os << ";\n\n";
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(gpu::GPUModuleOp gpuModuleOp)
    {
        assert(llvm::is_contained(_gpuModuleOps, gpuModuleOp));

        for (Operation& op : gpuModuleOp.getOps())
        {
            [[maybe_unused]] bool skipped{}; // not sure what to do with this

            RETURN_IF_FAILED(printer->printOperation(&op, &skipped, /* trailingSemiColon */ false));
        }

        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(ModuleEndOp)
    {
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(gpu::ReturnOp)
    {
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(LaunchFuncOp launchOp)
    {
        auto gridSizes = { launchOp.gridSizeX(), launchOp.gridSizeY(), launchOp.gridSizeZ() };
        auto blockSizes = { launchOp.blockSizeX(), launchOp.blockSizeY(), launchOp.blockSizeZ() };
        auto operands = launchOp->getOperands().drop_front(launchOp.kNumConfigOperands);
        auto getName = [&](Value operand) { return state.nameState.getName(operand); };
        auto pprint = [&](auto&& container) {
            llvm::interleave(
                llvm::map_range(
                    container,
                    getName),
                os,
                ", ");
        };

        auto kernelNameAttr = launchOp.getKernelName();
        // Printing the attr directly results in the kernel name being surrounded by quotes (since LLVM 14)
        auto kernelName = kernelNameAttr.str();

        os << kernelName << "<<<dim3(";
        pprint(gridSizes);
        os << "), dim3(";
        pprint(blockSizes);
        os << ")>>>(";
        pprint(operands);
        os << ")";

        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(gpu::GPUFuncOp funcOp)
    {
        SSANameState::Scope scope(state.nameState);
        auto usedNamesScope = state.nameState.createUsedNamesScope();

        auto& blocks = funcOp.getBlocks();
        auto numBlocks = blocks.size();
        if (numBlocks > 1)
            return funcOp.emitOpError() << "<<only single block functions supported>>";

        // print function declaration
        if (failed(printFunctionDeclaration(funcOp,
                                            /*trailingSemicolon*/ numBlocks == 0)))
        {
            return funcOp.emitOpError() << "<<failed to print function declaration>>";
        }

        if (numBlocks != 0)
        {
            // print function body
            if (failed(printer->printBlock(&(blocks.front()))))
                return funcOp.emitOpError() << "<<failed to print function body>>";
        }
        // Otherwise, just a declaration, so emit a newline and return

        os << "\n\n";

        return success();
    }

    LogicalResult GpuDialectCppPrinter::printDeclarations()
    {
        if (state.hasRuntime(Runtime::CUDA))
        {
            for (auto gpuModuleOp : _gpuModuleOps)
            {
                for (auto funcOp : gpuModuleOp.getOps<gpu::GPUFuncOp>())
                {
                    RETURN_IF_FAILED(printFunctionDeclaration(funcOp, /* trailingSemiColon */ true));
                }
            }
        }

        return success();
    }

    LogicalResult GpuDialectCppPrinter::printGPUIndexType()
    {
        os << "const ";
        return printer->printIndexType();
    }

} // namespace cpp_printer
} // namespace mlir
