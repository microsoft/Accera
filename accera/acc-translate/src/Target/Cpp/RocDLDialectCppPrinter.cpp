////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "RocDLDialectCppPrinter.h"

#include "AMDGPU.h"

using namespace mlir;

#define PP_CONCAT(A, B) PP_CONCAT_IMPL(A, B)
#define PP_CONCAT_IMPL(A, B) A##B

#define PP_STRINGIFY_IMPL(X) #X
#define PP_STRINGIFY(X) PP_STRINGIFY_IMPL(X)

#define MFMA_CPP_TYPE__f32 float
#define MFMA_CPP_TYPE__f16 __half
#define MFMA_CPP_TYPE__i8 int8_t
#define MFMA_CPP_TYPE__i32 int32_t
#define MFMA_CPP_TYPE__bf16 uint8_t

#define MFMA_CPP_TYPE(X) MFMA_CPP_TYPE__ #X

// Format is (OutTy, InTy, M, N, K)
#define MFMA_FUNCTION_LIST(X) \
    X(f32, f32, 4, 4, 1)      \
    X(f32, f32, 16, 16, 1)    \
    X(f32, f32, 16, 16, 4)    \
    X(f32, f32, 32, 32, 1)    \
    X(f32, f32, 32, 32, 2)    \
    X(f32, f16, 4, 4, 4)      \
    X(f32, f16, 16, 16, 16)   \
    X(f32, f16, 32, 32, 4)    \
    X(f32, f16, 32, 32, 8)    \
    X(f32, f16, 16, 16, 4)    \
    X(f32, bf16, 4, 4, 2)     \
    X(f32, bf16, 16, 16, 2)   \
    X(f32, bf16, 16, 16, 8)   \
    X(f32, bf16, 32, 32, 2)   \
    X(f32, bf16, 32, 32, 4)   \
    X(i32, i8, 4, 4, 4)       \
    X(i32, i8, 16, 16, 4)     \
    X(i32, i8, 16, 16, 16)    \
    X(i32, i8, 32, 32, 4)     \
    X(i32, i8, 32, 32, 8)

#define MANGLE_MFMA_ARGS(OutTy, InTy, M, N, K) OutTy##_##M##x##N##x##K##InTy
#define ROCDL_OP(OutTy, InTy, M, N, K) PP_CONCAT(::mlir::ROCDL::mfma_, MANGLE_MFMA_ARGS(OutTy, InTy, M, N, K))
#define CPP_BUILTIN_MFMA_FUNCTION(OutTy, InTy, M, N, K) PP_CONCAT(__builtin_amdgcn_mfma_, MANGLE_MFMA_ARGS(OutTy, InTy, M, N, K))
#define CPP_BUILTIN_MFMA_FUNCTION_NAME(OutTy, InTy, M, N, K) PP_STRINGIFY(CPP_BUILTIN_MFMA_FUNCTION(OutTy, InTy, M, N, K))

namespace mlir
{
namespace cpp_printer
{

    static bool isMFMAOp(Operation* op)
    {
#define RETURN_TRUE_IF_MFMA_OP(OutTy, InTy, M, N, K) \
    if (isa<ROCDL_OP(OutTy, InTy, M, N, K)>(op))     \
    {                                                \
        return true;                                 \
    }
        MFMA_FUNCTION_LIST(RETURN_TRUE_IF_MFMA_OP);
#undef RETURN_TRUE_IF_MFMA_OP
        return false;
    }

    LogicalResult RocDLDialectCppPrinter::printMFMAOp(Operation* op)
    {
        assert(isMFMAOp(op));
        if (!isMFMAOp(op))
        {
            return failure();
        }

#define PRINT_MFMA_OP(OutTy, InTy, M, N, K)                                \
    if (auto mfmaOp = dyn_cast<ROCDL_OP(OutTy, InTy, M, N, K)>(op))        \
    {                                                                      \
        auto res = mfmaOp.res();                                           \
        auto idx = state.nameState.getOrCreateName(                        \
            res, SSANameState::SSANameKind::Variable);                     \
        RETURN_IF_FAILED(printer->printType(res.getType()));               \
        os << " " << idx << " = ";                                         \
        os << CPP_BUILTIN_MFMA_FUNCTION_NAME(OutTy, InTy, M, N, K) << "("; \
    }
        MFMA_FUNCTION_LIST(PRINT_MFMA_OP);
#undef PRINT_MFMA_OP
        RETURN_IF_FAILED(printer->printOperationOperands(op));
        os << ")";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printBarrierOp(ROCDL::BarrierOp barrierOp)
    {
        if (!isCuda)
        {
            return barrierOp.emitError("non-cuda version is not supported yet");
        }

        os << "__syncthreads()";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printBlockDimXOp(ROCDL::BlockDimXOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = blockDim.x";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printBlockDimYOp(ROCDL::BlockDimYOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = blockDim.y";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printBlockDimZOp(ROCDL::BlockDimZOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = blockDim.z";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printBlockIdXOp(ROCDL::BlockIdXOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = blockIdx.x";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printBlockIdYOp(ROCDL::BlockIdYOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = blockIdx.y";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printBlockIdZOp(ROCDL::BlockIdZOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = blockIdx.z";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printGridDimXOp(ROCDL::GridDimXOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = gridDimx.x";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printGridDimYOp(ROCDL::GridDimYOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = gridDimx.y";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printGridDimZOp(ROCDL::GridDimZOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = gridDimx.z";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printThreadIdXOp(ROCDL::ThreadIdXOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = threadIdx.x";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printThreadIdYOp(ROCDL::ThreadIdYOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = threadIdx.y";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printThreadIdZOp(ROCDL::ThreadIdZOp op)
    {
        if (!isCuda)
        {
            return op.emitError("non-cuda version is not supported yet");
        }

        auto idx = state.nameState.getOrCreateName(
            op.getResult(), SSANameState::SSANameKind::Variable);
        os << "int " << idx << " = threadIdx.z";
        return success();
    }

    LogicalResult RocDLDialectCppPrinter::printMubufLoadOp(ROCDL::MubufLoadOp op)
    {
        return op.emitError("op is not supported yet");
    }

    LogicalResult RocDLDialectCppPrinter::printMubufStoreOp(ROCDL::MubufStoreOp op)
    {
        return op.emitError("op is not supported yet");
    }

    LogicalResult RocDLDialectCppPrinter::printDialectOperation(Operation* op,
                                                                bool* /*skipped*/,
                                                                bool* consumed)
    {
        *consumed = true;
        if (isMFMAOp(op))
            return printMFMAOp(op);

        if (auto barrierOp = dyn_cast<ROCDL::BarrierOp>(op))
            return printBarrierOp(barrierOp);

        if (auto blockDimOp = dyn_cast<ROCDL::BlockDimXOp>(op))
            return printBlockDimXOp(blockDimOp);
        if (auto blockDimOp = dyn_cast<ROCDL::BlockDimYOp>(op))
            return printBlockDimYOp(blockDimOp);
        if (auto blockDimOp = dyn_cast<ROCDL::BlockDimZOp>(op))
            return printBlockDimZOp(blockDimOp);

        if (auto blockIdxOp = dyn_cast<ROCDL::BlockIdXOp>(op))
            return printBlockIdXOp(blockIdxOp);
        if (auto blockIdxOp = dyn_cast<ROCDL::BlockIdYOp>(op))
            return printBlockIdYOp(blockIdxOp);
        if (auto blockIdxOp = dyn_cast<ROCDL::BlockIdZOp>(op))
            return printBlockIdZOp(blockIdxOp);

        if (auto gridimOp = dyn_cast<ROCDL::GridDimXOp>(op))
            return printGridDimXOp(gridimOp);
        if (auto gridimOp = dyn_cast<ROCDL::GridDimYOp>(op))
            return printGridDimYOp(gridimOp);
        if (auto gridimOp = dyn_cast<ROCDL::GridDimZOp>(op))
            return printGridDimZOp(gridimOp);

        if (auto threadIdxOp = dyn_cast<ROCDL::ThreadIdXOp>(op))
            return printThreadIdXOp(threadIdxOp);
        if (auto threadIdxOp = dyn_cast<ROCDL::ThreadIdYOp>(op))
            return printThreadIdYOp(threadIdxOp);
        if (auto threadIdxOp = dyn_cast<ROCDL::ThreadIdZOp>(op))
            return printThreadIdZOp(threadIdxOp);

        if (auto memBufLoadOp = dyn_cast<ROCDL::MubufLoadOp>(op))
            return printMubufLoadOp(memBufLoadOp);

        if (auto memBufStoreOp = dyn_cast<ROCDL::MubufStoreOp>(op))
            return printMubufStoreOp(memBufStoreOp);

        *consumed = false;
        return success();
    }

} // namespace cpp_printer
} // namespace mlir