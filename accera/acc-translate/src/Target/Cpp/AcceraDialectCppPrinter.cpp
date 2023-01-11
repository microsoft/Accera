////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa, Captain Jack Sparrow
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraDialectCppPrinter.h"
#include "AffineDialectCppPrinter.h"
#include "CppPrinterUtils.h"

#include "AMDGPU.h"
#include "NVGPU.h"
#include "ir/include/value/ValueDialect.h"

#include <ir/include/IRUtil.h>
#include <ir/include/argo/Utils.h>
#include <ir/include/nest/LoopNestOps.h>

#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace mlir::argo;

namespace vir = accera::ir::value;

namespace mlir
{
namespace cpp_printer
{
    LogicalResult AcceraDialectCppPrinter::printOp(vir::CallOp callOp)
    {
        auto callInterface = dyn_cast<CallOpInterface>(callOp.getOperation());
        auto callee = callInterface.resolveCallable();
        if (!callee) return callOp->emitError("Cannot find callee function");

        (void)printer->printDeclarationForOpResult(callOp);
        if (callOp->getNumResults() > 0)
            os << " = ";

        os << callOp.getCallee() << "(";
        RETURN_IF_FAILED(printer->printOperationOperands(callOp));
        os << ")";

        return success();
    }

    LogicalResult AcceraDialectCppPrinter::printOp(vir::ReturnOp returnOp)
    {
        os << "return";

        if (auto numOperands = returnOp.getNumOperands(); numOperands == 0)
        {
            // Nothing to do
        }
        else if (numOperands == 1)
        {
            os << " " << state.nameState.getName(returnOp.getOperand(0));
        }
        else
        {
            return returnOp.emitOpError() << "<<Returning tuple is not supported yet>>";
        }

        return success();
    }

    LogicalResult AcceraDialectCppPrinter::printOp(vir::WarpIdOp warpIdOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return warpIdOp.emitError("non-cuda version is not supported.");
        }

        RETURN_IF_FAILED(printer->printType(warpIdOp.result().getType()));
        auto idx = state.nameState.getOrCreateName(warpIdOp.result(), SSANameState::SSANameKind::Variable);
        auto tid = state.nameState.getOrCreateName(warpIdOp.threadId(), SSANameState::SSANameKind::Variable);
        os << " " << idx << " = ";
        if (state.hasRuntime(Runtime::ROCM))
        {
            os << "__builtin_amdgcn_readfirstlane(" << tid << ")";

            // ROCDL threadID ops will have a cast from i32 to index type, so navigate appropriately
            assert((gpu::Dimension{ warpIdOp.dimension() } == gpu::Dimension::x && warpIdOp.threadId().getDefiningOp<arith::IndexCastOp>().getIn().getDefiningOp<ROCDL::ThreadIdXOp>()) || (gpu::Dimension{ warpIdOp.dimension() } == gpu::Dimension::y && warpIdOp.threadId().getDefiningOp<arith::IndexCastOp>().getIn().getDefiningOp<ROCDL::ThreadIdYOp>()));
        }
        else
        {
            os << tid;
            assert(gpu::Dimension{ warpIdOp.dimension() } == warpIdOp.threadId().getDefiningOp<mlir::gpu::ThreadIdOp>().dimension());
        }

        if (gpu::Dimension{ warpIdOp.dimension() } == gpu::Dimension::x)
        {
            os << " / " << warpIdOp.warpSize();
        }

        return success();
    }

    LogicalResult AcceraDialectCppPrinter::printOp(vir::MMAAllocSyncOp allocMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return allocMatrixOp.emitError("non-cuda version is not supported.");
        }

        auto memRefType = allocMatrixOp.result().getType().cast<MemRefType>();
        const vir::MMAOp mfmaOpType{ static_cast<vir::MMAShape>(allocMatrixOp.mmaShapeType()) };
        const auto shape = std::make_tuple(mfmaOpType.getM(), mfmaOpType.getN(), mfmaOpType.getK());
        const vir::MMAOperandType opType{ allocMatrixOp.operandType() };
        const auto rowMajor = allocMatrixOp.rowMajor();
        return printMMAMatrixOp(state, printer, memRefType.getElementType(), shape, allocMatrixOp.result(), opType, mfmaOpType.getNumBlocks(), allocMatrixOp.blocks(), rowMajor);
    }

    // LogicalResult AcceraDialectCppPrinter::printOp(vir::MMAFillSyncOp constantMatrixOp)
    // {
    //     if (!state.hasRuntime(Runtime::CUDA))
    //     {
    //         return constantMatrixOp.emitError("non-cuda version is not supported.");
    //     }

    //     return printConstantMatrixOp(state, printer, constantMatrixOp.dest(), constantMatrixOp.value());
    // }

    LogicalResult AcceraDialectCppPrinter::printOp(vir::MMALoadSyncOp loadMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return loadMatrixOp.emitError("non-cuda version is not supported.");
        }

        const auto operandType = static_cast<vir::MMAOperandType>(loadMatrixOp.operandType());

        return printLoadMatrixOp(state, printer, loadMatrixOp.memref(), loadMatrixOp.dest(), operandType, loadMatrixOp.indices(), loadMatrixOp.rowMajor(), loadMatrixOp.blockThreadId(), loadMatrixOp.staticOffsets(), static_cast<vir::MMAFragmentOp>(loadMatrixOp.mmaPrologueOp()), loadMatrixOp.mmaPrologueArg());
    }

    LogicalResult AcceraDialectCppPrinter::printOp(vir::MMAComputeSyncOp computeMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return computeMatrixOp.emitError("non-cuda version is not supported.");
        }

        return printComputeMatrixOp(state, printer, computeMatrixOp.opA(), computeMatrixOp.opB(), computeMatrixOp.opC(), computeMatrixOp.opC(), computeMatrixOp.cbsz(), computeMatrixOp.abid(), computeMatrixOp.blgp());
    }

    LogicalResult AcceraDialectCppPrinter::printOp(vir::MMAStoreSyncOp storeMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return storeMatrixOp.emitError("non-cuda version is not supported.");
        }

        return printStoreMatrixOp(state, printer, storeMatrixOp.src(), storeMatrixOp.memref(), storeMatrixOp.indices(), storeMatrixOp.blockThreadId(), storeMatrixOp.staticOffsets(), static_cast<vir::MMAFragmentOp>(storeMatrixOp.mmaEpilogueOp()), storeMatrixOp.mmaEpilogueArg());
    }

    LogicalResult AcceraDialectCppPrinter::printVectorType(mlir::Type elementType, const uint32_t stride) const
    {
        if (state.hasRuntime(Runtime::ROCM))
        {
            RETURN_IF_FAILED(printer->printVectorType(elementType, stride));
        }
        else
        {
            std::string res;
            if (elementType.isF32())
                res = "float";
            else if (elementType.isF16())
                res = "half";

            res += stride == 1 ? "" : std::to_string(stride);
            os << res;
        }

        return success();
    }

    std::string getMemSpaceEnum(const uint64_t memSpace)
    {
        return "MemSpace::" + stringifyMemorySpace(*vir::symbolizeMemorySpace(memSpace)).str();
    }

    LogicalResult AcceraDialectCppPrinter::printOp(vir::GPUBlockCacheOp blockLoadOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return blockLoadOp.emitError("non-cuda version is not supported.");
        }

        auto srcMemref = blockLoadOp.memref();
        auto srcMemrefType = srcMemref.getType().cast<MemRefType>();
        auto srcMemSpace = srcMemrefType.getMemorySpaceAsInt();
        auto elementType = srcMemrefType.getElementType();
        AffineDialectCppPrinter* affineDialectPrinter = dynamic_cast<AffineDialectCppPrinter*>(printer->getDialectPrinter("Affine"));
        auto srcMap = mlir::getStridedLinearLayoutMap(srcMemrefType);
        auto srcRowMajor = mlir::canonicalizeStridedLayout(srcMemrefType).getLayout().isIdentity();

        auto dstMemrefType = blockLoadOp.dest().getType().cast<MemRefType>();
        auto destMemSpace = dstMemrefType.getMemorySpaceAsInt();
        auto dstRowMajor = blockLoadOp.dstRowMajor();

        const auto tileShape = accera::ir::util::ConvertArrayAttrToIntVector(blockLoadOp.tileShape());
        const auto var = SSANameState::SSANameKind::Variable;
        const auto accessMapName = srcMap.getNumResults() == 1 ? affineDialectPrinter->makeAffineIdxFuncName(affineDialectPrinter->getFuncBaseName(srcMap), 0) : "[](int, int)->int { /*unused*/ return 0; }";
        const auto src = state.nameState.getOrCreateName(srcMemref, var, "src_");
        const auto dst = state.nameState.getOrCreateName(blockLoadOp.dest(), var, "dst_");
        const auto srcOffsetRows = state.nameState.getOrCreateName(blockLoadOp.srcOffsetRows(), var, "src_off_r_");
        const auto srcOffsetCols = state.nameState.getOrCreateName(blockLoadOp.srcOffsetCols(), var, "src_off_c_");
        const auto blockThreadId = state.nameState.getOrCreateName(blockLoadOp.blockThreadId(), var, "tid_");
        const auto wpt = blockLoadOp.workPerThread();
        const auto vecWidth = blockLoadOp.vecWidth();
        const auto stride = std::min(wpt, vecWidth);
        const auto strategy = stringifyCacheStrategy(blockLoadOp.strategy());

        if (!blockLoadOp.srcToDst())
        {
            std::swap(srcMemSpace, destMemSpace);
            std::swap(srcRowMajor, dstRowMajor);
        }

        os << "block_copy<CopyMode::" << strategy << ", /*SRC_ROW_MAJOR*/ " << srcRowMajor << ", /*DST_ROW_MAJOR*/ " << dstRowMajor << ", /*STRIDE*/ " << stride << ", /*WPT*/ " << wpt;
        os << ", /*TILE_R,C*/" << tileShape[0] << ", " << tileShape[1] << ", /*BLOCK_DIM_X,Y,Z*/ " << blockLoadOp.blockDimX() << ", " << blockLoadOp.blockDimY();
        os << ", " << blockLoadOp.blockDimZ() << ", " << getMemSpaceEnum(srcMemSpace) << ", " << getMemSpaceEnum(destMemSpace) << ", ";
        RETURN_IF_FAILED(printVectorType(elementType, stride));
        os << ">(\n"
           << blockThreadId << ", (";
        RETURN_IF_FAILED(printer->printType(elementType));
        os << "*)" << src << ", " << srcOffsetRows << ", " << srcOffsetCols << ", " << accessMapName << ", (";
        RETURN_IF_FAILED(printer->printType(elementType));
        os << "*)" << dst << ")";

        return success();
    }

    LogicalResult AcceraDialectCppPrinter::printDialectOperation(
        Operation* op,
        bool* /*skipped*/,
        bool* consumed)
    {
        auto handler = [&, this](auto op_) {
            THROW_IF_FAILED(printOp(op_));
            *consumed = true;
        };

        TypeSwitch<Operation*>(op)
            .Case<vir::MMAAllocSyncOp>(handler)
            //.Case<vir::MMAFillSyncOp>(handler)
            .Case<vir::MMALoadSyncOp>(handler)
            .Case<vir::MMAComputeSyncOp>(handler)
            .Case<vir::MMAStoreSyncOp>(handler)
            .Case<vir::GPUBlockCacheOp>(handler)
            .Case<vir::CallOp>(handler)
            .Case<vir::ReturnOp>(handler)
            .Case<vir::WarpIdOp>(handler)
            .Default([&](Operation*) { *consumed = false; });

        return success();
    }

    LogicalResult AcceraDialectCppPrinter::printIntrinsicCallOp(Operation* callOp,
                                                                Operation* defFuncOp,
                                                                bool* consumed)
    {
        *consumed = false;

        llvm_unreachable("not valid mma kernel");
    }

    LogicalResult AcceraDialectCppPrinter::printPrologue()
    {

        return success();
    }

    LogicalResult AcceraDialectCppPrinter::printEpilogue()
    {
        // TODO: add a cmdline option to skip generating host launch func
        return success();
    }

    LogicalResult AcceraDialectCppPrinter::runPrePrintingPasses(Operation* op)
    {
        auto walkResult = op->walk([&](Operation* subOp) {
            if (auto funcOp = dyn_cast<FuncOp>(subOp))
            {
                CudaKernels.push_back(funcOp);
            }
            else if (auto affineForOp = dyn_cast<AffineForOp>(subOp))
            {
                // FIXME: This is a temprary heuristic. We may want to have an Argo pass
                // that performs some analysis and tags a loop as "unroll-able".
                if (hasAttrs(affineForOp->getAttrs(),
                             { argo::ArgoParallelForAttributeName }))
                {
                    state.unrolledForOps.insert(subOp);
                }
            }

            return WalkResult::advance();
        });

        return walkResult.wasInterrupted() ? failure() : success();
    }

} // namespace cpp_printer
} // namespace mlir
