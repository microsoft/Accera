////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GpuDialectCppPrinter.h"
#include <llvm/ADT/None.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LogicalResult.h>

#include <ir/include/IRUtil.h>

using namespace mlir::gpu;

namespace mlir
{
namespace cpp_printer
{

    LogicalResult GpuDialectCppPrinter::printBarrierOp(BarrierOp barrierOp)
    {
        if (!isCuda)
        {
            return barrierOp.emitError("non-cuda version is not supported yet");
        }

        os << "__syncthreads()";
        return success();
    }

    static int dimIndexToInteger(llvm::StringRef dim)
    {
        if (dim == "x")
        {
            return 0;
        }
        else if (dim == "y")
        {
            return 1;
        }
        else if (dim == "z")
        {
            return 2;
        }
        else
        {
            return -1;
        }
    }

    static Optional<uint64_t> getGridDim(Operation* op, llvm::StringRef dim)
    {

        if (auto fn = op->getParentOfType<FuncOp>())
        {
            if (!fn->hasAttrOfType<ArrayAttr>("gridSize"))
            {
                return llvm::None;
            }
            auto arrayAttr = accera::ir::util::ArrayAttrToVector<mlir::IntegerAttr>(fn->getAttrOfType<ArrayAttr>("gridSize"));
            auto idx = dimIndexToInteger(dim);
            if (idx == -1) return llvm::None;
            return arrayAttr[idx].getInt();
        }
        return llvm::None;
    }

    static Optional<uint64_t> getBlockDim(Operation* op, llvm::StringRef dim)
    {
        if (auto fn = op->getParentOfType<FuncOp>())
        {
            if (!fn->hasAttrOfType<ArrayAttr>("blockSize"))
            {
                return llvm::None;
            }
            auto arrayAttr = accera::ir::util::ArrayAttrToVector<mlir::IntegerAttr>(fn->getAttrOfType<ArrayAttr>("blockSize"));
            auto idx = dimIndexToInteger(dim);
            if (idx == -1) return llvm::None;
            return arrayAttr[idx].getInt();
        }
        return llvm::None;
    }

    LogicalResult GpuDialectCppPrinter::printGridDimOp(GridDimOp gridDimOp)
    {
        if (!isCuda)
        {
            return gridDimOp.emitError("non-cuda version is not supported yet");
        }

        const std::string varPrefix = std::string("gridDim_") + gridDimOp.dimension().str() + "_";
        auto idx = state.nameState.getOrCreateName(
            gridDimOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        os << "const uint " << idx << " = ";
        if (auto c = getGridDim(gridDimOp, gridDimOp.dimension()); c)
        {
            os << c.getValue();
        }
        else
        {
            os << "gridDim." << gridDimOp.dimension();
        }
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printBlockDimOp(BlockDimOp blockDimOp)
    {
        if (!isCuda)
        {
            return blockDimOp.emitError("non-cuda version is not supported yet");
        }

        const std::string varPrefix = std::string("blockDim_") + blockDimOp.dimension().str() + "_";
        auto idx = state.nameState.getOrCreateName(
            blockDimOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        os << "const uint " << idx << " = ";
        if (auto c = getBlockDim(blockDimOp, blockDimOp.dimension()); c)
        {
            os << c.getValue();
        }
        else
        {
            os << "blockDim." << blockDimOp.dimension();
        }
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printBlockIdOp(BlockIdOp bidOp)
    {
        if (!isCuda)
        {
            return bidOp.emitError("non-cuda version is not supported yet");
        }

        const std::string varPrefix = std::string("blockIdx_") + bidOp.dimension().str() + "_";
        auto idx = state.nameState.getOrCreateName(
            bidOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        os << "const uint " << idx << " = ";
        if (auto c = getGridDim(bidOp, bidOp.dimension()); c)
        {
            os << "(blockIdx." << bidOp.dimension() << "%" << c.getValue() << ")";
        }
        else
        {

            os << "blockIdx." << bidOp.dimension();
        }
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printThreadIdOp(ThreadIdOp tidOp)
    {
        if (!isCuda)
        {
            return tidOp.emitError("non-cuda version is not supported yet");
        }

        const std::string varPrefix = std::string("threadIdx_") + tidOp.dimension().str() + "_";
        auto idx = state.nameState.getOrCreateName(
            tidOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        os << "const uint " << idx << " = ";
        if (auto c = getBlockDim(tidOp, tidOp.dimension()); c)
        {
            os << "(threadIdx." << tidOp.dimension() << "%" << c.getValue() << ")";
        }
        else
        {
            os << "threadIdx." << tidOp.dimension();
        }
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printDialectOperation(Operation* op,
                                                              bool* /*skipped*/,
                                                              bool* consumed)
    {
        *consumed = true;

        if (auto barrierOp = dyn_cast<BarrierOp>(op))
            return printBarrierOp(barrierOp);

        if (auto gridDimOp = dyn_cast<GridDimOp>(op))
            return printGridDimOp(gridDimOp);

        if (auto blockDimOp = dyn_cast<BlockDimOp>(op))
            return printBlockDimOp(blockDimOp);

        if (auto bidOp = dyn_cast<BlockIdOp>(op))
            return printBlockIdOp(bidOp);

        if (auto tidOp = dyn_cast<ThreadIdOp>(op))
            return printThreadIdOp(tidOp);

        *consumed = false;
        return success();
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
        assert(isCuda && "not for cuda?");

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

} // namespace cpp_printer
} // namespace mlir
