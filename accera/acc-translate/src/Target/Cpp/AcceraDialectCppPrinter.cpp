////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraDialectCppPrinter.h"

#include <ir/include/IRUtil.h>
#include <ir/include/argo/Utils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include "AMDGPU.h"
#include "NVGPU.h"

using namespace mlir::argo;

namespace mlir
{
namespace cpp_printer
{

    static bool isMFMAComputeOp(Operation* op)
    {
        return llvm::isa<accera::ir::value::MFMAComputeOp>(op);
    }
    LogicalResult AcceraDialectCppPrinter::printMFMAComputeOp(Operation* op)
    {
        accera::ir::value::MFMAComputeOp mfmaOp = mlir::dyn_cast_or_null<accera::ir::value::MFMAComputeOp>(op);
        assert(mfmaOp);
        auto accumInputTy = mfmaOp.opC().getType();
        auto accumOutputTy = mfmaOp.res().getType();
        auto funName = GetAMDMFMAOpName(accumOutputTy, accumInputTy);
        if (!funName)
        {
            return failure();
        }
        auto idx = state.nameState.getOrCreateName(
            mfmaOp.res(), SSANameState::SSANameKind::Variable);
        auto ty = mfmaOp.res().getType();
        if (auto memrefTy = ty.dyn_cast<MemRefType>())
        {
            ty = VectorType::get(memrefTy.getNumElements(), memrefTy.getElementType());
        }

        RETURN_IF_FAILED(printer->printType(ty));
        os << " " << idx << " = ";
        os << funName << "(";
        os << state.nameState.getName(mfmaOp.opA()) << ", ";
        os << state.nameState.getName(mfmaOp.opB()) << ", ";
        os << state.nameState.getName(mfmaOp.opC()) << ", ";
        os << "0, 0, 0";
        os << ")";
        return success();
    }

    LogicalResult AcceraDialectCppPrinter::printDialectOperation(
        Operation* op,
        bool* /*skipped*/,
        bool* consumed)
    {
        *consumed = true;
        if (isMFMAComputeOp(op))
            return printMFMAComputeOp(op);
        *consumed = false;
        return success();
    }

    LogicalResult AcceraDialectCppPrinter::printIntrinsicCallOp(Operation* callOp,
                                                                Operation* defFuncOp,
                                                                bool* consumed)
    {
        *consumed = false;

        llvm_unreachable("not valid mma kernel");
    }

    LogicalResult AcceraDialectCppPrinter::printHostLaunchFunc()
    {
        if (!isCuda)
            return success();

        auto numCudaKernels = CudaKernels.size();
        // FIXME: we only support a single cuda kernel at the moment
        if (numCudaKernels != 1)
        {
            os << "<<only a single CUDA kernel is supported>>";
            return failure();
        }

        FuncOp kernel = CudaKernels[0];

        int gridSizeX = 1, gridSizeY = 1, gridSizeZ = 1;
        int blockSizeX = 1, blockSizeY = 1, blockSizeZ = 1;

        for (const auto& attr : kernel->getAttrs())
        {
            if (attr.first == "gridSize")
            {
                auto arrayAttr = accera::ir::util::ArrayAttrToVector<mlir::IntegerAttr>(attr.second.dyn_cast<ArrayAttr>());
                gridSizeX = arrayAttr[0].getInt();
                gridSizeY = arrayAttr[1].getInt();
                gridSizeZ = arrayAttr[2].getInt();
            }
            else if (attr.first == "blockSize")
            {
                auto arrayAttr = accera::ir::util::ArrayAttrToVector<mlir::IntegerAttr>(attr.second.dyn_cast<ArrayAttr>());
                blockSizeX = arrayAttr[0].getInt();
                blockSizeY = arrayAttr[1].getInt();
                blockSizeZ = arrayAttr[2].getInt();
            }
        }

        os << "void launch_kernel(";

        [[maybe_unused]] auto numArgs = static_cast<int>(kernel.getNumArguments());
        SmallVector<std::string, 0> argNames;
        argNames.reserve(kernel.getNumArguments());

        bool failedArgs = false;
        int argIdx = 0;
        interleaveComma(kernel.getArguments(), os, [&](Value argVal) {
            // We are out of the scope of nameState's ScopedHashTableScope, so let's
            // make our own arg names
            std::string name = "arg" + std::to_string(argIdx++);
            argNames.push_back(name);

            Type argType = argVal.getType();
            if (auto memRefType = argType.dyn_cast<MemRefType>())
            {
                if (failed(printer->printDecayedArrayDeclaration(memRefType, name)))
                {
                    failedArgs = true;
                }
            }
            else
            {
                if (failed(printer->printType(argType)))
                {
                    failedArgs = true;
                }
                os << name;
            }
        });

        os << ") {\n";

        os << "dim3 gridSize(" << gridSizeX << ", " << gridSizeY << ", " << gridSizeZ
           << ");\n";
        os << "dim3 blockSize(" << blockSizeX << ", " << blockSizeY << ", "
           << blockSizeZ << ");\n";

        os << kernel.getName() << "<<<gridSize, blockSize>>>(";

        interleaveComma(argNames, os, [&](const std::string& name) { os << name; });
        os << ");\n";

        os << "}\n";
        return success();
    }

    LogicalResult AcceraDialectCppPrinter::printPrologue()
    {

        return success();
    }

    LogicalResult AcceraDialectCppPrinter::printEpilogue()
    {
        // TODO: add a cmdline option to skip generating host launch func
        RETURN_IF_FAILED(printHostLaunchFunc());
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
