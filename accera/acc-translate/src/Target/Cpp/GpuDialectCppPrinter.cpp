////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GpuDialectCppPrinter.h"
#include <llvm/ADT/None.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LogicalResult.h>

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

    static int dimIndexToInteger(llvm::StringRef dim)
    {
        return StringSwitch<int>(dim)
            .Case("x", 0)
            .Case("y", 1)
            .Case("z", 2)
            .Default(-1);
    }

    static Optional<uint64_t> getGridDim(Operation* op, llvm::StringRef dim)
    {

        if (auto fn = op->getParentOfType<FuncOp>())
        {
            if (!fn->hasAttrOfType<ArrayAttr>("gridSize"))
            {
                return llvm::None;
            }
            auto arrayAttr = utilir::ArrayAttrToVector<mlir::IntegerAttr>(fn->getAttrOfType<ArrayAttr>("gridSize"));
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
            auto arrayAttr = utilir::ArrayAttrToVector<mlir::IntegerAttr>(fn->getAttrOfType<ArrayAttr>("blockSize"));
            auto idx = dimIndexToInteger(dim);
            if (idx == -1) return llvm::None;
            return arrayAttr[idx].getInt();
        }
        return llvm::None;
    }

    LogicalResult GpuDialectCppPrinter::printOp(GridDimOp gridDimOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return gridDimOp.emitError("non-cuda version is not supported yet");
        }

        const std::string varPrefix = std::string("gridDim_") + gridDimOp.dimension().str() + "_";
        auto idx = state.nameState.getOrCreateName(
            gridDimOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        os << "const unsigned int " << idx << " = ";
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

    LogicalResult GpuDialectCppPrinter::printOp(BlockDimOp blockDimOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return blockDimOp.emitError("non-cuda version is not supported yet");
        }

        const std::string varPrefix = std::string("blockDim_") + blockDimOp.dimension().str() + "_";
        auto idx = state.nameState.getOrCreateName(
            blockDimOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        os << "const unsigned int " << idx << " = ";
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

    LogicalResult GpuDialectCppPrinter::printOp(BlockIdOp bidOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return bidOp.emitError("non-cuda version is not supported yet");
        }

        const std::string varPrefix = std::string("blockIdx_") + bidOp.dimension().str() + "_";
        auto idx = state.nameState.getOrCreateName(
            bidOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        os << "const unsigned int " << idx << " = ";
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

    LogicalResult GpuDialectCppPrinter::printOp(ThreadIdOp tidOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return tidOp.emitError("non-cuda version is not supported yet");
        }

        const std::string varPrefix = std::string("threadIdx_") + tidOp.dimension().str() + "_";
        auto idx = state.nameState.getOrCreateName(
            tidOp.getResult(), SSANameState::SSANameKind::Variable, varPrefix);
        os << "const unsigned int " << idx << " = ";
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

        auto handler = [&, this](auto op_) {
            printOp(op_);
            *consumed = true;
        };

        TypeSwitch<Operation*>(op)
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
            .Case<ThreadIdOp>(handler)
            .Default([&](Operation*) { *consumed = false; });

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
                if (llvm::hasSingleElement(potentialGpuOps))
                {
                    state.setRuntime(Runtime::CUDA); // TODO: detect from the module

                    llvm::errs() << "GPU module detected, enabling CUDA runtime\n";
                    _gpuModuleOp = *potentialGpuOps.begin();
                }
                else
                {
                    return op->emitError("Multiple GPU modules are not currently supported");
                }
            }
        }

        return success();
    }

    LogicalResult GpuDialectCppPrinter::printHeaderFiles()
    {
        if (state.hasRuntime(Runtime::CUDA))
        {
            os << R"CUDA(

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
using vhalf = __fp16;
using vfloatx2_t = float __attribute__((ext_vector_type(2)));
using vfloatx4_t = float __attribute__((ext_vector_type(4)));
using vfloatx8_t = float __attribute__((ext_vector_type(8)));
using vfloatx16_t = float __attribute__((ext_vector_type(16)));
using vhalfx2_t = vhalf __attribute__((ext_vector_type(2)));
using vhalfx4_t = vhalf __attribute__((ext_vector_type(4)));
using vhalfx8_t = vhalf __attribute__((ext_vector_type(8)));
using vhalfx16_t = vhalf __attribute__((ext_vector_type(16)));
#elif defined(__CUDA__)
#include "cuda_fp16.h"
#endif // !defined(__HIP_PLATFORM_AMD__)

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
        if (execRuntime && (execRuntime != vir::ExecutionRuntime::CUDA &&
                            execRuntime != vir::ExecutionRuntime::ROCM &&
                            // TODO: ugh. remove
                            execRuntime != vir::ExecutionRuntime::DEFAULT))
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
        assert(gpuModuleOp == _gpuModuleOp);

#if 0
        for (auto funcOp : gpuModuleOp.getOps<GPUFuncOp>())
        {
            // TODO: We should probably not be printing the functions directly
            // like this, but instead handing control back to the printer
            RETURN_IF_FAILED(printOp(funcOp));
        }
#endif // 0

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

        os << launchOp.getKernelName() << "<<<dim3(";
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
            assert(_gpuModuleOp);

            for (auto funcOp : _gpuModuleOp.getOps<gpu::GPUFuncOp>())
            {
                RETURN_IF_FAILED(printFunctionDeclaration(funcOp, /* trailingSemiColon */ true));
            }
        }

        return success();
    }

} // namespace cpp_printer
} // namespace mlir
