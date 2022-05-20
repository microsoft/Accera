////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GpuDialectCppPrinter.h"
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
        RETURN_IF_FAILED(printGPUIndexType());

        os << " " << idx << " = ";

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
        RETURN_IF_FAILED(printGPUIndexType());

        os << " " << idx << " = ";

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
        RETURN_IF_FAILED(printGPUIndexType());

        os << " " << idx << " = ";

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
        RETURN_IF_FAILED(printGPUIndexType());

        os << " " << idx << " = ";

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

    LogicalResult GpuDialectCppPrinter::printOp(gpu::SubgroupMmaConstantMatrixOp constantMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return constantMatrixOp.emitError("non-cuda version is not supported.");
        }

        auto mmaMatrix = constantMatrixOp.res().getType().cast<MMAMatrixType>();
        auto fragName = state.nameState.getOrCreateName(
            constantMatrixOp.res(), SSANameState::SSANameKind::Variable, "mmaMatrix_");
        auto val = state.nameState.getOrCreateName(
            constantMatrixOp.value(), SSANameState::SSANameKind::Variable, "mmaFillValue_");
        RETURN_IF_FAILED(printAccType(mmaMatrix));
        os << " " << fragName << ";\n";
        os << "wmma::fill_fragment(" << fragName << ", " << val << ")";
        return success();
    }

    int64_t inferM(int64_t K, int64_t N)
    {
        if (N == 16 && K == 16)
            return 16;

        return {};
    }

    int64_t inferN(int64_t M, int64_t K)
    {
        if (M == 16 && K == 16)
            return 16;

        return {};
    }

    int64_t inferK(int64_t M, int64_t N)
    {
        if (M == 16 && N == 16)
            return 16;

        return {};
    }

    LogicalResult GpuDialectCppPrinter::printAccType(const MMAMatrixType& mmaMatrix)
    {
        auto matrixShape = mmaMatrix.getShape();
        auto m = matrixShape[0];
        auto n = matrixShape[1];
        auto k = inferK(m, n);
        os << "wmma::fragment<wmma::accumulator, ";
        os << m << ", " << n << ", " << k << ", ";

        RETURN_IF_FAILED(printer->printType(mmaMatrix.getElementType()));
        os << ">";
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(SubgroupMmaLoadMatrixOp loadMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return loadMatrixOp.emitError("non-cuda version is not supported.");
        }

        auto fragName = state.nameState.getOrCreateName(loadMatrixOp.res(), SSANameState::SSANameKind::Variable, "mmaMatrix_");
        auto rowIdx = state.nameState.getOrCreateName(loadMatrixOp.indices()[0], SSANameState::SSANameKind::Variable, "row_");
        auto colIdx = state.nameState.getOrCreateName(loadMatrixOp.indices()[1], SSANameState::SSANameKind::Variable, "col_");
        auto mmaMatrix = loadMatrixOp.res().getType().cast<MMAMatrixType>();
        auto leadingDim = loadMatrixOp.leadDimension();
        int64_t offset;
        SmallVector<int64_t, 2> strides;
        mlir::getStridesAndOffset(loadMatrixOp.srcMemref().getType().cast<MemRefType>(), strides, offset);

        // TODO: have a separate op for declaring a fragment
        if (mmaMatrix.getOperand() == "COp")
        {
            RETURN_IF_FAILED(printAccType(mmaMatrix));
        }
        else
        {
            int m{};
            int n{};
            int k{};
            auto matrixShape = mmaMatrix.getShape();
            os << "wmma::fragment<";
            if (mmaMatrix.getOperand() == "AOp")
            {
                os << "wmma::matrix_a, ";
                m = matrixShape[0];
                k = matrixShape[1];
                n = inferN(m, k);
            }
            else if (mmaMatrix.getOperand() == "BOp")
            {
                os << "wmma::matrix_b, ";
                k = matrixShape[0];
                n = matrixShape[1];
                m = inferM(k, n);
            }
            else
            {
                os << "UNSUPPORTED_MATRIX, ";
                return loadMatrixOp.emitError("Unsupported matrix used for MMA.");
            }

            os << m << ", " << n << ", " << k << ", ";

            RETURN_IF_FAILED(printer->printType(mmaMatrix.getElementType()));
            if (leadingDim.isOneValue())
                os << ", wmma::col_major>";
            else
                os << ", wmma::row_major>";
        }
        os << " " << fragName << ";\n";
        os << "wmma::load_matrix_sync(" << fragName << ", ";
        os << state.nameState.getName(loadMatrixOp.srcMemref()) << " + ";

        // The col major matrix has been transposed (metadata only), so strides[0] should always be the proper leading dim,
        // and the leadingDim has the stride in the first dim (before transpose) which tells us the actual layout of the matrix
        if (leadingDim.isOneValue())
            os << colIdx << " * " << strides[0] << " + " << rowIdx;
        else
            os << rowIdx << " * " << strides[0] << " + " << colIdx;
        os << ", " << strides[0];
        if (mmaMatrix.getOperand() == "COp")
        {
            if (leadingDim.isOneValue())
                os << ", wmma::layout_t::mem_col_major";
            else
                os << ", wmma::layout_t::mem_row_major";
        }
        os << ")";
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(gpu::SubgroupMmaComputeOp computeMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return computeMatrixOp.emitError("non-cuda version is not supported.");
        }

        auto opA = state.nameState.getName(computeMatrixOp.opA());
        auto opB = state.nameState.getName(computeMatrixOp.opB());
        auto opC = state.nameState.getName(computeMatrixOp.opC());
        auto fragName = state.nameState.getOrCreateName(computeMatrixOp.res(), SSANameState::SSANameKind::Variable, "mmaMatrix_");
        auto mmaMatrix = computeMatrixOp.res().getType().cast<MMAMatrixType>();
        RETURN_IF_FAILED(printAccType(mmaMatrix));
        os << " " << fragName << ";\n";
        os << "wmma::mma_sync(" << fragName << ", " << opA << ", " << opB << ", " << opC << ")";
        return success();
    }

    LogicalResult GpuDialectCppPrinter::printOp(gpu::SubgroupMmaStoreMatrixOp storeMatrixOp)
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            return storeMatrixOp.emitError("non-cuda version is not supported.");
        }

        auto rowIdx = state.nameState.getOrCreateName(storeMatrixOp.indices()[0], SSANameState::SSANameKind::Variable, "row_");
        auto colIdx = state.nameState.getOrCreateName(storeMatrixOp.indices()[1], SSANameState::SSANameKind::Variable, "col_");
        auto fragName = state.nameState.getName(storeMatrixOp.src());
        auto leadingDim = storeMatrixOp.leadDimension();

        int64_t offset;
        SmallVector<int64_t, 2> strides;
        mlir::getStridesAndOffset(storeMatrixOp.dstMemref().getType().cast<MemRefType>(), strides, offset);
        auto destMemref = state.nameState.getName(storeMatrixOp.dstMemref());
        os << "wmma::store_matrix_sync(" << destMemref << " + ";

        // The col major matrix has been transposed (metadata only), so strides[0] should always be the proper leading dim,
        // and the leadingDim has the stride in the first dim (before transpose) which tells us the actual layout of the matrix
        if (leadingDim.isOneValue())
            os << colIdx << " * " << strides[0] << " + " << rowIdx << ", " << fragName << ", " << strides[0] << ", wmma::layout_t::mem_col_major)";
        else
            os << rowIdx << " * " << strides[0] << " + " << colIdx << ", " << fragName << ", " << strides[0] << ", wmma::layout_t::mem_row_major)";
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
            .Case<SubgroupMmaConstantMatrixOp>(handler)
            .Case<SubgroupMmaLoadMatrixOp>(handler)
            .Case<SubgroupMmaComputeOp>(handler)
            .Case<SubgroupMmaStoreMatrixOp>(handler)
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
                llvm::errs() << "GPU module detected, enabling CUDA runtime\n";
                _gpuModuleOps = llvm::to_vector<4>(potentialGpuOps);
            }
        }

        for (auto gpuOp : _gpuModuleOps)
        {
            auto execRuntime = accera::ir::util::ResolveExecutionRuntime(gpuOp);
            if (!execRuntime)
            {
                llvm::errs() << "Device functions must specify a runtime\n";
                return failure();
            }
            switch (*execRuntime)
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

using float16_t = _Float16;
using vhalfx2_t = float16_t __attribute__((ext_vector_type(2)));
using vhalfx4_t = float16_t __attribute__((ext_vector_type(4)));
using vhalfx8_t = float16_t __attribute__((ext_vector_type(8)));
using vhalfx16_t = float16_t __attribute__((ext_vector_type(16)));
using vhalfx32_t = float16_t __attribute__((ext_vector_type(32)));
using vhalfx64_t = float16_t __attribute__((ext_vector_type(64)));
using vfloatx2_t = float __attribute__((ext_vector_type(2)));
using vfloatx4_t = float __attribute__((ext_vector_type(4)));
using vfloatx8_t = float __attribute__((ext_vector_type(8)));
using vfloatx16_t = float __attribute__((ext_vector_type(16)));
using vfloatx32_t = float __attribute__((ext_vector_type(32)));
using vfloatx64_t = float __attribute__((ext_vector_type(64)));
using int8_t = char;
using int16_t = short;
using uint8_t = unsigned char;
using uint16_t = unsigned short;
namespace std {
using ::uint8_t;
using ::uint16_t;
using ::int8_t;
using ::int16_t;
}

)CUDA";
        }
        else if (state.hasRuntime(Runtime::CUDA))
        {
            os << R"CUDA(
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

using float16_t = __half;
using uint32_t = unsigned int;

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
