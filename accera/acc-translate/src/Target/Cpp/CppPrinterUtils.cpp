////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Captain Jack Sparrow
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CppPrinterUtils.h"
#include "AffineDialectCppPrinter.h"
#include <mlir/Dialect/GPU/GPUDialect.h>

namespace mlir
{
namespace cpp_printer
{
    bool isPrivateOrWorkgroupMemSpace(unsigned memspace)
    {
        return (memspace == gpu::GPUDialect::getPrivateAddressSpace()) ||
               (memspace == gpu::GPUDialect::getWorkgroupAddressSpace());
    }

    int getIntTypeBitCount(int width)
    {
        int bitCount = -1;
        if (width <= 8)
        {
            bitCount = 8;
        }
        else if (width <= 16)
        {
            bitCount = 16;
        }
        else if (width <= 32)
        {
            bitCount = 32;
        }
        else if (width <= 64)
        {
            bitCount = 64;
        }
        else
        {
        }
        return bitCount;
    }

    std::string getLayout(const bool rowMajor)
    {
        if (rowMajor)
            return "row_major";

        return "col_major";
    }

    std::string getMmaLayout(const std::string& mmaNamespace, const bool rowMajor)
    {
        return mmaNamespace + "::layout_t::mem_" + getLayout(rowMajor);
    }

    int64_t getLeadingDim(MemRefType memRefType, const bool sharedMem, const bool rowMajor)
    {
        if (sharedMem)
        {
            auto shape = memRefType.getShape();
            for (int i = shape.size() - 1; i >= 0; --i)
            {
                if (shape[i] != 1)
                {
                    return shape[i];
                }
            }

            return 1;
        }

        int64_t offset;
        SmallVector<int64_t, 2> strides;
        (void)mlir::getStridesAndOffset(memRefType, strides, offset);
        return rowMajor ? strides.end()[-2] : strides.back();
    }

    std::string getMemrefAccessStr(CppPrinter* printer, const bool squareBracket, MemRefType memRefType, std::string memrefVar, mlir::Operation::operand_range indices)
    {
        auto indexOffsetStr = printer->getMemRefAccessOffset(squareBracket, memRefType, indices);
        std::string memrefStr;
        if (squareBracket)
        {
            memrefStr = "&" + memrefVar + indexOffsetStr;
        }
        else
        {
            memrefStr = memrefVar + " + " + indexOffsetStr;
        }

        return memrefStr;
    }

    std::string getWmmaNamespace(PrinterState& state)
    {
        if (state.hasRuntime(Runtime::ROCM))
            return "rocwmma";

        if (state.hasRuntime(Runtime::CUDA))
            return "wmma";

        return "";
    }

    std::string getFragmentEnum(PrinterState& state, const vir::MMAOperandType opType)
    {
        auto nsPrefix = getWmmaNamespace(state) + "::";
        switch (opType)
        {
        case vir::MMAOperandType::A:
            return nsPrefix + "matrix_a";
        case vir::MMAOperandType::B:
            return nsPrefix + "matrix_b";
        case vir::MMAOperandType::Acc:
            return nsPrefix + "accumulator";
        default:
            return "";
        }
    }

    Type GetROCMCastedOutputType(Type elementType)
    {
        // For FP16 or BF16 output, we need to load C in FP32 mode before passing to MFMA
        if (elementType.isF16() || elementType.isBF16())
            return FloatType::getF32(elementType.getContext());

        // For I8 output, we need to load C in I32 mode before passing to MFMA
        if (elementType.isInteger(8) || elementType.isInteger(16))
            return IntegerType::get(elementType.getContext(), 32);

        return elementType;
    }

    LogicalResult printFragmentType(PrinterState& state, CppPrinter* printer, Type elementType, const std::tuple<int, int, int>& matrixShape, const vir::MMAOperandType opType, const int totalBlocks, const int blocks, const bool rowMajor)
    {
        const auto ns = getWmmaNamespace(state);
        auto m = std::get<0>(matrixShape);
        auto n = std::get<1>(matrixShape);
        auto k = std::get<2>(matrixShape);
        auto&& os = printer->getOStream();
        os << ns << "::fragment<" << getFragmentEnum(state, opType) << ", ";
        os << m << ", " << n << ", " << k << ", ";

        if (state.hasRuntime(Runtime::ROCM))
        {
            if (opType == vir::MMAOperandType::Acc)
                elementType = GetROCMCastedOutputType(elementType);

            os << totalBlocks << ", " << blocks << ", ";
        }

        if (!state.hasRuntime(Runtime::ROCM) && opType != vir::MMAOperandType::Acc && elementType.isF32())
        {
            os << ns << "::precision::tf32";
        }
        else
        {
            RETURN_IF_FAILED(printer->printType(elementType));
        }

        if (opType != vir::MMAOperandType::Acc)
        {
            os << ", " << ns << "::" << getLayout(rowMajor);
        }
        os << ">";
        return success();
    }

    LogicalResult printMMAMatrixOp(PrinterState& state, CppPrinter* printer, Type elementType, const std::tuple<int, int, int>& matrixShape, Value val, const vir::MMAOperandType operandType, const int totalBlocks, const int blocks, const bool rowMajor)
    {
        auto fragName = state.nameState.getOrCreateName(val, SSANameState::SSANameKind::Variable, "mmaMatrix_");
        RETURN_IF_FAILED(printFragmentType(state, printer, elementType, matrixShape, operandType, totalBlocks, blocks, rowMajor));
        auto&& os = printer->getOStream();
        os << " " << fragName;
        return success();
    }

    LogicalResult printConstantMatrixOp(PrinterState& state, CppPrinter* printer, Value dest, Value value)
    {
        auto fragName = state.nameState.getName(dest);
        auto val = state.nameState.getOrCreateName(value, SSANameState::SSANameKind::Variable, "mmaFillValue_");
        auto&& os = printer->getOStream();
        os << getWmmaNamespace(state) << "::fill_fragment(" << fragName << ", " << val << ")";
        return success();
    }

    LogicalResult printFragmentOp(CppPrinter* printer, Type elementType, const StringRef& fragName, const vir::MMAFragmentOpType mmaFragmentOp, const StringRef& mmaFragmentArg)
    {
        auto&& os = printer->getOStream();
        auto fragDataName = fragName + "_data";

        os << "{\n"; // Additional scope required to avoid name collisions
        RETURN_IF_FAILED(printer->printType(elementType));
        os << "* " << fragDataName << " = reinterpret_cast<";
        RETURN_IF_FAILED(printer->printType(elementType));
        os << "*>(&" << fragName << ");\n";
        os << "for (int i = 0; i < sizeof(" << fragName << ") / sizeof(";
        RETURN_IF_FAILED(printer->printType(elementType));
        os << "); ++i) { ";
        switch (mmaFragmentOp)
        {
        case vir::MMAFragmentOpType::ReLU:
            os << "relu(" << fragDataName << "[i]);";
            break;
        case vir::MMAFragmentOpType::ReLU_NoConditional:
            os << "relu_no_conditional(" << fragDataName << "[i]);";
            break;
        case vir::MMAFragmentOpType::Set:
            os << "set(" << fragDataName << "[i], " << mmaFragmentArg << ");";
            break;
        case vir::MMAFragmentOpType::Scale:
            os << "scale(" << fragDataName << "[i], " << mmaFragmentArg << ");";
            break;
        default:
            break;
        }
        os << " }\n}\n";

        return success();
    }

    LogicalResult printLoadMatrixOp(PrinterState& state, CppPrinter* printer, Value src, Value dest, const vir::MMAOperandType operandType, mlir::Operation::operand_range indices, bool rowMajor, Value blockTid, const bool useStaticOffsets, const vir::MMAFragmentOpType mmaPrologueOp, Value mmaPrologueArg)
    {
        if (mmaPrologueOp == vir::MMAFragmentOpType::Set)
        {
            return printConstantMatrixOp(state, printer, dest, mmaPrologueArg);
        }

        const auto ns = getWmmaNamespace(state);
        auto&& os = printer->getOStream();
        int64_t offset;
        SmallVector<int64_t, 2> strides;
        auto memRefType = src.getType().cast<MemRefType>();
        auto memRefVarName = state.nameState.getName(src).str();
        RETURN_IF_FAILED(mlir::getStridesAndOffset(memRefType, strides, offset));
        if (operandType == vir::MMAOperandType::Acc)
        {
            rowMajor = strides.back() == 1;
        }

        const auto memspace = memRefType.getMemorySpaceAsInt();
        const auto sharedMem = isPrivateOrWorkgroupMemSpace(memspace);
        assert(!sharedMem || memRefType.getRank() >= 2); // sharedMem --> rank >= 2
        const auto ld = getLeadingDim(memRefType, sharedMem, rowMajor);

        const auto resName = state.nameState.getName(dest);
        auto srcMemrefStr = getMemrefAccessStr(printer, sharedMem, memRefType, memRefVarName, indices);
        os << ns << "::load_matrix_sync";
        if (state.hasRuntime(Runtime::ROCM))
        {
            os << "<";
            if (operandType == vir::MMAOperandType::Acc)
                os << useStaticOffsets << ", " << getMmaLayout(ns, rowMajor) << ", ";
            os << ld << ">(" << state.nameState.getName(blockTid) << ", " << resName << ", " << srcMemrefStr;
        }
        else
        {
            os << "(" << resName << ", " << srcMemrefStr << ", " << ld;
            if (operandType == vir::MMAOperandType::Acc)
                os << ", " << getMmaLayout(ns, rowMajor);
        }
        os << ")";

        if (mmaPrologueOp != vir::MMAFragmentOpType::None)
        {
            os << ";\n";

            auto dstElementType = dest.getType().cast<MemRefType>().getElementType();
            auto mmaPrologueArgName = state.nameState.getOrCreateName(mmaPrologueArg, SSANameState::SSANameKind::Variable, "mmaPrologueArg_");
            RETURN_IF_FAILED(printFragmentOp(printer, dstElementType, resName, mmaPrologueOp, mmaPrologueArgName));
        }

        return success();
    }

    LogicalResult printComputeMatrixOp(PrinterState& state, CppPrinter* printer, Value A, Value B, Value C, Value D, const int cbsz, const int abid, const int blgp)
    {
        auto opA = state.nameState.getName(A);
        auto opB = state.nameState.getName(B);
        auto opC = state.nameState.getName(C);
        auto fragName = state.nameState.getName(D);
        auto&& os = printer->getOStream();
        os << getWmmaNamespace(state) << "::mma_sync";
        if (state.hasRuntime(Runtime::ROCM))
        {
            os << "<" << cbsz << ", " << abid << ", " << blgp << ">";
        }
        os << "(" << fragName << ", " << opA << ", " << opB << ", " << opC << ")";
        return success();
    }

    LogicalResult printStoreMatrixOp(PrinterState& state, CppPrinter* printer, Value src, Value dest, mlir::Operation::operand_range indices, Value blockTid, const bool useStaticOffsets, const vir::MMAFragmentOpType mmaEpilogueOp, Value mmaEpilogueArg)
    {
        auto fragName = state.nameState.getName(src);

        int64_t offset;
        SmallVector<int64_t, 2> strides;
        auto memRefType = dest.getType().cast<MemRefType>();
        RETURN_IF_FAILED(mlir::getStridesAndOffset(memRefType, strides, offset));
        const bool rowMajor = strides.back() == 1;
        const auto memspace = memRefType.getMemorySpaceAsInt();
        const auto sharedMem = isPrivateOrWorkgroupMemSpace(memspace);
        assert(!sharedMem || memRefType.getRank() == 2); // sharedMem --> rank == 2
        const auto ld = getLeadingDim(memRefType, sharedMem, rowMajor);

        const auto ns = getWmmaNamespace(state);
        const auto dstMemrefStr = getMemrefAccessStr(printer, sharedMem, memRefType, state.nameState.getName(dest).str(), indices);
        auto&& os = printer->getOStream();

        if (mmaEpilogueOp != vir::MMAFragmentOpType::None)
        {
            auto srcElementType = src.getType().cast<MemRefType>().getElementType();
            auto mmaEpilogueArgName = state.nameState.getOrCreateName(mmaEpilogueArg, SSANameState::SSANameKind::Variable, "mmaEpilogueArg_");
            RETURN_IF_FAILED(printFragmentOp(printer, srcElementType, fragName, mmaEpilogueOp, mmaEpilogueArgName));
        }

        os << ns << "::store_matrix_sync";
        if (state.hasRuntime(Runtime::ROCM))
        {
            os << "<" << useStaticOffsets << ", " << getMmaLayout(ns, rowMajor) << ", " << ld << ">(" << state.nameState.getName(blockTid) << ", " << dstMemrefStr << ", " << fragName;
        }
        else
        {
            os << "(" << dstMemrefStr << ", " << fragName << ", " << ld << ", " << getMmaLayout(ns, rowMajor);
        }
        os << ")";
        return success();
    }
} // namespace cpp_printer
} // namespace mlir
