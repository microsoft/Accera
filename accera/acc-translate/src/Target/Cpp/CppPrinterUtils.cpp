////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Captain Jack Sparrow
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CppPrinterUtils.h"
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

    std::string getMemrefAccessStr(const bool sharedMem, std::string memrefVar, std::string row, std::string col, const int64_t leadingDim, const bool rowMajor)
    {
        // When accessing from shared memory, no need to account for layout.
        if (sharedMem)
            return std::string("&") + memrefVar + "[" + row + "][" + col + "]";

        if (!rowMajor)
            std::swap(row, col);

        return memrefVar + " + " + row + " * " + std::to_string(leadingDim) + " + " + col;
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

    LogicalResult printFragmentType(PrinterState& state, CppPrinter* printer, Type elementType, const vir::MMAOperandType opType, const std::tuple<int, int, int>& matrixShape, const bool rowMajor = {})
    {
        const auto ns = getWmmaNamespace(state);
        auto m = std::get<0>(matrixShape);
        auto n = std::get<1>(matrixShape);
        auto k = std::get<2>(matrixShape);
        auto&& os = printer->getOStream();
        os << ns << "::fragment<" << getFragmentEnum(state, opType) << ", ";
        os << m << ", " << n << ", " << k << ", ";

        if (opType != vir::MMAOperandType::Acc && elementType.isF32())
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

    LogicalResult printConstantMatrixOp(PrinterState& state, CppPrinter* printer, Type elementType, const std::tuple<int, int, int>& matrixShape, Value dest, Value value)
    {
        auto fragName = state.nameState.getOrCreateName(dest, SSANameState::SSANameKind::Variable, "mmaMatrix_");
        auto val = state.nameState.getOrCreateName(value, SSANameState::SSANameKind::Variable, "mmaFillValue_");
        RETURN_IF_FAILED(printFragmentType(state, printer, elementType, vir::MMAOperandType::Acc, matrixShape));
        auto&& os = printer->getOStream();
        os << " " << fragName << ";\n";
        os << getWmmaNamespace(state) << "::fill_fragment(" << fragName << ", " << val << ")";
        return success();
    }

    LogicalResult printLoadMatrixOp(PrinterState& state, CppPrinter* printer, Type elementType, const std::tuple<int, int, int>& matrixShape, Value src, Value dest, vir::MMAOperandType operandType, std::pair<Value, Value> rowcol, const bool rowMajor)
    {
        const auto fragName = state.nameState.getOrCreateName(dest, SSANameState::SSANameKind::Variable, "mmaMatrix_");
        const auto rowIdx = state.nameState.getOrCreateName(rowcol.first, SSANameState::SSANameKind::Variable, "row_");
        const auto colIdx = state.nameState.getOrCreateName(rowcol.second, SSANameState::SSANameKind::Variable, "col_");
        const auto ns = getWmmaNamespace(state);
        auto&& os = printer->getOStream();
        int64_t offset;
        SmallVector<int64_t, 2> strides;
        auto memRefType = src.getType().cast<MemRefType>();
        RETURN_IF_FAILED(mlir::getStridesAndOffset(memRefType, strides, offset));
        const auto memspace = memRefType.getMemorySpaceAsInt();
        const auto sharedMem = isPrivateOrWorkgroupMemSpace(memspace);
        const auto ld = rowMajor || sharedMem ? strides[0] : strides[1];
        assert(!sharedMem || memRefType.getRank() == 2); // sharedMem --> rank == 2

        RETURN_IF_FAILED(printFragmentType(state, printer, elementType, operandType, matrixShape, rowMajor));
        os << " " << fragName << ";\n";
        os << ns << "::load_matrix_sync(" << fragName << ", ";
        os << getMemrefAccessStr(sharedMem, state.nameState.getName(src).str(), rowIdx.str(), colIdx.str(), ld, rowMajor) << ", " << ld;
        if (operandType == vir::MMAOperandType::Acc)
        {
            os << ", " << getMmaLayout(ns, rowMajor);
        }
        os << ")";
        return success();
    }

    LogicalResult printComputeMatrixOp(PrinterState& state, CppPrinter* printer, Type elementType, const std::tuple<int, int, int>& resultShape, Value A, Value B, Value C, Value D)
    {
        auto opA = state.nameState.getName(A);
        auto opB = state.nameState.getName(B);
        auto opC = state.nameState.getName(C);
        auto fragName = state.nameState.getOrCreateName(D, SSANameState::SSANameKind::Variable, "mmaMatrix_");
        RETURN_IF_FAILED(printFragmentType(state, printer, elementType, vir::MMAOperandType::Acc, resultShape));
        auto&& os = printer->getOStream();
        os << " " << fragName << ";\n";
        os << getWmmaNamespace(state) << "::mma_sync(" << fragName << ", " << opA << ", " << opB << ", " << opC << ")";
        return success();
    }

    LogicalResult printStoreMatrixOp(PrinterState& state, CppPrinter* printer, Value src, Value dest, std::pair<Value, Value> rowcol)
    {
        auto rowIdx = state.nameState.getOrCreateName(rowcol.first, SSANameState::SSANameKind::Variable, "row_");
        auto colIdx = state.nameState.getOrCreateName(rowcol.second, SSANameState::SSANameKind::Variable, "col_");
        auto fragName = state.nameState.getName(src);

        int64_t offset;
        SmallVector<int64_t, 2> strides;
        auto memRefType = dest.getType().cast<MemRefType>();
        RETURN_IF_FAILED(mlir::getStridesAndOffset(memRefType, strides, offset));
        const bool rowMajor = strides[1] == 1;
        const auto ld = rowMajor ? strides[0] : strides[1];
        const auto memspace = memRefType.getMemorySpaceAsInt();
        const auto sharedMem = isPrivateOrWorkgroupMemSpace(memspace);
        assert(!sharedMem || memRefType.getRank() == 2); // sharedMem --> rank == 2

        const auto ns = getWmmaNamespace(state);
        auto&& os = printer->getOStream();
        os << ns << "::store_matrix_sync(";
        os << getMemrefAccessStr(sharedMem, state.nameState.getName(dest).str(), rowIdx.str(), colIdx.str(), ld, rowMajor);
        os << ", " << fragName << ", " << ld << ", " << getMmaLayout(ns, rowMajor) << ")";
        return success();
    }
} // namespace cpp_printer
} // namespace mlir
