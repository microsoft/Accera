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

    LogicalResult printConstantMatrixOp(PrinterState& state, CppPrinter* printer, Type elementType, const std::tuple<int, int, int>& matrixShape, Value dest, Value value);
    LogicalResult printLoadMatrixOp(PrinterState& state, CppPrinter* printer, Type elementType, const std::tuple<int, int, int>& matrixShape, Value src, Value dest, vir::MMAOperandType operandType, std::pair<Value, Value> rowcol, bool rowMajor);
    LogicalResult printComputeMatrixOp(PrinterState& state, CppPrinter* printer, Type elementType, const std::tuple<int, int, int>& matrixShape, Value A, Value B, Value C, Value D);
    LogicalResult printStoreMatrixOp(PrinterState& state, CppPrinter* printer, Value src, Value dest, std::pair<Value, Value> rowcol);
} // namespace cpp_printer
} // namespace mlir

#endif
