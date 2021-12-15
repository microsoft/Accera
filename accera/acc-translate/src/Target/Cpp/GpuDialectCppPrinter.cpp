//===- GpuDialectCppPrinter.cpp - GPU Dialect Printer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "GpuDialectCppPrinter.h"

using namespace mlir::gpu;

namespace mlir {
namespace cpp_printer {

LogicalResult GpuDialectCppPrinter::printBarrierOp(BarrierOp barrierOp) {
  if (!isCuda) {
    return barrierOp.emitError("non-cuda version is not supported yet");
  }

  os << "__syncthreads()";
  return success();
}

LogicalResult GpuDialectCppPrinter::printGridDimOp(GridDimOp gdimOp) {
  if (!isCuda) {
    return gdimOp.emitError("non-cuda version is not supported yet");
  }

  auto idx = state.nameState.getOrCreateName(
      gdimOp.getResult(), SSANameState::SSANameKind::Variable);
  os << "int " << idx << " = gridDim." << gdimOp.dimension();
  return success();
}

LogicalResult GpuDialectCppPrinter::printBlockDimOp(BlockDimOp bdimOp) {
  if (!isCuda) {
    return bdimOp.emitError("non-cuda version is not supported yet");
  }

  auto idx = state.nameState.getOrCreateName(
      bdimOp.getResult(), SSANameState::SSANameKind::Variable);
  os << "int " << idx << " = blockDim." << bdimOp.dimension();
  return success();
}

LogicalResult GpuDialectCppPrinter::printBlockIdOp(BlockIdOp bidOp) {
  if (!isCuda) {
    return bidOp.emitError("non-cuda version is not supported yet");
  }

  auto idx = state.nameState.getOrCreateName(
      bidOp.getResult(), SSANameState::SSANameKind::Variable);
  os << "int " << idx << " = blockIdx." << bidOp.dimension();
  return success();
}

LogicalResult GpuDialectCppPrinter::printThreadIdOp(ThreadIdOp tidOp) {
  if (!isCuda) {
    return tidOp.emitError("non-cuda version is not supported yet");
  }

  auto idx = state.nameState.getOrCreateName(
      tidOp.getResult(), SSANameState::SSANameKind::Variable);
  os << "int " << idx << " = threadIdx." << tidOp.dimension();
  return success();
}

LogicalResult GpuDialectCppPrinter::printDialectOperation(Operation *op,
                                                          bool * /*skipped*/,
                                                          bool *consumed) {
  *consumed = true;

  if (auto barrierOp = dyn_cast<BarrierOp>(op))
    return printBarrierOp(barrierOp);

  if (auto gdimOp = dyn_cast<GridDimOp>(op))
    return printGridDimOp(gdimOp);

  if (auto bdimOp = dyn_cast<BlockDimOp>(op))
    return printBlockDimOp(bdimOp);

  if (auto bidOp = dyn_cast<BlockIdOp>(op))
    return printBlockIdOp(bidOp);

  if (auto tidOp = dyn_cast<ThreadIdOp>(op))
    return printThreadIdOp(tidOp);

  *consumed = false;
  return success();
}

LogicalResult GpuDialectCppPrinter::printGpuFP16VectorType(VectorType vecType,
                                                           StringRef vecVar) {
  if (vecType.getNumDynamicDims()) {
    os << "<<VectorType with dynamic dims is not supported yet>>";
    return failure();
  }

  auto rank = vecType.getRank();
  if (rank == 0) {
    os << "<<zero-ranked Vectortype is not supported yet>>";
    return failure();
  }

  auto shape = vecType.getShape();
  if (shape[rank - 1] % 2) {
    os << "<<can't be represented by " << printer->float16VecT() << ">>";
    return failure();
  }

  os << printer->float16VecT() << " " << vecVar;
  int i = 0;
  for (; i < rank - 1; i++) {
    os << "[" << shape[i] << "]";
  }
  os << "[" << shape[i] / 2 << "]";
  return success();
}

LogicalResult GpuDialectCppPrinter::printVectorTypeArrayDecl(VectorType vecType,
                                                             StringRef vecVar) {
  assert(isCuda && "not for cuda?");

  auto elemType = vecType.getElementType();
  // TODO: support more vector types
  if (elemType.isa<Float16Type>()) {
    return printGpuFP16VectorType(vecType, vecVar);
  } else {
    os << "<<only support fp16 vec type>>";
    return failure();
  }
}

} // namespace cpp_printer
} // namespace mlir
