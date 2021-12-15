//===- AcceraDialectCppPrinter.cpp - Argo Dialect Printer -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "AcceraDialectCppPrinter.h"
// #include "mlir/Dialect/Argo/Target/NVGPU.h"
// #include "mlir/Dialect/Argo/Utils/Utils.h"

#include <ir/include/argo/Utils.h>

#include "NVGPU.h"

using namespace mlir::argo;

namespace mlir {
namespace cpp_printer {

LogicalResult AcceraDialectCppPrinter::printDialectOperation(
    Operation * /*op*/, bool * /*skipped*/, bool * /*consumed*/) {
  return success();
}

void AcceraDialectCppPrinter::printMMAm8n8k4RowColFP32Def(StringRef kernelName) {
  os << "__device__\n";
  os << "void " << kernelName << "(float *D, half2 *arg_A0, half2 *arg_A1, "
     << "half2 *arg_B0, half2 *arg_B1, float C0, float C1, float C2, float C3, "
     << "float C4, float C5, float C6, float C7) {\n";
  os << "  unsigned const *A0 = reinterpret_cast<unsigned const *>(arg_A0);\n";
  os << "  unsigned const *A1 = reinterpret_cast<unsigned const *>(arg_A1);\n";
  os << "  unsigned const *B0 = reinterpret_cast<unsigned const *>(arg_B0);\n";
  os << "  unsigned const *B1 = reinterpret_cast<unsigned const *>(arg_B1);\n";
  os << "  asm volatile(\"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
     << "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, \"\n";
  os << "      \"{%12,%13,%14,%15,%16,%17,%18,%19};\\n\"\n";
  os << "      : \"=f\"(D[0]),\n";
  os << "        \"=f\"(D[1]),\n";
  os << "        \"=f\"(D[2]),\n";
  os << "        \"=f\"(D[3]),\n";
  os << "        \"=f\"(D[4]),\n";
  os << "        \"=f\"(D[5]),\n";
  os << "        \"=f\"(D[6]),\n";
  os << "        \"=f\"(D[7])\n";
  os << "      : \"r\"(A0[0]),\n";
  os << "        \"r\"(A1[0]),\n";
  os << "        \"r\"(B0[0]),\n";
  os << "        \"r\"(B1[0]),\n";
  os << "        \"f\"(C0),\n";
  os << "        \"f\"(C1),\n";
  os << "        \"f\"(C2),\n";
  os << "        \"f\"(C3),\n";
  os << "        \"f\"(C4),\n";
  os << "        \"f\"(C5),\n";
  os << "        \"f\"(C6),\n";
  os << "        \"f\"(C7)\n";
  os << "  );\n";
  os << "}\n";
}

LogicalResult AcceraDialectCppPrinter::printMMAm8n8k4RowColFP32Call(
    Operation *callOp, FuncOp funcOp, StringRef kernelName) {
  int numResults = static_cast<int>(callOp->getNumResults());
  auto resName = state.nameState.getTempName();

  os << "float " << resName << "[" << numResults << "];\n";
  os << kernelName << "(" << resName << ", ";
  interleaveComma(callOp->getOperands(), os, [&](Value operand) {
    os << state.nameState.getName(operand);
  });
  os << ");\n";

  for (int idx = 0; idx < numResults; idx++) {
    auto resVal = callOp->getResult(idx);
    auto rVar = state.nameState.getOrCreateName(
        resVal, SSANameState::SSANameKind::Variable);
    RETURN_IF_FAILED(printer->printType(resVal.getType()));
    os << " " << rVar << " = " << resName << "[" << idx << "]";
    if (idx < numResults - 1)
      os << ";\n";
  }

  return success();
}

LogicalResult AcceraDialectCppPrinter::addMMAKernel(FuncOp funcOp) {
  StringAttr intrNameAttr =
      funcOp->getAttrOfType<StringAttr>(ArgoIntrNameAttributeName);
  if (!intrNameAttr) {
    return funcOp.emitError("no intr_name attribute");
  }

  StringAttr alayoutAttr =
      funcOp->getAttrOfType<StringAttr>(ArgoNVGPUMMAALayoutAttributeName);
  if (!alayoutAttr) {
    return funcOp.emitError("no alayout attribute");
  }

  StringAttr blayoutAttr =
      funcOp->getAttrOfType<StringAttr>(ArgoNVGPUMMABLayoutAttributeName);
  if (!blayoutAttr) {
    return funcOp.emitError("no blayout attribute");
  }

  auto intrName = intrNameAttr.getValue();
  auto alayout = alayoutAttr.getValue();
  auto blayout = blayoutAttr.getValue();

  FunctionType funcTy = funcOp.getType();
  int numInputs = funcTy.getNumInputs();
  assert(numInputs > 0);
  Type accType = funcTy.getInput(numInputs - 1);

  // FIXME: right now, we assume that we only have m8n8k4. Later Argo should a
  // tag to differentiate mma intrinsics

  // only support A-row-major and B-col-major mma
  if (intrName == "nvvm.mma.sync" && alayout == "row" && blayout == "col" &&
      accType.isa<FloatType>()) {
    FuncOpToMMAKernel.try_emplace(funcOp, MMAKernelKind::m8n8k4RowColfp32);
  } else {
    return funcOp.emitError("unsupported mma layout: ")
           << alayout << ", " << blayout;
  }

  return success();
}

// Rationale: we chose to print NV's mma from Argo dialect printer instead of
// Nvvm dialect printer due to the following reasons:
//   * intrinsic attributes such as intr_name, alayout and blayout are
//     part of the Argo dialect and should be transparent to other dialects
//   * nvvm dialect has its own MmaOp, but it's too low-level in our
//     lowering-pipeline. If we chose to print mma instruction there, we would
//     lose many high-level constructs such as affine/scf, which would have
//     been lowered to more fine-grained LLVM-style controls.
LogicalResult AcceraDialectCppPrinter::printIntrinsicCallOp(Operation *callOp,
                                                          Operation *defFuncOp,
                                                          bool *consumed) {
  *consumed = false;

  FuncOp funcOp = cast<FuncOp>(defFuncOp);
  auto kIter = FuncOpToMMAKernel.find(funcOp);
  if (kIter == FuncOpToMMAKernel.end()) {
    return success();
  }

  *consumed = true;

  StringRef kernelName = getMMAKernelName(kIter->second);
  switch (kIter->second) {
  case MMAKernelKind::m8n8k4RowColfp32:
    return printMMAm8n8k4RowColFP32Call(callOp, funcOp, kernelName);
  default:
    return funcOp.emitError("unsupported mma kernel kind");
  }

  llvm_unreachable("not valid mma kernel");
}

LogicalResult AcceraDialectCppPrinter::printHostLaunchFunc() {
  if (!isCuda)
    return success();

  auto numCudaKernels = CudaKernels.size();
  // FIXME: we only support a single cuda kernel at the moment
  if (numCudaKernels != 1) {
    os << "<<only a single CUDA kernel is supported>>";
    return failure();
  }

  FuncOp kernel = CudaKernels[0];

  int gridSizeX = 1, gridSizeY = 1, gridSizeZ = 1;
  int blockSizeX = 1, blockSizeY = 1, blockSizeZ = 1;

  for (const auto &attr : kernel->getAttrs()) {
    if (attr.first == "gridSizeX") {
      gridSizeX = attr.second.cast<mlir::IntegerAttr>().getInt();
    } else if (attr.first == "gridSizeY") {
      gridSizeY = attr.second.cast<mlir::IntegerAttr>().getInt();
    } else if (attr.first == "gridSizeZ") {
      gridSizeZ = attr.second.cast<mlir::IntegerAttr>().getInt();
    } else if (attr.first == "blockSizeX") {
      blockSizeX = attr.second.cast<mlir::IntegerAttr>().getInt();
    } else if (attr.first == "blockSizeY") {
      blockSizeY = attr.second.cast<mlir::IntegerAttr>().getInt();
    } else if (attr.first == "blockSizeZ") {
      blockSizeZ = attr.second.cast<mlir::IntegerAttr>().getInt();
    }
  }

  os << "void launch_kernel(";

  int numArgs = static_cast<int>(kernel.getNumArguments());
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
    if (auto memRefType = argType.dyn_cast<MemRefType>()) {
      if (failed(printer->printDecayedArrayDeclaration(memRefType, name))) {
        failedArgs = true;
      }
    } else {
      if (failed(printer->printType(argType))) {
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

  interleaveComma(argNames, os, [&](const std::string &name) { os << name; });
  os << ");\n";

  os << "}\n";
  return success();
}

LogicalResult AcceraDialectCppPrinter::printPrologue() {
  for (auto &iter : FuncOpToMMAKernel) {
    StringRef kernelName = getMMAKernelName(iter.second);
    switch (iter.second) {
    case MMAKernelKind::m8n8k4RowColfp32:
      printMMAm8n8k4RowColFP32Def(kernelName);
      break;
    default:
      os << "<<unsupported mma kernel kind>>";
      return failure();
    }
  }

  return success();
}

LogicalResult AcceraDialectCppPrinter::printEpilogue() {
  // TODO: add a cmdline option to skip generating host launch func
  RETURN_IF_FAILED(printHostLaunchFunc());
  return success();
}

LogicalResult AcceraDialectCppPrinter::runPrePrintingPasses(Operation *op) {
  MMAKernelNames.resize(static_cast<unsigned>(MMAKernelKind::InvalidKernel));
  addMMAKernelName(MMAKernelKind::m8n8k4RowColfp32,
                   "argo_mma_m8n8k4RowColfp32");

  auto walkResult = op->walk([&](Operation *subOp) {
    if (auto funcOp = dyn_cast<FuncOp>(subOp)) {
      if (hasAttrs(funcOp->getAttrs(), {argo::ArgoIntrinsicAttributeName})) {
        state.skippedOps.insert(subOp);
        state.intrinsicDecls.insert(subOp);
        if (failed(addMMAKernel(funcOp))) {
          return WalkResult::interrupt();
        }
      } else {
        // FIXME: currently Argo considers all functions to be cuda global
        // functions. Change printer's implementation later once Argo supports
        // __device__ functions.
        CudaKernels.push_back(funcOp);
      }
    } else if (auto affineForOp = dyn_cast<AffineForOp>(subOp)) {
      // FIXME: This is a temprary heuristic. We may want to have an Argo pass
      // that performs some analysis and tags a loop as "unroll-able".
      if (hasAttrs(affineForOp->getAttrs(),
                   {argo::ArgoParallelForAttributeName})) {
        state.unrolledForOps.insert(subOp);
      }
    }

    return WalkResult::advance();
  });

  return walkResult.wasInterrupted() ? failure() : success();
}

} // namespace cpp_printer
} // namespace mlir
