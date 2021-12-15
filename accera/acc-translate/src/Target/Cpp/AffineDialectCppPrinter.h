//===- AffineDialectCppPrinter.h - Affine Dialect Printer -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef AFFINE_DIALECT_CPP_PRINTER_H_
#define AFFINE_DIALECT_CPP_PRINTER_H_

#include <cassert>
#include <mlir/Dialect/Affine/IR/AffineOps.h>

#include "CppPrinter.h"

namespace mlir {
namespace cpp_printer {

struct AffineDialectCppPrinter : public DialectCppPrinter {
  AffineDialectCppPrinter(CppPrinter *printer)
      : DialectCppPrinter(printer), needAffineMemCpy(false) {}

  LogicalResult printPrologue() override;

  /// print Operation from Affine Dialect
  LogicalResult printDialectOperation(Operation *op, bool *skipped,
                                      bool *consumed) override;

  LogicalResult printAffineApplyOp(AffineApplyOp affineApplyOp);

  LogicalResult printAffineStoreOp(AffineStoreOp affineStoreOp);

  LogicalResult printAffineLoadOp(AffineLoadOp affineLoadOp);

  LogicalResult printAffineVectorLoadOp(AffineVectorLoadOp affineVecLoadOp);

  LogicalResult printAffineVectorStoreOp(AffineVectorStoreOp affineVecStoreOp);

  LogicalResult printAffineForOp(AffineForOp affineForOp);

  LogicalResult printAffineMapFunc(AffineMap map, StringRef funcName);

  LogicalResult printAffineExpr(AffineExpr affineExpr);

  LogicalResult runPrePrintingPasses(Operation *op) override;

  llvm::DenseMap<AffineMap, std::string> &getAffineMapToFuncBaseName() {
    return affineMapToFuncBaseName;
  }

  static constexpr const char *affineIdxTypeStr = "int64_t";

  static constexpr const char *affineMemCpyStr = "affine_memcpy";

  static constexpr const char *affineCeilDivStr = "affine_ceildiv";

  static constexpr const char *affineVecLoadFp16Str = "affine_vec_load_fp16";

  static constexpr const char *affineVecStoreFp16Str = "affine_vec_store_fp16";

  static constexpr const char *affineMapFuncPrefix = "affine_map_func_";

private:
  void printAffineMemCpy();

  void printAffineCeilDiv();

  void checkAffineMemCpyPass(Operation *op);

  void checkDeadAffineOpsPass(Operation *op);

  void collectAffineMapsPass(Operation *op);

  void printAffineMapResultIndices(AffineMap map,
                                   Operation::operand_range origIndices,
                                   llvm::SmallVector<StringRef, 4> &memIdxVars);

  LogicalResult
  printMemRefAccessPtr(Value memRef,
                       const llvm::SmallVector<StringRef, 4> &memIdxVars,
                       std::string &srcMemRefPtr);

  LogicalResult
  printMemRefAccessValue(Value memRef,
                         const llvm::SmallVector<StringRef, 4> &memIdxVars,
                         std::string &memRefVal);

  llvm::StringRef getFuncBaseName(AffineMap map) {
    auto iter = affineMapToFuncBaseName.find(map);
    assert(iter != affineMapToFuncBaseName.end());
    return iter->second;
  }

  bool needAffineMemCpy;

  // a map from an AffineMap to the base name of its corresponding function,
  // where the base name will be used to create affine_map_func for individual
  // indices.
  llvm::DenseMap<AffineMap, std::string> affineMapToFuncBaseName;
};

} // namespace cpp_printer
} // namespace mlir

#endif
