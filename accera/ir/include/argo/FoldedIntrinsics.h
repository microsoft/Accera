////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MLIR_DIALECT_ARGO_EDSC_FOLDEDINTRINSICS_H_
#define MLIR_DIALECT_ARGO_EDSC_FOLDEDINTRINSICS_H_

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/FoldUtils.h>

#include "Utils.h"

// FoldedValueBuilder is copied from Linalg.
// Currently, we only expose folded_std_constant_index.
// We can define more upon need.

namespace mlir {
namespace argo {
namespace intrinsics {

template <typename Op>
struct FoldedValueBuilder {
  // Builder-based
  template <typename... Args>
  FoldedValueBuilder(OpBuilder& b, Location loc, OperationFolder *folder, Args... args) {
    value = folder
                ? folder->create<Op>(b,
                                     loc,
                                     args...)
                : b.create<Op>(
                      loc, args...);
  }

  operator Value() { return value; }
  Value value;
};

template <>
struct FoldedValueBuilder<memref::DimOp> {
  // Builder-based
  template <typename... Args>
  FoldedValueBuilder(OpBuilder& b, Location loc, OperationFolder *folder, Args... args) {
    value = folder ? folder->create<memref::DimOp>(
                         b,
                         loc, args...)
                   : b.create<memref::DimOp>(
                         loc, args...);
  }

  FoldedValueBuilder(OpBuilder& b, Location loc, OperationFolder *folder, Value memrefOrTensor,
                     unsigned index) {
    int64_t dim = mlir::getMemrefOrTensorShape(memrefOrTensor, index);

    if (dim != ShapedType::kDynamicSize) {
      value = folder ? folder->create<ConstantIndexOp>(
                           b,
                           loc, dim)
                     : b
                           .create<ConstantIndexOp>(
                               loc, dim);
      return;
    }

    value =
        folder
            ? folder->create<memref::DimOp>(b,
                                    loc,
                                    memrefOrTensor, index)
            : b.create<memref::DimOp>(
                  loc, memrefOrTensor,
                  index);
  }

  operator Value() { return value; }
  Value value;
};

template <>
struct FoldedValueBuilder<AffineApplyOp> {
  FoldedValueBuilder(OpBuilder& b, Location loc, OperationFolder *folder, AffineMap map,
                     ValueRange mapOperands) {
    // support 1D for now
    if (map.getNumResults() == 1 && map.getNumInputs() <= 1) {
      AffineExpr expr = map.getResult(0);
      if (map.getNumInputs() == 0) {
        if (auto val = expr.dyn_cast<AffineConstantExpr>()) {
          value = FoldedValueBuilder<ConstantIndexOp>(b, loc, folder, val.getValue());
          return;
        }
      } else {
        // getNumInputs == 1
        if (expr.dyn_cast<AffineDimExpr>() ||
            expr.dyn_cast<AffineSymbolExpr>()) {
          value = mapOperands[0];
          return;
        }
      }
    }

    // roll-back case
    value =
        folder
            ? folder->create<AffineApplyOp>(
                  b,
                  loc, map, mapOperands)
            : b.create<AffineApplyOp>(
                  loc, map, mapOperands);
  }

  operator Value() { return value; }
  Value value;
};

using folded_std_constant_index = FoldedValueBuilder<ConstantIndexOp>;
using folded_std_dim = FoldedValueBuilder<memref::DimOp>;
using folded_std_addi = FoldedValueBuilder<AddIOp>;
using folded_std_muli = FoldedValueBuilder<MulIOp>;
using folded_std_subi = FoldedValueBuilder<SubIOp>;
using folded_std_diviu = FoldedValueBuilder<UnsignedDivIOp>;
using folded_affine_apply = FoldedValueBuilder<AffineApplyOp>;

} // namespace intrinsics
} // namespace argo
} // namespace mlir

#endif // MLIR_DIALECT_ARGO_EDSC_FOLDEDINTRINSICS_H_
