////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MLIR_DIALECT_ARGO_OPS_H_
#define MLIR_DIALECT_ARGO_OPS_H_

#include <mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/CopyOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/LLVM.h>

#include "ArgoTraits.h"
#include "ArgoTypes.h"

namespace mlir
{
namespace argo
{

    // Some literal
    constexpr const char* ArgoSpaceAttributeName = "space";
    constexpr const char* ArgoParallelForAttributeName = "argo_parallel_for";
    constexpr const char* ArgoAllReduceAttributeName = "argo_allreduce";
    constexpr const char* ArgoIntrinsicAttributeName = "argo_intrinsic";
    constexpr const char* ArgoIntrNameAttributeName = "intr_name";
    constexpr const char* ArgoIteratorMapAttributeName = "iterator_map";
    constexpr const char* ArgoIteratorVecWidthAttributeName = "vec_width";
    constexpr const char* ArgoIterationCountAttributeName = "iter_count";
    constexpr const char* ArgoIterationNoUnrollAttributeName = "iter_no_unroll";
    // TODO merge iter_count with iter_parm to a single list called iter_parms
    constexpr const char* ArgoIterationParmAttributeName = "iter_parm";
    constexpr const char* ArgoScatterLocalShapeAttributeName =
        "scattered_local_shape";

    /// Returns `maybeMap.get()` if `maybeMap` is set, otherwise returns the
    /// symbol-less identity map of `rank`.
    AffineMap extractOrIdentityMap(Optional<AffineMap> maybeMap, unsigned rank, mlir::MLIRContext* context);

#ifndef __ACCERA__
    #include "mlir/Dialect/Argo/IR/ArgoStructuredOpsInterfaces.h.inc"
#else
    #include "argo/ArgoStructuredOpsInterfaces.h.inc"
#endif
} // namespace argo

#define GET_OP_CLASSES
#ifndef __ACCERA__
#include "mlir/Dialect/Argo/IR/ArgoOps.h.inc"
#else
#include "argo/ArgoOps.h.inc"
#endif

#define GET_OP_CLASSES
#ifndef __ACCERA__
#include "mlir/Dialect/Argo/IR/ArgoStructuredOps.h.inc"
#else
#include "argo/ArgoStructuredOps.h.inc"
#endif
} // namespace mlir

#endif // MLIR_DIALECT_ARGO_OPS_H_
