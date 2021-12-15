////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/InliningUtils.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/raw_ostream.h>

#ifndef __ACCERA__
#include "mlir/Dialect/Argo/IR/ArgoOps.h"
#include "mlir/Dialect/Argo/IR/ArgoTypes.h"
#else
#include "argo/ArgoOps.h"
#include "argo/ArgoTypes.h"
#endif // !__ACCERA__

using namespace mlir;
using namespace mlir::argo;

//===----------------------------------------------------------------------===//
// ArgoInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Argo
/// operations.
struct ArgoInlinerInterface : public DialectInlinerInterface
{
    using DialectInlinerInterface::DialectInlinerInterface;

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /// Returns true if the given region 'src' can be inlined into the region
    /// 'dest' that is attached to an operation registered to the current dialect.
    bool isLegalToInline(Region* dest, Region* src, bool, BlockAndValueMapping& valueMapping) const final
    {
        // Conservatively don't allow inlining into affine structures.
        return false;
    }

    /// Returns true if the given operation 'op', that is registered to this
    /// dialect, can be inlined into the given region, false otherwise.
    bool isLegalToInline(Operation* op, Region* region, bool, BlockAndValueMapping& valueMapping) const final
    {
        return true;
    }

    bool shouldAnalyzeRecursively(Operation* op) const final
    {
        // Make opaque an exception
        return !isa<OpaqueOp>(op);
    }

    virtual void handleTerminator(Operation* /*op*/, Block* /*newDest*/) const
    {
    }

    virtual void handleTerminator(Operation* /*op*/,
                                  ArrayRef<Value> /*valuesToReplace*/) const {}
};

void mlir::argo::ArgoDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#ifndef __ACCERA__
#include "mlir/Dialect/Argo/IR/ArgoOps.cpp.inc"
#else
#include "argo/ArgoOps.cpp.inc"
#endif // !__ACCERA__
        >();

    addInterfaces<ArgoInlinerInterface>();
}
