////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <transforms/include/util/SnapshotUtilities.h>

#include <memory>

// fwd decls
namespace mlir
{
class MLIRContext;
class Pass;
template <typename OpT>
class OperationPass;

class RewritePatternSet;

} // namespace mlir

namespace accera
{
namespace ir::value
{
    class ValueFuncOp;
}
namespace transforms::loopnest
{
    struct LoopNestToValueFuncOptions
    {
        accera::transforms::IntraPassSnapshotOptions snapshotOptions;
        bool printLoops = false;
        bool printVecOpDetails = false;
    };

    void populateLoopnestToValueFuncPatterns(mlir::RewritePatternSet& patterns);
    std::unique_ptr<mlir::OperationPass<accera::ir::value::ValueFuncOp>> createLoopNestToValueFuncPass(const LoopNestToValueFuncOptions& options);
    std::unique_ptr<mlir::OperationPass<accera::ir::value::ValueFuncOp>> createLoopNestToValueFuncPass();
} // namespace transforms::loopnest
} // namespace accera
