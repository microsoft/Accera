////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace mlir
{
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;
} // namespace mlir

namespace accera::transforms
{
void populateRangeResolutionPatterns(mlir::OwningRewritePatternList& patterns);
void populateScheduleScaffoldingPatterns(bool printLoops, mlir::OwningRewritePatternList& patterns);
void populateScheduledOperationsPatterns(mlir::OwningRewritePatternList& patterns);
void populateScheduleToValueRewritePatterns(mlir::OwningRewritePatternList& patterns);
void populateScheduleToValuePatterns(mlir::OwningRewritePatternList& patterns);
void populateUnlinkSymbolicIndicesPatterns(mlir::OwningRewritePatternList& patterns);
void populateSymIndexCleanupPatterns(mlir::OwningRewritePatternList& patterns);
void populateGPUIndexMappingRewritePatterns(mlir::OwningRewritePatternList& patterns);
void populateLoopOptimizationPatterns(mlir::OwningRewritePatternList& patterns);
void populateLoopMergingPatterns(mlir::OwningRewritePatternList& patterns);
void populateLoopSimplificationPatterns(mlir::OwningRewritePatternList& patterns);
} // namespace accera::transforms
