////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace mlir
{
class MLIRContext;
class RewritePatternSet;
class Pass;
using OwningRewritePatternList = RewritePatternSet;
} // namespace mlir

namespace accera::transforms::executionPlan
{
void populateExecutionPlanMakeCachePatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanMultiCachePatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanCopyReducePatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanCacheRegionHoistingPatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanCacheRegionMergingPatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanCacheRegionPatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanCacheMappingPatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanAdjustHierarchicalCacheRegionPositionPatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanAdjustCacheMappingPositionPatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanMaxElementCacheRegionPatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanVectorizePatterns(bool printVectorizationDetails, mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanParallelizePatterns(mlir::OwningRewritePatternList& patterns);
void populateExecutionPlanScaleHoistingPatterns(mlir::OwningRewritePatternList& patterns);
void populateOutOfBoundsAccessHandlingPatterns(mlir::OwningRewritePatternList& patterns);
void populateConvergeLoadStoresPatterns(mlir::OwningRewritePatternList& patterns);

std::unique_ptr<mlir::Pass> createExecutionPlanMakeCachePass();
std::unique_ptr<mlir::Pass> createExecutionPlanCopyReducePass();
std::unique_ptr<mlir::Pass> createExecutionPlanCacheRegionLoweringPass();
std::unique_ptr<mlir::Pass> createExecutionPlanVectorizationPass();
std::unique_ptr<mlir::Pass> createExecutionPlanParallelizationPass();
std::unique_ptr<mlir::Pass> createExecutionPlanScaleHoistingPass();
std::unique_ptr<mlir::Pass> createOutOfBoundsAccessHandlingPass();
} // namespace accera::transforms::executionPlan
