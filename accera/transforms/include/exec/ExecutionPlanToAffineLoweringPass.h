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
using RewritePatternSet = RewritePatternSet;
} // namespace mlir

namespace accera::transforms::executionPlan
{
void populateExecutionPlanCacheFinalizePatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanMultiCachePatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanCopyReducePatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanCacheRegionHoistingPatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanCacheRegionMergingPatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanCacheRegionPatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanCacheMappingPatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanAdjustHierarchicalCacheRegionPositionPatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanAdjustCacheMappingPositionPatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanMaxElementCacheRegionPatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanVectorizePatterns(bool printVectorizationDetails, mlir::RewritePatternSet& patterns);
void populateExecutionPlanVectorizeUnrollPatterns(bool printVectorizationDetails, mlir::RewritePatternSet& patterns);
void populateExecutionPlanTensorizePatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanParallelizePatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanScaleHoistingPatterns(mlir::RewritePatternSet& patterns);
void populateOutOfBoundsAccessHandlingPatterns(mlir::RewritePatternSet& patterns);
void populateConvergeLoadStoresPatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanThriftyCachePatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanDelayedMappingPatterns(mlir::RewritePatternSet& patterns);
void populateExecutionPlanLoopUnswitchingPatterns(mlir::RewritePatternSet& patterns);

std::unique_ptr<mlir::Pass> createExecutionPlanMakeCachePass();
std::unique_ptr<mlir::Pass> createExecutionPlanCopyReducePass();
std::unique_ptr<mlir::Pass> createExecutionPlanCacheRegionLoweringPass();
std::unique_ptr<mlir::Pass> createExecutionPlanVectorizationPass();
std::unique_ptr<mlir::Pass> createExecutionPlanParallelizationPass();
std::unique_ptr<mlir::Pass> createExecutionPlanTensorizationPass();
std::unique_ptr<mlir::Pass> createExecutionPlanScaleHoistingPass();
std::unique_ptr<mlir::Pass> createOutOfBoundsAccessHandlingPass();
} // namespace accera::transforms::executionPlan
