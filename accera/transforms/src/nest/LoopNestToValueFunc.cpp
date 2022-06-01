////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/IRUtil.h>
#include <ir/include/exec/ExecutionPlanAttributes.h>
#include <ir/include/value/ValueDialect.h>

#include <transforms/include/nest/LoopNestToValue.h>
#include <transforms/include/util/SnapshotUtilities.h>

#include <mlir/Dialect/SPIRV/IR/SPIRVAttributes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/LoopUtils.h>

#include <llvm/ADT/TypeSwitch.h>

#include <memory>

using namespace mlir;
namespace lnir = accera::ir::loopnest;
namespace utilir = accera::ir::util;
namespace vir = accera::ir::value;
namespace xpir = accera::ir::executionPlan;

namespace tr = accera::transforms;
namespace lntr = accera::transforms::loopnest;
namespace vtr = accera::transforms::value;
namespace xptr = accera::transforms::executionPlan;
namespace affinetr = accera::transforms::affine;

namespace
{
struct LoopNestToValueFuncPass : public accera::transforms::LoopNestToValueFuncBase<LoopNestToValueFuncPass>
{
    LoopNestToValueFuncPass(const lntr::LoopNestToValueFuncOptions& options = {}) :
        _intrapassSnapshotter(options.snapshotOptions)
    {
        printVecOpDetails = options.printVecOpDetails;
        printLoops = options.printLoops;
    }

    void runOnOperation() final
    {
        auto* context = &getContext();
        auto vFuncOp = getOperation();

        bool shouldRun = true;

        auto snapshotter = _intrapassSnapshotter.MakeSnapshotPipe();
        snapshotter.Snapshot("Initial", vFuncOp);

        mlir::GreedyRewriteConfig topDownConfig; // Some patterns require a top-down handling of ops to ensure relative orders stay consistent
        topDownConfig.useTopDownTraversal = true;

        mlir::GreedyRewriteConfig singleIterationConfig;
        singleIterationConfig.maxIterations = 1;

        while (std::exchange(shouldRun, false))
        {
            {
                OwningRewritePatternList patterns(context);
                lntr::populateLoopnestToValueFuncPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("LoopnestToValueFunc", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                tr::populateRangeResolutionPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("RangeResolution", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                tr::populateScheduledOperationsPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ScheduledOperations", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                tr::populateScheduleScaffoldingPatterns(printLoops, patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ScheduleScaffolding", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                tr::populateScheduledOperationsPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ScheduledOperations", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                tr::populateScheduleToValueRewritePatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ScheduleToValue", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                vtr::populateValueSimplifyPatterns(patterns);
                utilir::FillCanonicalPatternsRecursively(vFuncOp, patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ValueSimplify", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                tr::populateScheduleScaffoldingPatterns(printLoops, patterns);
                tr::populateScheduleToValueRewritePatterns(patterns);
                tr::populateScheduledOperationsPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("Schedule_Scaffolding_Value_Operations", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                // TODO : this shouldn't be in the exec plan dialect
                xptr::populateConvergeLoadStoresPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanConvergeLoadStores", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                utilir::FillCanonicalPatternsRecursively(vFuncOp, patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("Canonicalize", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanMaxElementCacheRegionPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanMaxElementCachePositioning", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanCacheRegionHoistingPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns), topDownConfig);
                snapshotter.Snapshot("ExecutionPlanCacheRegionHoisting", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                tr::populateLoopMergingPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("LoopMerging", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanCacheRegionMergingPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanCacheRegionMerging", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanAdjustHierarchicalCacheRegionPositionPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns), topDownConfig);
                snapshotter.Snapshot("ExecutionPlanAdjustHierarchicalCacheRegionPosition", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanCacheRegionPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns), topDownConfig);
                snapshotter.Snapshot("ExecutionPlanCacheRegion", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanTensorizePatterns(patterns);
                utilir::FillCanonicalPatternsRecursively(vFuncOp, patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanTensorize", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanAdjustCacheMappingPositionPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanAdjustCacheMappingPosition", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanCacheMappingPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns), topDownConfig);
                snapshotter.Snapshot("ExecutionPlanCacheMapping", vFuncOp);
            }

            {
                // Note: A canonicalization cannot happen between ExecutionPlanCacheMapping and ExecutionPlanCheckAndElideThriftyCaches
                //       otherwise attributes on loads will be removed that this pass depends on
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanThriftyCachePatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanCheckAndElideThriftyCaches", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                tr::populateScheduledOperationsPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ScheduledOperations", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanScaleHoistingPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanScaleHoisting", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                utilir::FillCanonicalPatternsRecursively(vFuncOp, patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("Canonicalize", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanMultiCachePatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanMultiCacheCopy", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanCopyReducePatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanCopyReduce", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanDelayedMappingPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanDelayedMapping", vFuncOp);
            }

            {
                OwningRewritePatternList patterns(context);
                xptr::populateExecutionPlanLoopUnswitchingPatterns(patterns);
                (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
                snapshotter.Snapshot("ExecutionPlanLoopUnswitching", vFuncOp);
            }

            vFuncOp.walk([&shouldRun](lnir::NestOp) { shouldRun = true; return WalkResult::interrupt(); });
        }

        snapshotter.Snapshot("PostLoop", vFuncOp);

        {
            OwningRewritePatternList patterns(context);
            tr::populateLoopSimplificationPatterns(patterns);
            (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
            snapshotter.Snapshot("LoopSimplification", vFuncOp);
        }

        {
            OwningRewritePatternList patterns(context);
            utilir::FillCanonicalPatternsRecursively(vFuncOp, patterns);
            (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
            snapshotter.Snapshot("Canonicalize", vFuncOp);
        }

        {
            OwningRewritePatternList patterns(context);
            affinetr::populateAcceraAffineSimplificationPatterns(patterns);
            (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns), singleIterationConfig);
            snapshotter.Snapshot("AcceraAffineSimplification", vFuncOp);
        }

        {
            OwningRewritePatternList patterns(context);
            xptr::populateOutOfBoundsAccessHandlingPatterns(patterns);
            (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
            snapshotter.Snapshot("OutOfBoundsAccessHandling", vFuncOp);
        }

        {
            OwningRewritePatternList patterns(context);
            tr::populateGPUIndexMappingRewritePatterns(patterns);
            (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
            snapshotter.Snapshot("GPUIndexMapping", vFuncOp);
        }

        {
            OwningRewritePatternList patterns(context);
            xptr::populateExecutionPlanVectorizePatterns(printVecOpDetails, patterns);
            utilir::FillCanonicalPatternsRecursively(vFuncOp, patterns);
            (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
            snapshotter.Snapshot("ExecutionPlanVectorize_Canonicalize", vFuncOp);
        }

        {
            OwningRewritePatternList patterns(context);
            tr::populateLoopOptimizationPatterns(patterns);
            (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
            snapshotter.Snapshot("LoopOptimization", vFuncOp);
        }

        {
            OwningRewritePatternList patterns(context);
            xptr::populateExecutionPlanCacheFinalizePatterns(patterns);
            (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
            snapshotter.Snapshot("ExecutionPlanCacheFinalize", vFuncOp);
        }

        {
            OwningRewritePatternList patterns(context);
            xptr::populateExecutionPlanParallelizePatterns(patterns);
            utilir::FillCanonicalPatternsRecursively(vFuncOp, patterns);
            (void)applyPatternsAndFoldGreedily(vFuncOp, std::move(patterns));
            snapshotter.Snapshot("ExecutionPlanParallelize", vFuncOp);
        }
    }

    tr::IRSnapshotter _intrapassSnapshotter;
};

struct NestOpLowering : OpRewritePattern<lnir::NestOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(lnir::NestOp nestOp,
                                  PatternRewriter& rewriter) const override
    {
        auto loc = nestOp.getLoc();
        assert(nestOp.use_empty());
        if (nestOp->getParentOfType<lnir::KernelOp>())
        {
            return failure();
        }

        // Create the function that takes the arguments needed
        auto fnName = "NestFunction_" + std::to_string(accera::ir::util::GetUniqueId());

        auto scheduleOp = nestOp.getOrCreateSchedule();
        auto execPlanOp = scheduleOp.getOrCreateExecPlan();
        auto execTarget = execPlanOp.exec_target();

        auto fnType = rewriter.getFunctionType(llvm::None, llvm::None);

        auto nestFuncOp = [&]() -> vir::ValueLambdaOp {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(nestOp);

            return rewriter.create<vir::ValueLambdaOp>(loc,
                                                       fnName,
                                                       fnType,
                                                       execTarget);
        }();
        if (auto launchAttr = execPlanOp->getAttr(execPlanOp.getGPULaunchAttrName()))
        {
            nestFuncOp->setAttr(nestFuncOp.getGPULaunchAttrName(), launchAttr);
        }

        auto vectorizationInfoIdentifier = rewriter.getIdentifier(xpir::VectorizationInfoAttr::getKeyName());
        if (auto vectorizationInfoAttr = execPlanOp->getAttr(vectorizationInfoIdentifier))
        {
            nestFuncOp->setAttr(vectorizationInfoIdentifier, vectorizationInfoAttr);
        }

        rewriter.eraseOp(nestOp.getBody()->getTerminator());
        nestFuncOp.getBody().takeBody(nestOp.body());
        auto& nestBodyBlock = nestFuncOp.front();

        // Add the final return statement
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(&nestBodyBlock, nestBodyBlock.end());
            (void)rewriter.create<vir::ReturnOp>(loc);
        }

        rewriter.eraseOp(nestOp);
        return success();
    }
};

} // namespace

namespace accera::transforms::loopnest
{

void populateLoopnestToValueFuncPatterns(mlir::OwningRewritePatternList& patterns)
{
    uint16_t benefit = 1;
    auto context = patterns.getContext();
    patterns.insert<NestOpLowering>(context, benefit++);
}

std::unique_ptr<mlir::OperationPass<accera::ir::value::ValueFuncOp>> createLoopNestToValueFuncPass(const LoopNestToValueFuncOptions& options)
{
    return std::make_unique<LoopNestToValueFuncPass>(options);
}

std::unique_ptr<mlir::OperationPass<accera::ir::value::ValueFuncOp>> createLoopNestToValueFuncPass()
{
    return std::make_unique<LoopNestToValueFuncPass>();
}

} // namespace accera::transforms::loopnest
