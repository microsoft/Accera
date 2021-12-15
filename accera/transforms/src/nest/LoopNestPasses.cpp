////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/LoopNestPasses.h"
#include "AcceraPasses.h"
#include "nest/LoopNestToValue.h"

#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueDialect.h>

#include <mlir/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/LoopUtils.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/raw_os_ostream.h>

#include <iostream>

using namespace accera::ir;
using namespace accera::ir::loopnest;
namespace v = accera::ir::value;
namespace xp = accera::ir::executionPlan;
using namespace accera::transforms;
using namespace mlir;

namespace
{

struct ScheduledOperationsLoweringPass : public ConvertScheduledOperationsBase<ScheduledOperationsLoweringPass>
{
    void runOnFunction() final;
};

struct ScheduleToValueLoweringPass : public ConvertScheduleToValueBase<ScheduleToValueLoweringPass>
{
    void runOnFunction() final;
};

struct LoopNestOptPass : public ConvertLoopNestOptBase<LoopNestOptPass>
{
    void runOnOperation() final;
};

} // end anonymous namespace.

void ScheduledOperationsLoweringPass::runOnFunction()
{
    {
        OwningRewritePatternList patterns(&getContext());
        populateRangeResolutionPatterns(patterns);
        (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    {
        OwningRewritePatternList patterns(&getContext());
        populateScheduleScaffoldingPatterns(false, patterns);
        (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    ConversionTarget target(getContext());

    target.addLegalDialect<v::ValueDialect,
                           memref::MemRefDialect,
                           mlir::AffineDialect,
                           mlir::StandardOpsDialect,
                           LoopNestDialect,
                           xp::ExecutionPlanDialect>();

    target.addDynamicallyLegalOp<ScheduleOp>([](Operation* op) {
        // ScheduleOps still inside of kernels should be left alone for now
        return isa<KernelOp>(op->getParentOp());
    });

    target.addDynamicallyLegalOp<ScheduledLoopOp>([](ScheduledLoopOp op) {
        bool found = false;
        auto index = op.index().getValue();
        op.walk([&](Operation* innerOp) {
            for (auto operand : innerOp->getOperands())
            {
                if (operand)
                {
                    if (auto indexOp = dyn_cast_or_null<SymbolicIndexOp>(operand.getDefiningOp()); indexOp && index == indexOp.getValue())
                    {
                        found = true;
                    }
                }
            }
        });
        return !found;
    });

    target.addDynamicallyLegalOp<DimSizeOp>([](DimSizeOp op) {
        // DimSizeOps still inside of kernels should be left alone for now
        auto parentOp = op.getOperation()->getParentOp();
        return isa<KernelOp>(parentOp);
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the operations.
    OwningRewritePatternList patterns(&getContext());
    populateScheduledOperationsPatterns(patterns);

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
    {
        llvm::errs() << "ScheduledOperationsLoweringPass failed\n";
        llvm::errs().flush();

        signalPassFailure();
    }
}

void ScheduleToValueLoweringPass::runOnFunction()
{
    auto function = getFunction();

    {
        OwningRewritePatternList foldPatterns(&getContext());
        populateScheduleToValueRewritePatterns(foldPatterns);
        (void)applyPatternsAndFoldGreedily(function, std::move(foldPatterns));
    }

    ConversionTarget target(getContext());
    target.addLegalDialect<v::ValueDialect,
                           memref::MemRefDialect,
                           mlir::AffineDialect,
                           mlir::StandardOpsDialect,
                           xp::ExecutionPlanDialect>();

    // Now we only allow terminators and symbolic indices
    target.addIllegalDialect<LoopNestDialect>();
    target.addLegalOp<SymbolicIndexOp>();

    // Remove predicates if they aren't used anymore
    target.addDynamicallyLegalOp<ScheduledKernelOp,
                                 NullPredicateOp,
                                 ProloguePredicateOp,
                                 EpiloguePredicateOp,
                                 ConstantPredicateOp,
                                 FragmentTypePredicateOp,
                                 PlacementPredicateOp,
                                 IndexDefinedPredicateOp,
                                 ConjunctionPredicateOp,
                                 DisjunctionPredicateOp>([](Operation* op) {
        return !op->use_empty();
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the operations.
    OwningRewritePatternList patterns(&getContext());
    populateScheduleToValuePatterns(patterns);

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(function, target, std::move(patterns))))
    {
        signalPassFailure();
    }
}

void LoopNestOptPass::runOnOperation()
{
    auto func = getOperation();

    func.walk([&](AffineForOp op) {
        if (op->getAttrOfType<UnitAttr>("rcv_unrolled"))
        {
            auto tripCount = getConstantTripCount(op);
            if (tripCount && *tripCount >= 1)
                (void)loopUnrollFull(op);
        }
    });
}

namespace accera::transforms::loopnest
{
std::unique_ptr<Pass> createScheduledOperationsPass()
{
    return std::make_unique<ScheduledOperationsLoweringPass>();
}

std::unique_ptr<Pass> createScheduleToValuePass()
{
    return std::make_unique<ScheduleToValueLoweringPass>();
}

std::unique_ptr<Pass> createLoopNestOptPass()
{
    return std::make_unique<LoopNestOptPass>();
}

void addLoopNestStructureLoweringPasses(mlir::PassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createScheduledOperationsPass());
}

void addLoopNestFinalLoweringPasses(mlir::PassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createScheduleToValuePass());
    pm.addNestedPass<ir::value::ValueFuncOp>(createLoopNestOptPass());
}

void addLoopNestCleanupLoweringPasses(mlir::PassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
}

void addLoopNestLoweringPasses(mlir::PassManager& pm)
{
    addLoopNestStructureLoweringPasses(pm);
    addLoopNestFinalLoweringPasses(pm);
    addLoopNestCleanupLoweringPasses(pm);
}

} // namespace accera::transforms::loopnest
