////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include "value/ValueToLLVMLoweringPass.h"

#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/accera/AcceraOps.h>
#include <ir/include/value/ValueDialect.h>

#include <value/include/MLIREmitterContext.h>
#include <value/include/Matrix.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/VectorOps.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/raw_os_ostream.h>

#include <iostream>

namespace v = accera::ir::value;
namespace rc = accera::ir::rc;
namespace xp = accera::ir::executionPlan;
namespace ln = accera::ir::loopnest;

using namespace mlir;

namespace
{
std::string kAcceraFuncAttrName = "AcceraFunc";

MemRefType convertToMemRefType(Type type)
{
    MemRefType memRefType;
    auto tensorType = type.dyn_cast<TensorType>();
    if (tensorType)
    {
        assert(tensorType.hasRank() && "expected only ranked shapes");
        memRefType =
            MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    }
    else
    {
        memRefType = type.dyn_cast<MemRefType>();
    }
    return memRefType;
}

// Determine if current function returns the result value of the
// current op being lowered. If it does then dealloc should not be
// inserted.
bool checkInsertDealloc(Operation* currentOp, int resultIndex = 0)
{
    auto parentBlock = currentOp->getBlock();

    bool insertDealloc = true;
    parentBlock->walk([&insertDealloc, currentOp, resultIndex](ReturnOp op) {
        // If there is at least one result to investigate.
        if (currentOp->getNumResults() > 0)
        {
            auto result = currentOp->getResult(resultIndex);
            for (auto operand : op.getOperands())
                if (operand == result)
                    insertDealloc = false;
        }
    });

    return insertDealloc;
}

Value insertAllocAndDealloc(MemRefType type, Location loc, PatternRewriter& rewriter, bool insertDealloc)
{
    auto alloc = rewriter.create<memref::AllocOp>(loc, type);

    auto* parentBlock = alloc.getOperation()->getBlock();
    alloc.getOperation()->moveBefore(&parentBlock->front());

    if (insertDealloc)
    {
        auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
    }

    return alloc;
}

struct GemmOpLowering : public OpConversionPattern<rc::GemmOp>
{
    using OpConversionPattern<rc::GemmOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        rc::GemmOp op,
        ArrayRef<mlir::Value> operands,
        ConversionPatternRewriter& rewriter) const override;
};

struct AcceraLoweringPass : public accera::transforms::ConvertAcceraToLowerBase<AcceraLoweringPass>
{
    void runOnModule() final;
};

struct AcceraToLLVMPass : public accera::transforms::ConvertAcceraToLLVMBase<AcceraToLLVMPass>
{
    void runOnFunction() final;
};

struct GlobalAcceraToLLVMPass : public accera::transforms::ConvertGlobalAcceraToLLVMBase<GlobalAcceraToLLVMPass>
{
    void runOnModule() final;
};

} // namespace

void GlobalAcceraToLLVMPass::runOnModule()
{
    auto context = &getContext();
    LLVMConversionTarget target(*context);

    target.addLegalOp<ModuleOp>();
    target.addLegalOp<v::ModuleTerminatorOp>();

    OwningRewritePatternList patterns(context);

    // TODO: this is definitely out of date
    mlir::LowerToLLVMOptions options(context);
    options.useBarePtrCallConv = true;
    options.allocLowering = mlir::LowerToLLVMOptions::AllocLowering::AlignedAlloc;
    LLVMTypeConverter llvmTypeConverter(context, options);

    accera::transforms::rc::populateGlobalAcceraToLLVMPatterns(llvmTypeConverter, patterns);

    if (failed(applyPartialConversion(getModule(), target, std::move(patterns))))
    {
        signalPassFailure();
    }
}

void AcceraToLLVMPass::runOnFunction()
{
    auto f = getFunction();
    auto attr = f->getAttr(kAcceraFuncAttrName);
    if (!attr)
    {
        return;
    }

    llvm::errs() << "Running pass on " << f.getName() << "\n";

    PassManager pm(&getContext());

    auto fPM = pm.nest<FuncOp>();
    fPM.addPass(mlir::createCanonicalizerPass());
    fPM.addPass(accera::transforms::loopnest::createScheduledOperationsPass());
    fPM.addPass(accera::transforms::loopnest::createScheduleToValuePass());
    fPM.addPass(mlir::createCanonicalizerPass());
    fPM.addPass(accera::transforms::loopnest::createLoopNestOptPass());
    fPM.addPass(accera::transforms::value::createValueSimplifyPass());

    fPM.addPass(accera::transforms::executionPlan::createExecutionPlanCacheRegionLoweringPass());
    fPM.addPass(accera::transforms::executionPlan::createExecutionPlanVectorizationPass());
    fPM.addPass(accera::transforms::executionPlan::createExecutionPlanCopyReducePass());

    fPM.addPass(accera::transforms::loopnest::createScheduledOperationsPass());
    fPM.addPass(accera::transforms::loopnest::createScheduleToValuePass());
    fPM.addPass(accera::transforms::loopnest::createLoopNestOptPass());

    fPM.addPass(accera::transforms::executionPlan::createExecutionPlanVectorizationPass());
    fPM.addPass(accera::transforms::executionPlan::createExecutionPlanMakeCachePass());

    fPM.addPass(accera::transforms::value::createValueToStdPass());
    fPM.addPass(mlir::createCanonicalizerPass());
    fPM.addPass(mlir::createCSEPass());
    // fPM.addPass(mlir::createInlinerPass());

    fPM.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    fPM.addPass(mlir::createSimplifyAffineStructuresPass());
    fPM.addPass(mlir::createCanonicalizerPass());
    fPM.addPass(mlir::createLoopUnrollPass());
    fPM.addPass(mlir::createLowerAffinePass());
    fPM.addPass(mlir::createLowerToCFGPass());
    // fPM.addPass(accera::transforms::rc::createAcceraToLLVMPass());

    if (failed(pm.run(f->getParentOfType<ModuleOp>())))
        signalPassFailure();

    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<
        accera::ir::value::ValueDialect,
        accera::ir::loopnest::LoopNestDialect,
        accera::ir::executionPlan::ExecutionPlanDialect>();
    target.addLegalOp<mlir::FuncOp>();

    mlir::LowerToLLVMOptions options(&getContext());
    options.useBarePtrCallConv = true;
    options.allocLowering = mlir::LowerToLLVMOptions::AllocLowering::AlignedAlloc;
    LLVMTypeConverter llvmTypeConverter(&getContext(), options);

    OwningRewritePatternList patterns(&getContext());
    accera::transforms::value::populateLocalValueToLLVMPatterns(llvmTypeConverter, patterns);

    populateLinalgToLLVMConversionPatterns(llvmTypeConverter, patterns);
    populateStdToLLVMConversionPatterns(llvmTypeConverter, patterns);

    populateVectorToLLVMConversionPatterns(llvmTypeConverter, patterns, /*reassociateFPReductions*/ true);

    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
    {
        signalPassFailure();
    }
}

void AcceraLoweringPass::runOnModule()
{
    auto module = getModule();
    accera::value::ContextGuard<accera::value::MLIRContext> guard(module);

    ConversionTarget target(getContext());

    target.addLegalDialect<v::ValueDialect, xp::ExecutionPlanDialect, ln::LoopNestDialect, mlir::StandardOpsDialect>();
    target.addIllegalDialect<rc::AcceraDialect>();
    target.addLegalOp<ModuleOp, v::ModuleTerminatorOp, FuncOp>();

    OwningRewritePatternList patterns(&getContext());
    accera::transforms::rc::populateAcceraLoweringPatterns(patterns);

    if (failed(applyPartialConversion(getModule(), target, std::move(patterns))))
        signalPassFailure();
}

LogicalResult GemmOpLowering::matchAndRewrite(
    rc::GemmOp op,
    ArrayRef<mlir::Value> operands,
    ConversionPatternRewriter& rewriter) const
{
    rc::GemmOp::Adaptor adaptor(operands);
    auto loc = op.getLoc();

    auto f32Type = rewriter.getF32Type();
    bool hasBias = !op.getOperand(2).getType().isa<NoneType>();

    auto transposeA = op.transA() != 0;
    auto transposeB = op.transB() != 0;

    auto A = adaptor.A();
    auto B = adaptor.B();
    mlir::Value C;
    if (hasBias)
    {
        C = adaptor.C();
    }

    auto YMemRefType = convertToMemRefType(op.Y().getType());
    mlir::Value Y = [&] {
        bool insertDealloc = checkInsertDealloc(op);
        return insertAllocAndDealloc(YMemRefType, op.getLoc(), rewriter, insertDealloc);
    }();

    llvm::SmallVector<Type, 4> argTypes{ YMemRefType, A.getType(), B.getType() };
    if (hasBias)
    {
        argTypes.push_back(C.getType());
    }

    llvm::SmallVector<mlir::NamedAttribute, 1> argAttrElt{ { rewriter.getIdentifier("llvm.noalias"),
                                                             rewriter.getBoolAttr(true) } };

    FuncOp f;
    {
        auto module = op->getParentOfType<ModuleOp>();
        auto body = module.getBody();
        auto bodyIter = module.getBody()->begin();
        auto endIter = module.getBody()->end();
        while (bodyIter != endIter && !llvm::isa<mlir::FuncOp>(bodyIter))
        {
            ++bodyIter;
        }

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(body, bodyIter);
        f = rewriter.create<FuncOp>(
            rewriter.getFusedLoc(op.getLoc()),
            "gemm_f",
            rewriter.getFunctionType(
                argTypes,
                {}),
            std::vector{ rewriter.getNamedAttr(kAcceraFuncAttrName, rewriter.getUnitAttr()) });
        f.addEntryBlock();
    }
    {
        OpBuilder::InsertionGuard guard(rewriter);
        auto loc = f.getLoc();
        rewriter.setInsertionPoint(&f.front(), f.front().begin());
        [[maybe_unused]] auto alpha = rewriter.create<ConstantFloatOp>(loc, op.alpha(), f32Type );
        [[maybe_unused]] auto beta = rewriter.create<ConstantFloatOp>(loc, hasBias ? op.beta() : llvm::APFloat(0.f), f32Type );

        auto AShape = A.getType().cast<ShapedType>().getShape().vec();
        auto BShape = B.getType().cast<ShapedType>().getShape().vec();
        if (transposeA)
        {
            std::swap(AShape[0], AShape[1]);
        }
        if (transposeB)
        {
            std::swap(BShape[0], BShape[1]);
        }

        accera::value::Value YValue = accera::value::Wrap(f.getArgument(0), accera::utilities::MemoryLayout{ YMemRefType.getShape().vec() });
        accera::value::Value AValue = accera::value::Wrap(f.getArgument(1), accera::utilities::MemoryLayout{ AShape });
        accera::value::Value BValue = accera::value::Wrap(f.getArgument(2), accera::utilities::MemoryLayout{ BShape });

        accera::value::MatrixMatrixMultiply(AValue, BValue, YValue);

        (void)rewriter.create<ReturnOp>(loc);
    }

    llvm::SmallVector<mlir::Value, 4> fArgs{ Y, A, B };
    if (hasBias)
    {
        fArgs.push_back(C);
    }
    rewriter.create<CallOp>(loc, f, ValueRange{ fArgs } );

    rewriter.replaceOp(op, ValueRange{ Y });

    return success();
}

namespace accera::transforms::rc
{

void populateGlobalAcceraToLLVMPatterns(mlir::LLVMTypeConverter& typeConverter,mlir::OwningRewritePatternList& patterns)
{
    accera::transforms::value::populateGlobalValueToLLVMPatterns(typeConverter, patterns);
}

void populateAcceraToLLVMPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::OwningRewritePatternList& patterns)
{
    accera::transforms::value::populateLocalValueToLLVMPatterns(typeConverter, patterns);
}

void populateAcceraLoweringPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<GemmOpLowering>(patterns.getContext());
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraLoweringPass()
{
    return std::make_unique<AcceraLoweringPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createAcceraToLLVMPass()
{
    return std::make_unique<AcceraToLLVMPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGlobalAcceraToLLVMPass()
{
    return std::make_unique<GlobalAcceraToLLVMPass>();
}

// TODO: dead code - remove?
void addAcceraToStandardPasses(mlir::PassManager& pm)
{
    pm.addPass(createAcceraLoweringPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(transforms::loopnest::createScheduledOperationsPass());
    pm.addPass(transforms::loopnest::createScheduleToValuePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(transforms::loopnest::createLoopNestOptPass());
    pm.addPass(transforms::value::createValueSimplifyPass());

    pm.addPass(transforms::executionPlan::createExecutionPlanCacheRegionLoweringPass());

    // Adding this seems to cause errors when doing the final translation to LLVMIR
    // pm.addPass(transforms::executionPlan::createExecutionPlanVectorizationPass());

    pm.addPass(transforms::executionPlan::createExecutionPlanCopyReducePass());

    pm.addPass(transforms::loopnest::createScheduledOperationsPass());
    pm.addPass(transforms::loopnest::createScheduleToValuePass());
    pm.addPass(transforms::loopnest::createLoopNestOptPass());

    // Adding this seems to cause errors when doing the final translation to LLVMIR
    // pm.addPass(transforms::executionPlan::createExecutionPlanVectorizationPass());

    pm.addPass(transforms::executionPlan::createExecutionPlanMakeCachePass());

    pm.addPass(transforms::value::createValueToStdPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createInlinerPass());

    pm.addPass(createConvertLinalgToAffineLoopsPass());
    pm.addPass(createSimplifyAffineStructuresPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createLoopUnrollPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createLowerToCFGPass());
}

} // namespace accera::transforms::rc
