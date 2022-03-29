////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/IRUtil.h>
#include <ir/include/value/ValueDialect.h>
#include <mlir/IR/Location.h>
#include <value/include/MLIREmitterContext.h>

#include <mlir/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/LoopUtils.h>
#include <mlir/Transforms/RegionUtils.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>

#include <cassert>
#include <memory>

using namespace mlir;
namespace ir = accera::ir;
namespace utilir = accera::ir::util;
namespace vir = accera::ir::value;
namespace tr = accera::transforms;
namespace vtr = accera::transforms::value;

namespace
{
void HoistOpToParentAffineScope(Operation* op)
{
    Operation* parentOp = op->getParentOp();
    assert(parentOp != nullptr && "Can only hoist an op that has a parent op");
    if (!parentOp->hasTrait<OpTrait::AffineScope>())
    {
        auto affineScopeParent = op->getParentWithTrait<OpTrait::AffineScope>();
        auto& firstRegion = affineScopeParent->getRegion(0);
        auto& firstBlock = firstRegion.front();
        op->moveBefore(&firstBlock, firstBlock.begin());
    }
}

void HoistGPUBlockThreadIds(vir::ValueModuleOp vModule)
{
    // Hoist GPU block and thread ID ops to the top of the parent AffineScope (e.g. a mlir::FuncOp or a value::LambdaOp)
    // this enables the block and thread ID's to be used with Affine ops, as they
    // are index types that are defined in the top level of an AffineScope.
    // This is safe because the value of the block and thread IDs don't change based
    // on their position in the graph within an AffineScope
    vModule.walk([](mlir::gpu::ThreadIdOp op) {
        HoistOpToParentAffineScope(op.getOperation());
    });
    vModule.walk([](mlir::gpu::BlockIdOp op) {
        HoistOpToParentAffineScope(op.getOperation());
    });
}

constexpr auto kDefaultExecutionTarget = vir::ExecutionTarget::CPU;
constexpr size_t kLaunchConfigNumDims = 6;

struct ValueFuncToTargetPass : public tr::ValueFuncToTargetBase<ValueFuncToTargetPass>
{
    void runOnOperation() final
    {
        auto module = getOperation();
        auto context = module.getContext();

        for (auto vModule : make_early_inc_range(module.getOps<vir::ValueModuleOp>()))
        {
            {
                OwningRewritePatternList patterns(context);
                vtr::populateValueLambdaToFuncPatterns(context, patterns);
                (void)applyPatternsAndFoldGreedily(vModule, std::move(patterns));
            }

            {
                OwningRewritePatternList patterns(context);
                vtr::populateValueLaunchFuncInlinerPatterns(context, patterns);
                (void)applyPatternsAndFoldGreedily(vModule, std::move(patterns));
            }

            {
                OwningRewritePatternList patterns(context);
                vtr::populateValueFuncToTargetPatterns(context, patterns);
                (void)applyPatternsAndFoldGreedily(vModule, std::move(patterns));
            }

            HoistGPUBlockThreadIds(vModule);
        }

        module.walk([&](AffineForOp op) {
            if (op->getAttrOfType<UnitAttr>("accv_unrolled"))
            {
                auto tripCount = mlir::getConstantTripCount(op);
                if (tripCount && *tripCount >= 1)
                    (void)mlir::loopUnrollFull(op);
            }
            else if (auto jammed = op->getAttrOfType<IntegerAttr>("accv_unroll_jam"))
            {
                (void)mlir::loopUnrollJamByFactor(op, (uint64_t)jammed.getInt());
            }
            else
            {
                (void)mlir::promoteIfSingleIteration(op);
            }
        });
    }
};

struct ValueReturnOpConversion : OpRewritePattern<vir::ReturnOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(
        vir::ReturnOp returnOp,
        PatternRewriter& rewriter) const override
    {
        auto operands = returnOp.operands();
        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(returnOp, operands);
        return success();
    }
};

struct ValueFuncToTargetPattern : OpRewritePattern<vir::ValueFuncOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(
        vir::ValueFuncOp funcOp,
        PatternRewriter& rewriter) const override
    {
        auto loc = rewriter.getFusedLoc({ funcOp.getLoc(), accera::ir::util::GetLocation(rewriter, __FILE__, __LINE__) });
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(funcOp);

        [[maybe_unused]] auto target = funcOp.exec_target();

        auto newFuncName = funcOp.sym_name().str();

        auto newFuncOp = rewriter.create<mlir::FuncOp>(
            loc, newFuncName, funcOp.getType());
        newFuncOp.setVisibility(funcOp.getVisibility());

        if (!funcOp->getAttr("external"))
        {
            Region& newBody = newFuncOp.getBody();
            rewriter.inlineRegionBefore(funcOp.getBody(), newBody, newBody.begin());
        }

        // Carry forward attributes
        newFuncOp->setAttrs(funcOp->getAttrs());
        if (funcOp->getAttr(accera::ir::NoInlineAttrName))
        {
            newFuncOp->setAttr("passthrough", rewriter.getArrayAttr({ rewriter.getStringAttr("noinline") }));
        }

        rewriter.eraseOp(funcOp);
        return success();
    }
};

struct ValueLambdaRewritePattern : mlir::OpRewritePattern<vir::ValueLambdaOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult match(vir::ValueLambdaOp op) const final
    {
        auto lambdaFound = false;
        op.body().walk([&](vir::ValueLambdaOp) { lambdaFound = true; return WalkResult::interrupt(); });
        return success(!lambdaFound);
    }

    void rewrite(vir::ValueLambdaOp op, PatternRewriter& rewriter) const final
    {
        // get a list of all values used in the lambda that come from above
        llvm::SetVector<Value> valuesDefinedAbove;
        getUsedValuesDefinedAbove(op.body(), valuesDefinedAbove);

        llvm::SmallVector<Operation*, 8> constants;
        llvm::SmallVector<Value, 4> capturedValues;
        // This could be a std::partition, but mlir::Value doesn't have swap defined and idk how to work with their
        // non-owning lists yet
        for (const Value& v : valuesDefinedAbove)
        {
            if (auto constantOp = v.getDefiningOp(); constantOp && constantOp->hasTrait<mlir::OpTrait::ConstantLike>())
            {
                constants.push_back(constantOp);
            }
            else
            {
                capturedValues.push_back(v);
            }
        }

        // the new function will take all the args that the lambda took, plus all the implicitly captured values
        auto argTypes = llvm::to_vector<4>(op.getArgumentTypes());
        auto capturedValueTypes = mlir::ValueRange{ capturedValues }.getTypes();
        argTypes.append(capturedValueTypes.begin(), capturedValueTypes.end());

        // TODO: maybe support return types with lambdas
        auto fnType = rewriter.getFunctionType(argTypes, llvm::None);

        auto vFuncOp = [&] {
            auto vModuleOp = utilir::CastOrGetParentOfType<vir::ValueModuleOp>(op);
            assert(vModuleOp);

            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.restoreInsertionPoint(utilir::GetTerminalInsertPoint<vir::ValueModuleOp, vir::ModuleTerminatorOp>(vModuleOp));
            auto loc = accera::ir::util::GetLocation(rewriter, __FILE__, __LINE__);
            vir::ValueFuncOp vFuncOp = rewriter.create<vir::ValueFuncOp>(loc, op.sym_name(), fnType, op.exec_target());
            vFuncOp.setPrivate();

            return vFuncOp;
        }();

        auto& bodyBlock = vFuncOp.front();
        auto funcArgs = ValueRange{ vFuncOp.getArguments() };
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(&bodyBlock, bodyBlock.end());
            BlockAndValueMapping valueMapper;

            for (auto [fromValue, toValue] : llvm::zip(op.args(), funcArgs.take_front(op.args().size())))
            {
                if (!valueMapper.contains(fromValue))
                    valueMapper.map(fromValue, toValue);
            }
            for (auto [fromValue, toValue] : llvm::zip(capturedValues, funcArgs.drop_front(op.args().size())))
            {
                if (!valueMapper.contains(fromValue))
                    valueMapper.map(fromValue, toValue);
            }
            for (Operation* constant : constants)
            {
                rewriter.clone(*constant, valueMapper);
            }

            rewriter.cloneRegionBefore(op.body(), vFuncOp.body(), vFuncOp.body().end(), valueMapper);

            rewriter.mergeBlocks(&vFuncOp.back(), &vFuncOp.front(), vFuncOp.front().getArguments().take_front(vFuncOp.back().getNumArguments()));
        }

        vFuncOp.setType(rewriter.getFunctionType(vFuncOp.front().getArgumentTypes(), llvm::None));
        auto args = llvm::to_vector<4>(op.args());
        args.append(capturedValues.begin(), capturedValues.end());

        [[maybe_unused]] auto launchFuncOp = rewriter.create<vir::LaunchFuncOp>(accera::ir::util::GetLocation(rewriter, __FILE__, __LINE__), vFuncOp, args);

        if (auto launchAttr = op->getAttr(vir::ValueFuncOp::getGPULaunchAttrName()))
        {
            launchFuncOp->setAttr(vir::ValueFuncOp::getGPULaunchAttrName(), launchAttr);
            vFuncOp->setAttr(vir::ValueFuncOp::getGPULaunchAttrName(), launchAttr);
        }

        rewriter.eraseOp(op);
    }
};

struct ValueLaunchFuncOpInlinerPattern : OpRewritePattern<vir::LaunchFuncOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(vir::LaunchFuncOp op, PatternRewriter& rewriter) const final
    {
        auto target = op.exec_targetAttr();
        auto callee = op.callee().getLeafReference();

        auto parentFnOp = op->getParentWithTrait<mlir::OpTrait::FunctionLike>();
        if (parentFnOp->getAttr(ir::RawPointerAPIAttrName))
        {
            // Don't inline calls from RawPointerAPI functions
            return failure();
        }

        if (auto attr = parentFnOp->getAttrOfType<vir::ExecutionTargetAttr>(vir::ValueFuncOp::getExecTargetAttrName());
            attr && target == attr)
        {
            auto callInterface = mlir::dyn_cast<mlir::CallOpInterface>(op.getOperation());
            auto callable = callInterface.resolveCallable();
            if (!callable)
            {
                callable = mlir::SymbolTable::lookupNearestSymbolFrom(parentFnOp->getParentOp(), callee);
            }
            assert(llvm::isa<vir::ValueFuncOp>(callable) || llvm::isa<vir::ValueLambdaOp>(callable));
            if (callable->getAttr("external"))
            {
                return failure();
            }
            if (callable->getAttr(ir::NoInlineAttrName))
            {
                return failure();
            }

            auto& body = callable->getRegion(0);

            mlir::BlockAndValueMapping mapping;
            for (auto [src, dst] : llvm::zip(body.front().getArguments(), op.getOperands()))
            {
                mapping.map(src, dst);
            }
            for (auto& bodyOp : body.front().without_terminator())
            {
                rewriter.clone(bodyOp, mapping);
            }

            rewriter.eraseOp(op);
            return success();
        }

        return failure();
    }
};

} // namespace

namespace accera::transforms::value
{
void populateValueFuncToTargetPatterns(mlir::MLIRContext* context, mlir::OwningRewritePatternList& patterns)
{
    uint16_t benefit = 1;
    patterns.insert<ValueReturnOpConversion>(context, benefit++);
    patterns.insert<ValueFuncToTargetPattern>(context, benefit++);
}

void populateValueLambdaToFuncPatterns(mlir::MLIRContext* context, mlir::OwningRewritePatternList& patterns)
{
    uint16_t benefit = 1;
    patterns.insert<ValueLambdaRewritePattern>(context, benefit++);
}

void populateValueLaunchFuncInlinerPatterns(mlir::MLIRContext* context, mlir::OwningRewritePatternList& patterns)
{
    uint16_t benefit = 1;
    patterns.insert<ValueLaunchFuncOpInlinerPattern>(context, benefit++);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueFuncToTargetPass()
{
    return std::make_unique<ValueFuncToTargetPass>();
}

} // namespace accera::transforms::value
