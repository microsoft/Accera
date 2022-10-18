////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueFuncOp.h>
#include <value/include/Debugging.h>

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;
namespace ir = accera::ir;

namespace
{

ir::value::ValueFuncOp FindValueFunctionOp(const ir::value::ValueModuleOp& moduleOp, const std::string& name)
{
    ir::value::ValueFuncOp result;
    moduleOp->walk([name, &result](ir::value::ValueFuncOp op) {
        if (name == op.sym_name())
        {
            result = op;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return result;
}

ir::value::ValueFuncOp GetLaunchFunctionOp(const ir::value::ValueModuleOp& moduleOp, ir::value::ValueFuncOp& funcOp)
{
    ir::value::ValueFuncOp result = nullptr;
    auto name = funcOp.sym_name();

    moduleOp->walk([name, &result](ir::value::LaunchFuncOp op) {
        if (name == op.callee().getLeafReference())
        {
            result = ir::util::CastOrGetParentOfType<ir::value::ValueFuncOp>(op);
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });

    return result;
}

ir::loopnest::ScheduleOp GetScheduleOp(ir::value::ValueFuncOp& funcOp)
{
    // Find the ScheduleOp
    ir::loopnest::ScheduleOp result;
    if (auto region = &funcOp.body())
    {
        region->walk([&result](ir::loopnest::ScheduleOp op) {
            result = op;
            return WalkResult::interrupt();
        });
    }
    return result;
}

void MapArguments(PatternRewriter& rewriter, ir::value::ValueFuncOp& targetFnOp, ir::value::ValueFuncOp& wrapperFnOp, BlockAndValueMapping& valueMap)
{
    // Map target function args to debug function args
    for (auto [fromValue, toValue] : llvm::zip(targetFnOp.getArguments(), wrapperFnOp.getArguments()))
    {
        valueMap.map(fromValue, toValue);
    }

    targetFnOp->walk([&rewriter, &valueMap](Operation* op) {
        TypeSwitch<Operation*>(op)
            .Case([&](ir::value::AllocOp allocOp) {
                // Replicate local allocations (e.g. TEMP arrays)
                auto newOp = mlir::cast<ir::value::AllocOp>(rewriter.clone(*allocOp.getOperation()));
                valueMap.map(allocOp.getResult(), newOp.getResult());
            })
            .Case([&](ir::value::ReferenceGlobalOp refGlobalOp) {
                // Replicate references to globals (e.g. CONST arrays)
                auto newOp = mlir::cast<ir::value::ReferenceGlobalOp>(rewriter.clone(*refGlobalOp.getOperation()));
                valueMap.map(refGlobalOp.getResult(), newOp.getResult());
            });
    });
}

void CreateReferenceSchedules(PatternRewriter& rewriter, ir::loopnest::ScheduleOp& scheduleOp, BlockAndValueMapping& valueMap)
{
    auto targetNestOp = scheduleOp.getNest();
    if (auto fusedDomains = scheduleOp.getFusedDomains(); !fusedDomains.empty())
    {
        // Fusing case: split into multiple schedules (one per kernel), in kernel order
        auto kernels = targetNestOp.getKernels();

        // We currently only support 1 kernel per domain (corresponds to a "never-fused-before" schedule)
        // TODO: remove this limitation when the Python DSL supports adding multiple kernels to a schedule
        assert(fusedDomains.size() == kernels.size() && "Number of unfused domains != number of unfused kernels");
        for (auto [targetKernel, fusedDomain] : llvm::zip(kernels, fusedDomains))
        {
            auto nest = ir::loopnest::MakeNest(rewriter, fusedDomain);
            auto nestBuilder = nest.getBodyBuilder();

            // Map target symbolic indices to debug symbolic indices
            auto dims = fusedDomain.GetDimensions();
            std::unordered_set<ir::loopnest::Index> fusedDomainIndices(dims.begin(), dims.end());

            targetNestOp.walk([&](ir::loopnest::SymbolicIndexOp fromIndex) {
                if (!fromIndex.use_empty())
                {
                    auto sourceIndex = fromIndex.getValue();
                    for (auto fusedIndex : scheduleOp.getFusedIndices(sourceIndex))
                    {
                        // A reverse mapping of fused index to original index exists AND the original index
                        // belongs in the unfused domain
                        if (fusedDomainIndices.find(fusedIndex) != fusedDomainIndices.end())
                        {
                            sourceIndex = fusedIndex;
                            break;
                        }
                    }
                    auto toIndex = nest.getOrCreateSymbolicIndex(nestBuilder, sourceIndex);
                    valueMap.map(fromIndex.getResult(), toIndex.getResult());
                }
            });

            // Clone the kernel, referencing the re-mapped Values (this creates the symbolic indices)
            auto kernel = mlir::cast<ir::loopnest::KernelOp>(nestBuilder.clone(*targetKernel.getOperation(), valueMap));

            // Create the schedule and add the kernels (after the symbolic indices have been inserted into the IR)
            auto defaultSchedule = nest.getOrCreateSchedule();
            defaultSchedule.addKernel(kernel);
        }
    }
    else
    {
        // Non-fusing case: duplicate the nest with its kernel(s)
        auto domain = targetNestOp.getDomain().getValue();
        auto nest = ir::loopnest::MakeNest(rewriter, domain);
        auto nestBuilder = nest.getBodyBuilder();

        // Map target symbolic indices to debug symbolic indices
        targetNestOp.walk([&nestBuilder, &nest, &valueMap](ir::loopnest::SymbolicIndexOp fromIndex) {
            if (!fromIndex.use_empty())
            {
                auto toIndex = nest.getOrCreateSymbolicIndex(nestBuilder, fromIndex.getValue());
                valueMap.map(fromIndex.getResult(), toIndex.getResult());
            }
        });

        // Clone the kernels, referencing the re-mapped Values (this creates the symbolic indices)
        std::vector<ir::loopnest::KernelOp> kernels;
        auto targetKernels = targetNestOp.getKernels();
        std::transform(targetKernels.cbegin(), targetKernels.cend(), std::back_inserter(kernels), [&nestBuilder, &valueMap](auto knl) {
            return mlir::cast<ir::loopnest::KernelOp>(nestBuilder.clone(*knl.getOperation(), valueMap));
        });

        // Create the schedule and add the kernels (after the symbolic indices have been inserted into the IR)
        auto defaultSchedule = nest.getOrCreateSchedule();
        for (auto& kernel : kernels)
        {
            defaultSchedule.addKernel(kernel);
        }
    }
}

// Check functions are specified per input/output arguments and empty strings for input arguments
std::vector<std::string> GetCheckFunctions(ir::value::ValueFuncOp& targetFnOp)
{
    std::vector<std::string> result;
    if (targetFnOp->hasAttr(ir::GetOutputVerifiersAttrName()))
    {
        auto checkFunctionAttrs = targetFnOp->getAttrOfType<ArrayAttr>(ir::GetOutputVerifiersAttrName()).getValue();

        std::transform(checkFunctionAttrs.begin(), checkFunctionAttrs.end(), std::back_inserter(result), [](auto attr) {
            return attr.template cast<StringAttr>().getValue().str();
        });
    }

    return result;
}

std::vector<mlir::Value> GetTargetFunctionArgs(PatternRewriter& rewriter, Location loc, ir::value::ValueModuleOp& moduleOp, ir::value::ValueFuncOp& targetFnOp, ir::value::ValueFuncOp& dbgFnOp)
{
    std::vector<mlir::Value> result;

    // Replicate output args at the top of the function and copy the data
    std::string name = moduleOp.getName().str() + "_" + dbgFnOp.getName().str() + "_target_output_arg";
    auto dbgFnOpArgs = dbgFnOp.getArguments();
    for (auto [blockArg, checkFunction] : llvm::zip(dbgFnOpArgs, GetCheckFunctions(targetFnOp)))
    {
        if (!checkFunction.empty()) // output arguments have non-empty check functions
        {
            if (auto memrefType = blockArg.getType().dyn_cast<mlir::MemRefType>())
            {
                // Simplify any identity affine maps, e.g. (d0, d1) -> (d0 * 256 + d1) can become (d0, d1) -> (d0, d1)
                // Required by ConvertToLLVMPattern::isConvertibleAndHasIdentityMaps() in GlobalMemrefOpLowering
                // First, try simplifying the layout as it is
                memrefType = mlir::canonicalizeStridedLayout(memrefType);

                auto count = memrefType.getNumDynamicDims(); 
                if (count > 0)
                {
                    std::vector<mlir::BlockArgument> dbgFnOpArgsOfCount = dbgFnOpArgs.take_front(count).vec();
                    std::vector<mlir::Value> sizes;
                    std::transform(dbgFnOpArgsOfCount.begin(), dbgFnOpArgsOfCount.end(), std::back_inserter(sizes), [](mlir::BlockArgument d) { return (mlir::Value)(d); });
    
                    // TODO: need to check if this local scoped alloc is freed somewhere
                    auto localScopeAlloc = rewriter.create<ir::value::AllocOp>(loc,
                                        memrefType,
                                        llvm::None,
                                        llvm::None,
                                        mlir::ValueRange{ sizes});

                    (void)rewriter.create<ir::value::CopyOp>(loc, blockArg, localScopeAlloc);

                    result.push_back(localScopeAlloc);
                }
                else
                {
                    if (!memrefType.getLayout().isIdentity())
                    {
                        // The layout could not be simplified (e.g. SubArrays) - force an identity map
                        // The logical access indices will still work but there is a potential performance tradeoff with
                        // a change in the physical layout (acceptable for Debug mode)
                        memrefType = mlir::MemRefType::Builder(memrefType).setLayout({});
                    }

                    auto argCopy = ir::util::CreateGlobalBuffer(rewriter, dbgFnOp, memrefType, name);

                    // Replace the global-scoped ReferenceGlobalOp with one within the function context
                    auto globalScopeGlobalRef = mlir::dyn_cast_or_null<ir::value::ReferenceGlobalOp>(argCopy.getDefiningOp());
                    auto localScopeGlobalRef = rewriter.create<ir::value::ReferenceGlobalOp>(loc, globalScopeGlobalRef.getGlobal());

                    (void)rewriter.create<ir::value::CopyOp>(loc, blockArg, localScopeGlobalRef);

                    result.push_back(localScopeGlobalRef);
                    rewriter.eraseOp(globalScopeGlobalRef);
                }
            }
            else
            {
                throw std::logic_error{ "Argument is not a memRefType" }; // TODO: support additional function arg types as needed
            }
        }
        else
        {
            result.push_back(blockArg); // pass-through any input args
        }
    }
    return result;
}

LogicalResult EmitNestDebugFunction(ir::value::ValueFuncOp& targetFnOp, PatternRewriter& rewriter)
{
    // Find the ScheduleOp
    auto scheduleOp = GetScheduleOp(targetFnOp);
    if (!scheduleOp)
    {
        targetFnOp->removeAttr(ir::GetOutputVerifiersAttrName());
        return failure(); // no match
    }

    auto loc = targetFnOp.getLoc();
    auto moduleOp = ir::util::CastOrGetParentOfType<ir::value::ValueModuleOp>(targetFnOp.getOperation());

    // Find the LaunchFuncOp that calls the target function. This will be used for the debug function name prefix
    // and also for replacement with a new LaunchFuncOp that calls the debug wrapper function.
    // If no LaunchFuncOp exists (because this does not have a raw pointer API wrapper function), fallback to
    // the target function name as the debug function name prefix.
    auto targetLaunchFnOp = GetLaunchFunctionOp(moduleOp, targetFnOp);
    auto namePrefix = targetLaunchFnOp ? targetLaunchFnOp.sym_name().str() : targetFnOp.sym_name().str();
    auto dbgFnName = std::string("_debug_") + namePrefix;

    // Create a new function op with the same arguments and return value as the target function
    //      void dbgFnOp(args, ...)
    //      {
    //          Copy output args to output targetFnArgs
    //          Call targetFnOp(targetFnArgs, ...)
    //          Run default schedule impl using (args, ...)
    //          Call utility function to check output args vs output targetFnArgs
    //          Copy output targetFnArgs to output args
    //      }
    // TODO: The last copy can be avoided if we wrap the default schedule impl within its own ValueFuncOp
    auto dbgFnOp = [&rewriter, loc, &moduleOp, &targetFnOp, &scheduleOp, dbgFnName]() -> ir::value::ValueFuncOp {
        OpBuilder::InsertionGuard guard(rewriter);

        auto funcInsertPt = ir::util::GetTerminalInsertPoint<ir::value::ValueModuleOp,
                                                             ir::value::ModuleTerminatorOp>(moduleOp);
        rewriter.restoreInsertionPoint(funcInsertPt);

        auto argTypes = targetFnOp.getType().getInputs().vec();
        auto callingFnType = rewriter.getFunctionType(argTypes, targetFnOp.getType().getResults());
        auto wrapperFnOp = rewriter.create<ir::value::ValueFuncOp>(loc, dbgFnName + "_internal", callingFnType, targetFnOp.exec_target());

        // TODO : Clone more attributes?
        if (auto dynamicArgSizeRefs = targetFnOp->getAttrOfType<mlir::ArrayAttr>(ir::DynamicArgSizeReferencesAttrName))
        {
            wrapperFnOp->setAttr(ir::DynamicArgSizeReferencesAttrName, dynamicArgSizeRefs);
        }

        rewriter.setInsertionPointToStart(&wrapperFnOp.body().front());

        BlockAndValueMapping valueMap;
        MapArguments(rewriter, targetFnOp, wrapperFnOp, valueMap);
        CreateReferenceSchedules(rewriter, scheduleOp, valueMap);

        return wrapperFnOp;
    }();

    {
        // Collect arguments for calling the target function, then inject a call to the target function
        // Output arguments will be duplicated so that we can compare the results with the reference implementation
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&dbgFnOp.body().front());

        auto targetFnArgs = GetTargetFunctionArgs(rewriter, loc, moduleOp, targetFnOp, dbgFnOp);

        // Make a call to targetFnOp with the collected arguments
        auto msg = std::string("Checking ") + std::string(targetFnOp.sym_name()) + " ...\n";
        (void)rewriter.create<ir::value::PrintFOp>(loc, msg, /*toStderr=*/false);

        (void)rewriter.create<ir::value::LaunchFuncOp>(loc, targetFnOp, targetFnArgs);
        {
            // Set insertion point past the debug nest
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(&dbgFnOp.body().front());

            // For each output arg, call its designated utility function to check that the expected values match
            auto checkFunctions = GetCheckFunctions(targetFnOp);

            for (auto [targetArg, debugArg, checkFunction] : llvm::zip(targetFnArgs, dbgFnOp.getArguments(), checkFunctions))
            {
                if (!checkFunction.empty())
                {
                    if (auto utilityFnOp = FindValueFunctionOp(moduleOp, checkFunction))
                    {
                        auto memRefType = targetArg.getType().cast<MemRefType>();
                        if (memRefType.getNumDynamicDims() > 0)
                        {
                            std::vector<mlir::Value> operands = {targetArg, debugArg};
                            auto utilityFnOpArgs = utilityFnOp.getArguments();
                            int countOfArgsToBeCopied = utilityFnOpArgs.size() - operands.size();
                            for (int i = countOfArgsToBeCopied - 1; i >= 0; i--)
                            {
                                operands.insert(operands.begin(), targetFnArgs[i]);
                            }
                            (void)rewriter.create<ir::value::LaunchFuncOp>(loc, utilityFnOp, mlir::ValueRange(operands));
                        }
                        else
                        {
                            (void)rewriter.create<ir::value::LaunchFuncOp>(loc, utilityFnOp, mlir::ValueRange{ targetArg, debugArg });
                        }
                    }

                    // Set the output arguments of this function so that the caller gets the target result
                    // TODO: This last copy can be avoided if we wrap the default schedule impl within its own ValueFuncOp
                    (void)rewriter.create<ir::value::CopyOp>(loc, targetArg, debugArg);
                }
            }

            // Finally, add the terminator
            assert(dbgFnOp.getNumResults() == 0 && "Nest functions must return no results"); // future work?
            rewriter.create<ir::value::ReturnOp>(loc);
        }
    }

    if (targetLaunchFnOp)
    {
        // Replace the original launcher with one that calls the debug wrapper function
        auto newLaunchFnOp = ir::value::CreateRawPointerAPIWrapperFunction(rewriter, dbgFnOp, targetLaunchFnOp.sym_name());

        // Propagate the base name so that aliases can be created
        if (auto baseName = targetLaunchFnOp->getAttrOfType<mlir::StringAttr>(ir::BaseNameAttrName))
        {
            newLaunchFnOp->setAttr(ir::BaseNameAttrName, baseName);
        }

        if (auto dynamicArgSizeRefs = targetLaunchFnOp->getAttrOfType<mlir::ArrayAttr>(ir::DynamicArgSizeReferencesAttrName))
        {
            newLaunchFnOp->setAttr(ir::DynamicArgSizeReferencesAttrName, dynamicArgSizeRefs);
        }
        // TODO : Clone more attributes?

        rewriter.eraseOp(targetLaunchFnOp);
    }

    targetFnOp->removeAttr(ir::GetOutputVerifiersAttrName());

    return success();
}

struct EmitDebugFunctionPattern final : public OpRewritePattern<ir::value::ValueFuncOp>
{
    using OpRewritePattern<ir::value::ValueFuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ir::value::ValueFuncOp op, PatternRewriter& rewriter) const final
    {
        if (!op->hasAttr(ir::GetOutputVerifiersAttrName()))
        {
            return failure(); // no match
        }

        return EmitNestDebugFunction(op, rewriter);
    }
};

void populateEmitDebugFunctionPatterns(OwningRewritePatternList& patterns)
{
    patterns.insert<EmitDebugFunctionPattern>(patterns.getContext());
}

struct EmitDebugFunctionPass : public accera::transforms::EmitDebugFunctionBase<EmitDebugFunctionPass>
{
    void runOnModule() override
    {
        MLIRContext* context = &getContext();
        auto moduleOp = getOperation();

        RewritePatternSet patterns(context);
        populateEmitDebugFunctionPatterns(patterns);

        (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
    }
};
} // namespace

namespace accera::transforms
{
std::unique_ptr<OperationPass<ModuleOp>> createEmitDebugFunctionPass()
{
    return std::make_unique<EmitDebugFunctionPass>();
}
} // namespace accera::transforms
