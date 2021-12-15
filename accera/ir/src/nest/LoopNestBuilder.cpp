////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs, Kern Handa, Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/LoopNestBuilder.h"

#include "IRUtil.h"

#include <utilities/include/Exception.h>

#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <type_traits>

using namespace mlir;

namespace accera::ir
{
namespace loopnest
{
    namespace
    {
        llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Index& index)
        {
            os << index.GetName() << "(" << index.GetId() << ")";
            return os;
        }

        llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Range& r)
        {
            os << "[" << r.Begin() << "," << r.End() << ":" << r.Increment() << ")";
            return os;
        }

        llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Position& p)
        {
            switch (p)
            {
            case Position::prologue:
                os << "prologue";
                break;
            case Position::body:
                os << "body";
                break;
            case Position::epilogue:
                os << "epilogue";
                break;
            }
            return os;
        }

        template <typename T>
        T GetValue(mlir::Value value)
        {
            if constexpr (std::is_integral<T>::value)
            {
                if (auto op = mlir::dyn_cast_or_null<mlir::ConstantIntOp>(const_cast<mlir::Value&>(value).getDefiningOp()))
                    return static_cast<T>(op.getValue());
                else if (auto op = mlir::dyn_cast_or_null<mlir::ConstantIndexOp>(const_cast<mlir::Value&>(value).getDefiningOp()))
                    return static_cast<T>(op.getValue());
                else
                    assert(false && "Error: got bad op type for constant int");
            }
            else if constexpr (std::is_floating_point<T>::value)
            {
                using OpType = mlir::ConstantFloatOp;
                auto op = mlir::cast<OpType>(const_cast<mlir::Value&>(value).getDefiningOp());
                return static_cast<T>(const_cast<OpType&>(op).getValue());
            }
            else
            {
                static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "Invalid type for GetValue()");
            }

            llvm_unreachable("unexpected");
            return {};
        }

        // TODO : replace this with something less hacky - surely MLIR has a better way of finding ops recursively in a graph segment already
        std::vector<SymbolicIndexOp> RecurseGetSymbolicIndexOps(Region* region)
        {
            std::vector<SymbolicIndexOp> results;
            for (auto& block : region->getBlocks())
            {
                for (auto& op : block.getOperations())
                {
                    for (auto& subRegion : op.getRegions())
                    {
                        std::vector<SymbolicIndexOp> subResults = RecurseGetSymbolicIndexOps(&subRegion);
                        for (auto& subRes : subResults)
                        {
                            results.push_back(subRes);
                        }
                    }
                    for (auto val : op.getOperands())
                    {
                        if (auto definingOp = val.getDefiningOp(); !val.isa<BlockArgument>() && definingOp)
                        {
                            if (auto indexOp = dyn_cast<SymbolicIndexOp>(definingOp))
                            {
                                auto index = indexOp.getValue();
                                results.push_back(indexOp);
                            }
                        }
                    }
                }
            }
            return results;
        }

        // check for a "placement" predicate without an index
        bool IsBodyPlacementPredicate(KernelPredicateOpInterface predicate)
        {
            return false;
        }

        bool IsUsed(const Index& index, const std::vector<ScheduledKernelOp>& activeKernels, const TransformedDomain& domain)
        {
            // TODO: extract indices from kernels and evaluate
            // for (auto k : activeKernels)
            // {
            //     for (auto kernelIndexOp : RecurseGetSymbolicIndexOps(&(k.kernel.region())))
            //     {
            //         auto kernelIndex = kernelIndexOp.getValue();
            //         if (kernelIndex == index || domain.DependsOn(kernelIndex, index))
            //         {
            //             return true;
            //         }
            //     }
            // }

            return false;
        }

        Operation* FindKernelOp(StringRef id, ScheduleOp rootOp)
        {
            auto symTableOp = mlir::SymbolTable::getNearestSymbolTable(rootOp);
            auto symbolOp = mlir::SymbolTable::lookupNearestSymbolFrom(symTableOp, id);
            assert(symbolOp && "Kernel not found");
            return symbolOp;
        }

        Region* GetKernelBodyRegion(StringRef id, ScheduleOp scheduleOp)
        {
            auto op = FindKernelOp(id, scheduleOp);
            if (!op)
            {
                scheduleOp.emitError("Expected to find " + id + " as a kernel reference");
                return nullptr;
            }

            if (auto kernelOp = dyn_cast<KernelOp>(op))
            {
                return &kernelOp.region();
            }
            else if (auto scheduledKernelOp = dyn_cast<ScheduledKernelOp>(op))
            {
                throw std::runtime_error("Got ScheduledKernelOp when we expected a ScheduledKernelOp");
            }

            return nullptr;
        }

        std::vector<ScheduledLoopOp> GetInnerLoops(Block* loopBlock)
        {
            return { loopBlock->op_begin<ScheduledLoopOp>(), loopBlock->op_end<ScheduledLoopOp>() };
        }

        std::vector<InvokeKernelOp> GetInvokeKernelOps(Block* loopBlock)
        {
            return { loopBlock->op_begin<InvokeKernelOp>(), loopBlock->op_end<InvokeKernelOp>() };
        }
    } // namespace

    //
    // RecursionState
    //
    LoopNestBuilder::RecursionState::RecursionState(const LoopNestBuilder& builder)
    {
        for (const auto& id : builder.GetKernelIds())
        {
            validKernelGroups[id] = true;
        }

        // Compute the full domain shape
        auto splitDomain = builder.GetDomain();
        auto dimensionIndices = splitDomain.GetDimensions();
        subdomainSize.reserve(dimensionIndices.size());
        for (const auto& dimensionIndex : dimensionIndices)
        {
            subdomainSize.push_back(splitDomain.GetDimensionSize(dimensionIndex));
        }
    }

    //
    // LoopRange
    //

    LoopRange::LoopRange(mlir::Value start, mlir::Value stop, mlir::Value increment) :
        start(start), stop(stop), step(increment)
    {
    }

    Range LoopRange::GetConstantRange() const
    {
        auto startInt = GetValue<int64_t>(start);
        auto stopInt = GetValue<int64_t>(stop);
        auto stepInt = GetValue<int64_t>(step);
        return { startInt, stopInt, stepInt };
    }

    //
    // LoopNestBuilder
    //
    LoopNestBuilder::LoopNestBuilder(ScheduleOp op, mlir::PatternRewriter& builder, bool printLoops) :
        _schedule(op),
        _builder(builder),
        _constantOpBuilder(mlir::OpBuilder::atBlockBegin(builder.getBlock())),
        _printLoops(printLoops)
    {
        _kernelGroups = DiscoverKernelGroups();
    }

    void LoopNestBuilder::BuildLoopNest()
    {
        auto schedule = GetLoopSchedule();
        auto initialIndex = schedule.CurrentLoopIndex();

        // We need to create a RecursionState object here, because it's passed in as a reference
        RecursionState state(*this);

        GenerateLoopStructure(state, schedule);
        MergeAdjacentKernelBodies(_loops[initialIndex], schedule);
        EmitLoopBodies(_loops[initialIndex], state, schedule);

        ApplyInjectableMappings();

        EnsureTerminators();
    }

    LoopVisitSchedule LoopNestBuilder::GetLoopSchedule() const
    {
        std::vector<IndexRange> indexRanges;
        auto domain = GetDomain();

        auto loopSequence = const_cast<ScheduleOp&>(_schedule).getOrder();
        for (auto loopIndex : loopSequence)
        {
            auto range = domain.GetIndexRange(loopIndex);
            indexRanges.push_back(IndexRange{ loopIndex, range });
        }

        return { indexRanges };
    }

    LoopNestBuilder::RecursionState LoopNestBuilder::GenerateLoopStructure(const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        if (schedule.IsDone())
        {
            return state;
        }

        // 1) get list of prologue and epilogue kernels we'll hoist up to this level
        // 2) get splits/partitions for this loop range
        // 3) eval predicates and mark valid regions
        // 4) make a list of kernels that can possibly run for each partition (e.g., [1,2 | 2 | 2, 3])
        // 5) move adjacent fully-matching suffix on left into right partition (and expand)
        // 6) move adjacent fully-matching prefix on right into left partition (and expand)

        // ex, with S1: first(i), S2: all, S3: last(i):

        // step 1: partitions: (0..1), (1..N-1), (N-1..N)
        // step 2: partitions w/ kernels: (0..1: S1, S2, S3), (1..N-1: S1, S2, S3), (N-1..N: S1, S2, S3)
        // step 3: eval predicates and remove kernels: (0..1: S1, S2), (1..N-1: S2), (N-1..N: S2, S3)
        // step 4: ...
        // step 5: Suffix of first partition matches entirety of second: move
        //         --> (0..1: S1), (0..N-1: S2), (N1-..N: S2, S3)
        // step 6: prefix of last partition matches entirety of second: move
        //         --> (0..1: S1), (0..N: S2), (N1-..N: S3)

        auto loopIndex = schedule.CurrentLoopIndex();
        // Find the active range for the current loop dimension and reduce the end amount if it exceeds the active range (boundary case)
        auto fullRange = schedule.GetActiveLoopRange(GetDomain(), loopIndex, state.loopIndices);
        auto partitions = GetPartitions(loopIndex, fullRange, state, schedule);

        auto newState = state;
        for (const auto& p : partitions)
        {
            auto partitionRange = MakeLoopRange(_builder, p.range.Begin(), p.range.End(), p.range.Increment());
            UpdateSubdomainSizes(loopIndex, partitionRange, newState.subdomainSize);

            auto loop = EmitLoopOp(partitionRange, newState, schedule); // This creates the (empty) loop op
            _loops[loopIndex].push_back(loop);

            auto loopIndexOp = loop.getSymbolicIndex();
            newState.loopIndices.insert_or_assign(loopIndex, LoopIndexSymbolTableEntry{ loopIndexOp, partitionRange.GetConstantRange(), LoopIndexState::inProgress });
            GenerateLoopBody(loopIndexOp, partitionRange, newState, schedule); // This contains the recursive call to GenerateLoopStructure
            EndLoopRange(partitionRange, newState, schedule);
        }

        // set the loop index state to be "done"
        DefinePostLoopIndex(_builder, loopIndex, newState.loopIndices, schedule);
        return newState;
    }

    // TODO: move the loop-printing until after merging
    void LoopNestBuilder::MergeAdjacentKernelBodies(std::vector<ScheduledLoopOp> loops, const LoopVisitSchedule& schedule)
    {
        if (schedule.IsDone())
        {
            return;
        }

        // Do recursive call first
        for (auto loop : loops)
        {
            auto innerLoops = GetInnerLoops(loop.getBody());
            MergeAdjacentKernelBodies(innerLoops, schedule.Next());
        }

        // Now (maybe) merge partitions for this loop
        const bool mergePartitions = true;
        if (mergePartitions)
        {
            ScheduledLoopOp prevLoop = nullptr;
            for (auto loopIt = loops.begin(), nextIt = loops.begin() + 1; loopIt != loops.end(); ++loopIt)
            {
                auto thisLoop = *loopIt;
                auto nextLoop = nextIt == loops.end() ? nullptr : *nextIt;

                auto thisLoopKernels = GetInvokeKernelOps(thisLoop.getBody());
                if (thisLoopKernels.size() >= 0)
                {
                    if (prevLoop)
                    {
                        auto prevLoopKernels = GetInvokeKernelOps(prevLoop.getBody());
                        if (prevLoop.getNumIterations() == 1 &&
                            prevLoopKernels.size() > thisLoopKernels.size() &&
                            std::equal(thisLoopKernels.rbegin(), thisLoopKernels.rend(), prevLoopKernels.rbegin(), [](auto k1, auto k2) { return k1.kernel() == k2.kernel(); }))
                        {
                            thisLoop.setBegin(prevLoop.begin());
                            for (auto it = prevLoopKernels.rbegin(); it != prevLoopKernels.rbegin() + thisLoopKernels.size(); ++it)
                            {
                                _builder.eraseOp(*it);
                            }
                        }
                    }

                    if (nextLoop)
                    {
                        auto nextLoopKernels = GetInvokeKernelOps(nextLoop.getBody());
                        // Special case where this and next loops both have 1 iteration: make new loop with union of ranges for the common suffix of "this" and prefix of "next"

                        if (thisLoop.getNumIterations() == 1 && nextLoop.getNumIterations() == 1 && thisLoopKernels.size() > 0 && nextLoopKernels.size() > 0)
                        {
                            auto matchVal = nextLoopKernels[0];
                            auto matchStart = std::find_if(thisLoopKernels.begin(), thisLoopKernels.end(), [&matchVal](auto k) { return k.kernel() == matchVal.kernel(); });
                            if (matchStart != thisLoopKernels.end())
                            {
                                if (std::equal(matchStart, thisLoopKernels.end(), nextLoopKernels.begin(), [](auto k1, auto k2) { return k1.kernel() == k2.kernel(); }))
                                {
                                    auto newPartition = Range(thisLoop.begin(), nextLoop.end(), thisLoop.step());
                                    auto numBadPrefixKernels = matchStart - thisLoopKernels.begin();
                                    auto numMovedKernels = thisLoopKernels.end() - matchStart;
                                    auto builder = GetCurrentLoopBuilder(schedule);
                                    OpBuilder::InsertionGuard guard(builder);
                                    builder.setInsertionPointAfter(thisLoop);
                                    auto newLoop = cast<ScheduledLoopOp>(builder.clone(*thisLoop.getOperation()));
                                    newLoop.setEnd(nextLoop.end());
                                    auto loopIndex = schedule.CurrentLoopIndex();
                                    auto partitionRange = MakeLoopRange(builder, newPartition.Begin(), newPartition.End(), newPartition.Increment());
                                    auto subdomainSizes = newLoop.getSubdomainSize();
                                    UpdateSubdomainSizes(loopIndex, partitionRange, subdomainSizes);
                                    newLoop.setSubdomainSize(subdomainSizes);
                                    auto newLoopKernels = GetInvokeKernelOps(newLoop.getBody());

                                    std::vector<InvokeKernelOp> opsToDelete;
                                    opsToDelete.insert(opsToDelete.end(), matchStart, thisLoopKernels.end());
                                    opsToDelete.insert(opsToDelete.end(), newLoopKernels.begin(), newLoopKernels.begin() + numBadPrefixKernels);
                                    opsToDelete.insert(opsToDelete.end(), nextLoopKernels.begin(), nextLoopKernels.begin() + numMovedKernels);

                                    for (auto op : opsToDelete)
                                    {
                                        _builder.eraseOp(op);
                                    }
                                }
                            }
                        }
                        else if (nextLoop.getNumIterations() == 1 &&
                                 nextLoopKernels.size() > thisLoopKernels.size() &&
                                 std::equal(thisLoopKernels.begin(), thisLoopKernels.end(), nextLoopKernels.begin(), [](auto k1, auto k2) { return k1.kernel() == k2.kernel(); }))
                        {
                            thisLoop.setEnd(nextLoop.end());
                            for (auto it = nextLoopKernels.begin(); it != nextLoopKernels.begin() + thisLoopKernels.size(); ++it)
                            {
                                _builder.eraseOp(*it);
                            }
                        }
                    }
                }

                prevLoop = thisLoop;

                if (loopIt != loops.end() - 1) // avoid going out of bounds
                {
                    ++nextIt;
                }
            }
        }
    }

    LoopNestBuilder::RecursionState LoopNestBuilder::EmitLoopBodies(std::vector<ScheduledLoopOp> loops, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        auto newState = state;
        auto loopIndex = schedule.CurrentLoopIndex();

        for (auto loop : loops)
        {
            auto partitionRange = MakeLoopRange(_builder, loop.begin(), loop.end(), loop.step());
            auto loopIndexOp = loop.getSymbolicIndex();
            newState.loopIndices.insert_or_assign(loopIndex, LoopIndexSymbolTableEntry{ loopIndexOp, partitionRange.GetConstantRange(), LoopIndexState::inProgress });
            EmitLoopBody(loop, newState, schedule); // This contains the recursive call to EmitLoopBodies
        }

        // set the loop index state to be "done"
        DefinePostLoopIndex(_builder, loopIndex, newState.loopIndices, schedule); // #### not sure this is needed
        return newState;
    }

    void LoopNestBuilder::ApplyInjectableMappings()
    {
        auto loc = GetLocation();
        auto registeredInjectableMappings = GetScheduleOp().getInjectableMappings();
        for (auto& mapping : registeredInjectableMappings)
        {
            auto scheduledLoopOps = FindAllScheduledLoops(mapping.index());

            for (auto& scheduledLoopOp : scheduledLoopOps)
            {
                mlir::OpBuilder::InsertionGuard insertGuard(_builder);

                BlockAndValueMapping operandMap;
                _builder.setInsertionPoint(scheduledLoopOp);
                auto clonedBeginOp = _builder.clone(*(mapping.getOperation()), operandMap);

                _builder.setInsertionPointAfter(scheduledLoopOp);
                auto clonedEndOp = _builder.clone(*(mapping.getInjectionEndOp()), operandMap);
            }

            if (mapping.use_empty())
            {
                _builder.eraseOp(mapping);
            }
        }
    }

    // This function figures out the correct place to put the insertion point, but makes no guarantees about where it leaves it
    ScheduledLoopOp LoopNestBuilder::EmitLoopOp(const LoopRange& range, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        auto loopIndex = schedule.CurrentLoopIndex();
        auto constRange = range.GetConstantRange();
        if (_printLoops)
        {
            llvm::errs().indent(2 * schedule.CurrentLoopLevel()) << "for " << loopIndex << " in " << constRange << " {";
            if (constRange.NumIterations() == 1)
            {
                llvm::errs() << " -- single iteration";
            }
            llvm::errs() << "\n";
        }

        // Emit a ScheduledLoopOp
        auto builder = GetCurrentLoopBuilder(schedule);

        int begin = constRange.Begin();
        int end = constRange.End();
        int step = constRange.Increment();

        auto loc = GetLocation();
        auto symbolicIndex = GetSymbolicIndex(loopIndex);
        assert(symbolicIndex && "Error: bad symbolic index");
        auto domain = GetDomain();
        auto domainIndexOrder = domain.GetDimensions();
        auto loop = builder.create<ScheduledLoopOp>(loc, begin, end, step, symbolicIndex, state.subdomainSize, domainIndexOrder);

        // TODO : move these attributes to the loop attribute infra
        loop->setAttr("index", IndexAttr::get(loopIndex, builder.getContext()));
        if (auto val = GetUnrollIfRangeSmallerThan(loopIndex))
        {
            if (constRange.NumIterations() < (int64_t)*val)
            {
                loop->setAttr("rcv_unrolled", builder.getUnitAttr());
            }
        }

        if (auto val = GetUnrollAndJamFactor(loopIndex))
        {
            loop->setAttr("rcv_unroll_jam", builder.getI64IntegerAttr((int64_t)*val));
        }

        if (IsSaturated(loopIndex))
        {
            loop->setAttr("rcv_saturated", builder.getUnitAttr());
        }

        auto execPlan = GetScheduleOp().getOrCreateExecPlan();
        auto target = execPlan.exec_target();
        switch (target)
        {
        default:
        case ir::value::ExecutionTarget::CPU:
            break;
        case ir::value::ExecutionTarget::GPU:
            if (auto dictAttr = execPlan->getAttrOfType<DictionaryAttr>(execPlan.getGPUProcessorMapAttrName()))
            {
                for (auto [key, val] : dictAttr.getValue())
                {
                    auto indexAttr = val.dyn_cast<IndexAttr>();
                    if (loopIndex.GetId() == indexAttr.getValue().GetId())
                    {
                        loop->setAttr("rcv_gpu_map", builder.getStringAttr(key.str()));
                    }
                }
            }

            break;
        }

        // Carry forward any loop attributes tagged for this index
        auto loopAttrDict = _schedule.getLoopAttributes(loopIndex);
        if (loopAttrDict)
        {
            // We have attributes for this loop, so transfer them over to the ScheduledLoopOp
            auto loopAttrs = loopAttrDict->getValue();
            for (auto& [identifier, value] : loopAttrs)
            {
                loop->setAttr(identifier, value);
            }
        }

        return loop;
    }

    // This function gets run after the loop body is generatoed
    void LoopNestBuilder::EndLoopRange(const LoopRange& range, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        if (_printLoops)
        {
            llvm::errs().indent(2 * schedule.CurrentLoopLevel()) << "}\n";
        }
    }

    void LoopNestBuilder::GenerateLoopBody(mlir::Value index, const LoopRange& r, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        auto newState = state;
        auto loopIndex = schedule.CurrentLoopIndex();

        if (schedule.IsInnermostLoop())
        {
            InvokeKernels(r, Position::body, newState, schedule);
        }
        else
        {
            auto innerState = InvokeKernels(r, Position::prologue, newState, schedule);

            // Recursively call GenerateLoopStructure to generate the more-deeply-nested loops
            auto epilogueState = GenerateLoopStructure(innerState, schedule.Next());
            auto outerState = InvokeKernels(r, Position::epilogue, epilogueState, schedule);

            DefinePostLoopIndex(_builder, loopIndex, newState.loopIndices, schedule);
        }
    }

    void LoopNestBuilder::EmitLoopBody(ScheduledLoopOp loop, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        auto newState = state;
        auto loopIndex = schedule.CurrentLoopIndex();

        if (schedule.IsInnermostLoop())
        {
            ReplaceInvokeOps(loop.getBody(), newState);
        }
        else
        {
            ReplaceInvokeOps(loop.getPrologue(), newState); // prologue kernels
            auto innerLoops = GetInnerLoops(loop.getBody());

            // Recursively call EmitLoopBodies to emit the more-deeply-nested loops
            auto epilogueState = EmitLoopBodies(innerLoops, newState, schedule.Next());
            ReplaceInvokeOps(loop.getEpilogue(), epilogueState);

            DefinePostLoopIndex(_builder, loopIndex, newState.loopIndices, schedule);
        }
    }

    LoopNestBuilder::RecursionState LoopNestBuilder::InvokeKernels(const LoopRange& r, Position position, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        auto newState = state;
        if (position != Position::epilogue)
        {
            auto kernels = GetPossiblyValidKernels(state);
            DefineComputedIndexVariables(newState.loopIndices, kernels, schedule);
        }

        // TODO: need to iterate this in order
        for (const auto& id : GetPossiblyValidKernelIds(newState))
        {
            auto invoked = InvokeKernelGroup(id, position, newState.loopIndices, schedule);
            if (invoked)
            {
                newState.validKernelGroups[id] = false;
            }
        }

        return newState;
    }

    void LoopNestBuilder::ReplaceInvokeOps(Block* loopBlock, const RecursionState& state)
    {
        auto invokeOps = GetInvokeKernelOps(loopBlock);
        for (auto invokeOp : invokeOps)
        {
            auto builder = OpBuilder(loopBlock, std::prev(loopBlock->end()));
            EmitKernelBody(builder, invokeOp, state.loopIndices);
        }

        for (auto op : invokeOps)
        {
            op.erase();
        }
    }

    std::set<KernelId> LoopNestBuilder::GetEpilogueKernelIds(const LoopRange& r, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        std::set<KernelId> epilogueKernels;
        auto afterBodyState = GenerateAfterBodyState(r, state, schedule);
        for (const auto& id : GetPossiblyValidKernelIds(afterBodyState))
        {
            auto kernelGroup = GetKernelGroup(id);
            auto validKernels = GetValidKernels(kernelGroup, afterBodyState.loopIndices, schedule, Position::epilogue);
            auto invoked = !validKernels.empty();
            if (invoked)
            {
                epilogueKernels.insert(id);
            }
        }

        return epilogueKernels;
    }

    std::vector<std::string> LoopNestBuilder::GetPossiblyValidKernelIds(const RecursionState& state) const
    {
        auto ids = GetKernelIds();
        std::vector<std::string> validIds;
        std::copy_if(ids.begin(), ids.end(), std::back_inserter(validIds), [&](auto id) { return state.validKernelGroups.at(id); });
        return ids;
    }

    std::vector<ScheduledKernelOp> LoopNestBuilder::GetPossiblyValidKernels(const RecursionState& state) const
    {
        std::vector<ScheduledKernelOp> kernels;
        for (const auto& id : GetPossiblyValidKernelIds(state))
        {
            auto kernelGroup = GetKernelGroup(id);
            kernels.insert(kernels.end(), kernelGroup.begin(), kernelGroup.end());
        }

        return kernels;
    }

    bool LoopNestBuilder::InvokeKernelGroup(std::string id, Position position, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule)
    {
        auto kernelGroup = GetKernelGroup(id);
        auto loopIndex = schedule.CurrentLoopIndex();
        auto scheduledLoop = FindLatestScheduledLoop(loopIndex);

        // preprocess to get only valid kernels
        std::vector<ScheduledKernelOp> validKernels = GetValidKernels(kernelGroup, runtimeIndexVariables, schedule, position);

        if (validKernels.empty())
        {
            return false;
        }

        std::vector<Index> definedIndices;
        for (auto [key, value] : runtimeIndexVariables)
        {
            definedIndices.push_back(key);
        }

        OpBuilder builder = [&]() {
            switch (position)
            {
            case Position::prologue:
                return scheduledLoop.getPrologueBuilder();
            case Position::body:
                return scheduledLoop.getBodyBuilder();
            case Position::epilogue:
                return scheduledLoop.getEpilogueBuilder();
            default:
                throw std::runtime_error("Illegal Position type");
            }
        }();

        for (auto kernel : validKernels)
        {
            auto predicate = kernel.getPredicate();
            if (predicate == nullptr || isa<NullPredicateOp>(predicate))
            {
                if (position == Position::body)
                {
                    InvokeKernel(builder, kernel, position, runtimeIndexVariables, schedule);
                    return true;
                }
            }
            else
            {
                bool predicateResult = false;
                // TODO: Move to TypeSwitch
                if (auto castPredicate = dyn_cast<NullPredicateOp>(predicate))
                {
                    predicateResult = schedule.IsInnermostLoop();
                }
                else if (auto castPredicate = dyn_cast<KernelPredicateOpInterface>(predicate))
                {
                    auto result = castPredicate.evaluate(GetDomain(), runtimeIndexVariables, schedule);
                    if (result.has_value())
                        predicateResult = *result;
                }
                else if (auto castPredicate = dyn_cast<EvaluatablePredicateOpInterface>(predicate))
                {
                    predicateResult = castPredicate.evaluate(definedIndices, loopIndex, position);
                }

                if (predicateResult)
                {
                    InvokeKernel(builder, kernel, position, runtimeIndexVariables, schedule);
                    return true;
                }
            }
        }

        return false;
    }

    LoopNestBuilder::RecursionState LoopNestBuilder::GenerateAfterBodyState(const LoopRange& r, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        // get each inner index and set its state to 'done'
        auto newState = state;
        auto s = schedule.Next();
        while (!s.IsDone())
        {
            auto innerLoopIndex = s.CurrentLoopIndex();
            // Make sure _builder's insertion point is correctly set (?)
            DefinePostLoopIndex(_builder, innerLoopIndex, newState.loopIndices, s);
            s = s.Next();
        }
        return newState;
    }

    PartitionList LoopNestBuilder::GetPartitions(const Index& loopIndex, Range loopRange, const RecursionState& state, const LoopVisitSchedule& schedule) const
    {
        int begin = loopRange.Begin();
        int end = loopRange.End();
        int increment = loopRange.Increment();

        // Find conditions involving this index and add any relevant partition split points
        std::set<int64_t> splits;
        for (auto k : GetPossiblyValidKernels(state))
        {
            AddSplits(loopIndex, loopRange, k.getPredicate(), state.loopIndices, schedule, splits);
        }

        // Get index range
        PartitionList result;
        for (auto partitionEnd : splits)
        {
            result.push_back({ loopIndex, { begin, partitionEnd, increment } });
            begin = partitionEnd;
        }
        result.push_back({ loopIndex, { begin, end, increment } });

        return result;
    }

    void LoopNestBuilder::AddSplits(const Index& loopIndex, Range loopRange, Operation* predicateOp, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule, std::set<int64_t>& allSplits) const
    {
        // For some reason, simplifying a FragmentTypePredicate here erroneously returns a constant (empty) predicate
        // auto predicate = predicateOp.Simplify(GetDomain(), runtimeIndexVariables, schedule);

        const auto& domain = GetDomain();

        auto addTransformationSplits = [&domain, &loopIndex, &loopRange, &runtimeIndexVariables, &schedule](std::set<int64_t>& splits) -> void {
            // Apply splits resulting from schedule transformations on indices
            if (auto skewedOrReference = domain.IsSkewedOrReferenceIndex(loopIndex))
            {
                auto [isSkewedIndex, dependentIndex] = *skewedOrReference;
                if (!schedule.WasIterationVariableDefined(dependentIndex))
                {
                    // Two cases:
                    //  1. Partial unswitched conditions:
                    //       - Skewed index is ordered before the reference index
                    //       - Range(indexBeforeSkew) > 2 * Range(referenceIndex)
                    //
                    //      Limitation: When this condition is true the skewed index updates the *block* range
                    //      of the reference index. This means that each reference index should only be applied *once*
                    //      per skew transformation. Otherwise the behavior is undefined when (multiple outer skew indices
                    //      of different dimensions will result in conflicting reference index ranges).
                    //      Typically, for convolution, this is not an issue because each filter's dimension maps 1:1 to
                    //      an output's dimension (e.g. we usually don't convolve a 2D output with a 1D filter.)
                    //
                    //  2. Otherwise: fully unswitched - all dependent range begin/end tuples are unique per loopIndex value
                    auto dependentIndexRange = domain.GetIndexRange(dependentIndex);

                    // Calculate range of possible split points.
                    auto begin = loopRange.Begin() + loopRange.Increment();
                    auto end = loopRange.End(); // TODO: need to handle boundary conditions?

                    // S = N + M - 1 => N = S - M + 1
                    auto rangeBeforeSkew = loopRange.Size() - dependentIndexRange.Size() + 1;
                    if (isSkewedIndex && rangeBeforeSkew > dependentIndexRange.Size())
                    {
                        // Partially unswitched
                        // first triangle region
                        auto s = begin;
                        for (; s < dependentIndexRange.Size(); s += loopRange.Increment())
                        {
                            splits.insert(s);
                        }
                        // rectangle region
                        splits.insert(s);

                        // second triangle region
                        for (s = rangeBeforeSkew - loopRange.Increment(); s < end; s += loopRange.Increment())
                        {
                            splits.insert(s);
                        }
                    }
                    else
                    {
                        // Fully unswitched: every index value is a split point resulting in a unique Range
                        // for the dependent index added deeper in the loop
                        // (This logic is separated out for clarity)
                        //
                        //  Before skew:                 After skew (fully unswitched):
                        //  x x x x x                     x
                        //  x x x x x                     x x
                        //  x x x x x                     x x x
                        //  x x x x x                     x x x x    <-- triangle = height M-1
                        //  x x x x x  <-- N <= M         x x x x x  <-- horizontal strip = height 1
                        //                                  x x x x
                        //                                    x x x
                        //                                      x x
                        //                                        x  <-- triangle = height M-1
                        for (auto s = begin; s < end; s += loopRange.Increment())
                        {
                            splits.insert(s);
                        }
                    }
                } // else no splits needed, loopRange is already adjusted by LoopVisitSchedule::GetActiveLoopRange
            }
            else if (domain.IsSplitIndex(loopIndex, /*inner=*/false))
            {
                // An outer split index: some loops need to be unswitched depending on whether padding exists
                //
                //  Suppose padSize = 8, splitSize = 5 (padSize can easily be 3 as well):
                //  o o o o o
                //  o o o x x <-- unswitch at first ceiling boundary [padded outer indices]
                //  x x x x x
                //  x x x x x
                //  x x x x x
                //  x x x o o <-- unswitch at (rangeSize - extra) [all outer indices, if applicable]

                auto splitSize = loopRange.Increment();
                if (domain.HasPaddedParentIndex(loopIndex))
                {
                    // This is an outer split index that has a padded parent index
                    // As outer split indices are only split once, we've only needed to check the immediate parent
                    auto padSize = loopRange.Begin();
                    if (padSize != 0 && padSize != splitSize)
                    {
                        // Unswitch the first partial block, at the first ceiling boundary
                        // (i.e. the split boundary that immediately follows the padSize element)
                        auto firstCeilBound = (padSize / splitSize + 1) * splitSize;
                        if (firstCeilBound < loopRange.End())
                        {
                            splits.insert(firstCeilBound);
                        }
                    } // else front padding does not affect the size of the first split block
                }

                // Unswitch the final partial block if the range exceeds the split size AND is a non-multiple of the split size
                auto rangeSize = loopRange.End();
                auto extra = rangeSize % splitSize;
                if (rangeSize > splitSize && extra > 0)
                {
                    splits.insert(rangeSize - extra);
                }
            }
        };

        std::function<void(Operation*, std::set<int64_t>&)> proposeSplits =
            [&proposeSplits, &domain, &loopIndex, &loopRange](Operation* p, std::set<int64_t>& splits) -> void {
            if (auto fragmentTypePred = dyn_cast_or_null<FragmentTypePredicateOp>(p))
            {
                auto where = fragmentTypePred.fragment();
                if (where != FragmentType::all)
                {
                    auto predIndex = fragmentTypePred.index().cast<IndexAttr>().getValue();
                    if (predIndex == loopIndex || domain.DependsOn(predIndex, loopIndex))
                    {
                        std::vector<int64_t> splitVals;
                        auto indexValues = fragmentTypePred.getIndexValues();
                        switch (fragmentTypePred.fragment())
                        {
                        case FragmentType::first:
                            splitVals.push_back(loopRange.Begin() + loopRange.Increment());
                            break;
                        case FragmentType::last: {
                            // take into account last range being a boundary condition
                            auto extra = loopRange.End() % loopRange.Increment();
                            if (extra == 0)
                            {
                                splitVals.push_back(loopRange.End() - loopRange.Increment());
                            }
                            else
                            {
                                splitVals.push_back(loopRange.End() - extra);
                            }
                            break;
                        }
                        case FragmentType::select:
                            // single select value
                            // Note: not a special-case of FragmentType::range because it requires different
                            // treatment of split points to candidate ranges during evaluation (see addValidSplits below).
                            assert(indexValues.size() == 1 && "Invalid number of index values for select predicate");
                            splitVals.push_back(indexValues[0]);
                            break;
                        case FragmentType::endBoundary:
                            // already set by automatic boundary-handling code
                            break;
                        case FragmentType::range:
                            // custom begin & end (increment is ignored for now)
                            assert(indexValues.size() >= 2 && "Invalid number of index values for range predicate");
                            splitVals.push_back(indexValues[0]);
                            splitVals.push_back(indexValues[1]);
                            break;
                        default:
                            // nothing
                            break;
                        }

                        for (auto splitVal : splitVals)
                        {
                            if (splitVal > 0 && splitVal < loopRange.End())
                            {
                                splits.insert(splitVal);
                            }
                        }
                    }
                }
            }
            else if (auto indexDefinedPred = dyn_cast_or_null<IndexDefinedPredicateOp>(p))
            {
                // nothing
            }
            else if (auto conjunction = dyn_cast_or_null<ConjunctionPredicateOp>(p))
            {
                for (auto t : conjunction.values())
                {
                    proposeSplits(t.getDefiningOp(), splits);
                }
            }
            else if (auto disjunction = dyn_cast_or_null<DisjunctionPredicateOp>(p))
            {
                for (auto t : disjunction.values())
                {
                    proposeSplits(t.getDefiningOp(), splits);
                }
            }
        };

        std::function<bool(Operation * p, FragmentType type)> containsFragmentPredicate = [&containsFragmentPredicate](Operation* p, FragmentType type) -> bool {
            bool result = false;
            if (auto fragPred = dyn_cast<FragmentTypePredicateOp>(p);
                fragPred && fragPred.fragment() == type)
            {
                result = true;
            }
            else if (auto conjunction = dyn_cast_or_null<ConjunctionPredicateOp>(p))
            {
                for (auto t : conjunction.values())
                {
                    result = result && containsFragmentPredicate(t.getDefiningOp(), type);
                }
            }
            else if (auto disjunction = dyn_cast_or_null<DisjunctionPredicateOp>(p))
            {
                for (auto t : disjunction.values())
                {
                    result = result || containsFragmentPredicate(t.getDefiningOp(), type);
                }
            }
            return result;
        };

        auto addValidSplits = [&domain, &loopIndex, &loopRange, &runtimeIndexVariables, &schedule, containsFragmentPredicate](Operation* p, const std::set<int64_t>& splits, std::set<int64_t>& allSplits) -> void {
            auto pred = dyn_cast_or_null<KernelPredicateOpInterface>(p);
            if (pred)
            {
                // There isn't an entry in the symbol table yet, so we have to add a bogus ones
                auto symbolTable = runtimeIndexVariables;

                auto entry = LoopIndexSymbolTableEntry{ nullptr, loopRange, LoopIndexState::inProgress };
                symbolTable.insert_or_assign(loopIndex, entry);

                int64_t prevBegin = loopRange.Begin();
                entry.loopRange = { prevBegin, loopRange.End(), loopRange.Increment() };
                symbolTable.insert_or_assign(loopIndex, entry);
                auto prevVal = pred.evaluate(domain, symbolTable, schedule);

                const bool hasSelectFragment = containsFragmentPredicate(p, FragmentType::select);
                const bool hasRangeFragment = containsFragmentPredicate(p, FragmentType::range);

                for (auto it = splits.begin(); it != splits.end(); ++it)
                {
                    // eval at proposed split
                    auto splitPt = *it;
                    std::optional<bool> splitVal;

                    // evaluate() should handle conjunctions and disjunctions appropriately
                    // we just try all possible candidate ranges to see if any are valid
                    std::vector<Range> candidateRanges;
                    if (hasSelectFragment)
                        candidateRanges.push_back({ splitPt, splitPt + 1, loopRange.Increment() }); // index == splitPt
                    if (hasRangeFragment)
                        candidateRanges.push_back({ prevBegin, splitPt, loopRange.Increment() }); // prevSplitPt <= index < splitPt

                    candidateRanges.push_back({ splitPt, loopRange.End(), loopRange.Increment() }); // default one to try

                    for (const auto& candidateRange : candidateRanges)
                    {
                        entry.loopRange = candidateRange;
                        symbolTable.insert_or_assign(loopIndex, entry);
                        splitVal = splitVal.value_or(false) || pred.evaluate(domain, symbolTable, schedule);
                    }

                    if (splitVal.value_or(false))
                    {
                        allSplits.insert(splitPt);
                        prevBegin = splitPt;
                        prevVal = splitVal;
                    }
                }

                // take care of end case
                auto lastSplit = *(splits.rbegin());
                if (prevBegin != lastSplit && prevBegin != loopRange.Begin())
                {
                    allSplits.insert(prevBegin);
                }
            }
        };

        addTransformationSplits(allSplits);

        std::set<int64_t> splits;
        proposeSplits(predicateOp, splits);

        // Now add value new splits to incoming set
        if (splits.size() > 0)
            addValidSplits(predicateOp, splits, allSplits);
    }

    void LoopNestBuilder::UpdateSubdomainSizes(const Index& loopIndex, const LoopRange& range, std::vector<int64_t>& subdomainSize)
    {
        auto constantRange = range.GetConstantRange();
        auto partitionIncrement = std::min(constantRange.Size(), constantRange.Increment());
        for (auto pos : GetLogicalDimensionPositions(loopIndex))
        {
            subdomainSize[pos] = partitionIncrement;
        }
    }

    void LoopNestBuilder::DefineComputedIndexVariables(LoopIndexSymbolTable& indexVariables, const std::vector<ScheduledKernelOp>& activeKernels, const LoopVisitSchedule& schedule)
    {
        const auto& domain = GetDomain();

        // define all computed index variables (that are used)
        std::set<Index> usedIndices;
        for (auto d : domain.GetDimensions())
        {
            auto computedIndices = domain.GetComputedIndicesForDimension(d);
            for (auto index : computedIndices)
            {
                if (IsUsed(index, activeKernels, domain))
                {
                    usedIndices.insert(index);
                }
            }
        }

        for (const auto& index : usedIndices)
        {
            // NOTE: this never currently runs, because `usedIndices` is always empty
            auto expr = GetIndexExpression(_builder, index, domain);
            auto indexValue = EmitIndexExpression(_builder, _schedule.getLoc(), expr, domain);
            indexVariables.insert_or_assign(index, LoopIndexSymbolTableEntry{ indexValue, Range{ 0, 0, 0 }, LoopIndexState::inProgress });
        }
    }

    bool LoopNestBuilder::IsPlacementValid(ScheduledKernelOp kernel, const LoopIndexSymbolTable& runtimeLoopIndices, const LoopVisitSchedule& schedule, const Position& position) const
    {
        const auto& domain = GetDomain();
        auto placementPredicate = dyn_cast_or_null<KernelPredicateOpInterface>(kernel.getPlacementPredicate());
        auto isEmpty = !kernel.getPlacementPredicate() || static_cast<bool>(dyn_cast_or_null<NullPredicateOp>(kernel.getPlacementPredicate()));

        if (isEmpty || IsBodyPlacementPredicate(placementPredicate))
        {
            // TODO: doesn't need to be innermost, just inner enough that none of the "regular" predicates depend on any inner loops

            // HACK: check if there's an EvaluatablePredicate, to allow non-innermost "body" kernels with such a predicate
            //
            // TODO: merge the EvaluatablePredicate interface with the old "placement" predicate (or just replace the
            //       "placement" predicate concept with EvaluatablePredicate)
            auto hasEvaluatablePredicate = kernel.getPredicate() != nullptr && isa<EvaluatablePredicateOpInterface>(kernel.getPredicate());
            if (!schedule.IsInnermostLoop())
            {
                if (position == Position::body || !hasEvaluatablePredicate)
                {
                    return false;
                }
            }

            // TODO: put this in a function that preprocesses the kernel predicates when adding the kernels to the schedule
            // auto indices = kernel.kernel.GetIndices();
            std::vector<Index> indices;
            for (const auto& kernelIndex : indices)
            {
                for (const auto& loopIndex : domain.GetDependentLoopIndices(kernelIndex, true))
                {
                    // if not defined(loopIndex) return false;
                    if (runtimeLoopIndices.count(loopIndex) == 0 || runtimeLoopIndices.at(loopIndex).state == LoopIndexState::done)
                    {
                        return false;
                    }
                }
            }

            if (isEmpty)
            {
                return true;
            }
        }

        std::function<bool(Operation*)> evalPlacement = [&](Operation* p) -> bool {
            if (false /*p.IsAlwaysTrue()*/)
            {
                return true;
            }

            if (auto constantPred = dyn_cast_or_null<ConstantPredicateOp>(p))
            {
                return constantPred.value();
            }
            else if (auto fragmentTypePred = dyn_cast_or_null<FragmentTypePredicateOp>(p))
            {
                throw std::runtime_error("Fragment predicates not valid for placement");
            }
            else if (auto placementPred = dyn_cast_or_null<PlacementPredicateOp>(p))
            {
                if (schedule.IsInnermostLoop())
                {
                    // return !placementPred->HasIndex();
                    // TODO: change placement pred to have an optional index
                    return false;
                }

                auto nextLoopIndex = schedule.Next().CurrentLoopIndex();
                auto where = placementPred.placement();

                std::vector<Index> dependentLoopIndices;
                // if (placementPred->HasIndex())
                if (true)
                {
                    auto testIndex = placementPred.index().cast<IndexAttr>().getValue();

                    // get list of dependent indices
                    dependentLoopIndices = domain.GetDependentLoopIndices(testIndex, true);

                    // First check that we're not already inside any dependent loops
                    for (const auto& i : dependentLoopIndices)
                    {
                        if (runtimeLoopIndices.count(i) != 0 && runtimeLoopIndices.at(i).state == LoopIndexState::inProgress)
                        {
                            return false;
                        }
                    }
                }
                else
                {
                    dependentLoopIndices = { nextLoopIndex };
                }

                // Now check that the next loop at least partially defines the index in question
                if (std::find(dependentLoopIndices.begin(), dependentLoopIndices.end(), nextLoopIndex) != dependentLoopIndices.end())
                {
                    // Finally, check that we're in the correct position (before vs. after)
                    switch (where)
                    {
                    case PlacementType::before:
                        return (runtimeLoopIndices.count(nextLoopIndex) == 0 || runtimeLoopIndices.at(nextLoopIndex).state == LoopIndexState::notVisited);
                    case PlacementType::after:
                        return (runtimeLoopIndices.count(nextLoopIndex) != 0 && runtimeLoopIndices.at(nextLoopIndex).state == LoopIndexState::done);
                    default:
                        throw std::runtime_error("illegalState");
                    }
                }
                return false;
            }
            else if (auto definedPred = dyn_cast_or_null<IndexDefinedPredicateOp>(p))
            {
                auto definedIndex = definedPred.index().cast<IndexAttr>().getValue();
                return (runtimeLoopIndices.count(definedIndex) > 0) && (runtimeLoopIndices.at(definedIndex).state != LoopIndexState::done);
            }
            else if (auto conjunction = dyn_cast_or_null<ConjunctionPredicateOp>(p))
            {
                bool result = true;
                for (auto t : conjunction.values())
                {
                    result &= evalPlacement(t.getDefiningOp());
                }
                return result;
            }
            else if (auto disjunction = dyn_cast_or_null<DisjunctionPredicateOp>(p))
            {
                bool result = false;
                for (auto t : disjunction.values())
                {
                    result |= evalPlacement(t.getDefiningOp());
                }
                return result;
            }
            else
            {
                return false;
            }
        };

        return evalPlacement(placementPredicate);
    }

    std::vector<ScheduledKernelOp> LoopNestBuilder::GetValidKernels(const std::vector<ScheduledKernelOp>& kernelGroup, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule, const Position& position) const
    {
        std::vector<Index> definedIndices;
        for (auto [key, value] : runtimeIndexVariables)
        {
            definedIndices.push_back(key);
        }
        auto loopIndex = schedule.CurrentLoopIndex();
        std::vector<ScheduledKernelOp> validKernels;
        std::copy_if(kernelGroup.begin(), kernelGroup.end(), std::back_inserter(validKernels), [&](ScheduledKernelOp k) {
            if (!IsPlacementValid(k, runtimeIndexVariables, schedule, position))
            {
                return false;
            }

            auto predicate = k.getPredicate();
            if (auto castPredicate = dyn_cast_or_null<KernelPredicateOpInterface>(predicate))
            {
                auto simplifiedPredicateOp = castPredicate.simplify(GetDomain(), runtimeIndexVariables, schedule);
                auto simplifiedPredicate = dyn_cast_or_null<KernelPredicateOpInterface>(simplifiedPredicateOp);
                auto result = simplifiedPredicate.evaluate(GetDomain(), runtimeIndexVariables, schedule);
                if (result.has_value() && *result == false)
                {
                    return false;
                }
                return true;
            }
            else if (auto castPredicate = dyn_cast_or_null<EvaluatablePredicateOpInterface>(predicate))
            {
                auto predicateResult = castPredicate.evaluate(definedIndices, loopIndex, position);
                if (!predicateResult)
                {
                    return false;
                }
                return true;
            }
            return true;
        });

        return validKernels;
    }

    LoopIndexSymbolTable LoopNestBuilder::GetRuntimeIndexVariables(const LoopIndexSymbolTable& runtimeLoopIndices, const LoopVisitSchedule& schedule) const
    {
        auto domain = GetDomain();

        // Start with the concrete loop indices
        LoopIndexSymbolTable indexVariables = runtimeLoopIndices;

        // ...and add the variables we need to compute (because they represent an index that has been split)
        for (auto d : domain.GetDimensions())
        {
            auto computedIndices = domain.GetComputedIndicesForDimension(d);
            for (auto index : computedIndices)
            {
                if (auto runtimeVarIter = runtimeLoopIndices.find(index); runtimeVarIter != runtimeLoopIndices.end())
                {
                    indexVariables.insert_or_assign(index, runtimeVarIter->second);
                }
            }
        }
        return indexVariables;
    }

    mlir::ConstantIndexOp LoopNestBuilder::GetConstantIndex(OpBuilder& builder, int64_t indexVal)
    {
        if (_constantIndices.count(indexVal) == 0)
        {
            auto loc = GetLocation();
            auto block = _constantOpBuilder.getInsertionBlock();
            _constantOpBuilder.setInsertionPoint(block, block->begin());
            _constantIndices[indexVal] = _constantOpBuilder.create<ConstantIndexOp>(loc, indexVal);
        }
        return _constantIndices[indexVal];
    }

    LoopRange LoopNestBuilder::MakeLoopRange(mlir::OpBuilder& builder, int64_t start, int64_t stop, int64_t step)
    {
        return {
            GetConstantIndex(builder, start).getResult(),
            GetConstantIndex(builder, stop).getResult(),
            GetConstantIndex(builder, step).getResult()
        };
    }

    void LoopNestBuilder::DefinePostLoopIndex(OpBuilder& builder, const Index& loopIndex, LoopIndexSymbolTable& runtimeLoopIndices, const LoopVisitSchedule& schedule)
    {
        auto loopRange = schedule.GetActiveLoopRange(GetDomain(), loopIndex, runtimeLoopIndices);
        ConstantIndexOp firstVal = GetConstantIndex(builder, loopRange.Begin());
        auto entry = LoopIndexSymbolTableEntry{ firstVal, loopRange, LoopIndexState::done };

        if (runtimeLoopIndices.count(loopIndex) > 0)
        {
            entry.state = runtimeLoopIndices.at(loopIndex).state;
        }

        runtimeLoopIndices.insert_or_assign(loopIndex, entry);
    }

    bool LoopNestBuilder::IsFullyDefined(const Index& index, const LoopVisitSchedule& schedule) const
    {
        if (index == schedule.CurrentLoopIndex())
        {
            return true;
        }

        // For this index to be defined, it suffices that all the dependent leaf ("loop") indices are defined
        for (const auto& i : GetDomain().GetDependentLoopIndices(index))
        {
            if (!schedule.WasIterationVariableDefined(i))
            {
                return false;
            }
        }
        return true;
    }

    bool LoopNestBuilder::AreAllFullyDefined(const std::vector<Index>& indices, const LoopVisitSchedule& schedule) const
    {
        for (const auto& index : indices)
        {
            if (!IsFullyDefined(index, schedule))
            {
                return false;
            }
        }
        return true;
    }

    SymbolicIndexOp LoopNestBuilder::GetSymbolicIndex(Index index)
    {
        auto region = GetScheduleParentRegion();
        SymbolicIndexOp result;
        region->walk([&](Operation* op) {
            if (auto indexOp = dyn_cast_or_null<SymbolicIndexOp>(op))
            {
                if (indexOp.getValue() == index)
                {
                    result = indexOp;
                    return WalkResult::interrupt();
                }
            }

            return WalkResult::advance();
        });

        if (result)
            return result;

        auto symbolicIndex = _schedule.getOrCreateSymbolicIndex(_constantOpBuilder, index);
        return symbolicIndex;
    }

    mlir::OpBuilder LoopNestBuilder::GetCurrentLoopBuilder(const LoopVisitSchedule& schedule)
    {
        if (schedule.IsOutermostLoop())
        {
            return _builder;
        }

        // TODO : instead of using the latest scheduled loop here, if we move to a level-by-level processing of
        //        the loop tree we can loop over all of the scheduled loops for this level here and act on all
        //        of them simultaneously
        auto prevIndex = schedule.Prev().CurrentLoopIndex();
        auto prevScheduledLoop = FindLatestScheduledLoop(prevIndex);
        return prevScheduledLoop.getBodyBuilder();
    }

    std::vector<ScheduledLoopOp> LoopNestBuilder::FindAllScheduledLoops(Index value)
    {
        // #### TODO: need to get rid of _loops
        assert(_loops.count(value) > 0);
        return _loops.at(value);
    }

    ScheduledLoopOp LoopNestBuilder::FindLatestScheduledLoop(Index value)
    {
        return FindAllScheduledLoops(value).back();
    }

    std::optional<uint64_t> LoopNestBuilder::GetUnrollIfRangeSmallerThan(Index loopIndex) const
    {
        return const_cast<ScheduleOp&>(_schedule).getUnrollIfRangeSmallerThan(loopIndex);
    }

    bool LoopNestBuilder::IsSaturated(Index loopIndex) const
    {
        return const_cast<ScheduleOp&>(_schedule).isSaturated(loopIndex);
    }

    std::optional<uint64_t> LoopNestBuilder::GetUnrollAndJamFactor(Index loopIndex) const
    {
        return const_cast<ScheduleOp&>(_schedule).getUnrollAndJamFactor(loopIndex);
    }

    ScheduleOp LoopNestBuilder::GetScheduleOp() const
    {
        return _schedule;
    }

    mlir::Region* LoopNestBuilder::GetScheduleParentRegion() const
    {
        return const_cast<ScheduleOp&>(_schedule).getOperation()->getParentRegion();
    }

    std::vector<std::string> LoopNestBuilder::GetKernelIds() const
    {
        return const_cast<ScheduleOp&>(_schedule).getKernelIds();
    }

    TransformedDomain LoopNestBuilder::GetDomain() const
    {
        auto domain = const_cast<ScheduleOp&>(_schedule).getDomain().getValue();
        return domain;
    }

    const std::vector<ScheduledKernelOp>& LoopNestBuilder::GetKernelGroup(std::string id) const
    {
        return _kernelGroups.at(id);
    }

    std::map<KernelId, std::vector<ScheduledKernelOp>> LoopNestBuilder::DiscoverKernelGroups() const
    {
        std::map<KernelId, std::vector<ScheduledKernelOp>> kernels;
        auto kernelIds = GetKernelIds();
        for (auto id : kernelIds)
        {
            if (auto op = FindKernelOp(id, GetScheduleOp()))
            {
                if (auto scheduledKernelOp = dyn_cast<ScheduledKernelOp>(op))
                {
                    kernels[id].push_back({ scheduledKernelOp });
                }
            }
            else
            {
                llvm::errs() << "Couldn't find kernel " << id << "\n";
                assert(false);
            }
        }

        return kernels;
    }

    mlir::Value LoopNestBuilder::EmitIndexExpression(OpBuilder& builder, Location loc, const AffineExpression& expr, const TransformedDomain& domain)
    {
        std::vector<mlir::Value> symbols;
        AffineExpr affineExpr = expr.GetAffineExpr();
        auto exprIndices = expr.GetIndices();
        std::vector<mlir::Value> indices;
        std::transform(exprIndices.begin(), exprIndices.end(), std::back_inserter(indices), [this](auto i) {
            return GetSymbolicIndex(i);
        });
        auto map = AffineMap::get(indices.size(), symbols.size(), affineExpr);
        indices.insert(indices.end(), symbols.begin(), symbols.end());
        auto exprOp = builder.create<AffineApplyOp>(loc, map, indices);

        return exprOp;
    }

    AffineExpression LoopNestBuilder::GetIndexExpression(OpBuilder& builder, const Index& index, const TransformedDomain& domain) const
    {
        if (!domain.IsComputedIndex(index))
        {
            throw std::runtime_error("illegalState");
        }

        return domain.GetReducedIndexExpr(index, builder.getContext());
    }

    std::vector<size_t> LoopNestBuilder::GetLogicalDimensionPositions(const Index& index) const
    {
        auto domain = GetDomain();
        auto orderedDomainDims = domain.GetDimensions();
        std::vector<size_t> result;
        for (auto dimensionIndex : domain.GetBaseIndices(index))
        {
            auto iter = std::find(orderedDomainDims.begin(), orderedDomainDims.end(), dimensionIndex);
            assert(iter != orderedDomainDims.end() && "Trying to use an index that isn't part of this split iteration domain");
            result.push_back(std::distance(orderedDomainDims.begin(), iter));
        }
        return result;
    }

    void LoopNestBuilder::InvokeKernel(OpBuilder& builder, ScheduledKernelOp kernel, Position position, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule)
    {
        if (_printLoops)
        {
            llvm::errs().indent(2 * schedule.CurrentLoopLevel() + 1) << position << "(" << kernel.getId() << ")\n";
        }

        [[maybe_unused]] auto invokeOp = builder.create<InvokeKernelOp>(GetLocation(), kernel.getId());
    }

    void LoopNestBuilder::EmitKernelBody(OpBuilder& builder, InvokeKernelOp invokeOp, const LoopIndexSymbolTable& runtimeIndexVariables)
    {
        auto domain = GetDomain();
        auto op = FindKernelOp(invokeOp.getKernel(), GetScheduleOp());
        assert(isa<ScheduledKernelOp>(op) && "Didn't find scheduled kernel");

        auto scheduledKernelOp = cast<ScheduledKernelOp>(op);
        auto kernelId = scheduledKernelOp.getKernel();
        auto loc = GetLocation();

        Region* kernelBodyRegion = GetKernelBodyRegion(kernelId, GetScheduleOp());
        auto& kernelBody = kernelBodyRegion->front();

        // Define old indices in terms of new ones
        std::vector<mlir::Value> newLoopIndices;
        std::map<Index, mlir::Value> computedIndexVariables;
        for (auto originalLoopIndex : domain.GetDimensions())
        {
            if (runtimeIndexVariables.count(originalLoopIndex) > 0)
            {
                newLoopIndices.push_back(runtimeIndexVariables.at(originalLoopIndex).value);
            }
            else
            {
                AffineExpression expr;
                if (!domain.IsComputedIndex(originalLoopIndex))
                {
                    AffineExpr affineExpr = builder.getAffineConstantExpr(domain.GetDimensionBegin(originalLoopIndex));
                    expr = AffineExpression(affineExpr, {});
                }
                else
                {
                    expr = GetIndexExpression(builder, originalLoopIndex, domain);
                }
                auto indexValue = EmitIndexExpression(builder, loc, expr, domain);
                newLoopIndices.push_back(indexValue);
                computedIndexVariables[originalLoopIndex] = indexValue;
            }
        }

        // TODO: uniquify the symbolic index ops for the nest and keep a list of them, so we don't have to scan through all the body ops to generate the mapping
        BlockAndValueMapping argMap;
        std::vector<SymbolicIndexOp> symbolicIndexOps = RecurseGetSymbolicIndexOps(kernelBodyRegion);
        for (auto symbIdxOp : symbolicIndexOps)
        {
            auto index = symbIdxOp.getValue();
            if (runtimeIndexVariables.count(index) > 0)
            {
                argMap.map(symbIdxOp.getResult(), runtimeIndexVariables.at(index).value);
            }

            if (computedIndexVariables.count(index) > 0)
            {
                argMap.map(symbIdxOp.getResult(), computedIndexVariables.at(index));
            }
        }

        // Actually clone the operations
        for (auto& op : kernelBody.without_terminator())
        {
            builder.clone(op, argMap);
        }

        auto scheduledLoopOp = invokeOp->getParentOp();
        llvm::SmallVector<Attribute> kernelIds{ builder.getStringAttr(kernelId) };
        if (auto kernelsAttr = scheduledLoopOp->getAttrOfType<ArrayAttr>("kernels"))
        {
            kernelIds.insert(kernelIds.begin(), kernelsAttr.begin(), kernelsAttr.end());
        }
        scheduledLoopOp->setAttr("kernels", builder.getArrayAttr(kernelIds));
    }

    void LoopNestBuilder::EnsureTerminators()
    {
        for (auto [_, loopOps] : _loops)
        {
            for (auto scheduledLoopOp : loopOps)
            {
                auto loc = GetLocation();
                ScheduledLoopOp::ensureTerminator(scheduledLoopOp.prologue(), _builder, loc);
                ScheduledLoopOp::ensureTerminator(scheduledLoopOp.body(), _builder, loc);
                ScheduledLoopOp::ensureTerminator(scheduledLoopOp.epilogue(), _builder, loc);
            }
        }
    }

    mlir::Location LoopNestBuilder::GetLocation()
    {
        std::string tag = "LoopNestBuilder";
        return util::GetLocation(_builder, tag, _schedule.getLoc());
    }

} // namespace loopnest
} // namespace accera::ir
