////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs, Kern Handa, Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/LoopNestBuilder.h"

#include "IRUtil.h"
#include "nest/LoopIndexInfo.h"
#include "nest/LoopNestOps.h"
#include "value/ValueDialect.h"
#include <utilities/include/Exception.h>

#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>
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
            os << "[" << r.Begin() << ",";
            if (r.HasVariableEnd())
            {
                auto arg = r.VariableEnd().dyn_cast<mlir::BlockArgument>();
                os << "arg" << arg.getArgNumber();
            }
            else
            {
                os << r.End();
            }
            os << ":" << r.Increment() << ")";
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
                if (auto constIntOp = mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(const_cast<mlir::Value&>(value).getDefiningOp()))
                    return static_cast<T>(constIntOp.value());
                else if (auto constIdxOp = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(const_cast<mlir::Value&>(value).getDefiningOp()))
                    return static_cast<T>(constIdxOp.value());
                else
                    assert(false && "Error: got bad op type for constant int");
            }
            else if constexpr (std::is_floating_point<T>::value)
            {
                using OpType = mlir::arith::ConstantFloatOp;
                auto op = mlir::cast<OpType>(const_cast<mlir::Value&>(value).getDefiningOp());
                return static_cast<T>(const_cast<OpType&>(op).value());
            }
            else
            {
                static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "Invalid type for GetValue()");
            }

            llvm_unreachable("unexpected");
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

        // TODO: use StringAttr for id to avoid the extra conversion
        Operation* FindKernelOp(StringRef id, ScheduleOp rootOp)
        {
            auto symTableOp = mlir::SymbolTable::getNearestSymbolTable(rootOp);
            auto idAttr = StringAttr::get(rootOp->getContext(), id);
            auto symbolOp = mlir::SymbolTable::lookupNearestSymbolFrom(symTableOp, idAttr);
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

        LoopIndexSymbolTable GetEnclosingLoopIndices(Operation* where)
        {
            // Contents of the loop index symbol table:
            //   mlir::Value value; // this is the induction variable for the loop
            //   Range loopRange;
            //   LoopIndexState state;
            LoopIndexSymbolTable result;
            auto loop = where->getParentOfType<ScheduledLoopOp>();
            while (loop)
            {
                auto index = loop.getIndex();
                result.insert_or_assign(index, LoopIndexSymbolTableEntry{ {}, loop.getRange(), LoopIndexState::inProgress });
                loop = loop->getParentOfType<ScheduledLoopOp>();
            }
            return result;
        }
    } // namespace

    //
    // RecursionState
    //
    LoopNestBuilder::RecursionState::RecursionState(LoopNestBuilder& builder) :
        affineConstraints(builder.GetInitialConstraints())
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

    LoopRange::LoopRange(mlir::Value start_, mlir::Value stop_, mlir::Value increment) :
        start(start_), stop(stop_), step(increment)
    {
    }

    Range LoopRange::GetConstantRange() const
    {
        assert(!IsVariable() && "GetConstantRange called on a variable range");

        auto startInt = GetValue<int64_t>(start);
        auto stopInt = GetValue<int64_t>(stop);
        auto stepInt = GetValue<int64_t>(step);
        return { startInt, stopInt, stepInt };
    }

    mlir::AffineValueMap GetAffineValueMap(mlir::Value val)
    {
        // Create a 1-dimensional <(d0) -> (d0)> affine map
        auto map = mlir::AffineMap::getMultiDimIdentityMap(1 /* numDims */, val.getContext());
        llvm::SmallVector<mlir::Value, 4> operands{ val };

        // Compose and canonicalize the <(d0) -> (d0)> map with the { val } operand:
        // - detects if val can be a symbol and converts the map to <(s0) -> (s0)>
        // - detects if val is a constant and converts the map to give that contant and have no operands. e.g. <() -> (16)>
        mlir::fullyComposeAffineMapAndOperands(&map, &operands);
        mlir::canonicalizeMapAndOperands(&map, &operands);
        return mlir::AffineValueMap(map, operands);
    }

    Range LoopRange::GetVariableRange() const
    {
        assert(IsVariable() && "GetVariableRange called on a constant range");

        // TODO: support variable increment
        mlir::AffineValueMap startValueMap = GetAffineValueMap(start);
        mlir::AffineValueMap stopValueMap = GetAffineValueMap(stop);
        auto stepInt = GetValue<int64_t>(step);
        return { startValueMap, stopValueMap, stepInt };
    }

    Range LoopRange::GetRange() const
    {
        return IsVariable() ? GetVariableRange() : GetConstantRange();
    }

    bool LoopRange::IsVariableStart() const
    {
        auto constStartOp = start.getDefiningOp<mlir::arith::ConstantIndexOp>();
        return (constStartOp == nullptr);
    }

    bool LoopRange::IsVariableStop() const
    {
        auto constStopOp = stop.getDefiningOp<mlir::arith::ConstantIndexOp>();
        return (constStopOp == nullptr);
    }

    bool LoopRange::IsVariable() const
    {
        return IsVariableStart() || IsVariableStop();
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

    std::vector<ScheduledLoopOp> LoopNestBuilder::BuildLoopNest()
    {
        auto schedule = GetLoopSchedule();
        auto initialIndex = schedule.CurrentLoopIndex();

        // We need to create a RecursionState object here, because it's passed in as a (const) reference
        RecursionState state(*this);

        GenerateInitialLoopStructure(state, schedule);
        auto baseLoops = _loops[initialIndex];
        _loops.clear();

        UnswitchLoops(baseLoops, state, schedule, state.affineConstraints);

        AddInvokeOps(_loops[initialIndex], state, schedule);
        VerifyPredicates(_loops[initialIndex], schedule);

        MergeAdjacentKernelBodies(_loops[initialIndex], schedule);

        EmitLoopBodies(_loops[initialIndex], state, schedule);

        ApplyInjectableMappings();

        return _loops[initialIndex];
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

    LoopNestAffineConstraints LoopNestBuilder::GetInitialConstraints()
    {
        auto loopSequence = const_cast<ScheduleOp&>(_schedule).getOrder();
        auto domain = GetDomain();
        auto context = _schedule->getContext();
        auto constraints = domain.GetLoopNestConstraints(loopSequence, context);
        // Set the value for each loop index to the SymbolicIndexOp mlir::Value handle
        for (auto& index : loopSequence)
        {
            mlir::Value symbolicIndexVal = GetSymbolicIndex(index);
            constraints.SetValue(index, symbolicIndexVal);
        }
        return constraints;
    }

    LoopNestBuilder::RecursionState LoopNestBuilder::GenerateInitialLoopStructure(const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        if (schedule.IsDone())
        {
            return state;
        }

        const auto& domain = GetDomain();
        auto loopIndex = schedule.CurrentLoopIndex();

        // Find the active range for the current loop dimension and reduce the end amount if it exceeds the active range (boundary case)
        auto fullRange = schedule.GetActiveLoopRange(domain, loopIndex, state.loopIndices);
        Partition p = { loopIndex, fullRange, LoopPartitionConstraints{ state.affineConstraints, state.affineConstraints } };

        auto newState = state;
        LoopRange partitionRange = MakeLoopRange(_builder, p.range);
        UpdateSubdomainSizes(loopIndex, partitionRange, newState.subdomainSize);

        auto loop = EmitLoopOp(partitionRange, newState, schedule); // This creates the (empty) loop op

        bool shouldGuardLoopBounds = IsGpuLoop(loopIndex);
        if (shouldGuardLoopBounds)
        {
            AddLoopLimitMetadata(loop);
        }

        _loops[loopIndex].push_back(loop);

        SymbolicIndexOp loopIndexOp = loop.getSymbolicIndex();
        newState.loopIndices.insert_or_assign(loopIndex, LoopIndexSymbolTableEntry{ loopIndexOp, partitionRange.GetRange(), LoopIndexState::inProgress });
        GenerateInitialLoopBody(loop, partitionRange, newState, schedule); // This contains the recursive call to GenerateInitialLoopStructure
        EndLoopRange(partitionRange, newState, schedule);

        // set the loop index state to be "done"
        DefinePostLoopIndex(_builder, loopIndex, newState.loopIndices, schedule);
        return newState;
    }

    void LoopNestBuilder::AddLoopLimitMetadata(ScheduledLoopOp loop)
    {
        const auto& domain = GetDomain();
        auto loopIndex = loop.getIndex();

        auto getParentIndex = [](const TransformedDomain& domain, Index i) {
            assert(domain.HasParentIndex(i));
            return domain.GetParentIndices(i)[0];
        };

        // TODO: rewrite this part to be less duplicative
        if (domain.IsSplitIndex(loopIndex, /*inner=*/true))
        {
            auto parentIndex = getParentIndex(domain, loopIndex);
            auto parentRange = domain.GetIndexRange(parentIndex);
            auto outerIndex = domain.GetOtherSplitIndex(loopIndex);
            auto outerRange = domain.GetIndexRange(outerIndex);
            assert(outerRange.Size() == parentRange.Size());
            if (parentRange.Size() % outerRange.Increment() != 0)
            {
                loop->setAttr("accv_upper_limit", _builder.getI64IntegerAttr(parentRange.End()));
                loop->setAttr("accv_upper_limit_index", IndexAttr::get(parentIndex, _builder.getContext()));
            }
        }
        else if (domain.IsSplitIndex(loopIndex, /*inner=*/false))
        {
            auto parentIndex = getParentIndex(domain, loopIndex);
            if (domain.HasParentIndex(parentIndex))
            {
                auto grandparentIndex = getParentIndex(domain, parentIndex);
                auto grandparentRange = domain.GetIndexRange(grandparentIndex);
                if (domain.IsSplitIndex(parentIndex, /*inner=*/true))
                {
                    auto parentRange = domain.GetIndexRange(parentIndex);
                    auto outerIndex = domain.GetOtherSplitIndex(parentIndex);
                    auto outerRange = domain.GetIndexRange(outerIndex);
                    assert(outerRange.Size() == grandparentRange.Size());
                    if (grandparentRange.Size() % outerRange.Increment() != 0)
                    {
                        loop->setAttr("accv_upper_limit", _builder.getI64IntegerAttr(grandparentRange.End()));
                        loop->setAttr("accv_upper_limit_index", IndexAttr::get(outerIndex, _builder.getContext()));
                    }
                }
            }
        }
    }

    LoopNestBuilder::RecursionState LoopNestBuilder::AddInvokeOps(const std::vector<ScheduledLoopOp>& loops, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        if (schedule.IsDone())
        {
            return state;
        }

        auto loopIndex = schedule.CurrentLoopIndex();

        // for each loop op
        RecursionState newState = state;
        for (auto loop : loops)
        {
            newState = state;
            LoopRange partitionRange = MakeLoopRange(_builder, loop.getRange());
            UpdateSubdomainSizes(loopIndex, partitionRange, newState.subdomainSize);
            loop.setSubdomainSize(newState.subdomainSize);
            SymbolicIndexOp loopIndexOp = loop.getSymbolicIndex();
            newState.loopIndices.insert_or_assign(loopIndex, LoopIndexSymbolTableEntry{ loopIndexOp, partitionRange.GetRange(), LoopIndexState::inProgress });
            GenerateLoopBody(loop, partitionRange, newState, schedule); // This contains the recursive call to AddInvokeOps
        }

        // set the loop index state to be "done"
        DefinePostLoopIndex(_builder, loopIndex, newState.loopIndices, schedule);
        return newState;
    }

    void LoopNestBuilder::VerifyPredicates(const std::vector<ScheduledLoopOp>& loops, const LoopVisitSchedule& schedule)
    {
        if (schedule.IsDone())
        {
            return;
        }

        // For each invoke kernel op:
        //   - get all enclosing loops
        //   - set index values/ranges for all enclosing loops
        //   - eval the predicate based on those index->value mappings
        const auto& domain = GetDomain();
        for (auto loop : loops)
        {
            loop.walk([&](InvokeKernelOp invokeOp) {
                auto loopIndices = GetEnclosingLoopIndices(invokeOp);
                auto kernel = FindKernelOp(invokeOp.getKernel(), GetScheduleOp());
                if (auto scheduledKernelOp = dyn_cast<ScheduledKernelOp>(kernel))
                {
                    auto pred = scheduledKernelOp.getKernelPredicate();
                    auto predVal = pred.evaluate(domain, loopIndices, schedule);
                    // verify pred evaluates to 'true' or 'unknown'
                    if (predVal.has_value())
                    {
                        if (!predVal.value())
                        {
                            throw std::runtime_error("Error: predicate evaluates to false!");
                        }
                    }
                }
            });
        }
    }

    ScheduledLoopOp CloneWithNewRange(mlir::PatternRewriter& rewriter, ScheduledLoopOp loopToClone, const Range& range)
    {
        auto newLoop = rewriter.create<ScheduledLoopOp>(loopToClone->getLoc(),
                                                        range,
                                                        loopToClone.getSymbolicIndex(),
                                                        loopToClone.getSubdomainSize(), // TODO : remove subdomain size, this is not accurate anymore with a new range
                                                        loopToClone.getSubdomainIndexOrder());

        rewriter.eraseBlock(newLoop.getPrologue());
        rewriter.eraseBlock(newLoop.getBody());
        rewriter.eraseBlock(newLoop.getEpilogue());
        newLoop.prologue().getBlocks().clear();
        newLoop.body().getBlocks().clear();
        newLoop.epilogue().getBlocks().clear();
        rewriter.cloneRegionBefore(loopToClone.prologue(), newLoop.prologue(), newLoop.prologue().end());
        rewriter.cloneRegionBefore(loopToClone.body(), newLoop.body(), newLoop.body().end());
        rewriter.cloneRegionBefore(loopToClone.epilogue(), newLoop.epilogue(), newLoop.epilogue().end());

        // Clone attributes that aren't the beginMap, endMap, step, or operand_segment_sizes attrs
        // TODO : make this a more general "copy all attrs from X dialect" type of copy so we don't have to keep this list up to date eternally
        //        currently not all of our attributes follow the required naming convention to make this easy
        std::vector<mlir::StringAttr> attrsToSkip = { loopToClone.beginMapAttrName(),
                                                      loopToClone.endMapAttrName(),
                                                      loopToClone.stepAttrName(),
                                                      loopToClone.operand_segment_sizesAttrName() };
        auto srcLoopAttrs = loopToClone->getAttrs();
        for (auto namedAttr : srcLoopAttrs)
        {
            auto nameStringAttr = namedAttr.getName();
            auto findIter = std::find(attrsToSkip.begin(), attrsToSkip.end(), nameStringAttr);
            if (findIter == attrsToSkip.end())
            {
                newLoop->setAttr(namedAttr.getName(), namedAttr.getValue());
            }
        }

        return newLoop;
    }

    LoopNestBuilder::RecursionState LoopNestBuilder::UnswitchLoops(const std::vector<ScheduledLoopOp>& loops, const RecursionState& state, const LoopVisitSchedule& schedule, const LoopNestAffineConstraints& currentConstraints)
    {
        if (schedule.IsDone())
        {
            return state;
        }

        auto newState = state;
        newState.affineConstraints = currentConstraints;
        OpBuilder::InsertionGuard guard(_builder);
        for (auto loop : loops)
        {
            auto loopIndex = loop.getIndex();
            assert(loopIndex == schedule.CurrentLoopIndex());
            auto domain = GetDomain();
            auto fullRange = schedule.GetActiveLoopRange(domain, loopIndex, newState.loopIndices);
            std::vector<Partition> partitions;
            bool shouldGuardLoopBounds = IsGpuLoop(loopIndex);
            if (shouldGuardLoopBounds)
            {
                partitions.push_back({ loopIndex, fullRange, LoopPartitionConstraints{ currentConstraints, currentConstraints } });
            }
            else
            {
                partitions = GetPartitions(loopIndex, fullRange, newState, schedule);
            }

            _builder.setInsertionPointAfter(loop);
            _loops[loopIndex].erase(std::remove(_loops[loopIndex].begin(), _loops[loopIndex].end(), loop), _loops[loopIndex].end());

            for (auto& p : partitions)
            {
                newState = state;
                newState.affineConstraints = currentConstraints;

                LoopRange partitionRange = MakeLoopRange(_builder, p.range);
                auto newLoop = CloneWithNewRange(_builder, loop, partitionRange.GetRange());

                // TODO : remove subdomain sizes
                auto subdomainSizes = newLoop.getSubdomainSize();
                UpdateSubdomainSizes(loopIndex, partitionRange, subdomainSizes);
                newLoop.setSubdomainSize(subdomainSizes);

                if (shouldGuardLoopBounds)
                {
                    AddLoopLimitMetadata(newLoop);
                }

                SymbolicIndexOp loopIndexOp = newLoop.getSymbolicIndex();
                newState.loopIndices.insert_or_assign(loopIndex, LoopIndexSymbolTableEntry{ loopIndexOp, partitionRange.GetRange(), LoopIndexState::inProgress });
                EndLoopRange(partitionRange, newState, schedule);
                _loops[loopIndex].push_back(newLoop);

                auto innerLoops = GetInnerLoops(newLoop.getBody());
                UnswitchLoops(innerLoops, newState, schedule.Next(), p.constraints.innerConstraints);

                // set the loop index state to be "done"
                DefinePostLoopIndex(_builder, loopIndex, newState.loopIndices, schedule);
            }

            _builder.eraseOp(loop);
        }

        return newState;
    }

    // TODO: move the loop-printing until after merging
    void LoopNestBuilder::MergeAdjacentKernelBodies(std::vector<ScheduledLoopOp> loops, const LoopVisitSchedule& schedule)
    {
        // Suffix of first partition matches entirety of second: move
        // --> (0..1: S1), (0..N-1: S2), (N1-..N: S2, S3)
        // prefix of last partition matches entirety of second: move
        // --> (0..1: S1), (0..N: S2), (N1-..N: S3)
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

                if (!thisLoop.hasConstantRange() || (nextLoop && !nextLoop.hasConstantRange()))
                {
                    // TODO : support merging dynamic ranges
                    continue;
                }

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
                            thisLoop.setConstantBegin(prevLoop.getConstantBegin());
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
                                    auto newPartition = Range(thisLoop.getConstantBegin(), nextLoop.getConstantEnd(), thisLoop.step());
                                    auto numBadPrefixKernels = matchStart - thisLoopKernels.begin();
                                    auto numMovedKernels = thisLoopKernels.end() - matchStart;
                                    auto builder = GetCurrentLoopBuilder(schedule);
                                    OpBuilder::InsertionGuard guard(builder);
                                    builder.setInsertionPointAfter(thisLoop);
                                    auto newLoop = cast<ScheduledLoopOp>(builder.clone(*thisLoop.getOperation()));
                                    newLoop.setConstantEnd(nextLoop.getConstantEnd());
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
                            thisLoop.setConstantEnd(nextLoop.getConstantEnd());
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
            // TODO : fix with dynamic ranges
            auto partitionRange = MakeLoopRange(_builder, 0, 1, 1);
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
        mlir::OpBuilder::InsertionGuard insertGuard(_builder);
        [[maybe_unused]] auto loc = GetLocation();
        auto registeredInjectableMappings = GetScheduleOp().getInjectableMappings();
        for (auto& mapping : registeredInjectableMappings)
        {
            auto scheduledLoopOps = FindAllScheduledLoops(mapping.index());

            for (auto& scheduledLoopOp : scheduledLoopOps)
            {
                BlockAndValueMapping operandMap;
                _builder.setInsertionPoint(scheduledLoopOp);
                [[maybe_unused]] auto clonedBeginOp = _builder.clone(*(mapping.getOperation()), operandMap);

                _builder.setInsertionPointAfter(scheduledLoopOp);
                [[maybe_unused]] auto clonedEndOp = _builder.clone(*(mapping.getInjectionEndOp()), operandMap);
            }

            if (mapping.use_empty())
            {
                _builder.eraseOp(mapping);
            }
        }
    }

    // This function figures out the correct place to put the insertion point, but makes no guarantees about where it leaves it
    ScheduledLoopOp LoopNestBuilder::EmitLoopOp(const LoopRange& loopRange, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        auto loopIndex = schedule.CurrentLoopIndex();
        auto range = loopRange.GetRange();

        if (_printLoops)
        {
            llvm::errs().indent(2 * schedule.CurrentLoopLevel()) << "for " << loopIndex << " in " << range << " {";
            if (!range.HasVariableEnd() && range.NumIterations() == 1)
            {
                llvm::errs() << " -- single iteration";
            }
            llvm::errs() << "\n";
        }

        // Emit a ScheduledLoopOp
        auto builder = GetCurrentLoopBuilder(schedule);

        auto loc = GetLocation();
        auto symbolicIndex = GetSymbolicIndex(loopIndex);
        assert(symbolicIndex && "Error: bad symbolic index");
        auto domain = GetDomain();
        auto domainIndexOrder = domain.GetDimensions();

        auto loop = builder.create<ScheduledLoopOp>(loc, range, symbolicIndex, state.subdomainSize, domainIndexOrder);

        // TODO : move these attributes to the loop attribute infra
        loop->setAttr("index", IndexAttr::get(loopIndex, builder.getContext()));

        if (!range.HasVariableEnd())
        {
            if (auto val = GetUnrollIfRangeSmallerThan(loopIndex))
            {
                if (range.NumIterations() < (int64_t)*val)
                {
                    loop->setAttr("accv_unrolled", builder.getUnitAttr());
                }
            }

            if (auto val = GetUnrollAndJamFactor(loopIndex))
            {
                loop->setAttr("accv_unroll_jam", builder.getI64IntegerAttr((int64_t)*val));
            }

            if (IsSaturated(loopIndex))
            {
                loop->setAttr("accv_saturated", builder.getUnitAttr());
            }
        }

        auto execPlan = GetScheduleOp().getOrCreateExecPlan();
        auto target = execPlan.exec_target();
        switch (target)
        {
        default:
        case ir::value::ExecutionTarget::CPU:
            break;
        case ir::value::ExecutionTarget::GPU:
            auto bindingPairOpt = execPlan.getBinding(loopIndex);
            if (bindingPairOpt.has_value())
            {
                auto [proc, map] = bindingPairOpt.value();
                auto procStr = ir::value::stringifyEnum(proc);

                std::vector<mlir::NamedAttribute> loopBoundAttrs;
                loopBoundAttrs.emplace_back(builder.getStringAttr("proc"), builder.getStringAttr(procStr));
                loopBoundAttrs.emplace_back(builder.getStringAttr("map"), mlir::AffineMapAttr::get(map));
                loop->setAttr("accv_gpu_map", builder.getDictionaryAttr(loopBoundAttrs));
            }
            break;
        }

        // Carry forward any loop attributes tagged for this index
        auto loopAttrDict = _schedule.getLoopAttributes(loopIndex);
        if (loopAttrDict)
        {
            // We have attributes for this loop, so transfer them over to the ScheduledLoopOp
            auto loopAttrs = loopAttrDict->getValue();
            for (auto& loopAttr : loopAttrs)
            {
                loop->setAttr(loopAttr.getName(), loopAttr.getValue());
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

    void LoopNestBuilder::GenerateInitialLoopBody(ScheduledLoopOp loop, const LoopRange& r, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        auto newState = state;
        auto loopIndex = schedule.CurrentLoopIndex();

        if (schedule.IsInnermostLoop())
        {
            UpdateKernelState(loop, r, Position::body, newState, schedule);
        }
        else
        {
            auto innerState = UpdateKernelState(loop, r, Position::prologue, newState, schedule);

            // Recursively call GenerateInitialLoopStructure to generate the more-deeply-nested loops
            auto epilogueState = GenerateInitialLoopStructure(innerState, schedule.Next());

            auto outerState = UpdateKernelState(loop, r, Position::epilogue, epilogueState, schedule);
            DefinePostLoopIndex(_builder, loopIndex, newState.loopIndices, schedule);
        }
    }

    void LoopNestBuilder::GenerateLoopBody(ScheduledLoopOp loop, const LoopRange& r, const RecursionState& state, const LoopVisitSchedule& schedule)
    {
        auto newState = state;
        auto loopIndex = schedule.CurrentLoopIndex();

        if (schedule.IsInnermostLoop())
        {
            InvokeKernels(loop, r, Position::body, newState, schedule);
        }
        else
        {
            auto innerState = InvokeKernels(loop, r, Position::prologue, newState, schedule);

            // Recursively call AddInvokeOps to generate the more-deeply-nested loops
            auto innerLoops = GetInnerLoops(loop.getBody());
            auto epilogueState = AddInvokeOps(innerLoops, innerState, schedule.Next());
            auto outerState = InvokeKernels(loop, r, Position::epilogue, epilogueState, schedule);

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

    LoopNestBuilder::RecursionState LoopNestBuilder::InvokeKernels(ScheduledLoopOp loop, const LoopRange& r, Position position, const RecursionState& state, const LoopVisitSchedule& schedule)
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
            auto invoked = MaybeInvokeKernelGroup(id, true, position, loop, newState.loopIndices, schedule);
            if (invoked)
            {
                newState.validKernelGroups[id] = false;
            }
        }

        return newState;
    }

    LoopNestBuilder::RecursionState LoopNestBuilder::UpdateKernelState(ScheduledLoopOp loop, const LoopRange& r, Position position, const RecursionState& state, const LoopVisitSchedule& schedule)
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
            auto wouldInvoke = MaybeInvokeKernelGroup(id, false, position, loop, newState.loopIndices, schedule);
            if (wouldInvoke)
            {
                newState.validKernelGroups[id] = false;
            }
        }

        return newState;
    }

    void LoopNestBuilder::ReplaceInvokeOps(Block* loopBlock, const RecursionState& state)
    {
        auto builder = OpBuilder(loopBlock, std::prev(loopBlock->end()));
        auto invokeOps = GetInvokeKernelOps(loopBlock);
        for (auto invokeOp : invokeOps)
        {
            EmitKernelBody(builder, invokeOp, state.loopIndices);
            _builder.eraseOp(invokeOp);
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
            if (!validKernels.empty())
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

        return validIds;
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

    bool LoopNestBuilder::MaybeInvokeKernelGroup(std::string id, bool invoke, Position position, ScheduledLoopOp loop, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule)
    {
        auto kernelGroup = GetKernelGroup(id);
        auto loopIndex = schedule.CurrentLoopIndex();

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
                return loop.getPrologueBuilder();
            case Position::body:
                return loop.getBodyBuilder();
            case Position::epilogue:
                return loop.getEpilogueBuilder();
            default:
                throw std::runtime_error("Illegal Position type");
            }
        }();

        for (auto kernel : validKernels)
        {
            auto kernelPredicate = kernel.getKernelPredicate();
            auto evaluatablePredicate = kernel.getEvaluatablePredicate();
            if ((!kernelPredicate && !evaluatablePredicate) || isa<NullPredicateOp>(kernelPredicate))
            {
                if (position == Position::body)
                {
                    if (invoke)
                        InvokeKernel(builder, kernel, position, runtimeIndexVariables, schedule);
                    return true;
                }
            }
            else
            {
                bool predicateResult = false;
                // TODO: Move to TypeSwitch
                if (kernelPredicate && isa<NullPredicateOp>(kernelPredicate)) // BUG? This case looks like it gets handled (differently) above
                {
                    predicateResult = schedule.IsInnermostLoop();
                }
                else if (kernelPredicate)
                {
                    auto result = kernelPredicate.evaluate(GetDomain(), runtimeIndexVariables, schedule);
                    if (result.has_value())
                        predicateResult = *result;
                }
                else if (evaluatablePredicate)
                {
                    predicateResult = evaluatablePredicate.evaluate(definedIndices, loopIndex, position);
                }

                if (predicateResult)
                {
                    if (invoke)
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

    std::vector<Partition> LoopNestBuilder::GetPartitions(const Index& loopIndex, Range loopRange, const RecursionState& state, const LoopVisitSchedule& schedule) const
    {
        // TODO : better integration between the dynamic and static scenarios
        // If this loop index is part of a dynamic dimension, then its partition begin and end values may be functions of that dynamic value
        auto loc = const_cast<ScheduleOp&>(_schedule).getLoc();
        auto domain = GetDomain();
        auto baseIndices = domain.GetBaseIndices(loopIndex);
        bool isConstantSizeDimensionIndex = true;
        for (auto& baseIndex : baseIndices)
        {
            isConstantSizeDimensionIndex &= domain.HasConstantDimensionSize(baseIndex);
        }
        if (!isConstantSizeDimensionIndex)
        {
            // This loop range, even if it is a statically-sized split index by itself, is part of a dynamic dimension and therefore may have a dynamic begin or end value
            auto splitSize = loopRange.Increment();
            auto constraintPartitions = state.affineConstraints.SplitIndex(_builder, loc, loopIndex, splitSize);
            std::vector<Partition> result;
            for (auto& constraintPartition : constraintPartitions)
            {
                // TODO : move to SplitIndex and let SplitIndex return a variable number of constraints
                if (!constraintPartition.resolveConstraints.IsEmpty())
                {
                    auto [lbValueMap, ubValueMap] = constraintPartition.resolveConstraints.GetLowerAndUpperBound(loopIndex, _builder, loc);
                    result.push_back({ loopIndex, { lbValueMap, ubValueMap, splitSize }, constraintPartition });
                }
            }
            return result;
        }

        // Find conditions involving this index and add any relevant partition split points
        std::set<int64_t> splits;
        for (auto k : GetPossiblyValidKernels(state))
        {
            AddSplits(loopIndex, loopRange, k.getKernelPredicate(), state.loopIndices, schedule, splits);
        }

        // Get constraint partitions
        auto allConstraintPartitions = state.affineConstraints.PartitionIndex(_builder, loc, loopIndex, splits);
        auto lastConstraintPartition = allConstraintPartitions.back();
        std::vector<LoopPartitionConstraints> allButLastConstraintPartitions(allConstraintPartitions.begin(), allConstraintPartitions.end() - 1);

        // Get index ranges
        int begin = loopRange.Begin();
        int end = loopRange.End();
        int increment = loopRange.Increment();
        std::vector<Partition> result;
        for (auto [partitionEnd, constraintPartition] : llvm::zip(splits, allButLastConstraintPartitions))
        {
            result.push_back({ loopIndex, { begin, partitionEnd, increment }, constraintPartition });
            begin = partitionEnd;
        }
        result.push_back({ loopIndex, { begin, end, increment }, lastConstraintPartition });

        return result;
    }

    void LoopNestBuilder::AddSplits(const Index& loopIndex, const Range& loopRange, KernelPredicateOpInterface predicateOp, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule, std::set<int64_t>& allSplits) const
    {
        // Adds split points for a fixed loopRange. This does not change the top level boundaries of the loop.
        // To adjust the top level boundaries, see LoopVisitSchedule::GetActiveLoopRange

        if (predicateOp)
        {
            auto builder = const_cast<LoopNestBuilder*>(this)->GetCurrentLoopBuilder(schedule);
            predicateOp = predicateOp.simplify(builder, GetDomain(), runtimeIndexVariables, schedule);
        }

        const auto& domain = GetDomain();

        auto addTransformationSplits = [&domain, &loopIndex, &loopRange, &schedule](std::set<int64_t>& splits) -> void {
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
                    auto end = loopRange.End();

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
                auto splitSize = loopRange.Increment();

                // An outer split index: some loops need to be unswitched depending on whether padding exists
                //
                // Front-padding:
                //  Suppose padSize = 8, splitSize = 5 (padSize can easily be 3 as well):
                //
                //  o o o o o
                //  o o o x x <-- unswitch at first ceiling boundary [padded outer indices]
                //  x x x x x
                //  x x x x x
                //  x x x x x
                //  x x x o o <-- unswitch at (rangeSize - extra) [all outer indices, if applicable]
                //
                // Back-padding: handled by unswitching the end boundary block

                // Reconcile the domain constraints with the active loop range by taking their overlap:
                // - The domain constraints represent the active iteration space (with padding excluded)
                // - The active loop range represents the loop variable's current state, such as whether we are within a boundary block
                const auto constraints = domain.GetConstraints();
                auto [begin, end] = constraints.GetEffectiveRangeBounds(loopIndex);
                begin = std::max(loopRange.Begin(), begin);
                end = std::min(loopRange.End(), end);
                assert(begin < end && "Could not reconcile loop ranges"); // likely a boundary block splitting logic error

                if (domain.IsPaddedIndex(loopIndex) && begin > 0)
                {
                    // Front-padded index
                    // Unswitch the first partial block, at the first ceiling boundary
                    // (i.e. the split boundary that immediately follows padding)
                    // TODO: is begin > 0 sufficient or do we need to add logic to detect if we are in a boundary block?
                    auto padSize = begin;
                    if (padSize != 0 && padSize != splitSize)
                    {
                        // Unswitch the first partial block, at the first ceiling boundary
                        // (i.e. the split boundary that immediately follows the padSize element)
                        auto firstCeilBound = (padSize / splitSize + 1) * splitSize;
                        if (firstCeilBound < end)
                        {
                            splits.insert(firstCeilBound);
                        }
                    } // else front-padding does not affect the size of the first split block
                }

                // Unswitch the final partial block if the end-block exceeds the split size AND is a non-multiple of the split size
                // (also handles the end-padding case)
                auto extra = end % splitSize;
                if (end > splitSize && extra > 0)
                {
                    splits.insert(end - extra);
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
                            // single select value (split points are just before the selected value, and 1 increment after)
                            assert(indexValues.size() == 1 && "Invalid number of index values for select predicate");
                            splitVals.push_back(indexValues[0]);
                            splitVals.push_back(indexValues[0] + 1);
                            break;
                        case FragmentType::endBoundary:
                            // already set by automatic boundary-handling code
                            break;
                        case FragmentType::range: {
                            // custom begin & end (increment is ignored for now)
                            assert(indexValues.size() >= 2 && "Invalid number of index values for range predicate");
                            auto customBegin = indexValues[0];
                            auto customEnd = indexValues[1];
                            splitVals.push_back(customBegin);

                            if (domain.IsSplitIndex(loopIndex, /*inner=*/false))
                            {
                                // compute the boundary blocks
                                // first possible range is [customBegin, customEnd)
                                // second possible range is [customEnd, loopRange.End())
                                // Note: we ignore/don't care if other intermediate ranges are defined elsewhere
                                // and err on the side of splitting more often than necessary
                                std::vector<int64_t> blockEnds = { customEnd, loopRange.End() };
                                auto splitSize = loopRange.Increment();
                                int64_t start = customBegin;
                                for (auto e : blockEnds)
                                {
                                    auto rangeSize = e - start;
                                    auto extra = rangeSize % splitSize;
                                    if (rangeSize > splitSize && extra > 0)
                                    {
                                        splitVals.push_back(e - extra);
                                    }
                                    start = e;
                                }
                            }
                            splitVals.push_back(customEnd);
                            break;
                        }
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

        auto addValidSplits = [&domain, &loopIndex, &loopRange, &runtimeIndexVariables, &schedule, containsFragmentPredicate](Operation* p, const std::set<int64_t>& splits, std::set<int64_t>& allSplits_) -> void {
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
                        allSplits_.insert(splitPt);
                        prevBegin = splitPt;
                        prevVal = splitVal;
                    }
                }

                // take care of end case
                auto lastSplit = *(splits.rbegin());
                if (prevBegin != lastSplit && prevBegin != loopRange.Begin())
                {
                    allSplits_.insert(prevBegin);
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
        if (range.IsVariable())
            return;

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
        auto placementPredicate = kernel.getPlacementPredicate();
        auto isEmpty = !kernel.getPlacementPredicate() || static_cast<bool>(dyn_cast_or_null<NullPredicateOp>(kernel.getPlacementPredicate().getOperation()));

        if (isEmpty || IsBodyPlacementPredicate(placementPredicate))
        {
            // TODO: doesn't need to be innermost, just inner enough that none of the "regular" predicates depend on any inner loops

            // HACK: check if there's an EvaluatablePredicate, to allow non-innermost "body" kernels with such a predicate
            //
            // TODO: merge the EvaluatablePredicate interface with the old "placement" predicate (or just replace the
            //       "placement" predicate concept with EvaluatablePredicate)
            auto hasEvaluatablePredicate = kernel.getEvaluatablePredicate() != nullptr;
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

            auto kernelPredicate = k.getKernelPredicate();
            auto evaluatablePredicate = k.getEvaluatablePredicate();
            if (kernelPredicate)
            {
                auto builder = const_cast<LoopNestBuilder*>(this)->GetCurrentLoopBuilder(schedule);
                auto simplifiedPredicate = kernelPredicate.simplify(builder, GetDomain(), runtimeIndexVariables, schedule);
                auto result = simplifiedPredicate.evaluate(GetDomain(), runtimeIndexVariables, schedule);
                if (result.has_value() && *result == false)
                {
                    return false;
                }
                return true;
            }
            else if (evaluatablePredicate)
            {
                auto predicateResult = evaluatablePredicate.evaluate(definedIndices, loopIndex, position);
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

    arith::ConstantIndexOp LoopNestBuilder::GetConstantIndex(OpBuilder& builder, int64_t indexVal)
    {
        OpBuilder::InsertionGuard guard(builder);

        if (_constantIndices.count(indexVal) == 0)
        {
            auto loc = GetLocation();
            auto block = _constantOpBuilder.getInsertionBlock();
            _constantOpBuilder.setInsertionPoint(block, block->begin());
            _constantIndices[indexVal] = _constantOpBuilder.create<arith::ConstantIndexOp>(loc, indexVal);
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

    LoopRange LoopNestBuilder::MakeLoopRange(mlir::OpBuilder& builder, const Range& range)
    {
        // TODO : merge these scenarios better
        if (range.HasValueMapBegin() && range.HasValueMapEnd())
        {
            // Create affine apply ops for the begin and end values
            auto beginApplyOps = util::AffineValueMapToAffineApplyOps(builder, GetLocation(), range.ValueMapBegin());
            auto endApplyOps = util::AffineValueMapToAffineApplyOps(builder, GetLocation(), range.ValueMapEnd());

            // TODO : do we need to support situations with multiple begin/end values. (These turn into mins/maxes that get emitted and evaluated at runtime)
            assert(beginApplyOps.size() == 1);
            assert(endApplyOps.size() == 1);
            return {
                beginApplyOps[0].getResult(),
                endApplyOps[0].getResult(),
                GetConstantIndex(builder, range.Increment()).getResult()
            };
        }
        else if (range.HasVariableEnd())
        {
            return {
                GetConstantIndex(builder, range.Begin()).getResult(),
                range.VariableEnd(),
                GetConstantIndex(builder, range.Increment()).getResult()
            };
        }
        return MakeLoopRange(builder, range.Begin(), range.End(), range.Increment());
    }

    void LoopNestBuilder::DefinePostLoopIndex(OpBuilder& builder, const Index& loopIndex, LoopIndexSymbolTable& runtimeLoopIndices, const LoopVisitSchedule& schedule)
    {
        auto loopRange = schedule.GetActiveLoopRange(GetDomain(), loopIndex, runtimeLoopIndices);
        arith::ConstantIndexOp firstVal = GetConstantIndex(builder, loopRange.Begin());
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

    mlir::OpBuilder LoopNestBuilder::GetCurrentLoopBuilder(const LoopVisitSchedule& schedule, ScheduledLoopOp innerLoop)
    {
        if (schedule.IsOutermostLoop())
        {
            return _builder;
        }

        auto loop = innerLoop->getParentOfType<ScheduledLoopOp>();
        return loop.getBodyBuilder();
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

    // TODO: make a more general "unswitch this loop" function
    bool LoopNestBuilder::IsGpuLoop(Index loopIndex) const
    {
        auto execPlan = GetScheduleOp().getOrCreateExecPlan();
        if (execPlan.exec_target() != ir::value::ExecutionTarget::GPU)
        {
            return false;
        }

        return execPlan.hasBinding(loopIndex);
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
        for (auto& kernelOp : kernelBody.without_terminator())
        {
            builder.clone(kernelOp, argMap);
        }

        auto scheduledLoopOp = invokeOp->getParentOp();
        llvm::SmallVector<Attribute> kernelIds{ builder.getStringAttr(kernelId) };
        if (auto kernelsAttr = scheduledLoopOp->getAttrOfType<ArrayAttr>("kernels"))
        {
            kernelIds.insert(kernelIds.begin(), kernelsAttr.begin(), kernelsAttr.end());
        }
        scheduledLoopOp->setAttr("kernels", builder.getArrayAttr(kernelIds));
    }

    mlir::Location LoopNestBuilder::GetLocation()
    {
        std::string tag = "LoopNestBuilder";
        return util::GetLocation(_builder, tag, _schedule.getLoc());
    }

} // namespace loopnest
} // namespace accera::ir
