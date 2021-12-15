////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "AffineExpression.h"
#include "LoopIndexInfo.h"
#include "LoopNestOps.h"
#include "LoopVisitSchedule.h"
#include "TransformedDomain.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/InliningUtils.h>

#include <functional>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace accera::ir
{
namespace loopnest
{
    class LoopNestBuilder;

    struct Partition
    {
        Index index;
        Range range;
    };
    using PartitionList = std::vector<Partition>;

    struct LoopRange
    {
        LoopRange(mlir::Value start, mlir::Value stop, mlir::Value increment);

        mlir::Value start;
        mlir::Value stop;
        mlir::Value step;

        Range GetConstantRange() const;
    };

    /// <summary>
    /// Abstract base class for objects that visit a loop nest (e.g., code generators)
    /// </summary>
    class LoopNestBuilder
    {
    public:
        LoopNestBuilder(ScheduleOp op, mlir::PatternRewriter& builder, bool printLoops = false);

        void BuildLoopNest();

        const std::vector<ScheduledKernelOp>& GetKernelGroup(std::string id) const;
        std::vector<std::string> GetKernelIds() const;

    private:
        struct RecursionState
        {
            RecursionState(const LoopNestBuilder&);
            RecursionState(const RecursionState&) = default;

            // loopIndices is a map from a loop Index variable to:
            //   - the actual induction variable for that loop,
            //   - the range visited by that variable in this branch of the code (for loops that have already been visited), and
            //   - the state of the variable's loop (before, inside, after)
            LoopIndexSymbolTable loopIndices;

            std::unordered_map<KernelId, bool> validKernelGroups;
            std::vector<int64_t> subdomainSize;
        };

        // The main "passes" in code generation
        RecursionState GenerateLoopStructure(const RecursionState& state, const LoopVisitSchedule& schedule);
        void MergeAdjacentKernelBodies(std::vector<ScheduledLoopOp> loops, const LoopVisitSchedule& schedule);
        RecursionState EmitLoopBodies(std::vector<ScheduledLoopOp> loops, const RecursionState& state, const LoopVisitSchedule& schedule);
        // TODO: unswitch loops
        // TODO: add kernels
        void ApplyInjectableMappings();

        void InvokeKernel(OpBuilder& builder, ScheduledKernelOp kernel, Position position, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule);

        TransformedDomain GetDomain() const;
        LoopVisitSchedule GetLoopSchedule() const;

        ScheduledLoopOp EmitLoopOp(const LoopRange& range, const RecursionState& state, const LoopVisitSchedule& schedule);
        void GenerateLoopBody(mlir::Value index, const LoopRange& r, const RecursionState& state, const LoopVisitSchedule& schedule);
        void EmitLoopBody(ScheduledLoopOp loop, const RecursionState& state, const LoopVisitSchedule& schedule);
        void EndLoopRange(const LoopRange& range, const RecursionState& state, const LoopVisitSchedule& schedule);

        /// Returns `true` if the current loop body is inside the loop for the given index (so, "inside" counts the current loop being emitted)
        bool IsFullyDefined(const Index& index, const LoopVisitSchedule& schedule) const;

        /// Returns `true` if the current loop body is inside the loop for the given index (so, "inside" counts the current loop being emitted)
        bool AreAllFullyDefined(const std::vector<Index>& indices, const LoopVisitSchedule& schedule) const;

        RecursionState InvokeKernels(const LoopRange& r, Position position, const RecursionState& state, const LoopVisitSchedule& schedule);
        void ReplaceInvokeOps(Block* loopBlock, const RecursionState& state);

        bool InvokeKernelGroup(std::string id, Position position, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule);
        RecursionState GenerateAfterBodyState(const LoopRange& r, const RecursionState& state, const LoopVisitSchedule& schedule);
        std::set<KernelId> GetEpilogueKernelIds(const LoopRange& r, const RecursionState& state, const LoopVisitSchedule& schedule);
        std::vector<std::string> GetPossiblyValidKernelIds(const RecursionState& state) const;
        std::vector<ScheduledKernelOp> GetPossiblyValidKernels(const RecursionState& state) const;

        PartitionList GetPartitions(const Index& loopIndex, Range loopRange, const RecursionState& state, const LoopVisitSchedule& schedule) const;

        // TODO: change this to take a KernelPredicate instead of Operation*
        void AddSplits(const Index& loopIndex, Range loopRange, Operation* predicate, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule, std::set<int64_t>& splits) const;

        void UpdateSubdomainSizes(const Index& loopIndex, const LoopRange& range, std::vector<int64_t>& subdomainSize);

        void DefineComputedIndexVariables(LoopIndexSymbolTable& runtimeLoopIndices, const std::vector<ScheduledKernelOp>& activeKernels, const LoopVisitSchedule& schedule);
        LoopIndexSymbolTable GetRuntimeIndexVariables(const LoopIndexSymbolTable& runtimeLoopIndices, const LoopVisitSchedule& schedule) const;
        void DefinePostLoopIndex(OpBuilder& builder, const Index& loopIndex, LoopIndexSymbolTable& runtimeLoopIndices, const LoopVisitSchedule& schedule);

        bool IsPlacementValid(ScheduledKernelOp kernel, const LoopIndexSymbolTable& runtimeLoopIndices, const LoopVisitSchedule& schedule, const Position& position) const;
        std::vector<ScheduledKernelOp> GetValidKernels(const std::vector<ScheduledKernelOp>& kernelGroup, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule, const Position& position) const;

        void EmitKernelBody(OpBuilder& builder, InvokeKernelOp invokeOp, const LoopIndexSymbolTable& runtimeIndexVariables);
        mlir::Value EmitIndexExpression(OpBuilder& builder, mlir::Location loc, const AffineExpression& expr, const TransformedDomain& domain);
        AffineExpression GetIndexExpression(OpBuilder& builder, const Index& index, const TransformedDomain& domain) const;

        std::vector<size_t> GetLogicalDimensionPositions(const Index& index) const;

        SymbolicIndexOp GetSymbolicIndex(Index index);
        mlir::ConstantIndexOp GetConstantIndex(mlir::OpBuilder& builder, int64_t value);
        LoopRange MakeLoopRange(mlir::OpBuilder& builder, int64_t start, int64_t stop, int64_t increment);
        std::vector<ScheduledLoopOp> FindAllScheduledLoops(Index index);
        mlir::OpBuilder GetCurrentLoopBuilder(const LoopVisitSchedule& schedule);
        ScheduledLoopOp FindLatestScheduledLoop(Index index);

        bool IsSaturated(Index loopIndex) const;
        std::optional<uint64_t> GetUnrollIfRangeSmallerThan(Index loopIndex) const;
        std::optional<uint64_t> GetUnrollAndJamFactor(Index loopIndex) const;
        ScheduleOp GetScheduleOp() const;
        mlir::Region* GetScheduleParentRegion() const;
        std::map<std::string, std::vector<ScheduledKernelOp>> DiscoverKernelGroups() const;
        mlir::Location GetLocation();
        void EnsureTerminators();

        ScheduleOp _schedule;
        std::map<std::string, std::vector<ScheduledKernelOp>> _kernelGroups;
        std::map<Index, std::vector<ScheduledLoopOp>> _loops;
        std::map<int64_t, mlir::ConstantIndexOp> _constantIndices;
        mlir::PatternRewriter& _builder;
        mlir::OpBuilder _constantOpBuilder;
        bool _printLoops = true;
    };

} // namespace loopnest
} // namespace accera::ir
