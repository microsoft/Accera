////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "FunctionDeclaration.h"
#include "Index.h"
#include "Scalar.h"
#include "ScalarIndex.h"

#include <utilities/include/TupleUtils.h>

#include <cassert>
#include <memory>
#include <vector>

namespace accera::ir::loopnest
{
class ScheduleOp;
class KernelPredicateOpInterface;
} // namespace accera::ir::loopnest

namespace accera
{
namespace value
{
    using loopnests::Index;
    using loopnests::SplitIndex;

    class Nest;
    class Kernel;
    class KernelPredicate;
    class Plan;
    class GPUPlan;
    class ScheduleImpl;

    using ScalarIndexPair = std::pair<ScalarIndex, ScalarIndex>;

    class Schedule
    {
    public:
        Schedule(Schedule&&) noexcept;
        Schedule(const Schedule&);
        Schedule& operator=(const Schedule&);
        Schedule& operator=(Schedule&&) noexcept;
        ~Schedule();

        /// <summary> Returns the indices for this schedule, starting from the outermost index </summary>
        std::vector<ScalarIndex> GetIndices();

        /// <summary> Returns the indices for this schedule, starting from the outermost index </summary>
        template <int N>
        utilities::RepeatTuple<ScalarIndex, N> GetIndices();

        /// <summary> Split a loop along a dimension </summary>
        /// <param name="i"> The dimension to split </param>
        /// <param name="factor"> The blocksize to split </param>
        /// <returns> A SplitIndex, containing the inner and outer indices.</returns>
        SplitIndex Split(Index i, int factor);

        /// <summary> Split a loop along a dimension </summary>
        /// <param name="i"> The dimension to split </param>
        /// <param name="factor"> The blocksize to split </param>
        /// <returns> An std::pair of scalars indicating the outer and inner indices </returns>
        ScalarIndexPair Split(ScalarIndex i, int factor); // `i` must be backed by a SymbolicIndexOp

        /// <summary> Pads one dimension with empty (no-op) elements </summary>
        /// <param name="i"> The dimension to pad </param>
        /// <param name="size"> The number of elements to pad </param>
        /// <param name="padFront"> If set to true, inserts padding before the zeroth element, else appends padding</param>
        /// <returns> The padded index </returns>
        Index Pad(Index i, int size, bool padFront=true);

        /// <summary> Pads one dimension with empty (no-op) elements </summary>
        /// <param name="i"> The dimension to pad </param>
        /// <param name="size"> The number of elements to pad </param>
        /// <param name="padFront"> If set to true, inserts padding before the zeroth element, else appends padding</param>
        /// <returns> The padded index </returns>
        ScalarIndex Pad(ScalarIndex i, int size, bool padFront=true);

        /// <summary> Skews one dimension along another dimension </summary>
        /// <param name="i"> The dimension to skew </param>
        /// <param name="reference"> The reference dimension </param>
        /// <returns> The skewed index </returns>
        Index Skew(Index i, Index reference);

        /// <summary> Skews one dimension along another dimension </summary>
        /// <param name="i"> The dimension to skew </param>
        /// <param name="reference"> The reference dimension </param>
        /// <returns> The skewed index </returns>
        ScalarIndex Skew(ScalarIndex i, ScalarIndex reference);

        /// <summary> Unroll the loop along a dimension </summary>
        /// <param name="i"> The dimension to unroll </param>
        /// <param name="size"> Unroll only if the range is smaller than size (exclusive) </param>
        void Unroll(Index i, std::optional<uint64_t> size = std::nullopt);

        /// <summary> Unroll the loop along a dimension </summary>
        /// <param name="i"> The dimension to unroll </param>
        /// <param name="size"> Unroll only if the range is smaller than size (exclusive) </param>
        void Unroll(ScalarIndex i, std::optional<uint64_t> size = std::nullopt); // `i` must be backed by a SymbolicIndexOp

        /// <summary> Partially unroll the loop along a dimension </summary>
        /// <param name="i"> The dimension to unroll </param>
        /// <param name="factor"> The number of times to unroll the loop </param>
        void InterleavedUnroll(Index i, uint64_t factor);

        /// <summary> Partially unroll the loop along a dimension </summary>
        /// <param name="i"> The dimension to unroll </param>
        /// <param name="factor"> The number of times to unroll the loop </param>
        void InterleavedUnroll(ScalarIndex i, uint64_t factor);

        /// <summary> Sets the nest ordering </summary>
        /// <param name="order"> The order of loop indices to use, starting from the outermost loop </param>
        void SetOrder(std::vector<Index> order);

        /// <summary> Sets the nest ordering </summary>
        /// <param name="order"> The order of loop indices to use, starting from the outermost loop </param>
        void SetOrder(std::vector<ScalarIndex> order); // ScalarIndexes must be backed by a SymbolicIndexOp

        /// <summary> Adds a kernel to the schedule </summary>
        /// <param name="kernel"> The kernel to execute </param>
        void AddKernel(const Kernel& kernel);

        /// <summary> Adds a kernel to the schedule </summary>
        /// <param name="kernel"> The kernel to execute </param>
        /// <param name="pred"> The predicate that determines when the kernel should run  </param>
        void AddKernel(const Kernel& kernel, const KernelPredicate& pred);

        /// <summary> Adds a kernel to the schedule </summary>
        /// <param name="kernel"> The kernel to execute </param>
        /// <param name="pred"> The predicate that determines when the kernel should start  </param>
        /// <param name="placement"> The predicate that determines where the kernel code should be placed </param>
        void AddKernel(const Kernel& kernel, const KernelPredicate& pred, const KernelPredicate& placement);

        /// Fuse other schedules into this one, destroying the other ones.
        ///
        /// `indexCorrespondences` is a list of index lists, indicating which indices are being fused together.
        ///
        /// Returns the "fusing" index, which will be the outermost loop index. The rest
        /// follow the order in `indexCorrespondences`. Any indices in the original nests not in `indexCorrespondences` will be added
        /// to the end (they will be the innermost loops), starting with the ones from the first nest, and then the second.
        ScalarIndex Fuse(std::vector<Schedule>& others, const std::vector<std::vector<ScalarIndex>>& indexCorrespondences);

        /// <summary> Creates an execution plan for the host </summary>
        /// <returns> The execution plan </returns>
        Plan CreatePlan();

        /// <summary> Creates an execution plan for the GPU </summary>
        /// <param name="gpuOptions"> The target GPU options </param>
        /// <param name="execRuntime"> The target execution runtime </param>
        /// <returns> The execution plan </returns>
        GPUPlan CreateGPUPlan(targets::GPU gpuOptions, ExecutionRuntime execRuntime = ExecutionRuntime::Default);

        void dump();

        ScheduleImpl& GetImpl();

        /// returns the `Index` object represented by a `ScalarIndex` (which much be an instance of SymbolicIndexOp)
        Index ResolveIndex(ScalarIndex index);

        ScalarIndex LookUpIndex(Index index);

        void SetSaturatedFlag(Index i);
        void SetSaturatedFlag(ScalarIndex i);

    private:
        friend class Nest;
        friend class Plan;
        friend class GPUPlan;
        Schedule(Nest& nest);
        Schedule(accera::ir::loopnest::ScheduleOp op);

        accera::ir::loopnest::ScheduleOp GetOp() const;

        std::unique_ptr<ScheduleImpl> _impl;
    };

} // namespace value
} // namespace accera

#pragma region implementation

namespace accera
{
namespace value
{
    template <int N>
    utilities::RepeatTuple<ScalarIndex, N> Schedule::GetIndices()
    {
        using std::begin;
        using std::end;
        utilities::RepeatTuple<ScalarIndex, N> result;
        auto indices = GetIndices();
        assert(indices.size() >= N);

        return utilities::VectorToTuple<N>(indices);
    }

} // namespace value
} // namespace accera
