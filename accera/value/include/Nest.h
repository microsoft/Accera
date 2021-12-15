////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <vector>

#include "Index.h"
#include "IterationDomain.h"
#include "Range.h"
#include "Scalar.h"
#include "ScalarIndex.h"

#include <utilities/include/MemoryLayout.h>
#include <utilities/include/TupleUtils.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

namespace accera::ir::loopnest
{
class NestOp;
}

namespace accera
{
namespace value
{
    using loopnests::Index;
    using loopnests::IterationDomain;
    using loopnests::Range;
    using utilities::MemoryLayout;

    using ScalarIndexPair = std::pair<ScalarIndex, ScalarIndex>;
    class Schedule;
    class NestImpl;

    class Nest
    {
    public:
        /// <summary> Constructor that creates a nest from a MemoryShape </summary>
        /// <param name="sizes"> Memory shape describing the sizes </param>
        Nest(const utilities::MemoryShape& sizes);

        /// <summary> Constructor that creates a nest from a vector of ranges </summary>
        /// <param name="ranges"> A vector of accera::ir::loopnest::Range 's</param>
        Nest(const std::vector<Range>& ranges);

        Nest(const IterationDomain& domain);
        Nest(Nest&& other);

        ~Nest();

        /// <summary> Returns the specified index for this nest, with the outermost index being index 0 </summary>
        ScalarIndex GetIndex(int pos);

        /// <summary> Returns the indices for this nest, starting from the outermost index </summary>
        std::vector<ScalarIndex> GetIndices();

        /// <summary> Returns the indices for this nest, starting from the outermost index </summary>
        template <int N>
        utilities::RepeatTuple<ScalarIndex, N> GetIndices();

        IterationDomain GetDomain() const;

        /// <summary> Sets the default kernel function to be run in the innermost loop </summary>
        void Set(std::function<void()> kernelFn);

        /// <summary> Creates a schedule to run this nest </summary>
        Schedule CreateSchedule();

        void dump();

    private:
        friend class Schedule;

        accera::ir::loopnest::NestOp GetOp();

        std::unique_ptr<NestImpl> _impl;
    };

} // namespace value
} // namespace accera

#pragma region implementation

namespace accera
{
namespace value
{
    template <int N>
    utilities::RepeatTuple<ScalarIndex, N> Nest::GetIndices()
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
