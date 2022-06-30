////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Index.h"
#include "OperandIndex.h"

#include <mlir/IR/Value.h>

#include <iosfwd>
#include <variant>

namespace accera::ir
{
namespace loopnest
{
    /// <summary>
    /// A class representing the half-open interval `[begin, end)`, with an increment between points of _increment.
    /// </summary>
    class Range
    {
    public:
        /// <summary> Constructor that creates a range </summary>
        /// <param name="begin"> The begin index </param>
        /// <param name="end"> The end index </param>
        /// <param name="increment"> The increment </param>
        Range(int64_t begin, int64_t end, int64_t increment = 1);
        Range(int64_t begin, Index endIndex, int64_t increment = 1);
        Range(int64_t begin, mlir::Value end, int64_t increment = 1);
        Range(int64_t begin, OperandIndex end, int64_t increment = 1);

        int64_t Begin() const;
        int64_t End() const;
        mlir::Value VariableEnd() const;
        bool HasConstantEnd() const;
        bool HasVariableEnd() const;
        bool HasIndexEnd() const;
        bool HasOperandIndexEnd() const;
        Index EndIndex() const;
        OperandIndex EndOperandIndex() const;
        int64_t Size() const;
        int64_t Increment() const;
        int64_t NumIterations() const;
        int64_t LastIterationBegin() const;

    private:
        int64_t _begin;
        std::variant<int64_t, Index, OperandIndex, mlir::Value> _end;
        int64_t _increment;
    };

    bool operator==(const Range& i1, const Range& i2);
    bool operator!=(const Range& i1, const Range& i2);
    bool operator<(const Range& i1, const Range& i2);
    bool operator<=(const Range& i1, const Range& i2);

    bool Intersects(const Range& a, const Range& b);

    std::ostream& operator<<(std::ostream& os, const Range& r);
} // namespace loopnest
} // namespace accera::ir
