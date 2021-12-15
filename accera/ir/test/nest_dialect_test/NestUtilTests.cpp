////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

#include <ir/include/nest/AffineExpression.h>
#include <ir/include/nest/Index.h>
#include <ir/include/nest/IndexRange.h>
#include <ir/include/nest/IterationDomain.h>
#include <ir/include/nest/OperandIndex.h>
#include <ir/include/nest/Range.h>
#include <ir/include/nest/TransformedDomain.h>

#include "LoopNestTestVerification.h"

#include <mlir/IR/Builders.h>

using namespace accera::ir::loopnest;

TEST_CASE("Index tests")
{
    Index i1;
    Index i2("i2");
    Index i2_2("i2");
    Index i3("i3", 123);
    Index i1_copy = i1;
    Index iNone = Index::none;

    REQUIRE(i1 == i1);
    REQUIRE(i1 == i1_copy);
    REQUIRE(i1.GetId() == i1_copy.GetId());
    REQUIRE(i1 != i2);
    REQUIRE(i1 != i2_2);
    REQUIRE(i1 != i3);

    REQUIRE(i2 != i2_2);
    REQUIRE(i2.GetName() == "i2");
    REQUIRE(i2_2.GetName() == "i2");
    REQUIRE(i2.GetId() != i2_2.GetId());

    REQUIRE(i3.GetId() == 123);
}

TEST_CASE("Constant Range tests")
{
    using namespace accera::ir::loopnest;

    Range r_0_10(0, 10);
    Range r_0_5(0, 5);
    Range r_5_10(5, 10);
    Range r_0_10_1(0, 10, 1);
    Range r_0_10_2(0, 10, 2);
    Range r_5_10_2(5, 10, 2);

    REQUIRE(r_0_10 == r_0_10);
    REQUIRE(r_0_10 != r_5_10);
    REQUIRE(r_0_10 == r_0_10_1);
    REQUIRE(r_0_10.HasConstantEnd());
    REQUIRE(!r_0_10.HasIndexEnd());
    REQUIRE(!r_0_10.HasOperandIndexEnd());
    REQUIRE(r_0_10.Begin() == 0);
    REQUIRE(r_0_10.End() == 10);
    REQUIRE(r_0_10.Size() == 10);
    REQUIRE(r_0_10.Increment() == 1);
    REQUIRE(r_0_10.NumIterations() == 10);

    REQUIRE(r_0_10_2.Begin() == 0);
    REQUIRE(r_0_10_2.End() == 10);
    REQUIRE(r_0_10_2.Size() == 10);
    REQUIRE(r_0_10_2.Increment() == 2);
    REQUIRE(r_0_10_2.NumIterations() == 5);

    REQUIRE(r_5_10_2.Begin() == 5);
    REQUIRE(r_5_10_2.End() == 10);
    REQUIRE(r_5_10_2.Size() == 5);
    REQUIRE(r_5_10_2.Increment() == 2);
    REQUIRE(r_5_10_2.NumIterations() == 3);
    REQUIRE(r_5_10_2.LastIterationBegin() == 9);

    REQUIRE(Intersects(r_0_10, r_5_10));
    REQUIRE(Intersects(r_0_10, r_5_10_2));
    REQUIRE(!Intersects(r_0_5, r_5_10));

    // Ranges sort lexicographically by (start, end, step)
    std::vector<Range> ranges{ r_5_10, r_5_10_2, r_0_5, r_0_10 };
    std::sort(ranges.begin(), ranges.end());
    REQUIRE(ranges == std::vector<Range>{ r_0_5, r_0_10, r_5_10, r_5_10_2 });
}

TEST_CASE("IndexRange tests")
{
    Index i("i");
    Range r(0, 10);

    IndexRange ir(i, r);
    IndexRange ir2(i, r);
    IndexRange jr("j", r);

    REQUIRE(ir.GetIndex() == i);
    REQUIRE(jr.GetName() == "j");
    REQUIRE(ir.Begin() == 0);
    REQUIRE(ir.End() == 10);
    REQUIRE(ir.Size() == 10);
    REQUIRE(ir.Increment() == 1);
    REQUIRE(ir.GetRange() == r);

    REQUIRE(ir == ir2);
    REQUIRE(ir != jr);
    REQUIRE(ir < jr);
    REQUIRE(!(jr < ir));
}

TEST_CASE("Domain tests")
{
    auto& b = GetTestBuilder();
    auto context = b.getContext();

    const int64_t M = 20, N = 10;
    Index i("i"), j("j"), k("");
    IterationDomain d({ { i, { 0, M } }, { j, { 0, N } } });
    REQUIRE(d.NumDimensions() == 2);

    REQUIRE(d.GetDimensionRange(i).GetIndex() == i);

    REQUIRE(d.GetDimensionRange(i) == d.GetDimensionRange(0));
    REQUIRE(d.GetDimensionRange(j) == d.GetDimensionRange(1));

    REQUIRE(d.GetDimensionRange(0).GetRange() == Range(0, M));
    REQUIRE(d.GetDimensionRange(i).GetRange() == Range(0, M));
    REQUIRE(d.GetDimensionRange(1).GetRange() == Range(0, N));
    REQUIRE(d.GetDimensionRange(j).GetRange() == Range(0, N));

    TransformedDomain td(d);
    REQUIRE(td.NumDimensions() == 2);
    REQUIRE(td.NumLoopIndices() == 2);
    REQUIRE(td.NumIndices() == 2);

    auto [i0, i1] = td.Split(i, 10, context);
    auto [i2, i3] = td.Split(i1, 5, context);
    REQUIRE(td.NumDimensions() == 2);
    REQUIRE(td.NumLoopIndices() == 4); // i0, j, i2, i3
    REQUIRE(td.NumIndices() == 6); // i, i0, j, i1, i2, i3

    REQUIRE(td.Exists(i));
    REQUIRE(td.Exists(i0));
    REQUIRE(td.Exists(i1));
    REQUIRE(td.Exists(i2));
    REQUIRE(td.Exists(i3));
    REQUIRE(td.Exists(j));
    REQUIRE(!td.Exists(k));

    REQUIRE(td.IsDimension(i));
    REQUIRE(!td.IsDimension(i0));
    REQUIRE(!td.IsDimension(i1));
    REQUIRE(!td.IsDimension(i2));
    REQUIRE(!td.IsDimension(i3));
    REQUIRE(td.IsDimension(j));
    REQUIRE(!td.IsDimension(k));

    REQUIRE(!td.IsLoopIndex(i));
    REQUIRE(td.IsLoopIndex(i0));
    REQUIRE(!td.IsLoopIndex(i1));
    REQUIRE(td.IsLoopIndex(i2));
    REQUIRE(td.IsLoopIndex(i3));
    REQUIRE(td.IsLoopIndex(j));
    REQUIRE(!td.IsLoopIndex(k));

    REQUIRE(td.IsComputedIndex(i));
    REQUIRE(!td.IsComputedIndex(i0));
    REQUIRE(td.IsComputedIndex(i1));
    REQUIRE(!td.IsComputedIndex(i2));
    REQUIRE(!td.IsComputedIndex(i3));
    REQUIRE(!td.IsComputedIndex(j));
    REQUIRE(!td.IsComputedIndex(k));

    REQUIRE(td.DependsOn(i, i0));
    REQUIRE(td.DependsOn(i, i1));
    REQUIRE(td.DependsOn(i1, i2));
    REQUIRE(!td.DependsOn(i0, i2));
    REQUIRE(!td.DependsOn(i2, i3));

    REQUIRE(td.IsSplitIndex(i1, /*inner=*/true));
    REQUIRE(td.IsSplitIndex(i3, /*inner=*/true));
    REQUIRE(!td.IsSplitIndex(i2, /*inner=*/true));
    REQUIRE(!td.IsSplitIndex(i0, /*inner=*/true));
    REQUIRE(!td.IsSplitIndex(i, /*inner=*/true));
    REQUIRE(td.GetOtherSplitIndex(i3) == i2);
    REQUIRE(td.GetOtherSplitIndex(i1) == i0);
}

TEST_CASE("Domain skew tests")
{
    auto context = GetTestBuilder().getContext();

    const int64_t M = 20, N = 10;
    Index i("i"), j("j"), k("");
    IterationDomain d({ { i, { 0, M } }, { j, { 0, N } } });
    TransformedDomain td(d);
    REQUIRE(td.NumDimensions() == 2);
    REQUIRE(td.NumLoopIndices() == 2);
    REQUIRE(td.NumIndices() == 2);

    auto [i0, i1] = td.Split(i, 10, context);
    auto [i2, i3] = td.Split(i1, 5, context);
    auto i4 = td.Skew(i3, j, context);

    REQUIRE(td.NumDimensions() == 2);
    REQUIRE(td.NumLoopIndices() == 4); // i0, j, i2, i4
    REQUIRE(td.NumIndices() == 7); // i, i0, j, i1, i2, i3, i4
    REQUIRE(td.IsComputedIndex(i3));
    REQUIRE(!td.IsComputedIndex(i4));
    REQUIRE(!td.IsComputedIndex(j));
    REQUIRE(td.DependsOn(i3, i4));
    REQUIRE(td.DependsOn(i3, j));
    REQUIRE(!td.IsLoopIndex(i3));
    REQUIRE(td.IsLoopIndex(i4));
    REQUIRE(td.IsLoopIndex(j));
    REQUIRE(!td.IsSplitIndex(i4, /*inner=*/true));
    auto result = td.IsSkewedOrReferenceIndex(i4);
    REQUIRE(result);
    REQUIRE((*result).first);
    REQUIRE((*result).second == j);
    result = td.IsSkewedOrReferenceIndex(j);
    REQUIRE(result);
    REQUIRE((*result).first == false);
    REQUIRE((*result).second == i4);
    result = td.IsSkewedOrReferenceIndex(i3);
    REQUIRE(!result);

    auto constraints = td.GetConstraints();
    auto indices = td.GetIndices();
    for (const auto& index : indices)
    {
        auto range = td.GetIndexRange(index);
        std::cout << index.GetName() << ": " << range << std::endl;
        auto [begin, end] = constraints.GetEffectiveRangeBounds(index);
        REQUIRE(begin == range.Begin());
        REQUIRE(end == range.End());
    }

    // bounds verification for skew
    // fix the skew index to a value, and verify the range of the reference index
    // horizontal bands, starting from width = 1, expanding to width = 5, then shrinking back to 1
    // intervals are closed
    std::vector<std::tuple<int64_t, int64_t, int64_t>> iValuesToJBounds = {
        // i4, [j_begin, j_end]
        { 0, 0, 0 }, // width = 1
        { 1, 0, 1 }, // width = 2
        { 4, 0, 4 }, // width = 4
        { 5, 1, 5 }, // width = 5
        { 6, 2, 6 }, // width = 5
        { 9, 5, 9 }, // width = 5
        { 12, 8, 9 }, // width = 2
        { 13, 9, 9 }, // width = 1
    };
    for (const auto& iValueToJBound : iValuesToJBounds)
    {
        auto constraintsTest = td.GetConstraints();
        auto [iValue, jLB, jUB] = iValueToJBound;
        constraintsTest.AddConstraint(i4, Range(iValue, iValue + 1));
        auto [begin, end] = constraintsTest.GetEffectiveRangeBounds(j);
        REQUIRE(begin == jLB);
        REQUIRE(end == jUB + 1);
    }

    // fix the reference index to a value, verify the range of the skew index
    // vertical strips of length 5
    // intervals are closed
    std::vector<std::tuple<int64_t, int64_t, int64_t>> jValuesToIBounds = {
        // j, [i4_begin, i4_end]
        { 0, 0, 4 },
        { 4, 4, 8 },
        { 9, 9, 13 },
    };
    for (const auto& jValueToIBound : jValuesToIBounds)
    {
        auto constraintsTest = td.GetConstraints();
        auto [jValue, iLB, iUB] = jValueToIBound;
        constraintsTest.AddConstraint(j, Range(jValue, jValue + 1));
        auto [begin, end] = constraintsTest.GetEffectiveRangeBounds(i4);
        REQUIRE(begin == iLB);
        REQUIRE(end == iUB + 1);
    }

    // find an intersection of bounds
    // intervals are closed
    // Note: the result is a union of bounds on the j axis
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> iBoundsToJBounds = {
        // [iBegin, iEnd], [jBegin, jEnd]
        { 0, 0, 0, 0 }, // top triangle
        { 0, 1, 0, 1 }, // top triangle
        { 0, 5, 0, 5 },
        { 1, 6, 0, 6 }, // middle
        { 2, 7, 0, 7 }, // middle
        { 3, 8, 0, 8 }, // middle
        { 10, 13, 6, 9 }, // bottom triangle
        { 12, 13, 8, 9 }, // bottom triangle
    };

    for (const auto& iBoundsToJBound : iBoundsToJBounds)
    {
        auto constraintsTest = td.GetConstraints();
        auto [iLB, iUB, jLB, jUB] = iBoundsToJBound;
        constraintsTest.AddConstraint(i4, Range(iLB, iUB + 1));
        auto [begin, end] = constraintsTest.GetEffectiveRangeBounds(j);
        REQUIRE(begin == jLB);
        REQUIRE(end == jUB + 1);
    }
}

TEST_CASE("Domain pad tests")
{
    auto context = GetTestBuilder().getContext();

    const int64_t M = 21, N = 10;
    Index i("i"), j("j"), k("");
    IterationDomain d({ { i, { 0, M } }, { j, { 0, N } } });
    TransformedDomain td(d);
    REQUIRE(td.NumDimensions() == 2);
    REQUIRE(td.NumLoopIndices() == 2);
    REQUIRE(td.NumIndices() == 2);

    // (21 + 7) split 3 => 28 split 3 => partial(size=2), full(size=3) x 6, partial(size=1)
    const int64_t padSize = 7;
    const int64_t splitSize = 3;

    // pad then split
    // this should result in both front and back padding
    auto i1 = td.Pad(i, padSize, context);
    REQUIRE(td.GetIndexRange(i1) == Range(padSize, M + padSize, 1));
    REQUIRE(td.IsPaddedIndex(i1));

    auto [i2, i3] = td.Split(i1, splitSize, context);
    REQUIRE(td.GetIndexRange(i2) == Range(padSize, M + padSize, splitSize));
    REQUIRE(td.GetIndexRange(i3) == Range(0, splitSize, 1));
    REQUIRE(td.HasPaddedParentIndex(i2));
    REQUIRE(td.HasPaddedParentIndex(i3));

    auto constraints = td.GetConstraints();
    constraints.dump();

    auto [begin, end] = constraints.GetEffectiveRangeBounds(i);
    REQUIRE(begin == 0);
    REQUIRE(end == M);

    auto [begin1, end1] = constraints.GetEffectiveRangeBounds(i1);
    REQUIRE(begin1 == padSize); // shift forward
    REQUIRE(end1 == M + padSize);

    auto [begin2, end2] = constraints.GetEffectiveRangeBounds(i2);
    REQUIRE(begin2 == padSize); // shift forward
    REQUIRE(end2 == M + padSize);

    auto [begin3, end3] = constraints.GetEffectiveRangeBounds(i3);
    REQUIRE(begin3 == 0); // default range (before clamping)
    REQUIRE(end3 == splitSize); // default range (before clamping)

    // Test clamping given the current outer index range by applying additional constraints
    // on the parent index (where parent = outer + inner)
    // intervals are closed
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> parentBoundToInnerBounds = {
        // [parentBegin, parentEnd], [innerBegin, innerEnd]
        { 7, 8, 0, 1 }, // first block (partial, size=2)
        { 9, 11, 0, 2 }, // 2nd block (full, size=3)
        { 12, 14, 0, 2 }, // 3rd block (full, size=3)
        { 24, 26, 0, 2 }, // penultimate block (full, size=3)
        { 27, 27, 0, 0 }, // final block (partial, size=1)
    };

    REQUIRE(td.GetParentIndices(i3).size() > 0);
    auto parentIndex = td.GetParentIndices(i3)[0];
    REQUIRE(i1 == parentIndex); // parent, aka the padded index

    for (const auto& parentBoundToInnerBound : parentBoundToInnerBounds)
    {
        auto constraintsTest = td.GetConstraints();
        auto [pLB, pUB, iLB, iUB] = parentBoundToInnerBound;
        auto [beginOld, endOld] = constraintsTest.GetEffectiveRangeBounds(i3);

        // One limitation of FlatAffineConstraints is that constant ranges are dervied
        // from constant ranges, therefore we simply constrain the inner range size
        // with the parent range size
        auto innerEnd = pUB - pLB;
        constraintsTest.AddConstraint(i3, Range(beginOld, beginOld + innerEnd + 1));

        auto [begin, end] = constraintsTest.GetEffectiveRangeBounds(i3);
        REQUIRE(begin == iLB);
        REQUIRE(end == iUB + 1);
    }
}

TEST_CASE("Domain fusion tests")
{
    const int64_t M = 21, N0 = 10, N1 = 8, K = 16;
    Index i0("i0"), j0("j0"), i1("i1"), j1("j1"), k("k");
    IterationDomain d0({ { i0, { 0, M } }, { j0, { 0, N0 } }, {k, { 0, K }} });
    TransformedDomain td0(d0);

    IterationDomain d1({ { i1, { 0, M } }, { j1, { 0, N1 } } });
    TransformedDomain td1(d1);

    // fuse unequally-shaped iteration domains
    auto td2 = TransformedDomain::Fuse({td0, td1}, { {i0, i1}, {j0, j1} });
    REQUIRE(td2.NumDimensions() == 3); // 2 fused, 1 unfused
    REQUIRE(td2.NumLoopIndices() == 3);
    REQUIRE(td2.NumIndices() == 3);
}
