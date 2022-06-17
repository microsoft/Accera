////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_all.hpp>

#include <utilities/include/Exception.h>
#include <utilities/include/MemoryLayout.h>

#include <algorithm>
#include <numeric>

namespace accera
{
using namespace utilities;

TEST_CASE("TestMemoryCoordinates")
{
    auto coords1 = MemoryCoordinates({ 1, 2, 3 });
    CHECK(coords1.NumDimensions() == 3);
    CHECK(coords1[0] == 1);
    CHECK(coords1[1] == 2);
    CHECK(coords1[2] == 3);

    auto coords2 = MemoryCoordinates(4, 5, 6, 7);
    CHECK(coords2.NumDimensions() == 4);
    CHECK(coords2[0] == 4);
    CHECK(coords2[1] == 5);
    CHECK(coords2[2] == 6);
    CHECK(coords2[3] == 7);
}

TEST_CASE("TestDimensionOrder")
{
    auto order1 = DimensionOrder({ 0, 1, 2 });
    CHECK(order1.NumDimensions() == 3);
    CHECK(order1[0] == 0);
    CHECK(order1[1] == 1);
    CHECK(order1[2] == 2);
    CHECK(order1.IsCanonicalOrder());

    auto order2 = DimensionOrder({ 1, 2, 0 });
    CHECK(!order2.IsCanonicalOrder());

    MemoryShape referenceShape(2, 3, 4);
    auto shape1 = order1.Permute(referenceShape);
    CHECK(shape1 == referenceShape);

    auto shape2 = order2.Permute(referenceShape);
    CHECK(shape2 == MemoryShape(3, 4, 2));
}

TEST_CASE("TestMemoryLayoutCtors")
{
    auto rows = GENERATE(range(1u, 3u));
    auto columns = GENERATE(range(1u, 3u));
    auto rowExtentExtra = GENERATE(range(0u, 2u));
    auto columnExtentExtra = GENERATE(range(0u, 2u));
    auto rowOffset = rowExtentExtra != 0 ? GENERATE_COPY(range(0u, rowExtentExtra)) : 0u;
    auto columnOffset = columnExtentExtra != 0 ? GENERATE_COPY(range(0u, columnExtentExtra)) : 0u;
    auto rowExtent = rows + rowExtentExtra;
    auto columnExtent = columns + columnExtentExtra;

    SECTION("Shape Ctor")
    {
        MemoryLayout layout{ MemoryShape{ rows, columns } };
        CHECK(layout.NumDimensions() == 2);
        CHECK(layout.NumElements() == (rows * columns));
        CHECK(layout.GetActiveSize() == MemoryShape({ rows, columns }));
        CHECK(layout.GetExtent() == MemoryShape({ rows, columns }));
        CHECK(layout.GetOffset() == MemoryShape({ 0, 0 }));
        CHECK(layout.GetIncrement() == MemoryShape({ columns, 1 }));
    }

    SECTION("Shape Extent Offset Ctor")
    {
        MemoryLayout layout({ rows, columns }, { rowExtent, columnExtent }, { rowOffset, columnOffset });

        CHECK(layout.NumDimensions() == 2);
        CHECK(layout.NumElements() == (rows * columns));
        CHECK(layout.GetActiveSize() == MemoryShape({ rows, columns }));
        CHECK(layout.GetExtent() == MemoryShape({ rowExtent, columnExtent }));
        CHECK(layout.GetOffset() == MemoryShape({ rowOffset, columnOffset }));
        CHECK(layout.GetIncrement() == MemoryShape({ columnExtent, 1 }));
    }
}

TEST_CASE("TestMemoryLayoutSlice")
{
    constexpr int64_t rows = 3, columns = 5, channels = 7, outerExtent = 4;
    auto physicalSize = GENERATE_COPY(chunk(GENERATE(range(1, 4)), values({ rows, columns, channels, outerExtent })));

    std::vector<int64_t> order(physicalSize.size());
    std::iota(order.begin(), order.end(), 0);
    MemoryLayout layout(physicalSize, DimensionOrder(order));
    CHECK(order.size() == (size_t)layout.NumDimensions());

    SECTION("TestLayout")
    {
        auto numDimensions = layout.NumDimensions();
        decltype(numDimensions) zero{};
        auto sliceDimension = GENERATE_COPY(range(zero, numDimensions));
        auto sliced = layout.GetSliceLayout(sliceDimension);

        CHECK(sliced.NumDimensions() == (layout.NumDimensions() - 1));
        CHECKED_IF(sliceDimension == 0)
        {
            CHECK(sliced.NumElements() == (layout.NumElements() / layout.GetExtent(0)));
        }
        CHECKED_ELSE(sliceDimension == 0)
        {
            CHECK(sliced.NumElements() == (layout.NumElements() / layout.GetExtent(0)));
        }

        auto slicedNumDimensions = sliced.NumDimensions();
        CHECKED_ELSE(slicedNumDimensions == 0)
        {
            auto dimension = GENERATE_COPY(range(zero, slicedNumDimensions));
            CHECKED_IF(dimension < sliceDimension)
            {
                CHECK(sliced.GetActiveSize(dimension) == layout.GetActiveSize(dimension));
                CHECK(sliced.GetIncrement(dimension) == layout.GetIncrement(dimension));
                CHECKED_IF(dimension == (sliceDimension - 1))
                {
                    CHECK(sliced.GetExtent(dimension) == layout.GetExtent(dimension) * layout.GetExtent(dimension + 1));
                }
                CHECKED_ELSE(dimension == (sliceDimension - 1))
                {
                    CHECK(sliced.GetExtent(dimension) == layout.GetExtent(dimension));
                }
            }
            CHECKED_ELSE(dimension < sliceDimension)
            {
                CHECK(sliced.GetActiveSize(dimension) == layout.GetActiveSize(dimension + 1));
                CHECK(sliced.GetIncrement(dimension) == layout.GetIncrement(dimension + 1));
                CHECK(sliced.GetExtent(dimension) == layout.GetExtent(dimension + 1));
            }
        }
    }
}

TEST_CASE("TestMergeDimensions")
{
    SECTION("CanonicalOrder")
    {
        MemoryLayout layout(MemoryShape{ 2, 3, 4 });

        MemoryLayout mergedLayout1 = layout.GetMergedDimensionsLayout(0, 1);
        CHECK(mergedLayout1.NumDimensions() == layout.NumDimensions() - 1);
        CHECK(mergedLayout1.GetActiveSize() == MemoryShape{ 6, 4 });
        CHECK(mergedLayout1.GetExtent() == MemoryShape{ 6, 4 });
        CHECK(mergedLayout1.GetOffset() == MemoryShape{ 0, 0 });
        CHECK(mergedLayout1.GetIncrement() == MemoryShape{ 4, 1 });
        CHECK(mergedLayout1.GetDimensionOrder() == DimensionOrder{ 0, 1 });

        MemoryLayout mergedLayout2 = layout.GetMergedDimensionsLayout(1, 2);
        CHECK(mergedLayout2.NumDimensions() == layout.NumDimensions() - 1);
        CHECK(mergedLayout2.GetActiveSize() == MemoryShape{ 2, 12 });
        CHECK(mergedLayout2.GetExtent() == MemoryShape{ 2, 12 });
        CHECK(mergedLayout2.GetOffset() == MemoryShape{ 0, 0 });
        CHECK(mergedLayout2.GetIncrement() == MemoryShape{ 12, 1 });
        CHECK(mergedLayout2.GetDimensionOrder() == DimensionOrder{ 0, 1 });

        MemoryLayout mergedLayout3;
        CHECK_THROWS(mergedLayout3 = layout.GetMergedDimensionsLayout(2, 0));
    }

    SECTION("ReverseOrder")
    {
        MemoryLayout layout(MemoryShape{ 2, 3, 4 }, DimensionOrder{ 2, 1, 0 });

        MemoryLayout mergedLayout1 = layout.GetMergedDimensionsLayout(0, 1);
        CHECK(mergedLayout1.NumDimensions() == layout.NumDimensions() - 1);
        CHECK(mergedLayout1.GetActiveSize() == MemoryShape{ 6, 4 });
        CHECK(mergedLayout1.GetExtent() == MemoryShape{ 6, 4 });
        CHECK(mergedLayout1.GetOffset() == MemoryShape{ 0, 0 });
        CHECK(mergedLayout1.GetIncrement() == MemoryShape{ 1, 6 });
        CHECK(mergedLayout1.GetDimensionOrder() == DimensionOrder{ 1, 0 });

        MemoryLayout mergedLayout2 = layout.GetMergedDimensionsLayout(1, 2);
        CHECK(mergedLayout2.NumDimensions() == layout.NumDimensions() - 1);
        CHECK(mergedLayout2.GetActiveSize() == MemoryShape{ 2, 12 });
        CHECK(mergedLayout2.GetExtent() == MemoryShape{ 2, 12 });
        CHECK(mergedLayout2.GetOffset() == MemoryShape{ 0, 0 });
        CHECK(mergedLayout2.GetIncrement() == MemoryShape{ 1, 2 });
        CHECK(mergedLayout2.GetDimensionOrder() == DimensionOrder{ 1, 0 });

        MemoryLayout mergedLayout3;
        CHECK_THROWS(mergedLayout3 = layout.GetMergedDimensionsLayout(2, 0));
    }
}

TEST_CASE("TestSplitDimensions")
{
    SECTION("CanonicalOrder")
    {
        MemoryLayout layout(MemoryShape{ 4, 6 });
        CHECK(layout.GetIncrement() == MemoryShape{ 6, 1 });

        MemoryLayout splitLayout1 = layout.GetSplitDimensionLayout(0, 2);
        CHECK(splitLayout1.NumDimensions() == layout.NumDimensions() + 1);
        CHECK(splitLayout1.GetActiveSize() == MemoryShape{ 2, 2, 6 });
        CHECK(splitLayout1.GetExtent() == MemoryShape{ 2, 2, 6 });
        CHECK(splitLayout1.GetOffset() == MemoryShape{ 0, 0, 0 });
        CHECK(splitLayout1.GetIncrement() == MemoryShape{ 12, 6, 1 });
        CHECK(splitLayout1.GetDimensionOrder() == DimensionOrder{ 0, 1, 2 });

        MemoryLayout splitLayout2 = layout.GetSplitDimensionLayout(1, 2);
        CHECK(splitLayout2.NumDimensions() == layout.NumDimensions() + 1);
        CHECK(splitLayout2.GetActiveSize() == MemoryShape{ 4, 3, 2 });
        CHECK(splitLayout2.GetExtent() == MemoryShape{ 4, 3, 2 });
        CHECK(splitLayout2.GetOffset() == MemoryShape{ 0, 0, 0 });
        CHECK(splitLayout2.GetIncrement() == MemoryShape{ 6, 2, 1 });
        CHECK(splitLayout2.GetDimensionOrder() == DimensionOrder{ 0, 1, 2 });
    }

    SECTION("ReverseOrder")
    {
        MemoryLayout layout(MemoryShape{ 4, 6 }, DimensionOrder{ 1, 0 });
        CHECK(layout.GetIncrement() == MemoryShape{ 1, 4 });

        MemoryLayout splitLayout1 = layout.GetSplitDimensionLayout(0, 2);
        CHECK(splitLayout1.NumDimensions() == layout.NumDimensions() + 1);
        CHECK(splitLayout1.GetActiveSize() == MemoryShape{ 2, 2, 6 });
        CHECK(splitLayout1.GetExtent() == MemoryShape{ 2, 2, 6 });
        CHECK(splitLayout1.GetOffset() == MemoryShape{ 0, 0, 0 });
        CHECK(splitLayout1.GetIncrement() == MemoryShape{ 2, 1, 4 });
        CHECK(splitLayout1.GetDimensionOrder() == DimensionOrder{ 1, 2, 0 });

        MemoryLayout splitLayout2 = layout.GetSplitDimensionLayout(1, 2);
        CHECK(splitLayout2.NumDimensions() == layout.NumDimensions() + 1);
        CHECK(splitLayout2.GetActiveSize() == MemoryShape{ 4, 3, 2 });
        CHECK(splitLayout2.GetExtent() == MemoryShape{ 4, 3, 2 });
        CHECK(splitLayout2.GetOffset() == MemoryShape{ 0, 0, 0 });
        CHECK(splitLayout2.GetIncrement() == MemoryShape{ 1, 8, 4 });
        CHECK(splitLayout2.GetDimensionOrder() == DimensionOrder{ 2, 0, 1 });
    }
}

TEST_CASE("TestLayoutDimensionOrder")
{
    MemoryLayout layout({ 7, 5, 3 }, ChannelMajorTensorOrder);

    SECTION("MemoryLayout::GetPhysicalDimension")
    {
        CHECK(layout.GetPhysicalDimension(2) == 0);
        CHECK(layout.GetPhysicalDimension(0) == 1);
        CHECK(layout.GetPhysicalDimension(1) == 2);
    }

    SECTION("MemoryLayout::GetLogicalDimension")
    {
        CHECK(layout.GetLogicalDimension(0) == 2);
        CHECK(layout.GetLogicalDimension(1) == 0);
        CHECK(layout.GetLogicalDimension(2) == 1);
    }
}

TEST_CASE("TestMemoryLayoutHash")
{
    MemoryLayout layout1({ 7, 5, 3 }, ChannelMajorTensorOrder);
    MemoryLayout layout2({ 7, 5, 3 }, ChannelMajorTensorOrder);
    MemoryLayout layout3({ 7, 5, 3 }, RowMajorTensorOrder);
    // MemoryLayout layout4({ 7, 5, 3 }, { 1, 1, 0 }, ChannelMajorTensorOrder);

    std::hash<MemoryLayout> layoutHash;

    SECTION("std::hash<MemoryLayout>")
    {
        CHECK(layoutHash(layout1) == layoutHash(layout2));
        CHECK_FALSE(layoutHash(layout1) == layoutHash(layout3));
        //        CHECK_FALSE(layoutHash(layout1) == layoutHash(layout4));
    }

    std::hash<DimensionVector> dimHash;
    SECTION("std::hash<DimensionVector> DimensionOrder")
    {
        DimensionOrder order1({ 0, 1, 2, 3 });
        DimensionOrder order2({ 0, 1, 2 });
        CHECK(dimHash(order1) == dimHash(order1));
        CHECK_FALSE(dimHash(order1) == dimHash(order2));
    }

    SECTION("std::hash<DimensionVector> MemoryCoordinates")
    {
        MemoryCoordinates coords1({ 1, 2, 3 });
        MemoryCoordinates coords2({ 1, 2 });
        CHECK(dimHash(coords1) == dimHash(coords1));
        CHECK_FALSE(dimHash(coords1) == dimHash(coords2));
    }

    SECTION("std::hash<DimensionVector> MemoryShape")
    {
        MemoryShape shape1({ 1, 2, 3 });
        MemoryShape shape2({ 1, 2 });
        CHECK(dimHash(shape1) == dimHash(shape1));
        CHECK_FALSE(dimHash(shape1) == dimHash(shape2));
    }
}

TEST_CASE("TestScalarLayout")
{
    CHECK(ScalarLayout.GetMemorySize() == 1u);
    CHECK(ScalarLayout.HasPadding() == false);
    CHECK(ScalarLayout.IsCanonicalOrder() == true);
    CHECK(ScalarLayout.IsContiguous() == true);
    CHECK(ScalarLayout.NumDimensions() == 0);
    CHECK(ScalarLayout.NumElements() == 1u);
}

TEST_CASE("TestInflateMemoryLayout")
{
    // Test with a 3-dimensional input layout
    MemoryLayout layout({ 7, 5, 3 }, ChannelMajorTensorOrder);
    MemoryLayout layout2 = layout.CopyWithExtraDimensions(0); // should be the same
    MemoryLayout layout3 = layout.CopyWithExtraDimensions(2); // should be the same

    SECTION("MemoryLayout::CopyWithExtraDimensions(0)")
    {
        CHECK(layout.GetActiveSize() == layout2.GetActiveSize());
        CHECK(layout.GetExtent() == layout2.GetExtent());
        CHECK(layout.GetOffset() == layout2.GetOffset());
        CHECK(layout.GetIncrement() == layout2.GetIncrement());
        CHECK(layout.GetDimensionOrder() == layout2.GetDimensionOrder());
    }

    SECTION("MemoryLayout::CopyWithExtraDimensions(2)")
    {
        CHECK_FALSE(layout.GetActiveSize() == layout3.GetActiveSize());
        CHECK_FALSE(layout.GetExtent() == layout3.GetExtent());
        CHECK_FALSE(layout.GetOffset() == layout3.GetOffset());
        CHECK_FALSE(layout.GetIncrement() == layout3.GetIncrement());
        CHECK_FALSE(layout.GetDimensionOrder() == layout3.GetDimensionOrder());
        CHECK(layout.NumElements() == layout3.NumElements());
        CHECK(layout.GetMemorySize() == layout3.GetMemorySize());
        CHECK(layout.NumDimensions() == layout3.NumDimensions() - 2);
        CHECK(layout.GetEntryOffset({ 1, 2, 3 }) == layout3.GetEntryOffset({ 0, 0, 1, 2, 3 }));
        CHECK(layout.GetEntryOffset({ 3, 2, 1 }) == layout3.GetEntryOffset({ 0, 0, 3, 2, 1 }));
    }
}

TEST_CASE("TestInflateNullMemoryLayout")
{
    // Test with a 3-dimensional input layout
    MemoryLayout layout{};
    MemoryLayout layout2 = layout.CopyWithExtraDimensions(0); // should be the same ({})
    MemoryLayout layout3 = layout.CopyWithExtraDimensions(2); // should not be the same ({1, 1})

    SECTION("Null MemoryLayout::CopyWithExtraDimensions(0)")
    {
        CHECK(layout.GetActiveSize() == layout2.GetActiveSize());
        CHECK(layout.GetExtent() == layout2.GetExtent());
        CHECK(layout.GetOffset() == layout2.GetOffset());
        CHECK(layout.GetIncrement() == layout2.GetIncrement());
        CHECK(layout.GetDimensionOrder() == layout2.GetDimensionOrder());
    }

    SECTION("Null MemoryLayout::CopyWithExtraDimensions(2)")
    {
        CHECK_FALSE(layout.GetActiveSize() == layout3.GetActiveSize());
        CHECK_FALSE(layout.GetExtent() == layout3.GetExtent());
        CHECK_FALSE(layout.GetOffset() == layout3.GetOffset());
        CHECK_FALSE(layout.GetIncrement() == layout3.GetIncrement());
        CHECK_FALSE(layout.GetDimensionOrder() == layout3.GetDimensionOrder());
        CHECK(layout.NumElements() == layout3.NumElements());
        CHECK(layout.GetMemorySize() == layout3.GetMemorySize());
        CHECK(layout.NumDimensions() == layout3.NumDimensions() - 2);
        CHECK(layout3.GetActiveSize() == MemoryShape{ 1, 1 });
        CHECK(layout3.GetExtent() == MemoryShape{ 1, 1 });
        CHECK(layout3.GetOffset() == MemoryShape{ 0, 0 });
        CHECK(layout3.GetIncrement() == MemoryShape{ 1, 1 });
        CHECK(layout3.GetDimensionOrder() == DimensionOrder{ 0, 1 });
    }
}
} // namespace accera
