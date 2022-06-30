////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MemoryLayout.h"
#include "Exception.h"
#include "Hash.h"

#include <cassert>
#include <numeric>
#include <sstream>

#include <mlir/IR/BuiltinTypes.h>

#define BOUNDS_CHECK 0

namespace accera
{
namespace utilities
{
    namespace
    {
        MemoryShape ContiguousCumulativeIncrement(const MemoryShape& extent, const DimensionOrder& order)
        {
            const auto numDimensions = extent.NumDimensions();
            std::vector<int64_t> result(numDimensions);
            int64_t prevScale = 1;
            bool hasVariableSize = false;

            for (int64_t index = numDimensions - 1; index >= 0; --index)
            {
                result[order[index]] = prevScale;
                if (extent[order[index]] != mlir::ShapedType::kDynamicSize && !hasVariableSize)
                {
                    prevScale = prevScale * extent[order[index]];
                }
                else // variable size
                {
                    prevScale = mlir::ShapedType::kDynamicStrideOrOffset;
                    hasVariableSize = true; // stop the cumulative increment computation
                }
            }
            return { result };
        }

        MemoryShape ContiguousCumulativeIncrement(const MemoryShape& extent)
        {
            return ContiguousCumulativeIncrement(extent, DimensionOrder(extent.NumDimensions()));
        }

        MemoryShape StridedIncrement(const MemoryShape& extent, const MemoryShape& strides)
        {
            const auto numDimensions = extent.NumDimensions();
            std::vector<int64_t> result(numDimensions);
            int64_t prevScale = 1;
            for (int64_t index = numDimensions - 1; index >= 0; --index)
            {
                if (extent[index] == mlir::ShapedType::kDynamicSize)
                {
                    // Unsupported until there is a use case
                    throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument,
                                                            "Runtime-sizes are not supported for strided increments.");
                }

                result[index] = prevScale * strides[index];
                prevScale *= extent[index];
            }

            return { result };
        }
    } // namespace

    //
    // DimensionOrder
    //

    DimensionOrder::DimensionOrder(int64_t numDimensions) :
        DimensionVector(std::vector<int64_t>(numDimensions))
    {
        std::iota(_data.begin(), _data.end(), 0);
    }

    DimensionOrder::DimensionOrder(const std::vector<int64_t>& order) :
        DimensionVector(order)
    {
        Validate();
    }

    DimensionOrder::DimensionOrder(const std::initializer_list<int64_t>& order) :
        DimensionVector(order)
    {
        Validate();
    }

    void DimensionOrder::Validate() const
    {
        std::vector<int64_t> test(_data.size());
        std::iota(test.begin(), test.end(), 0);
        if (!std::is_permutation(_data.begin(), _data.end(), test.begin()))
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Dimension order must be a valid permutation vector.");
        }
    }

    bool DimensionOrder::IsCanonicalOrder() const
    {
        for (int64_t index = 0; index < NumDimensions(); ++index)
        {
            if (index != (*this)[index])
            {
                return false;
            }
        }
        return true;
    }

    std::string DimensionOrder::ToString() const
    {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    //
    // MemoryShape / Coordinates
    //

    int64_t MemoryShape::NumElements() const
    {
        return std::accumulate(_data.begin(), _data.end(), 1, std::multiplies<void>());
    }

    void MemoryShape::Resize(int64_t numDimensions)
    {
        if (numDimensions > static_cast<int64_t>(_data.size()))
        {
            int64_t extraDimensions = numDimensions - static_cast<int64_t>(_data.size());
            _data.insert(_data.begin(), extraDimensions, 1);
        }
        while (numDimensions < static_cast<int64_t>(_data.size()))
        {
            _data[1] *= _data[0];
            _data.erase(_data.begin());
        }
    }

    std::string MemoryShape::ToString() const
    {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    std::string MemoryCoordinates::ToString() const
    {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    //
    // MemoryLayout
    //
    MemoryLayout::MemoryLayout(const MemoryShape& size) :
        MemoryLayout(size, size, MemoryShape(std::vector<int64_t>(size.NumDimensions(), 0)))
    {}

    MemoryLayout::MemoryLayout(const MemoryShape& size, const MemoryShape& extent, const MemoryShape& offset) :
        MemoryLayout(size, extent, offset, ContiguousCumulativeIncrement(extent))
    {}

    MemoryLayout::MemoryLayout(const MemoryShape& size, const MemoryShape& extent, const MemoryShape& offset, const MemoryShape& increment) :
        _size(size),
        _extent(extent),
        _offset(offset),
        _increment(increment),
        _dimensionOrder(size.NumDimensions())
    {
        for (int index = 0; index < _size.NumDimensions(); ++index)
        {
            if (!IsVariableSized(index) && _size[index] + _offset[index] > _extent[index])
            {
                throw InputException(InputExceptionErrors::invalidArgument,
                                     "Extent must be larger or equal to the size plus offset.");
            }
        }
    }

    MemoryLayout::MemoryLayout(const MemoryLayout& originalLayout, const MemoryShape& size, const MemoryShape& strides) :
        MemoryLayout(size, originalLayout.GetExtent(), originalLayout.GetOffset(), StridedIncrement(originalLayout.GetExtent(), strides))
    {}

    // Constructors that deal with ordering
    MemoryLayout::MemoryLayout(const MemoryShape& size, const DimensionOrder& order) :
        MemoryLayout(size, size, MemoryShape(std::vector<int64_t>(size.NumDimensions(), 0)), order)
    {}

    MemoryLayout::MemoryLayout(const MemoryShape& size, const MemoryShape& extent, const MemoryShape& offset, const DimensionOrder& order) :
        MemoryLayout(size, extent, offset, ContiguousCumulativeIncrement(extent, order), order)
    {}

    MemoryLayout::MemoryLayout(const MemoryShape& size, const MemoryShape& extent, const MemoryShape& offset, const MemoryShape& increment, const DimensionOrder& order) :
        _size(size),
        _extent(extent),
        _offset(offset),
        _increment(increment),
        _dimensionOrder(order)
    {
        for (int index = 0; index < _size.NumDimensions(); ++index)
        {
            if (!IsVariableSized(index) && _size[index] + _offset[index] > _extent[index])
            {
                throw InputException(InputExceptionErrors::invalidArgument,
                                     "Extent must be larger or equal to the size plus offset.");
            }
        }
    }

    size_t MemoryLayout::NumElements() const
    {
        return static_cast<size_t>(_size.NumElements());
    }

    size_t MemoryLayout::GetMemorySize() const
    {
        if (*this == ScalarLayout)
        {
            return 1u;
        }

        if (IsVariableSized())
        {
            return mlir::ShapedType::kDynamicSize;
        }

        auto outermostDimension = GetOutermostDimension();
        return static_cast<size_t>(_extent[outermostDimension] * _increment[outermostDimension]);
    }

    bool MemoryLayout::IsOutOfBounds(const MemoryCoordinates& logicalCoordinates) const
    {
        const int numDimensions = NumDimensions();
        for (int index = 0; index < numDimensions; ++index)
        {
            if (logicalCoordinates[index] + _offset[index] < 0 ||
                logicalCoordinates[index] - _offset[index] >= _extent[index])
            {
                return true;
            }
        }
        return false;
    }

    bool MemoryLayout::IsContiguous() const
    {
        if (*this == ScalarLayout)
            return true;

        const auto outermostDim = GetOutermostDimension();
        const auto numDimensions = NumDimensions();
        for (int64_t d = 0; d < numDimensions; ++d)
        {
            if (d != outermostDim && _size[d] != _extent[d])
                return false;
        }
        return true;
    }

    bool MemoryLayout::IsCanonicalOrder() const
    {
        return _dimensionOrder.IsCanonicalOrder();
    }

    bool MemoryLayout::HasPadding() const
    {
        return _size != _extent;
    }

    bool MemoryLayout::IsVariableSized() const
    {
        return std::any_of(_size.begin(), _size.end(), [](auto s) { return s == mlir::ShapedType::kDynamicSize; });
    }

    int64_t MemoryLayout::GetFirstEntryOffset() const
    {
        return GetEntryOffset(GetOrigin());
    }

    MemoryCoordinates MemoryLayout::GetOrigin() const
    {
        const auto numDimensions = NumDimensions();
        MemoryCoordinates firstEntry(std::vector<int64_t>(numDimensions, 0));
        return firstEntry;
    }

    MemoryShape MemoryLayout::LogicalToPhysical(const MemoryShape& logicalShape) const
    {
        return detail::Permute(logicalShape, _dimensionOrder);
    }
    MemoryCoordinates MemoryLayout::LogicalToPhysical(const MemoryCoordinates& logicalCoordinates) const
    {
        return detail::Permute(logicalCoordinates, _dimensionOrder);
    }
    MemoryShape MemoryLayout::PhysicalToLogical(const MemoryShape& physicalShape) const
    {
        return detail::Permute(physicalShape, _dimensionOrder);
    }
    MemoryCoordinates MemoryLayout::PhysicalToLogical(const MemoryCoordinates& physicalCoordinates) const
    {
        return detail::Permute(physicalCoordinates, _dimensionOrder);
    }

    bool MemoryLayout::IsVariableSized(size_t index) const
    {
        return _extent[index] == mlir::ShapedType::kDynamicSize || _increment[index] == mlir::ShapedType::kDynamicStrideOrOffset;
    }

    int64_t MemoryLayout::GetActiveSize(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        ConstantSizeCheckDimensionIndex(index); // computed value, enforce
        return GetActiveSize()[index];
    }

    int64_t MemoryLayout::GetExtent(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        return GetExtent()[index];
    }

    int64_t MemoryLayout::GetOffset(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        return GetOffset()[index];
    }

    size_t MemoryLayout::GetDataOffset() const
    {
        return GetEntryOffset(GetOrigin());
    }

    int64_t MemoryLayout::GetIncrement(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        return GetIncrement()[index];
    }

    int64_t MemoryLayout::GetEntryOffset(const MemoryCoordinates& logicalCoordinates) const
    {
        const auto& offset = GetOffset();
        const auto& increment = GetIncrement();
        const auto numDimensions = NumDimensions();
        size_t result = 0;

#if BOUNDS_CHECK
        if (IsOutOfBounds(logicalCoordinates))
            throw InputException(InputExceptionErrors::indexOutOfRange);
#endif

        for (int index = 0; index < numDimensions; ++index)
        {
            result += increment[index] * (logicalCoordinates[index] + offset[index]);
        }
        return result;
    }

    MemoryCoordinates MemoryLayout::GetCoordinatesFromOffset(size_t offsetVal) const
    {
        const auto numDim = NumDimensions();
        std::vector<int64_t> result(numDim);
        auto offset = static_cast<int64_t>(offsetVal);
        for (int d = 0; d < numDim; ++d)
        {
            const auto thisExtent = static_cast<int64_t>(GetIncrement(d));
            const auto x = offset / thisExtent;
            result[d] = x - GetOffset(d);
            offset = (offset % thisExtent);
        }
        return result;
    }

    int64_t MemoryLayout::GetInnermostDimension() const
    {
        return GetLogicalDimension(NumDimensions() - 1);
    }

    int64_t MemoryLayout::GetOutermostDimension() const
    {
        return GetLogicalDimension(0);
    }

    int64_t MemoryLayout::GetPhysicalDimension(int64_t logicalDimension) const
    {
        if (logicalDimension < 0 || logicalDimension >= _dimensionOrder.NumDimensions())
        {
            throw InputException(InputExceptionErrors::indexOutOfRange);
        }

        if (auto it = std::find(_dimensionOrder.begin(), _dimensionOrder.end(), logicalDimension);
            it != _dimensionOrder.end())
        {
            return static_cast<int64_t>(std::distance(_dimensionOrder.begin(), it));
        }
        else
        {
            throw InputException(InputExceptionErrors::indexOutOfRange);
        }
    }

    int64_t MemoryLayout::GetLogicalDimension(int64_t physicalDimension) const
    {
        if (physicalDimension < 0 || physicalDimension >= _dimensionOrder.NumDimensions())
        {
            throw InputException(InputExceptionErrors::indexOutOfRange);
        }

        return _dimensionOrder[physicalDimension];
    }

    MemoryLayout MemoryLayout::ReorderedCopy(const DimensionOrder& newOrder) const
    {
        MemoryLayout result{ GetActiveSize(),
                             GetExtent(),
                             GetOffset(),
                             newOrder };
        return result;
    }

    MemoryLayout MemoryLayout::GetSliceLayout(int logicalDimension) const
    {
        if (logicalDimension >= NumDimensions())
        {
            throw InputException(InputExceptionErrors::indexOutOfRange,
                                 "Can't slice along a dimension greater than the number of dimensions");
        }

        if (NumDimensions() == 1)
        {
            return ScalarLayout;
        }

        auto physicalDimension = GetPhysicalDimension(logicalDimension);
        auto size = LogicalToPhysical(_size).ToVector();
        auto extent = LogicalToPhysical(_extent).ToVector();
        auto offset = LogicalToPhysical(_offset).ToVector();
        auto increment = LogicalToPhysical(_increment).ToVector();
        auto order = _dimensionOrder.ToVector();

        for (auto v : { &size, &extent, &offset, &increment, &order })
        {
            v->erase(v->begin() + physicalDimension);
        }

        // Recompute extent from increment
        int64_t prevIncrement = IsVariableSized() ? mlir::ShapedType::kDynamicStrideOrOffset : GetMemorySize();

        std::transform(
            increment.begin(),
            increment.end(),
            extent.begin(),
            [&](int64_t thisIncrement) {
                int64_t temp = (prevIncrement == mlir::ShapedType::kDynamicStrideOrOffset) ? mlir::ShapedType::kDynamicSize : prevIncrement / thisIncrement;
                prevIncrement = thisIncrement;
                return temp;
            });

        // TODO: verify this:
        // If the chosen logical dimension maps to a logical dimension that's anything
        // other than the innermost (logical) dimension, decrease the remaining dimensions by 1
        // There is expected to be, at the most, one dimension that ends up at 0. Either a dimension
        // 1 that gets decreased to 0, or a dimension 0 that goes to -1 and then gets clamped to 0.
        auto dimensionRemoved = _dimensionOrder[physicalDimension];
        for (auto& i : order)
        {
            if (i > dimensionRemoved)
            {
                --i;
            }
        }

        MemoryLayout result = { detail::ReversePermute(MemoryShape{ size }, order),
                                detail::ReversePermute(MemoryShape{ extent }, order),
                                detail::ReversePermute(MemoryShape{ offset }, order),
                                detail::ReversePermute(MemoryShape{ increment }, order),
                                DimensionOrder{ order } };

        return result;
    }

    MemoryLayout MemoryLayout::GetMergedDimensionsLayout(int dimension1, int dimension2) const
    {
        // Get the in-memory ordering of the dimensions, and sort them with the outer dimension first
        auto physicalDimension1 = GetPhysicalDimension(dimension1);
        auto physicalDimension2 = GetPhysicalDimension(dimension2);

        if (physicalDimension1 > physicalDimension2)
        {
            std::swap(dimension1, dimension2);
            std::swap(physicalDimension1, physicalDimension2);
        }
        ThrowIf(physicalDimension2 != physicalDimension1 + 1, InputExceptionErrors::invalidArgument, "Merged dimensions must be adjacent in memory");
        ThrowIf(_size[dimension2] != _extent[dimension2], InputExceptionErrors::invalidArgument, "Inner dimension to merge must have full extent");

        // Let's work in memory-order dimensions from now on
        auto size = LogicalToPhysical(_size).ToVector();
        auto extent = LogicalToPhysical(_extent).ToVector();
        auto offset = LogicalToPhysical(_offset).ToVector();
        auto increment = LogicalToPhysical(_increment).ToVector();
        auto order = _dimensionOrder.ToVector();

        const auto innerSize = size[physicalDimension2];
        const auto innerIncrement = increment[physicalDimension2];

        // Compute the properties of the merged dimension
        const auto mergeDimSize = size[physicalDimension1] * innerSize;
        const auto mergeDimExtent = extent[physicalDimension1] * innerSize;
        const auto mergeDimOffset = offset[physicalDimension1] * innerSize;
        const auto mergeDimIncrement = innerIncrement;

        // Now update the outer dimension entry and delete the inner dimension one
        size[physicalDimension1] = mergeDimSize;
        extent[physicalDimension1] = mergeDimExtent;
        offset[physicalDimension1] = mergeDimOffset;
        increment[physicalDimension1] = mergeDimIncrement;

        for (auto v : { &size, &extent, &offset, &increment, &order })
        {
            v->erase(v->begin() + physicalDimension2);
        }

        for (auto& i : order)
        {
            if (i > dimension2)
            {
                --i;
            }
        }

        return { detail::ReversePermute(MemoryShape{ size }, order),
                 detail::ReversePermute(MemoryShape{ extent }, order),
                 detail::ReversePermute(MemoryShape{ offset }, order),
                 detail::ReversePermute(MemoryShape{ increment }, order),
                 DimensionOrder{ order } };
    }

    MemoryLayout MemoryLayout::GetSplitDimensionLayout(int dimension, int innerSize) const
    {
        ThrowIf(_size[dimension] != _extent[dimension], InputExceptionErrors::invalidArgument, "Dimension to split must have full extent");
        ThrowIf(_size[dimension] % innerSize != 0, InputExceptionErrors::invalidArgument, "Dimension to split must be a multiple of the split size");
        ThrowIf(_offset[dimension] != 0, LogicExceptionErrors::illegalState, "Dimension to split should have an offset of 0"); // sanity check

        auto size = _size.ToVector();
        auto extent = _extent.ToVector();
        auto offset = _offset.ToVector();
        auto increment = _increment.ToVector();
        auto order = _dimensionOrder.ToVector();

        // Insert new inner dimension (with a copy of the outer dimension's values)
        for (auto v : { &size, &extent, &offset, &increment, &order })
        {
            int64_t prevVal = (*v)[dimension];
            v->insert(v->begin() + dimension, prevVal);
        }

        for (auto& i : order)
        {
            if (i > _dimensionOrder[dimension])
            {
                ++i;
            }
        }

        // Update size and extent of outer dimension
        size[dimension] /= innerSize;
        extent[dimension] /= innerSize;
        increment[dimension] *= innerSize;

        // Update increment of inner dimension
        size[dimension + 1] = innerSize;
        extent[dimension + 1] = innerSize;
        order[dimension + 1] = order[dimension] + 1;

        return { MemoryShape{ size },
                 MemoryShape{ extent },
                 MemoryShape{ offset },
                 MemoryShape{ increment },
                 DimensionOrder{ order } };
    }

    MemoryLayout MemoryLayout::CopyWithExtraDimensions(int addedDimensions) const
    {
        if (addedDimensions < 0)
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Number of dimensions to add must be non-negative.");
        }

        // Create prefixes of new layout properties
        std::vector<int64_t> size(addedDimensions, 1);
        std::vector<int64_t> extent(addedDimensions, 1);
        std::vector<int64_t> offset(addedDimensions, 0);
        std::vector<int64_t> increment(addedDimensions, static_cast<int64_t>(GetMemorySize()));
        std::vector<int64_t> order(addedDimensions);
        std::iota(order.begin(), order.end(), 0);

        // Append existing layout properties
        size.insert(size.end(), _size.begin(), _size.end());
        extent.insert(extent.end(), _extent.begin(), _extent.end());
        offset.insert(offset.end(), _offset.begin(), _offset.end());
        increment.insert(increment.end(), _increment.begin(), _increment.end());
        order.insert(order.end(), _dimensionOrder.begin(), _dimensionOrder.end());

        // Fix up order
        auto start = order.begin() + addedDimensions;
        std::transform(start, order.end(), start, [addedDimensions](auto x) {
            return x + addedDimensions;
        });

        return { MemoryShape{ size },
                 MemoryShape{ extent },
                 MemoryShape{ offset },
                 MemoryShape{ increment },
                 DimensionOrder{ order } };
    }

    void MemoryLayout::BoundsCheckDimensionIndex(size_t index) const
    {
        if (static_cast<int>(index) >= NumDimensions())
        {
            throw InputException(InputExceptionErrors::indexOutOfRange, "Dimension index out-of-bounds.");
        }
    }

    void MemoryLayout::ConstantSizeCheckDimensionIndex(size_t index) const
    {
        if (IsVariableSized(index))
        {
            throw LogicException(LogicExceptionErrors::notImplemented, "Not implemented for variable sizes.");
        }
    }

    std::string MemoryLayout::ToString() const
    {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    bool Equal(const DimensionVector& shape1, const DimensionVector& shape2)
    {
        auto size = shape1.NumDimensions();
        if (size != shape2.NumDimensions())
        {
            return false;
        }

        for (int index = 0; index < size; ++index)
        {
            if (shape1[index] != shape2[index])
            {
                return false;
            }
        }
        return true;
    }

    bool operator==(const DimensionOrder& order1, const DimensionOrder& order2)
    {
        return Equal(order1, order2);
    }

    bool operator!=(const DimensionOrder& order1, const DimensionOrder& order2)
    {
        return !Equal(order1, order2);
    }

    bool operator==(const MemoryShape& shape1, const MemoryShape& shape2)
    {
        return Equal(shape1, shape2);
    }

    bool operator!=(const MemoryShape& shape1, const MemoryShape& shape2)
    {
        return !Equal(shape1, shape2);
    }

    bool operator==(const MemoryCoordinates& shape1, const MemoryCoordinates& shape2)
    {
        return Equal(shape1, shape2);
    }

    bool operator!=(const MemoryCoordinates& shape1, const MemoryCoordinates& shape2)
    {
        return !Equal(shape1, shape2);
    }

    bool MemoryLayoutsEqual(const MemoryLayout& layout1, const MemoryLayout& layout2)
    {
        return (layout1.GetExtent() == layout2.GetExtent()) && (layout1.GetActiveSize() == layout2.GetActiveSize()) &&
               (layout1.GetOffset() == layout2.GetOffset() && layout1.GetDimensionOrder() == layout2.GetDimensionOrder());
    }

    bool operator==(const MemoryLayout& layout1, const MemoryLayout& layout2)
    {
        return MemoryLayoutsEqual(layout1, layout2);
    }

    bool operator!=(const MemoryLayout& layout1, const MemoryLayout& layout2)
    {
        return !MemoryLayoutsEqual(layout1, layout2);
    }

    bool operator==(const MemoryAffineCoefficients& coeff1, const MemoryAffineCoefficients& coeff2)
    {
        return (coeff1.coefficients == coeff2.coefficients) && (coeff1.offset == coeff2.offset);
    }

    bool operator!=(const MemoryAffineCoefficients& coeff1, const MemoryAffineCoefficients& coeff2)
    {
        return !(coeff1 == coeff2);
    }

    std::ostream& operator<<(std::ostream& out, const utilities::DimensionOrder& order)
    {
        if (order.NumDimensions() == 0)
        {
            out << "{}";
        }
        else
        {
            out << "d" << order[0];
            auto numDimensions = order.NumDimensions();
            for (int index = 1; index < numDimensions; ++index)
            {
                out << ", "
                    << "d" << order[index];
            }
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, const utilities::MemoryCoordinates& coords)
    {
        if (coords.NumDimensions() == 0)
        {
            out << "{}";
        }
        else
        {
            out << coords[0];
            auto numDimensions = coords.NumDimensions();
            for (int index = 1; index < numDimensions; ++index)
            {
                out << ", " << coords[index];
            }
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, const utilities::MemoryShape& shape)
    {
        if (shape.NumDimensions() == 0)
        {
            out << "{}";
        }
        else
        {
            out << shape[0];
            auto numDimensions = shape.NumDimensions();
            for (int index = 1; index < numDimensions; ++index)
            {
                out << " x " << shape[index];
            }
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, const utilities::MemoryLayout& layout)
    {
        if (layout == ScalarLayout)
        {
            out << "scalar layout";
        }
        else
        {
            out << "active size: " << layout.GetActiveSize();
            out << " memory size: " << layout.GetExtent();
            out << " memory strides: " << layout.GetIncrement();
            out << " dimension order: " << layout.GetDimensionOrder();
        }
        return out;
    }

    MemoryLayout MemoryLayout::Flatten() const
    {
        if (IsContiguous())
        {
            return MemoryLayout(static_cast<int64_t>(GetMemorySize()));
        }
        else
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Cannot flatten a discontiguous MemoryLayout.");
        }
    }

    /*extern*/ MemoryLayout ScalarLayout{};

} // namespace utilities
} // namespace accera

size_t std::hash<::accera::utilities::DimensionVector>::operator()(const ::accera::utilities::DimensionVector& v) const noexcept
{
    return ::accera::utilities::HashValue(v.ToVector());
}

std::size_t std::hash<::accera::utilities::MemoryLayout>::operator()(const accera::utilities::MemoryLayout& arg) const noexcept
{
    using ::accera::utilities::HashCombine;

    size_t hashVal = 0;
    HashCombine(hashVal, arg.GetActiveSize().ToVector());
    HashCombine(hashVal, arg.GetExtent().ToVector());
    HashCombine(hashVal, arg.GetOffset().ToVector());
    HashCombine(hashVal, arg.GetIncrement().ToVector());
    HashCombine(hashVal, arg.GetDimensionOrder().ToVector());
    return hashVal;
}
