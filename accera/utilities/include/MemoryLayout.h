////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "FunctionUtils.h"
#include "TypeTraits.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

namespace accera
{
namespace utilities
{
    /// <summary> For performing an affine mapping from access indices to memory locations. </summary>
    struct MemoryAffineCoefficients
    {
        std::vector<int64_t> coefficients;
        int64_t offset;
    };

    /// <summary> An abstract base class for DimensionOrder,  MemoryShape, and MemoryCoordinates. </summary>
    class DimensionVector
    {
    public:
        /// <summary> Convert to a std::vector of integers. </summary>
        ///
        /// <returns> The elements as a std::vector. </returns>
        const std::vector<int64_t>& ToVector() const { return _data; }

        /// <summary> Element access operator. </summary>
        int64_t operator[](int64_t index) const { return _data[index]; }

        /// <summary> Element access operator. </summary>
        int64_t& operator[](int64_t index) { return _data[index]; }

        /// <summary> Get the number of dimensions. </summary>
        auto NumDimensions() const { return static_cast<int64_t>(_data.size()); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "DimensionVector"; }

        /// <summary> std::begin customization point </summary>
        auto begin() { return _data.begin(); }
        auto begin() const { return _data.cbegin(); }

        /// <summary> std::end customization point </summary>
        auto end() { return _data.end(); }
        auto end() const { return _data.cend(); }

        using iterator = std::vector<int64_t>::iterator;
        using const_iterator = std::vector<int64_t>::const_iterator;

    protected:
        DimensionVector() = default;
        DimensionVector(const std::vector<int64_t>& elements) :
            _data(elements.begin(), elements.end()) {}
        DimensionVector(const std::initializer_list<int64_t>& elements) :
            _data(elements.begin(), elements.end()) {}

        template <typename... IndexType, std::enable_if_t<utilities::AllSame<int, utilities::RemoveCVRefT<IndexType>...>, int> = 0>
        DimensionVector(IndexType... indices) :
            DimensionVector(static_cast<int64_t>(indices)...)
        {}

        template <typename... IndexType, std::enable_if_t<utilities::AllSame<int64_t, utilities::RemoveCVRefT<IndexType>...>, int> = 0>
        DimensionVector(IndexType... indices)
        {
            _data.reserve(sizeof...(IndexType));
            auto addIndex = [&](auto index) {
                _data.push_back(index);
            };
            ApplyToEach(addIndex, indices...);
        }

        std::vector<int64_t> _data;
    };

    class MemoryShape;

    /// <summary> A vector of dimension indices representing the ordering of the logical dimensions (e.g., 'row', 'column') in memory. </summary>
    class DimensionOrder : public DimensionVector
    {
    public:
        DimensionOrder() = default;

        /// <summary> Constructor for the canonical order with a given number of dimensions. </summary>
        ///
        /// <param name="numDimensions"> The number of dimensions. </param>
        DimensionOrder(int64_t numDimensions);

        /// <summary> Constructor from a vector of integers. </summary>
        ///
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major). </param>
        DimensionOrder(const std::vector<int64_t>& order);

        /// <summary> Constructor from a list of integers </summary>
        ///
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major. </param>
        DimensionOrder(const std::initializer_list<int64_t>& order);

        /// <summary> Constructor from an array of integers. </summary>
        ///
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major). </param>
        template <size_t N>
        DimensionOrder(const std::array<int64_t, N>& order) :
            DimensionOrder({ order.begin(), order.end() })
        {
            Validate();
        }

        /// <summary> Constructor from a parameter pack of int64_ts </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        template <typename... IndexType, std::enable_if_t<utilities::AllSame<int64_t, utilities::RemoveCVRefT<IndexType>...>, int> = 0>
        DimensionOrder(IndexType... coordinates) :
            DimensionVector(coordinates...)
        {
            Validate();
        }

        /// <summary> Indicates if this object represents the canonical memory order (0, 1, 2, ...) </summary>
        bool IsCanonicalOrder() const;

        /// <summary> Permutes a shape from canonical order into this order </summary>
        template <typename ShapeType, std::enable_if_t<std::is_base_of_v<DimensionVector, ShapeType>, int> = 0>
        ShapeType Permute(const ShapeType& shape) const;

        /// <summary> Reverse-permutes a shape from this order into this canonical order </summary>
        template <typename ShapeType, std::enable_if_t<std::is_base_of_v<DimensionVector, ShapeType>, int> = 0>
        ShapeType ReversePermute(const ShapeType& shape) const;

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "DimensionOrder"; }

        std::string ToString() const;

    private:
        void Validate() const;
    };

    /// <summary> A vector of numbers representing shape (extent) information of a multidimensional array. </summary>
    class MemoryShape : public DimensionVector
    {
    public:
        MemoryShape() = default;

        /// <summary> Constructor from a vector of integers </summary>
        ///
        /// <param name="shape"> The size per dimension of the shape </param>
        MemoryShape(const std::vector<int64_t>& shape) :
            DimensionVector(shape) {}

        /// <summary> Constructor from a list of integers </summary>
        ///
        /// <param name="shape"> The size per dimension of the shape </param>
        MemoryShape(const std::initializer_list<int64_t>& shape) :
            DimensionVector(shape) {}

        /// <summary> Constructor from a parameter pack of ints </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        template <typename... IndexType, std::enable_if_t<utilities::AllSame<int, utilities::RemoveCVRefT<IndexType>...>, int> = 0>
        MemoryShape(IndexType... coordinates) :
            DimensionVector(coordinates...)
        {}

        /// <summary> Constructor from a parameter pack of int64_ts </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        template <typename... IndexType, std::enable_if_t<utilities::AllSame<int64_t, utilities::RemoveCVRefT<IndexType>...>, int> = 0>
        MemoryShape(IndexType... coordinates) :
            DimensionVector(coordinates...)
        {}

        /// <summary> Get the total number of elements. </summary>
        int64_t NumElements() const;

        /// <summary>
        /// Resize to a different number of dimensions.
        /// If the new dimensionality is greater than the existing dimensionality, '1' will be appended to the front.
        /// For instance, resizing the shape (3, 4) to have 4 dimensions will result in the shape (1, 1, 3, 4).
        /// If the new dimensionality is less than the existing dimensionality, the leading dimensions will be squashed
        /// together. For instance, resizing the shape (1, 2, 3, 4) to 2 dimensions will result in the shape (6, 4)
        /// </summary>
        void Resize(int64_t numDimensions);

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "MemoryShape"; }

        std::string ToString() const;
    };

    /// <summary> A vector of numbers representing an index into a multidimensional array. </summary>
    class MemoryCoordinates : public DimensionVector
    {
    public:
        MemoryCoordinates() = default;

        /// <summary> Constructor from a vector of integers </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        MemoryCoordinates(const std::vector<int64_t>& coordinates) :
            DimensionVector(coordinates) {}

        /// <summary> Constructor from a list of integers </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        MemoryCoordinates(const std::initializer_list<int64_t>& coordinates) :
            DimensionVector(coordinates) {}

        /// <summary> Constructor from a parameter pack of ints </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        template <typename... IndexType, std::enable_if_t<utilities::AllSame<int, utilities::RemoveCVRefT<IndexType>...>, int> = 0>
        MemoryCoordinates(IndexType... coordinates) :
            DimensionVector(coordinates...)
        {}

        /// <summary> Constructor from a parameter pack of int64_ts </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        template <typename... IndexType, std::enable_if_t<utilities::AllSame<int64_t, utilities::RemoveCVRefT<IndexType>...>, int> = 0>
        MemoryCoordinates(IndexType... coordinates) :
            DimensionVector(coordinates...)
        {}

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "MemoryCoordinates"; }

        std::string ToString() const;
    };

    // TODO: dedupe with the definition in ir/include/value/ValueEnums.h
    enum class MemorySpace : uint64_t
    {
        None = 0,
        Global = 1,
        Shared = 3,
        Private = 5,
    };

    /// <summary> A class representing layout of a block of data in memory where the block can also
    /// contain padding such that a certain offset is required to access the "active" memory inside the
    /// padded block. </summary>
    class MemoryLayout
    {
    public:
        MemoryLayout() = default;

        //
        // Constructors using the canonical dimension order
        //

        /// <summary> Constructor from size only (no padding). </summary>
        ///
        /// <param name="size"> The shape of the active area of the memory region, using the canonical memory order
        ///   (the first element is the size of the slowest-changing dimension, and the last element is the size of the
        ///   fastest-changing dimension). </param>
        MemoryLayout(const MemoryShape& size);

        /// <summary> Constructor from size only (no padding). </summary>
        ///
        /// <param name="space"> The memory space for this layout </param>
        /// <param name="size"> The shape of the active area of the memory region, using the canonical memory order
        ///   (the first element is the size of the slowest-changing dimension, and the last element is the size of the
        ///   fastest-changing dimension). </param>
        MemoryLayout(MemorySpace space, const MemoryShape& size);

        /// <summary> Constructor from a parameter pack of ints specifying the size only (no padding). </summary>
        ///
        /// <param name="sizes"> The dimension sizes </param>
        template <typename... IndexType, std::enable_if_t<utilities::AllSame<int, utilities::RemoveCVRefT<IndexType>...>, int> = 0>
        MemoryLayout(IndexType... sizes) :
            MemoryLayout(MemoryShape(sizes...))
        {}

        /// <summary> Constructor from a parameter pack of int64_ts specifying the size only (no padding). </summary>
        ///
        /// <param name="sizes"> The dimension sizes </param>
        template <typename... IndexType, std::enable_if_t<utilities::AllSame<int64_t, utilities::RemoveCVRefT<IndexType>...>, int> = 0>
        MemoryLayout(IndexType... sizes) :
            MemoryLayout(MemoryShape(sizes...))
        {}

        /// <summary> General constructor. </summary>
        ///
        /// <param name="size"> The shape of the active area of the memory region (the first element is the size of
        ///   the slowest-changing dimension, and the last element is the size of the fastest-changing dimension).
        /// </param>
        /// <param name="extent"> The extent of the allocated memory of the memory region. </param>
        /// <param name="offset"> The offset into memory to the active area of the memory region. </param>
        MemoryLayout(const MemoryShape& size, const MemoryShape& extent, const MemoryShape& offset);

        /// <summary> Constructor for strided layout. </summary>
        ///
        /// <param name="originalLayout"> The layout from which to construct this layout. </param>
        /// <param name="size"> The shape of the active area of the memory region. </param>
        /// <param name="strides"> The strides into memory to the active area of the memory region. </param>
        MemoryLayout(const MemoryLayout& originalLayout, const MemoryShape& size, const MemoryShape& strides);

        //
        // Constructors with a user-supplied logical dimension ordering
        //

        /// <summary> Constructor from size and ordering (no padding). </summary>
        ///
        /// <param name="size"> The extent of the active area of the memory region, expressed in logical dimensions
        ///   (`MemoryLayout({M, N}, {0, 1})` creates a row-major array with M rows and N columns, and
        ///   `MemoryLayout({M, N}, {1, 0})` creates a column-major array with M rows and N columns)
        /// </param>
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for the canonical row-major
        ///     ordering of 2D arrays, and [1, 0] for column-major. </param>
        MemoryLayout(const MemoryShape& size, const DimensionOrder& order);

        /// <summary> General constructor. </summary>
        ///
        /// <param name="size"> The extent of the active area of the memory region, expressed in logical dimensions
        ///   (`MemoryLayout({M, N}, {0, 1})` creates a row-major array with M rows and N columns, and
        ///   `MemoryLayout({M, N}, {1, 0})` creates a column-major array with M rows and N columns)
        /// </param>
        /// <param name="extent"> The extent of the allocated memory of the memory region. </param>
        /// <param name="offset"> The offset into memory to the active area of the memory region. </param>
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major. </param>
        MemoryLayout(const MemoryShape& size, const MemoryShape& extent, const MemoryShape& offset, const DimensionOrder& order);

        MemoryLayout(const MemoryShape& size, const MemoryShape& extent, const MemoryShape& offset, const MemoryShape& increment);

        /// <summary> Returns the number of dimensions in this memory layout </summary>
        ///
        /// <returns> The number of dimensions </summary>
        int64_t NumDimensions() const { return _size.NumDimensions(); }

        /// <summary> Returns the number of active elements in this memory layout </summary>
        ///
        /// <returns> The number of elements </returns>
        size_t NumElements() const;

        /// <summary> Returns the number of total (active plus extra extent) elements in this memory layout </summary>
        ///
        /// <returns> The number of elements </returns>
        size_t GetMemorySize() const;

        /// <summary> Checks if a location is outside of the stored memory extent in any dimension </summary>
        ///
        /// <param name="coordinates"> The coordinates of the entry </param>
        /// <returns> `true` if the location is out of bounds </returns>
        bool IsOutOfBounds(const MemoryCoordinates& coordinates) const;

        /// <summary> Checks if the memory defined by this layout is contiguous </summary>
        bool IsContiguous() const;

        /// <summary> Checks if the memory defined by this layout is in the canonical memory order (0, 1, 2, ...) </summary>
        bool IsCanonicalOrder() const;

        /// <summary>
        /// Indicates if this layout has any extra padding
        /// </summary>
        ///
        /// <returns> Returns `true` if there is any extra padding around the active area. </returns>
        bool HasPadding() const;

        /// <summary> Gets the offset into memory for the first active entry </summary>
        ///
        /// <returns> The offset to the first active entry (from the beginning of memory) </returns>
        int64_t GetFirstEntryOffset() const;

        /// <summary> Helper method to construct a MemoryCoordinates object of the correct rank, containing the coordinates (0,0,...,0) </summary>
        MemoryCoordinates GetOrigin() const;

        //
        // Getting information about logical layout
        //

        /// <summary>
        /// Returns the size of the "active" memory area (not counting any padding), in the logical coordinates for this layout.
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the size of the memory area. </returns>
        const MemoryShape& GetActiveSize() const { return _size; }

        /// <summary>
        /// Returns the size of the "active" memory area (not counting any padding) for the given logical dimension.
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> A `MemoryShape` object containing the size of the memory area. </returns>
        int64_t GetActiveSize(size_t index) const;

        /// <summary>
        /// Returns the allocated size of the memory (including padding), in the logical coordinates for this layout.
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the allocated size in each dimension </returns>
        const MemoryShape& GetExtent() const { return _extent; }

        /// <summary>
        /// Returns the allocated size of the memory (including padding) for the given logical dimension.
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> A `MemoryShape` object containing the allocated size in each dimension </returns>
        int64_t GetExtent(size_t index) const;

        /// <summary>
        /// Returns the offsets to the "active" area of memory, in the logical coordinates for this layout.
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the offset to the active part of memory for that dimension. </returns>
        const MemoryShape& GetOffset() const { return _offset; }

        /// <summary>
        /// Returns the offsets to the "active" area of memory for the given logical dimension.
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> A `MemoryShape` object containing the offset to the active part of memory for that dimension. </returns>
        int64_t GetOffset(size_t index) const;

        /// <summary> Gets the logical coordinates that point to the element at a given offset from the beginning of memory. </summary>
        /// This is the inverse operation from GetEntryOffset.
        /// This function doesn't do any bounds-checking on the output. If the layout includes padding, the coordinates may be negative
        /// or outside the active memory bounds.
        MemoryCoordinates GetCoordinatesFromOffset(size_t index) const;

        /// <summary>
        /// Returns the cumulative increments in the logical coordinates for this layout. This is the distance in memory
        /// between two entries that are adjacent in that dimension.
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the increments for each dimension. </returns>
        const MemoryShape& GetIncrement() const { return _increment; }

        /// <summary>
        /// Returns the cumulative increment for the requested logical dimension. This is the distance in memory
        /// between two entries that are adjacent in that dimension.
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> The cumulative increment for the given dimension. </returns>
        int64_t GetIncrement(size_t index) const;

        /// <summary> Gets the offset into memory for an entry, given logical coordinates </summary>
        ///
        /// <param name="logicalCoordinates"> The logical coordinates of the entry </param>
        /// <returns> The offset to the entry (from the beginning of memory) </returns>
        int64_t GetEntryOffset(const MemoryCoordinates& logicalCoordinates) const;

        /// <summary> Gets the offset into memory for an entry </summary>
        ///
        /// <param name="logicalCoordinates"> The coordinates of the entry </param>
        /// <returns> The offset to the entry (from the beginning of memory) </returns>
        template <typename... IndexType, std::enable_if_t<utilities::AllSame<int64_t, utilities::RemoveCVRefT<IndexType>...>, int> = 0>
        int64_t GetEntryOffset(IndexType... logicalCoordinates) const;

        /// <summary> Returns the ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major. </summary>
        const DimensionOrder& GetDimensionOrder() const { return _dimensionOrder; }

        MemorySpace GetMemorySpace() const { return _memorySpace; }
        MemoryLayout SetMemorySpace(MemorySpace space) const
        {
            auto copy = *this;
            copy._memorySpace = space;
            return copy;
        }
        //
        // Converting between logical and physical dimensions
        //

        /// <summary> Returns the dimension with the smallest stride over memory. </summary>
        int64_t GetInnermostDimension() const;

        /// <summary> Returns the dimension with the largest stride over memory. </summary>
        int64_t GetOutermostDimension() const;

        /// <summary> Returns the corresponding physical dimension for the given logical dimension. </summary>
        int64_t GetPhysicalDimension(int64_t logicalDimension) const;

        /// <summary> Returns the corresponding logical dimension for the given physical dimension. </summary>
        int64_t GetLogicalDimension(int64_t physicalDimension) const;

        MemoryShape LogicalToPhysical(const MemoryShape& logicalShape) const;

        MemoryShape PhysicalToLogical(const MemoryShape& physicalShape) const;

        //
        // Creating related memory layouts
        //

        /// <summary> Creates a new MemoryLayout with the same memory layout, but with a new order for the dimensions </summary>
        ///
        /// <param name="newOrder"> The new order for the dimensions </param>
        /// <returns> A new MemoryLayout instance with the order switched </returns>
        MemoryLayout ReorderedCopy(const DimensionOrder& newOrder) const;

        /// <summary> Creates a new MemoryLayout with the same memory layout, but with the specified dimensions merged into one </summary>
        ///
        /// <param name="dimension1"> The dimension to merge </param>
        /// <param name="dimension2"> The other dimension to merge </param>
        /// <returns> A new memory layout that matches this one, except all information at the specified dimensions have been merged.
        /// The dimension order of the new memory layout has also been adjusted accordingly. </returns>
        MemoryLayout GetMergedDimensionsLayout(int dimension1, int dimension2) const;

        /// <summary> Creates a new MemoryLayout with the same memory layout, but with the specified dimension split into 2 dimensions </summary>
        ///
        /// <param name="dimension"> The dimension to split </param>
        /// <param name="innerSize"> The size of the new inner dimension </param>
        /// <returns> A new memory layout that matches this one, except all information at the specified dimensions have been merged.
        /// The dimension order of the new memory layout has also been adjusted accordingly. </returns>
        MemoryLayout GetSplitDimensionLayout(int dimension, int innerSize) const;

        /// <summary> Creates a new MemoryLayout with the same memory layout, but with the specified dimension sliced out </summary>
        ///
        /// <param name="dimension"> The dimension to slice out </param>
        /// <returns> A new memory layout that matches this one, except all information at the specified dimension has been removed.
        /// The dimension order of the new memory layout has also been adjusted accordingly. </returns>
        /// <remarks> If there's a tensor that needs to be sliced into a row-channel matrix, the
        /// MemoryLayout that represents that matrix can be expressed by doing `layout.GetSliceLayout(1)`, where
        /// `layout` is the MemoryLayout that describes the original tensor.</remarks>
        MemoryLayout GetSliceLayout(int dimension) const;

        /// <summary> Creates a new MemoryLayout by adding empty dimensions. </summary>
        ///
        /// <param name="addedDimensions"> The number of dimensions to add. </param>
        /// <returns> A new memory layout that matches this one, but with extra leading "empty" dimensions. </returns>
        /// <remarks>
        /// The new layout is created by appending dimensions of size (and extent) `1` and offset `0`.
        /// For instance, calling `CopyWithExtraDimensions(2)` on a layout of size `{2,3,4}` would result
        /// in a layout of size `{1,1,2,3,4}`.
        MemoryLayout CopyWithExtraDimensions(int addedDimensions) const;

        /// <summary> If the layout is contiguous, return a new layout that interprets this block as
        /// a simple one dimensional vector, otherwise throws an exception. </summary>
        MemoryLayout Flatten() const;

        std::string ToString() const;

    private:
        MemoryLayout(const MemoryShape& size, const MemoryShape& extent, const MemoryShape& offset, const MemoryShape& increment, const DimensionOrder& order);
        void BoundsCheckDimensionIndex(size_t index) const;
        size_t GetDataOffset() const; // offset for entry {0,0,0...}

        MemoryCoordinates LogicalToPhysical(const MemoryCoordinates& logicalCoordinates) const;
        MemoryCoordinates PhysicalToLogical(const MemoryCoordinates& physicalCoordinates) const;

        // "Logical dimension" quantities
        MemoryShape _size; // The "active" area of the memory
        MemoryShape _extent; // The allocated size along each dimension
        MemoryShape _offset; // The offset to the active area for each dimension
        MemoryShape _increment; // The distance in memory between adjacent elements for each dimension in logical coordinates

        // The memory order of the logical dimensions, encoded as a permutation of the physical order.
        // So, [0, 1, 2] means the physical and logical order are the same. An order of [2, 0, 1]
        // means that physical dimension 2 is logically first. In other words, logical element (r, c, d)
        // would map to physical element (d, r, c)
        DimensionOrder _dimensionOrder;

        MemorySpace _memorySpace = MemorySpace::None;
    };

    /// <summary> Helper value to denote a scalar (degree 0) memory layout </summary>
    extern MemoryLayout ScalarLayout;

    /// <summary> Checks if two dimension-order vectors are equal. </summary>
    ///
    /// <param name="order1"> The first order vector. </param>
    /// <param name="order2"> The other order vector. </param>
    bool operator==(const DimensionOrder& order1, const DimensionOrder& order2);
    bool operator!=(const DimensionOrder& order1, const DimensionOrder& order2);

    /// <summary> Checks if two shapes are equal. </summary>
    ///
    /// <param name="shape1"> The first shape. </param>
    /// <param name="shape2"> The other shape. </param>
    bool operator==(const MemoryShape& shape1, const MemoryShape& shape2);
    bool operator!=(const MemoryShape& shape1, const MemoryShape& shape2);

    /// <summary> Checks if two coordinates are equal. </summary>
    ///
    /// <param name="shape1"> The first shape. </param>
    /// <param name="shape2"> The other shape. </param>
    bool operator==(const MemoryCoordinates& shape1, const MemoryCoordinates& shape2);
    bool operator!=(const MemoryCoordinates& shape1, const MemoryCoordinates& shape2);

    /// <summary> Checks if two memory layouts are equal. </summary>
    ///
    /// <param name="layout1"> The first memory layout. </param>
    /// <param name="layout2"> The other memory layout. </param>
    bool MemoryLayoutsEqual(const MemoryLayout& layout1, const MemoryLayout& layout2);
    bool operator==(const MemoryLayout& shape1, const MemoryLayout& shape2);
    bool operator!=(const MemoryLayout& shape1, const MemoryLayout& shape2);

    /// <summary> Checks if two affine coefficient memory maps are equal. </summary>
    ///
    /// <param name="order1"> The first affine coefficient memory map. </param>
    /// <param name="order2"> The other affine coefficient memory map. </param>
    bool operator==(const MemoryAffineCoefficients& coeff1, const MemoryAffineCoefficients& coeff2);
    bool operator!=(const MemoryAffineCoefficients& coeff1, const MemoryAffineCoefficients& coeff2);

    /// <summary> Represents row-major matrix order </summary>
    constexpr std::array<int64_t, 2> RowMajorMatrixOrder({ 0, 1 });

    /// <summary> Represents column-major matrix order </summary>
    constexpr std::array<int64_t, 2> ColumnMajorMatrixOrder{ 1, 0 };

    /// <summary> Represents row-major 3D tensor order </summary>
    constexpr std::array<int64_t, 3> RowMajorTensorOrder{ 0, 1, 2 };

    /// <summary> Represents channel-major 3D tensor order </summary>
    constexpr std::array<int64_t, 3> ChannelMajorTensorOrder{ 2, 0, 1 };

    /// <summary> Writes a `DimensionOrder` to an output stream </summary>
    std::ostream& operator<<(std::ostream& out, const utilities::DimensionOrder& order);

    /// <summary> Writes a `MemoryCoordinates` to an output stream </summary>
    std::ostream& operator<<(std::ostream& out, const utilities::MemoryCoordinates& order);

    /// <summary> Writes a `MemoryShape`'s dimensions to an output stream </summary>
    std::ostream& operator<<(std::ostream& out, const utilities::MemoryShape& shape);

    /// <summary> Writes a `MemoryLayout` to an output stream </summary>
    std::ostream& operator<<(std::ostream& out, const utilities::MemoryLayout& layout);
} // namespace utilities
} // namespace accera

//
// Implementation
//
namespace accera
{
namespace utilities
{
    namespace detail
    {
        template <typename ShapeType, std::enable_if_t<std::is_base_of_v<DimensionVector, ShapeType>, int> = 0>
        ShapeType Permute(const ShapeType& shape, const DimensionOrder& order)
        {
            const int64_t numDimensions = shape.NumDimensions();
            std::vector<int64_t> result(numDimensions);
            for (int64_t index = 0; index < numDimensions; ++index)
            {
                result[index] = shape[order[index]];
            }
            return { result };
        }

        template <typename ShapeType, std::enable_if_t<std::is_base_of_v<DimensionVector, ShapeType>, int> = 0>
        ShapeType ReversePermute(const ShapeType& shape, const DimensionOrder& order)
        {
            const int64_t numDimensions = shape.NumDimensions();
            std::vector<int64_t> result(numDimensions);
            for (int64_t index = 0; index < numDimensions; ++index)
            {
                result[order[index]] = shape[index];
            }
            return { result };
        }

    } // namespace detail

    template <typename ShapeType, std::enable_if_t<std::is_base_of_v<DimensionVector, ShapeType>, int>>
    ShapeType DimensionOrder::Permute(const ShapeType& shape) const
    {
        return detail::Permute(shape, *this);
    }

    template <typename ShapeType, std::enable_if_t<std::is_base_of_v<DimensionVector, ShapeType>, int>>
    ShapeType DimensionOrder::ReversePermute(const ShapeType& shape) const
    {
        return detail::ReversePermute(shape, *this);
    }

    template <typename... IndexType, std::enable_if_t<utilities::AllSame<int64_t, utilities::RemoveCVRefT<IndexType>...>, int>>
    int64_t MemoryLayout::GetEntryOffset(IndexType... logicalCoordinates) const
    {
        MemoryCoordinates coords{ logicalCoordinates... };
        return GetEntryOffset(coords);
    }

} // namespace utilities
} // namespace accera
namespace std
{
template <>
struct hash<::accera::utilities::DimensionVector>
{
    using Type = ::accera::utilities::DimensionVector;

    /// <summary> Computes a hash of the input value. </summary>
    ///
    /// <returns> A hash value for the given input. </returns>
    [[nodiscard]] size_t operator()(const Type& value) const noexcept;
};

template <>
struct hash<::accera::utilities::MemoryLayout>
{
    using Type = ::accera::utilities::MemoryLayout;

    /// <summary> Computes a hash of the input value. </summary>
    ///
    /// <returns> A hash value for the given input. </returns>
    [[nodiscard]] size_t operator()(const Type& value) const noexcept;
};

} // namespace std
