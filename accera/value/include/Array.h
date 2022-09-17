////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "EmitterContext.h"

#include <utilities/include/FunctionUtils.h>
#include <utilities/include/MemoryLayout.h>

#include <functional>
#include <tuple>

namespace accera
{
namespace value
{
    /// <summary> A View type that wraps a Value instance and enforces a memory layout that represents a multidimensional array </summary>
    class Array
    {
    public:
        Array();

        /// <summary> Constructor that wraps the provided instance of Value </summary>
        /// <param name="value"> The Value instance to wrap </param>
        /// <param name="name"> An optional name for the emitted construct </param>
        Array(Value value, const std::string& name = "");

        /// <summary> Constructs an instance from a 1D std::vector reshaped into the given array shape </summary>
        /// <typeparam name="T"> Any fundamental type accepted by Value </typeparam>
        /// <param name="data"> The data represented as a std::vector, in canonical row-major layout </param>
        /// <param name="layout"> The layout of the memory </param>
        /// <param name="name"> An optional name for the emitted construct </param>
        template <typename T>
        Array(const std::vector<T>& data, std::optional<utilities::MemoryLayout> layout, const std::string& name = "");

        Array(const Array&);
        Array(Array&&) noexcept;
        Array& operator=(const Array&);
        Array& operator=(Array&&);
        ~Array();

        /// <summary> Array element access operator. </summary>
        /// <returns> The Scalar value wrapping the value that is at the specified index within the array </returns>
        Scalar operator()(const std::vector<Scalar>& indices);

        /// <summary> Array element access operator. </summary>
        /// <returns> The Scalar value wrapping the value that is at the specified index within the array </returns>
        template <typename... IndicesTypes>
        std::enable_if_t<utilities::AllSame<Scalar, utilities::RemoveCVRefT<IndicesTypes>...>, Scalar>
        operator()(IndicesTypes&&... indices);

        template <typename... IndicesTypes>
        std::enable_if_t<utilities::AllSame<int, utilities::RemoveCVRefT<IndicesTypes>...>, Scalar>
        operator()(IndicesTypes&&... indices);

        /// <summary> Gets the underlying wrapped Value instance </summary>
        Value GetValue() const;

        /// <summary> Creates a new Array instance that contains the same data as this instance </summary>
        /// <returns> A new Array instance that points to a new, distinct memory that contains the same data as this instance </returns>
        Array Copy() const;

        /// <summary> Get a subarray view of the data </summary>
        /// <param name="offsets"> The origin of the view --- the indices of the first entry in the subarray </param>
        /// <param name="shape"> The shape of the view </param>
        /// <returns> The resulting subarray block </returns>
        Array SubArray(const std::vector<Scalar>& offsets, const utilities::MemoryShape& shape, std::optional<std::vector<int64_t>> strides = {}) const;

        /// <summary> Get a reduced-rank slice of the data </summary>
        /// <param name="slicedDimensions"> The dimensions to remove from the domain. </param>
        /// <param name="sliceOffsets"> The index of the slice to keep for each of the sliced dimensions. </param>
        /// <returns> The resulting slice of the array </returns>
        Array Slice(std::vector<int64_t> slicedDimensions, std::vector<Scalar> sliceOffsets) const;

// TODO: Enable when functionality is needed and semantics are fully cleared
#if 0
        /// <summary> Get a view of the data where 2 (contiguous) dimensions are merged </summary>
        /// <param name="dim1"> One of the dimensions to merge </param>
        /// <param name="dim2"> The other dimension to merge </param>
        /// <remarks>
        /// The given dimensions must be adjacent in memory, and the smaller dimension to be merged
        /// must have the full extent of the underlying memory
        /// </remarks>
        Array MergeDimensions(int64_t dim1, int64_t dim2) const;

        /// <summary> Get a view of the data where a dimension is split into 2 dimensions </summary>
        /// <param name="dim"> The dimension to split </param>
        /// <param name="size"> The extent of the new inner dimension </param>
        /// <remarks>
        /// The new dimension will be placed immediately after the split dimension.
        /// The dimension being split must have full extent (and not be a sub-view of some other array).
        /// The extent (and size) of the dimension being split must be a multiple of the split size.
        /// </remarks>
        Array SplitDimension(int64_t dim, int64_t size) const;

        /// <summary> Get a view of the data using a completely different layout </summary>
        /// <param name="layout"> The memory layout for the new array view to use </param>
        /// <remarks> The total memory size of the original layout and the new layout must match </remarks>
        Array Reshape(const utilities::MemoryLayout& layout) const;
#endif // 0

        /// <summary> Get a view of the data using a different logical ordering of dimensions </summary>
        /// <param name="order"> The order for the new array view to use </param>
        /// <remarks> This operation doesn't alter any memory: it just returns a view of the array with a different logical ordering. </remarks>
        Array Reorder(const utilities::DimensionOrder& order) const;

        /// <summary> Returns the shape of the array </summary>
        /// <returns> The shape of the array </returns>
        utilities::MemoryShape Shape() const;

        /// <summary> Returns the memory layout of the array </summary>
        utilities::MemoryLayout GetLayout() const;

        /// <summary> Returns the number of dimensions </summary>
        int64_t Rank() const;

        /// <summary> Returns the number of active elements </summary>
        /// <returns> The size of the array </returns>
        int64_t Size() const;

        /// <summary> Retrieves the type of data stored in the wrapped Value instance </summary>
        /// <returns> The type </returns>
        ValueType GetType() const;

        void SetName(const std::string& name);
        std::string GetName() const;

        /// <summary> Returns `true` if the layout of Array is variable-sized. </summary>
        bool IsVariableSized() const;

        Array& operator+=(Scalar);
        Array& operator+=(Array);

        Array& operator-=(Scalar);
        Array& operator-=(Array);

        Array& operator*=(Scalar);

        Array& operator/=(Scalar);

    private:
        Value _value;

        template <typename T>
        static Value MakeValue(const std::vector<T>& data, const std::optional<utilities::MemoryLayout> layout, const std::string& name = "")
        {
            if (layout.has_value())
            {
                using namespace utilities;
                auto size = data.size();
                if (size != layout->GetMemorySize())
                {
                    throw InputException(InputExceptionErrors::invalidSize);
                }
            }
            return { data, layout, name };
        }
    };

    /// <summary> Creates a for loop over the array </summary>
    /// <param name="array"> The instance of Array that references the data over which to iterate </param>
    /// <param name="fn"> The function to be called for each coordinate where there is an active element </param>
    void For(Array array, std::function<void(const std::vector<Scalar>&)> fn);

    /// <summary> Constructs an allocated instance with the specified dimensions </summary>
    /// <param name="layout"> The layout of the memory </param>
    /// <param name="type"> The type of the elements </param>
    /// <param name="name"> The name of the allocated array </param>
    inline Array MakeArray(const utilities::MemoryLayout& layout, ValueType type, const std::string& name = "", AllocateFlags flags = AllocateFlags::None, const std::vector<ScalarDimension>& runtimeSizes = {})
    {
        return Array(Allocate(type, layout, flags, runtimeSizes), name);
    }

    inline Array MakeArray(const utilities::MemoryLayout& layout, ValueType type, AllocateFlags flags, const std::vector<ScalarDimension>& runtimeSizes = {})
    {
        return Array(Allocate(type, layout, flags, runtimeSizes));
    }

    /// <summary> Constructs an allocated instance with the specified dimensions </summary>
    /// <typeparam name="T"> Any fundamental type accepted by Value </typeparam>
    /// <param name="layout"> The layout of the memory </param>
    /// <param name="name"> The name of the allocated array </param>
    template <typename T>
    Array MakeArray(const utilities::MemoryLayout& layout, const std::string& name = "", AllocateFlags flags = AllocateFlags::None, const std::vector<ScalarDimension>& runtimeSizes = {})
    {
        return Array(Allocate<T>(layout, flags, runtimeSizes), name);
    }

    template <typename T>
    Array MakeArray(const utilities::MemoryLayout& layout, AllocateFlags flags, const std::vector<ScalarDimension>& runtimeSizes = {})
    {
        return Array(Allocate<T>(layout, flags, runtimeSizes));
    }
} // namespace value
} // namespace accera

#pragma region implementation

namespace accera
{
namespace value
{

    template <typename T>
    Array::Array(const std::vector<T>& data, const std::optional<utilities::MemoryLayout> layout, const std::string& name) :
        Array(MakeValue(data, layout, name))
    {}

    template <typename... IndicesTypes>
    std::enable_if_t<utilities::AllSame<Scalar, utilities::RemoveCVRefT<IndicesTypes>...>, Scalar>
    Array::operator()(IndicesTypes&&... indices)
    {
        std::initializer_list<Scalar> indexList{ std::forward<IndicesTypes&&>(indices)... };
        if (sizeof...(IndicesTypes) != GetValue().GetLayout().NumDimensions())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidSize, "wrong number of indices for accessing this array");
        }
        return this->operator()(std::vector<Scalar>(indexList));
    }

    template <typename... IndicesTypes>
    std::enable_if_t<utilities::AllSame<int, utilities::RemoveCVRefT<IndicesTypes>...>, Scalar>
    Array::operator()(IndicesTypes&&... indices)
    {
        return this->operator()(Scalar(indices)...);
    }

} // namespace value
} // namespace accera

#pragma endregion implementation
