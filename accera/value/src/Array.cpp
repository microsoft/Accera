////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Array.h"
#include "EmitterContext.h"

#include <utilities/include/Exception.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>

#include <iostream>

namespace accera
{
using namespace utilities;

namespace value
{
    namespace
    {
        MemoryLayout GetSliceLayout(const MemoryLayout& originalLayout, std::vector<int64_t> slicedDimensions)
        {
            std::sort(slicedDimensions.begin(), slicedDimensions.end(), std::greater<int64_t>());

            MemoryLayout result = originalLayout;
            for (auto dim : slicedDimensions)
            {
                result = result.GetSliceLayout(dim);
            }
            return result;
        }

    } // namespace

    Array::Array() = default;

    Array::Array(Value value, const std::string& name) :
        _value(value)
    {
        if (!_value.IsDefined() || !_value.IsConstrained())
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must be defined and have a memory layout");
        }
        if (_value.GetLayout() == ScalarLayout)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must not be scalar");
        }
        if (!name.empty())
        {
            SetName(name);
        }
    }

    Array::~Array() = default;
    Array::Array(const Array&) = default;
    Array::Array(Array&&) noexcept = default;

    Array& Array::operator=(const Array& other)
    {
        if (this != &other)
        {
            _value = other._value;
        }
        return *this;
    }

    Array& Array::operator=(Array&& other)
    {
        if (this != &other)
        {
            _value = std::move(other._value);
            other._value = Value();
        }
        return *this;
    }

    Value Array::GetValue() const { return _value; }

    Array Array::Copy() const
    {
        auto newValue = Allocate(_value.GetBaseType(), _value.GetLayout());
        newValue = _value;
        return newValue;
    }

    Scalar Array::operator()(const std::vector<Scalar>& indices)
    {
        if (static_cast<int64_t>(indices.size()) != GetValue().GetLayout().NumDimensions())
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }
        std::vector<int64_t> dims(indices.size());
        std::iota(dims.begin(), dims.end(), 0);
        Value indexedValue = GetContext().Slice(_value, dims, indices);

        return indexedValue;
    }

    Array Array::SubArray(const std::vector<Scalar>& offsets, const MemoryShape& shape, std::optional<std::vector<int64_t>> strides) const
    {
        assert(offsets.size() == (size_t) Rank() && shape.NumDimensions() == Rank());
        if (!strides)
        {
            strides = std::vector<int64_t>(Rank(), 1LL);
        }

        assert(strides->size() == static_cast<size_t>(Rank()));

        return GetContext().View(_value, offsets, shape, *strides);
    }

    Array Array::Slice(std::vector<int64_t> slicedDimensions, std::vector<Scalar> sliceOffsets) const
    {
        auto newLayout = GetSliceLayout(_value.GetLayout(), slicedDimensions);
        return GetContext().Slice(_value, slicedDimensions, sliceOffsets);
    }

    Array Array::Reorder(const DimensionOrder& order) const
    {
        return GetContext().Reorder(_value, order);
    }

// TODO: Enable when functionality is needed and semantics are fully cleared
#if 0
    Array Array::MergeDimensions(int64_t dim1, int64_t dim2) const
    {
        return GetContext().MergeDimensions(_value, dim1, dim2);
    }

    Array Array::SplitDimension(int64_t dim, int64_t size) const
    {
        return GetContext().SplitDimension(_value, dim, size);
    }

    Array Array::Reshape(const MemoryLayout& layout) const
    {
        if (GetLayout().GetMemorySize() != layout.GetMemorySize())
        {
            throw InputException(InputExceptionErrors::invalidSize, "Total memory size of a reshape op must remain constant");
        }

        return GetContext().Reshape(_value, layout);
    }
#endif // 0

    utilities::MemoryShape Array::Shape() const { return _value.GetLayout().GetActiveSize(); }

    utilities::MemoryLayout Array::GetLayout() const { return _value.GetLayout(); }

    int64_t Array::Size() const 
    { 
        return static_cast<int64_t>(_value.GetLayout().NumElements()); 
    }

    int64_t Array::Rank() const { return static_cast<int64_t>(_value.GetLayout().NumDimensions()); }

    ValueType Array::GetType() const { return _value.GetBaseType(); }

    void Array::SetName(const std::string& name) { _value.SetName(name); }

    std::string Array::GetName() const { return _value.GetName(); }

    bool Array::IsVariableSized() const
    {
        return _value.GetLayout().IsVariableSized();
    }

    void For(Array array, std::function<void(const std::vector<Scalar>&)> fn)
    {
        auto layout = array.GetValue().GetLayout();
        GetContext().For(layout, [fn = std::move(fn), &layout](std::vector<Scalar> coordinates) {
            if (layout.NumDimensions() != static_cast<int>(coordinates.size()))
            {
                throw InputException(InputExceptionErrors::invalidSize);
            }

            fn(coordinates);
        });
    }

    Array& Array::operator+=(Array m)
    {
        if (m.Shape() != Shape())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }

        if (m.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &m](const std::vector<Scalar>& indices) {
            (*this)(indices) += m(indices);
        });

        return *this;
    }

    Array& Array::operator-=(Array m)
    {
        if (m.Shape() != Shape())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }

        if (m.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &m](const std::vector<Scalar>& indices) {
            (*this)(indices) -= m(indices);
        });

        return *this;
    }

    Array& Array::operator+=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](const std::vector<Scalar>& indices) {
            (*this)(indices) += s;
        });

        return *this;
    }

    Array& Array::operator-=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](const std::vector<Scalar>& indices) {
            (*this)(indices) -= s;
        });

        return *this;
    }

    Array& Array::operator*=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](const std::vector<Scalar>& indices) {
            (*this)(indices) *= s;
        });

        return *this;
    }

    Array& Array::operator/=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](const std::vector<Scalar>& indices) {
            (*this)(indices) /= s;
        });

        return *this;
    }
} // namespace value
} // namespace accera
