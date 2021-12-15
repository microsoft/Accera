////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Tensor.h"
#include "EmitterContext.h"

#include <utilities/include/Exception.h>

#include <cassert>
#include <functional>
#include <numeric>

namespace accera
{
using namespace utilities;

namespace value
{
    Tensor::Tensor() = default;

    Tensor::Tensor(Value value, const std::string& name) :
        _value(value)
    {
        if (!_value.IsDefined() || !_value.IsConstrained() || _value.GetLayout().NumDimensions() != 3)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must be three-dimensional");
        }
        if (!name.empty())
        {
            SetName(name);
        }
    }

    Tensor::~Tensor() = default;
    Tensor::Tensor(const Tensor&) = default;
    Tensor::Tensor(Tensor&&) noexcept = default;

    Tensor& Tensor::operator=(const Tensor& other)
    {
        if (this != &other)
        {
            _value = other._value;
        }
        return *this;
    }

    Tensor& Tensor::operator=(Tensor&& other)
    {
        if (this != &other)
        {
            _value = std::move(other._value);
            other._value = Value();
        }
        return *this;
    }

    Scalar Tensor::operator()(Scalar rowIndex, Scalar columnIndex, Scalar channelIndex)
    {
        Value indexedValue = GetContext().Slice(_value, { 0, 1, 2 }, { rowIndex, columnIndex, channelIndex });

        return indexedValue;
    }

    Value Tensor::GetValue() const { return _value; }

    Tensor Tensor::SubTensor(Scalar row, Scalar column, Scalar channel, int numRows, int numColumns, int numChannels) const
    {
        const MemoryLayout& currentLayout = _value.GetLayout();

        if (numRows > currentLayout.GetActiveSize(0) ||
            numColumns > currentLayout.GetActiveSize(1) ||
            numChannels > currentLayout.GetActiveSize(2))
        {
            throw InputException(InputExceptionErrors::indexOutOfRange);
        }

        // Need to cast so that row and numRows are the same type (and similarly with columns and channels)
        Value indexedValue = GetContext().View(_value, { row, column, channel }, { numRows, numColumns, numChannels });
        return indexedValue;
    }

    Tensor Tensor::Copy() const
    {
        auto newValue = Allocate(_value.GetBaseType(), _value.GetLayout());
        newValue = _value;
        return newValue;
    }

    size_t Tensor::Size() const { return _value.GetLayout().NumElements(); }

    Matrix Tensor::Slice(Scalar row, value::Slice mode1, value::Slice mode2) const
    {
        const auto& currentLayout = _value.GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0));

        // TODO: generate 2-tensor for col, channel "offsets"
        Value indexedValue = GetContext().Slice(_value, { 0 }, { row });

        return indexedValue;
    }

    Matrix Tensor::Slice(value::Slice mode1, Scalar column, value::Slice mode2) const
    {
        const auto& currentLayout = _value.GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(1));
        Value indexedValue = GetContext().Slice(_value, { 1 }, { column });

        return indexedValue;
    }

    Matrix Tensor::Slice(value::Slice mode1, value::Slice mode2, Scalar channel) const
    {
        const auto& currentLayout = _value.GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(2));
        Value indexedValue = GetContext().Slice(_value, { 2 }, { channel });

        return indexedValue;
    }

    Vector Tensor::Slice(Scalar row, Scalar column, value::Slice /*mode*/) const
    {
        const auto& currentLayout = _value.GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(0));
        Value indexedValue = GetContext().Slice(_value, { 0, 1 }, { row, column });

        return indexedValue;
    }

    Vector Tensor::Slice(Scalar row, value::Slice /*mode*/, Scalar channel) const
    {
        const auto& currentLayout = _value.GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(1));
        Value indexedValue = GetContext().Slice(_value, { 0, 2 }, { row, channel });

        return indexedValue;
    }

    Vector Tensor::Slice(value::Slice /*mode*/, Scalar column, Scalar channel) const
    {
        const auto& currentLayout = _value.GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(1));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(1));
        Value indexedValue = GetContext().Slice(_value, { 1, 2 }, { column, channel });

        return indexedValue;
    }

    size_t Tensor::Rows() const { return static_cast<size_t>(_value.GetLayout().GetActiveSize(0)); }

    size_t Tensor::Columns() const { return static_cast<size_t>(_value.GetLayout().GetActiveSize(1)); }

    size_t Tensor::Channels() const { return static_cast<size_t>(_value.GetLayout().GetActiveSize(2)); }

    ValueType Tensor::GetType() const { return _value.GetBaseType(); }

    void Tensor::SetName(const std::string& name) { _value.SetName(name); }

    std::string Tensor::GetName() const { return _value.GetName(); }

    Tensor& Tensor::operator+=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [&, this](Scalar row, Scalar column, Scalar channel) {
            (*this)(row, column, channel) += s;
        });

        return *this;
    }

    Tensor& Tensor::operator-=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [&, this](Scalar row, Scalar column, Scalar channel) {
            (*this)(row, column, channel) -= s;
        });

        return *this;
    }

    Tensor& Tensor::operator*=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [&, this](Scalar row, Scalar column, Scalar channel) {
            (*this)(row, column, channel) *= s;
        });

        return *this;
    }

    Tensor& Tensor::operator/=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [&, this](Scalar row, Scalar column, Scalar channel) {
            (*this)(row, column, channel) /= s;
        });

        return *this;
    }

    // Free function operator overloads
    Tensor operator+(Tensor t, Scalar s)
    {
        Tensor copy = t.Copy();
        return copy += s;
    }

    Tensor operator-(Tensor t, Scalar s)
    {
        Tensor copy = t.Copy();
        return copy -= s;
    }

    Tensor operator*(Tensor t, Scalar s)
    {
        Tensor copy = t.Copy();
        return copy *= s;
    }

    Tensor operator/(Tensor t, Scalar s)
    {
        Tensor copy = t.Copy();
        return copy /= s;
    }

} // namespace value
} // namespace accera
