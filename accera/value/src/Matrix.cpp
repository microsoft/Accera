////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Matrix.h"
#include "EmitterContext.h"

#include <utilities/include/Exception.h>

#include <cassert>
#include <functional>

namespace accera
{
using namespace utilities;

namespace value
{

    Matrix::Matrix() = default;

    Matrix::Matrix(Value value, const std::string& name) :
        _value(value)
    {
        if (!_value.IsDefined() || !_value.IsConstrained() || _value.GetLayout().NumDimensions() != 2)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must be two-dimensional");
        }

        if (!name.empty())
        {
            SetName(name);
        }
    }

    Matrix::~Matrix() = default;
    Matrix::Matrix(const Matrix&) = default;
    Matrix::Matrix(Matrix&&) noexcept = default;

    Matrix& Matrix::operator=(const Matrix& other)
    {
        if (this != &other)
        {
            _value = other._value;
        }
        return *this;
    }

    Matrix& Matrix::operator=(Matrix&& other)
    {
        if (this != &other)
        {
            _value = std::move(other._value);
            other._value = Value();
        }
        return *this;
    }

    Matrix::operator Array() const
    {
        return { GetValue() };
    }

    Scalar Matrix::operator()(Scalar rowIndex, Scalar columnIndex)
    {
        Value indexedValue = GetContext().Slice(_value, { 0, 1 }, { rowIndex, columnIndex });
        return indexedValue;
    }

    Value Matrix::GetValue() const { return _value; }

    Matrix Matrix::SubMatrix(Scalar row, Scalar column, int numRows, int numColumns, const int rowStride, const int colStride) const
    {
        const MemoryLayout& currentLayout = _value.GetLayout();

        if (numRows > currentLayout.GetActiveSize(0) ||
            numColumns > currentLayout.GetActiveSize(1))
        {
            throw InputException(InputExceptionErrors::indexOutOfRange);
        }

        // Need to cast so that row and numRows are the same type (and similarly with columns)
        // Value indexedValue = GetContext().View(_value, { { row, row + Cast(Scalar(numRows), row.GetType()) }, { column, column + Cast(Scalar(numColumns), column.GetType()) } });
        Value indexedValue = GetContext().View(_value, { row, column }, { numRows, numColumns }, {rowStride, colStride});
        return indexedValue;
    }

    Matrix Matrix::Copy() const
    {
        auto newValue = Allocate(_value.GetBaseType(), _value.GetLayout());
        newValue = _value;
        return newValue;
    }

    size_t Matrix::Size() const
    {
        return _value.GetLayout().NumElements();
    }

    Vector Matrix::Row(Scalar index) const
    {
        Value indexedValue = GetContext().Slice(_value, { 0 }, { index });
        return indexedValue;
    }

    Vector Matrix::Column(Scalar index) const
    {
        Value indexedValue = GetContext().Slice(_value, { 1 }, { index });
        return indexedValue;
    }

    Matrix Matrix::TransposedView() const
    {
        return GetContext().Reorder(_value, { 1, 0 });
    }

    // TODO: This should return int64_t
    size_t Matrix::Rows() const
    {
        return static_cast<size_t>(_value.GetLayout().GetActiveSize(0));
    }

    // TODO: This should return int64_t
    size_t Matrix::Columns() const
    {
        return static_cast<size_t>(_value.GetLayout().GetActiveSize(1));
    }

    Matrix::MatrixLayout Matrix::GetMatrixLayout() const
    {
        return _value.GetLayout().IsCanonicalOrder() ? MatrixLayout::rowMajor : MatrixLayout::columnMajor;
    }

    ValueType Matrix::GetType() const
    {
        return _value.GetBaseType();
    }

    void Matrix::SetName(const std::string& name)
    {
        _value.SetName(name);
    }

    std::string Matrix::GetName() const
    {
        return _value.GetName();
    }

    Matrix& Matrix::operator+=(Matrix m)
    {
        if (m.Rows() != Rows() && m.Columns() != Columns())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }

        if (m.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(m, [this, &m](Scalar row, Scalar column) {
            (*this)(row, column) += m(row, column);
        });

        return *this;
    }

    Matrix& Matrix::operator-=(Matrix m)
    {
        if (m.Rows() != Rows() && m.Columns() != Columns())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }

        if (m.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(m, [this, &m](Scalar row, Scalar column) {
            (*this)(row, column) -= m(row, column);
        });

        return *this;
    }

    Matrix& Matrix::operator+=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar row, Scalar column) {
            (*this)(row, column) += s;
        });

        return *this;
    }

    Matrix& Matrix::operator-=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar row, Scalar column) {
            (*this)(row, column) -= s;
        });

        return *this;
    }

    Matrix& Matrix::operator*=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar row, Scalar column) {
            (*this)(row, column) *= s;
        });

        return *this;
    }

    Matrix& Matrix::operator/=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar row, Scalar column) {
            (*this)(row, column) /= s;
        });

        return *this;
    }

    // Free function operator overloads
    Matrix operator+(Matrix m1, Matrix m2)
    {
        Matrix copy = m1.Copy();
        return copy += m2;
    }

    Matrix operator+(Matrix m, Scalar s)
    {
        Matrix copy = m.Copy();
        return copy += s;
    }

    Matrix operator-(Matrix m1, Matrix m2)
    {
        Matrix copy = m1.Copy();
        return copy -= m2;
    }
    Matrix operator-(Matrix m, Scalar s)
    {
        Matrix copy = m.Copy();
        return copy -= s;
    }

    Matrix operator*(Matrix m, Scalar s)
    {
        Matrix copy = m.Copy();
        return copy *= s;
    }

    Matrix operator/(Matrix m, Scalar s)
    {
        Matrix copy = m.Copy();
        return copy /= s;
    }

} // namespace value
} // namespace accera
