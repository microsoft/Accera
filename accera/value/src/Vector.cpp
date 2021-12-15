////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Vector.h"
#include "EmitterContext.h"

namespace accera
{
namespace value
{
    using namespace utilities;

    Vector::Vector() = default;

    Vector::Vector(Value value, const std::string& name) :
        _value(value)
    {
        if (!_value.IsDefined() || !_value.IsConstrained() || _value.GetLayout().NumDimensions() != 1)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must be one-dimensional");
        }
        if (!name.empty())
        {
            SetName(name);
        }
    }

    Vector::~Vector() = default;
    Vector::Vector(const Vector&) = default;
    Vector::Vector(Vector&&) noexcept = default;

    Vector& Vector::operator=(const Vector& other)
    {
        if (this != &other)
        {
            _value = other._value;
        }
        return *this;
    }

    Vector& Vector::operator=(Vector&& other)
    {
        if (this != &other)
        {
            _value = std::move(other._value);
            other._value = Value();
        }
        return *this;
    }

    Scalar Vector::operator[](Scalar index) { return (*this)(index); }

    Scalar Vector::operator()(Scalar index)
    {
        Value indexedValue = GetContext().Slice(_value, { 0 }, { index });

        return indexedValue;
    }

    Value Vector::GetValue() const { return _value; }

    Vector Vector::SubVector(Scalar offset, int size) const
    {
        Value indexedValue = GetContext().View(_value, { offset }, { size });
        return indexedValue;
    }

    Vector Vector::Copy() const
    {
        auto newValue = Allocate(_value.GetBaseType(), _value.GetLayout());
        newValue = _value;
        return newValue;
    }

    size_t Vector::Size() const
    {
        return _value.GetLayout().NumElements();
    }

    ValueType Vector::GetType() const
    {
        return _value.GetBaseType();
    }

    void Vector::SetName(const std::string& name)
    {
        _value.SetName(name);
    }

    std::string Vector::GetName() const
    {
        return _value.GetName();
    }

    Vector& Vector::operator+=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar index) {
            (*this)(index) += s;
        });

        return *this;
    }

    Vector& Vector::operator+=(Vector v)
    {
        if (v.Size() != Size())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }

        if (v.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(v, [this, &v](Scalar index) {
            (*this)(index) += v(index);
        });

        return *this;
    }

    Vector& Vector::operator-=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar index) {
            (*this)(index) -= s;
        });

        return *this;
    }

    Vector& Vector::operator-=(Vector v)
    {
        if (v.Size() != Size())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }

        if (v.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(v, [this, &v](Scalar index) {
            (*this)(index) -= v(index);
        });

        return *this;
    }

    Vector& Vector::operator*=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar index) {
            (*this)(index) *= s;
        });

        return *this;
    }

    Vector& Vector::operator/=(Scalar s)
    {
        if (s.GetType() != GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar index) {
            (*this)(index) /= s;
        });

        return *this;
    }

    // Free function operator overloads
    Vector operator+(Scalar s, Vector v)
    {
        return v + s;
    }

    Vector operator+(Vector v, Scalar s)
    {
        Vector copy = v.Copy();
        return copy += s;
    }

    Vector operator+(Vector v1, Vector v2)
    {
        Vector copy = v1.Copy();
        return copy += v2;
    }

    Vector operator-(Scalar s, Vector v)
    {
        Vector copy = v.Copy();
        For(copy, [&](Scalar index) {
            copy(index) = s - copy(index);
        });
        return copy;
    }

    Vector operator-(Vector v, Scalar s)
    {
        Vector copy = v.Copy();
        return copy -= s;
    }

    Vector operator-(Vector v1, Vector v2)
    {
        Vector copy = v1.Copy();
        return copy -= v2;
    }

    Vector operator*(Scalar s, Vector v)
    {
        return v * s;
    }

    Vector operator*(Vector v, Scalar s)
    {
        Vector copy = v.Copy();
        return copy *= s;
    }

    Vector operator*(Vector v, Vector u)
    {
        Vector copy = v.Copy();
        For(copy, [&](Scalar index) {
            copy(index) = copy(index) * u(index);
        });
        return copy;
    }

    Vector operator/(Scalar s, Vector v)
    {
        Vector copy = v.Copy();
        For(copy, [&](Scalar index) {
            copy(index) = s / copy(index);
        });
        return copy;
    }

    Vector operator/(Vector v, Scalar s)
    {
        Vector copy = v.Copy();
        return copy /= s;
    }

    Vector operator/(Vector v, Vector u)
    {
        Vector copy = v.Copy();
        For(copy, [&](Scalar index) {
            copy(index) = copy(index) / u(index);
        });
        return copy;
    }

    Vector AsVector(ViewAdapter view)
    {
        Value value = view;
        value.SetLayout(value.GetLayout().Flatten());
        return value;
    }

} // namespace value
} // namespace accera
