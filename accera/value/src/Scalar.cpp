////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Scalar.h"
#include "EmitterContext.h"

#include <utilities/include/Exception.h>

namespace accera
{
namespace value
{
    using namespace utilities;

    Scalar::Scalar() = default;

    Scalar::Scalar(Value value, const std::string& name) :
        _value(std::move(value))
    {
        if (auto& layout = _value.GetLayout();
            !_value.IsDefined() || !_value.IsConstrained() ||
            !(layout == ScalarLayout ||
              (layout.NumDimensions() == 1 && layout.GetExtent(0) == 1)))
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }
        if (!name.empty())
        {
            SetName(name);
        }
    }

    Scalar::~Scalar() = default;
    Scalar::Scalar(const Scalar&) = default;
    Scalar::Scalar(Scalar&&) noexcept = default;

    Scalar& Scalar::operator=(const Scalar& other)
    {
        if (this != &other)
        {
            _value = other._value;
        }
        return *this;
    }

    Scalar& Scalar::operator=(Scalar&& other) noexcept
    {
        if (this != &other)
        {
            _value = std::move(other._value);
            other._value = Value();
        }
        return *this;
    }

    Value Scalar::GetValue() const { return _value; }

    Scalar Scalar::Copy() const
    {
        auto s = MakeScalar(GetType());
        s = *this;
        return s;
    }

    ValueType Scalar::GetType() const { return _value.GetBaseType(); }

    void Scalar::SetName(const std::string& name) { _value.SetName(name); }

    std::string Scalar::GetName() const { return _value.GetName(); }

    Scalar& Scalar::operator+=(Scalar s)
    {
        _value = GetContext().BinaryOperation(ValueBinaryOperation::add, _value, s._value);
        return *this;
    }

    Scalar& Scalar::operator-=(Scalar s)
    {
        _value = GetContext().BinaryOperation(ValueBinaryOperation::subtract, _value, s._value);
        return *this;
    }

    Scalar& Scalar::operator*=(Scalar s)
    {
        _value = GetContext().BinaryOperation(ValueBinaryOperation::multiply, _value, s._value);
        return *this;
    }

    Scalar& Scalar::operator/=(Scalar s)
    {
        _value = GetContext().BinaryOperation(ValueBinaryOperation::divide, _value, s._value);
        return *this;
    }

    Scalar& Scalar::operator%=(Scalar s)
    {
        _value = GetContext().BinaryOperation(ValueBinaryOperation::modulus, _value, s._value);
        return *this;
    }

    // Free function operator overloads
    Scalar operator+(Scalar s1, Scalar s2)
    {
        return Add(s1, s2);
    }

    Scalar operator-(Scalar s1, Scalar s2)
    {
        return Subtract(s1, s2);
    }

    Scalar operator*(Scalar s1, Scalar s2)
    {
        return Multiply(s1, s2);
    }

    Scalar operator/(Scalar s1, Scalar s2)
    {
        return Divide(s1, s2);
    }

    Scalar operator%(Scalar s1, Scalar s2)
    {
        return Modulo(s1, s2);
    }

    Scalar operator-(Scalar s)
    {
        return Cast(0, s.GetType()) - s;
    }

    Scalar operator++(Scalar s)
    {
        if (!s.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        return s += Cast(1, s.GetType());
    }

    Scalar operator++(Scalar s, int)
    {
        if (!s.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        Scalar copy = s.Copy();
        s += Cast(1, s.GetType());
        return copy;
    }

    Scalar operator--(Scalar s)
    {
        if (!s.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        return s -= Cast(1, s.GetType());
    }

    Scalar operator--(Scalar s, int)
    {
        if (!s.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        Scalar copy = s.Copy();
        s -= Cast(1, s.GetType());
        return copy;
    }

    Scalar operator==(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::equality, s1.GetValue(), s2.GetValue());
    }

    Scalar operator!=(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::inequality, s1.GetValue(), s2.GetValue());
    }

    Scalar operator<=(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::lessthanorequal, s1.GetValue(), s2.GetValue());
    }

    Scalar operator<(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::lessthan, s1.GetValue(), s2.GetValue());
    }

    Scalar operator>=(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::greaterthanorequal, s1.GetValue(), s2.GetValue());
    }

    Scalar operator>(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::greaterthan, s1.GetValue(), s2.GetValue());
    }

    Scalar operator&&(Scalar s1, Scalar s2)
    {
        if (!s1.GetValue().IsIntegral() || !s2.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        return GetContext().BinaryOperation(ValueBinaryOperation::logicalAnd, s1.GetValue(), s2.GetValue());
    }

    Scalar operator||(Scalar s1, Scalar s2)
    {
        if (!s1.GetValue().IsIntegral() || !s2.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        return GetContext().BinaryOperation(ValueBinaryOperation::logicalOr, s1.GetValue(), s2.GetValue());
    }

    Scalar MakeScalar(ValueType type, const std::string&)
    {
        // TODO: figure out how to name these scalars
        return Scalar(Value(type, ScalarLayout));
    }

} // namespace value
} // namespace accera
