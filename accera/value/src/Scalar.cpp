////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Scalar.h"
#include "EmitterContext.h"
#include "ValueType.h"

#include <initializer_list>
#include <utilities/include/Exception.h>

#include <llvm/ADT/STLExtras.h>

namespace accera
{
namespace value
{
    namespace
    {
        template <typename T, typename C>
        constexpr bool ItemIsOneOf(T&& t, C&& c)
        {
            return llvm::any_of(c, [=](auto arg) { return t == arg; });
        }

        bool IsImplcitTypeCastable(ValueType source, ValueType target)
        {
#define MAP_TARGET_TO_POSSIBLE_SOURCES(TARGET, ...) \
    case TARGET:                                    \
        return ItemIsOneOf(source, std::initializer_list<ValueType>{ __VA_ARGS__ })

            switch (target)
            {
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Int8, ValueType::Boolean);
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Byte, ValueType::Boolean, ValueType::Int8);
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Int16, ValueType::Boolean, ValueType::Int8, ValueType::Byte, ValueType::Uint16);
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Uint16, ValueType::Boolean, ValueType::Int8, ValueType::Byte, ValueType::Int16);
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Int32, ValueType::Boolean, ValueType::Int8, ValueType::Byte, ValueType::Int16, ValueType::Uint16, ValueType::Uint32);
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Uint32, ValueType::Boolean, ValueType::Int8, ValueType::Byte, ValueType::Int16, ValueType::Uint16, ValueType::Int32);
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Int64, ValueType::Boolean, ValueType::Int8, ValueType::Byte, ValueType::Int16, ValueType::Uint16, ValueType::Int32, ValueType::Uint32, ValueType::Uint64);
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Uint64, ValueType::Boolean, ValueType::Int8, ValueType::Byte, ValueType::Int16, ValueType::Uint16, ValueType::Int32, ValueType::Uint32, ValueType::Int64);
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Float16, ValueType::Boolean, ValueType::Int8, ValueType::Byte, ValueType::Int16, ValueType::Uint16);
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Float, ValueType::Boolean, ValueType::Int8, ValueType::Byte, ValueType::Int16, ValueType::Uint16, ValueType::Int32, ValueType::Uint32, ValueType::Int64, ValueType::Uint64, ValueType::Float16);
                MAP_TARGET_TO_POSSIBLE_SOURCES(ValueType::Double, ValueType::Boolean, ValueType::Int8, ValueType::Byte, ValueType::Int16, ValueType::Uint16, ValueType::Int32, ValueType::Uint32, ValueType::Int64, ValueType::Uint64, ValueType::Float16, ValueType::Float);

            default:
                return false;
            }

#undef MAP_TARGET_TO_POSSIBLE_SOURCES
        }
    } // namespace
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
            if (GetType() != other.GetType() && IsImplcitTypeCastable(other.GetType(), GetType()))
            {
                Scalar castedScalar = Cast(other, GetType());
                _value = castedScalar._value;
            }
            else
            {
                _value = other._value;
            }
        }
        return *this;
    }

    Scalar& Scalar::operator=(Scalar&& other) noexcept
    {
        if (this != &other)
        {
            if (GetType() != other.GetType() && IsImplcitTypeCastable(other.GetType(), GetType()))
            {
                Scalar castedScalar = Cast(other, GetType());
                _value = std::move(castedScalar._value);
            }
            else
            {
                _value = std::move(other._value);
            }
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
