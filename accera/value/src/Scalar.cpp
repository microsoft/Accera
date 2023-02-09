////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Scalar.h"
#include "Emittable.h"
#include "EmitterContext.h"
#include "ScalarOperations.h"
#include "Value.h"
#include "ValueType.h"

#include <initializer_list>
#include <utilities/include/Exception.h>

#include <llvm/ADT/STLExtras.h>

namespace accera
{
namespace value
{
    bool IsLogicalComparable(ValueType type1, ValueType type2)
    {
        if ((type1 == ValueType::Index && IsIntegerType(type2)) ||
            (type2 == ValueType::Index && IsIntegerType(type1)))
            return true;
        else
            return false;
    }

    using namespace utilities;

    Scalar::Scalar() = default;

    Scalar::Scalar(Value value, const std::string& name, Role role) :
        _role{ role }
    {
        SetValue(value);

        if (!name.empty())
        {
            SetName(name);
        }
    }

    void Scalar::SetValue(Value value)
    {
        if (value.PointerLevel() && (_role == Role::InputOutput || _role == Role::Output))
        {
            value = value.PointerTo();
        }

        _value.Reset();
        _value = std::move(value);
        if (auto& layout = _value.GetLayout();
            !_value.IsDefined() || !_value.IsConstrained() ||
            !(layout == ScalarLayout ||
              (layout.NumDimensions() == 1 && layout.GetExtent(0) == 1)))
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }
    }

    Scalar::~Scalar() = default;
    Scalar::Scalar(const Scalar&) = default;
    Scalar::Scalar(Scalar&&) noexcept = default;

    Scalar& Scalar::operator=(const Scalar& other)
    {
        if (this != &other)
        {
            auto e1 = _value.TryGet<Emittable>();
            auto e2 = other._value.TryGet<Emittable>();
            if (e1 && e2 && e1->GetDataAs<void*>() == e2->GetDataAs<void*>())
            {
                return *this;
            }
            if (GetType() != other.GetType() && IsImplicitlyCastable(other, *this))
            {
                Scalar castedScalar = Cast(other, GetType());
                GetContext().CopyData(castedScalar._value, _value);
            }
            else
            {
                _value = other._value;
            }

            _role = other.GetRole();
        }
        return *this;
    }

    Scalar& Scalar::operator=(Scalar&& other) noexcept
    {
        if (this != &other)
        {
            auto e1 = _value.TryGet<Emittable>();
            auto e2 = other._value.TryGet<Emittable>();
            if (e1 && e2 && e1->GetDataAs<void*>() == e2->GetDataAs<void*>())
            {
                return *this;
            }
            if (GetType() != other.GetType() && IsImplicitlyCastable(other, *this))
            {
                Scalar castedScalar = Cast(other, GetType());
                GetContext().MoveData(castedScalar._value, _value);
            }
            else
            {
                _value = std::move(other._value);
            }
            other._value.Clear();

            _role = other.GetRole();
        }
        return *this;
    }

    void Scalar::Set(const Scalar& other)
    {
        if (_role == Role::Input)
            throw LogicException(LogicExceptionErrors::illegalState);

        *this = other;
    }

    Value Scalar::GetValue() const
    {
        return _value;
    }

    Role Scalar::GetRole() const
    {
        return _role;
    }

    Scalar Scalar::Copy() const
    {
        auto s = MakeScalar(GetType());
        s = *this;
        return s;
    }

    ValueType Scalar::GetType() const
    {
        return _value.GetBaseType();
    }

    void Scalar::SetName(const std::string& name)
    {
        _value.SetName(name);
    }

    std::string Scalar::GetName() const
    {
        return _value.GetName();
    }

    Scalar& Scalar::operator+=(Scalar s)
    {
        Scalar rhs;
        if (s.GetType() != GetType())
        {
            if (IsImplicitlyCastable(s, *this))
            {
                rhs = Cast(s, GetType());
            }
            else
            {
                throw TypeMismatchException(GetType(), s.GetType());
            }
        }
        else
        {
            rhs = s;
        }

        _value = GetContext().BinaryOperation(ValueBinaryOperation::add, _value, rhs._value);
        return *this;
    }

    Scalar& Scalar::operator-=(Scalar s)
    {
        Scalar rhs;
        if (s.GetType() != GetType())
        {
            if (IsImplicitlyCastable(s, *this))
            {
                rhs = Cast(s, GetType());
            }
            else
            {
                throw TypeMismatchException(GetType(), s.GetType());
            }
        }
        else
        {
            rhs = s;
        }

        _value = GetContext().BinaryOperation(ValueBinaryOperation::subtract, _value, rhs._value);
        // auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(*this, s);

        // _value = GetContext().BinaryOperation(ValueBinaryOperation::subtract, lhs._value, rhs._value);
        return *this;
    }

    Scalar& Scalar::operator*=(Scalar s)
    {
        Scalar rhs;
        if (s.GetType() != GetType())
        {
            if (IsImplicitlyCastable(s, *this))
            {
                rhs = Cast(s, GetType());
            }
            else
            {
                throw TypeMismatchException(GetType(), s.GetType());
            }
        }
        else
        {
            rhs = s;
        }

        _value = GetContext().BinaryOperation(ValueBinaryOperation::multiply, _value, rhs._value);
        // auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(*this, s);

        // _value = GetContext().BinaryOperation(ValueBinaryOperation::multiply, lhs._value, rhs._value);
        return *this;
    }

    Scalar& Scalar::operator/=(Scalar s)
    {
        Scalar rhs;
        if (s.GetType() != GetType())
        {
            if (IsImplicitlyCastable(s, *this))
            {
                rhs = Cast(s, GetType());
            }
            else
            {
                throw TypeMismatchException(GetType(), s.GetType());
            }
        }
        else
        {
            rhs = s;
        }

        _value = GetContext().BinaryOperation(ValueBinaryOperation::divide, _value, rhs._value);
        // auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(*this, s);

        // _value = GetContext().BinaryOperation(ValueBinaryOperation::divide, lhs._value, rhs._value);
        return *this;
    }

    Scalar& Scalar::operator%=(Scalar s)
    {
        Scalar rhs;
        if (s.GetType() != GetType())
        {
            if (IsImplicitlyCastable(s, *this))
            {
                rhs = Cast(s, GetType());
            }
            else
            {
                throw TypeMismatchException(GetType(), s.GetType());
            }
        }
        else
        {
            rhs = s;
        }

        _value = GetContext().BinaryOperation(ValueBinaryOperation::modulus, _value, rhs._value);
        // auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(*this, s);

        // _value = GetContext().BinaryOperation(ValueBinaryOperation::modulus, lhs._value, rhs._value);
        return *this;
    }

    std::pair<Scalar, Scalar> Scalar::MakeTypeCompatible(Scalar s1, Scalar s2)
    {
        auto s1Type = s1.GetType();
        auto s2Type = s2.GetType();

        if (s1Type == s2Type)
        {
            return { s1, s2 };
        }

        assert((!s1.IsConstant() || !s2.IsConstant()) && "Unexpected scenario");

        Scalar newS1, newS2;
        if (s1.IsConstant())
        {
            newS1 = Cast(s1, s2Type);
            newS2 = s2;
        }
        else if (s2.IsConstant())
        {
            newS1 = s1;
            newS2 = Cast(s2, s1Type);
        }
        else if (IsImplicitlyCastable(s1, s2))
        {
            newS1 = Cast(s1, s2Type);
            newS2 = s2;
        }
        else if (IsImplicitlyCastable(s2, s1))
        {
            newS1 = s1;
            newS2 = Cast(s2, s1Type);
        }

        return { newS1, newS2 };
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
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(s1, s2);

        return GetContext().LogicalOperation(ValueLogicalOperation::equality, lhs.GetValue(), rhs.GetValue());
    }

    Scalar operator!=(Scalar s1, Scalar s2)
    {
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(s1, s2);

        return GetContext().LogicalOperation(ValueLogicalOperation::inequality, lhs.GetValue(), rhs.GetValue());
    }

    Scalar operator<=(Scalar s1, Scalar s2)
    {
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(s1, s2);

        return GetContext().LogicalOperation(ValueLogicalOperation::lessthanorequal, lhs.GetValue(), rhs.GetValue());
    }

    Scalar operator<(Scalar s1, Scalar s2)
    {
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(s1, s2);

        return GetContext().LogicalOperation(ValueLogicalOperation::lessthan, lhs.GetValue(), rhs.GetValue());
    }

    Scalar operator>=(Scalar s1, Scalar s2)
    {
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(s1, s2);

        return GetContext().LogicalOperation(ValueLogicalOperation::greaterthanorequal, lhs.GetValue(), rhs.GetValue());
    }

    Scalar operator>(Scalar s1, Scalar s2)
    {
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(s1, s2);

        return GetContext().LogicalOperation(ValueLogicalOperation::greaterthan, lhs.GetValue(), rhs.GetValue());
    }

    Scalar operator&&(Scalar s1, Scalar s2)
    {
        if (!s1.GetValue().IsIntegral() || !s2.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(s1, s2);

        return GetContext().BinaryOperation(ValueBinaryOperation::logicalAnd, lhs.GetValue(), rhs.GetValue());
    }

    Scalar operator||(Scalar s1, Scalar s2)
    {
        if (!s1.GetValue().IsIntegral() || !s2.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(s1, s2);

        return GetContext().BinaryOperation(ValueBinaryOperation::logicalOr, lhs.GetValue(), rhs.GetValue());
    }

    Scalar MakeScalar(ValueType type, const std::string&, Role role)
    {
        // TODO: figure out how to name these scalars
        return Scalar(Value(type, ScalarLayout), "", role);
    }

} // namespace value
} // namespace accera
