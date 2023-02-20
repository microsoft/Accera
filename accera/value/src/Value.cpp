////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Value.h"
#include "EmitterContext.h"
#include "ValueType.h"

#include <utilities/include/Hash.h>
#include <utilities/include/TypeAliases.h>
#include <utilities/include/TypeTraits.h>

namespace accera
{
namespace value
{
    using namespace detail;
    using namespace utilities;

    Value::Value() = default;

    Value::~Value() = default;

    Value::Value(const Value&) = default;

    Value::Value(Value&& other) noexcept
    {
        // GCC 8's stdlib doesn't have std::swap for std::variant
        auto temp = _data;
        _data = other._data;
        other._data = temp;
        std::swap(_type, other._type);
        std::swap(_layout, other._layout);
        std::swap(_hasName, other._hasName);
        std::swap(_name, other._name);
    }

    // clang-format off
    /****************************************************************************
    Copy assignment has to be context aware. The terms "defined", "empty", and
    "constrained" match their definitions of the respective Value member functions.
    Shallow copies of data is done in this function. Deep copies are done by the
    global EmitterContext object.
    * if lhs is not defined
      * and rhs is not defined: don't do anything
      * and rhs is defined: copy everything
    * if lhs is defined and not constrained and empty
      * and rhs is not defined: throw
      * and rhs is defined: if types don't match, throw. copy everything
    * if lhs is defined and constrained and empty
      * and rhs is not defined: throw
      * and rhs is defined and not constrained and empty: if types don't match, throw.
      * and rhs is defined and constrained and empty: if types or layout don't match, throw
      * and rhs is defined and not constrained and not empty: if types don't match, throw. shallow copy of data
      * and rhs is defined and constrained and not empty: if types or layout don't match, throw. shallow copy of data
    * if lhs is defined and not constrained and not empty:
      * throw
    * if lhs is defined and constrained and not empty:
      * and rhs does not match type or layout: throw
      * and rhs matches type and layout: deep copy of data
    ****************************************************************************/
    // clang-format on
    Value& Value::operator=(const Value& other)
    {
        if (this != &other)
        {
            if (!IsDefined())
            {
                // lhs is not defined
                if (other.IsDefined())
                {
                    _layout = other._layout;
                    _data = other._data;
                    _type = other._type;
                    _hasName = other._hasName;
                    _name = other._name;
                }
            }
            else
            {
                // lhs is defined
                if (!other.IsDefined())
                {
                    throw InputException(InputExceptionErrors::invalidArgument, "Value assignment, lhs is defined, but rhs is not");
                }
                if (GetBaseType() != other.GetBaseType())
                {
                    throw InputException(InputExceptionErrors::typeMismatch, "Value assignment type mismatch");
                }

                if (!IsConstrained())
                {
                    // lhs is not constrained
                    if (IsEmpty())
                    {
                        _layout = other._layout;
                        _data = other._data;
                        _type = other._type;
                        _hasName = other._hasName;
                        _name = other._name;
                    }
                    else
                    {
                        throw LogicException(LogicExceptionErrors::illegalState, "Value assignment, lhs was expected to be constrained");
                    }
                }
                else
                {
                    // lhs is constrained
                    if (!other.IsConstrained() || GetLayout() != other.GetLayout())
                    {
                        throw InputException(InputExceptionErrors::sizeMismatch, "Value assignment, layout mismatch");
                    }

                    if (IsEmpty())
                    {
                        _type = other._type;
                    }
                    GetContext().CopyData(other, *this);
                }
            }
        }

        return *this;
    }

    // Behaves similarly to the copy assignment, except rhs is reset afterwards
    Value& Value::operator=(Value&& other)
    {
        if (this != &other)
        {
            if (!IsDefined())
            {
                // lhs is not defined
                if (other.IsDefined())
                {
                    _data = std::move(other._data);
                    _layout = std::move(other._layout);
                    _type = std::move(other._type);
                    _hasName = std::move(other._hasName);
                    _name = std::move(other._name);
                }
            }
            else
            {
                // lhs is defined
                if (!other.IsDefined())
                {
                    throw InputException(InputExceptionErrors::invalidArgument, "Value assignment, lhs is defined, but rhs is not");
                }
                if (GetBaseType() != other.GetBaseType())
                {
                    throw InputException(InputExceptionErrors::typeMismatch, "Value assignment type mismatch");
                }

                if (!IsConstrained())
                {
                    // lhs is not constrained
                    if (IsEmpty())
                    {
                        _data = std::move(other._data);
                        _layout = std::move(other._layout);
                        _type = std::move(other._type);
                        _hasName = std::move(other._hasName);
                        _name = std::move(other._name);
                    }
                    else
                    {
                        throw LogicException(LogicExceptionErrors::illegalState, "Value assignment, lhs was expected to be constrained");
                    }
                }
                else
                {
                    // lhs is constrained
                    if (!other.IsConstrained() || GetLayout() != other.GetLayout())
                    {
                        throw InputException(InputExceptionErrors::sizeMismatch, "Value assignment, layout mismatch");
                    }

                    if (IsEmpty())
                    {
                        _type = other._type;
                    }
                    GetContext().MoveData(other, *this);
                }
            }
            other.Reset();
        }

        return *this;
    }

    Value::Value(Emittable data, std::optional<MemoryLayout> layout) :
        _data(data),
        _type(GetContext().GetType(data)),
        _layout(layout)
    {}

    Value::Value(ValueType type, std::optional<MemoryLayout> layout, int pointerLevel) :
        _type({ type, pointerLevel }),
        _layout(layout)
    {}

    Value::Value(detail::ValueTypeDescription typeDescription, std::optional<MemoryLayout> layout) noexcept :
        _type(typeDescription),
        _layout(layout)
    {}

    void Value::Reset()
    {
        _type = { ValueType::Undefined, 0 };
        _layout.reset();
        _data = {};
        _hasName = false;
        _name = "";
    }

    void Value::SetData(Value value, bool force)
    {
        if (!force && IsConstrained() && value.IsConstrained() && value.GetLayout() != GetLayout())
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Layout of the current value (" + GetLayout().ToString() + ") does not match that of the input value ("
                                  + value.GetLayout().ToString() + "). Alternatively, use force = 'true' to overwrite current layout.");
        }

        std::visit(VariantVisitor{ [this, force](Emittable emittable) {
                                      auto type = GetContext().GetType(emittable);
                                      if (!force && type.first != _type.first)
                                      {
                                          throw TypeMismatchException("Value::SetData (Emittable)", _type.first, type.first);
                                      }

                                      _data = emittable;
                                      _type = type;
                                  },
                                   [this, force](auto&& arg) {
                                       if (auto type = GetValueType<std::decay_t<decltype(arg)>>(); !force && type != _type.first)
                                       {
                                           throw TypeMismatchException("Value::SetData (auto&&)", _type.first, type);
                                       }

                                       _data = arg;
                                   } },
                   value._data);
    }

    bool Value::IsDefined() const
    {
        return _type.first != ValueType::Undefined;
    }

    bool Value::IsUndefined() const
    {
        return !IsDefined();
    }

    bool Value::IsEmpty() const
    {
        return std::visit(
            VariantVisitor{
                [](Emittable data) -> bool { return data.GetDataAs<void*>() == nullptr; },
                [](auto&& data) -> bool { return data == nullptr; } },
            _data);
    }

    bool Value::IsConstant() const
    {
        return GetContext().IsConstantData(*this);
    }

    bool Value::IsIntegral() const
    {
        switch (_type.first)
        {
        case ValueType::Boolean:
            [[fallthrough]];
        case ValueType::Byte:
            [[fallthrough]];
        case ValueType::Int8:
            [[fallthrough]];
        case ValueType::Int16:
            [[fallthrough]];
        case ValueType::Int32:
            [[fallthrough]];
        case ValueType::Int64:
            [[fallthrough]];
        case ValueType::Uint16:
            [[fallthrough]];
        case ValueType::Uint32:
            [[fallthrough]];
        case ValueType::Uint64:
            [[fallthrough]];
        case ValueType::Index:
            return true;
        default:
            return false;
        }
    }

    bool Value::IsBoolean() const
    {
        return _type.first == ValueType::Boolean;
    }

    bool Value::IsByte() const
    {
        return _type.first == ValueType::Byte;
    }

    bool Value::IsInt8() const
    {
        return _type.first == ValueType::Int8;
    }

    bool Value::IsInt16() const
    {
        return _type.first == ValueType::Int16;
    }

    bool Value::IsInt32() const
    {
        return _type.first == ValueType::Int32;
    }

    bool Value::IsInt64() const
    {
        return _type.first == ValueType::Int64;
    }

    bool Value::IsUint16() const
    {
        return _type.first == ValueType::Uint16;
    }

    bool Value::IsUint32() const
    {
        return _type.first == ValueType::Uint32;
    }

    bool Value::IsUint64() const
    {
        return _type.first == ValueType::Uint64;
    }

    bool Value::IsIndex() const
    {
        return _type.first == ValueType::Index;
    }

    bool Value::IsFloatingPoint() const
    {
        return (_type.first == ValueType::Float16 || _type.first == ValueType::Float || _type.first == ValueType::Double || _type.first == ValueType::BFloat16);
    }

    bool Value::IsFloat16() const
    {
        return _type.first == ValueType::Float16;
    }

    bool Value::IsFloat32() const
    {
        return _type.first == ValueType::Float;
    }

    bool Value::IsDouble() const
    {
        return _type.first == ValueType::Double;
    }

    bool Value::IsConstrained() const
    {
        return _layout.has_value();
    }

    const MemoryLayout& Value::GetLayout() const
    {
        return _layout.value();
    }

    ValueType Value::GetBaseType() const
    {
        return _type.first;
    }

    void Value::SetLayout(MemoryLayout layout)
    {
        GetContext().SetLayout(*this, layout);
    }

    Value Value::PointerTo() const
    {
        Value copy = *this;
        ++copy._type.second;
        return copy;
    }

    int Value::PointerLevel() const
    {
        return _type.second;
    }

    Value::UnderlyingDataType& Value::GetUnderlyingData()
    {
        return _data;
    }

    const Value::UnderlyingDataType& Value::GetUnderlyingData() const
    {
        return _data;
    }

    void Value::SetName(const std::string& name)
    {
        _name = name;
        if (!IsEmpty() && IsDefined())
            GetContext().SetName(*this, name);
        _hasName = true;
    }

    std::string Value::GetName() const
    {
        return _name;
    }

    bool Value::HasCustomName() const
    {
        return _hasName;
    }

    namespace detail
    {
        Value StoreConstantData(ConstantData data, std::optional<MemoryLayout> layout, const std::string& name)
        {
            auto size = std::visit(
                [](auto&& data) -> size_t {
                    return data.size();
                },
                data);

            auto dataLayout = layout.value_or(MemoryLayout(static_cast<int64_t>(size)));
            return GetContext().StoreConstantData(data, dataLayout, name);
        }
    } // namespace detail

#define ADD_TO_STRING_ENTRY(NAMESPACE, OPERATOR) \
    case NAMESPACE::OPERATOR:                    \
        return #OPERATOR;
#define BEGIN_FROM_STRING if (false)
#define ADD_FROM_STRING_ENTRY(NAMESPACE, OPERATOR) else if (name == #OPERATOR) return NAMESPACE::OPERATOR

    std::string ToString(ValueType vt)
    {
        switch (vt)
        {
            ADD_TO_STRING_ENTRY(ValueType, Undefined);
            ADD_TO_STRING_ENTRY(ValueType, Void);
            ADD_TO_STRING_ENTRY(ValueType, Boolean);
            ADD_TO_STRING_ENTRY(ValueType, Byte);
            ADD_TO_STRING_ENTRY(ValueType, Int8);
            ADD_TO_STRING_ENTRY(ValueType, Int16);
            ADD_TO_STRING_ENTRY(ValueType, Int32);
            ADD_TO_STRING_ENTRY(ValueType, Int64);
            ADD_TO_STRING_ENTRY(ValueType, Uint16);
            ADD_TO_STRING_ENTRY(ValueType, Uint32);
            ADD_TO_STRING_ENTRY(ValueType, Uint64);
            ADD_TO_STRING_ENTRY(ValueType, Index);
            ADD_TO_STRING_ENTRY(ValueType, Float16);
            ADD_TO_STRING_ENTRY(ValueType, BFloat16);
            ADD_TO_STRING_ENTRY(ValueType, Float);
            ADD_TO_STRING_ENTRY(ValueType, Double);

        default:
            return "Undefined";
        }
    }

    ValueType FromString(std::string name)
    {
        BEGIN_FROM_STRING;
        ADD_FROM_STRING_ENTRY(ValueType, Undefined);
        ADD_FROM_STRING_ENTRY(ValueType, Void);
        ADD_FROM_STRING_ENTRY(ValueType, Boolean);
        ADD_FROM_STRING_ENTRY(ValueType, Byte);
        ADD_FROM_STRING_ENTRY(ValueType, Int8);
        ADD_FROM_STRING_ENTRY(ValueType, Int16);
        ADD_FROM_STRING_ENTRY(ValueType, Int32);
        ADD_FROM_STRING_ENTRY(ValueType, Int64);
        ADD_FROM_STRING_ENTRY(ValueType, Uint16);
        ADD_FROM_STRING_ENTRY(ValueType, Uint32);
        ADD_FROM_STRING_ENTRY(ValueType, Uint64);
        ADD_FROM_STRING_ENTRY(ValueType, Index);
        ADD_FROM_STRING_ENTRY(ValueType, Float16);
        ADD_FROM_STRING_ENTRY(ValueType, BFloat16);
        ADD_FROM_STRING_ENTRY(ValueType, Float);
        ADD_FROM_STRING_ENTRY(ValueType, Double);

        return ValueType::Undefined;
    }

    namespace
    {
        template <typename T, typename C>
        constexpr bool ItemIsOneOf(T&& t, C&& c)
        {
            return llvm::any_of(c, [=](auto arg) { return t == arg; });
        }

    } // namespace

    bool IsImplicitlyCastable(ValueType source, ValueType target)
    {
        return GetContext().IsImplicitlyCastable(source, target);
    }

    bool IsImplicitlyCastable(ViewAdapter v1, ViewAdapter v2)
    {
        auto source = v1.GetValue().GetBaseType();
        auto target = v2.GetValue().GetBaseType();
        return IsImplicitlyCastable(source, target);
    }
} // namespace value
} // namespace accera

size_t std::hash<::accera::value::Value>::operator()(const ::accera::value::Value& value) const noexcept
{
    using ::accera::utilities::HashCombine;
    using ::accera::utilities::IntPtrT;
    using ::accera::utilities::VariantVisitor;
    using ::accera::value::Emittable;

    size_t hash = 0;

    HashCombine(hash,
                std::visit(
                    VariantVisitor{
                        [](Emittable emittable) { return std::hash<IntPtrT>{}(emittable.GetDataAs<IntPtrT>()); },
                        [](auto&& data) { return std::hash<std::decay_t<decltype(data)>>{}(data); } },
                    value.GetUnderlyingData()));
    HashCombine(hash, value.GetBaseType());
    HashCombine(hash, value.PointerLevel());

    if (value.IsConstrained())
    {
        HashCombine(hash, value.GetLayout());
    }
    else
    {
        // Special random value for an unconstrained Value
        HashCombine(hash, 0x87654321);
    }

    return hash;
}
