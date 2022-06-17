////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <utilities/include/Boolean.h>
#include <utilities/include/Exception.h>
#include <utilities/include/TypeTraits.h>

#include <cstdint>
#include <string>

namespace accera
{
namespace value
{

    enum class index_t : int64_t {};

    struct float16_t {
        using underlying_type = float;
        float data;
    };

    struct bfloat16_t {
        using underlying_type = float;
        float data;
    };

    /// <summary> An enumeration of primitive types supported by the value library </summary>
    enum class ValueType
    {
        /// <summary> undefined type </summary>
        Undefined = -1,
        /// <summary> void type </summary>
        Void = 0,
        /// <summary> index type </summary>
        Index,
        /// <summary> 1 byte boolean </summary>
        Boolean,
        /// <summary> 1 byte unsigned integer </summary>
        Byte,
        /// <summary> 1 byte signed integer </summary>
        Int8,
        /// <summary> 2 byte signed integer </summary>
        Int16,
        /// <summary> 4 byte signed integer </summary>
        Int32,
        /// <summary> 8 byte signed integer </summary>
        Int64,
        /// <summary> 2 byte unsigned integer </summary>
        Uint16,
        /// <summary> 4 byte unsigned integer </summary>
        Uint32,
        /// <summary> 8 byte unsigned integer </summary>
        Uint64,
        /// <summary> 2 byte floating point </summary>
        Float16,
        /// <summary> 2 byte Brain floating point </summary>
        BFloat16,
        /// <summary> 4 byte floating point </summary>
        Float,
        /// <summary> 8 byte floating point </summary>
        Double,
    };

    /// <summary> An enumeration of unary operations supported by the value library </summary>
    enum class ValueUnaryOperation
    {
        /// <summary> Logical negation </summary>
        LogicalNot,
    };

    /// <summary> An enumeration of binary operations supported by the value library </summary>
    enum class ValueBinaryOperation
    {
        /// <summary> Addition operation </summary>
        add,
        /// <summary> Subtraction operation </summary>
        subtract,
        /// <summary> Multiplication operation </summary>
        multiply,
        /// <summary> Division operation </summary>
        divide,
        /// <summary> Remainder operation </summary>
        modulus,
        logicalAnd,
        logicalOr
    };

    enum class ValueLogicalOperation
    {
        equality,
        inequality,
        lessthan,
        lessthanorequal,
        greaterthan,
        greaterthanorequal
    };

    /// <summary> Helper template function that does a mapping from C++ type to ValueType </summary>
    /// <typeparam name="Ty"> The C++ type that should be used to get the corresponding enum value </typeparam>
    /// <returns> The ValueType enum value that matches the C++ type provided </returns>
    template <typename Ty>
    constexpr ValueType GetValueType()
    {
        using T = std::decay_t<utilities::RemoveAllPointersT<Ty>>;

        if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, utilities::Boolean>)
        {
            return ValueType::Boolean;
        }
        else if constexpr (std::is_same_v<T, uint8_t>)
        {
            return ValueType::Byte;
        }
        else if constexpr (std::is_same_v<T, char> || std::is_same_v<T, int8_t>)
        {
            return ValueType::Int8;
        }
        else if constexpr (std::is_same_v<T, short>  || std::is_same_v<T, int16_t>)
        {
            return ValueType::Int16;
        }
        else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>)
        {
            return ValueType::Int32;
        }
        else if constexpr (std::is_same_v<T, int64_t>)
        {
            return ValueType::Int64;
        }
        else if constexpr (std::is_same_v<T, uint16_t>)
        {
            return ValueType::Uint16;
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
            return ValueType::Uint32;
        }
        else if constexpr (std::is_same_v<T, uint64_t>)
        {
            return ValueType::Uint64;
        }
        else if constexpr (std::is_same_v<T, index_t>)
        {
            return ValueType::Index;
        }
        else if constexpr (std::is_same_v<T, float16_t>)
        {
            return ValueType::Float16;
        }
        else if constexpr (std::is_same_v<T, bfloat16_t>)
        {
            return ValueType::BFloat16;
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return ValueType::Float;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return ValueType::Double;
        }
        else
        {
            static_assert(utilities::FalseType<T>::value, "Unknown value type");
        }
    }

    /// <summary> Get a string representation of the enum value </summary>
    std::string ToString(ValueType t);
    ValueType FromString(std::string name);

} // namespace value
} // namespace accera
