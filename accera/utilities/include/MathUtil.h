////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <type_traits>

namespace accera
{
namespace utilities
{
    namespace
    {
        template <typename T1, typename T2>
        using CommonType = typename std::common_type_t<T1, T2>;

    } // namespace

    template <typename NumeratorType, typename DenominatorType>
    CommonType<NumeratorType, DenominatorType> FloorDiv(NumeratorType numerator, DenominatorType denominator)
    {
        static_assert(std::is_integral_v<NumeratorType> && std::is_integral_v<DenominatorType>, "FloorDiv requires integral arguments");
        return numerator / denominator;
    }

    template <typename NumeratorType, typename DenominatorType>
    CommonType<NumeratorType, DenominatorType> CeilDiv(NumeratorType numerator, DenominatorType denominator)
    {
        static_assert(std::is_integral_v<NumeratorType> && std::is_integral_v<DenominatorType>, "CeilDiv requires integral arguments");
        return ((numerator - 1) / denominator) + 1;
    }

    template <typename NumToRoundType, typename MultipleType>
    CommonType<NumToRoundType, MultipleType> RoundDownToMultiple(NumToRoundType numToRound0, MultipleType multiple0)
    {
        static_assert(std::is_integral_v<NumToRoundType> && std::is_integral_v<MultipleType>, "RoundDownToMultiple requires integral arguments");
        using ResultType = CommonType<NumToRoundType, MultipleType>;
        ResultType numToRound = numToRound0, multiple = multiple0;
        assert(multiple);
        int isNegative;
        if constexpr (std::is_unsigned_v<NumToRoundType>)
        {
            isNegative = 1;
        }
        else
        {
            isNegative = static_cast<int>(numToRound < 0);
        }
        return ((numToRound + isNegative * (multiple - 1)) / multiple) * multiple;
    }

    template <typename NumToRoundType, typename MultipleType>
    CommonType<NumToRoundType, MultipleType> RoundUpToMultiple(NumToRoundType numToRound0, MultipleType multiple0)
    {
        static_assert(std::is_integral_v<NumToRoundType> && std::is_integral_v<MultipleType>, "RoundUpToMultiple requires integral arguments");
        using ResultType = CommonType<NumToRoundType, MultipleType>;
        ResultType numToRound = numToRound0, multiple = multiple0;
        assert(multiple);
        int isPositive;
        if constexpr (std::is_unsigned_v<NumToRoundType>)
        {
            isPositive = 1;
        }
        else
        {
            isPositive = static_cast<int>(numToRound >= 0);
        }
        return ((numToRound + isPositive * (multiple - 1)) / multiple) * multiple;
    }

} // namespace utilities
} // namespace accera
