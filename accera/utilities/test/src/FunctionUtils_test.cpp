////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include <utilities/include/FunctionUtils.h>

#include <type_traits>

namespace accera
{

TEST_CASE("FunctionUtils")
{
    int g_Value = 0;

    auto VoidFunction1 = [&] {
        g_Value = 1;
    };

    auto VoidFunction2 = [&] {
        g_Value = 2;
    };

    auto VoidFunction3 = [&] {
        g_Value = 3;
    };

    auto VoidFunction4 = [&] {
        g_Value = 4;
    };

    auto AddToGlobalValue = [&](int value) {
        g_Value += value;
    };

    auto ReturnIntFunction = [&] {
        return 1;
    };

    SECTION("TestInOrderFunctionEvaluator")
    {
        utilities::InOrderFunctionEvaluator(VoidFunction1, VoidFunction2, VoidFunction3, VoidFunction4);
        REQUIRE(g_Value == 4);
    }

    SECTION("TestApplyToEach")
    {
        utilities::ApplyToEach(AddToGlobalValue, 1, 2, 3, 4, 5);
        REQUIRE(g_Value == 1 + 2 + 3 + 4 + 5);
    }

    SECTION("TestFunctionTraits")
    {
        STATIC_REQUIRE(std::is_same_v<utilities::FunctionReturnType<decltype(ReturnIntFunction)>, int>);
        STATIC_REQUIRE(std::is_same_v<std::tuple_element_t<0, utilities::FunctionArgTypes<decltype(AddToGlobalValue)>>, int>);
    }
}

} // namespace accera
