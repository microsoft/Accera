////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include <utilities/include/TunableParameters.h>

#include <algorithm>
#include <vector>

namespace accera
{

using namespace utilities;

TEST_CASE("TunableParameters_test1")
{
    std::vector expected{ 1, 2, 3, 4 };
    std::vector<int> actual;
    TunableParameter p(expected, "expected");
    TuningEngine engine(p);
    do
    {
        actual.push_back(p);
    } while (engine.Next());

    CHECK(actual == expected);
}

TEST_CASE("TunableParameters_test2")
{
    std::vector<int> p1Values{ 1, 2, 3 };
    std::vector<int> p2Values{ 4, 5 };
    std::vector<int> p3Values{ 6, 7, 8 };
    TunableParameter p1(p1Values, "p1");
    TunableParameter p2(p2Values, "p2");
    TunableParameter p3(p3Values, "p3");
    std::vector expected{
        std::vector{ 1, 4, 6 },
        std::vector{ 1, 4, 7 },
        std::vector{ 1, 4, 8 },
        std::vector{ 1, 5, 6 },
        std::vector{ 1, 5, 7 },
        std::vector{ 1, 5, 8 },
        std::vector{ 2, 4, 6 },
        std::vector{ 2, 4, 7 },
        std::vector{ 2, 4, 8 },
        std::vector{ 2, 5, 6 },
        std::vector{ 2, 5, 7 },
        std::vector{ 2, 5, 8 },
        std::vector{ 3, 4, 6 },
        std::vector{ 3, 4, 7 },
        std::vector{ 3, 4, 8 },
        std::vector{ 3, 5, 6 },
        std::vector{ 3, 5, 7 },
        std::vector{ 3, 5, 8 },
    };
    std::vector<std::vector<int>> actual;
    TuningEngine engine(p1, p2, p3);
    do
    {
        actual.push_back(std::vector<int>{ p1, p2, p3 });
    } while (engine.Next());

    CHECK(std::equal(expected.begin(), expected.end(), actual.begin(), actual.end()));

    engine.Reset();
    CHECK(expected[0] == std::vector<int>{ p1, p2, p3 });
}

} // namespace accera
