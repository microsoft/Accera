////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include <utilities/include/PropertyBag.h>

#include <string>
#include <vector>

namespace accera
{
using namespace utilities;

TEST_CASE("TestPropertyBag")
{
    PropertyBag metadata;

    REQUIRE(metadata.IsEmpty());
    REQUIRE_FALSE(metadata.HasEntry("a"));

    metadata.SetEntry("a", std::string("1"));
    REQUIRE(metadata.HasEntry("a"));
    CHECK(metadata.GetEntry<std::string>("a") == "1");

    metadata.SetEntry("a", std::string("2"));
    CHECK(metadata.GetEntry<std::string>("a") == "2");

    auto foo = metadata.GetEntry<std::string>("a");
    CHECK(foo == "2");

    auto foo2 = metadata.GetEntry("a");
    CHECK(foo2.type() == typeid(std::string));
    CHECK(std::any_cast<std::string>(foo2) == "2");

    auto removedEntry = metadata.RemoveEntry("a");
    CHECK(std::any_cast<std::string>(removedEntry) == "2");

    auto removedEntries = metadata.RemoveEntry("a");
    CHECK(metadata.IsEmpty());

    SECTION("PropertyBag property access has side effect")
    {
        CHECK(!metadata["f"].has_value());
    }
}

} // namespace accera
