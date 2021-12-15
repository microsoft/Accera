////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chris Lovett
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include <utilities/include/Files.h>
#include <utilities/include/StringUtil.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
namespace fs = std::filesystem;

using namespace accera;

TEST_CASE("Stringf with args")
{
    REQUIRE(utilities::FormatString("test %d is %s", 10, "fun") == "test 10 is fun");
}

std::string GetUnicodeTestPath(const std::string& basePath, const std::string& utf8test)
{
    std::string testing = utilities::JoinPaths(basePath, "Testing");
    std::string unicode = utilities::JoinPaths(testing, "Unicode");

    std::string testdir = utilities::JoinPaths(unicode, utf8test);

    return testdir;
}

TEST_CASE("PathTests", "[.][!throws][!mayfail]")
{
    auto basePath = utilities::GetWorkingDirectory();

    SECTION("JoinPaths")
    {
        std::vector<std::string> parts = utilities::SplitPath(basePath);
        std::string result = utilities::JoinPaths("", parts);

        // normalize path for platform differences (just for testing)
        std::string norm = basePath;
        std::replace(norm.begin(), norm.end(), '\\', '/');
        std::replace(result.begin(), result.end(), '\\', '/');

        REQUIRE(norm == result);
    }

    SECTION("Unicode")
    {
#ifdef WIN32
        // chinese for 'test'
        std::wstring test(L"\u6D4B\u8bd5");
        auto path = fs::path(test);
        auto utf8test = path.u8string();
#else
        std::string utf8test{ "测试" };
#endif
        auto testdir = GetUnicodeTestPath(basePath, utf8test);
        std::cout << "writing test output to " << testdir << std::endl;

        // bugbug: the rolling build for Linux is giving us EACCESSDENIED on this testdir for some reason...
        utilities::EnsureDirectoryExists(testdir);
        REQUIRE(utilities::DirectoryExists(testdir));

        std::string testContent = "this is a test";
        auto testContentLength = testContent.size();

        // chinese for 'banana'
#ifdef WIN32
        std::wstring banana(L"\u9999\u8549");
        std::string utf8banana = fs::path(banana).u8string();
#else
        std::string utf8banana{ "香蕉" };
#endif
        utf8banana += ".txt";

        std::string testfile = utilities::JoinPaths(testdir, utf8banana);
        {
            auto outputStream = utilities::OpenOfstream(testfile);
            outputStream.write(testContent.c_str(), testContentLength);
        }
        {
            auto inputStream = utilities::OpenIfstream(testfile);
            char buffer[100];
            inputStream.read(buffer, testContentLength);
            buffer[testContentLength] = '\0';
            std::string actual(buffer);

            REQUIRE(actual == testContent);
        }
    }
}
