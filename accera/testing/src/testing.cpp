////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Ofer Dekel
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "testing.h"

#include <utilities/include/Logger.h>

#include <iostream>
#include <string>
#include <vector>

namespace accera
{
namespace testing
{
    namespace
    {
        bool testFailedFlag = false;
        std::vector<std::string> testFailures;
        std::vector<std::string> testSuccesses;
        std::vector<std::string> testWarnings;
    } // namespace

    TestFailureException::TestFailureException(const std::string& testDescription) :
        std::runtime_error(std::string("TestFailureException: ") + testDescription)
    {
    }

    TestNotImplementedException::TestNotImplementedException(const std::string& testDescription) :
        std::runtime_error(std::string("TestNotImplementedException: ") + testDescription)
    {
    }

    //
    // vectors
    //

    template <typename ValueType>
    bool IsVectorEqual(const std::vector<ValueType>& a, const std::vector<ValueType>& b)
    {
        auto size = a.size();
        if (size != b.size())
        {
            return false;
        }

        for (size_t index = 0; index < size; ++index)
        {
            if (a[index] != b[index])
            {
                return false;
            }
        }
        return true;
    }

    template <typename ValueType1, typename ValueType2, typename ValueType3>
    bool IsVectorApproxEqual(const std::vector<ValueType1>& a, const std::vector<ValueType2>& b, ValueType3 tolerance)
    {
        // allow vectors of different size, provided that they differ by a suffix of zeros
        size_t size = a.size();
        if (b.size() < size)
        {
            size = b.size();
        }

        for (size_t i = 0; i < size; ++i)
        {
            if (IsEqual(a[i], b[i], tolerance) == false)
            {
                return false;
            }
        }

        // confirm suffix of zeros
        for (size_t i = size; i < a.size(); ++i)
        {
            if (IsEqual(a[i], static_cast<ValueType1>(0), tolerance) == false)
            {
                return false;
            }
        }

        for (size_t i = size; i < b.size(); ++i)
        {
            if (IsEqual(b[i], static_cast<ValueType2>(0), tolerance) == false)
            {
                return false;
            }
        }

        return true;
    }

    bool IsEqual(const std::vector<bool>& a, const std::vector<bool>& b)
    {
        return IsVectorEqual(a, b);
    }

    bool IsEqual(const std::vector<int>& a, const std::vector<int>& b)
    {
        return IsVectorEqual(a, b);
    }

    bool IsEqual(const std::vector<int64_t>& a, const std::vector<int64_t>& b)
    {
        return IsVectorEqual(a, b);
    }

    bool IsEqual(const std::vector<std::string>& a, const std::vector<std::string>& b)
    {
        return IsVectorEqual(a, b);
    }

    bool IsEqual(const std::vector<float>& a, const std::vector<float>& b, float tolerance)
    {
        return IsVectorApproxEqual(a, b, tolerance);
    }

    bool IsEqual(const std::vector<double>& a, const std::vector<double>& b, double tolerance)
    {
        return IsVectorApproxEqual(a, b, tolerance);
    }

    bool IsEqual(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, float tolerance)
    {
        for (size_t index = 0; index < a.size(); ++index)
        {
            if (!IsVectorApproxEqual(a[index], b[index], tolerance))
            {
                return false;
            }
        }
        return true;
    }

    template <typename ValueType1, typename ValueType2>
    bool IsEqual(const std::vector<std::vector<ValueType1>>& a, const std::vector<std::vector<ValueType2>>& b, double tolerance)
    {
        if (a.size() != b.size())
        {
            return false;
        }

        for (size_t index = 0; index < a.size(); ++index)
        {
            if (!IsVectorApproxEqual(a[index], b[index], tolerance))
            {
                return false;
            }
        }
        return true;
    }

    bool ProcessTest(const std::string& testDescription, bool success)
    {
        if (!success)
        {
            TestFailed(testDescription);
        }
        else
        {
            TestSucceeded(testDescription);
        }

        return success;
    }

    bool ProcessQuietTest(const std::string& testDescription, bool success)
    {
        if (!success)
        {
            TestFailed(testDescription);
        }

        return success;
    }

    void ProcessCriticalTest(const std::string& testDescription, bool success)
    {
        if (!ProcessTest(testDescription, success))
        {
            throw TestFailureException(testDescription);
        }
    }

    void TestFailed(const std::string& message)
    {
        std::cout << message << " ... Failed" << std::endl;
        testFailedFlag = true;
        testFailures.push_back(message);
    }

    void TestSucceeded(const std::string& message)
    {
        std::cout << message << " ... Success" << std::endl;
        testSuccesses.push_back(message);
    }

    void TestWarning(const std::string& message)
    {
        std::cout << "[Warning]\t" << message << std::endl;
        testWarnings.push_back(message);
    }

    bool DidTestFail()
    {
        return testFailedFlag;
    }

    std::vector<std::string> GetFailedTests()
    {
        return testFailures;
    }

    std::vector<std::string> GetSuccessfulTests()
    {
        return testSuccesses;
    }

    std::vector<std::string> GetTestWarnings()
    {
        return testWarnings;
    }

    int GetExitCode()
    {
        return DidTestFail() ? 1 : 0;
    }

    void Reset()
    {
        testFailedFlag = false;
        testFailures.clear();
        testSuccesses.clear();
        testWarnings.clear();
    }

    void PrintTestSummary()
    {
        auto total = testFailures.size() + testSuccesses.size() + testWarnings.size();
        std::cout << "Test summary" << std::endl;
        std::cout << "  " << testSuccesses.size() << " / " << total << " tests succeeded" << std::endl;
        std::cout << "  " << testFailures.size() << " / " << total << " tests failed" << std::endl;
        std::cout << "  " << testWarnings.size() << " / " << total << " tests produced warnings" << std::endl;
    }

    void PrintTestDetails()
    {
        std::cout << "Test details" << std::endl;
        if (!testSuccesses.empty())
        {
            std::cout << "  Successful tests:" << std::endl;
            for (const auto& t : testSuccesses)
            {
                std::cout << "    " << t << std::endl;
            }
            std::cout << std::endl;
        }

        if (!testFailures.empty())
        {
            std::cout << "  Failed tests:" << std::endl;
            for (const auto& t : testFailures)
            {
                std::cout << "    " << t << std::endl;
            }
            std::cout << std::endl;
        }

        if (!testWarnings.empty())
        {
            std::cout << "  Warnings:" << std::endl;
            for (const auto& t : testWarnings)
            {
                std::cout << "    " << t << std::endl;
            }
        }
    }

    EnableLoggingHelper::EnableLoggingHelper()
    {
        logging::ShouldLog() = true;
    }

    EnableLoggingHelper::~EnableLoggingHelper()
    {
        logging::ShouldLog() = false;
    }

    template bool IsEqual(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, double tolerance);
    template bool IsEqual(const std::vector<std::vector<float>>& a, const std::vector<std::vector<double>>& b, double tolerance);
    template bool IsEqual(const std::vector<std::vector<double>>& a, const std::vector<std::vector<float>>& b, double tolerance);
    template bool IsEqual(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b, double tolerance);
} // namespace testing
} // namespace accera
