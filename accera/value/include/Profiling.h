////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace accera
{
namespace value
{
    void EnterProfileRegion(const std::string& regionName);
    void ExitProfileRegion(const std::string& regionName);
    void PrintProfileResults();

    class ProfileRegion
    {
    public:
        explicit ProfileRegion(const std::string& regionName);
        ~ProfileRegion();

    private:
        std::string _regionName;
    };
} // namespace value
} // namespace accera
