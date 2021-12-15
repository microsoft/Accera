////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Profiling.h"
#include "EmitterContext.h"

namespace accera
{
namespace value
{
    void EnterProfileRegion(const std::string& regionName)
    {
        GetContext().EnterProfileRegion(regionName);
    }

    void ExitProfileRegion(const std::string& regionName)
    {
        GetContext().ExitProfileRegion(regionName);
    }

    void PrintProfileResults()
    {
        GetContext().PrintProfileResults();
    }

    ProfileRegion::ProfileRegion(const std::string& regionName) :
        _regionName(regionName)
    {
        EnterProfileRegion(_regionName);
    }

    ProfileRegion::~ProfileRegion()
    {
        ExitProfileRegion(_regionName);
    }
} // namespace value
} // namespace accera
