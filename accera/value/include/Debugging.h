////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

#include "Array.h"

#define GET_LOCATION() \
    FileLocation { __FILE__, __LINE__ }

namespace accera
{
namespace value
{
    struct FileLocation
    {
        std::string file;
        int64_t line;
    };

    class LocationGuardImpl;
    class LocationGuard
    {
    public:
        LocationGuard(FileLocation location);
        ~LocationGuard();
        mlir::Location GetLocation() const;

    private:
        std::unique_ptr<LocationGuardImpl> _impl;
    };

    void CheckAllClose(Array actual, Array desired, Scalar tolerance);

} // namespace value

namespace ir
{
    inline std::string GetDebugModeAttrName()
    {
        return "accv.debug";
    }

    inline std::string GetPrintErrorFunctionName()
    {
        return "_acc_eprintf_";
    }

} // namespace ir

} // namespace accera
