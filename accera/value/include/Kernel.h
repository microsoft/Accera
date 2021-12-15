////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Scalar.h"

#include <utilities/include/FunctionUtils.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

namespace accera::ir::loopnest
{
class KernelOp;
}

namespace accera
{
namespace value
{
    class KernelImpl;

    class Kernel
    {
    public:
        Kernel( std::string id, std::function<void()> kernelFn);
        Kernel(Kernel&& other);
        ~Kernel();

        std::vector<Scalar> GetIndices() const;
        void dump();

    private:
        friend class Nest;
        friend class Schedule;
        accera::ir::loopnest::KernelOp GetOp() const;

        std::unique_ptr<KernelImpl> _impl;
    };

} // namespace value
} // namespace accera
#pragma region implementation
