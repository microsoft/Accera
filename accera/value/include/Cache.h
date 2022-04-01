////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "FunctionDeclaration.h"
#include "Index.h"
#include "IterationDomain.h"
#include "Plan.h"
#include "Range.h"
#include "Scalar.h"
#include "VectorizationInformation.h"

#include <ir/include/exec/ExecutionPlanEnums.h>
#include <ir/include/value/ValueEnums.h>

#include <memory>
#include <variant>
#include <vector>

namespace accera::ir::loopnest
{
class ScheduleOp;
}

namespace accera
{
namespace value
{
    using loopnests::Index;
    using ScalarIndex = Scalar;

    class CacheImpl;

    using CacheIndexing = accera::ir::executionPlan::CacheIndexing;

    using CacheAllocation = accera::ir::executionPlan::CacheAllocation;

    using utilities::DimensionOrder;
    using utilities::MemoryAffineCoefficients;
    using utilities::MemorySpace;

    class Cache
    {
    public:
        // Automatic caching version
        Cache(accera::ir::loopnest::ScheduleOp schedule,
              ViewAdapter value,
              const std::optional<ScalarIndex>& keySliceIndex,
              const std::optional<int64_t>& maxElements,
              CacheIndexing mapping = CacheIndexing::GlobalToPhysical,
              CacheAllocation allocation = CacheAllocation::Automatic,
              MemorySpace memorySpace = MemorySpace::None,
              ExecutionTarget execTarget = targets::CPU{});

        // Manual caching versions
        Cache(accera::ir::loopnest::ScheduleOp schedule,
              std::variant<ViewAdapter, Cache*> value,
              const std::optional<ScalarIndex>& keySliceIndex,
              const std::optional<ScalarIndex>& triggerIndex,
              const std::optional<int64_t>& maxElements,
              const MemoryAffineCoefficients& memoryCoefficients,
              bool thrifty,
              bool doubleBufferCache = false,
              const std::optional<VectorizationInformation>& vectorizationInfo = std::nullopt,
              CacheIndexing mapping = CacheIndexing::GlobalToPhysical,
              CacheAllocation allocation = CacheAllocation::Automatic,
              MemorySpace memorySpace = MemorySpace::None,
              MemorySpace doubleBufferMemorySpace = MemorySpace::None,
              ExecutionTarget execTarget = targets::CPU{});

        Cache(accera::ir::loopnest::ScheduleOp schedule,
              std::variant<ViewAdapter, Cache*> value,
              const std::optional<ScalarIndex>& keySliceIndex,
              const std::optional<ScalarIndex>& triggerIndex,
              const std::optional<int64_t>& maxElements,
              const DimensionOrder& dimOrder,
              bool thrifty,
              bool doubleBufferCache = false,
              const std::optional<VectorizationInformation>& vectorizationInfo = std::nullopt,
              CacheIndexing mapping = CacheIndexing::GlobalToPhysical,
              CacheAllocation allocation = CacheAllocation::Automatic,
              MemorySpace memorySpace = MemorySpace::None,
              MemorySpace doubleBufferMemorySpace = MemorySpace::None,
              ExecutionTarget execTarget = targets::CPU{});

        // Runtime-Init caching version
        Cache(accera::ir::loopnest::ScheduleOp schedule,
              ViewAdapter value,
              const std::string& packingFunctionName,
              const std::string& packedBufferSizeFnName,
              CacheIndexing mapping = CacheIndexing::GlobalToPhysical);

        // Emit-time packed caching version
        Cache(accera::ir::loopnest::ScheduleOp schedule,
              ViewAdapter value,
              ViewAdapter constantData,
              const std::string& wrapperFnName,
              const std::string& packedBufferName,
              CacheIndexing mapping = CacheIndexing::GlobalToPhysical);

        Cache(const Cache&) = delete;
        Cache(Cache&&) noexcept;
        Cache& operator=(const Cache&) = delete;
        Cache& operator=(Cache&&) noexcept;
        ~Cache();

        Value GetBaseValue();

    private:
        std::unique_ptr<CacheImpl> _impl;
    };

} // namespace value
} // namespace accera
