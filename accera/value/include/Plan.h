////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Cache.h"
#include "FunctionDeclaration.h"
#include "Index.h"
#include "IterationDomain.h"
#include "Range.h"
#include "Scalar.h"
#include "VectorizationInformation.h"

#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/value/ValueEnums.h>
#include <utilities/include/MemoryLayout.h>

#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace accera
{
namespace value
{
    using ir::value::Processor;
    using utilities::DimensionOrder;
    using utilities::MemoryAffineCoefficients;
    using utilities::MemoryLayout;
    using utilities::MemorySpace;

    using accera::ir::executionPlan::CacheAllocation;
    using accera::ir::executionPlan::CacheIndexing;
    using accera::ir::value::CacheStrategyType;

    using loopnests::Index;
    using loopnests::IterationDomain;
    using loopnests::Range;
    using loopnests::SplitIndex;
    using ScalarIndex = Scalar;

    class Schedule;
    class Cache;
    class PlanImpl;

    enum class ParallelizationPolicy : int
    {
        Static,
        Dynamic
    };

    class Plan
    {
    public:
        Plan(const Plan&) = delete;
        Plan(Plan&&) noexcept;
        Plan& operator=(const Plan&) = delete;
        Plan& operator=(Plan&&) noexcept;
        ~Plan();

        /// <summary> Adds a manual active block cache for a view target or different cache </summary>
        /// <param name="target"> The target being cached (e.g Array, Matrix, etc) </param>
        /// <param name="outermostIncludedSplitIndex"> The outermost index in one of the cached dimensions to include in the cache </param>
        /// <param name="triggerIndex"> The index to fill the cache at, must be the same as outermostIncludedSplitIndex or precede it in the schedule order </param>
        /// <param name="memoryMap"> The affine coefficients to use to map from active block position to cache position in the cache buffer </param>
        /// <param name="elementType"> The element type to use in the cache </param>
        /// <param name="thrifty"> Whether to make this a thrifty cache </param>
        /// <param name="doubleBuffer"> Whether or not to use double-buffering to fill this cache </param>
        /// <param name="vectorizationInfo"> Optional vectorization configuration for the cache ops </param>
        /// <param name="indexing"> The cache indexing </param>
        /// <param name="allocation"> The cache allocation </param>
        /// <param name="memorySpace"> The memory space</param>
        /// <param name="doubleBufferMemorySpace"> The memory space to put the double buffer temporary buffer in </param>
        /// <returns> An instance of Cache </returns>
        Cache AddCache(std::variant<ViewAdapter, Cache*> target, const ScalarIndex& outermostIncludedSplitIndex, const ScalarIndex& triggerIndex, const MemoryAffineCoefficients& memoryMap, const std::optional<value::ValueType>& elementType = std::nullopt, bool thrifty = false, bool doubleBuffer = false, const std::optional<VectorizationInformation>& vectorizationInfo = std::nullopt, CacheIndexing indexing = CacheIndexing::GlobalToPhysical, CacheAllocation allocation = CacheAllocation::Automatic, MemorySpace memorySpace = MemorySpace::None, MemorySpace doubleBufferMemorySpace = MemorySpace::None);

        /// <summary> Adds a manual active block cache for a view target or different cache </summary>
        /// <param name="target"> The target being cached (e.g Array, Matrix, etc) </param>
        /// <param name="outermostIncludedSplitIndex"> The outermost index in one of the cached dimensions to include in the cache </param>
        /// <param name="triggerIndex"> The index to fill the cache at, must be the same as outermostIncludedSplitIndex or precede it in the schedule order </param>
        /// <param name="dimOrder"> The dimension order permutation to use to map from active block position to cache position in the cache buffer </param>
        /// <param name="elementType"> The element type to use in the cache </param>
        /// <param name="thrifty"> Whether to make this a thrifty cache </param>
        /// <param name="doubleBuffer"> Whether or not to use double-buffering to fill this cache </param>
        /// <param name="vectorizationInfo"> Optional vectorization configuration for the cache ops </param>
        /// <param name="indexing"> The cache indexing </param>
        /// <param name="allocation"> The cache allocation </param>
        /// <param name="memorySpace"> The memory space</param>
        /// <param name="doubleBufferMemorySpace"> The memory space to put the double buffer temporary buffer in </param>
        /// <returns> An instance of Cache </returns>
        Cache AddCache(std::variant<ViewAdapter, Cache*> target, const ScalarIndex& outermostIncludedSplitIndex, const ScalarIndex& triggerIndex, const DimensionOrder& dimOrder, const std::optional<value::ValueType>& elementType = std::nullopt, bool thrifty = false, bool doubleBuffer = false, const std::optional<VectorizationInformation>& vectorizationInfo = std::nullopt, CacheIndexing indexing = CacheIndexing::GlobalToPhysical, CacheAllocation allocation = CacheAllocation::Automatic, MemorySpace memorySpace = MemorySpace::None, MemorySpace doubleBufferMemorySpace = MemorySpace::None);

        /// <summary> Adds a manual active block cache for a view target or different cache with an identity dimension ordering </summary>
        /// <param name="target"> The target being cached (e.g Array, Matrix, etc) </param>
        /// <param name="outermostIncludedSplitIndex"> The outermost index in one of the cached dimensions to include in the cache </param>
        /// <param name="elementType"> The element type to use in the cache </param>
        /// <param name="thrifty"> Whether to make this a thrifty cache </param>
        /// <param name="doubleBuffer"> Whether or not to use double-buffering to fill this cache </param>
        /// <param name="vectorizationInfo"> Optional vectorization configuration for the cache ops </param>
        /// <param name="indexing"> The cache indexing </param>
        /// <param name="allocation"> The cache allocation </param>
        /// <param name="memorySpace"> The memory space</param>
        /// <param name="doubleBufferMemorySpace"> The memory space to put the double buffer temporary buffer in </param>
        /// <returns> An instance of Cache </returns>
        Cache AddCache(std::variant<ViewAdapter, Cache*> target, const ScalarIndex& outermostIncludedSplitIndex, const std::optional<value::ValueType>& elementType = std::nullopt, bool thrifty = false, bool doubleBuffer = false, const std::optional<VectorizationInformation>& vectorizationInfo = std::nullopt, CacheIndexing indexing = CacheIndexing::GlobalToPhysical, CacheAllocation allocation = CacheAllocation::Automatic, MemorySpace memorySpace = MemorySpace::None, MemorySpace doubleBufferMemorySpace = MemorySpace::None);

        /// <summary> Adds a manual active block cache for a view target or different cache </summary>
        /// <param name="target"> The target being cached (e.g Array, Matrix, etc) </param>
        /// <param name="maxElements"> A cutoff budget that will be used to select the outermost index in one of the cached dimensions to include in the cache (in order not to exceed the budget) </param>
        /// <param name="memoryMap"> The affine coefficients to use to map from active block position to cache position in the cache buffer </param>
        /// <param name="elementType"> The element type to use in the cache </param>
        /// <param name="thrifty"> Whether to make this a thrifty cache </param>
        /// <param name="doubleBuffer"> Whether or not to use double-buffering to fill this cache </param>
        /// <param name="vectorizationInfo"> Optional vectorization configuration for the cache ops </param>
        /// <param name="indexing"> The cache indexing </param>
        /// <param name="allocation"> The cache allocation </param>
        /// <param name="memorySpace"> The memory space</param>
        /// <param name="doubleBufferMemorySpace"> The memory space to put the double buffer temporary buffer in </param>
        /// <returns> An instance of Cache </returns>
        Cache AddCache(std::variant<ViewAdapter, Cache*> target, int64_t maxElements, const utilities::MemoryAffineCoefficients& memoryMap, const std::optional<value::ValueType>& elementType = std::nullopt, bool thrifty = false, bool doubleBuffer = false, const std::optional<VectorizationInformation>& vectorizationInfo = std::nullopt, CacheIndexing indexing = CacheIndexing::GlobalToPhysical, CacheAllocation allocation = CacheAllocation::Automatic, MemorySpace memorySpace = MemorySpace::None, MemorySpace doubleBufferMemorySpace = MemorySpace::None);

        /// <summary> Adds a manual active block cache for a view target or different cache </summary>
        /// <param name="target"> The target being cached (e.g Array, Matrix, etc) </param>
        /// <param name="maxElements"> A cutoff budget that will be used to select the outermost index in one of the cached dimensions to include in the cache (in order not to exceed the budget) </param>
        /// <param name="dimOrder"> The dimension order permutation to use to map from active block position to cache position in the cache buffer </param>
        /// <param name="elementType"> The element type to use in the cache </param>
        /// <param name="thrifty"> Whether to make this a thrifty cache </param>
        /// <param name="doubleBuffer"> Whether or not to use double-buffering to fill this cache </param>
        /// <param name="vectorizationInfo"> Optional vectorization configuration for the cache ops </param>
        /// <param name="indexing"> The cache indexing </param>
        /// <param name="allocation"> The cache allocation </param>
        /// <param name="memorySpace"> The memory space</param>
        /// <param name="doubleBufferMemorySpace"> The memory space to put the double buffer temporary buffer in </param>
        /// <returns> An instance of Cache </returns>
        Cache AddCache(std::variant<ViewAdapter, Cache*> target, int64_t maxElements, const DimensionOrder& dimOrder, const std::optional<value::ValueType>& elementType = std::nullopt, bool thrifty = false, bool doubleBuffer = false, const std::optional<VectorizationInformation>& vectorizationInfo = std::nullopt, CacheIndexing indexing = CacheIndexing::GlobalToPhysical, CacheAllocation allocation = CacheAllocation::Automatic, MemorySpace memorySpace = MemorySpace::None, MemorySpace doubleBufferMemorySpace = MemorySpace::None);

        /// <summary> Adds a manual active element cache for a view target or different cache with an identity dimension ordering </summary>
        /// <param name="target"> The target being cached (e.g Array, Matrix, etc) </param>
        /// <param name="maxElements"> A cutoff budget that will be used to select the outermost index in one of the cached dimensions to include in the cache (in order not to exceed the budget) </param>
        /// <param name="thrifty"> Whether to make this a thrifty cache </param>
        /// <param name="doubleBuffer"> Whether or not to use double-buffering to fill this cache </param>
        /// <param name="vectorizationInfo"> Optional vectorization configuration for the cache ops </param>
        /// <param name="indexing"> The cache indexing </param>
        /// <param name="allocation"> The cache allocation </param>
        /// <param name="memorySpace"> The memory space</param>
        /// <param name="doubleBufferMemorySpace"> The memory space to put the double buffer temporary buffer in </param>
        /// <returns> An instance of Cache </returns>
        Cache AddCache(std::variant<ViewAdapter, Cache*> target, int64_t maxElements, const std::optional<value::ValueType>& elementType = std::nullopt, bool thrifty = false, bool doubleBuffer = false, const std::optional<VectorizationInformation>& vectorizationInfo = std::nullopt, CacheIndexing indexing = CacheIndexing::GlobalToPhysical, CacheAllocation allocation = CacheAllocation::Automatic, MemorySpace memorySpace = MemorySpace::None, MemorySpace doubleBufferMemorySpace = MemorySpace::None);

        /// <summary> Emits an offline packing function for the given target and changes its usage in the function to assume a packed representation </summary>
        /// <param name="target"> The target being cached (e.g Array, Matrix, etc) </param>
        /// <param name="packingFnName"> The name of the packing function to emit </param>
        /// <param name="packedBufferSizeFnName"> The name of the function giving the size of the packed buffer to emit </param>
        /// <param name="indexing"> The cache indexing </param>
        /// <returns> An instance of Cache </returns>
        Cache EmitRuntimeInitPacking(ViewAdapter target, const std::string& packingFnName, const std::string& packedBufferSizeFnName, CacheIndexing indexing = CacheIndexing::GlobalToPhysical);

        /// <summary> Packs and embeds the given buffer of data into the binary following the offline packing format for the given target and changes its usage in the function to assume a packed representation. Also removes the given buffer as an argument to the function </summary>
        /// <param name="target"> The target being cached (e.g Array, Matrix, etc) </param>
        /// <param name="constantData"> The data to pack and cache </param>
        /// <param name="wrapperFnName"> The name to give the wrapping function that calls the base function with the packed data </param>
        /// <param name="packedBufferName"> The string name to give the buffer in the binary </param>
        /// <param name="indexing"> The cache indexing </param>
        /// <returns> An instance of Cache </returns>
        Cache PackAndEmbedBuffer(ViewAdapter target, ViewAdapter constantData, const std::string& wrapperFnName, const std::string& packedBufferName, CacheIndexing indexing = CacheIndexing::GlobalToPhysical);

        /// <summary> Vectorizes along an index </summary>
        /// <param name="i"> The scalar index indicating the axis to vectorize </param>
        /// <param name="vectorizationInfo"> The vectorization configuration </param>
        void Vectorize(ScalarIndex i, const VectorizationInformation& vectorizationInfo); // `i` must be backed by a SymbolicIndexOp

        /// <summary> Parallelizes one or more iteration space dimensions </summary>
        /// <param name="indices"> The scalar indices to parallelize. Specifying multiple indices is equivalent to the `collapse` argument in OpenMP. Therefore, the dimensions must be contiguous in the iteration space dimension order. </param>
        /// <param name="numThreads"> The number of threads to schedule. </param>
        /// <param name="policy"> The policy used to schedule work across the threads. </param>
        void Parallelize(std::vector<ScalarIndex> indices, int64_t numThreads, ParallelizationPolicy policy);

        void _EraseLoop(const value::ScalarIndex& index);

    private:
        friend class Schedule;
        Plan(Schedule& sched, ExecutionRuntime execRuntime = ExecutionRuntime::DEFAULT);

        std::unique_ptr<PlanImpl> _impl;
    };

    class GPUPlan
    {
    public:
        GPUPlan(const GPUPlan&) = delete;
        GPUPlan(GPUPlan&&) noexcept;
        GPUPlan& operator=(const GPUPlan&) = delete;
        GPUPlan& operator=(GPUPlan&&) noexcept;
        ~GPUPlan();

        /// <summary> Adds a cache for a view target </summary>
        /// <param name="target"> The target being cached (e.g Array, Matrix, etc) </param>
        /// <param name="outermostIncludedSplitIndex"> The outermost index in one of the cached dimensions to include in the cache </param>
        /// <param name="triggerIndex"> The index to fill the cache at, must be the same as outermostIncludedSplitIndex or precede it in the schedule order </param>
        /// <param name="dimOrder"> The dimension order permutation to use to map from active block position to cache position in the cache buffer </param>
        /// <param name="elementType"> The element type to use in the cache </param>
        /// <param name="thrifty"> Whether to make this a thrifty cache </param>
        /// <param name="doubleBuffer"> Whether or not to use double-buffering to fill this cache </param>
        /// <param name="vectorizationInfo"> Optional vectorization configuration for the cache ops </param>
        /// <param name="mapping"> The cache mapping </param>
        /// <param name="allocation"> The cache allocation </param>
        /// <param name="memorySpace"> The memory space</param>
        /// <param name="doubleBufferMemorySpace"> The memory space to put the double buffer temporary buffer in </param>
        /// <returns> An instance of Cache </returns>
        Cache AddCache(std::variant<ViewAdapter, Cache*> target, const ScalarIndex& outermostIncludedSplitIndex, const value::ScalarIndex& triggerIndex, const DimensionOrder& dimOrder, const std::optional<value::ValueType>& elementType, bool thrifty, bool doubleBuffer, CacheStrategyType strategy, const std::optional<VectorizationInformation>& vectorizationInfo, CacheIndexing mapping, CacheAllocation allocation, MemorySpace memorySpace, MemorySpace doubleBufferMemorySpace, const std::optional<uint64_t>& sharedMemOffset);

        /// <summary> Adds a cache for a view target </summary>
        /// <param name="target"> The target being cached (e.g Array, Matrix, etc) </param>
        /// <param name="maxElements"> A cutoff budget that can be used to infer the outermost index to include the cache </param>
        /// <param name="memorySpace"> The memory space</param>
        /// <returns> An instance of Cache </returns>
        Cache AddCache(ViewAdapter target, int64_t maxElements, CacheStrategyType strategy, MemorySpace memorySpace, const std::optional<uint64_t>& sharedMemOffset);

        /// <summary> Assigns an ordered sequence of loop indices to a GPU processor </summary>
        /// <param name="indices"> The loop indices </param>
        /// <param name="proc"> The GPU processor, indicating a block or thread </param>
        void MapIndicesToProcessor(std::vector<ScalarIndex> indices, Processor proc);

        /// <summary> Assigns a loop indes to a GPU processor </summary>
        /// <param name="index"> The loop index </param>
        /// <param name="proc"> The GPU processor, indicating a block or thread </param>
        void MapIndexToProcessor(ScalarIndex index, Processor proc);

        /// <summary> Tensorize three iteration space dimensions </summary>
        /// <param name="indices"> The scalar indices to tensorize. Three indices must be specified whose dimensions must be contiguous in the iteration space dimension order. </param>
        /// <param name="dims"> The dimension of the tensor operation. </param>
        /// <param name="numTotalPasses"> Total number of passes of the tensor operation to run. </param>
        /// <param name="useStaticOffsets"> Use precomputed index offsets for address calculation (potential optimization). </param>
        /// <param name="numFusedPasses"> Number of passes of the tensor operation for which to allocate register, higher value indicates higher register allocation. </param>
        /// <param name="schedulingPolicy"> Determines whether we iterate over blocks or passes. </param>
        void Tensorize(std::vector<ScalarIndex> indices, ir::value::MMAShapeType dims, int numTotalPasses = 1, bool useStaticOffsets = false, int numFusedPasses = -1, ir::value::MMASchedulingPolicyType schedulingPolicy = ir::value::MMASchedulingPolicyType::PassOrder, ir::value::MMAFragmentOpType prologueOp = ir::value::MMAFragmentOpType::None, double prologueArg = {}, ir::value::MMAFragmentOpType epilogueOp = ir::value::MMAFragmentOpType::None, double epilogueArg = {}, bool _useRocWMMA=false);

    private:
        friend class Schedule;
        GPUPlan(targets::GPU gpuOptions, Schedule& sched, ExecutionRuntime execRuntime = ExecutionRuntime::DEFAULT);

        std::unique_ptr<PlanImpl> _impl;
    };
} // namespace value
} // namespace accera
