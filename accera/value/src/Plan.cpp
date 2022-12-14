////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plan.h"
#include "Cache.h"
#include "MLIREmitterContext.h"
#include "Schedule.h"

#include <ir/include/IRUtil.h>
#include <ir/include/exec/ExecutionPlanAttributes.h>
#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/exec/ParallelizationInfo.h>
#include <ir/include/exec/TensorizationInfo.h>
#include <ir/include/exec/VectorizationInfo.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueAttributes.h>
#include <ir/include/value/ValueDialect.h>
#include <ir/include/value/ValueEnums.h>

#include <utilities/include/Exception.h>

#include <mlir/IR/Attributes.h>

#include <cassert>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

using namespace accera::ir::value;
using namespace accera::ir::loopnest;
using namespace accera::ir::executionPlan;

namespace accera
{
using namespace utilities;

namespace value
{
    // Implementation class
    class PlanImpl
    {
    public:
        PlanImpl(
            value::ExecutionTarget execTarget,
            ScheduleOp scheduleOp,
            value::ExecutionRuntime execRuntime) :
            _execRuntime(execRuntime),
            _execTarget(execTarget),
            _scheduleOp(scheduleOp)
        {
            std::visit(
                [this](auto options) {
                    // TODO: formalize setting exec target by using an interface
                    auto nestOp = mlir::dyn_cast<NestOp>(_scheduleOp->getParentOp());
                    _execPlanOp = _scheduleOp.getOrCreateExecPlan();

                    mlir::OpBuilder b(nestOp);
                    if constexpr (std::is_same_v<decltype(options), targets::CPU>)
                    {
                        auto execTargetAttr = ir::value::ExecutionTargetAttr::get(b.getContext(), ir::value::ExecutionTarget::CPU);
                        nestOp.exec_targetAttr(execTargetAttr);
                        _execPlanOp.exec_targetAttr(execTargetAttr);
                    }
                    else if constexpr (std::is_same_v<decltype(options), targets::GPU>)
                    {
                        auto execTargetAttr = ir::value::ExecutionTargetAttr::get(b.getContext(), ir::value::ExecutionTarget::GPU);
                        nestOp.exec_targetAttr(execTargetAttr);
                        _execPlanOp.exec_targetAttr(execTargetAttr);

                        if (_execRuntime != ExecutionRuntime::DEFAULT && _execRuntime != ExecutionRuntime::NONE && _execRuntime != ExecutionRuntime::OPENMP)
                        {
                            auto execRuntimeAttrName = ValueModuleOp::getExecRuntimeAttrName();
                            auto execRuntimeAttrValue = ir::value::ExecutionRuntimeAttr::get(
                                b.getContext(), (ir::value::ExecutionRuntime)_execRuntime);
                            if (auto mod = nestOp->getParentOfType<mlir::ModuleOp>())
                            {
                                mod->setAttr(execRuntimeAttrName, execRuntimeAttrValue);
                            }
                            if (auto mod = nestOp->getParentOfType<ValueModuleOp>())
                            {
                                mod->setAttr(execRuntimeAttrName, execRuntimeAttrValue);
                            }
                        }

                        _execPlanOp->setAttr(_execPlanOp.getGPULaunchAttrName(), options.ToArrayAttr(b.getContext()));
                    }
                    else
                        llvm_unreachable("Unexpected");
                },
                execTarget);
        }

        Cache AddAutomaticCache(ViewAdapter target, const std::optional<ScalarIndex>& keySliceIndex, const std::optional<int64_t>& maxElements, CacheIndexing mapping, CacheAllocation allocation, MemorySpace memorySpace, const std::optional<uint64_t>& sharedMemOffset, CacheStrategy strategy)
        {
            return { _scheduleOp, target, keySliceIndex, maxElements, sharedMemOffset, strategy, mapping, allocation, memorySpace, _execTarget };
        }

        Cache AddManualCache(std::variant<ViewAdapter, Cache*> target,
                             const std::optional<ScalarIndex>& keySliceIndex,
                             const std::optional<ScalarIndex>& triggerIndex,
                             const std::optional<int64_t>& maxElements,
                             const std::optional<value::ValueType>& elementType,
                             bool thrifty,
                             bool doubleBuffer,
                             const std::optional<VectorizationInformation>& vectorizationInfo,
                             CacheIndexing mapping,
                             CacheAllocation allocation,
                             MemorySpace memorySpace,
                             MemorySpace doubleBufferMemorySpace,
                             const MemoryAffineCoefficients& memoryMap,
                             CacheStrategy strategy = CacheStrategy::Striped)
        {
            return { _scheduleOp,
                     target,
                     keySliceIndex,
                     triggerIndex,
                     maxElements,
                     memoryMap,
                     elementType,
                     thrifty,
                     strategy,
                     doubleBuffer,
                     vectorizationInfo,
                     mapping,
                     allocation,
                     memorySpace,
                     doubleBufferMemorySpace,
                     _execTarget };
        }

        Cache AddManualCache(std::variant<ViewAdapter, Cache*> target,
                             const std::optional<ScalarIndex>& keySliceIndex,
                             const std::optional<ScalarIndex>& triggerIndex,
                             const std::optional<int64_t>& maxElements,
                             const std::optional<value::ValueType>& elementType,
                             bool thrifty,
                             bool doubleBuffer,
                             const std::optional<VectorizationInformation>& vectorizationInfo,
                             CacheIndexing mapping,
                             CacheAllocation allocation,
                             MemorySpace memorySpace,
                             MemorySpace doubleBufferMemorySpace,
                             const DimensionOrder& dimOrder,
                             const std::optional<uint64_t>& sharedMemOffset = {},
                             CacheStrategy strategy = CacheStrategy::Striped)
        {
            return { _scheduleOp,
                     target,
                     keySliceIndex,
                     triggerIndex,
                     maxElements,
                     dimOrder,
                     elementType,
                     thrifty,
                     sharedMemOffset,
                     strategy,
                     doubleBuffer,
                     vectorizationInfo,
                     mapping,
                     allocation,
                     memorySpace,
                     doubleBufferMemorySpace,
                     _execTarget };
        }

        Cache AddRuntimeInitCache(ViewAdapter target, const std::string& packingFnName, const std::string& packedBufferSizeFnName, CacheIndexing indexing)
        {
            return { _scheduleOp, target, packingFnName, packedBufferSizeFnName, indexing };
        }

        Cache PackAndEmbedBuffer(ViewAdapter target, ViewAdapter constantData, const std::string& wrapperFnName, const std::string& packedBufferName, CacheIndexing indexing)
        {
            return { _scheduleOp, target, constantData, wrapperFnName, packedBufferName, indexing };
        }

        void Vectorize(ScalarIndex i, const VectorizationInformation& dslVectorizationInfo)
        {
            auto& builder = GetBuilder();
            auto symbolicIndexOp = GetIndexOp(i);
            auto index = symbolicIndexOp.getValue();

            VectorizationInfo vectorizationInfo{ dslVectorizationInfo.vectorBytes, dslVectorizationInfo.vectorUnitCount, dslVectorizationInfo.unrollOnly };
            auto vectorizationInfoIdentifier = builder.getStringAttr(VectorizationInfoAttr::getKeyName());
            auto vectorizationInfoAttr = VectorizationInfoAttr::get(vectorizationInfo, builder.getContext());
            _scheduleOp.addLoopAttribute(index, vectorizationInfoIdentifier, vectorizationInfoAttr);

            // tag the ExecPlanOp with this vectorization info so that other cache ops
            // can extract this info from the loopnest graph later
            _execPlanOp->setAttr(vectorizationInfoIdentifier, vectorizationInfoAttr);
        }

        void Parallelize(std::vector<ScalarIndex> indices, int64_t numThreads, ParallelizationPolicy policy)
        {
            auto& builder = GetBuilder();

            ParallelizationInfo parallelizationInfo{ numThreads, policy == ParallelizationPolicy::Dynamic };
            auto parallelizationInfoIdentifier = builder.getStringAttr(ParallelizationInfoAttr::getKeyName());
            auto parallelizationInfoAttr = ParallelizationInfoAttr::get(parallelizationInfo, builder.getContext());

            // mark each index as parallelized
            // during lowering, indices are continguous in the schedule ordering will be collapsed
            for (auto& i : indices)
            {
                auto symbolicIndexOp = GetIndexOp(i);
                auto index = symbolicIndexOp.getValue();
                _scheduleOp.addLoopAttribute(index, parallelizationInfoIdentifier, parallelizationInfoAttr);
            }
        }

        void Tensorize(std::vector<ScalarIndex> indices, MMAShape dims, int numTotalPasses, bool useStaticOffsets, int numFusedPasses, MMASchedulingPolicy schedulingPolicy, bool _useRocWMMA)
        {
            auto& builder = GetBuilder();

            TensorizationInfo tensorizationInfo{ static_cast<accera::ir::value::MMAShape>(dims), numTotalPasses, useStaticOffsets, numFusedPasses, static_cast<accera::ir::value::MMASchedulingPolicy>(schedulingPolicy), _useRocWMMA };
            auto tensorizationInfoIdentifier = builder.getStringAttr(TensorizationInfoAttr::getKeyName());
            auto tensorizationInfoAttr = TensorizationInfoAttr::get(tensorizationInfo, builder.getContext());

            // mark each index as tensorized
            // during lowering, indices are continguous in the schedule ordering will be collapsed
            for (auto& i : indices)
            {
                auto symbolicIndexOp = GetIndexOp(i);
                auto index = symbolicIndexOp.getValue();
                _scheduleOp.addLoopAttribute(index, tensorizationInfoIdentifier, tensorizationInfoAttr);
            }

            // tag the ExecPlanOp with this tensorization info so that other cache ops
            // can extract this info from the loopnest graph later
            _execPlanOp->setAttr(tensorizationInfoIdentifier, tensorizationInfoAttr);
        }

        void MapIndicesToProcessor(std::vector<ScalarIndex>& scalarIndices, Processor proc)
        {
            // Create a mapping for each index bound to this processor

            // The mapping expressions are computed following a first-major-like mapping where the last index
            // has a stride of 1 in the mapping, the second to last index has a stride equal to the iteration
            // count of the last index, and so on
            // E.g. for ordered indices (i, j, k) bound to THREAD_Y the mappings will be:
            //      THREAD_Y = (i * num_iters(j) * num_iters(k)) + (j * num_iters(k)) + k
            //      But more usefully from this we can compute:
            //      mapping_k = ( THREAD_Y ) % num_iters(k)
            //      mapping_j = ( THREAD_Y // num_iters(k) ) % num_iters(j)
            //      mapping_i = ( THREAD_Y // (num_iters(j) * num_iters(k)) ) % num_iters(i)

            auto& builder = GetBuilder();

            // Position the builder inside the nest before the schedule op
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(_scheduleOp);

            // Get the current mapping dictionary
            auto planOp = _scheduleOp.getOrCreateExecPlan();

            std::vector<SymbolicIndexOp> symbolicIndexOps;
            std::transform(scalarIndices.begin(), scalarIndices.end(), std::back_inserter(symbolicIndexOps), [&](const ScalarIndex& index) { return GetIndexOp(index); });

            // Traverse the symbolicIndexOps from back to front to compute iteration shape stride information
            std::reverse(symbolicIndexOps.begin(), symbolicIndexOps.end());
            auto transformedDomain = _scheduleOp.getDomain().getValue();
            int64_t accumulatedStride = 1;
            std::vector<mlir::NamedAttribute> boundSymNameIndexPairs;
            for (auto& symbolicIndexOp : symbolicIndexOps)
            {
                auto index = symbolicIndexOp.getValue();
                auto range = transformedDomain.GetIndexRange(index);
                auto iterCount = range.NumIterations();
                auto currentStride = accumulatedStride;
                accumulatedStride *= iterCount;

                mlir::AffineMap boundMap = mlir::AffineMap::get(0, 1, (builder.getAffineSymbolExpr(0).floorDiv(currentStride)) % iterCount);

                planOp.addBinding(builder.getContext(), index, proc, boundMap);
            }
        }

        void _EraseLoop(const value::ScalarIndex& scalarIndex)
        {
            auto builder = GetBuilder();
            auto symbolicIndexOp = GetIndexOp(scalarIndex);
            auto index = symbolicIndexOp.getValue();
            _scheduleOp.addLoopAttribute(index, builder.getStringAttr("_erase"), builder.getUnitAttr());
        }

    private:
        mlir::OpBuilder& GetBuilder()
        {
            return ::accera::value::GetMLIRContext().GetOpBuilder();
        }

        // TODO : de-dupe with ScheduleImpl
        SymbolicIndexOp GetIndexOp(ScalarIndex val)
        {
            auto mlirValue = mlir::Value::getFromOpaquePointer(val.GetValue().Get<Emittable>().GetDataAs<MLIRContext::EmittableInfo*>()->data);
            auto op = mlirValue.getDefiningOp();
            assert(op);

            auto indexOp = llvm::dyn_cast_or_null<SymbolicIndexOp>(op);
            assert(indexOp);
            return indexOp;
        }

        value::ExecutionRuntime _execRuntime;
        value::ExecutionTarget _execTarget;
        ScheduleOp _scheduleOp;
        ExecPlanOp _execPlanOp;
    };

    //
    // Plan impl
    //

    Plan::Plan(Plan&& other) noexcept :
        _impl(std::move(other._impl))
    {}

    Plan& Plan::operator=(Plan&& other) noexcept
    {
        if (this != &other)
        {
            using std::swap;
            swap(_impl, other._impl);
        }
        return *this;
    }

    Plan::Plan(
        Schedule& schedule,
        value::ExecutionRuntime runtime /* = value::ExecutionRuntime::DEFAULT */) :
        _impl(std::make_unique<PlanImpl>(
            value::targets::CPU{},
            schedule.GetOp(),
            runtime))
    {}

    Plan::~Plan() = default;

    Cache Plan::AddCache(std::variant<ViewAdapter, Cache*> target, const ScalarIndex& outermostIncludedSplitIndex, const ScalarIndex& triggerIndex, const MemoryAffineCoefficients& memoryMap, const std::optional<value::ValueType>& elementType, bool thrifty, bool doubleBuffer, const std::optional<VectorizationInformation>& vectorizationInfo, CacheIndexing mapping, CacheAllocation allocation, MemorySpace memorySpace, MemorySpace doubleBufferMemorySpace)
    {
        return _impl->AddManualCache(target, outermostIncludedSplitIndex, triggerIndex, std::nullopt, elementType, thrifty, doubleBuffer, vectorizationInfo, mapping, allocation, memorySpace, doubleBufferMemorySpace, memoryMap);
    }

    Cache Plan::AddCache(std::variant<ViewAdapter, Cache*> target, const ScalarIndex& outermostIncludedSplitIndex, const ScalarIndex& triggerIndex, const DimensionOrder& dimOrder, const std::optional<value::ValueType>& elementType, bool thrifty, bool doubleBuffer, const std::optional<VectorizationInformation>& vectorizationInfo, CacheIndexing mapping, CacheAllocation allocation, MemorySpace memorySpace, MemorySpace doubleBufferMemorySpace)
    {
        return _impl->AddManualCache(target, outermostIncludedSplitIndex, triggerIndex, std::nullopt, elementType, thrifty, doubleBuffer, vectorizationInfo, mapping, allocation, memorySpace, doubleBufferMemorySpace, dimOrder);
    }

    Cache Plan::AddCache(std::variant<ViewAdapter, Cache*> target, int64_t maxElements, const MemoryAffineCoefficients& memoryMap, const std::optional<value::ValueType>& elementType, bool thrifty, bool doubleBuffer, const std::optional<VectorizationInformation>& vectorizationInfo, CacheIndexing mapping, CacheAllocation allocation, MemorySpace memorySpace, MemorySpace doubleBufferMemorySpace)
    {
        return _impl->AddManualCache(target, std::nullopt, std::nullopt, maxElements, elementType, thrifty, doubleBuffer, vectorizationInfo, mapping, allocation, memorySpace, doubleBufferMemorySpace, memoryMap);
    }

    Cache Plan::AddCache(std::variant<ViewAdapter, Cache*> target, int64_t maxElements, const DimensionOrder& dimOrder, const std::optional<value::ValueType>& elementType, bool thrifty, bool doubleBuffer, const std::optional<VectorizationInformation>& vectorizationInfo, CacheIndexing mapping, CacheAllocation allocation, MemorySpace memorySpace, MemorySpace doubleBufferMemorySpace)
    {
        return _impl->AddManualCache(target, std::nullopt, std::nullopt, maxElements, elementType, thrifty, doubleBuffer, vectorizationInfo, mapping, allocation, memorySpace, doubleBufferMemorySpace, dimOrder);
    }

    Cache Plan::AddCache(std::variant<ViewAdapter, Cache*> target, const ScalarIndex& outermostIncludedSplitIndex, const std::optional<value::ValueType>& elementType, bool thrifty, bool doubleBuffer, const std::optional<VectorizationInformation>& vectorizationInfo, CacheIndexing mapping, CacheAllocation allocation, MemorySpace memorySpace, MemorySpace doubleBufferMemorySpace)
    {
        Value baseValue;
        if (std::holds_alternative<Cache*>(target))
        {
            auto cache = std::get<Cache*>(target);
            baseValue = cache->GetBaseValue();
        }
        else
        {
            auto viewAdapter = std::get<ViewAdapter>(target);
            baseValue = viewAdapter.GetValue();
        }
        int64_t rank = baseValue.GetLayout().NumDimensions();
        DimensionOrder dimOrder(rank);
        return _impl->AddManualCache(target, outermostIncludedSplitIndex, outermostIncludedSplitIndex, std::nullopt, elementType, thrifty, doubleBuffer, vectorizationInfo, mapping, allocation, memorySpace, doubleBufferMemorySpace, dimOrder);
    }

    Cache Plan::AddCache(std::variant<ViewAdapter, Cache*> target, int64_t maxElements, const std::optional<value::ValueType>& elementType, bool thrifty, bool doubleBuffer, const std::optional<VectorizationInformation>& vectorizationInfo, CacheIndexing mapping, CacheAllocation allocation, MemorySpace memorySpace, MemorySpace doubleBufferMemorySpace)
    {
        Value baseValue;
        if (std::holds_alternative<Cache*>(target))
        {
            auto cache = std::get<Cache*>(target);
            baseValue = cache->GetBaseValue();
        }
        else
        {
            auto viewAdapter = std::get<ViewAdapter>(target);
            baseValue = viewAdapter.GetValue();
        }
        int64_t rank = baseValue.GetLayout().NumDimensions();
        DimensionOrder dimOrder(rank);
        auto viewAdapter = std::get<ViewAdapter>(target);
        return _impl->AddManualCache(target, std::nullopt, std::nullopt, maxElements, elementType, thrifty, doubleBuffer, vectorizationInfo, mapping, allocation, memorySpace, doubleBufferMemorySpace, dimOrder);
    }

    Cache Plan::EmitRuntimeInitPacking(ViewAdapter target, const std::string& packingFnName, const std::string& packedBufferSizeFnName, CacheIndexing indexing)
    {
        return _impl->AddRuntimeInitCache(target, packingFnName, packedBufferSizeFnName, indexing);
    }

    Cache Plan::PackAndEmbedBuffer(ViewAdapter target, ViewAdapter constantData, const std::string& wrapperFnName, const std::string& packedBufferName, CacheIndexing indexing)
    {
        return _impl->PackAndEmbedBuffer(target, constantData, wrapperFnName, packedBufferName, indexing);
    }

    void Plan::Vectorize(ScalarIndex i, const VectorizationInformation& vectorizationInfo)
    {
        _impl->Vectorize(i, vectorizationInfo);
    }

    void Plan::Parallelize(std::vector<ScalarIndex> indices, int64_t numThreads, ParallelizationPolicy policy)
    {
        _impl->Parallelize(indices, numThreads, policy);
    }

    void Plan::_EraseLoop(const value::ScalarIndex& index)
    {
        _impl->_EraseLoop(index);
    }

    //
    // GPUPlan impl
    //
    GPUPlan::GPUPlan(GPUPlan&& other) noexcept :
        _impl(std::move(other._impl))
    {}

    GPUPlan& GPUPlan::operator=(GPUPlan&& other) noexcept
    {
        if (this != &other)
        {
            using std::swap;
            swap(_impl, other._impl);
        }
        return *this;
    }

    GPUPlan::~GPUPlan() = default;

    GPUPlan::GPUPlan(value::targets::GPU gpuOptions, Schedule& sched, ExecutionRuntime runtime) :
        _impl(std::make_unique<PlanImpl>(gpuOptions, sched.GetOp(), runtime))
    {
    }

    Cache GPUPlan::AddCache(std::variant<ViewAdapter, Cache*> target, const ScalarIndex& outermostIncludedSplitIndex, const value::ScalarIndex& triggerIndex, const DimensionOrder& dimOrder, const std::optional<value::ValueType>& elementType, bool thrifty, bool doubleBuffer, CacheStrategy strategy, const std::optional<VectorizationInformation>& vectorizationInfo, CacheIndexing mapping, CacheAllocation allocation, MemorySpace memorySpace, MemorySpace doubleBufferMemorySpace, const std::optional<uint64_t>& sharedMemOffset)
    {
        return _impl->AddManualCache(target, outermostIncludedSplitIndex, triggerIndex, std::nullopt, elementType, thrifty, doubleBuffer, vectorizationInfo, mapping, allocation, memorySpace, doubleBufferMemorySpace, dimOrder, sharedMemOffset, strategy);
    }

    Cache GPUPlan::AddCache(ViewAdapter target, int64_t maxElements, CacheStrategy strategy, MemorySpace memorySpace, const std::optional<uint64_t>& sharedMemOffset)
    {
        return _impl->AddAutomaticCache(target, std::nullopt, maxElements, CacheIndexing::GlobalToPhysical, CacheAllocation::Automatic, memorySpace, sharedMemOffset, strategy);
    }

    void GPUPlan::Tensorize(std::vector<ScalarIndex> indices, MMAShape dims, int numTotalPasses, bool useStaticOffsets, int numFusedPasses, MMASchedulingPolicy schedulingPolicy, bool _useRocWMMA)
    {
        _impl->Tensorize(indices, dims, numTotalPasses, useStaticOffsets, numFusedPasses, schedulingPolicy, _useRocWMMA);
    }

    void GPUPlan::MapIndicesToProcessor(std::vector<ScalarIndex> indices, Processor proc)
    {
        _impl->MapIndicesToProcessor(indices, proc);
    }

    void GPUPlan::MapIndexToProcessor(ScalarIndex index, Processor proc)
    {
        MapIndicesToProcessor({ index }, proc);
    }
} // namespace value
} // namespace accera
