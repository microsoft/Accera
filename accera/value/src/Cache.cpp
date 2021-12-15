////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Cache.h"
#include "EmitterContext.h"
#include "MLIREmitterContext.h"

#include <ir/include/IRUtil.h>
#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueDialect.h>

#include <utilities/include/Exception.h>
#include <utilities/include/MemoryLayout.h>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace accera::ir::loopnest;
using namespace accera::ir::executionPlan;
namespace vir = accera::ir::value;

namespace accera
{
using namespace utilities;

namespace value
{
    namespace
    {
        Index GetIndex(const ScalarIndex& scalarIndex)
        {
            Index underlyingIndex;
            auto indexValue = mlir::Value::getFromOpaquePointer(scalarIndex.GetValue().Get<Emittable>().GetDataAs<MLIRContext::EmittableInfo*>()->data);
            assert(mlir::isa<SymbolicIndexOp>(indexValue.getDefiningOp()));
            auto symbolicIndex = mlir::dyn_cast<SymbolicIndexOp>(indexValue.getDefiningOp());
            return symbolicIndex.getValue();
        }

        std::vector<Index> GetIndices(const std::vector<ScalarIndex>& dimensions)
        {
            std::vector<Index> underlyingIndices;
            std::transform(dimensions.begin(), dimensions.end(), std::back_inserter(underlyingIndices), [](const ScalarIndex& scalarIndex) { return GetIndex(scalarIndex); });
            return underlyingIndices;
        }

        enum class CopyDirection : int
        {
            SourceToCache,
            CacheToSource,
        };
    } // namespace

    class CacheImpl
    {
    public:
        Value GetBaseValue()
        {
            return _input;
        }

    protected:
        CacheImpl(ScheduleOp schedule, std::variant<Value, CacheImpl*> input, CacheIndexing cacheIndexMapping) :
            _scheduleOp(schedule),
            _cacheIndexMapping(cacheIndexMapping),
            _cacheId(accera::ir::util::GetUniqueId())
        {
            if (std::holds_alternative<Value>(input))
            {
                _input = std::get<Value>(input);
                _baseMlirValueInput = mlir::Value::getFromOpaquePointer(_input.Get<Emittable>().GetDataAs<MLIRContext::EmittableInfo*>()->data);
                _mlirValueInput = _baseMlirValueInput;
                _hierarchicalCacheLevel = 0;
            }
            else
            {
                auto parentCacheImpl = std::get<CacheImpl*>(input);
                _input = parentCacheImpl->_input;
                _mlirValueInput = parentCacheImpl->_cacheValue;
                _baseMlirValueInput = parentCacheImpl->_baseMlirValueInput;
                _hierarchicalCacheLevel = parentCacheImpl->_hierarchicalCacheLevel + 1;
                if (_mlirValueInput == nullptr)
                {
                    throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Source cache for hierarchical cache is invalid becaase it doesn't have a cache mlir::Value");
                }
            }
        }

        mlir::OpBuilder GetBuilder()
        {
            return mlir::OpBuilder(_scheduleOp);
        }

        mlir::Location GetLocation()
        {
            return _scheduleOp.getLoc();
        }

        mlir::MemRefType GetInputType()
        {
            auto inputValueType = _mlirValueInput.getType();
            assert(inputValueType.isa<mlir::MemRefType>());
            return inputValueType.cast<mlir::MemRefType>();
        }

        std::vector<int64_t> GetInputShape()
        {
            auto inputType = GetInputType();
            return inputType.getShape().vec();
        }

        std::vector<int64_t> GetCacheShape()
        {
            return _cacheInfo.cacheType.getShape().vec();
        }

        mlir::Type GetElementType()
        {
            auto inputType = GetInputType();
            return inputType.getElementType();
        }

        ValueType GetValueElementType()
        {
            return _input.GetBaseType();
        }

        ScheduleOp _scheduleOp;
        Value _input;
        mlir::Value _baseMlirValueInput;
        mlir::Value _mlirValueInput;
        CacheIndexing _cacheIndexMapping;
        int64_t _cacheId;
        int64_t _hierarchicalCacheLevel;
        CacheInfo _cacheInfo; // Subclasses set this manually
        mlir::Value _cacheValue; // Subclasses set this manually
    };

    class AutomaticCacheImpl : public CacheImpl
    {
    public:
        AutomaticCacheImpl(ScheduleOp schedule,
                           Value value,
                           const std::optional<Index>& outermostIncludedSplitIndex,
                           const std::optional<int64_t>& maxElements,
                           CacheIndexing mapping,
                           CacheAllocation allocation,
                           MemorySpace dslMemorySpace,
                           ExecutionOptions execOptions) :
            CacheImpl(schedule, value, mapping),
            _execOptions(execOptions)
        {
            auto memorySpace = *ir::value::symbolizeMemorySpace((uint64_t)dslMemorySpace);

            auto builder = GetBuilder();
            auto loc = GetLocation();

            _cacheInfo = MakeAutomaticCacheInfo(builder, _mlirValueInput, allocation, schedule, outermostIncludedSplitIndex, maxElements, memorySpace);

            if (allocation == CacheAllocation::Automatic)
            {
                auto makeCache = builder.create<MakeCacheOp>(loc, _cacheInfo.cacheType, memorySpace);
                _cacheValue = makeCache;
                auto op = makeCache;
                op->moveBefore(schedule.getOperation());
            }
            else
            {
                _cacheValue = _mlirValueInput;
            }

            _cacheAccessContext = MakeCacheAccessContext(_cacheValue, _cacheInfo);
            _cacheAccessContext.cacheRegionRelevantScheduleIndexRanges = _cacheInfo.cacheRegionRelevantScheduleIndexRanges;
            _cacheAccessContext.cacheRegionBaseIndices = _cacheInfo.cacheRegionBaseIndices;

            BeginCacheRegionOp regionOp = builder.create<BeginCacheRegionOp>(loc, _mlirValueInput, _cacheAccessContext, _mlirValueInput, *_cacheInfo.cacheIndex, *_cacheInfo.triggerIndex, _cacheId, _hierarchicalCacheLevel, false, false);
            auto endOp = builder.create<EndCacheRegionOp>(loc, regionOp);
            _scheduleOp.injectMapping(regionOp);
        }

        CacheAccessContext _cacheAccessContext;
        ExecutionOptions _execOptions;
    };

    class ActiveBlockCacheImpl : public CacheImpl
    {
    public:
        ActiveBlockCacheImpl(ScheduleOp schedule,
                             std::variant<Value, CacheImpl*> value,
                             const std::optional<Index>& keySliceIndex,
                             const std::optional<Index>& triggerIndex,
                             const std::optional<int64_t>& maxElements,
                             const std::variant<MemoryAffineCoefficients, DimensionOrder>& cacheMapping,
                             CacheIndexing mapping,
                             CacheAllocation allocation,
                             MemorySpace dslMemorySpace,
                             ExecutionOptions execOptions) :
            CacheImpl(schedule, value, mapping),
            _execOptions(execOptions)
        {
            auto builder = GetBuilder();
            auto loc = builder.getUnknownLoc();
            auto memorySpace = *ir::value::symbolizeMemorySpace((uint64_t)dslMemorySpace);

            _cacheInfo = MakeManualCacheInfo(builder, _baseMlirValueInput, allocation, schedule, keySliceIndex, triggerIndex, maxElements, cacheMapping, memorySpace);

            if (allocation == CacheAllocation::Automatic)
            {
                auto makeCache = builder.create<MakeCacheOp>(loc, _cacheInfo.cacheType, memorySpace);
                _cacheValue = makeCache.getResult();
                auto op = makeCache;
                op->moveBefore(schedule.getOperation());
            }
            else
            {
                _cacheValue = _mlirValueInput;
            }

            _cacheAccessContext = MakeCacheAccessContext(_cacheValue, _cacheInfo);
            _cacheAccessContext.cacheRegionRelevantScheduleIndexRanges = _cacheInfo.cacheRegionRelevantScheduleIndexRanges;
            _cacheAccessContext.cacheRegionBaseIndices = _cacheInfo.cacheRegionBaseIndices;

            mlir::Operation* cacheRegionOp;
            if (maxElements.has_value())
            {
                auto loopIndices = schedule.getOrder();
                assert(!loopIndices.empty());
                auto innermostIndex = loopIndices.back();

                BeginMaxElementCacheRegionOp regionOp = builder.create<BeginMaxElementCacheRegionOp>(loc,
                                                                                                     _mlirValueInput,
                                                                                                     _cacheValue,
                                                                                                     _baseMlirValueInput,
                                                                                                     _cacheInfo.accessMaps,
                                                                                                     _cacheInfo.maxElementBudget,
                                                                                                     innermostIndex,
                                                                                                     _cacheId,
                                                                                                     _hierarchicalCacheLevel,
                                                                                                     _cacheInfo.dimReorderCache);
                cacheRegionOp = regionOp;
            }
            else
            {
                BeginCacheRegionOp regionOp = builder.create<BeginCacheRegionOp>(loc, _mlirValueInput, _cacheAccessContext, _baseMlirValueInput, *_cacheInfo.cacheIndex, *_cacheInfo.triggerIndex, _cacheId, _hierarchicalCacheLevel, true, _cacheInfo.dimReorderCache);
                cacheRegionOp = regionOp;
            }
            auto regionHandle = cacheRegionOp->getResult(0);
            auto endOp = builder.create<EndCacheRegionOp>(loc, regionHandle);
            _scheduleOp.injectMapping(cacheRegionOp);
        }

        CacheAccessContext _cacheAccessContext;
        ExecutionOptions _execOptions;
    };

    class OfflineCacheImpl : public CacheImpl
    {
    protected:
        OfflineCacheImpl(ScheduleOp schedule, Value input, CacheIndexing cacheIndexMapping) :
            CacheImpl(schedule, input, cacheIndexMapping)
        {
            _vModuleOp = accera::ir::util::CastOrGetParentOfType<vir::ValueModuleOp>(_scheduleOp);
        }

        int64_t GetCacheVolume()
        {
            auto cacheShape = _cacheInfo.cacheType.getShape().vec();
            int64_t cacheVolume = 1;
            for (auto dimSize : cacheShape)
            {
                cacheVolume *= dimSize;
            }
            return cacheVolume;
        }

        vir::ValueFuncOp GetScheduleFuncOp()
        {
            return accera::ir::util::CastOrGetParentOfType<vir::ValueFuncOp>(_scheduleOp);
        }

        void UpdateScheduleOpToUsePackedBuffer(mlir::OpBuilder& builder)
        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);
            auto loc = GetLocation();

            // Set insertion point to the schedule area
            // Need to change the input argument to have the cache shape
            // and change all accesses to use the
            // split iteration dimensions -> physical cache position map

            auto scheduleFuncOp = GetScheduleFuncOp();

            // TODO : fix hack - As a quick hack, instead of replacing the argument type, just reshape the argument
            // and replace all uses to use the reshaped version. This won't result in the right shape being output
            // in the HAT file, but will work for MLIR purposes for now
            builder.setInsertionPointToStart(&scheduleFuncOp.body().front());

            llvm::SmallVector<int64_t> strides;
            int64_t offset{};
            (void)mlir::getStridesAndOffset(_cacheInfo.cacheType, strides, offset);
            auto reshapedMemrefOp = builder.create<mlir::memref::ReinterpretCastOp>(
                loc,
                _cacheInfo.cacheType,
                _mlirValueInput,
                offset, // offset,
                GetCacheShape(), // shape
                strides);
            mlir::Value reshapedMemref = reshapedMemrefOp;

            _mlirValueInput.replaceAllUsesExcept(reshapedMemref, reshapedMemrefOp);

            // Get the schedule's split indices that are used for each dimension of the cache
            // Each element of the outer vector corresponds to a dimension in the cache buffer
            // The element contains all of the split indices that are summed to produce the
            // cache dimension index value
            // TODO : support more than simple addition of indices
            std::vector<std::vector<Index>> indicesUsedForCacheDims = _shardMapping.relevantScheduleIndices;
            assert(indicesUsedForCacheDims.size() == _cacheInfo.cacheType.getRank());

            // Change slice ops to access the packed version of the buffer
            // TODO : handle more than slice ops - find or create a common interface for modifying buffer access patterns
            _vModuleOp.walk([&](vir::SliceOp sliceOp) {
                if (sliceOp.source() == reshapedMemref)
                {
                    builder.setInsertionPoint(sliceOp);
                    std::vector<mlir::Value> cacheSliceOperands;
                    cacheSliceOperands.reserve(indicesUsedForCacheDims.size());
                    auto transformedDomain = _scheduleOp.getDomain().getValue();
                    for (const auto& dimIndices : indicesUsedForCacheDims)
                    {
                        assert(!dimIndices.empty());

                        // We want the iteration index, not the iteration variable value, so take the current iteration value and divide it by the step size for this loop
                        mlir::Value firstIndexOp = _scheduleOp.getOrCreateSymbolicIndex(builder, dimIndices.front());
                        int64_t currentLoopStepSize = transformedDomain.GetIndexRange(dimIndices.front()).Increment();

                        // Use AffineApplyOps on index calculations in order to work with vector and affine dialect utilities
                        auto dimZero = builder.getAffineDimExpr(0);
                        auto iterationCountDivExpr = dimZero.floorDiv(builder.getAffineConstantExpr(currentLoopStepSize));
                        auto iterationCountDivMap = mlir::AffineMap::get(1, 0, iterationCountDivExpr);
                        mlir::Value accumulatedIndexOps = builder.create<mlir::AffineApplyOp>(loc, iterationCountDivMap, mlir::ValueRange{ firstIndexOp });

                        for (size_t idx = 1; idx < dimIndices.size(); ++idx)
                        {
                            // When multiple indices are collapsed into a single cache dimension, the distance between elements
                            // identified by indices increases based on how many indices are combined
                            // e.g. if we have a cache shape [64, 32, 16, 4] with input mapping [0, 1, 1, 0],
                            //      then we combine the inner dimension-1 mapped elements to produce
                            //      [64, 512, 4] with input mapping [0, 1, 0]
                            //      and suppose that [index0, index1, index2, index3] map to the [64, 32, 16, 4] shape,
                            //      then we would construct [ index0 / step0, index1 / step1, index2 / step2, index3 / step3 ] as our accessing indices
                            //      but if we've collapsed the inner two dimensions of the cache, we need to map index1 and index2 differently
                            //      to account for this:
                            //      [ index0 / step0, (index1 / step1) * iterationCount2 + (index2 / step2), index3 / step3 ]

                            // Multiple the previously accumulated indices by the current index iteration count
                            int64_t currentLoopIterationCount = transformedDomain.GetIndexRange(dimIndices[idx]).NumIterations();
                            auto upscalePreviousIndicesExpr = dimZero * builder.getAffineConstantExpr(currentLoopIterationCount);
                            auto upscalePreviousIndicesMap = mlir::AffineMap::get(1, 0, upscalePreviousIndicesExpr);
                            mlir::Value upScaledPreviousIndices = builder.create<mlir::AffineApplyOp>(loc, upscalePreviousIndicesMap, mlir::ValueRange{ accumulatedIndexOps });

                            // Produce the current index iteration count
                            mlir::Value symbolicIndexOp = _scheduleOp.getOrCreateSymbolicIndex(builder, dimIndices[idx]);
                            currentLoopStepSize = transformedDomain.GetIndexRange(dimIndices[idx]).Increment();
                            iterationCountDivExpr = dimZero.floorDiv(builder.getAffineConstantExpr(currentLoopStepSize));
                            iterationCountDivMap = mlir::AffineMap::get(1, 0, iterationCountDivExpr);
                            mlir::Value iterationCount = builder.create<mlir::AffineApplyOp>(loc, iterationCountDivMap, mlir::ValueRange{ symbolicIndexOp });

                            // Update the accumulation to be the sum of the upscaled previous indices and the current index iteration count
                            auto accumulationExpr = dimZero + builder.getAffineDimExpr(1);
                            auto accumulationMap = mlir::AffineMap::get(2, 0, accumulationExpr);
                            accumulatedIndexOps = builder.create<mlir::AffineApplyOp>(loc, accumulationMap, mlir::ValueRange{ upScaledPreviousIndices, iterationCount });
                        }
                        cacheSliceOperands.push_back(accumulatedIndexOps);
                    }
                    std::vector<int64_t> sliceDimensions(indicesUsedForCacheDims.size());
                    std::iota(sliceDimensions.begin(), sliceDimensions.end(), 0);
                    vir::SliceOp newSliceOp = builder.create<vir::SliceOp>(loc,
                                                                           reshapedMemref,
                                                                           sliceDimensions,
                                                                           cacheSliceOperands,
                                                                           sliceOp.getType());
                    sliceOp.result().replaceAllUsesWith(newSliceOp);
                }
            });

            // TODO : actually change the arg type to the function, add an attribute to the function arg indicating that happened
            // and spread the changed arg shape out through callers in the module
#if 0
            // This ifdef'd out code replaces the function argument type but doesn't flow that type back to callers
            int replacementArgIdx = 0;
            for (const auto& arg : scheduleFuncOp.getArguments())
            {
                if (arg == _mlirValueInput)
                {
                    break;
                }
                replacementArgIdx++;
            }
            assert(replacementArgIdx < scheduleFuncOp.getNumArguments());
            auto initFnType = scheduleFuncOp.getType();
            auto argTypes = initFnType.getInputs().vec();
            argTypes[replacementArgIdx] = _cacheInfo.cacheType;
            auto newFnType = builder.getFunctionType(argTypes, initFnType.getResults());
            scheduleFuncOp.setType(newFnType);
            // Set the type of the block argument too
            _mlirValueInput.setType(_cacheInfo.cacheType);
#endif
        }

        ScheduleShardMapping _shardMapping;
        vir::ValueModuleOp _vModuleOp;
    };

    // Runtime initialized cache Implementation class
    class RuntimeInitCacheImpl : public OfflineCacheImpl
    {
    public:
        RuntimeInitCacheImpl(ScheduleOp schedule, Value value, const std::string& packingFunctionName, const std::string& packedBufferSizeFnName, CacheIndexing mapping) :
            OfflineCacheImpl(schedule, value, mapping)
        {
            auto loc = GetLocation();
            auto builder = GetBuilder();

            auto memorySpace = *ir::value::symbolizeMemorySpace((uint64_t)MemorySpace::Global);

            _cacheInfo = MakeFullBufferAutomaticCacheInfo(builder, _mlirValueInput, CacheAllocation::Automatic, schedule, memorySpace);
            _shardMapping = _cacheInfo.shardMapping;

            // Make the packing function
            // TODO : do we want to emit an un-pack function by default?
            CreatePackingFunction(builder, packingFunctionName);

            // Make the packed buffer size function
            CreatePackedBufferSizeFunction(builder, packedBufferSizeFnName);

            // Change the schedule to assume a packed verison of the buffer
            UpdateScheduleOpToUsePackedBuffer(builder);
        }

    private:
        void AddCacheZero(mlir::OpBuilder& builder, mlir::Value cache)
        {
            auto loc = builder.getUnknownLoc();
            [[maybe_unused]] auto cacheZero = builder.create<CacheZeroOp>(loc, cache);
        }

        void AddCacheCopy(mlir::OpBuilder& builder, mlir::Value input, CacheAccessContext cacheAccessContext, CopyDirection direction)
        {
            auto loc = builder.getUnknownLoc();
            if (direction == CopyDirection::SourceToCache)
            {
                builder.create<ActiveElementCacheCopyOp>(loc, input, cacheAccessContext);
            }
            else
            {
                builder.create<ActiveElementCacheCopyOp>(loc, cacheAccessContext, input);
            }
        }

        void CreatePackingFunction(mlir::OpBuilder& builder, const std::string& packingFunctionName)
        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);

            auto loc = GetLocation();
            auto insertionPoint = accera::ir::util::GetTerminalInsertPoint<vir::ValueModuleOp, vir::ModuleTerminatorOp>(_vModuleOp);
            builder.restoreInsertionPoint(insertionPoint);

            auto packingFnType = builder.getFunctionType({ GetInputType(), _cacheInfo.cacheType }, llvm::None);
            vir::ValueFuncOp packingFuncOp = builder.create<vir::ValueFuncOp>(loc, packingFunctionName + "_internal", packingFnType, ir::value::ExecutionTarget::CPU);

            builder.setInsertionPointToStart(&packingFuncOp.body().front());

            auto inputArrayVal = packingFuncOp.getArgument(0);
            auto outputArrayVal = packingFuncOp.getArgument(1);

            auto packingCacheAccessContext = MakeCacheAccessContext(outputArrayVal, _cacheInfo);
            AddCacheZero(builder, outputArrayVal);

            AddCacheCopy(builder, inputArrayVal, packingCacheAccessContext, CopyDirection::SourceToCache);
            builder.create<vir::ReturnOp>(loc);

            // Now create the Raw pointer API function for this wrapper function
            vir::CreateRawPointerAPIWrapperFunction(builder, packingFuncOp, packingFunctionName);
        }

        void CreatePackedBufferSizeFunction(mlir::OpBuilder& builder, const std::string& packedBufferSizeFnName)
        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);
            auto loc = GetLocation();

            auto insertionPoint = accera::ir::util::GetTerminalInsertPoint<vir::ValueModuleOp, vir::ModuleTerminatorOp>(_vModuleOp);
            builder.restoreInsertionPoint(insertionPoint);

            // We know the packed buffer size at emit-time, so just build and return a constant
            auto bufferSizeFnType = builder.getFunctionType({}, { builder.getI64Type() });
            vir::ValueFuncOp bufferSizeFuncOp = builder.create<vir::ValueFuncOp>(loc, packedBufferSizeFnName, bufferSizeFnType, ir::value::ExecutionTarget::CPU);
            bufferSizeFuncOp->setAttr(ir::HeaderDeclAttrName, builder.getUnitAttr());
            bufferSizeFuncOp->setAttr(ir::RawPointerAPIAttrName, builder.getUnitAttr());

            builder.setInsertionPointToStart(&bufferSizeFuncOp.body().front());

            mlir::Value constantSize = builder.create<mlir::ConstantIntOp>(loc, GetCacheVolume(), builder.getI64Type());
            builder.create<vir::ReturnOp>(loc, constantSize);
        }
    };

    // Emit-time packed cache Implementation class
    class EmitTimePackedCacheImpl : public OfflineCacheImpl
    {
    public:
        EmitTimePackedCacheImpl(ScheduleOp schedule, Value value, Value constantData, const std::string& wrapperFnName, const std::string& packedBufferName, CacheIndexing mapping) :
            OfflineCacheImpl(schedule, value, mapping)
        {
            auto builder = GetBuilder();

            auto constData = mlir::Value::getFromOpaquePointer(constantData.Get<Emittable>().GetDataAs<MLIRContext::EmittableInfo*>()->data);

            auto memorySpace = *ir::value::symbolizeMemorySpace((uint64_t)MemorySpace::Global);

            _cacheInfo = MakeFullBufferAutomaticCacheInfo(builder, _mlirValueInput, CacheAllocation::Automatic, schedule, memorySpace);

            // Create a new Array and pack it now, then store the packed array in the binary
            switch (GetValueElementType())
            {
            case ValueType::Int8:
                _packedBuffer = EmbedPackedBuffer<signed char>(builder, constData, packedBufferName);
                break;
            case ValueType::Byte:
                _packedBuffer = EmbedPackedBuffer<unsigned char>(builder, constData, packedBufferName);
                break;
            case ValueType::Int16:
                _packedBuffer = EmbedPackedBuffer<short>(builder, constData, packedBufferName);
                break;
            case ValueType::Int32:
                _packedBuffer = EmbedPackedBuffer<int>(builder, constData, packedBufferName);
                break;
            case ValueType::Int64:
                _packedBuffer = EmbedPackedBuffer<int64_t>(builder, constData, packedBufferName);
                break;
            case ValueType::Float:
                _packedBuffer = EmbedPackedBuffer<float>(builder, constData, packedBufferName);
                break;
            case ValueType::Double:
                _packedBuffer = EmbedPackedBuffer<double>(builder, constData, packedBufferName);
                break;
            default:
                throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Unsupported element type for caching");
            }
            _packedBufferMlirValue = mlir::Value::getFromOpaquePointer(_packedBuffer.GetValue().Get<Emittable>().GetDataAs<MLIRContext::EmittableInfo*>()->data);

            // Create a wrapper function that calls the accera function with the schedule op but hard-codes the arg corresponding
            // to the packed buffer and removes it as an arg in the wrapper function
            CreateWrapperFunction(builder, wrapperFnName);

            // Change the schedule to assume the given buffer is packed
            UpdateScheduleOpToUsePackedBuffer(builder);
        }

    private:
        template <typename Fn>
        void RecursiveWalkPackedShape(const ScheduleShardMapping& packingShapeInfo, size_t currentDimIdx, std::vector<int64_t>& currentCacheIndices, Fn&& callback)
        {
            if (currentDimIdx == packingShapeInfo.shardSizes.size())
            {
                // We're at the innermost layer, representing a single element of the packed buffer, call the callback with the current indices
                callback(currentCacheIndices);
            }
            else
            {
                for (int64_t currentCacheDimIdx = 0; currentCacheDimIdx < packingShapeInfo.shardSizes[currentDimIdx]; ++currentCacheDimIdx)
                {
                    currentCacheIndices[currentDimIdx] = currentCacheDimIdx;
                    RecursiveWalkPackedShape(packingShapeInfo, currentDimIdx + 1, currentCacheIndices, callback);
                }
            }
        }

        std::vector<int64_t> MapPackedIndicesToUnpackedIndices(int64_t dimCount, const ScheduleShardMapping& packingShapeInfo, const std::vector<int64_t>& cacheIndices)
        {
            std::vector<int64_t> unpackedIndices(dimCount, 0);
            for (size_t cacheDimIdx = 0; cacheDimIdx < packingShapeInfo.affinePerDimCoefficients.size(); ++cacheDimIdx)
            {
                unpackedIndices[packingShapeInfo.logicalDimensionMappings[cacheDimIdx]] += cacheIndices[cacheDimIdx] * packingShapeInfo.affinePerDimCoefficients[cacheDimIdx];
            }
            return unpackedIndices;
        }

        size_t FlattenIndices(const std::vector<int64_t>& affineCoefficients, const std::vector<int64_t>& cacheIndices)
        {
            size_t flattenedIndex = 0;
            for (size_t cacheDimIdx = 0; cacheDimIdx < affineCoefficients.size(); ++cacheDimIdx)
            {
                flattenedIndex += cacheIndices[cacheDimIdx] * affineCoefficients[cacheDimIdx];
            }
            return flattenedIndex;
        }

        template <typename ElementType>
        void PackBuffer(const MemoryLayout& inputLayout, mlir::DenseElementsAttr inputData, std::vector<ElementType>& packedBuffer, const ScheduleShardMapping& packingShapeInfo)
        {
            // We loop over the cache shape and compute the input element we should be copying using the affine dim coefficients and copy that

            std::vector<int64_t> currentCacheIndices(packingShapeInfo.shardSizes.size(), 0);
            RecursiveWalkPackedShape(packingShapeInfo, 0, currentCacheIndices, [&](const std::vector<int64_t>& cacheIndices) {
                auto unpackedIndices = MapPackedIndicesToUnpackedIndices(inputLayout.NumDimensions(), packingShapeInfo, cacheIndices);
                size_t flattenedPackedBufferIndex = FlattenIndices(packingShapeInfo.affineCoefficients, cacheIndices);
                MemoryCoordinates inputCoordinates{ unpackedIndices };
                if (inputLayout.IsOutOfBounds(inputCoordinates))
                {
                    packedBuffer[flattenedPackedBufferIndex] = static_cast<ElementType>(0);
                }
                else
                {
                    size_t flattenedUnpackedIndex = static_cast<size_t>(inputLayout.GetEntryOffset(inputCoordinates));
                    packedBuffer[flattenedPackedBufferIndex] = inputData.getValue<ElementType>({ flattenedUnpackedIndex });
                }
            });
        }

        template <typename ElementType>
        accera::value::Array EmbedPackedBuffer(mlir::OpBuilder& builder, mlir::Value constData, const std::string& packedBufferName)
        {
            auto packedLayout = accera::utilities::MemoryLayout{ accera::utilities::MemoryShape{ GetCacheShape() } };

            auto constantDataGlobalRef = mlir::dyn_cast_or_null<accera::ir::value::ReferenceGlobalOp>(constData.getDefiningOp());
            assert(constantDataGlobalRef != nullptr);
            auto constantDataAttr = constantDataGlobalRef.getGlobal().value().getValue().cast<mlir::DenseElementsAttr>();

            std::vector<ElementType> packedBufferInitData(GetCacheVolume());
            PackBuffer<ElementType>(_input.GetLayout(), constantDataAttr, packedBufferInitData, _shardMapping);

            accera::value::Array packedBuffer;
            {
                // Jump to top of module and create the global
                mlir::OpBuilder::InsertionGuard insertGuard(builder);

                auto insertionPoint = accera::ir::util::GetTerminalInsertPoint<vir::ValueModuleOp, vir::ModuleTerminatorOp>(_vModuleOp);
                builder.restoreInsertionPoint(insertionPoint);
                packedBuffer = accera::value::Array(packedBufferInitData, packedLayout, packedBufferName);
            }
            return packedBuffer;
        }

        void CreateWrapperFunction(mlir::OpBuilder& builder, const std::string& wrapperFnName)
        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);
            auto loc = GetLocation();

            auto insertionPoint = accera::ir::util::GetTerminalInsertPoint<vir::ValueModuleOp, vir::ModuleTerminatorOp>(_vModuleOp);
            builder.restoreInsertionPoint(insertionPoint);

            // Construct the arguments to the main function without the input that is being packed
            auto scheduleFuncOp = GetScheduleFuncOp();
            int targetArgIdx = 0;
            for (const auto& arg : scheduleFuncOp.getArguments())
            {
                if (arg == _mlirValueInput)
                {
                    break;
                }
                targetArgIdx++;
            }
            assert(targetArgIdx < scheduleFuncOp.getNumArguments());
            auto argsWithTargetRemoved = scheduleFuncOp.getType().getInputs().vec();
            argsWithTargetRemoved.erase(argsWithTargetRemoved.begin() + targetArgIdx);

            auto callingFnType = builder.getFunctionType(argsWithTargetRemoved, scheduleFuncOp.getType().getResults());

            vir::ValueFuncOp wrappingFn = builder.create<vir::ValueFuncOp>(loc, wrapperFnName + "_internal", callingFnType, ir::value::ExecutionTarget::CPU);

            builder.setInsertionPointToStart(&wrappingFn.body().front());

            auto wrapperFnArgs = wrappingFn.getArguments().vec();
            std::vector<mlir::Value> constantInjectedArgs;
            constantInjectedArgs.insert(constantInjectedArgs.begin(), wrapperFnArgs.begin(), wrapperFnArgs.end());

            auto globalScopeGlobalRef = mlir::dyn_cast_or_null<accera::ir::value::ReferenceGlobalOp>(_packedBufferMlirValue.getDefiningOp());
            auto localScopeGlobalRef = builder.create<accera::ir::value::ReferenceGlobalOp>(GetLocation(), globalScopeGlobalRef.getGlobal());
            // Re-view the packed buffer memref to match the function argument
            // TODO : update the function arguments to match the packed shape
            mlir::Value shapelessMemref = builder.create<mlir::memref::CastOp>(loc, localScopeGlobalRef, mlir::UnrankedMemRefType::get(GetElementType(), GetInputType().getMemorySpace()));
            auto reshapedMemref = builder.create<mlir::memref::CastOp>(loc, shapelessMemref, GetInputType());

            constantInjectedArgs.insert(constantInjectedArgs.begin() + targetArgIdx, reshapedMemref);
            auto launchFuncOp = builder.create<vir::LaunchFuncOp>(GetLocation(), scheduleFuncOp, constantInjectedArgs);

            if (launchFuncOp.getNumResults() > 0)
            {
                builder.create<vir::ReturnOp>(loc, launchFuncOp.getResults());
            }
            else
            {
                builder.create<vir::ReturnOp>(loc);
            }

            // Now create the Raw pointer API function for this wrapper function
            vir::CreateRawPointerAPIWrapperFunction(builder, wrappingFn, wrapperFnName);
        }

        accera::value::Array _packedBuffer;
        mlir::Value _packedBufferMlirValue;
    };

    //
    // Main class implementation
    //

    // Automatic caching version
    Cache::Cache(ScheduleOp schedule,
                 ViewAdapter value,
                 const std::optional<ScalarIndex>& keySliceIndex,
                 const std::optional<int64_t>& maxElements,
                 CacheIndexing mapping,
                 CacheAllocation allocation,
                 MemorySpace memorySpace,
                 ExecutionOptions execOptions)
    {
        std::optional<Index> keySlice;
        if (keySliceIndex.has_value())
        {
            keySlice = GetIndex(*keySliceIndex);
        }
        _impl = std::make_unique<AutomaticCacheImpl>(schedule, value, keySlice, maxElements, mapping, allocation, memorySpace, execOptions);
    }

    // Manual caching version
    Cache::Cache(ScheduleOp schedule,
                 std::variant<ViewAdapter, Cache*> value,
                 const std::optional<ScalarIndex>& keySliceIndex,
                 const std::optional<ScalarIndex>& triggerIndex,
                 const std::optional<int64_t>& maxElements,
                 const MemoryAffineCoefficients& memoryMap,
                 CacheIndexing mapping,
                 CacheAllocation allocation,
                 MemorySpace memorySpace,
                 ExecutionOptions execOptions)
    {
        std::optional<Index> keySlice;
        if (keySliceIndex.has_value())
        {
            keySlice = GetIndex(*keySliceIndex);
        }
        std::optional<Index> resolvedTriggerIndex;
        if (triggerIndex.has_value())
        {
            resolvedTriggerIndex = GetIndex(*triggerIndex);
        }
        if (std::holds_alternative<ViewAdapter>(value))
        {
            _impl = std::make_unique<ActiveBlockCacheImpl>(schedule, std::get<ViewAdapter>(value), keySlice, resolvedTriggerIndex, maxElements, memoryMap, mapping, allocation, memorySpace, execOptions);
        }
        else
        {
            _impl = std::make_unique<ActiveBlockCacheImpl>(schedule, std::get<Cache*>(value)->_impl.get(), keySlice, resolvedTriggerIndex, maxElements, memoryMap, mapping, allocation, memorySpace, execOptions);
        }
    }

    Cache::Cache(ScheduleOp schedule,
                 std::variant<ViewAdapter, Cache*> value,
                 const std::optional<ScalarIndex>& keySliceIndex,
                 const std::optional<ScalarIndex>& triggerIndex,
                 const std::optional<int64_t>& maxElements,
                 const DimensionOrder& dimOrder,
                 CacheIndexing mapping,
                 CacheAllocation allocation,
                 MemorySpace memorySpace,
                 ExecutionOptions execOptions)
    {
        std::optional<Index> keySlice;
        if (keySliceIndex.has_value())
        {
            keySlice = GetIndex(*keySliceIndex);
        }
        std::optional<Index> resolvedTriggerIndex;
        if (triggerIndex.has_value())
        {
            resolvedTriggerIndex = GetIndex(*triggerIndex);
        }

        if (std::holds_alternative<ViewAdapter>(value))
        {
            _impl = std::make_unique<ActiveBlockCacheImpl>(schedule, std::get<ViewAdapter>(value), keySlice, resolvedTriggerIndex, maxElements, dimOrder, mapping, allocation, memorySpace, execOptions);
        }
        else
        {
            _impl = std::make_unique<ActiveBlockCacheImpl>(schedule, std::get<Cache*>(value)->_impl.get(), keySlice, resolvedTriggerIndex, maxElements, dimOrder, mapping, allocation, memorySpace, execOptions);
        }
    }

    // Runtime-init caching version
    Cache::Cache(ScheduleOp schedule,
                 ViewAdapter value,
                 const std::string& packingFunctionName,
                 const std::string& packedBufferSizeFnName,
                 CacheIndexing mapping) :
        _impl(std::make_unique<RuntimeInitCacheImpl>(schedule, value, packingFunctionName, packedBufferSizeFnName, mapping))
    {
    }

    // Emit-time packed caching version
    Cache::Cache(ScheduleOp schedule,
                 ViewAdapter value,
                 ViewAdapter constantData,
                 const std::string& wrapperFnName,
                 const std::string& packedBufferName,
                 CacheIndexing mapping) :
        _impl(std::make_unique<EmitTimePackedCacheImpl>(schedule, value, constantData, wrapperFnName, packedBufferName, mapping))
    {
    }

    Cache::Cache(Cache&& other) noexcept :
        _impl(std::move(other._impl))
    {}

    Cache& Cache::operator=(Cache&& other) noexcept
    {
        if (this != &other)
        {
            using std::swap;
            swap(_impl, other._impl);
        }
        return *this;
    }

    Cache::~Cache() = default;

    Value Cache::GetBaseValue()
    {
        return _impl->GetBaseValue();
    }

} // namespace value
} // namespace accera
