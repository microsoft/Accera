////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "exec/ExecutionPlanOps.h"

#include "IRUtil.h"
#include "exec/ExecutionPlanAttributes.h"
#include "exec/ExecutionPlanDialect.cpp.inc"
#include "exec/ExecutionPlanEnums.cpp.inc"
#include "nest/Index.h"
#include "nest/LoopNestAttributes.h"
#include "nest/LoopNestOps.h"
#include "nest/TransformedDomain.h"
#include "value/ValueEnums.h"
#include "value/include/MLIREmitterContext.h"
#include <utilities/include/MemoryLayout.h>

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>

#include <iostream>
#include <numeric>
#include <set>
#include <stack>

using namespace accera::ir;
using namespace loopnest;
using namespace executionPlan;
using namespace mlir;

namespace accera::ir
{
namespace executionPlan
{
    using accera::ir::value::MemorySpace;

    namespace
    {
        [[maybe_unused]] int64_t GetCutoffEntry(const std::vector<int64_t>& cacheDimSizes, int64_t maxCacheElements)
        {
            // Loop from the innermost tier of the cache to the outermost to determine at which
            // cache dimension to cut off the cache such that at most maxCacheElements are cached
            int64_t accumulatedCacheSize = 1; // The innermost cache dimension shards are of size 1
            int64_t numDimensions = static_cast<int64_t>(cacheDimSizes.size());
            int64_t cutoffIdx = numDimensions - 1;
            for (int64_t loopIdx = 0; loopIdx < numDimensions; ++loopIdx)
            {
                int64_t cacheIdx = (numDimensions - 1) - loopIdx; // Work from back to front in the cache
                // Each cache dimension is a count of shards of the lower dimensions of the cache held by that
                // dimension. So to find the current size, start from the innermost dimension where each element
                // represents a shard of size 1 and multiply by shard counts moving outward through the dimensions
                accumulatedCacheSize *= cacheDimSizes[cacheIdx];
                if (accumulatedCacheSize > maxCacheElements)
                {
                    // Including this cache dimension put the cache size over the threshold, so don't update the
                    // cutoffIdx and let the previously examined dimension be the outermost dimension of the cache
                    break;
                }

                cutoffIdx = cacheIdx;
            }

            return cutoffIdx;
        }

        int64_t GetCutoffEntry(const std::vector<std::vector<Index>>& relevantIndices, ArrayRef<Index> cutoffIndices)
        {
            std::set<Index> remainingIndices(cutoffIndices.begin(), cutoffIndices.end());
            // Loop from the innermost tier of the cache to the outermost to determine at which
            // cache dimension to cut off the cache such all of the cutoffIndices are seen
            int64_t numDimensions = static_cast<int64_t>(relevantIndices.size());
            int64_t cutoffIdx = numDimensions - 1;
            for (int64_t loopIdx = 0; loopIdx < numDimensions; ++loopIdx)
            {
                if (remainingIndices.empty())
                {
                    break;
                }

                int64_t cacheIdx = (numDimensions - 1) - loopIdx; // Work from back to front in the cache
                // Each cache dimension is a count of shards of the lower dimensions of the cache held by that
                // dimension. So to find the current size, start from the innermost dimension where each element
                // represents a shard of size 1 and multiply by shard counts moving outward through the dimensions
                auto loopIndex = relevantIndices[cacheIdx].front();
                remainingIndices.erase(loopIndex);
                cutoffIdx = cacheIdx;
            }

            return cutoffIdx;
        }

        std::vector<Index> FlattenUnfoldedRelevantScheduleIndices(const std::vector<std::vector<Index>>& relevantScheduleIndices)
        {
            std::vector<Index> result;
            result.reserve(relevantScheduleIndices.size());
            for (const auto& indices : relevantScheduleIndices)
            {
                assert(indices.size() == 1 && "Folded relevant schedule indices aren't valid for flattening");
                result.push_back(indices.front());
            }
            return result;
        }

        std::vector<Index> GetRelevantCutoffEntryIndicesForKeySliceIndices(const std::vector<Index>& scheduleOrder, const std::vector<std::vector<Index>>& relevantScheduleIndices, const std::vector<Index>& cutoffEntryIndices)
        {
            // For each index in cutoffEntryIndices, find the index in relevantScheduleIndices that either
            // matches it or is the closest index that is deeper in the schedule and create a vector of all of these
            // indices and return it
            std::vector<Index> result;
            std::vector<Index> flattenedRelevantScheduleIndices = FlattenUnfoldedRelevantScheduleIndices(relevantScheduleIndices);
            result.reserve(cutoffEntryIndices.size());
            for (const auto& cutoffIndex : cutoffEntryIndices)
            {
                if (std::find(flattenedRelevantScheduleIndices.begin(), flattenedRelevantScheduleIndices.end(), cutoffIndex) != flattenedRelevantScheduleIndices.end())
                {
                    result.push_back(cutoffIndex);
                }
                else
                {
                    auto iter = std::find(scheduleOrder.begin(), scheduleOrder.end(), cutoffIndex);
                    assert(iter != scheduleOrder.end() && "Key slice index specified isn't part of the schedule");
                    auto nextInnerRelevantIdxIter = std::find_if(iter, scheduleOrder.end(), [&](const Index& scheduleIdx) {
                        return std::find(flattenedRelevantScheduleIndices.begin(), flattenedRelevantScheduleIndices.end(), scheduleIdx) != flattenedRelevantScheduleIndices.end();
                    });
                    assert(nextInnerRelevantIdxIter != scheduleOrder.end() && "Key slice is a single element"); // TODO : do we need to support this case?
                    result.push_back(*nextInnerRelevantIdxIter);
                }
            }
            return result;
        }
    } // namespace

    //
    // ExecutionPlanDialect
    //
    void ExecutionPlanDialect::initialize()
    {
        addOperations<
#define GET_OP_LIST
#include "exec/ExecutionPlanOps.cpp.inc"
            >();
        addAttributes<VectorizationInfoAttr>();
        addAttributes<ParallelizationInfoAttr>();
        addAttributes<TensorizationInfoAttr>();
        addAttributes<InPlaceUnrollInfoAttr>();
    }

    //
    // MakeCache utilities
    //

    void ComputeAffineCoefficients(ScheduleShardMapping& shardMapping)
    {
        assert(shardMapping.shardSizes.size() == shardMapping.logicalDimensionMappings.size());

        shardMapping.affinePerDimCoefficients.resize(shardMapping.logicalDimensionMappings.size(), 1);
        shardMapping.affineCoefficients.resize(shardMapping.logicalDimensionMappings.size(), 1);

        // The affineCoefficients are the coefficients to multiply the cache index value in each dimension by and sum
        // with other terms in that input dimension when computing the input value index for a given cache index
        // So compute those once we have the completed shape
        auto maxIter = std::max_element(shardMapping.logicalDimensionMappings.begin(), shardMapping.logicalDimensionMappings.end());
        int64_t largestMappedDim = *maxIter;
        std::vector<int64_t> currentCoefficientsPerDim(largestMappedDim + 1, 1);

        int64_t currentCoefficient = 1;
        for (size_t idx = 0; idx < shardMapping.logicalDimensionMappings.size(); ++idx)
        {
            // Loop from the inner part of the shape outwards
            size_t currentIdx = shardMapping.logicalDimensionMappings.size() - idx - 1;

            shardMapping.affineCoefficients[currentIdx] = currentCoefficient;
            shardMapping.affinePerDimCoefficients[currentIdx] = currentCoefficientsPerDim[shardMapping.logicalDimensionMappings[currentIdx]];

            currentCoefficient *= shardMapping.shardSizes[currentIdx];
            currentCoefficientsPerDim[shardMapping.logicalDimensionMappings[currentIdx]] *= shardMapping.shardSizes[currentIdx];
        }
    }

    // TODO - this isn't considering clamped split sizes when computing shapes
    ScheduleShardMapping ParseShardSizes(loopnest::ScheduleOp schedule, const std::vector<loopnest::Index>& inputIndices, bool matchBaseIndex)
    {
        // Walk the schedule and build a high-dimensional shape giving the shard sizes and dimension assignments
        ScheduleShardMapping result;

        auto scheduleOrder = schedule.getOrder();
        auto transformedDomainAttr = schedule.getDomain();
        auto domain = transformedDomainAttr.getValue();
        for (size_t idx = 0; idx < scheduleOrder.size(); ++idx)
        {
            auto& index = scheduleOrder[idx];
            for (auto dimensionIndex : domain.GetBaseIndices(index))
            {
                loopnest::Range indexRange = domain.GetIndexRange(index);

                std::vector<loopnest::Index>::const_iterator indexIter;
                if (matchBaseIndex)
                {
                    indexIter = std::find(inputIndices.begin(), inputIndices.end(), dimensionIndex);
                }
                else
                {
                    indexIter = std::find(inputIndices.begin(), inputIndices.end(), index);
                }
                if (indexIter != inputIndices.end())
                {
                    int64_t vecIdx = std::distance(inputIndices.begin(), indexIter);
                    result.logicalDimensionMappings.push_back(vecIdx);

                    // Record each split index (size / increment) to account for how many cache
                    // shards are represented at that level of the cache, and therefore how many
                    // elements should be in that dimension of the cache
                    result.shardSizes.push_back(indexRange.NumIterations());
                    result.relevantScheduleIndexPositions.push_back({ idx });
                    result.relevantScheduleIndices.push_back({ index });
                }
            }
        }
        ComputeAffineCoefficients(result);

        return result;
    }

    ScheduleShardMapping TrimUpToIndex(const ScheduleShardMapping& shardMapping, int64_t cutoffIdx)
    {
        ScheduleShardMapping result;
        std::copy(shardMapping.shardSizes.begin() + cutoffIdx, shardMapping.shardSizes.end(), std::back_inserter(result.shardSizes));
        std::copy(shardMapping.logicalDimensionMappings.begin() + cutoffIdx, shardMapping.logicalDimensionMappings.end(), std::back_inserter(result.logicalDimensionMappings));
        std::copy(shardMapping.relevantScheduleIndices.begin() + cutoffIdx, shardMapping.relevantScheduleIndices.end(), std::back_inserter(result.relevantScheduleIndices));
        std::copy(shardMapping.relevantScheduleIndexPositions.begin() + cutoffIdx, shardMapping.relevantScheduleIndexPositions.end(), std::back_inserter(result.relevantScheduleIndexPositions));
        return result;
    }

    ScheduleShardMapping CollapseSingleElementDimensions(const ScheduleShardMapping& shardMapping)
    {
        // e.g. if we have a cache shape [1, 64, 64] with input mapping [0, 1, 0],
        //      then elide the outer 1-element shard to produce
        //      [64, 64] with input mapping [1, 0]
        ScheduleShardMapping result;

        for (unsigned idx = 0; idx < shardMapping.shardSizes.size(); ++idx)
        {
            if (shardMapping.shardSizes[idx] > 1)
            {
                result.shardSizes.push_back(shardMapping.shardSizes[idx]);
                result.logicalDimensionMappings.push_back(shardMapping.logicalDimensionMappings[idx]);
                result.relevantScheduleIndices.push_back(shardMapping.relevantScheduleIndices[idx]);
                result.relevantScheduleIndexPositions.push_back(shardMapping.relevantScheduleIndexPositions[idx]);
            }
        }
        ComputeAffineCoefficients(result);

        return result;
    }

    ScheduleShardMapping CollapseRepeatedDimensions(const ScheduleShardMapping& shardMapping)
    {
        // e.g. if we have a cache shape [64, 32, 16, 4] with input mapping [0, 1, 1, 0] and affine coefficients [ 4, 16, 1, 1 ],
        //      then combine the inner dimension-1 mapped elements to produce
        //      [64, 512, 4] with input mapping [0, 1, 0] and affine coefficients [ 4, 1, 1 ]
        ScheduleShardMapping result;

        int64_t currentSize = shardMapping.shardSizes.front();
        int64_t currentMapping = shardMapping.logicalDimensionMappings.front();
        result.relevantScheduleIndices.push_back(shardMapping.relevantScheduleIndices.front());
        result.relevantScheduleIndexPositions.push_back(shardMapping.relevantScheduleIndexPositions.front());

        for (unsigned idx = 1; idx < shardMapping.logicalDimensionMappings.size(); ++idx)
        {
            if (shardMapping.logicalDimensionMappings[idx] == currentMapping)
            {
                currentSize *= shardMapping.shardSizes[idx];
                result.relevantScheduleIndices.back().emplace_back(shardMapping.relevantScheduleIndices[idx].front());
                result.relevantScheduleIndexPositions.back().emplace_back(shardMapping.relevantScheduleIndexPositions[idx].front());
            }
            else
            {
                result.shardSizes.push_back(currentSize);
                result.logicalDimensionMappings.push_back(currentMapping);

                currentSize = shardMapping.shardSizes[idx];
                currentMapping = shardMapping.logicalDimensionMappings[idx];
                result.relevantScheduleIndices.push_back(shardMapping.relevantScheduleIndices[idx]);
                result.relevantScheduleIndexPositions.push_back(shardMapping.relevantScheduleIndexPositions[idx]);
            }
        }
        result.shardSizes.push_back(currentSize);
        result.logicalDimensionMappings.push_back(currentMapping);
        ComputeAffineCoefficients(result);

        return result;
    }

    ScheduleShardMapping GetScheduleShardMapping(loopnest::ScheduleOp schedule, const std::vector<loopnest::Index>& accessLogicalIndices)
    {
        // Parse the ordered indices in the schedule and compare against the TransformedDomain and the
        // schedule kernel usage to determine which of the split indices in the schedule are relevant for the given
        // input.
        // The cache shape will follow the order in which the schedule will operate on elements in the input

        auto shardMapping = ParseShardSizes(schedule, accessLogicalIndices, true);

        assert(shardMapping.shardSizes.size() == shardMapping.logicalDimensionMappings.size() && "Must have one dimension mapping per cache dimension");

        // Collapse repeated dimensions in the cache to simplify operations with it
        ScheduleShardMapping simplifiedShardMapping = CollapseRepeatedDimensions(shardMapping);

        // Collapse cache dimensions with only a single shard
        simplifiedShardMapping = CollapseSingleElementDimensions(simplifiedShardMapping);

        return simplifiedShardMapping;
    }

    mlir::AffineExpr RecursiveBinOpToAffineExprHelper(mlir::Value currentValue, std::vector<loopnest::Index>& orderedAccessIndices)
    {
        auto currentOp = currentValue.getDefiningOp();
        if (!currentOp)
        {
            return {};
        }

        OpBuilder builder{ currentOp };

        if (auto binOp = mlir::dyn_cast<value::BinOp>(currentOp))
        {
            auto lhsResults = RecursiveBinOpToAffineExprHelper(binOp.lhs(), orderedAccessIndices);
            auto rhsResults = RecursiveBinOpToAffineExprHelper(binOp.rhs(), orderedAccessIndices);
            mlir::AffineExpr expr;
            switch (binOp.getPredicate())
            {
            case value::BinaryOpPredicate::ADD:
                expr = lhsResults + rhsResults;
                break;
            case value::BinaryOpPredicate::SUB:
                expr = lhsResults - rhsResults;
                break;
            case value::BinaryOpPredicate::MUL:
                expr = lhsResults * rhsResults;
                break;
            case value::BinaryOpPredicate::DIV:
                expr = lhsResults.floorDiv(rhsResults);
                break;
            case value::BinaryOpPredicate::MOD:
                expr = lhsResults % rhsResults;
                break;
            default:
                assert(false && "Unsupported Binary Op in index computation");
            }
            return expr;
        }
        else if (auto symbolicIndexOp = mlir::dyn_cast<loopnest::SymbolicIndexOp>(currentOp))
        {
            auto index = symbolicIndexOp.getValue();
            auto accessIndexPosIter = std::find(orderedAccessIndices.begin(), orderedAccessIndices.end(), index);
            size_t accessIndexPos = std::distance(orderedAccessIndices.begin(), accessIndexPosIter);
            if (accessIndexPosIter == orderedAccessIndices.end())
            {
                orderedAccessIndices.push_back(index);
            }
            return builder.getAffineDimExpr(accessIndexPos);
        }
        else if (auto constantOp = mlir::dyn_cast<arith::ConstantIntOp>(currentOp))
        {
            return builder.getAffineConstantExpr(constantOp.value());
        }
        // we expect value CastOps and/or IndexCastOps when the DSL code indexes into an array with a constant integer
        // e.g. A[0,0] will produce an index cast from int64 to index for each 0
        else if (auto indexCast = mlir::dyn_cast_or_null<arith::IndexCastOp>(currentOp))
        {
            return RecursiveBinOpToAffineExprHelper(indexCast.getIn(), orderedAccessIndices);
        }
        else if (auto valueCast = mlir::dyn_cast_or_null<accera::ir::value::CastOp>(currentOp); valueCast.result().getType().isIndex())
        {
            return RecursiveBinOpToAffineExprHelper(valueCast.source(), orderedAccessIndices);
        }
        llvm_unreachable("Unsupported op type used for indexing into a value that is being cached");
    }

    std::vector<loopnest::Index> GetBaseIndicesCombinedToMakeValue(mlir::Value combinedIndex, const loopnest::TransformedDomain& domain)
    {
        // Only supports value::BinOps and mlir::AffineApplyOps for combinations currently
        // And only supports loopnest::SymbolicIndexOps and mlir::arith::ConstantOps as the base values

        std::vector<loopnest::Index> results;
        std::stack<mlir::Value> valuesToFollow;
        valuesToFollow.push(combinedIndex);
        while (!valuesToFollow.empty())
        {
            auto currentValue = valuesToFollow.top();
            valuesToFollow.pop();
            auto currentOp = currentValue.getDefiningOp();
            if (auto binOp = mlir::dyn_cast_or_null<value::BinOp>(currentOp))
            {
                valuesToFollow.push(binOp.lhs());
                valuesToFollow.push(binOp.rhs());
            }
            else if (auto affineApplyOp = mlir::dyn_cast_or_null<mlir::AffineApplyOp>(currentOp))
            {
                for (auto operand : affineApplyOp.getOperands())
                {
                    valuesToFollow.push(operand);
                }
            }
            else if (auto symbolicIndexOp = mlir::dyn_cast_or_null<loopnest::SymbolicIndexOp>(currentOp))
            {
                auto index = symbolicIndexOp.getValue();
                auto baseIndices = domain.GetBaseIndices(index);
                for (const auto& baseIndex : baseIndices)
                {
                    if (std::find(results.begin(), results.end(), baseIndex) == results.end())
                    {
                        results.push_back(baseIndex);
                    }
                }
            }
            else if (auto constantOp = mlir::dyn_cast_or_null<arith::ConstantOp>(currentOp))
            {
                // Nothing to do currently
                // TODO : when we support multiple lanes based on different combinations of indices or constants
                // we'll need to store this constant so that we can check if we see the same constant or a different
                // constant when selecting a lane mapping
            }
            else
            {
                assert(false && "Unsupported op type used for indexing into a value that is being cached");
            }
        }
        return results;
    }

    // Compute the AffineMap that maps from access indices to cache buffer position using the given MemoryAffineCoefficients
    // e.g. with MemoryAffineCoefficients = { coefficients = [ 1, 2, 3 ], offset = 4 }
    //          compute (d0, d1, d2) -> ( d0*1 + d1*2 + d2*3 + 4 )
    mlir::AffineMap ComputeFlatAffineMapFromAffineCoefficients(OpBuilder& builder, const utilities::MemoryAffineCoefficients& affineMapping)
    {
        mlir::AffineExpr flatAffinePositionExpr = builder.getAffineConstantExpr(affineMapping.offset);
        for (size_t logicalAccessIndexPos = 0; logicalAccessIndexPos < affineMapping.coefficients.size(); ++logicalAccessIndexPos)
        {
            auto coefficientScaledIndex = builder.getAffineDimExpr(logicalAccessIndexPos) * affineMapping.coefficients[logicalAccessIndexPos];
            flatAffinePositionExpr = flatAffinePositionExpr + coefficientScaledIndex;
        }
        return mlir::AffineMap::get(affineMapping.coefficients.size(), 0, flatAffinePositionExpr, builder.getContext());
    }

    // TODO : de-dupe utility code with GetAccessExpressions()
    std::vector<loopnest::Index> GetAccessBaseIndices(mlir::Value input, loopnest::ScheduleOp schedule)
    {
        // Find uses of this input inside of schedule kernels
        auto scheduledKernels = schedule.getKernels();

        std::vector<loopnest::Index> orderedAccessIndices;

        auto transformedDomainAttr = schedule.getDomain();
        auto domain = transformedDomainAttr.getValue();

        for (auto& scheduledKernel : scheduledKernels)
        {
            auto foundKernelOp = util::FindOpWithSymbolName(scheduledKernel.getKernel(), schedule);
            auto kernelOp = mlir::dyn_cast_or_null<KernelOp>(foundKernelOp);
            assert(kernelOp != nullptr);
            auto kernelBlock = kernelOp.getBody();
            for (auto userOp : input.getUsers())
            {
                auto ancestorOp = kernelBlock->findAncestorOpInBlock(*userOp);
                if (ancestorOp == nullptr)
                    // This user op is not in the kernel
                    continue;

                // This operation is either inside the kernel block directly
                // or has an ancestor op inside the kernel block

                // We need to specifically enable op types here since we need to understand
                // the access pattern.
                // TODO : create an interface or make accv.slice adhere to an existing interface that will enable us to support more op types here
                auto sliceOp = mlir::dyn_cast_or_null<value::SliceOp>(userOp);
                assert(sliceOp != nullptr && "Only supports SliceOps into values for caching currently");
                value::SliceOp::Adaptor sliceAdaptor{ sliceOp };
                [[maybe_unused]] auto inputMemRefType = sliceOp.getSourceMemRefType();
                auto sliceDimensions = util::ConvertArrayAttrToIntVector(sliceAdaptor.sliceDimensions());
                assert(inputMemRefType.getRank() > 0 && sliceDimensions.size() == (size_t)inputMemRefType.getRank() && "Only support single-element slicing currently");
                std::vector<mlir::Value> sliceOffsets{ sliceAdaptor.offsets().begin(), sliceAdaptor.offsets().end() };
                for (const auto& offsetValue : sliceOffsets)
                {
                    RecursiveBinOpToAffineExprHelper(offsetValue, orderedAccessIndices);
                }
            }
        }
        return orderedAccessIndices;
    }

    std::pair<std::vector<mlir::AffineExpr>, std::vector<loopnest::Index>> GetAccessExpressions(mlir::Value input, loopnest::ScheduleOp schedule)
    {
        // Find uses of this input inside of schedule kernels
        auto scheduledKernels = schedule.getKernels();
        std::vector<mlir::AffineExpr> accessExpressions;

        std::vector<loopnest::Index> orderedAccessIndices;

        auto transformedDomainAttr = schedule.getDomain();
        auto domain = transformedDomainAttr.getValue();

        for (auto& scheduledKernel : scheduledKernels)
        {
            auto foundKernelOp = util::FindOpWithSymbolName(scheduledKernel.getKernel(), schedule);
            auto kernelOp = mlir::dyn_cast_or_null<KernelOp>(foundKernelOp);
            assert(kernelOp != nullptr);
            auto kernelBlock = kernelOp.getBody();
            for (auto userOp : input.getUsers())
            {
                auto ancestorOp = kernelBlock->findAncestorOpInBlock(*userOp);
                if (ancestorOp == nullptr)
                    // This user op is not in the kernel
                    continue;

                // This operation is either inside the kernel block directly
                // or has an ancestor op inside the kernel block

                // We need to specifically enable op types here since we need to understand
                // the access pattern.
                // TODO : create an interface or make accv.slice adhere to an existing interface that will enable us to support more op types here
                auto sliceOp = mlir::dyn_cast_or_null<value::SliceOp>(userOp);
                assert(sliceOp != nullptr && "Only supports SliceOps into values for caching currently");
                value::SliceOp::Adaptor sliceAdaptor{ sliceOp };
                [[maybe_unused]] auto inputMemRefType = sliceOp.getSourceMemRefType();
                auto sliceDimensions = util::ConvertArrayAttrToIntVector(sliceAdaptor.sliceDimensions());
                assert(inputMemRefType.getRank() > 0 && sliceDimensions.size() == (size_t)inputMemRefType.getRank() && "Only support single-element slicing currently");
                std::vector<mlir::Value> sliceOffsets{ sliceAdaptor.offsets().begin(), sliceAdaptor.offsets().end() };

                // Find the SymbolicIndexOps and their corresponding loopnest Indices used in each dimension offset
                // TODO : support "lane" duplication for different combinations of the same indices when accessing the buffer
                //        e.g. A[i, j + k] and A[i, j - k] in the same kernel
                std::vector<mlir::AffineExpr> currentAccessExpressions;
                std::vector<loopnest::Index> currentOrderedAccessIndices;
                for (const auto& offsetValue : sliceOffsets)
                {
                    auto accessExpr = RecursiveBinOpToAffineExprHelper(offsetValue, currentOrderedAccessIndices);
                    currentAccessExpressions.push_back(accessExpr);
                    // TODO : support multiple caches when there are multiple access patterns on input arrays?
                }

                if (!accessExpressions.empty())
                {
                    // Currently only supports one access pattern in the kernel
                    [[maybe_unused]] bool accessPatternMatches = accessExpressions.size() == currentAccessExpressions.size() &&
                                                                 std::equal(accessExpressions.begin(), accessExpressions.end(), currentAccessExpressions.begin());
                    assert(accessPatternMatches && "Only supports one access pattern per cached buffer");
                }
                else
                {
                    accessExpressions = currentAccessExpressions;
                }
                if (!orderedAccessIndices.empty())
                {
                    // Currently only supports one access pattern in the kernel
                    [[maybe_unused]] bool accessIndicesMatch = orderedAccessIndices.size() == currentOrderedAccessIndices.size() &&
                                                               std::equal(orderedAccessIndices.begin(), orderedAccessIndices.end(), currentOrderedAccessIndices.begin());
                    assert(accessIndicesMatch && "Only supports one access pattern per cached buffer");
                }
                else
                {
                    orderedAccessIndices = currentOrderedAccessIndices;
                }
            }
        }
        return std::make_pair(accessExpressions, orderedAccessIndices);
    }

    CacheInfo ComputeCacheIndexInfo(
        mlir::OpBuilder& builder,
        mlir::Value input,
        ScheduleOp schedule,
        const std::optional<loopnest::Index>& keySliceIndex,
        const std::optional<loopnest::Index>& triggerIndex,
        std::optional<int64_t> overrideShardMappingCutoffIdx,
        bool matchBaseIndicesWhenFillingShardMapping)
    {
        auto transformedDomainAttr = schedule.getDomain();
        auto domain = transformedDomainAttr.getValue();

        auto accessBaseIndices = GetAccessBaseIndices(input, schedule);

        // to determine the active block for the given key slice and schedule kernel
        // we want to use MLIR's MemRefRegion from mlir\include\mlir\Analysis\Utils.h
        // but that requires that the loopnest is already formed and in the affine dialect
        // so we compute the cache region here and add mapping metadata but move computing
        // the active block to a lowering pass

        auto shardMapping = ParseShardSizes(schedule, accessBaseIndices, matchBaseIndicesWhenFillingShardMapping);
        std::vector<Index> activeCacheCutoffEntryIndices = keySliceIndex.has_value() ? std::vector<Index>{ *keySliceIndex } : accessBaseIndices;
        std::vector<Index> cacheTriggerCutoffEntryIndices = triggerIndex.has_value() ? std::vector<Index>{ *triggerIndex } : accessBaseIndices;

        // TODO : move cutoff index computation to lowering passes so we can enable max element caching for manual caches
        int64_t shardMappingCutoffIdx;
        if (overrideShardMappingCutoffIdx.has_value())
        {
            shardMappingCutoffIdx = *overrideShardMappingCutoffIdx;
        }
        else
        {
            auto relevantCutoffEntryIndices = GetRelevantCutoffEntryIndicesForKeySliceIndices(schedule.getOrder(), shardMapping.relevantScheduleIndices, activeCacheCutoffEntryIndices);
            shardMappingCutoffIdx = GetCutoffEntry(shardMapping.relevantScheduleIndices, relevantCutoffEntryIndices);
        }

        auto cutoffShardMapping = TrimUpToIndex(shardMapping, shardMappingCutoffIdx);

        assert(cutoffShardMapping.shardSizes.size() == cutoffShardMapping.logicalDimensionMappings.size() && "Must have one dimension mapping per cache dimension");

        // Accumulate the full relevant schedule indices as a single flattened vector
        // (an index is "relevant" if it is used to index into the input value)
        // And the corresponding index ops for the schedule
        std::vector<loopnest::Index> flatRelevantScheduleIndices;
        std::vector<mlir::Value> relevantScheduleIndexOps;
        flatRelevantScheduleIndices.reserve(shardMapping.relevantScheduleIndices.size());
        relevantScheduleIndexOps.reserve(shardMapping.relevantScheduleIndices.size());
        for (const auto& relevantIndices : shardMapping.relevantScheduleIndices)
        {
            assert(relevantIndices.size() == 1 && "Should only have one relevant index per dimension before index folding");
            flatRelevantScheduleIndices.push_back(relevantIndices.front());
            auto op = schedule.getOrCreateSymbolicIndex(builder, relevantIndices.front());
            relevantScheduleIndexOps.push_back(op.getResult());
        }

        // Store the IndexRanges of the cache active region relevant indices, and the
        // base indices for the relevant cache region base indices while we have the
        // ScheduleOp in hand
        // We'll want this info to reconstruct a comparable loopnest to copy data to/from the cache
        // And the base indices are used for checking dim size for clamping purposes
        std::vector<loopnest::IndexRange> cacheRegionRelevantScheduleIndexRanges;
        std::vector<std::vector<loopnest::Index>> cacheRegionBaseIndices;
        cacheRegionRelevantScheduleIndexRanges.reserve(cutoffShardMapping.relevantScheduleIndices.size());
        cacheRegionBaseIndices.reserve(cutoffShardMapping.relevantScheduleIndices.size());
        for (const auto& relevantIndices : cutoffShardMapping.relevantScheduleIndices)
        {
            assert(relevantIndices.size() == 1 && "Should only have one relevant index per dimension before index folding");
            cacheRegionRelevantScheduleIndexRanges.push_back(IndexRange(relevantIndices.front(), domain.GetIndexRange(relevantIndices.front())));
            std::vector<loopnest::Index> currentBaseIndices;
            for (const auto& relevantIndex : relevantIndices)
            {
                auto baseIndices = domain.GetBaseIndices(relevantIndex);
                currentBaseIndices.insert(currentBaseIndices.end(), baseIndices.begin(), baseIndices.end());
            }
            cacheRegionBaseIndices.push_back(currentBaseIndices);
        }

        // Keep track of the relevant schedule indices that are outside of the cache region so we can index
        // into the right base input position
        std::vector<loopnest::SymbolicIndexOp> externalRelevantScheduleIndexOps;
        size_t numExternalRelevantIndices = relevantScheduleIndexOps.size() - cacheRegionRelevantScheduleIndexRanges.size();
        externalRelevantScheduleIndexOps.reserve(numExternalRelevantIndices);
        for (size_t idx = 0; idx < numExternalRelevantIndices; ++idx)
        {
            auto index = shardMapping.relevantScheduleIndices[idx].front();
            externalRelevantScheduleIndexOps.push_back(schedule.getOrCreateSymbolicIndex(builder, index));
        }

        // The shardMappingCutoffIdx is the cutoff in the order of relevant indices in this schedule given the accessDimensions.
        // Map this shardMappingCutoffIdx back to the full schedule ordering (including dimensions we don't care about for this cache)
        // to get the scheduleCutoffIdx

        size_t cacheKeySliceCutoffIdx;
        if (shardMappingCutoffIdx == 0)
        {
            // cutoffIdx is at the outermost level of schedule that is relevant this input, so put the cacheKeySliceCutoffIdx at 0
            // so the cache copying / reducing happens at the outermost level in the schedule
            cacheKeySliceCutoffIdx = 0;
        }
        else
        {
            // shardMappingCutoffIdx is not at the outermost level of the schedule relevant to this input, so put the cacheKeySliceCutoffIdx
            // at the level of the full schedule that the cutoff occurs at in the relevant schedule indices
            cacheKeySliceCutoffIdx = shardMapping.relevantScheduleIndexPositions[shardMappingCutoffIdx].front();
        }

        auto scheduleOrder = schedule.getOrder();

        CacheInfo result;

        result.cacheIndex = scheduleOrder[cacheKeySliceCutoffIdx];
        result.triggerIndex = triggerIndex.has_value() ? triggerIndex : result.cacheIndex;
        result.fullShardMapping = shardMapping;
        result.shardMapping = cutoffShardMapping;
        result.accessBaseIndices = accessBaseIndices;
        result.fullRelevantScheduleIndices = llvm::SmallVector<mlir::Value, 4>(llvm::iterator_range(relevantScheduleIndexOps.begin(), relevantScheduleIndexOps.end()));
        result.externalRelevantScheduleIndices = llvm::SmallVector<mlir::Value, 4>(llvm::iterator_range(externalRelevantScheduleIndexOps.begin(), externalRelevantScheduleIndexOps.end()));
        result.cacheRegionRelevantScheduleIndexRanges = cacheRegionRelevantScheduleIndexRanges;
        result.cacheRegionBaseIndices = cacheRegionBaseIndices;

        return result;
    }

    CacheInfo MakeAutomaticCacheInfoCommon(mlir::OpBuilder& builder, mlir::Value input, ScheduleOp schedule, const CacheAllocation& cacheAllocation, int64_t shardMappingCutoffIdx, const MemorySpace& memorySpace, const std::optional<loopnest::Index>& keySliceIndex, bool matchBaseIndicesWhenFillingShardMapping)
    {
        // Parse the ordered indices in the schedule and compare against the TransformedDomain and the
        // schedule kernel usage to determine which of the split indices in the schedule are relevant for the given
        // input.
        // The cache shape will follow the order in which the schedule will operate on elements in the input
        auto transformedDomainAttr = schedule.getDomain();
        auto domain = transformedDomainAttr.getValue();

        auto cacheInfo = ComputeCacheIndexInfo(builder, input, schedule, keySliceIndex, keySliceIndex, shardMappingCutoffIdx, matchBaseIndicesWhenFillingShardMapping);
        auto flatRelevantScheduleIndices = FlattenUnfoldedRelevantScheduleIndices(cacheInfo.fullShardMapping.relevantScheduleIndices);

        auto inputType = input.getType();
        assert(inputType.isa<MemRefType>());
        auto inputMemRefType = inputType.cast<MemRefType>();
        auto inputElementType = inputMemRefType.getElementType();

        auto [accessExpressions, accessIndices] = GetAccessExpressions(input, schedule);

        // Collapse/fold repeated base input dimensions or single element dimensions in
        // the cache.
        // E.g. suppose a cache shape (16, 32, 4, 2, 8) with base input mapping (1, 0, 0, 1, 1)
        //      (this is 128x256 B tile caching from MLAS MatMul), then we can fold the repeated
        //      0 and 1 dimensions to get { (16, 128, 16) , (1, 0, 1) }

        // Collapse repeated dimensions in the cache to simplify operations with it
        ScheduleShardMapping foldedCutoffShardMapping = CollapseRepeatedDimensions(cacheInfo.shardMapping);

        // Collapse cache dimensions with only a single shard
        foldedCutoffShardMapping = CollapseSingleElementDimensions(foldedCutoffShardMapping);

        // Create a map that maps from base relevant indices to folded cutoff indices for folded cache access
        std::vector<mlir::AffineExpr> cacheDimFoldingExprs;
        cacheDimFoldingExprs.reserve(foldedCutoffShardMapping.relevantScheduleIndices.size());
        // TODO : de-dupe with cache.cpp
        for (const auto& relevantIndices : foldedCutoffShardMapping.relevantScheduleIndices)
        {
            assert(!relevantIndices.empty());

            mlir::AffineExpr currentExpr = builder.getAffineConstantExpr(0);

            for (auto relevantIndex : relevantIndices)
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
                auto range = domain.GetIndexRange(relevantIndex);
                currentExpr = currentExpr * range.NumIterations();

                // Get the position of this index relative to the other relevant base indices
                auto idxPosIter = std::find(flatRelevantScheduleIndices.begin(), flatRelevantScheduleIndices.end(), relevantIndex);
                assert(idxPosIter != flatRelevantScheduleIndices.end());
                auto scheduleIndexPosition = std::distance(flatRelevantScheduleIndices.begin(), idxPosIter);

                auto dimExpr = builder.getAffineDimExpr(scheduleIndexPosition);
                auto diffExpr = dimExpr - range.Begin();
                auto divExpr = diffExpr.floorDiv(range.Increment());
                currentExpr = currentExpr + divExpr;
            }

            cacheDimFoldingExprs.push_back(currentExpr);
        }
        auto relevantScheduleIndexToCachePositionMap = mlir::AffineMap::get(flatRelevantScheduleIndices.size(), 0, cacheDimFoldingExprs, builder.getContext());

        // Now construct the mappings from the relevant schedule indices to input data position in the cache region
        // This map will be used to copy data in and out of the cache by creating a loop structure over the relevant indices of the cache region
        // and using this mapping to determine the correct input element
        // This mapping also needs to account for whatever operations went into forming the input index

        // the accessExpressions and accessIndices deal in base indices, we need to figure out the mapping from relevant schedule indices -> relevant base indices
        // then compose that map with the access maps to get the relevant schedule indices -> input position map
        // E.g. Suppose we have:
        //      - indices i, j, k, l
        //      - matrix A
        //      - kernel access A[i + l, k], so the relevant base indices are { i, l, k }
        //      - split i, j, k, l each once and set order { iOuter, jOuter, kOuter, lOuter, i, j, k, l }
        //      - cache A at index lOuter (inclusive)
        //      Then:
        //      - we should expect to have a 4-D cache since there are 4 relevant non-collapsable split indices (lOuter, i, k, l)
        //      - cache region is { lOuter, i, j, k, l }
        //      - relevant base indices -> input position map is (d0, d1, d2) -> (d0 + d1, d2), d0 = i, d1 = l, d2 = k
        //      - full schedule indices are { iOuter, jOuter, kOuter, lOuter,  i,  j,  k,  l }
        //                                        d0,     d1,     d2,     d3, d4, d5, d6, d7
        //      - full schedule indices -> base indices map is (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0 + d4, d1 + d5, d2 + d6, d3 + d7)
        //      - relevant schedule indices are { iOuter, kOuter, lOuter,  i,  k,  l }
        //                                            d0,     d1,     d2, d3, d4, d5
        //      - relevant schedule indices -> cache position map is (d0, d1, d2, d3, d4, d5) -> (d2 / lSplit, d3, d4, d5)
        //      - relevant schedule indices -> relevant base indices map is (d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d2 + d5)
        //      - relevant schedule indices -> base indices map is a composition of the relevant schedule indices -> relevant base indices map
        //              and the relevant base indices -> input position map, which is:
        //              (d0, d1, d2, d3, d4, d5) -> (d0 + d3 + d1 + d4, d2 + d5)

        // Compute the relevant schedule indices -> input position map
        // E.g. for A[i + l, k] above, the base indices -> input position map is (d0, d1, d2, d3) -> (d0 + d3, d2)
        //      and the full schedule indices -> base indices map is (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0 + d4, d1 + d5, d2 + d6, d3 + d7)
        //      so the composition of these maps gives (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0 + d4 + d3 + d7, d2 + d6)

        // Compute relevant schedule indices -> base indices:
        std::vector<mlir::AffineExpr> relevantScheduleIndicesToBaseIndicesExprs(accessIndices.size(), builder.getAffineConstantExpr(0));
        for (size_t scheduleIndexPos = 0; scheduleIndexPos < flatRelevantScheduleIndices.size(); ++scheduleIndexPos)
        {
            auto& relevantScheduleIndex = flatRelevantScheduleIndices[scheduleIndexPos];
            auto baseIndices = domain.GetBaseIndices(relevantScheduleIndex);
            auto scheduleIndexDimExpr = builder.getAffineDimExpr(scheduleIndexPos);
            for (auto& baseIndex : baseIndices)
            {
                auto baseIndexPosIter = std::find(accessIndices.begin(), accessIndices.end(), baseIndex);
                assert(baseIndexPosIter != accessIndices.end());
                auto baseIndexPos = std::distance(accessIndices.begin(), baseIndexPosIter);
                relevantScheduleIndicesToBaseIndicesExprs[baseIndexPos] = relevantScheduleIndicesToBaseIndicesExprs[baseIndexPos] + scheduleIndexDimExpr;
            }
        }
        auto relevantScheduleIndicesToRelevantBaseIndicesMap = mlir::AffineMap::get(flatRelevantScheduleIndices.size(), 0, relevantScheduleIndicesToBaseIndicesExprs, builder.getContext());

        // Assemble the relevant base indices -> input position map from the derived access expressions
        auto relevantBaseIndicesToInputPositionMap = mlir::AffineMap::get(accessIndices.size(), 0, accessExpressions, builder.getContext());

        // Compose the maps to produce a mapping from relevant schedule indices to input position
        auto relevantScheduleIndicesToInputPositionMap = relevantBaseIndicesToInputPositionMap.compose(relevantScheduleIndicesToRelevantBaseIndicesMap);

        CacheAccessMaps accessMaps;

        accessMaps.relevantIndicesToActiveElementCache = relevantScheduleIndexToCachePositionMap;
        accessMaps.relevantIndicesToInput = relevantScheduleIndicesToInputPositionMap;

        int memoryLocation = 0;

        switch (memorySpace)
        {
        case MemorySpace::Global:
            memoryLocation = 0; //Todo: figure what api to call
            break;
        default:
            [[fallthrough]];
        case MemorySpace::None:
            [[fallthrough]];
        case MemorySpace::Shared:
            memoryLocation = gpu::GPUDialect::getWorkgroupAddressSpace();
            break;
        case MemorySpace::Private:
            memoryLocation = gpu::GPUDialect::getPrivateAddressSpace();
            break;
        }

        if (memoryLocation >= 0)
        {
            cacheInfo.cacheType = MemRefType::get(foldedCutoffShardMapping.shardSizes, inputElementType, {}, memoryLocation);
        }
        else
        {
            cacheInfo.cacheType = MemRefType::get(foldedCutoffShardMapping.shardSizes, inputElementType);
        }
        cacheInfo.cacheAllocation = cacheAllocation;
        cacheInfo.accessMaps = accessMaps;
        cacheInfo.activeBlockCache = false;
        cacheInfo.shardMapping = foldedCutoffShardMapping;

        return cacheInfo;
    }

    CacheInfo MakeAutomaticCacheInfo(mlir::OpBuilder& builder, mlir::Value input, CacheAllocation cacheAllocation, loopnest::ScheduleOp schedule, const std::optional<loopnest::Index>& outermostIncludedSplitIndex, const std::optional<int64_t>& maxElements, MemorySpace memorySpace)
    {
        auto [accessExpressions, accessIndices] = GetAccessExpressions(input, schedule);

        auto shardMapping = ParseShardSizes(schedule, accessIndices, outermostIncludedSplitIndex.has_value());
        std::vector<Index> cutoffEntryIndices = outermostIncludedSplitIndex.has_value() ? std::vector<Index>{ *outermostIncludedSplitIndex } : accessIndices;

        auto relevantCutoffEntryIndices = GetRelevantCutoffEntryIndicesForKeySliceIndices(schedule.getOrder(), shardMapping.relevantScheduleIndices, cutoffEntryIndices);
        auto cutoffIdx = maxElements.has_value() ? GetCutoffEntry(shardMapping.shardSizes, *maxElements) : GetCutoffEntry(shardMapping.relevantScheduleIndices, relevantCutoffEntryIndices);

        return MakeAutomaticCacheInfoCommon(builder, input, schedule, cacheAllocation, cutoffIdx, memorySpace, outermostIncludedSplitIndex, outermostIncludedSplitIndex.has_value());
    }

    // Simplified version of MakeAutomaticCacheInfo that represents a cache of the entire input for the given schedule
    CacheInfo MakeFullBufferAutomaticCacheInfo(mlir::OpBuilder& builder, mlir::Value input, CacheAllocation cacheAllocation, loopnest::ScheduleOp schedule, MemorySpace memorySpace)
    {
        return MakeAutomaticCacheInfoCommon(builder, input, schedule, cacheAllocation, 0 /*shardMappingCutoffIdx*/, memorySpace, std::nullopt, true /* matchBaseIndicesWhenFillingShardMapping */);
    }

    CacheInfo MakeManualCacheInfo(mlir::OpBuilder& builder,
                                  mlir::Value input,
                                  CacheAllocation cacheAllocation,
                                  loopnest::ScheduleOp schedule,
                                  const std::optional<accera::value::ValueType>& elementType,
                                  const std::optional<loopnest::Index>& keySliceIndex,
                                  const std::optional<loopnest::Index>& triggerIndex,
                                  const std::optional<int64_t>& maxElements,
                                  const std::variant<utilities::MemoryAffineCoefficients, utilities::DimensionOrder>& cacheMappingInfo,
                                  MemorySpace memorySpace)
    {
        CacheInfo cacheInfo;
        if (maxElements.has_value())
        {
            cacheInfo.maxElementBudget = *maxElements;
        }
        else
        {
            cacheInfo = ComputeCacheIndexInfo(builder, input, schedule, keySliceIndex, triggerIndex, std::nullopt, keySliceIndex.has_value());
        }
        CacheAccessMaps accessMaps;

        if (std::holds_alternative<utilities::MemoryAffineCoefficients>(cacheMappingInfo))
        {
            accessMaps.coefficients = std::get<utilities::MemoryAffineCoefficients>(cacheMappingInfo);
            cacheInfo.dimReorderCache = false;
        }
        else
        {
            accessMaps.dimOrder = std::get<utilities::DimensionOrder>(cacheMappingInfo);
            cacheInfo.dimReorderCache = true;
        }

        int memoryLocation = 0;

        switch (memorySpace)
        {
        case MemorySpace::Global:
            memoryLocation = 0; //Todo: figure what api to call
            break;
        default:
            [[fallthrough]];
        case MemorySpace::None:
            [[fallthrough]];
        case MemorySpace::Shared:
            memoryLocation = (int)value::MemorySpace::Shared;
            break;
        case MemorySpace::Private:
            memoryLocation = (int)value::MemorySpace::Private;
            break;
        case MemorySpace::Tensor:
            memoryLocation = (int)value::MemorySpace::Tensor;
            break;
        }

        // The cache shape isn't determined for active block caches until lowering the cache ops, so give it a dynamic shape for now to be replaced during lowering

        auto inputType = input.getType();
        assert(inputType.isa<MemRefType>());
        auto inputMemRefType = inputType.cast<MemRefType>();
        auto inputElementType = inputMemRefType.getElementType();
        auto cacheElementType = elementType.has_value() ? accera::value::ValueTypeToMLIRType(builder, *elementType) : inputElementType;

        std::vector<int64_t> dynamicSizedMemrefShape(1, DynamicSizeSentinelValue);
        cacheInfo.cacheType = MemRefType::get(dynamicSizedMemrefShape, cacheElementType, {}, memoryLocation);
        cacheInfo.cacheAllocation = cacheAllocation;
        cacheInfo.accessMaps = accessMaps;
        cacheInfo.activeBlockCache = true;

        return cacheInfo;
    }

    CacheAccessContext MakeCacheAccessContext(mlir::Value cache, CacheInfo& cacheInfo)
    {
        CacheAccessContext result;
        result.value = cache;
        result.accessMaps = cacheInfo.accessMaps;
        result.activeBlockCache = cacheInfo.activeBlockCache;
        result.dimReorderCache = cacheInfo.dimReorderCache;
        result.fullRelevantScheduleIndices = cacheInfo.fullRelevantScheduleIndices;
        result.externalRelevantScheduleIndices = cacheInfo.externalRelevantScheduleIndices;
        result.cacheRegionRelevantScheduleIndexRanges = cacheInfo.cacheRegionRelevantScheduleIndexRanges;
        result.cacheRegionBaseIndices = cacheInfo.cacheRegionBaseIndices;
        return result;
    }

    //
    // MakeCacheOp
    //

    void MakeCacheOp::build(OpBuilder& builder,
                            OperationState& result,
                            mlir::MemRefType cacheType,
                            accera::ir::value::MemorySpace memorylocation)
    {
        build(
            builder,
            result,
            cacheType,
            memorylocation,
            mlir::AffineMap::getMultiDimIdentityMap(cacheType.getRank(), builder.getContext()),
            mlir::AffineMap::getMultiDimIdentityMap(cacheType.getRank(), builder.getContext()),
            std::vector<Index>{},
            std::vector<Index>{});
    }

    void MakeCacheOp::build(OpBuilder& builder,
                            OperationState& result,
                            mlir::MemRefType cacheType,
                            accera::ir::value::MemorySpace memorylocation,
                            AffineMap activeBlockToCacheMap,
                            AffineMap offsetArrayToCacheAccessMap,
                            const std::vector<Index>& offsetAccessIndices,
                            const std::vector<Index>& multiCacheAccessIndices)
    {
        auto offsetAccessIndexAttrs = util::ConvertIndexVectorToArrayAttr(offsetAccessIndices, builder.getContext());
        auto multiCacheAccessIndexAttrs = util::ConvertIndexVectorToArrayAttr(multiCacheAccessIndices, builder.getContext());

        build(builder,
              result,
              cacheType,
              memorylocation,
              activeBlockToCacheMap,
              offsetArrayToCacheAccessMap,
              offsetAccessIndexAttrs,
              multiCacheAccessIndexAttrs);
    }

    mlir::AffineValueMap MakeCacheOp::insertCachePosition(const std::vector<mlir::Value>& multiCacheIndexIterationCounters, const std::vector<mlir::Value>& offsetAccessIVs, const std::vector<mlir::Value>& baseArrayIndices)
    {
        std::vector<mlir::Value> allIndices(multiCacheIndexIterationCounters.begin(), multiCacheIndexIterationCounters.end());
        allIndices.insert(allIndices.end(), offsetAccessIVs.begin(), offsetAccessIVs.end());
        allIndices.insert(allIndices.end(), baseArrayIndices.begin(), baseArrayIndices.end());
        auto map = offsetArrayToCacheAccessMap();
        mlir::AffineValueMap result(map, allIndices);
        return result;
    }

    mlir::AffineValueMap MakeCacheOp::insertCachePosition(mlir::Operation* where, const std::vector<mlir::Value>& baseArrayIndices)
    {
        return insertCachePosition(where->getBlock(), baseArrayIndices);
    }

    mlir::AffineValueMap MakeCacheOp::insertCachePosition(mlir::Block* where, const std::vector<mlir::Value>& baseArrayIndices)
    {
        std::vector<loopnest::Index> cacheMultiCacheIndices = util::ConvertArrayAttrToIndexVector(multiCacheAccessIndices());
        std::vector<loopnest::Index> cacheOffsetAccessIndices = util::ConvertArrayAttrToIndexVector(offsetAccessIndices());

        std::vector<mlir::Value> multiCacheIVs = util::GetCurrentIndexIVs(cacheMultiCacheIndices, where);
        std::vector<mlir::Value> offsetAccessIVs = util::GetCurrentIndexIVs(cacheOffsetAccessIndices, where);

        mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(where);

        std::vector<mlir::Value> multiCacheIterationCounters;
        std::transform(multiCacheIVs.begin(), multiCacheIVs.end(), std::back_inserter(multiCacheIterationCounters), [&](mlir::Value iv) {
            mlir::AffineForOp loop = mlir::getForInductionVarOwner(iv);
            mlir::Value multiCacheIterationCounter = util::CreateConstantRangeForOpIterationCounter(builder, builder.getUnknownLoc(), loop);
            return multiCacheIterationCounter;
        });

        return insertCachePosition(multiCacheIterationCounters, offsetAccessIVs, baseArrayIndices);
    }

    // Note : this doesn't always work after canonicalization potentially removes operands
    template <typename OpType>
    std::vector<mlir::Value> GetBaseArrayLoadStorePosition(OpType op, const mlir::ArrayAttr& multiCacheAccessIndices, const mlir::ArrayAttr& offsetAccessIndices)
    {
        std::vector<loopnest::Index> cacheMultiCacheIndices = util::ConvertArrayAttrToIndexVector(multiCacheAccessIndices);
        std::vector<loopnest::Index> cacheOffsetAccessIndices = util::ConvertArrayAttrToIndexVector(offsetAccessIndices);

        size_t multiCacheIdxCount = cacheMultiCacheIndices.size();
        size_t offsetIdxCount = cacheOffsetAccessIndices.size();

        auto operands = op.getMapOperands().drop_front(multiCacheIdxCount + offsetIdxCount);

        return std::vector<mlir::Value>(operands.begin(), operands.end());
    }

    std::vector<mlir::Value> MakeCacheOp::getBaseArrayPosition(mlir::AffineReadOpInterface loadOp)
    {
        assert(loadOp.getMemRef() == cache());
        return GetBaseArrayLoadStorePosition(loadOp, multiCacheAccessIndices(), offsetAccessIndices());
    }

    std::vector<mlir::Value> MakeCacheOp::getBaseArrayPosition(mlir::AffineWriteOpInterface storeOp)
    {
        assert(storeOp.getMemRef() == cache());
        return GetBaseArrayLoadStorePosition(storeOp, multiCacheAccessIndices(), offsetAccessIndices());
    }

    //
    // ActiveElementCacheCopyOp
    //

    void ActiveElementCacheCopyOp::build(OpBuilder& builder,
                                         OperationState& result,
                                         Value src,
                                         Value dst,
                                         ValueRange externalRelevantIndices,
                                         const std::vector<IndexRange>& cacheRegionRelevantIndexRanges,
                                         const std::vector<std::vector<Index>>& cacheRegionBaseIndices,
                                         AffineMap relevantIndicesToSrcMap,
                                         AffineMap relevantIndicesToDstMap)
    {
        std::vector<mlir::Value> indices(externalRelevantIndices.begin(), externalRelevantIndices.end());
        auto cacheRegionRelevantIndexRangeAttrs = util::VectorToArrayAttr<IndexRange, IndexRangeAttr>(
            cacheRegionRelevantIndexRanges,
            [&](const IndexRange& indexRange) -> IndexRangeAttr {
                return IndexRangeAttr::get(indexRange, builder.getContext());
            },
            builder.getContext());

        auto cacheRegionBaseIndexAttrs = util::VectorToArrayAttr<std::vector<Index>, mlir::ArrayAttr>(
            cacheRegionBaseIndices,
            [&](const std::vector<Index>& currentBaseIndices) -> mlir::ArrayAttr {
                return util::ConvertIndexVectorToArrayAttr(currentBaseIndices, builder.getContext());
            },
            builder.getContext());
        build(builder,
              result,
              src,
              dst,
              externalRelevantIndices,
              cacheRegionRelevantIndexRangeAttrs,
              cacheRegionBaseIndexAttrs,
              relevantIndicesToSrcMap,
              relevantIndicesToDstMap);
    }

    void ActiveElementCacheCopyOp::build(OpBuilder& builder,
                                         OperationState& result,
                                         Value src,
                                         CacheAccessContext dstContext)
    {
        build(builder,
              result,
              src,
              dstContext.value,
              dstContext.externalRelevantScheduleIndices,
              dstContext.cacheRegionRelevantScheduleIndexRanges,
              dstContext.cacheRegionBaseIndices,
              dstContext.accessMaps.relevantIndicesToInput,
              dstContext.accessMaps.relevantIndicesToActiveElementCache);
    }

    void ActiveElementCacheCopyOp::build(OpBuilder& builder,
                                         OperationState& result,
                                         CacheAccessContext srcContext,
                                         Value dst)
    {
        build(builder,
              result,
              srcContext.value,
              dst,
              srcContext.externalRelevantScheduleIndices,
              srcContext.cacheRegionRelevantScheduleIndexRanges,
              srcContext.cacheRegionBaseIndices,
              srcContext.accessMaps.relevantIndicesToActiveElementCache,
              srcContext.accessMaps.relevantIndicesToInput);
    }

    //
    // ActiveElementCacheReduceOp
    //
    void ActiveElementCacheReduceOp::build(OpBuilder& builder,
                                           OperationState& result,
                                           Value src,
                                           Value dst,
                                           ValueRange externalRelevantIndices,
                                           const std::vector<IndexRange>& cacheRegionRelevantIndexRanges,
                                           const std::vector<std::vector<Index>>& cacheRegionBaseIndices,
                                           AffineMap relevantIndicesToSrcCacheMap,
                                           AffineMap relevantIndicesToDstMap,
                                           ValueRange scaleValues)
    {
        auto cacheRegionRelevantIndexRangeAttrs = util::VectorToArrayAttr<IndexRange, IndexRangeAttr>(
            cacheRegionRelevantIndexRanges,
            [&](const IndexRange& indexRange) -> IndexRangeAttr {
                return IndexRangeAttr::get(indexRange, builder.getContext());
            },
            builder.getContext());

        auto cacheRegionBaseIndexAttrs = util::VectorToArrayAttr<std::vector<Index>, mlir::ArrayAttr>(
            cacheRegionBaseIndices,
            [&](const std::vector<Index>& currentBaseIndices) -> mlir::ArrayAttr {
                return util::ConvertIndexVectorToArrayAttr(currentBaseIndices, builder.getContext());
            },
            builder.getContext());
        build(builder,
              result,
              src,
              dst,
              externalRelevantIndices,
              scaleValues,
              cacheRegionRelevantIndexRangeAttrs,
              cacheRegionBaseIndexAttrs,
              relevantIndicesToSrcCacheMap,
              relevantIndicesToDstMap);
    }

    void ActiveElementCacheReduceOp::build(OpBuilder& builder,
                                           OperationState& result,
                                           Value src,
                                           CacheAccessContext dstContext)
    {
        build(builder,
              result,
              src,
              dstContext.value,
              dstContext.externalRelevantScheduleIndices,
              dstContext.cacheRegionRelevantScheduleIndexRanges,
              dstContext.cacheRegionBaseIndices,
              dstContext.accessMaps.relevantIndicesToActiveElementCache,
              dstContext.accessMaps.relevantIndicesToInput,
              llvm::None); // scaleValues
    }

    void ActiveElementCacheReduceOp::build(OpBuilder& builder,
                                           OperationState& result,
                                           CacheAccessContext srcContext,
                                           Value dst)
    {
        build(builder,
              result,
              srcContext.value,
              dst,
              srcContext.externalRelevantScheduleIndices,
              srcContext.cacheRegionRelevantScheduleIndexRanges,
              srcContext.cacheRegionBaseIndices,
              srcContext.accessMaps.relevantIndicesToActiveElementCache,
              srcContext.accessMaps.relevantIndicesToInput,
              llvm::None); // scaleValues
    }

    //
    // BeginCacheMappingOp
    //
    void BeginCacheMappingOp::build(OpBuilder& builder,
                                    OperationState& result,
                                    Value fromValue,
                                    Value baseCacheValue,
                                    Value baseInput,
                                    CacheAccessContext toValueContext,
                                    int64_t id,
                                    bool activeBlockCache)
    {
        auto cacheRegionRelevantIndexRangeAttrs = util::VectorToArrayAttr<IndexRange, IndexRangeAttr>(
            toValueContext.cacheRegionRelevantScheduleIndexRanges,
            [&](const IndexRange& indexRange) -> IndexRangeAttr {
                return IndexRangeAttr::get(indexRange, builder.getContext());
            },
            builder.getContext());

        auto cacheRegionBaseIndexAttrs = util::VectorToArrayAttr<std::vector<Index>, mlir::ArrayAttr>(
            toValueContext.cacheRegionBaseIndices,
            [&](const std::vector<Index>& currentBaseIndices) -> mlir::ArrayAttr {
                return util::ConvertIndexVectorToArrayAttr(currentBaseIndices, builder.getContext());
            },
            builder.getContext());

        result.addTypes(builder.getIndexType());
        result.addOperands(fromValue);
        result.addOperands(baseCacheValue);
        result.addOperands(baseInput);
        result.addOperands(toValueContext.value);
        result.addOperands(toValueContext.fullRelevantScheduleIndices);
        result.addOperands(toValueContext.externalRelevantScheduleIndices);
        result.addAttribute("cacheRegionRelevantIndexRanges", cacheRegionRelevantIndexRangeAttrs);
        result.addAttribute("cacheRegionBaseIndices", cacheRegionBaseIndexAttrs);
        result.addAttribute("toValueAccessMaps", toValueContext.accessMaps.ToAttr(builder));
        result.addAttribute("id", builder.getI64IntegerAttr(id));
        if (activeBlockCache)
        {
            result.addAttribute("activeBlockCache", builder.getUnitAttr());
        }
        result.addAttribute("operand_segment_sizes", builder.getI32VectorAttr({ 1 /* fromValue */, 1 /* baseCacheValue */, 1 /* toValue */, static_cast<int32_t>(toValueContext.fullRelevantScheduleIndices.size()), static_cast<int32_t>(toValueContext.externalRelevantScheduleIndices.size()) }));
    }

    CacheAccessContext BeginCacheMappingOp::getToValueAccessContext()
    {
        BeginCacheMappingOp::Adaptor adaptor{ *this };

        auto cacheRegionRelevantIndexRanges = util::ArrayAttrToVector<IndexRange, IndexRangeAttr>(
            adaptor.cacheRegionRelevantIndexRanges(),
            [&](const IndexRangeAttr& indexRangeAttr) -> IndexRange {
                return indexRangeAttr.getValue();
            });

        auto cacheRegionBaseIndices = util::ArrayAttrToVector<std::vector<Index>, mlir::ArrayAttr>(
            adaptor.cacheRegionBaseIndices(),
            util::ConvertArrayAttrToIndexVector);

        CacheAccessContext result;
        result.value = toValue();
        result.activeBlockCache = activeBlockCache();
        result.fullRelevantScheduleIndices = adaptor.fullRelevantIndices();
        result.externalRelevantScheduleIndices = adaptor.externalRelevantIndices();
        result.cacheRegionRelevantScheduleIndexRanges = cacheRegionRelevantIndexRanges;
        result.cacheRegionBaseIndices = cacheRegionBaseIndices;
        result.accessMaps = CacheAccessMaps::FromAttr(toValueAccessMaps());
        return result;
    }

    mlir::Operation* BeginCacheMappingOp::getEndOp()
    {
        auto uses = util::getUsesOfType<EndCacheMappingOp>(resultId());
        assert(uses.size() == 1);
        return uses.front();
    }

    int64_t BeginCacheMappingOp::getId()
    {
        return id();
    }

    //
    // EndCacheMappingOp
    //
    mlir::Operation* EndCacheMappingOp::getBeginOp()
    {
        auto op = mappingId().getDefiningOp();
        assert(op != nullptr);
        return op;
    }

    //
    // BeginCreateCacheOp
    //
    void BeginCreateCacheOp::build(OpBuilder& builder,
                                   OperationState& result,
                                   Value inputValue,
                                   CacheAccessContext cacheAccessContext,
                                   Value baseInput,
                                   loopnest::Index cacheIndex,
                                   loopnest::Index triggerIndex,
                                   int64_t id,
                                   int64_t cacheHierarchyLevel,
                                   bool activeBlockCache,
                                   bool dimReorderCache,
                                   bool thrifty,
                                   value::CacheStrategy strategy,
                                   bool doubleBufferCache,
                                   accera::ir::value::MemorySpace doubleBufferMemorySpace,
                                   const VectorizationInfo& vecInfo)
    {
        auto cacheRegionRelevantIndexRangeAttrs = util::VectorToArrayAttr<IndexRange, IndexRangeAttr>(
            cacheAccessContext.cacheRegionRelevantScheduleIndexRanges,
            [&](const IndexRange& indexRange) -> IndexRangeAttr {
                return IndexRangeAttr::get(indexRange, builder.getContext());
            },
            builder.getContext());

        auto cacheRegionBaseIndexAttrs = util::VectorToArrayAttr<std::vector<Index>, mlir::ArrayAttr>(
            cacheAccessContext.cacheRegionBaseIndices,
            [&](const std::vector<Index>& currentBaseIndices) -> mlir::ArrayAttr {
                return util::ConvertIndexVectorToArrayAttr(currentBaseIndices, builder.getContext());
            },
            builder.getContext());

        result.addTypes(builder.getIndexType());
        result.addOperands(inputValue);
        result.addOperands(cacheAccessContext.value);
        result.addOperands(baseInput);
        result.addOperands(cacheAccessContext.fullRelevantScheduleIndices);
        result.addOperands(cacheAccessContext.externalRelevantScheduleIndices);
        result.addAttribute("cacheRegionRelevantIndexRanges", cacheRegionRelevantIndexRangeAttrs);
        result.addAttribute("cacheRegionBaseIndices", cacheRegionBaseIndexAttrs);
        result.addAttribute("cacheAccessMaps", cacheAccessContext.accessMaps.ToAttr(builder));
        result.addAttribute("triggerIndex", IndexAttr::get(triggerIndex, builder.getContext()));
        result.addAttribute("cacheIndex", IndexAttr::get(cacheIndex, builder.getContext()));
        result.addAttribute("id", builder.getI64IntegerAttr(id));
        result.addAttribute("cacheHierarchyLevel", builder.getI64IntegerAttr(cacheHierarchyLevel));
        if (activeBlockCache)
        {
            result.addAttribute("activeBlockCache", builder.getUnitAttr());
        }
        if (dimReorderCache)
        {
            result.addAttribute("dimReorderCache", builder.getUnitAttr());
        }
        if (thrifty)
        {
            result.addAttribute("thrifty", builder.getUnitAttr());
        }
        if (doubleBufferCache)
        {
            result.addAttribute("doubleBufferCache", builder.getUnitAttr());
            result.addAttribute("doubleBufferMemorySpace", value::MemorySpaceAttr::get(builder.getContext(), doubleBufferMemorySpace));
        }
        result.addAttribute("strategy", value::CacheStrategyAttr::get(builder.getContext(), strategy));
        result.addAttribute("vectorizationInfo", VectorizationInfoAttr::get(vecInfo, builder.getContext()));
        result.addAttribute("operand_segment_sizes", builder.getI32VectorAttr({ 1 /* fromValue */, 1 /* toValue */, 1 /* baseInput */, static_cast<int32_t>(cacheAccessContext.fullRelevantScheduleIndices.size()), static_cast<int32_t>(cacheAccessContext.externalRelevantScheduleIndices.size()) }));
    }

    CacheAccessContext BeginCreateCacheOp::getCacheAccessContext()
    {
        BeginCreateCacheOp::Adaptor adaptor{ *this };

        auto cacheRegionRelevantIndexRanges = util::ArrayAttrToVector<IndexRange, IndexRangeAttr>(
            adaptor.cacheRegionRelevantIndexRanges(),
            [&](const IndexRangeAttr& indexRangeAttr) -> IndexRange {
                return indexRangeAttr.getValue();
            });

        auto cacheRegionBaseIndices = util::ArrayAttrToVector<std::vector<Index>, mlir::ArrayAttr>(
            adaptor.cacheRegionBaseIndices(),
            util::ConvertArrayAttrToIndexVector);

        CacheAccessContext result;
        result.value = cache();
        result.activeBlockCache = activeBlockCache();
        result.dimReorderCache = dimReorderCache();
        result.fullRelevantScheduleIndices = adaptor.fullRelevantIndices();
        result.externalRelevantScheduleIndices = adaptor.externalRelevantIndices();
        result.cacheRegionRelevantScheduleIndexRanges = cacheRegionRelevantIndexRanges;
        result.cacheRegionBaseIndices = cacheRegionBaseIndices;
        result.accessMaps = CacheAccessMaps::FromAttr(cacheAccessMaps());
        return result;
    }

    Index BeginCreateCacheOp::index()
    {
        return triggerIndex().getValue();
    }

    Position BeginCreateCacheOp::position()
    {
        return Position::prologue;
    }

    mlir::Operation* BeginCreateCacheOp::getInjectionEndOp()
    {
        return getEndOp();
    }

    mlir::Operation* BeginCreateCacheOp::getEndOp()
    {
        auto uses = util::getUsesOfType<EndCacheRegionOp>(resultId());
        assert(uses.size() == 1);
        return uses.front();
    }

    int64_t BeginCreateCacheOp::getId()
    {
        return id();
    }

    //
    // BeginCreateMaxElementCacheOp
    //
    void BeginCreateMaxElementCacheOp::build(OpBuilder& builder,
                                             OperationState& result,
                                             Value input,
                                             Value cache,
                                             Value baseInput,
                                             CacheAccessMaps accessMaps,
                                             int64_t maxElements,
                                             loopnest::Index innermostLoopNestIndex,
                                             int64_t id,
                                             int64_t cacheHierarchyLevel,
                                             bool dimReorderCache,
                                             bool thrifty,
                                             value::CacheStrategy strategy,
                                             bool doubleBufferCache,
                                             accera::ir::value::MemorySpace doubleBufferMemorySpace,
                                             const VectorizationInfo& vecInfo)
    {
        result.addTypes(builder.getIndexType());
        result.addOperands(input);
        result.addOperands(cache);
        result.addOperands(baseInput);
        result.addAttribute("cacheAccessMaps", accessMaps.ToAttr(builder));
        result.addAttribute("id", builder.getI64IntegerAttr(id));
        result.addAttribute("cacheHierarchyLevel", builder.getI64IntegerAttr(cacheHierarchyLevel));
        result.addAttribute("maxElements", builder.getI64IntegerAttr(maxElements));
        result.addAttribute("innermostLoopNestIndex", IndexAttr::get(innermostLoopNestIndex, builder.getContext()));
        if (dimReorderCache)
        {
            result.addAttribute("dimReorderCache", builder.getUnitAttr());
        }
        if (thrifty)
        {
            result.addAttribute("thrifty", builder.getUnitAttr());
        }
        if (doubleBufferCache)
        {
            result.addAttribute("doubleBufferCache", builder.getUnitAttr());
            result.addAttribute("doubleBufferMemorySpace", value::MemorySpaceAttr::get(builder.getContext(), doubleBufferMemorySpace));
        }
        result.addAttribute("strategy", value::CacheStrategyAttr::get(builder.getContext(), strategy));
        result.addAttribute("vectorizationInfo", VectorizationInfoAttr::get(vecInfo, builder.getContext()));
    }

    Index BeginCreateMaxElementCacheOp::index()
    {
        return innermostLoopNestIndex().getValue();
    }

    Position BeginCreateMaxElementCacheOp::position()
    {
        return Position::prologue;
    }

    mlir::Operation* BeginCreateMaxElementCacheOp::getInjectionEndOp()
    {
        return getEndOp();
    }

    mlir::Operation* BeginCreateMaxElementCacheOp::getEndOp()
    {
        auto uses = util::getUsesOfType<EndCacheRegionOp>(resultId());
        assert(uses.size() == 1);
        return uses.front();
    }

    int64_t BeginCreateMaxElementCacheOp::getId()
    {
        return id();
    }

    //
    // BeginActiveCacheRegionOp
    //
    void BeginActiveCacheRegionOp::build(OpBuilder& builder,
                                         OperationState& result,
                                         mlir::Value cache,
                                         int64_t id)
    {
        result.addTypes(builder.getIndexType());
        result.addOperands(cache);
        result.addAttribute("id", builder.getI64IntegerAttr(id));
    }

    mlir::Operation* BeginActiveCacheRegionOp::getEndOp()
    {
        auto uses = util::getUsesOfType<EndCacheRegionOp>(resultId());
        assert(uses.size() == 1);
        return uses.front();
    }

    int64_t BeginActiveCacheRegionOp::getId()
    {
        return id();
    }

    //
    // EndCacheRegionOp
    //
    Operation* EndCacheRegionOp::getBeginOp()
    {
        auto op = regionId().getDefiningOp();
        assert(op != nullptr);
        return op;
    }

    //
    // DelayedMappingRegionOp
    //
    void DelayedMappingRegionOp::build(mlir::OpBuilder& builder,
                                       mlir::OperationState& result,
                                       mlir::Value from,
                                       mlir::Value to)
    {
        result.addOperands({ from, to });
        mlir::Region* region = result.addRegion();
        mlir::Block* bodyBlock = new mlir::Block;
        region->getBlocks().push_back(bodyBlock);
        ensureTerminator(*region, builder, result.location);
    }

    DelayedMappingRegionOp MakeDelayedMappingRegion(mlir::OpBuilder& builder, mlir::Value from, mlir::Value to, std::function<void(mlir::OpBuilder&)> body)
    {
        auto loc = from.getLoc();
        auto mappingRegionOp = builder.create<DelayedMappingRegionOp>(loc, from, to);
        auto bodyBuilder = mappingRegionOp.getBodyBuilder();
        body(bodyBuilder);

        return mappingRegionOp;
    }

    std::optional<std::pair<value::Processor, mlir::AffineMap>> GetBindingForLoop(mlir::AffineForOp loop)
    {
        auto gpuMapAttr = loop->getAttrOfType<mlir::DictionaryAttr>("accv_gpu_map");
        if (!gpuMapAttr)
        {
            return std::nullopt;
        }
        auto gpuProcStr = gpuMapAttr.get("proc").cast<mlir::StringAttr>().getValue();
        auto gpuProcOpt = value::symbolizeEnum<value::Processor>(gpuProcStr);
        assert(gpuProcOpt.hasValue() && "Unrecognized proc tag found");
        auto gpuProc = gpuProcOpt.getValue();
        auto map = gpuMapAttr.get("map").cast<mlir::AffineMapAttr>().getValue();
        return std::make_pair(gpuProc, map);
    }

    // Parse an instance of an attribute registered to the execution plan dialect.
    mlir::Attribute ExecutionPlanDialect::parseAttribute(mlir::DialectAsmParser& parser, mlir::Type type) const
    {
        // Parse the main keyword for the attribute.
        StringRef keyword;
        if (failed(parser.parseKeyword(&keyword)))
            return {};

        if (keyword == "vectorizationinfo")
        {
            return parseVectorizationInfo(parser);
        }
        else if (keyword == "parallelizationinfo")
        {
            return parseParallelizationInfo(parser);
        }
        else if (keyword == "tensorizationinfo")
        {
            return parseTensorizationInfo(parser);
        }
        else if (keyword == "inplaceunrollinfo")
        {
            return parseInPlaceUnrollInfo(parser);
        }

        parser.emitError(parser.getNameLoc(), "unknown execution plan attribute: " + keyword);
        return {};
    }

    // Print an instance of a type registered to the execution plan dialect.
    void ExecutionPlanDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer) const
    {
        if (auto vecInfoAttr = attr.dyn_cast<VectorizationInfoAttr>())
        {
            print(vecInfoAttr, printer);
        }
        else if (auto parInfoAttr = attr.dyn_cast<ParallelizationInfoAttr>())
        {
            print(parInfoAttr, printer);
        }
        else if (auto tensorInfoAttr = attr.dyn_cast<TensorizationInfoAttr>())
        {
            print(tensorInfoAttr, printer);
        }
        else if (auto unrollInfoAttr = attr.dyn_cast<InPlaceUnrollInfoAttr>())
        {
            print(unrollInfoAttr, printer);
        }
    }

} // namespace executionPlan
} // namespace accera::ir

//
// TableGen'd op method definitions
//

#define GET_OP_CLASSES
#include "exec/ExecutionPlanInterfaces.cpp.inc"
#include "exec/ExecutionPlanOps.cpp.inc"
