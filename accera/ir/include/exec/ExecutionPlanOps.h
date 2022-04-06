////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <optional>
#include <variant>

#include "CacheAccessMaps.h"
#include "ExecutionPlanAttributes.h"
#include "ExecutionPlanEnums.h"
#include "ir/include/nest/Index.h"
#include "ir/include/nest/IndexRange.h"
#include "ir/include/nest/LoopNestAttributes.h"
#include "ir/include/nest/LoopNestOps.h"
#include "ir/include/value/ValueEnums.h"
#include <utilities/include/MemoryLayout.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/LoopLikeInterface.h>

namespace accera::ir
{
// mlir-tblgen currently creates files with the assumption that the following
// symbols are present in the current namespace, so we have to import them
// explicitly
using llvm::ArrayRef;
using llvm::iterator_range;
using llvm::StringRef;
using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::AffineValueMap;
using mlir::ArrayAttr;
using mlir::Attribute;
using mlir::Block;
using mlir::Builder;
using mlir::IntegerAttr;
using mlir::LogicalResult;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::OwningRewritePatternList;
using mlir::ParseResult;
using mlir::Region;
using mlir::ShapedType;
using mlir::TensorType;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

using loopnest::Index;
using loopnest::IndexAttr;
using loopnest::IndexRange;
using loopnest::InjectableMapping;
using loopnest::Position;

namespace OpTrait = mlir::OpTrait;

namespace executionPlan
{
    using accera::ir::value::MemorySpace;

    struct CacheAccessContext
    {
        mlir::Value value;
        CacheAccessMaps accessMaps;
        bool activeBlockCache;
        bool dimReorderCache;
        ValueRange fullRelevantScheduleIndices;
        ValueRange externalRelevantScheduleIndices; // Relevant schedule indices that are external to the cache region
        std::vector<IndexRange> cacheRegionRelevantScheduleIndexRanges;
        std::vector<std::vector<Index>> cacheRegionBaseIndices;
    };

    // Copied from ShapedType::kDynamicSize in mlir\include\mlir\IR\StandardTypes.h becuase gcc has linker issues with static constexpr constants
    const int64_t DynamicSizeSentinelValue = -1;

#include "exec/ExecutionPlanInterfaces.h.inc"
} // namespace executionPlan
} // namespace accera::ir

// Include the auto-generated header file containing the declarations of the execution plan operations.
#define GET_OP_CLASSES
#include "exec/ExecutionPlanDialect.h.inc"
#include "exec/ExecutionPlanOps.h.inc"

namespace accera::ir::executionPlan
{

// Unit attr name for controlling whether bounds checking is done for ops within a marked op
const mlir::StringRef AccessBoundsCheckAttrName = "accxp.access_bounds_check";

//
// Utility functions and EDSC-type intrinsics
//

struct ScheduleShardMapping
{
    std::vector<int64_t> shardSizes;
    std::vector<int64_t> logicalDimensionMappings;
    std::vector<int64_t> affinePerDimCoefficients;
    std::vector<int64_t> affineCoefficients;
    std::vector<std::vector<loopnest::Index>> relevantScheduleIndices;
    std::vector<std::vector<size_t>> relevantScheduleIndexPositions;
};

struct CacheInfo
{
    MemRefType cacheType;
    bool activeBlockCache;
    bool dimReorderCache;
    int64_t maxElementBudget = -1;
    CacheAllocation cacheAllocation;
    std::optional<loopnest::Index> cacheIndex;
    std::optional<loopnest::Index> triggerIndex;
    std::vector<Index> accessBaseIndices;
    CacheAccessMaps accessMaps;
    ScheduleShardMapping fullShardMapping;
    ScheduleShardMapping shardMapping;
    llvm::SmallVector<mlir::Value, 4> fullRelevantScheduleIndices;
    llvm::SmallVector<mlir::Value, 4> externalRelevantScheduleIndices; // Relevant schedule indices that are external to the cache region
    std::vector<IndexRange> cacheRegionRelevantScheduleIndexRanges;
    std::vector<std::vector<Index>> cacheRegionBaseIndices;
};

CacheInfo MakeAutomaticCacheInfo(
    mlir::OpBuilder& builder,
    mlir::Value input,
    CacheAllocation cacheAllocation,
    loopnest::ScheduleOp schedule,
    const std::optional<loopnest::Index>& outermostIncludedSplitIndex,
    const std::optional<int64_t>& maxElements = std::nullopt,
    MemorySpace memorySpace = MemorySpace::Shared);

CacheInfo MakeFullBufferAutomaticCacheInfo(
    mlir::OpBuilder& builder,
    mlir::Value input,
    CacheAllocation cacheAllocation,
    loopnest::ScheduleOp schedule,
    MemorySpace memorySpace = MemorySpace::Global);

CacheInfo MakeManualCacheInfo(
    mlir::OpBuilder& builder,
    mlir::Value input,
    CacheAllocation cacheAllocation,
    loopnest::ScheduleOp schedule,
    const std::optional<loopnest::Index>& keySliceIndex,
    const std::optional<loopnest::Index>& triggerIndex,
    const std::optional<int64_t>& maxElements,
    const std::variant<utilities::MemoryAffineCoefficients, utilities::DimensionOrder>& cacheMappingInfo,
    MemorySpace memorySpace);

mlir::AffineMap ComputeFlatAffineMapFromAffineCoefficients(
    mlir::OpBuilder& builder,
    const utilities::MemoryAffineCoefficients& affineMapping);

ScheduleShardMapping GetScheduleShardMapping(
    loopnest::ScheduleOp schedule,
    const std::vector<loopnest::Index>& accessLogicalIndices);

CacheAccessContext MakeCacheAccessContext(
    mlir::Value cache,
    CacheInfo& cacheInfo);

DelayedMappingRegionOp MakeDelayedMappingRegion(mlir::OpBuilder& builder, mlir::Value from, mlir::Value to, std::function<void(mlir::OpBuilder&)> body);

} // namespace accera::ir::executionPlan
