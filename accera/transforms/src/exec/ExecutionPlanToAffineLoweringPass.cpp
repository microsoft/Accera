////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "exec/ExecutionPlanToAffineLoweringPass.h"
#include "AcceraPasses.h"
#include "util/VectorizationUtil.h"

#include <ir/include/IRUtil.h>
#include <ir/include/exec/ExecutionOptions.h>
#include <ir/include/exec/ExecutionPlanAttributes.h>
#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/exec/VectorizationInfo.h>
#include <ir/include/nest/LoopNestAttributes.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueDialect.h>
#include <ir/include/value/ValueEnums.h>

#include <mlir/Support/LLVM.h>
#include <utilities/include/Boolean.h>
#include <utilities/include/MathUtil.h>
#include <utilities/include/TypeTraits.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Analysis/LoopAnalysis.h>
#include <mlir/Analysis/Utils.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Identifier.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/InliningUtils.h>
#include <mlir/Transforms/LoopUtils.h>
#include <mlir/Transforms/Utils.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <queue>
#include <stack>
#include <stdexcept>

using namespace accera::ir;
using namespace accera::ir::executionPlan;
using namespace accera::ir::value;
using namespace accera::ir::loopnest;
namespace v = accera::ir::value;
using namespace accera::transforms;
using namespace mlir;
using namespace accera::utilities;

#define DEBUG_TYPE "execution-plat-to-affine-lowering"

namespace
{
// Here we prefer using std::string for these attr names so they can be used more flexibly
// in internal utilities as well as MLIR APIs as a mlir::StringRef. Note that mlir::StringRef
// has a constructor that takes a const std::string& for convenience

const std::string BoundsCheckedAttrName = "accxp_bounds_checked";
const std::string BaseArrayAccessMapAttrName = "accxp_base_array_access_map";
const std::string BaseArrayAccessIndicesAttrName = "accxp_base_array_access_indices";

// These strings are used to create predictable index names for internally-generated GPU-related loops
// for the purposes of cache accesses. MakeCacheOps identify the loop indices to look for and combine those
// with a map to access the appropriate position in the cache, however that mechanism does not currently
// distinguish between a general active block position and a specific GPU thread's responsibility region
// within that active block.
// E.g. suppose you're mapping from a shared memory cache to a private memory cache but instead of having
// different loop levels with different active blocks, you want the private memory cache to hold only the
// region that each thread is responsible for in the shared memory cache, so instead of being a new active
// block, it is a subset of an existing active block identified by thread indices
// TODO : come up with a better way of standardizing how a GPU thread maps from a shared active block
//        to the subset of the active block that it is responsible for
const std::string ActionsPerThreadIndexName = "accxp_actions_per_thread_loop_index";
const std::string ThreadVectorizationIndexName = "accxp_thread_vectorization_loop_index";
const std::string ThreadXIndexName = "accxp_thread_x_loop_index";
const std::string ThreadYIndexName = "accxp_thread_y_loop_index";
const std::string ThreadZIndexName = "accxp_thread_z_loop_index";

// Attribute names used for partially unrolling loops
const std::string UnswitchPrefixItersName = "accxp_unswitch_prefix_iters";
const std::string UnswitchSuffixItersName = "accxp_unswitch_suffix_iters";

// #### TODO: move this somewhere that makes sense
enum class GPUIndexDimension
{
    X,
    Y,
    Z
};

struct MakeCacheOpLowering : public OpRewritePattern<MakeCacheOp>
{
    using OpRewritePattern<MakeCacheOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(MakeCacheOp makeCacheOp, PatternRewriter& rewriter) const final;
};

struct CacheZeroOpRewrite : public OpRewritePattern<CacheZeroOp>
{
    using OpRewritePattern<CacheZeroOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CacheZeroOp cacheZeroOp, PatternRewriter& rewriter) const final;
};

struct ActiveElementCacheCopyOpRewrite : public OpRewritePattern<ActiveElementCacheCopyOp>
{
    using OpRewritePattern<ActiveElementCacheCopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ActiveElementCacheCopyOp cacheCopyOp, PatternRewriter& rewriter) const final;
};

struct ThriftyCacheMultiCopyOpRewrite : public OpRewritePattern<MultiCacheCopyOp>
{
    using OpRewritePattern<MultiCacheCopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(MultiCacheCopyOp cacheCopyOp, PatternRewriter& rewriter) const final;
};

struct ThriftyCacheCopyOpRewrite : public OpRewritePattern<ActiveBlockCacheCopyOp>
{
    using OpRewritePattern<ActiveBlockCacheCopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ActiveBlockCacheCopyOp cacheCopyOp, PatternRewriter& rewriter) const final;
};

struct ThriftyCacheReduceOpRewrite : public OpRewritePattern<ActiveBlockCacheReduceOp>
{
    using OpRewritePattern<ActiveBlockCacheReduceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ActiveBlockCacheReduceOp cacheReduceOp, PatternRewriter& rewriter) const final;
};

struct MultiCacheCopyOpRewrite : public OpRewritePattern<MultiCacheCopyOp>
{
    using OpRewritePattern<MultiCacheCopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(MultiCacheCopyOp cacheCopyOp, PatternRewriter& rewriter) const final;
};

struct ActiveBlockCacheCopyOpRewrite : public OpRewritePattern<ActiveBlockCacheCopyOp>
{
    using OpRewritePattern<ActiveBlockCacheCopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ActiveBlockCacheCopyOp cacheCopyOp, PatternRewriter& rewriter) const final;
};

struct ActiveElementCacheReduceOpRewrite : public OpRewritePattern<ActiveElementCacheReduceOp>
{
    using OpRewritePattern<ActiveElementCacheReduceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ActiveElementCacheReduceOp cacheReduceOp, PatternRewriter& rewriter) const final;
};

struct ActiveBlockCacheReduceOpRewrite : public OpRewritePattern<ActiveBlockCacheReduceOp>
{
    using OpRewritePattern<ActiveBlockCacheReduceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ActiveBlockCacheReduceOp cacheReduceOp, PatternRewriter& rewriter) const final;
};

struct BeginCacheMappingOpRewrite : public OpRewritePattern<BeginCacheMappingOp>
{
    using OpRewritePattern<BeginCacheMappingOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(BeginCacheMappingOp beginCacheMappingOp, PatternRewriter& rewriter) const final;
};

struct AdjustHierarchicalCacheRegionPositionRewrite : public OpRewritePattern<BeginCacheRegionOp>
{
    using OpRewritePattern<BeginCacheRegionOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(BeginCacheRegionOp beginCacheRegionOp, PatternRewriter& rewriter) const final;
};

struct AdjustCacheMappingPositionRewrite : public OpRewritePattern<BeginCacheMappingOp>
{
    using OpRewritePattern<BeginCacheMappingOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(BeginCacheMappingOp beginCacheMappingOp, PatternRewriter& rewriter) const final;
};

struct BeginCacheRegionOpRewrite : public OpRewritePattern<BeginCacheRegionOp>
{
    using OpRewritePattern<BeginCacheRegionOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(BeginCacheRegionOp beginCacheRegionOp, PatternRewriter& rewriter) const final;
};

struct HoistCacheRegionOpsRewrite : public OpRewritePattern<BeginCacheRegionOp>
{
    using OpRewritePattern<BeginCacheRegionOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(BeginCacheRegionOp beginCacheRegionOp, PatternRewriter& rewriter) const final;
};

struct MergeCacheRegionOpsRewrite : public OpRewritePattern<BeginCacheRegionOp>
{
    using OpRewritePattern<BeginCacheRegionOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(BeginCacheRegionOp beginCacheRegionOp, PatternRewriter& rewriter) const final;
};

struct MaxElementCacheRegionOpRewrite : public OpRewritePattern<BeginMaxElementCacheRegionOp>
{
    using OpRewritePattern<BeginMaxElementCacheRegionOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(BeginMaxElementCacheRegionOp beginMaxElementCacheRegionOp, PatternRewriter& rewriter) const final;
};

struct VectorizeAffineForOpConversion : public OpRewritePattern<AffineForOp>
{
    using OpRewritePattern<AffineForOp>::OpRewritePattern;
    VectorizeAffineForOpConversion(MLIRContext* context, bool printVectorizationDetails = false) :
        OpRewritePattern(context, /* benefit */ 1),
        printVectorizationDetails(printVectorizationDetails)
    {}

    LogicalResult matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const final;

    // status-reporting helper methods
    void emitVectorizationRemark(mlir::Operation* sourceOp, const std::string& remark) const;
    void didVectorizeOp(mlir::Operation* sourceOp, VectorizedOp& vectorizedOp) const;
    void vectorizeOpsInBlock(PatternRewriter& rewriter,
                             mlir::Block::iterator begin,
                             mlir::Block::iterator end,
                             mlir::Value unrollingIV,
                             const VectorizationInfo& vectorInfo,
                             VectorizedOpMap& vectorizedOps,
                             std::vector<BlockAndValueMapping>& laneMappings,
                             int64_t step,
                             int64_t unrollMax) const;

    bool printVectorizationDetails = false;
};

struct InPlaceUnrollAffineForOpConversion : public OpRewritePattern<AffineForOp>
{
    using OpRewritePattern<AffineForOp>::OpRewritePattern;
    InPlaceUnrollAffineForOpConversion(MLIRContext* context, bool printVectorizationDetails = false) :
        OpRewritePattern(context, /* benefit */ 1),
        printVectorizationDetails(printVectorizationDetails)
    {}

    LogicalResult matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const final;

    bool printVectorizationDetails = false;
};

struct TensorizeAffineForOpConversion : public OpRewritePattern<AffineForOp>
{
    using OpRewritePattern<AffineForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const final;
};

struct ParallelizeAffineForOpConversion : public OpRewritePattern<AffineForOp>
{
    using OpRewritePattern<AffineForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const final;
};

struct CollapseAffineParallelOpsRewrite : public OpRewritePattern<AffineParallelOp>
{
    using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(AffineParallelOp affineParallelOp, PatternRewriter& rewriter) const final;
};

struct HoistScalingToCacheReduceRewrite : public OpRewritePattern<mlir::AffineStoreOp>
{
    using OpRewritePattern<mlir::AffineStoreOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::AffineStoreOp affineStoreOp, PatternRewriter& rewriter) const final;
};

struct OutOfBoundsLoadRewrite : public OpRewritePattern<mlir::memref::LoadOp>
{
    using OpRewritePattern<mlir::memref::LoadOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::memref::LoadOp loadOp, PatternRewriter& rewriter) const final;
};

struct OutOfBoundsAffineLoadRewrite : public OpRewritePattern<mlir::AffineLoadOp>
{
    using OpRewritePattern<mlir::AffineLoadOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::AffineLoadOp affineLoadOp, PatternRewriter& rewriter) const final;
};

struct OutOfBoundsStoreRewrite : public OpRewritePattern<mlir::memref::StoreOp>
{
    using OpRewritePattern<mlir::memref::StoreOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::memref::StoreOp toreOp, PatternRewriter& rewriter) const final;
};

struct OutOfBoundsAffineStoreRewrite : public OpRewritePattern<mlir::AffineStoreOp>
{
    using OpRewritePattern<mlir::AffineStoreOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::AffineStoreOp affineStoreOp, PatternRewriter& rewriter) const final;
};

struct ConvertLoadsToAffineRewrite : public OpRewritePattern<mlir::memref::LoadOp>
{
    using OpRewritePattern<mlir::memref::LoadOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::memref::LoadOp loadOp, PatternRewriter& rewriter) const final;
};

struct ConvertStoresToAffineRewrite : public OpRewritePattern<mlir::memref::StoreOp>
{
    using OpRewritePattern<mlir::memref::StoreOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::memref::StoreOp storeOp, PatternRewriter& rewriter) const final;
};

struct ConvertValueLoadsToAffineRewrite : public OpRewritePattern<v::LoadOp>
{
    using OpRewritePattern<v::LoadOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(v::LoadOp loadOp, PatternRewriter& rewriter) const final;
};

struct ConvertValueStoresToAffineRewrite : public OpRewritePattern<v::StoreOp>
{
    using OpRewritePattern<v::StoreOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(v::StoreOp storeOp, PatternRewriter& rewriter) const final;
};

struct DelayedMappingRegionOpRewrite : public OpRewritePattern<DelayedMappingRegionOp>
{
    using OpRewritePattern<DelayedMappingRegionOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(DelayedMappingRegionOp mappingRegionOp, PatternRewriter& rewriter) const final;
};

struct LoopUnswitchingOpRewrite : public OpRewritePattern<mlir::AffineForOp>
{
    using OpRewritePattern<mlir::AffineForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::AffineForOp forOp, PatternRewriter& rewriter) const final;
};

struct ExecutionPlanMakeCacheLoweringPass : public ConvertExecutionPlanMakeCacheBase<ExecutionPlanMakeCacheLoweringPass>
{
    void runOnFunction() final;
};

struct ExecutionPlanCopyReduceLoweringPass : public ConvertExecutionPlanCopyReduceBase<ExecutionPlanCopyReduceLoweringPass>
{
    void runOnOperation() final;
};

struct ExecutionPlanCacheRegionLoweringPass : public ConvertExecutionPlanCacheRegionBase<ExecutionPlanCacheRegionLoweringPass>
{
    void runOnOperation() final;
};

struct ExecutionPlanVectorizationPass : public ConvertExecutionPlanVectorizationBase<ExecutionPlanVectorizationPass>
{
    void runOnOperation() final;
};

struct ExecutionPlanTensorizationPass : public ConvertExecutionPlanTensorizationBase<ExecutionPlanTensorizationPass>
{
    void runOnOperation() final;
};

struct ExecutionPlanParallelizationPass : public ConvertExecutionPlanParallelizationBase<ExecutionPlanParallelizationPass>
{
    void runOnOperation() final;
};

struct ExecutionPlanScaleHoistingPass : public ConvertExecutionPlanScaleHoistingBase<ExecutionPlanScaleHoistingPass>
{
    void runOnFunction() final;
};

struct OutOfBoundsAccessHandlingPass : public HandleOutOfBoundsAccessBase<OutOfBoundsAccessHandlingPass>
{
    void runOnFunction() final;
};

// Vectorization-related functions and types

Type GetInnerElementType(Value val)
{
    auto valType = val.getType();
    assert(valType.isa<MemRefType>());
    auto memRefType = valType.cast<MemRefType>();
    auto elementType = memRefType.getElementType();
    if (elementType.isa<mlir::VectorType>())
    {
        auto vectorType = elementType.cast<mlir::VectorType>();
        elementType = vectorType.getElementType();
    }
    return elementType;
}

bool HasVectorizationInfo(Operation* op)
{
    auto vectorizationInfoAttr = op->getAttrOfType<VectorizationInfoAttr>(VectorizationInfoAttr::getKeyName());

    return vectorizationInfoAttr != nullptr;
}

VectorizationInfo GetVectorizationInfo(Operation* op)
{
    auto vectorizationInfoAttr = op->getAttrOfType<VectorizationInfoAttr>(VectorizationInfoAttr::getKeyName());
    assert(vectorizationInfoAttr != nullptr);

    return vectorizationInfoAttr.getValue();
}

template <typename CacheOpType>
VectorizationInfo GetCacheOpVectorizationInfoOrDefault(CacheOpType cacheOp)
{
    VectorizationInfo vecInfo;
    auto vecInfoAttr = cacheOp.vectorizationInfoAttr();
    if (vecInfoAttr)
    {
        vecInfo = vecInfoAttr.getValue();
    }
    return vecInfo;
}

[[maybe_unused]] void SetVectorizationInfo(Operation* op, const VectorizationInfo& vecInfo)
{
    op->setAttr(VectorizationInfoAttr::getKeyName(), VectorizationInfoAttr::get(vecInfo, op->getContext()));
}

// TODO: Remove need for builder
void SetVectorizationInfo(ScheduleOp op, Index index, const VectorizationInfo& vecInfo)
{
    OpBuilder builder(op);
    auto vectorizationInfoIdentifier = builder.getIdentifier(VectorizationInfoAttr::getKeyName());
    op.addLoopAttribute(index, vectorizationInfoIdentifier, VectorizationInfoAttr::get(vecInfo, builder.getContext()));
}

[[maybe_unused]] void SetVectorizationInfo(ScheduleOp op, SymbolicIndexOp index, const VectorizationInfo& vecInfo)
{
    SetVectorizationInfo(op, index.getValue(), vecInfo);
}

void RemoveVectorizationInfo(Operation* op)
{
    OpBuilder builder(op);
    auto vectorizationInfoIdentifier = builder.getIdentifier(VectorizationInfoAttr::getKeyName());
    op->removeAttr(vectorizationInfoIdentifier);
}

// In-place-unroll-related functions

bool HasInPlaceUnrollInfo(Operation* op)
{
    auto inPlaceUnrollInfoAttr = op->getAttrOfType<InPlaceUnrollInfoAttr>(InPlaceUnrollInfoAttr::getKeyName());

    return inPlaceUnrollInfoAttr != nullptr;
}

InPlaceUnrollInfo GetInPlaceUnrollInfo(Operation* op)
{
    auto inPlaceUnrollInfoAttr = op->getAttrOfType<InPlaceUnrollInfoAttr>(InPlaceUnrollInfoAttr::getKeyName());
    assert(inPlaceUnrollInfoAttr != nullptr);

    return inPlaceUnrollInfoAttr.getValue();
}

[[maybe_unused]] void SetInPlaceUnrollInfo(Operation* op, const InPlaceUnrollInfo& inPlaceUnrollInfo)
{
    op->setAttr(InPlaceUnrollInfoAttr::getKeyName(), InPlaceUnrollInfoAttr::get(inPlaceUnrollInfo, op->getContext()));
}

[[maybe_unused]] void SetInPlaceUnrollInfo(ScheduleOp op, Index index, const InPlaceUnrollInfo& inPlaceUnrollInfo)
{
    OpBuilder builder(op);
    auto inPlaceUnrollInfoIdentifier = builder.getIdentifier(InPlaceUnrollInfoAttr::getKeyName());
    op.addLoopAttribute(index, inPlaceUnrollInfoIdentifier, InPlaceUnrollInfoAttr::get(inPlaceUnrollInfo, builder.getContext()));
}

[[maybe_unused]] void SetInPlaceUnrollInfo(ScheduleOp op, SymbolicIndexOp index, const InPlaceUnrollInfo& inPlaceUnrollInfo)
{
    SetInPlaceUnrollInfo(op, index.getValue(), inPlaceUnrollInfo);
}

void RemoveInPlaceUnrollInfo(Operation* op)
{
    OpBuilder builder(op);
    auto inPlaceUnrollInfoIdentifier = builder.getIdentifier(InPlaceUnrollInfoAttr::getKeyName());
    op->removeAttr(inPlaceUnrollInfoIdentifier);
}

// Tensorization-related functions

bool HasTensorizationInfo(Operation* op)
{
    OpBuilder builder(op);
    auto tensorizationInfoIdentifier = builder.getIdentifier(TensorizationInfoAttr::getKeyName());
    auto tensorizationInfoAttr = op->getAttrOfType<TensorizationInfoAttr>(tensorizationInfoIdentifier);

    return tensorizationInfoAttr != nullptr;
}

TensorizationInfo GetTensorizationInfo(Operation* op)
{
    OpBuilder builder(op);
    auto tensorizationInfoIdentifier = builder.getIdentifier(TensorizationInfoAttr::getKeyName());
    auto tensorizeInfoAttr = op->getAttrOfType<TensorizationInfoAttr>(tensorizationInfoIdentifier);
    assert(tensorizeInfoAttr);

    return tensorizeInfoAttr.getValue();
}

void RemoveTensorizationInfo(Operation* op)
{
    OpBuilder builder(op);
    auto tensorizationInfoIdentifier = builder.getIdentifier(TensorizationInfoAttr::getKeyName());
    op->removeAttr(tensorizationInfoIdentifier);
}

// Parallelization-related functions

bool HasParallelizationInfo(Operation* op)
{
    OpBuilder builder(op);
    auto parallelizationInfoIdentifier = builder.getIdentifier(ParallelizationInfoAttr::getKeyName());
    auto parallelizationInfoAttr = op->getAttrOfType<ParallelizationInfoAttr>(parallelizationInfoIdentifier);

    return parallelizationInfoAttr != nullptr;
}

ParallelizationInfo GetParallelizationInfo(Operation* op)
{
    OpBuilder builder(op);
    auto parallelizationInfoIdentifier = builder.getIdentifier(ParallelizationInfoAttr::getKeyName());
    auto parallelizationInfoAttr = op->getAttrOfType<ParallelizationInfoAttr>(parallelizationInfoIdentifier);
    assert(parallelizationInfoAttr != nullptr);

    return parallelizationInfoAttr.getValue();
}

bool IsTerminalOp(mlir::Operation* op)
{
    // TODO: change this to also look for terminator ops
    return op->getNumResults() == 0;
}

mlir::Value CreateProductOfValues(OpBuilder& builder, Location loc, Type elementType, ValueRange values)
{
    mlir::Value currentProduct = builder.create<mlir::ConstantOp>(loc, util::GetOneAttr(builder, elementType));
    for (auto currentValue : values)
    {
        currentProduct = builder.create<v::BinOp>(loc, BinaryOpPredicate::MUL, currentProduct, currentValue);
    }
    return currentProduct;
}

std::optional<int64_t> GetDimSizeForBaseIndices(const std::vector<Index>& baseIndices, Operation* where)
{
    std::optional<int64_t> dimSize;
    for (const auto& baseIndex : baseIndices)
    {
        auto dimSizeOpt = util::GetDimSizeAt(baseIndex, where);
        if (dimSizeOpt.has_value())
        {
            if (dimSize.has_value())
            {
                assert(*dimSizeOpt == dimSize && "Each originating base index for a given index should have the same dim size");
            }
            else
            {
                dimSize = *dimSizeOpt;
            }
        }
    }
    return dimSize;
}

bool IsBoundsChecked(Operation* op)
{
    return op->getAttr(BoundsCheckedAttrName) != nullptr;
}

void SetBoundsChecked(OpBuilder& builder, Operation* op)
{
    op->setAttr(BoundsCheckedAttrName, builder.getUnitAttr());
}

template <typename LoadOrStoreOp>
bool HasOutOfBoundsAccess(LoadOrStoreOp op, mlir::Location loc)
{
    // This is a pared down version of mlir::boundCheckLoadOrStoreOp, which has a bug currently where it only returns failure (out of bounds)
    // if the last thing it checks has a failure, rather than anything it checks.

    mlir::MemRefRegion accessRegion(loc);
    auto memRefType = op.getMemRefType();
    unsigned rank = memRefType.getRank();
    (void)accessRegion.compute(op, 0, nullptr /*sliceState*/, false /*addMemRefDimBounds */);
    bool outOfBounds = false;
    // For each dimension, check for out of bounds.
    for (unsigned dim = 0; dim < rank; ++dim)
    {
        assert(!memRefType.isDynamicDim(dim) && "Dynamic dimensions are not currently supported");

        // Intersect memory region with constraint capturing out of bounds (both out
        // of upper and out of lower), and check if the constraint system is
        // feasible. If it is, there is at least one point out of bounds.

        // Check for overflow: d_i >= memref dim size.
        FlatAffineConstraints upperConstraints(*accessRegion.getConstraints());
        int64_t dimSize = memRefType.getDimSize(dim);
        upperConstraints.addConstantLowerBound(dim, dimSize);

        // Check for a negative index: d_i <= -1.
        FlatAffineConstraints lowerConstraints(*accessRegion.getConstraints());
        lowerConstraints.addConstantUpperBound(dim, -1);

        if (!upperConstraints.isEmpty() || !lowerConstraints.isEmpty())
        {
            outOfBounds = true;
            break;
        }
    }
    return outOfBounds;
}

// Returns whether left and right contain the same elements (possibly reordered)
template <typename ElementType>
bool ContainsSameElements(const std::vector<ElementType>& left, const std::vector<ElementType>& right)
{
    if (left.size() != right.size())
    {
        return false;
    }
    else if (left.empty() && right.empty())
    {
        return true;
    }
    else
    {
        for (const auto& element : left)
        {
            if (std::find(right.begin(), right.end(), element) == right.end())
            {
                return false;
            }
        }
    }
    return true;
}

struct LoopnestInfo
{
    std::vector<int64_t> baseIterationShape;
    std::vector<size_t> preferredTraversalOrder;
    std::vector<accera::ir::loopnest::Range> fullySplitRanges;
    std::vector<std::vector<int64_t>> splits; // outer vector: one entry per domain dimension, inner vector: one entry per split to perform (empty vector means no splits)
    std::vector<std::vector<int64_t>> indexOrder; // outer vector: one entry per domain dimension, inner vector: one entry per index with that base dimension giving its overall order position (single-element vector means no splits occured in this dim)
};

LoopnestInfo ConstructCacheLoopnestInfo(Operation* baseOp, const std::vector<IndexRange>& cacheRegionIndexRanges, const std::vector<std::vector<Index>>& cacheRegionBaseIndices)
{
    // Walk the cacheRegionIndexRanges in order to determine split sizes, use cacheRegionBaseIndices to determine which iteration domain dim the split originates from
    std::vector<std::vector<Index>> baseIndicesSeen;
    LoopnestInfo result;
    for (int64_t cacheRegionIdx = 0; cacheRegionIdx < (int64_t)cacheRegionIndexRanges.size(); ++cacheRegionIdx)
    {
        const auto& currentBaseIndices = cacheRegionBaseIndices[cacheRegionIdx];
        const auto& currentIndexRange = cacheRegionIndexRanges[cacheRegionIdx];
        auto findIter = std::find_if(baseIndicesSeen.begin(), baseIndicesSeen.end(), [&](const std::vector<Index>& alreadySeenBaseIndices) {
            return ContainsSameElements(alreadySeenBaseIndices, currentBaseIndices);
        });
        if (findIter == baseIndicesSeen.end())
        {
            int64_t shapeDimSize = static_cast<int64_t>(currentIndexRange.Size());
            auto dimSizeOpt = GetDimSizeForBaseIndices(currentBaseIndices, baseOp);
            if (dimSizeOpt.has_value())
            {
                shapeDimSize = std::min(*dimSizeOpt, shapeDimSize);
            }
            baseIndicesSeen.push_back(currentBaseIndices);
            result.baseIterationShape.push_back(shapeDimSize);
            result.splits.push_back(std::vector<int64_t>{}); // first time we're seeing this dimension, so no splits yet
            result.indexOrder.push_back(std::vector<int64_t>{ cacheRegionIdx });
        }
        else
        {
            size_t dimIdx = std::distance(baseIndicesSeen.begin(), findIter);
            result.splits[dimIdx].push_back(currentIndexRange.Size());
            result.indexOrder[dimIdx].push_back(cacheRegionIdx);
        }
        // TODO : these ranges don't account for clamping
        result.fullySplitRanges.push_back(currentIndexRange.GetRange());
    }
    return result;
}

IterationDomain CreateLoopNestIterationDomain(const std::vector<std::string>& domainDimNames,
                                              const std::vector<int64_t>& domainDimSizes)
{
    assert(domainDimNames.size() == domainDimSizes.size());
    std::vector<IndexRange> indexRanges;
    for (auto [domainDimName, domainDimSize] : llvm::zip(domainDimNames, domainDimSizes))
    {
        accera::ir::loopnest::Range dimRange(0, domainDimSize);
        indexRanges.emplace_back(domainDimName, dimRange);
    }
    return { indexRanges };
}

std::tuple<NestOp, ScheduleOp, ExecPlanOp> CreateCacheLoopnestHelper(
    OpBuilder& builder,
    Location loc,
    const LoopnestInfo& loopnestInfo,
    const std::vector<std::string>& activeBlockDimNames,
    const std::optional<VectorizationInfo>& vectorizationInfoOpt,
    int elementByteWidth,
    const v::ExecutionTarget& execTarget,
    const std::string& kernelSuffix,
    const std::function<void(OpBuilder&, const std::vector<mlir::Value>&, const std::vector<mlir::Value>&)>& kernelFn)
{
    // TODO : make this more like a loopnest that the DSL could create
    //        this currently requires all the split indices as separate values,
    //        which requires the schedule to be set before the kernel is created
    NestOp cacheNest;
    if (!activeBlockDimNames.empty())
    {
        // If we have dim names, then make the nest via a custom IterationDomain
        auto iterationDomain = CreateLoopNestIterationDomain(activeBlockDimNames, loopnestInfo.baseIterationShape);
        cacheNest = MakeNest(builder, iterationDomain);
    }
    else
    {
        cacheNest = MakeNest(builder, loopnestInfo.baseIterationShape);
    }
    auto cacheNestBodyBuilder = cacheNest.getBodyBuilder();

    auto cacheNestSchedule = cacheNest.getOrCreateSchedule();
    auto cacheNestSymbolicIndices = cacheNest.getIndices(cacheNestBodyBuilder);
    std::vector<mlir::Value> cacheRegionDomainIndices;
    std::copy(cacheNestSymbolicIndices.begin(), cacheNestSymbolicIndices.end(), std::back_inserter(cacheRegionDomainIndices));

    // create all the splits and set the order based on the cacheRegion info
    std::vector<std::vector<SymbolicIndexOp>> unorderedSplitIndices;
    size_t totalSplitIndexCount = 0;
    for (size_t domainDimIdx = 0; domainDimIdx < loopnestInfo.splits.size(); ++domainDimIdx)
    {
        std::vector<SymbolicIndexOp> currentDomainDimSplitIndices{ cacheNestSymbolicIndices[domainDimIdx] };
        totalSplitIndexCount++; // at least 1 index per domain dim
        for (size_t splitCount = 0; splitCount < loopnestInfo.splits[domainDimIdx].size(); ++splitCount)
        {
            auto [outerSplitIdx, innerSplitIdx] = cacheNestSchedule.split(currentDomainDimSplitIndices.back(), loopnestInfo.splits[domainDimIdx][splitCount]);
            currentDomainDimSplitIndices[currentDomainDimSplitIndices.size() - 1] = outerSplitIdx;
            currentDomainDimSplitIndices.push_back(innerSplitIdx);
            totalSplitIndexCount++; // 1 additional index per split
        }
        unorderedSplitIndices.push_back(currentDomainDimSplitIndices);
    }

    std::vector<mlir::Value> orderedSymbolicIndexOpValues(totalSplitIndexCount, mlir::Value());
    std::vector<Index> cacheNestScheduleOrder(totalSplitIndexCount, Index());

    if (loopnestInfo.preferredTraversalOrder.empty())
    {
        for (size_t domainDimIdx = 0; domainDimIdx < loopnestInfo.indexOrder.size(); ++domainDimIdx)
        {
            for (size_t splitIndexIdx = 0; splitIndexIdx < loopnestInfo.indexOrder[domainDimIdx].size(); ++splitIndexIdx)
            {
                auto position = loopnestInfo.indexOrder[domainDimIdx][splitIndexIdx];
                auto splitIndexSymbolicOp = unorderedSplitIndices[domainDimIdx][splitIndexIdx];
                assert(position < (int)cacheNestScheduleOrder.size());
                cacheNestScheduleOrder[position] = splitIndexSymbolicOp.getValue();
                orderedSymbolicIndexOpValues[position] = splitIndexSymbolicOp;
            }
        }
    }
    else
    {
        for (size_t posIdx = 0; posIdx < loopnestInfo.preferredTraversalOrder.size(); ++posIdx)
        {
            auto domainDimIdx = loopnestInfo.preferredTraversalOrder[posIdx];
            for (size_t splitIndexIdx = 0; splitIndexIdx < loopnestInfo.indexOrder[domainDimIdx].size(); ++splitIndexIdx)
            {
                auto splitIndexSymbolicOp = unorderedSplitIndices[domainDimIdx][splitIndexIdx];
                assert(posIdx < cacheNestScheduleOrder.size());
                cacheNestScheduleOrder[posIdx] = splitIndexSymbolicOp.getValue();
                orderedSymbolicIndexOpValues[posIdx] = splitIndexSymbolicOp;
            }
        }
    }
    cacheNestSchedule.setOrder(cacheNestScheduleOrder);

    // Now create the kernel using all of the split indices
    std::string kernelName = "cache_internal_loopnest_kernel_" + kernelSuffix;
    auto cacheNestKernel = MakeKernel(cacheNestBodyBuilder, kernelName, [&](mlir::OpBuilder& builder, mlir::Location) {
        kernelFn(builder, cacheRegionDomainIndices, orderedSymbolicIndexOpValues);
    });

    cacheNestSchedule.addKernel(cacheNestKernel);
    auto cacheNestExecPlanOp = cacheNestSchedule.getOrCreateExecPlan();
    auto execAttr = ExecutionTargetAttr::get(builder.getContext(), execTarget);
    cacheNest.exec_targetAttr(execAttr);
    cacheNestExecPlanOp.exec_targetAttr(execAttr);

    if (vectorizationInfoOpt.has_value())
    {
        // What we want is a per-op unroll surrounding a single loop vectorization

        auto vecInfo = *vectorizationInfoOpt;

        int64_t elementsPerVector = vecInfo.vectorBytes / elementByteWidth;
        int64_t budget = vecInfo.vectorUnitCount * elementsPerVector;

        if (budget > 0)
        {
            // Vectorize the innermost loop then apply in-place unrolling / per-op unrolling to the loops outside of it subject to the vectorization budget based on the size and number of vector registers
            SetVectorizationInfo(cacheNestSchedule, cacheNestScheduleOrder.back(), vecInfo);

            // reduce the budget based on how large the innermost vectorized loop is
            auto innermostLoopRange = loopnestInfo.fullySplitRanges.back();

            // If the budget is greater than the number of iterations of the innermost loop
            // then we can vectorize more loops in the nest
            if (budget > innermostLoopRange.NumIterations())
            {
                // Determine how much of the nest can be vectorized and set the vectorization info on those loops
                budget /= innermostLoopRange.NumIterations();
                int numVectorizedLoops = 1;
                for (size_t loopCounter = 1; loopCounter < loopnestInfo.fullySplitRanges.size(); ++loopCounter)
                {
                    size_t loopIdx = loopnestInfo.fullySplitRanges.size() - loopCounter - 1; // Vectorize loops from the innermost to the outermost as long as we still have vector registers to work with
                    auto loopRange = loopnestInfo.fullySplitRanges[loopIdx];
                    auto loopUnrollFactor = std::min(budget, loopRange.NumIterations());
                    InPlaceUnrollInfo inPlaceUnrollInfo{ loopUnrollFactor };
                    numVectorizedLoops++;
                    SetInPlaceUnrollInfo(cacheNestSchedule, cacheNestScheduleOrder[loopIdx], inPlaceUnrollInfo);
                    budget /= loopUnrollFactor;
                    if (budget <= 1) // if there is only 1 in-place op unroll left in the budget then we're done vectorizing
                    {
                        break;
                    }
                }
            }
        }
    }

    return { cacheNest, cacheNestSchedule, cacheNestExecPlanOp };
}

std::vector<size_t> GetMajorToMinorDimensionTraversal(const mlir::MemRefType& sourceType)
{
    llvm::SmallVector<int64_t, 4> strides;
    int64_t offset;
    auto strideResult = mlir::getStridesAndOffset(sourceType, strides, offset);
    assert(succeeded(strideResult));
    std::vector<std::pair<int64_t, size_t>> strideAndLogicalDims;
    size_t dim = 0;
    for (auto& stride : strides)
    {
        strideAndLogicalDims.push_back(std::make_pair(stride, dim++));
    }

    std::sort(strideAndLogicalDims.begin(), strideAndLogicalDims.end(), [](const std::pair<int64_t, size_t>& left, const std::pair<int64_t, size_t>& right) {
        // Want the larger strides ordered earlier
        return left.first > right.first;
    });

    std::vector<size_t> result;
    std::transform(strideAndLogicalDims.begin(), strideAndLogicalDims.end(), std::back_inserter(result), [](const std::pair<int64_t, size_t>& strideAndDim) {
        return strideAndDim.second;
    });
    return result;
}

std::tuple<NestOp, ScheduleOp, ExecPlanOp> CreateActiveBlockCacheLoopnest(
    mlir::OpBuilder& builder,
    Location loc,
    const mlir::MemRefType& sourceType,
    const std::vector<int64_t>& activeBlockShape,
    const std::vector<std::string>& activeBlockDimNames,
    const std::optional<VectorizationInfo>& vectorizationInfoOpt,
    int elementByteWidth,
    const v::ExecutionTarget& execTarget,
    const std::string& kernelSuffix,
    const std::function<void(OpBuilder&, const std::vector<mlir::Value>&, const std::vector<mlir::Value>&)>& kernelFn)
{
    LoopnestInfo loopnestInfo;
    loopnestInfo.baseIterationShape = activeBlockShape;

    std::transform(activeBlockShape.begin(), activeBlockShape.end(), std::back_inserter(loopnestInfo.fullySplitRanges), [&](int64_t indexRange) -> accera::ir::loopnest::Range {
        return accera::ir::loopnest::Range(0, indexRange, 1);
    });
    std::transform(activeBlockShape.begin(), activeBlockShape.end(), std::back_inserter(loopnestInfo.splits), [&](int64_t indexRange) -> std::vector<int64_t> {
        return std::vector<int64_t>(); // TODO : Do we want to create any splits for efficiency reasons?
    });
    int64_t idx = 0;
    std::transform(activeBlockShape.begin(), activeBlockShape.end(), std::back_inserter(loopnestInfo.indexOrder), [&](int64_t indexRange) -> std::vector<int64_t> {
        return { idx++ }; // TODO : Do we want to make any reorders for efficiency reasons?
    });

    // Set the preferred traversal order based on the strides in the array we're copying from. Larger strides -> earlier in the schedule
    // The activeBlockShape is given in logical ordering, so the preferred traversal order is a permutation array of those dimensions
    if (auto rank = sourceType.getRank(); rank >= 0 && (size_t)rank == activeBlockShape.size())
    {
        // Only set the preferred order if we're creating a loopnest with the same number of dimensions as our source memref type
        loopnestInfo.preferredTraversalOrder = GetMajorToMinorDimensionTraversal(sourceType);
    }

    std::string fullKernelSuffix = "active_block_" + kernelSuffix;
    return CreateCacheLoopnestHelper(builder, loc, loopnestInfo, activeBlockDimNames, vectorizationInfoOpt, elementByteWidth, execTarget, fullKernelSuffix, kernelFn);
}

template <typename CacheOp>
std::tuple<NestOp, ScheduleOp, ExecPlanOp> CreateActiveElementCacheLoopnest(OpBuilder& builder,
                                                                            CacheOp cacheOp,
                                                                            int elementByteWidth,
                                                                            const std::string& kernelSuffix,
                                                                            const std::function<void(OpBuilder&, const std::vector<mlir::Value>&, const std::vector<mlir::Value>&)>& kernelFn)
{
    auto loc = cacheOp.getLoc();

    std::optional<v::ExecutionTarget> execTargetOpt = util::ResolveExecutionTarget(cacheOp);
    assert(execTargetOpt.has_value());
    auto execTarget = *execTargetOpt;

    auto vLambdaOp = cacheOp->template getParentOfType<v::ValueLambdaOp>();

    std::optional<VectorizationInfo> vectorizationInfoOpt;
    if (vLambdaOp && HasVectorizationInfo(vLambdaOp))
    {
        vectorizationInfoOpt = GetVectorizationInfo(vLambdaOp);
    }
    // TODO : make this a utility function on a cache op interface
    auto cacheRegionIndexRanges = util::ArrayAttrToVector<IndexRange, IndexRangeAttr>(cacheOp.cacheRegionRelevantIndexRanges(),
                                                                                      [](const IndexRangeAttr& indexRangeAttr) {
                                                                                          return indexRangeAttr.getValue();
                                                                                      });

    auto cacheRegionBaseIndices = util::ArrayAttrToVector<std::vector<Index>, mlir::ArrayAttr>(
        cacheOp.cacheRegionBaseIndices(),
        util::ConvertArrayAttrToIndexVector);
    assert(cacheRegionIndexRanges.size() == cacheRegionBaseIndices.size());

    LoopnestInfo loopnestInfo = ConstructCacheLoopnestInfo(cacheOp, cacheRegionIndexRanges, cacheRegionBaseIndices);

    std::string fullKernelSuffix = "active_element_" + kernelSuffix;
    return CreateCacheLoopnestHelper(builder, loc, loopnestInfo, {}, vectorizationInfoOpt, elementByteWidth, execTarget, fullKernelSuffix, kernelFn);
}

// Contains shape and access information for an active block of an array
struct ActiveBlockInfo
{
    int64_t activeBlockVolume;
    std::vector<int64_t> shape;
    std::vector<mlir::Value> externalSymbols;
    std::vector<mlir::AffineMap> lbMaps;
    std::vector<mlir::AffineMap> ubMaps;
    mlir::AffineMap activeBlockOffsetMap;
};

// Contains the MemRefRegion and access characteristics of an active block
struct ArrayAccessInfo
{
    ArrayAccessInfo(mlir::Location loc) :
        activeBlock(loc) {}
    mlir::Value array;
    mlir::MemRefRegion activeBlock;
    bool valueWritten = false;
    bool valueRead = false;
    bool onlyReadsAreAccumulates = true;
    bool cacheUsedInRegion = false;
};

// Contains multi-cache information, such as the combined ArrayAccessInfo and ActiveBlockInfo
// as well as the multi-cache shape information, cache reference, and access information for each
// different active block region that is part of the same multicache
struct MultiCacheInfo
{
    MultiCacheInfo(mlir::Location loc) :
        arrayAccessInfo(loc) {}

    struct ActiveBlockRegionInfo
    {
        std::vector<mlir::Value> allCacheExternalSymbols;
        CacheAccessContext cacheAccessContext;
    };
    std::unordered_map<mlir::Operation*, ActiveBlockRegionInfo> activeBlockRegionInfos;
    MakeCacheOp originalCacheOp;
    MakeCacheOp multiCache;
    ArrayAccessInfo arrayAccessInfo;
    ActiveBlockInfo activeBlockInfo;
    mlir::AffineMap activeBlockToCacheMap;
    mlir::AffineMap multiCacheExternalSymbolsPermutationMap;
    std::vector<mlir::Value> multiCacheExternalSymbols;
    std::vector<mlir::AffineMap> multiCacheLBMaps;
    std::vector<mlir::AffineMap> multiCacheUBMaps;
    std::vector<int64_t> multiCacheStepSizes;
    std::vector<Index> multiCacheLoopIndexIds;
    std::vector<uint64_t> multiCacheIterationCounts;
    std::vector<mlir::AffineForOp> multiCacheLoops;
};

std::pair<mlir::Value, mlir::ValueRange> GetAccessValueAndIndices(Operation* loadOrStoreOp)
{
    bool isLoadOrStore = isa<memref::StoreOp, mlir::AffineStoreOp, v::StoreOp, v::MMAStoreSyncOp, mlir::memref::LoadOp, mlir::AffineLoadOp, v::LoadOp, v::MMALoadSyncOp>(loadOrStoreOp);
    assert(isLoadOrStore);
    if (auto stdStoreOp = dyn_cast_or_null<memref::StoreOp>(loadOrStoreOp))
    {
        memref::StoreOp::Adaptor adaptor{ stdStoreOp };
        return std::make_pair(adaptor.memref(), adaptor.indices());
    }
    else if (auto affineStoreOp = dyn_cast_or_null<mlir::AffineStoreOp>(loadOrStoreOp))
    {
        mlir::AffineStoreOp::Adaptor adaptor{ affineStoreOp };
        return std::make_pair(adaptor.memref(), adaptor.indices());
    }
    else if (auto valueStoreOp = dyn_cast_or_null<v::StoreOp>(loadOrStoreOp))
    {
        v::StoreOp::Adaptor adaptor{ valueStoreOp };
        return std::make_pair(adaptor.memref(), adaptor.indices());
    }
    else if (auto valueMMAStoreSyncOp = dyn_cast_or_null<v::MMAStoreSyncOp>(loadOrStoreOp))
    {
        v::MMAStoreSyncOp::Adaptor adaptor{ valueMMAStoreSyncOp };
        return std::make_pair(adaptor.memref(), adaptor.indices());
    }
    else if (auto stdLoadOp = dyn_cast_or_null<memref::LoadOp>(loadOrStoreOp))
    {
        memref::LoadOp::Adaptor adaptor{ stdLoadOp };
        return std::make_pair(adaptor.memref(), adaptor.indices());
    }
    else if (auto affineLoadOp = dyn_cast_or_null<mlir::AffineLoadOp>(loadOrStoreOp))
    {
        mlir::AffineLoadOp::Adaptor adaptor{ affineLoadOp };
        return std::make_pair(adaptor.memref(), adaptor.indices());
    }
    else if (auto valueLoadSyncOp = dyn_cast_or_null<v::LoadOp>(loadOrStoreOp))
    {
        v::LoadOp::Adaptor adaptor{ valueLoadSyncOp };
        return std::make_pair(adaptor.memref(), adaptor.indices());
    }
    else if (auto valueMMALoadSyncOp = dyn_cast_or_null<v::MMALoadSyncOp>(loadOrStoreOp))
    {
        v::MMALoadSyncOp::Adaptor adaptor{ valueMMALoadSyncOp };
        return std::make_pair(adaptor.memref(), adaptor.indices());
    }
    assert(false && "Unhandled load/store case");
}

bool ComputeRegionAccessedByOp(PatternRewriter& rewriter, mlir::MemRefRegion& activeBlockRegion, mlir::Operation* op, unsigned loopDepth)
{
    mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPoint(op);

    if (isa<mlir::AffineLoadOp, mlir::AffineStoreOp, v::MMALoadSyncOp, v::MMAStoreSyncOp>(op))
    {
        auto result = activeBlockRegion.compute(op, loopDepth, nullptr, false);
        assert(succeeded(result));
        return true;
    }
    return false;
}

// Computes the active block for the array for the ops in the graph in half-open graph interval [startOp, endOp)
ArrayAccessInfo ComputeAccessInfoForArrayAtLevel(PatternRewriter& rewriter, mlir::Value array, mlir::Block::iterator startOp, mlir::Block::iterator endOp, bool computeActiveBlock)
{
    auto loc = startOp->getLoc();
    unsigned loopDepth = mlir::getNestingDepth(&(*startOp));
    bool firstMemRefRegionSeen = true;

    auto parentBlock = startOp->getBlock();

    ArrayAccessInfo result(loc);
    result.array = array;
    for (Operation* arrayUserOp : array.getUsers())
    {
        // Check if this use is inside of the cache region
        bool isInRegion = false;
        // TODO : fix this pattern, this seems like a really inefficient way of doing this
        parentBlock->walk(mlir::Block::iterator(startOp), mlir::Block::iterator(endOp), [&](Operation* op) {
            if (op == arrayUserOp)
            {
                isInRegion = true;
                result.cacheUsedInRegion = true;
                return WalkResult::interrupt();
            }
            else
            {
                return WalkResult::advance();
            }
        });

        if (isInRegion)
        {
            // while we're examining this op, compute the active block that is accessed by this op within the cache region
            mlir::MemRefRegion activeBlockRegion(loc);

            // TODO : make value load/store implement load/store interfaces from std dialect
            if (isa<mlir::memref::StoreOp, mlir::AffineStoreOp, v::StoreOp, v::MMAStoreSyncOp>(arrayUserOp))
            {
                result.valueWritten = true;
            }
            if (isa<mlir::memref::LoadOp, mlir::AffineLoadOp, v::LoadOp, v::MMALoadSyncOp>(arrayUserOp))
            {
                result.valueRead = true;
                if (result.onlyReadsAreAccumulates)
                {
                    // Check if this read is only used for a simple accumulation, i.e.:
                    // %0 = some value...
                    // %1 = load %array[indices]
                    // %2 = add %0, %1
                    // store %2 %array[indices]
                    // Note: the load and store memrefs and indices must match, otherwise it's writing to a different location or a different memref and is not a simple accumulate

                    assert(arrayUserOp->getNumResults() == 1);
                    auto loadedValue = arrayUserOp->getResult(0);
                    auto inputValueAndIndices = GetAccessValueAndIndices(arrayUserOp);

                    for (auto loadedValueUser : loadedValue.getUsers())
                    {
                        if (isa<v::BinOp, mlir::AddFOp, mlir::AddIOp>(loadedValueUser))
                        {
                            if (auto binOp = dyn_cast<v::BinOp>(loadedValueUser))
                            {
                                if (binOp.predicate() != v::BinaryOpPredicate::ADD)
                                {
                                    result.onlyReadsAreAccumulates = false;
                                    break;
                                }
                            }
                            auto addResult = loadedValueUser->getResult(0);

                            for (auto addedValueUser : addResult.getUsers())
                            {
                                if (!isa<mlir::memref::StoreOp, mlir::AffineStoreOp, v::StoreOp, v::MMAStoreSyncOp>(addedValueUser))
                                {
                                    result.onlyReadsAreAccumulates = false;
                                    break;
                                }
                                // Check that the destination is the same as the source
                                auto dstValueAndIndices = GetAccessValueAndIndices(addedValueUser);
                                if (dstValueAndIndices.first != inputValueAndIndices.first || dstValueAndIndices.second != inputValueAndIndices.second)
                                {
                                    result.onlyReadsAreAccumulates = false;
                                    break;
                                }
                            }
                            if (!result.onlyReadsAreAccumulates)
                            {
                                // The inner loop was broken out of, now break of this loop
                                break;
                            }
                        }
                        else if (isa<mlir::memref::StoreOp, mlir::AffineStoreOp, v::StoreOp, v::MMAStoreSyncOp>(loadedValueUser))
                        {
                            // If it is just a load-and-store without any other uses of the loaded value, then treat that as a simple accumulate equivalent to accumulating the value 0
                            auto dstValueAndIndices = GetAccessValueAndIndices(loadedValueUser);
                            if (dstValueAndIndices.first != inputValueAndIndices.first || dstValueAndIndices.second != inputValueAndIndices.second)
                            {
                                result.onlyReadsAreAccumulates = false;
                                break;
                            }
                        }
                        else
                        {
                            result.onlyReadsAreAccumulates = false;
                            break;
                        }
                    }
                }
            }
            if (computeActiveBlock)
            {
                if (ComputeRegionAccessedByOp(rewriter, activeBlockRegion, arrayUserOp, loopDepth))
                {
                    if (firstMemRefRegionSeen)
                    {
                        result.activeBlock = activeBlockRegion;
                        firstMemRefRegionSeen = false;
                    }
                    else
                    {
                        auto unionResult = result.activeBlock.unionBoundingBox(activeBlockRegion);
                        assert(succeeded(unionResult));

                        result.activeBlock.cst.removeRedundantConstraints();
                    }
                }
            }
        }
    }
    return result;
}

int64_t GetActiveBlockVolume(const mlir::MemRefRegion& activeBlock)
{
    if (!activeBlock.memref)
    {
        return 0;
    }
    mlir::SmallVector<int64_t, 4> shape;
    std::vector<mlir::SmallVector<int64_t, 4>> lbs;
    mlir::SmallVector<int64_t, 4> lbDivisors;

    auto activeBlockVolumeOpt = activeBlock.getConstantBoundingSizeAndShape(&shape, &lbs, &lbDivisors);
    assert(activeBlockVolumeOpt.hasValue());
    return activeBlockVolumeOpt.getValue();
}

ActiveBlockInfo ConvertMemRefRegionToActiveBlockInfo(OpBuilder& builder, const mlir::MemRefRegion& activeBlockRegion)
{
    ActiveBlockInfo activeBlockInfo;

    // This shape computation is duplicated from GetActiveBlockVolume() since this function is the only one that needs the shape, lbs, and lbDivisors
    mlir::SmallVector<int64_t, 4> shape;
    std::vector<mlir::SmallVector<int64_t, 4>> lbs;
    mlir::SmallVector<int64_t, 4> lbDivisors; // TODO : do we have scenarios where we need to consider these?

    auto activeBlockVolumeOpt = activeBlockRegion.getConstantBoundingSizeAndShape(&shape, &lbs, &lbDivisors);
    assert(activeBlockVolumeOpt.hasValue());
    activeBlockInfo.activeBlockVolume = activeBlockVolumeOpt.getValue();

    activeBlockInfo.shape.insert(activeBlockInfo.shape.begin(), shape.begin(), shape.end());

    std::vector<std::vector<int64_t>> lbsVec;
    std::transform(lbs.begin(), lbs.end(), std::back_inserter(lbsVec), [](const mlir::SmallVector<int64_t, 4>& vec) { return std::vector<int64_t>(vec.begin(), vec.end()); });

    unsigned rank = activeBlockRegion.getRank();
    activeBlockInfo.lbMaps.resize(rank);
    activeBlockInfo.ubMaps.resize(rank);
    for (auto dim = 0u; dim < rank; ++dim)
    {
        activeBlockRegion.getLowerAndUpperBound(dim, activeBlockInfo.lbMaps[dim], activeBlockInfo.ubMaps[dim]);
    }

    const FlatAffineConstraints* cst = activeBlockRegion.getConstraints();
    mlir::SmallVector<mlir::Value, 8> regionSymbols;
    cst->getIdValues(rank, cst->getNumIds(), &regionSymbols);
    std::vector<mlir::Value> regionSymbolsVec;
    regionSymbolsVec.insert(regionSymbolsVec.end(), regionSymbols.begin(), regionSymbols.end());
    activeBlockInfo.externalSymbols = regionSymbolsVec;

    // Adapted from generateCopy() in llvm-project\mlir\lib\Transforms\Utils\LoopUtils.cpp
    // Index start offsets for active block relative to the original array
    std::vector<mlir::AffineExpr> activeBlockOffsetExprs;
    activeBlockOffsetExprs.reserve(rank);
    int64_t numConstraints = cst->getNumCols() - rank - 1;
    for (unsigned arrayDim = 0; arrayDim < rank; arrayDim++)
    {
        assert(lbs[arrayDim].size() == cst->getNumCols() - rank && "incorrect bound size");

        mlir::AffineExpr offset = builder.getAffineConstantExpr(0);
        for (unsigned constraintCol = 0; constraintCol < numConstraints; constraintCol++)
            offset = offset + lbs[arrayDim][constraintCol] * builder.getAffineDimExpr(constraintCol);
        assert(lbDivisors[arrayDim] > 0);
        offset = (offset + lbs[arrayDim][cst->getNumCols() - 1 - rank]).floorDiv(lbDivisors[arrayDim]);

        // Record the offsets since they are needed to remap the memory accesses of
        // the original memref further below.
        activeBlockOffsetExprs.push_back(offset);
    }

    activeBlockInfo.activeBlockOffsetMap = mlir::AffineMap::get(numConstraints, 0, activeBlockOffsetExprs, builder.getContext());

    return activeBlockInfo;
}

MakeCacheOp UpdateActiveBlockCacheShape(PatternRewriter& rewriter,
                                        MakeCacheOp baseMakeCacheOp,
                                        const CacheAccessContext& cacheAccessContext,
                                        const ActiveBlockInfo& activeBlockInfo,
                                        const std::vector<uint64_t>& multiCacheShape)
{
    // TODO : detect when the access region exceeds the MemRefRegion and pad the memref

    // Duplicate the given MakeCacheOp for this cache and give it a large enough 1-D memref to hold activeBlockVolume
    auto currentCacheType = baseMakeCacheOp.cache().getType();
    assert(currentCacheType.isa<mlir::MemRefType>());
    auto currentCacheMemRefType = currentCacheType.cast<mlir::MemRefType>();
    if (!cacheAccessContext.dimReorderCache)
    {
        assert(currentCacheMemRefType.getRank() == 1 && "Active block caches with custom coefficients should be 1-dimensional");
    }

    auto cacheShape = currentCacheMemRefType.getShape().vec();

    if (cacheAccessContext.dimReorderCache)
    {
        assert((cacheShape[0] == DynamicSizeSentinelValue || cacheShape.size() == activeBlockInfo.shape.size()) && "Inconsistent cache rank");

        // reorder cacheShape based on the dimension reorder
        auto reorderVec = cacheAccessContext.accessMaps.dimOrder.ToVector();
        assert(reorderVec.size() == activeBlockInfo.shape.size());
        cacheShape.resize(activeBlockInfo.shape.size(), DynamicSizeSentinelValue);
        for (unsigned cacheDimIdx = 0; cacheDimIdx < activeBlockInfo.shape.size(); ++cacheDimIdx)
        {
            cacheShape[cacheDimIdx] = std::max(cacheShape[cacheDimIdx], activeBlockInfo.shape[reorderVec[cacheDimIdx]]);
        }
    }
    else
    {
        assert(cacheShape.size() == 1 && "Affine coefficient caches must be rank 1 buffers");
        int64_t volumePlusOffset = activeBlockInfo.activeBlockVolume + cacheAccessContext.accessMaps.coefficients.offset;
        if (cacheShape[0] == DynamicSizeSentinelValue)
        {
            cacheShape[0] = volumePlusOffset;
        }
        else
        {
            cacheShape[0] = std::max(cacheShape[0], volumePlusOffset);
        }
    }
    // insert the multiCache dimensions as the outer dimensions
    cacheShape.insert(cacheShape.begin(), multiCacheShape.begin(), multiCacheShape.end());

    auto newCacheType = mlir::MemRefType::get(cacheShape, currentCacheMemRefType.getElementType(), {}, currentCacheMemRefType.getMemorySpace());

    mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPoint(baseMakeCacheOp);
    auto replacementOp = rewriter.create<MakeCacheOp>(baseMakeCacheOp.getLoc(), newCacheType, baseMakeCacheOp.memorySpace());
    return replacementOp;
}

mlir::AffineMap CreateActiveBlockToCacheMap(PatternRewriter& rewriter,
                                            CacheAccessContext& cacheAccessContext)
{
    mlir::AffineMap activeBlockToCacheMap;
    if (cacheAccessContext.dimReorderCache)
    {
        auto dimReorderVec = cacheAccessContext.accessMaps.dimOrder.ToVector();
        std::vector<unsigned int> unsignedDimReorderVec(dimReorderVec.begin(), dimReorderVec.end());
        activeBlockToCacheMap = mlir::AffineMap::getPermutationMap(unsignedDimReorderVec, rewriter.getContext());
    }
    else
    {
        activeBlockToCacheMap = ComputeFlatAffineMapFromAffineCoefficients(rewriter, cacheAccessContext.accessMaps.coefficients);
    }
    return activeBlockToCacheMap;
}

mlir::AffineMap CreateArrayToCacheMap(mlir::OpBuilder& builder,
                                      size_t arrayRank,
                                      mlir::AffineMap activeBlockOffsetMap,
                                      mlir::AffineMap activeBlockToCacheMap,
                                      size_t offsetAccessIndexCount,
                                      size_t multiCacheDimCount)
{
    // Our final map has (offsetAccessIndexCount + arrayRank) inputs and (arrayRank) results unless it is a multicache,
    // in which case it has (multiCacheDimCount + offsetAccessIndexCount + arrayRank) inputs and (multiCacheDimCount + arrayRank) results

    // Create a simple passthrough map for the base array position values
    auto passthroughMap = mlir::AffineMap::getMultiDimIdentityMap(arrayRank, builder.getContext());

    // The activeBlockOffsetMap maps from the loop indices used to perform an offset into the base array (the "offset access indices") to the values
    // for each dimension that the base array position should be offset to access the active block.

    // Now concatenate the passthrough map onto the activeBlockOffsetMap to produce a map that maps from ( offset access index IVs..., base array positions... ) -> ( offset amounts..., base array positions... )
    auto activeBlockOffsetWithPassthroughMap = util::ConcatenateAndShiftAffineDimsAndMaps(builder, activeBlockOffsetMap, passthroughMap);
    assert(activeBlockOffsetWithPassthroughMap.getNumResults() == 2 * arrayRank);

    // Now create a map that subtracts each offset amount from the corresponding base array position
    std::vector<mlir::AffineExpr> subtractOffsetExprs;
    for (auto dimIdx = 0u; dimIdx < arrayRank; ++dimIdx)
    {
        subtractOffsetExprs.push_back(builder.getAffineDimExpr(dimIdx + arrayRank) - builder.getAffineDimExpr(dimIdx));
    }
    auto offsetSubtractMap = mlir::AffineMap::get(2 * arrayRank, 0, subtractOffsetExprs, builder.getContext());

    // Now compose the activeBlockOffsetWithPassthroughMap and offsetSubtractMap to map from ( offset access index IVs..., base array positions... ) -> ( active block position... )
    auto offsetAccessIVsArrayIndicesToActiveBlockPositionMap = offsetSubtractMap.compose(activeBlockOffsetWithPassthroughMap);

    // Now map from the active block position to the cache position
    auto fullActiveBlockCacheMap = activeBlockToCacheMap.compose(offsetAccessIVsArrayIndicesToActiveBlockPositionMap);

    // Now account for multicache slicing

    // Insert the multiCache slice operands into the map and operands
    // Shift the dimensions used in the fullActiveBlockCacheMap so that the first several
    // dimensions are the multiCache slicing dimensions
    auto multiCacheSliceMap = mlir::AffineMap::getMultiDimIdentityMap(multiCacheDimCount, builder.getContext());
    auto fullCacheMap = util::ConcatenateAndShiftAffineDimsAndMaps(builder, multiCacheSliceMap, fullActiveBlockCacheMap);

    // Now fullCacheMap maps ( multiCache indices..., offset access index IVs..., base array positions... ) -> ( multicache active block position... )
    return fullCacheMap;
}

MakeCacheOp UpdateActiveBlockCacheAccess(PatternRewriter& rewriter,
                                         MakeCacheOp shapedMakeCacheOp,
                                         size_t arrayRank,
                                         mlir::AffineMap activeBlockOffsetMap,
                                         mlir::AffineMap activeBlockToCacheMap,
                                         const std::vector<Index>& offsetAccessIndices,
                                         const std::vector<Index>& multiCacheAccessIndices)
{
    mlir::AffineMap arrayToCacheMap = CreateArrayToCacheMap(rewriter,
                                                            arrayRank,
                                                            activeBlockOffsetMap,
                                                            activeBlockToCacheMap,
                                                            offsetAccessIndices.size(),
                                                            multiCacheAccessIndices.size());

    mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPoint(shapedMakeCacheOp);
    auto replacementOp = rewriter.create<MakeCacheOp>(shapedMakeCacheOp.getLoc(),
                                                      shapedMakeCacheOp.getType(),
                                                      shapedMakeCacheOp.memorySpace(),
                                                      arrayToCacheMap,
                                                      offsetAccessIndices,
                                                      multiCacheAccessIndices);

    rewriter.eraseOp(shapedMakeCacheOp);
    return replacementOp;
}

void UpdateCacheAccessContextForActiveBlockCache(PatternRewriter& rewriter,
                                                 CacheAccessContext& cacheAccessContext,
                                                 const ActiveBlockInfo& activeBlockInfo,
                                                 mlir::AffineMap activeBlockToCacheMap,
                                                 const std::vector<mlir::Value>& multiCacheExternalSymbols)
{
    // Update the cache region relevant schedule index ranges and the access maps in the cacheAccessContext
    // based on the given ActiveBlockInfo

    std::vector<IndexRange> activeBlockIndexRanges;
    auto rank = activeBlockInfo.shape.size();
    activeBlockIndexRanges.reserve(rank);
    for (size_t activeBlockDim = 0; activeBlockDim < rank; ++activeBlockDim)
    {
        accera::ir::loopnest::Range dimRange(0, activeBlockInfo.shape[activeBlockDim]);
        std::string dimName = "active_block_dim_" + std::to_string(activeBlockDim);
        activeBlockIndexRanges.emplace_back(dimName, dimRange);
    }

    cacheAccessContext.cacheRegionRelevantScheduleIndexRanges = activeBlockIndexRanges;

    std::vector<mlir::AffineExpr> remapExprs;
    remapExprs.reserve(rank);
    for (unsigned i = 0; i < rank; i++)
    {
        // The starting operands of indexRemap will be the active block operands (the symbols on
        // which the memref region is parametric); then those corresponding to
        // the memref's original indices follow.
        auto dimExpr = rewriter.getAffineDimExpr(activeBlockInfo.externalSymbols.size() + i);
        auto offsetExpr = activeBlockInfo.activeBlockOffsetMap.getResult(i);
        remapExprs.push_back(dimExpr - offsetExpr);
    }
    auto indexRemap = AffineMap::get(activeBlockInfo.externalSymbols.size() + rank, 0, remapExprs, rewriter.getContext());

    cacheAccessContext.externalRelevantScheduleIndices = activeBlockInfo.externalSymbols;
    auto inputToCacheMap = activeBlockToCacheMap.compose(indexRemap);
    cacheAccessContext.accessMaps.inputIndicesToActiveBlockCache = inputToCacheMap;
}

bool ShouldMergeMultiCacheInfos(const MultiCacheInfo& lhs, const MultiCacheInfo& rhs)
{
    // If either cache usage is not read-only, then they have to be merged
    // If both are read-only caches, then check if their active blocks
    // have the same shape and external symbols, in which case they can be merged

    // In general we should prefer to merge cache buffers to reduce complexity and unnecessary memory bloat.
    // Only when the caches are read-only or accumulate-only and have differing shapes then we should leave them as separate bufers

    bool cachesAreReadOnly = !lhs.arrayAccessInfo.valueWritten && !rhs.arrayAccessInfo.valueWritten;
    bool cachesAreAccumulateOnly = lhs.arrayAccessInfo.onlyReadsAreAccumulates && rhs.arrayAccessInfo.onlyReadsAreAccumulates;
    bool cachesHaveSameShape = lhs.activeBlockInfo.shape.size() == rhs.activeBlockInfo.shape.size() &&
                               std::equal(lhs.activeBlockInfo.shape.begin(), lhs.activeBlockInfo.shape.end(), rhs.activeBlockInfo.shape.begin());
    bool multiCachesHaveSameShape = std::equal(lhs.multiCacheIterationCounts.begin(), lhs.multiCacheIterationCounts.end(), rhs.multiCacheIterationCounts.begin());

    bool cachesHaveSameExternalSymbols = lhs.multiCacheExternalSymbols.size() == rhs.multiCacheExternalSymbols.size() &&
                                         std::equal(lhs.multiCacheExternalSymbols.begin(), lhs.multiCacheExternalSymbols.end(), rhs.multiCacheExternalSymbols.begin());

    bool sameShapeAndSymbols = multiCachesHaveSameShape && cachesHaveSameShape && cachesHaveSameExternalSymbols;

    bool keepSeparateCacheBuffers = (cachesAreReadOnly || cachesAreAccumulateOnly) && !sameShapeAndSymbols;

    return !keepSeparateCacheBuffers;
}

mlir::Value GetOriginalIV(mlir::Value possiblyOffsetIV)
{
    // Requires that possiblyOffsetIV is constructed from a single IV and constants
    if (possiblyOffsetIV.isa<mlir::BlockArgument>())
    {
        return possiblyOffsetIV;
    }
    else
    {
        auto definingOp = possiblyOffsetIV.getDefiningOp();
        assert(definingOp != nullptr);
        if (auto affineApplyOp = mlir::dyn_cast<mlir::AffineApplyOp>(definingOp))
        {
            for (auto operand : affineApplyOp.getOperands())
            {
                if (auto originalIV = GetOriginalIV(operand))
                {
                    return originalIV;
                }
            }
            return nullptr;
        }
        else if (auto constantOp = mlir::dyn_cast<mlir::ConstantOp>(definingOp))
        {
            return nullptr;
        }
        else
        {
            assert(false && "Offset IVs must be offset with AffineApplyOps and constants");
        }
    }
}

mlir::AffineMap ComputeLoopIVToDefinitionOrderMap(const std::vector<mlir::Value>& ivs, mlir::MLIRContext* context)
{
    // This is currently limited to nested AffineForOp induction variables for simplicity

    // returns the permutation map that would map from the current order of the IVs to the order they were defined in
    // e.g. if the values are loop indices
    //      for %arg0 ...
    //          for %arg1 ...
    //              for %arg2 ...
    //                  for %arg3 ...
    // and they are passed in in the order [%arg1, %arg3, %arg0, %arg2]
    // Then this order has the positions [0,1,2,3] and the order they are defined in would be [ 2, 0, 3, 1 ]

    std::vector<unsigned> creationOrderIndices(ivs.size());
    std::iota(creationOrderIndices.begin(), creationOrderIndices.end(), 0);
    std::sort(creationOrderIndices.begin(), creationOrderIndices.end(), [&](unsigned currentIdx, unsigned otherIdx) {
        // returns true if current is ordered before other, which in this case happens if
        // current is defined in a higher loop level than other
        const auto& currentIV = ivs[currentIdx];
        const auto& otherIV = ivs[otherIdx];
        auto currentOriginalIV = GetOriginalIV(currentIV);
        auto otherOriginalIV = GetOriginalIV(otherIV);
        auto currentDefiningOp = mlir::getForInductionVarOwner(currentOriginalIV);
        auto otherDefiningOp = mlir::getForInductionVarOwner(otherOriginalIV);
        assert(currentDefiningOp != nullptr);
        assert(otherDefiningOp != nullptr);
        bool currentIsAncestor = currentDefiningOp->isAncestor(otherDefiningOp);
        bool otherIsAncestor = otherDefiningOp->isAncestor(currentDefiningOp);
        assert((currentIsAncestor || otherIsAncestor) && "ComputeLoopIVDefinitionOrder only works on nested AffineForOp IVs");
        return currentIsAncestor;
    });

    return mlir::AffineMap::getPermutationMap(creationOrderIndices, context);
}

mlir::AffineMap ComputeLoopIVDefinitionOrderToCurrentOrderMap(const std::vector<mlir::Value>& ivs, mlir::MLIRContext* context)
{
    // This is currently limited to nested AffineForOp induction variables for simplicity

    // returns the permutation that would map from the definition order of the given IVs to their positions in the given vector
    // e.g. if the values are loop indices
    //      for %arg0 ...
    //          for %arg1 ...
    //              for %arg2 ...
    //                  for %arg3 ...
    // and they are passed in in the order [%arg1, %arg3, %arg0, %arg2]
    // Then this order has the positions [0,1,2,3] and the order they are defined in would be [ 2, 0, 3, 1 ]
    // And the permutation map (d0, d1, d2, d3) -> (1, 3, 0, 2) would map from their definition order to their current order

    // if these same loop IVs are passed in in the order [%arg1, %arg0, %arg3, %arg2]
    // Then this order has the positions [0,1,2,3] and the order they are defined in would be [ 1, 0, 3, 2 ]
    // And the permutation map (d0, d1, d2, d3) -> (1, 0, 3, 2) would map from their definition order to their current order

    // The permutation map that maps from the definition order to the current order is the inverse of the map returned by ComputeLoopIVToDefinitionOrderMap()
    // For example: for arg order [%arg1, %arg3, %arg0, %arg2], the definition order is [2,0,3,1]
    //      because element 2 of those IVs (%arg0) is defined first, element 0 (%arg1) is defined second, etc.
    //      however the mapping that would take [%arg0, %arg1, %arg2, %arg3] in definition order and
    //      permute it to the given [%arg1, %arg3, %arg0, %arg2] order is (d0,d1,d2,d3) -> (1,3,0,2).
    //      If we consider the definition order as a mapping (d0,d1,d2,d3)->(d2,d0,d3,d1)
    //      and then invert it we get (d2,d0,d3,d1) -> (d0,d1,d2,d3)
    //      which simplifies to (d0,d1,d2,d3) -> (d1,d3,d0,d2)
    // Another example: for arg order [%arg1, %arg0, %arg3, %arg2], the definition order is [1,0,3,2].
    //      The mapping that would permute [%arg0, %arg1, %arg2, %arg3] to [%arg1, %arg0, %arg3, %arg2]
    //      is (d0,d1,d2,d3) -> (d1,d0,d3,d2).
    //      If we consider the definition order as a mapping (d0,d1,d2,d3)->(d1,d0,d3,d2)
    //      and then invert it we get (d1,d0,d3,d2) -> (d0,d1,d2,d3)
    //      which simplifies to (d0,d1,d2,d3) -> (d1,d0,d3,d2)

    auto currentToDefinitionOrderMap = ComputeLoopIVToDefinitionOrderMap(ivs, context);
    auto definitionToCurrentOrderMap = mlir::inversePermutation(currentToDefinitionOrderMap);
    return definitionToCurrentOrderMap;
}

// Create an AffineLoadOp that understands how to access caches
mlir::AffineLoadOp CreateLoad(mlir::OpBuilder& builder,
                              mlir::Location loc,
                              mlir::Value src,
                              const std::vector<mlir::Value>& baseArrayPosition,
                              const std::vector<std::pair<Index, mlir::Value>>& unrealizedLoopNestIndices = {})
{
    if (auto srcCacheOp = mlir::dyn_cast_or_null<MakeCacheOp>(src.getDefiningOp()))
    {
        mlir::AffineValueMap loadAccessInfo = srcCacheOp.insertCachePosition(builder.getInsertionBlock(), baseArrayPosition, unrealizedLoopNestIndices);
        return builder.create<mlir::AffineLoadOp>(loc, src, loadAccessInfo.getAffineMap(), loadAccessInfo.getOperands());
    }
    else
    {
        return builder.create<mlir::AffineLoadOp>(loc, src, baseArrayPosition);
    }
}

// Create an MMALoadSyncOp that understands how to access caches
v::MMALoadSyncOp CreateMMALoad(mlir::OpBuilder& builder,
                               mlir::Location loc,
                               mlir::Type resultType,
                               mlir::Value src,
                               MMAShape mmaShapeType,
                               MMAOperandType operandType,
                               const std::vector<mlir::Value>& baseArrayPosition)
{
    if (auto srcCacheOp = mlir::dyn_cast_or_null<MakeCacheOp>(src.getDefiningOp()))
    {
        mlir::AffineValueMap loadAccessInfo = srcCacheOp.insertCachePosition(builder.getInsertionBlock(), baseArrayPosition, {});
        return builder.create<v::MMALoadSyncOp>(loc, resultType, src, mmaShapeType, operandType, loadAccessInfo.getAffineMap(), loadAccessInfo.getOperands());
    }
    else
    {
        return builder.create<v::MMALoadSyncOp>(loc, resultType, src, mmaShapeType, operandType, baseArrayPosition);
    }
}

v::MMAStoreSyncOp CreateMMAStore(mlir::OpBuilder& builder,
                                 mlir::Location loc,
                                 mlir::Value value,
                                 mlir::Value dst,
                                 MMAShape mmaShapeType,
                                 const std::vector<mlir::Value>& baseArrayPosition)
{
    if (auto dstCacheOp = mlir::dyn_cast_or_null<MakeCacheOp>(dst.getDefiningOp()))
    {
        mlir::AffineValueMap storeAccessInfo = dstCacheOp.insertCachePosition(builder.getInsertionBlock(), baseArrayPosition, {});
        return builder.create<v::MMAStoreSyncOp>(loc, value, dst, mmaShapeType, storeAccessInfo.getAffineMap(), storeAccessInfo.getOperands());
    }
    else
    {
        return builder.create<v::MMAStoreSyncOp>(loc, value, dst, mmaShapeType, baseArrayPosition);
    }
}

// Create an AffineStoreOp that understands how to access caches
template <typename StoreOp = mlir::AffineStoreOp>
StoreOp CreateStore(mlir::OpBuilder& builder,
                    mlir::Location loc,
                    mlir::Value value,
                    mlir::Value dst,
                    const std::vector<mlir::Value>& baseArrayPosition,
                    const std::vector<std::pair<Index, mlir::Value>>& unrealizedLoopNestIndices = {})
{
    if (auto dstCacheOp = mlir::dyn_cast_or_null<MakeCacheOp>(dst.getDefiningOp()))
    {
        mlir::AffineValueMap storeAccessInfo = dstCacheOp.insertCachePosition(builder.getInsertionBlock(), baseArrayPosition, unrealizedLoopNestIndices);
        return builder.create<StoreOp>(loc, value, dst, storeAccessInfo.getAffineMap(), storeAccessInfo.getOperands());
    }
    else
    {
        return builder.create<StoreOp>(loc, value, dst, baseArrayPosition);
    }
}

bool HasBaseArrayAccessAttrs(mlir::Operation* op)
{
    return op->hasAttr(BaseArrayAccessMapAttrName) && op->hasAttr(BaseArrayAccessIndicesAttrName);
}

void SetBaseArrayAccessAttrs(mlir::Operation* op, mlir::AffineMap accessMap, const std::vector<IndexAttr>& indices)
{
    auto indexArrayAttr = util::VectorToArrayAttr<IndexAttr>(indices, op->getContext());
    op->setAttr(BaseArrayAccessMapAttrName, mlir::AffineMapAttr::get(accessMap));
    op->setAttr(BaseArrayAccessIndicesAttrName, indexArrayAttr);
}

void CopyBaseArrayAccessAttrs(mlir::Operation* from, mlir::Operation* to)
{
    assert(HasBaseArrayAccessAttrs(from));
    to->setAttr(BaseArrayAccessMapAttrName, from->getAttr(BaseArrayAccessMapAttrName));
    to->setAttr(BaseArrayAccessIndicesAttrName, from->getAttr(BaseArrayAccessIndicesAttrName));
}

mlir::AffineValueMap GetBaseArrayAccessAffineValueMap(mlir::Operation* op)
{
    assert(HasBaseArrayAccessAttrs(op));
    auto affineMapAttr = op->getAttrOfType<mlir::AffineMapAttr>(BaseArrayAccessMapAttrName);
    auto indexArrayAttr = op->getAttrOfType<mlir::ArrayAttr>(BaseArrayAccessIndicesAttrName);
    auto affineMap = affineMapAttr.getValue();
    auto indices = util::ConvertArrayAttrToIndexVector(indexArrayAttr);
    auto indexValues = util::GetCurrentIndexIVs(indices, op);

    return mlir::AffineValueMap(affineMap, indexValues);
}

// Get the base array position for an AffineLoadOp that understands how to access caches
template <typename LoadStoreOp>
std::vector<mlir::Value> GetBaseArrayPosition(mlir::OpBuilder& builder, mlir::Location loc, LoadStoreOp loadStoreOp)
{
    if (HasBaseArrayAccessAttrs(loadStoreOp))
    {
        mlir::AffineValueMap accessAffineValueMap = GetBaseArrayAccessAffineValueMap(loadStoreOp);
        auto map = accessAffineValueMap.getAffineMap();
        auto operands = accessAffineValueMap.getOperands().vec();
        return util::MultiDimAffineApply(builder, loc, map, operands);
    }
    else
    {
        auto memref = loadStoreOp.memref();
        typename LoadStoreOp::Adaptor adaptor{ loadStoreOp };
        if (auto cache = mlir::dyn_cast_or_null<MakeCacheOp>(memref.getDefiningOp()))
        {
            // Note : this doesn't always work after canonicalization has run and omitted some operands
            return cache.getBaseArrayPosition(loadStoreOp);
        }
        else
        {
            auto accessMap = loadStoreOp.getAffineMapAttr().getValue();
            std::vector<mlir::Value> affineIndices(adaptor.indices().begin(), adaptor.indices().end());
            return util::MultiDimAffineApply(builder, loc, accessMap, affineIndices);
        }
    }
}

bool IsCacheRegionEmpty(BeginCacheRegion beginRegion)
{
    auto endOp = mlir::dyn_cast<EndCacheRegion>(beginRegion.getEndOp());

    auto parentBlock = beginRegion->getBlock();

    bool emptyRegion = true;
    parentBlock->walk(++mlir::Block::iterator(beginRegion), mlir::Block::iterator(endOp), [&](Operation* op) {
        emptyRegion = false;
        return WalkResult::interrupt();
    });

    return emptyRegion;
}

template <typename LoadStoreOp>
mlir::AffineMap GetLoadStoreAccessMap(LoadStoreOp op)
{
    return op.getAffineMapAttr().getValue();
}

template <typename LoadStoreOp>
std::vector<IndexAttr> GetLoadStoreAccessIndexAttrs(LoadStoreOp op)
{
    std::vector<mlir::Value> indexValues(op.indices().begin(), op.indices().end());
    std::vector<IndexAttr> indexAttrs;
    for (auto indexValue : indexValues)
    {
        mlir::AffineForOp forOp = mlir::getForInductionVarOwner(indexValue);
        if (!forOp) continue; // TODO: Is this correct???
        assert(forOp != nullptr);
        assert(forOp->hasAttrOfType<IndexAttr>("index"));
        auto indexAttr = forOp->getAttrOfType<IndexAttr>("index");
        indexAttrs.push_back(indexAttr);
    }
    return indexAttrs;
}

template <typename LoadStoreOp>
void TransferOrSetAccessAttrs(LoadStoreOp from, LoadStoreOp to)
{
    if (HasBaseArrayAccessAttrs(from))
    {
        CopyBaseArrayAccessAttrs(from, to);
    }
    else
    {
        auto accessMap = GetLoadStoreAccessMap(from);
        auto accessIndexAttrs = GetLoadStoreAccessIndexAttrs(from);
        SetBaseArrayAccessAttrs(to, accessMap, accessIndexAttrs);
    }
}

struct MultiCacheLoopInfo
{
    std::vector<mlir::AffineForOp> multiCacheLoops;
    std::vector<mlir::Value> multiCacheIVs;
    std::vector<mlir::Value> multiCacheIterCounters;
    std::vector<mlir::Value> activeBlockExternalSymbols;
    std::vector<int64_t> multiCacheShape;
    std::vector<int64_t> multiCacheStepSizes;
};

MultiCacheLoopInfo CreateMultiCacheLoops(mlir::OpBuilder& builder, MultiCacheCopyOp copyOp, const std::function<void(mlir::OpBuilder&, const MultiCacheLoopInfo&)>& fn)
{
    mlir::OpBuilder::InsertionGuard guard(builder);
    MultiCacheCopyOp::Adaptor adaptor{ copyOp };
    MultiCacheLoopInfo result;

    auto loc = copyOp.getLoc();

    std::optional<v::ExecutionTarget> execTargetOpt = util::ResolveExecutionTarget(copyOp);
    auto execTarget = *execTargetOpt;
    if (execTarget == v::ExecutionTarget::GPU)
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(copyOp);
        (void)util::CreateGPUControlBarrier(builder, "Block", loc);
        builder.setInsertionPointAfter(copyOp);
        (void)util::CreateGPUControlBarrier(builder, "Block", loc);
    }

    auto multiCacheLBMapsArrayAttr = adaptor.multiCacheLoopLowerBoundMaps();
    auto multiCacheUBMapsArrayAttr = adaptor.multiCacheLoopUpperBoundMaps();
    auto multiCacheStepsArrayAttr = adaptor.multiCacheLoopStepSizes();
    auto multiCacheLBMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(multiCacheLBMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });
    auto multiCacheUBMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(multiCacheUBMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });
    result.multiCacheStepSizes = util::ConvertArrayAttrToIntVector(multiCacheStepsArrayAttr);

    std::vector<Index> multiCacheIndexIds = util::ConvertArrayAttrToIndexVector(adaptor.multiCacheLoopIndexIds());

    assert(multiCacheLBMaps.size() == multiCacheUBMaps.size());
    assert(multiCacheLBMaps.size() == result.multiCacheStepSizes.size());
    assert(multiCacheLBMaps.size() == multiCacheIndexIds.size());
    auto multiCacheLoopCount = multiCacheLBMaps.size();

    // Construct the multiCache loops
    // Are we able to replace these with loopnests? we don't have a way to construct loopnests with affine map lower/upper bounds currently
    mlir::OpBuilder currentBuilder = builder;
    mlir::ValueRange emptyOperands;
    for (unsigned multiCacheDim = 0; multiCacheDim < multiCacheLoopCount; ++multiCacheDim)
    {
        auto forOp = mlir::createCanonicalizedAffineForOp(currentBuilder, loc, emptyOperands, multiCacheLBMaps[multiCacheDim], emptyOperands, multiCacheUBMaps[multiCacheDim], result.multiCacheStepSizes[multiCacheDim]);
        forOp->setAttr("index", IndexAttr::get(multiCacheIndexIds[multiCacheDim], currentBuilder.getContext()));
        currentBuilder = mlir::OpBuilder::atBlockTerminator(forOp.getBody());
        mlir::Value iterCounter = util::CreateConstantRangeForOpIterationCounter(currentBuilder, loc, forOp);
        result.multiCacheIterCounters.push_back(iterCounter);

        result.multiCacheIVs.push_back(forOp.getInductionVar());

        auto constantTripCountOpt = mlir::getConstantTripCount(forOp);
        assert(constantTripCountOpt.hasValue() && "AffineForOps in Accera loop nests must have constant trip counts");
        result.multiCacheShape.push_back(constantTripCountOpt.getValue());

        result.multiCacheLoops.push_back(forOp);
    }

    // Now that we have the multiCache IVs we can permute the multiCache external symbols and these IVs to make the full external symbols for the ActiveBlockCacheCopyOp
    auto externalSymbolsPermutationMap = copyOp.externalSymbolsPermutationMap();
    auto multiCacheExternalSymbolsValueRange = adaptor.multiCacheExternalSymbols();
    std::vector<mlir::Value> unpermutedExternalSymbols(multiCacheExternalSymbolsValueRange.begin(), multiCacheExternalSymbolsValueRange.end());
    unpermutedExternalSymbols.insert(unpermutedExternalSymbols.end(), result.multiCacheIVs.begin(), result.multiCacheIVs.end());

    // Permute the external symbols into their creation order
    // as the externalSymbolsPermutationMap will map from their creation
    // order to their expected order for the maps

    if (!unpermutedExternalSymbols.empty())
    {
        auto externalSymbolsToDefOrderMap = ComputeLoopIVToDefinitionOrderMap(unpermutedExternalSymbols, currentBuilder.getContext());
        std::vector<mlir::Value> activeBlockExternalSymbolDefinitionOrdered = util::MultiDimAffineApply(currentBuilder, loc, externalSymbolsToDefOrderMap, unpermutedExternalSymbols);
        result.activeBlockExternalSymbols = util::MultiDimAffineApply(currentBuilder, loc, externalSymbolsPermutationMap, activeBlockExternalSymbolDefinitionOrdered);
    }

    fn(currentBuilder, result);

    return result;
}

bool SameMemorySpace(mlir::Value left, mlir::Value right)
{
    auto leftType = left.getType();
    assert(leftType.isa<mlir::MemRefType>());
    auto leftMemRefType = leftType.cast<mlir::MemRefType>();
    auto rightType = right.getType();
    assert(rightType.isa<mlir::MemRefType>());
    auto rightMemRefType = rightType.cast<mlir::MemRefType>();

    return leftMemRefType.getMemorySpace() == rightMemRefType.getMemorySpace();
}

bool IsCacheOp(mlir::Operation* op)
{
    return mlir::isa<MakeCacheOp,
                     BeginCacheMappingOp,
                     EndCacheMappingOp,
                     BeginCacheRegionOp,
                     EndCacheRegionOp,
                     BeginMaxElementCacheRegionOp,
                     MultiCacheCopyOp,
                     ActiveBlockCacheCopyOp,
                     ActiveBlockCacheReduceOp,
                     CacheZeroOp,
                     ActiveElementCacheCopyOp,
                     ActiveElementCacheReduceOp>(op);
}

} // namespace

LogicalResult MakeCacheOpLowering::matchAndRewrite(MakeCacheOp makeCacheOp, PatternRewriter& rewriter) const
{
    auto loc = makeCacheOp.getLoc();

    auto cacheArray = makeCacheOp.cache();

    if (cacheArray.use_empty())
    {
        // No uses of the cache array anymore, so just erase this cache and move on
        rewriter.eraseOp(makeCacheOp);
        return success();
    }

    auto cacheBaseType = makeCacheOp.cache().getType();
    assert(cacheBaseType.isa<MemRefType>() && "Cache must be a memref");
    auto cacheType = cacheBaseType.cast<MemRefType>();
    auto elementBitWidth = cacheType.getElementTypeBitWidth();
    auto elementByteWidth = elementBitWidth / 8;

    if (!cacheType.hasStaticShape())
    {
        // MakeCacheOps that produce dynamic size memrefs are the base ops used for cloning
        rewriter.eraseOp(makeCacheOp);
        return success();
    }

    mlir::Value cacheGlobalBuffer;

    auto vLambdaOp = makeCacheOp->getParentOfType<v::ValueLambdaOp>();
    if (makeCacheOp.memorySpace() == v::MemorySpace::None || makeCacheOp.memorySpace() == v::MemorySpace::Global)
    {
        bool stackAllocateBuffer = false;
        if (vLambdaOp && HasVectorizationInfo(vLambdaOp))
        {
            auto vecInfo = GetVectorizationInfo(vLambdaOp);
            auto elementsPerVector = vecInfo.vectorBytes / elementByteWidth;
            stackAllocateBuffer = cacheType.getNumElements() <= (elementsPerVector * vecInfo.vectorUnitCount);
        }
        if (stackAllocateBuffer)
        {
            cacheGlobalBuffer = rewriter.create<mlir::memref::AllocaOp>(loc, cacheType, mlir::ValueRange{}, rewriter.getI64IntegerAttr(32));
        }
        else
        {
            cacheGlobalBuffer = util::CreateGlobalBuffer(rewriter, makeCacheOp, cacheType, "cache");
        }
    }
    else
    {
        // Shared or Private
        cacheGlobalBuffer = rewriter.create<v::AllocOp>(loc, cacheType, llvm::None);
    }

    rewriter.replaceOp(makeCacheOp, ValueRange{ cacheGlobalBuffer });

    return success();
}

LogicalResult CacheZeroOpRewrite::matchAndRewrite(CacheZeroOp cacheZeroOp, PatternRewriter& rewriter) const
{
    auto loc = cacheZeroOp.getLoc();

    auto cache = cacheZeroOp.cache();
    auto innerElementType = GetInnerElementType(cache);
    auto cacheType = cache.getType().cast<MemRefType>();
    auto cacheShape = cacheType.getShape();

    OpBuilder currentBuilder = rewriter;
    std::vector<mlir::Value> loopIndices;
    std::vector<AffineForOp> loops;
    for (size_t cacheDim = 0; cacheDim < cacheShape.size(); ++cacheDim)
    {
        auto newLoop = currentBuilder.create<AffineForOp>(loc, 0, cacheShape[cacheDim], 1);
        currentBuilder = util::MakeBodyBuilder(newLoop);
        loopIndices.push_back(newLoop.getInductionVar());
        loops.push_back(newLoop);
    }
    auto constantZero = currentBuilder.create<ConstantOp>(loc, innerElementType, currentBuilder.getZeroAttr(innerElementType));
    currentBuilder.create<mlir::memref::StoreOp>(loc, constantZero, cache, loopIndices);

    rewriter.eraseOp(cacheZeroOp);
    return success();
}

LogicalResult ActiveElementCacheCopyOpRewrite::matchAndRewrite(ActiveElementCacheCopyOp cacheCopyOp, PatternRewriter& rewriter) const
{
    // Copy data from the source buffer to the destination buffer by iterating over the cache region shape
    // and mapping from cache region indices to the source buffer and destination buffer

    auto loc = cacheCopyOp.getLoc();

    ActiveElementCacheCopyOp::Adaptor cacheCopyOpAdaptor{ cacheCopyOp };

    auto src = cacheCopyOp.src();
    assert(src.getType().isa<MemRefType>());
    auto memRefType = src.getType().cast<MemRefType>();
    [[maybe_unused]] auto baseSrcElementType = GetInnerElementType(src); // e.g. f32

    auto elementBitWidth = memRefType.getElementTypeBitWidth();
    auto elementByteWidth = elementBitWidth / 8;

    auto dst = cacheCopyOp.dst();
    assert(dst.getType().isa<MemRefType>());
    auto dstMemRefType = dst.getType().cast<MemRefType>();
    unsigned dstMemRefSpace = dstMemRefType.getMemorySpaceAsInt();
    auto baseDstElementType = GetInnerElementType(dst); // e.g. f32

    assert(baseSrcElementType == baseDstElementType && "Copy source and dest data types don't match");

    std::optional<v::ExecutionTarget> execTargetOpt = util::ResolveExecutionTarget(cacheCopyOp);
    assert(execTargetOpt.has_value());
    auto execTarget = *execTargetOpt;

    // TODO : make this a utility function on a cache op interface
    auto cacheRegionIndexRanges = util::ArrayAttrToVector<IndexRange, IndexRangeAttr>(
        cacheCopyOp.cacheRegionRelevantIndexRanges(),
        [](const IndexRangeAttr& indexRangeAttr) {
            return indexRangeAttr.getValue();
        });

    auto cacheRegionBaseIndices = util::ArrayAttrToVector<std::vector<Index>, mlir::ArrayAttr>(
        cacheCopyOp.cacheRegionBaseIndices(),
        util::ConvertArrayAttrToIndexVector);

    // If this op has no volume to operate over due to unswitched boundary conditions, just erase the op and return
    for (const auto& indexRange : cacheRegionIndexRanges)
    {
        if (indexRange.Size() == 0)
        {
            rewriter.eraseOp(cacheCopyOp);
            return success();
        }
    }

    if (execTarget == v::ExecutionTarget::GPU && dstMemRefSpace != static_cast<unsigned int>(v::MemorySpace::Private))
    {
        // TODO : should this be in a better place? This barrier is trying to prevent loops from getting
        //        too far ahead of their counterparts and trying to fill a cache before every thread is
        //        done with the previous set of data in the cache. This could possibly be an epilogue
        //        at some level of the loopnest outside of the innermost kernel but at the outermost
        //        loop level handled by every thread in cases where threads need to loop over a series
        //        of block (e.g. blocks in the K dimension in the GEMM case)
        (void)util::CreateGPUControlBarrier(rewriter, "Block", loc);
    }

    auto [copyNestOp, copyScheduleOp, copyExecPlanOp] = CreateActiveElementCacheLoopnest(rewriter, cacheCopyOp, elementByteWidth, "copy", [&](OpBuilder& cacheCopyBuilder, const std::vector<mlir::Value>& /*domainIndices*/, const std::vector<mlir::Value>& orderedSymbolicIndexOpValues) {
        std::vector<mlir::Value> combinedRelevantIndices;
        combinedRelevantIndices.insert(combinedRelevantIndices.end(), cacheCopyOpAdaptor.externalRelevantIndices().begin(), cacheCopyOpAdaptor.externalRelevantIndices().end());
        combinedRelevantIndices.insert(combinedRelevantIndices.end(), orderedSymbolicIndexOpValues.begin(), orderedSymbolicIndexOpValues.end());

        // Create a load op to use for bounds checking evaluation
        // We may need to delete this load and re-create it inside of a conditional if this load potentially has an out-of-bounds access
        auto loadOp = cacheCopyBuilder.create<AffineLoadOp>(loc, src, cacheCopyOp.relevantIndicesToSrcMap(), combinedRelevantIndices);
        cacheCopyBuilder.create<AffineStoreOp>(loc, loadOp.getResult(), dst, cacheCopyOp.relevantIndicesToDstMap(), combinedRelevantIndices);
    });

    // Bounds check cache copy loads/stores so we don't introduce
    // a bug by adding a cache copy
    auto copyOrder = copyScheduleOp.getOrder();
    for (const auto& loopIndex : copyOrder)
    {
        copyScheduleOp.addLoopAttribute(loopIndex, rewriter.getIdentifier(AccessBoundsCheckAttrName), rewriter.getUnitAttr());
    }

    if (execTarget == v::ExecutionTarget::GPU && dstMemRefSpace != static_cast<unsigned int>(v::MemorySpace::Private))
    {
        // Create thread mappings for the different levels of the copy loopnest
        // TODO : restructure the loopnest to ensure that there is always
        //        the appropriate number and size of splits to optimize this

        auto vLambdaOp = cacheCopyOp->getParentOfType<v::ValueLambdaOp>();
        // If we're inside a lambda then our ultimate exec target may be different
        // from the ValueFuncOp target. E.g. for GPU loopnests, the loopnest lambda
        // becomes a GPU function while the wrapping function stays as a CPU function
        // that launches the GPU func

        auto launchAttr = vLambdaOp->getAttrOfType<mlir::ArrayAttr>(vLambdaOp.getGPULaunchAttrName());
        assert(launchAttr != nullptr);
        auto gpuParams = accera::ir::targets::GPU::FromArrayAttr(launchAttr);
        std::vector<int64_t> blockDimSizes = { gpuParams.block.x, gpuParams.block.y, gpuParams.block.z };

        // Assign thread dimensions if it's not a private memory cache.
        auto threadXProcStr = v::stringifyEnum(v::Processor::ThreadX);
        auto threadYProcStr = v::stringifyEnum(v::Processor::ThreadY);
        auto threadZProcStr = v::stringifyEnum(v::Processor::ThreadZ);
        std::vector<llvm::StringRef> procStrs{ threadXProcStr,
                                               threadYProcStr,
                                               threadZProcStr };

        auto finalLoopNestOrder = copyScheduleOp.getOrder();
        std::vector<mlir::NamedAttribute> mappings;
        for (auto dimIdx = 0u; dimIdx < blockDimSizes.size(); ++dimIdx)
        {
            if (finalLoopNestOrder.size() <= dimIdx)
            {
                // Currently if we have fewer dimensions than block dim assignments, keep assignments as they
                // are even if threads wind up duplicating work
                // TODO : determine the best dimensions to favor splitting more heavily and split
                //        the same logical cache dimension multiple times and assign to different thread dims
                break;
            }

            mappings.push_back({ rewriter.getIdentifier(procStrs[dimIdx]),
                                 IndexAttr::get(finalLoopNestOrder[dimIdx], rewriter.getContext()) });
        }

        auto procMapAttrName = copyExecPlanOp.getGPUProcessorMapAttrName();
        auto procMap = rewriter.getDictionaryAttr({ mappings });
        copyExecPlanOp->setAttr(procMapAttrName, procMap);

        (void)util::CreateGPUControlBarrier(rewriter, "Block", loc);
    }

    rewriter.eraseOp(cacheCopyOp);

    return success();
}

std::optional<std::vector<int64_t>> GetConstantActiveBlockShape(const std::vector<mlir::AffineMap>& lbMaps,
                                                                const std::vector<mlir::AffineMap>& ubMaps)
{
    assert(lbMaps.size() == ubMaps.size() && "Must have same number of lower bound and upper bound maps");

    std::vector<int64_t> activeBlockShape;
    [[maybe_unused]] bool hasConstantShape = true;
    for (auto dim = 0u; dim < lbMaps.size(); ++dim)
    {
        assert(lbMaps[dim].getNumResults() == 1);
        assert(ubMaps[dim].getNumResults() == 1);
        assert(lbMaps[dim].getNumDims() == ubMaps[dim].getNumDims());
        assert(lbMaps[dim].getNumSymbols() == ubMaps[dim].getNumSymbols());
        auto diffExpr = ubMaps[dim].getResult(0) - lbMaps[dim].getResult(0);
        auto simplifiedExpr = mlir::simplifyAffineExpr(diffExpr, lbMaps[dim].getNumDims(), lbMaps[dim].getNumSymbols());
        if (!simplifiedExpr.isa<mlir::AffineConstantExpr>())
        {
            return std::nullopt;
        }
        else
        {
            auto constantExpr = simplifiedExpr.cast<mlir::AffineConstantExpr>();
            activeBlockShape.push_back(constantExpr.getValue());
        }
    }
    return activeBlockShape;
}

LogicalResult MultiCacheCopyOpRewrite::matchAndRewrite(MultiCacheCopyOp multiCacheCopyOp, PatternRewriter& rewriter) const
{
    auto loc = multiCacheCopyOp.getLoc();

    MultiCacheCopyOp::Adaptor adaptor{ multiCacheCopyOp };

    if (util::IsSubdomainEmpty(multiCacheCopyOp))
    {
        // We're in a zero-volume subdomain so any code here will get optimized out,
        // however, the memref region mappings don't gracefully handle this situation
        // currently so just remove this op and return rather than creating loopnests
        // that will get erased anyways
        rewriter.eraseOp(multiCacheCopyOp);
        return success();
    }

    MultiCacheLoopInfo multiCacheInfo = CreateMultiCacheLoops(rewriter, multiCacheCopyOp, [&](mlir::OpBuilder& currentBuilder, const MultiCacheLoopInfo& info) {
        currentBuilder.create<ActiveBlockCacheCopyOp>(loc,
                                                      multiCacheCopyOp.array(),
                                                      multiCacheCopyOp.cache(),
                                                      info.activeBlockExternalSymbols,
                                                      info.activeBlockExternalSymbols,
                                                      info.multiCacheIterCounters,
                                                      multiCacheCopyOp.activeBlockLowerBoundMaps(),
                                                      multiCacheCopyOp.activeBlockUpperBoundMaps(),
                                                      multiCacheCopyOp.activeBlockToCacheMap(),
                                                      multiCacheCopyOp.toCache(),
                                                      multiCacheCopyOp.activeBlockTag(),
                                                      multiCacheCopyOp.thrifty(),
                                                      true, // skipBarriers : this copy will already be guarded by barriers at the multicache level, so skip creating them internally
                                                      multiCacheCopyOp.vectorizationInfoAttr());
    });

    rewriter.eraseOp(multiCacheCopyOp);

    return success();
}

LogicalResult ActiveBlockCacheCopyOpRewrite::matchAndRewrite(ActiveBlockCacheCopyOp cacheCopyOp, PatternRewriter& rewriter) const
{
    // Copy data from the source buffer to the destination buffer by iterating over the cache region shape described
    // by the lower and upper bound maps and operands and mapping from cache region indices to the source buffer and destination buffer

    auto loc = cacheCopyOp.getLoc();

    ActiveBlockCacheCopyOp::Adaptor adaptor{ cacheCopyOp };

    if (util::IsSubdomainEmpty(cacheCopyOp))
    {
        // We're in a zero-volume subdomain so any code here will get optimized out,
        // however, the memref region mappings don't gracefully handle this situation
        // currently so just remove this op and return rather than creating loopnests
        // that will get erased anyways
        rewriter.eraseOp(cacheCopyOp);
        return success();
    }

    auto array = cacheCopyOp.array();
    assert(array.getType().isa<MemRefType>());
    auto memRefType = array.getType().cast<MemRefType>();
    unsigned outerArrayMemRefSpace = memRefType.getMemorySpaceAsInt();
    [[maybe_unused]] auto baseArrayElementType = GetInnerElementType(array); // e.g. f32
    unsigned outerArrayRank = memRefType.getRank();

    auto elementBitWidth = memRefType.getElementTypeBitWidth();
    auto elementByteWidth = elementBitWidth / 8;

    auto cache = cacheCopyOp.cache();
    assert(cache.getType().isa<MemRefType>());
    auto cacheMemRefType = cache.getType().cast<MemRefType>();
    unsigned cacheMemRefSpace = cacheMemRefType.getMemorySpaceAsInt();
    auto baseCacheElementType = GetInnerElementType(cache); // e.g. f32
    [[maybe_unused]] unsigned fullCacheRank = cacheMemRefType.getRank();

    assert(baseArrayElementType == baseCacheElementType && "Copy source and dest data types don't match");

    bool arrayToCache = cacheCopyOp.toCache();

    // Similar to generatePointWiseCopy() from llvm-project\mlir\lib\Transforms\Utils\LoopUtils.cpp however
    // we have a custom mapping from the active block to the cache position

    auto lbMapsArrayAttr = adaptor.lbMaps();
    auto ubMapsArrayAttr = adaptor.ubMaps();
    auto lbMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(lbMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });
    auto ubMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(ubMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });

    auto lbOperands = adaptor.lbOperands();
    auto ubOperands = adaptor.ubOperands();

    assert(llvm::all_of(lbMaps, [&](mlir::AffineMap lbMap) {
        return lbMap.getNumInputs() == lbOperands.size();
    }));
    assert(llvm::all_of(ubMaps, [&](mlir::AffineMap ubMap) {
        return ubMap.getNumInputs() == ubOperands.size();
    }));

    assert(lbMaps.size() == ubMaps.size() && "mismatched number of lb and ub maps");
    unsigned activeBlockRank = lbMaps.size();

    OpBuilder currentBuilder = rewriter;

    auto constantShapeOpt = GetConstantActiveBlockShape(lbMaps, ubMaps);

    std::optional<v::ExecutionTarget> execTargetOpt = util::ResolveExecutionTarget(cacheCopyOp);
    auto execTarget = *execTargetOpt;

    std::optional<VectorizationInfo> vecInfo;
    auto vecInfoLLVMOpt = cacheCopyOp.vectorizationInfo();
    if (vecInfoLLVMOpt.hasValue())
    {
        vecInfo = vecInfoLLVMOpt.getValue().getValue();
    }

    if (constantShapeOpt.has_value())
    {
        auto activeBlockShape = *constantShapeOpt;

        if (execTarget == v::ExecutionTarget::GPU)
        {
            if (!cacheCopyOp.skipBarriers())
            {
                (void)util::CreateGPUControlBarrier(rewriter, "Block", loc);
            }
            auto vLambdaOp = cacheCopyOp->getParentOfType<v::ValueLambdaOp>();
            // If we're inside a lambda then our ultimate exec target may be different
            // from the ValueFuncOp target. E.g. for GPU loopnests, the loopnest lambda
            // becomes a GPU function while the wrapping function stays as a CPU function
            // that launches the GPU func

            auto launchAttr = vLambdaOp->getAttrOfType<mlir::ArrayAttr>(vLambdaOp.getGPULaunchAttrName());
            assert(launchAttr != nullptr);
            auto gpuParams = accera::ir::targets::GPU::FromArrayAttr(launchAttr);
            std::vector<int64_t> blockDimSizes = { gpuParams.block.x, gpuParams.block.y, gpuParams.block.z };

            auto activeBlockVolume = std::accumulate(activeBlockShape.begin(), activeBlockShape.end(), 1, std::multiplies<int64_t>());

            // Use thread mappings any time one of the arrays we're indexing into is non-private
            bool useThreadMappings = outerArrayMemRefSpace != static_cast<unsigned int>(v::MemorySpace::Private) ||
                                     cacheMemRefSpace != static_cast<unsigned int>(v::MemorySpace::Private);

            if (!useThreadMappings)
            {
                // If we're copying from private memory to private memory, then don't consider the block sizes as we won't
                // have any threads to map relative to either of these buffers
                blockDimSizes = { 1, 1, 1 };
            }
            int64_t totalLoadsPerThread = activeBlockVolume / (blockDimSizes[0] * blockDimSizes[1] * blockDimSizes[2]);

            int64_t vectorSizePerThread = 1;
            if (vecInfo.has_value() && vecInfo->vectorBytes > 0)
            {
                vectorSizePerThread = std::min(vecInfo->vectorBytes / elementByteWidth, totalLoadsPerThread);
            }

            auto loadsPerThread = std::max((int64_t)1, (int64_t)(totalLoadsPerThread / vectorSizePerThread));

            std::vector<int64_t> activeBlockIterationShape{ loadsPerThread,
                                                            blockDimSizes[2],
                                                            blockDimSizes[1],
                                                            blockDimSizes[0],
                                                            vectorSizePerThread };
            std::vector<std::string> activeBlockDimNames{ ActionsPerThreadIndexName,
                                                          ThreadZIndexName,
                                                          ThreadYIndexName,
                                                          ThreadXIndexName,
                                                          ThreadVectorizationIndexName };
            auto [copyNestOp, copyScheduleOp, copyExecPlanOp] = CreateActiveBlockCacheLoopnest(rewriter, loc, memRefType, activeBlockIterationShape, activeBlockDimNames, vecInfo, elementByteWidth, execTarget, "copy", [&](OpBuilder& currentBuilder, const std::vector<mlir::Value>& domainIndices, const std::vector<mlir::Value>& /*orderedSymbolicIndexOpValues*/) {
                // The induction variables have been shifted to represent the constant iteration space
                // however, the maps expect they are constructed based on the original mappings so we
                // need to offset each IV by its lower bound map applied to its lower bound operands
                // e.g. affine.for %arg5 = affine_map<(d0) -> (d0)>(%arg4) to affine_map<(d0) -> (d0 + 256)>(%arg4) {
                //      became
                //      affine.for %arg5 = 0 to 256 {
                //      so now we need to do
                //      %lb_resolve = affine.apply affine_map<(d0) -> (d0)>(%arg4)
                //      %real_arg5 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%lb_resolve, %arg5)
                [[maybe_unused]] auto lptIndex = domainIndices[0];
                [[maybe_unused]] auto threadZ = domainIndices[1];
                [[maybe_unused]] auto threadY = domainIndices[2];
                [[maybe_unused]] auto threadX = domainIndices[3];
                [[maybe_unused]] auto vecIndex = domainIndices[4];

                // Map domainIndices -> flat buffer w/ affine expr + map
                mlir::AffineExpr cacheFillNestToFlatExpr = (currentBuilder.getAffineDimExpr(0) * (blockDimSizes[2] * blockDimSizes[1] * blockDimSizes[0] * vectorSizePerThread)) +
                                                           (currentBuilder.getAffineDimExpr(1) * (blockDimSizes[1] * blockDimSizes[0] * vectorSizePerThread)) +
                                                           (currentBuilder.getAffineDimExpr(2) * (blockDimSizes[0] * vectorSizePerThread)) +
                                                           (currentBuilder.getAffineDimExpr(3) * (vectorSizePerThread)) +
                                                           (currentBuilder.getAffineDimExpr(4));

                mlir::AffineMap cacheFillNestMap = mlir::AffineMap::get(5, 0, cacheFillNestToFlatExpr);

                llvm::SmallVector<int64_t, 4> multiCacheStrides;
                int64_t activeBlockOffset; // TODO : do we need to leverage this in any way? we're currently just arranging the threads according to fast/slow dimensions of the logical memref
                auto strideResult = mlir::getStridesAndOffset(memRefType, multiCacheStrides, activeBlockOffset);
                assert(succeeded(strideResult));
                auto numMultiCacheDims = multiCacheStrides.size() - activeBlockRank;
                std::vector<int64_t> activeBlockStrides(multiCacheStrides.begin() + numMultiCacheDims, multiCacheStrides.end());

                // We want to traverse the dimensions of the active block in increasing stride order, so keep track of the logical dimensions and sort them
                std::vector<std::pair<size_t, int64_t>> activeBlockLogicalDimAndStride;
                size_t dimIdxCounter = 0;
                std::transform(activeBlockStrides.begin(), activeBlockStrides.end(), std::back_inserter(activeBlockLogicalDimAndStride), [&](int64_t stride) {
                    return std::make_pair(dimIdxCounter++, stride);
                });

                std::sort(activeBlockLogicalDimAndStride.begin(), activeBlockLogicalDimAndStride.end(), [](const std::pair<size_t, int64_t>& left, const std::pair<size_t, int64_t>& right) {
                    return left.second < right.second;
                });

                auto cumulativeStride = 1;
                std::vector<mlir::AffineExpr> flatToActiveBlockExprs(activeBlockRank);
                for (const auto& [activeBlockLogicalDimIdx, stride] : activeBlockLogicalDimAndStride)
                {
                    auto curDimSize = activeBlockShape[activeBlockLogicalDimIdx];
                    flatToActiveBlockExprs[activeBlockLogicalDimIdx] = ((currentBuilder.getAffineDimExpr(0).floorDiv(cumulativeStride)) % curDimSize);
                    cumulativeStride *= curDimSize;
                }

                mlir::AffineMap flatBufferToActiveBlockMap = mlir::AffineMap::get(1, 0, flatToActiveBlockExprs, currentBuilder.getContext());
                auto gpuFillNestToActiveBlockMap = flatBufferToActiveBlockMap.compose(cacheFillNestMap);

                std::vector<mlir::Value> loopNestIVs(domainIndices.begin(), domainIndices.end());
                std::vector<mlir::Value> gpuFillMapApplied = util::MultiDimAffineApply(currentBuilder, loc, gpuFillNestToActiveBlockMap, loopNestIVs);

                // Map from above map (flat buffer) to active block position
                // TODO: Above combined map, needs to map from fastest moving global index, to fastest moving flat index

                std::vector<mlir::Value> lowerBoundOffsetIVs;
                lowerBoundOffsetIVs.reserve(gpuFillMapApplied.size());
                assert(lbMaps.size() == gpuFillMapApplied.size());
                mlir::AffineExpr sumExpr = currentBuilder.getAffineDimExpr(0) + currentBuilder.getAffineDimExpr(1);
                mlir::AffineMap sumMap = mlir::AffineMap::get(2, 0, sumExpr);
                for (unsigned arrayDim = 0; arrayDim < gpuFillMapApplied.size(); ++arrayDim)
                {
                    mlir::Value lbMapApplied = currentBuilder.create<mlir::AffineApplyOp>(loc, lbMaps[arrayDim], lbOperands);
                    mlir::Value lbOffsetIV = currentBuilder.create<mlir::AffineApplyOp>(loc, sumMap, mlir::ValueRange{ lbMapApplied, gpuFillMapApplied[arrayDim] });
                    lowerBoundOffsetIVs.push_back(lbOffsetIV);
                }

                // Get the pairs of loopnest Index objects and their corresponding mlir::Values to use to access the
                // caches if needed
                std::vector<std::pair<Index, mlir::Value>> unrealizedLoopNestIndices;
                for (auto& loopnestIV : domainIndices)
                {
                    auto indexOp = mlir::dyn_cast<SymbolicIndexOp>(loopnestIV.getDefiningOp());
                    auto index = indexOp.index().getValue();
                    unrealizedLoopNestIndices.emplace_back(index, loopnestIV);
                }
                if (arrayToCache)
                {
                    mlir::Value loadedValue = CreateLoad(currentBuilder, loc, array, lowerBoundOffsetIVs, unrealizedLoopNestIndices);
                    CreateStore(currentBuilder, loc, loadedValue, cache, lowerBoundOffsetIVs, unrealizedLoopNestIndices);
                }
                else
                {
                    mlir::Value loadedValue = CreateLoad(currentBuilder, loc, cache, lowerBoundOffsetIVs, unrealizedLoopNestIndices);
                    CreateStore(currentBuilder, loc, loadedValue, array, lowerBoundOffsetIVs, unrealizedLoopNestIndices);
                }
            });

            if (useThreadMappings)
            {
                auto threadZProcStr = v::stringifyEnum(v::Processor::ThreadZ);
                auto threadYProcStr = v::stringifyEnum(v::Processor::ThreadY);
                auto threadXProcStr = v::stringifyEnum(v::Processor::ThreadX);
                std::vector<llvm::StringRef> procStrs{ threadZProcStr,
                                                       threadYProcStr,
                                                       threadXProcStr };

                std::vector<mlir::NamedAttribute> mappings;
                auto copySymbolicIndices = copyNestOp.getIndices(rewriter);
                for (auto i = 0u; i < procStrs.size(); ++i)
                {
                    mappings.push_back({ rewriter.getIdentifier(procStrs[i]),
                                         IndexAttr::get(copySymbolicIndices[i + 1].getValue(), rewriter.getContext()) });
                }

                auto procMapAttrName = copyExecPlanOp.getGPUProcessorMapAttrName();
                auto procMap = rewriter.getDictionaryAttr({ mappings });
                copyExecPlanOp->setAttr(procMapAttrName, procMap);

                if (!cacheCopyOp.skipBarriers())
                {
                    (void)util::CreateGPUControlBarrier(rewriter, "Block", loc);
                }
            }
        }
        else
        {
            auto [copyNestOp, copyScheduleOp, copyExecPlanOp] = CreateActiveBlockCacheLoopnest(rewriter, loc, memRefType, activeBlockShape, {}, vecInfo, elementByteWidth, execTarget, "copy", [&](OpBuilder& currentBuilder, const std::vector<mlir::Value>& domainIndices, const std::vector<mlir::Value>& /*orderedSymbolicIndexOpValues*/) {
                // The induction variables have been shifted to represent the constant iteration space
                // however, the maps expect they are constructed based on the original mappings so we
                // need to offset each IV by its lower bound map applied to its lower bound operands
                // e.g. affine.for %arg5 = affine_map<(d0) -> (d0)>(%arg4) to affine_map<(d0) -> (d0 + 256)>(%arg4) {
                //      became
                //      affine.for %arg5 = 0 to 256 {
                //      so now we need to do
                //      %lb_resolve = affine.apply affine_map<(d0) -> (d0)>(%arg4)
                //      %real_arg5 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%lb_resolve, %arg5)

                std::vector<mlir::Value> lowerBoundOffsetIVs;
                lowerBoundOffsetIVs.reserve(domainIndices.size());
                assert(lbMaps.size() == domainIndices.size());
                mlir::AffineExpr sumExpr = currentBuilder.getAffineDimExpr(0) + currentBuilder.getAffineDimExpr(1);
                mlir::AffineMap sumMap = mlir::AffineMap::get(2, 0, sumExpr);
                for (unsigned arrayDim = 0; arrayDim < domainIndices.size(); ++arrayDim)
                {
                    mlir::Value lbMapApplied = currentBuilder.create<mlir::AffineApplyOp>(loc, lbMaps[arrayDim], lbOperands);
                    mlir::Value lbOffsetIV = currentBuilder.create<mlir::AffineApplyOp>(loc, sumMap, mlir::ValueRange{ lbMapApplied, domainIndices[arrayDim] });
                    lowerBoundOffsetIVs.push_back(lbOffsetIV);
                }

                if (arrayToCache)
                {
                    mlir::Value loadedValue = CreateLoad(currentBuilder, loc, array, lowerBoundOffsetIVs);
                    CreateStore(currentBuilder, loc, loadedValue, cache, lowerBoundOffsetIVs);
                }
                else
                {
                    mlir::Value loadedValue = CreateLoad(currentBuilder, loc, cache, lowerBoundOffsetIVs);
                    CreateStore(currentBuilder, loc, loadedValue, array, lowerBoundOffsetIVs);
                }
            });
            // Bounds check cache copy loads/stores so we don't introduce
            // a bug by adding a cache copy
            auto copyOrder = copyScheduleOp.getOrder();
            for (const auto& loopIndex : copyOrder)
            {
                copyScheduleOp.addLoopAttribute(loopIndex, rewriter.getIdentifier(AccessBoundsCheckAttrName), rewriter.getUnitAttr());
            }
        }
    }
    else
    {
        std::vector<mlir::Value> copyIVs;

        // Are we able to replace these with loopnests? we don't have a way to construct loopnests with affine map lower/upper bounds currently
        for (unsigned arrayDim = 0; arrayDim < outerArrayRank; ++arrayDim)
        {
            auto forOp = mlir::createCanonicalizedAffineForOp(currentBuilder, loc, lbOperands, lbMaps[arrayDim], ubOperands, ubMaps[arrayDim]);
            currentBuilder = mlir::OpBuilder::atBlockTerminator(forOp.getBody());

            // Subscript for the slow memref being copied.
            copyIVs.push_back(forOp.getInductionVar());
        }

        if (arrayToCache)
        {
            mlir::Value loadedValue = CreateLoad(currentBuilder, loc, array, copyIVs);
            CreateStore(currentBuilder, loc, loadedValue, cache, copyIVs);
        }
        else
        {
            mlir::Value loadedValue = CreateLoad(currentBuilder, loc, cache, copyIVs);
            CreateStore(currentBuilder, loc, loadedValue, array, copyIVs);
        }
    }

    rewriter.eraseOp(cacheCopyOp);

    return success();
}

LogicalResult ActiveBlockCacheReduceOpRewrite::matchAndRewrite(ActiveBlockCacheReduceOp cacheReduceOp, PatternRewriter& rewriter) const
{
    // TODO : handle gpu
    // Copy data from the source buffer to the destination buffer by iterating over the cache region shape described
    // by the lower and upper bound maps and operands and mapping from cache region indices to the source buffer and destination buffer

    auto loc = cacheReduceOp.getLoc();

    ActiveBlockCacheReduceOp::Adaptor adaptor{ cacheReduceOp };

    if (util::IsSubdomainEmpty(cacheReduceOp))
    {
        // We're in a zero-volume subdomain so any code here will get optimized out,
        // however, the memref region mappings don't gracefully handle this situation
        // currently so just remove this op and return rather than creating loopnests
        // that will get erased anyways
        rewriter.eraseOp(cacheReduceOp);
        return success();
    }

    auto array = cacheReduceOp.array();
    assert(array.getType().isa<MemRefType>());
    auto memRefType = array.getType().cast<MemRefType>();
    [[maybe_unused]] auto baseArrayElementType = GetInnerElementType(array); // e.g. f32

    auto elementBitWidth = memRefType.getElementTypeBitWidth();
    auto elementByteWidth = elementBitWidth / 8;

    auto cache = cacheReduceOp.cache();
    assert(cache.getType().isa<MemRefType>());
    auto cacheMemRefType = cache.getType().cast<MemRefType>();
    [[maybe_unused]] unsigned cacheMemRefSpace = cacheMemRefType.getMemorySpaceAsInt();
    auto baseCacheElementType = GetInnerElementType(cache); // e.g. f32

    assert(baseArrayElementType == baseCacheElementType && "Copy source and dest data types don't match");

    // Similar to generatePointWiseCopy() from llvm-project\mlir\lib\Transforms\Utils\LoopUtils.cpp however
    // we have a custom mapping from the active block to the cache position

    auto lbMapsArrayAttr = adaptor.lbMaps();
    auto ubMapsArrayAttr = adaptor.ubMaps();
    auto lbMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(lbMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });
    auto ubMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(ubMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });

    auto lbOperands = adaptor.lbOperands();
    auto ubOperands = adaptor.ubOperands();

    assert(llvm::all_of(lbMaps, [&](mlir::AffineMap lbMap) {
        return lbMap.getNumInputs() == lbOperands.size();
    }));
    assert(llvm::all_of(ubMaps, [&](mlir::AffineMap ubMap) {
        return ubMap.getNumInputs() == ubOperands.size();
    }));

    unsigned rank = memRefType.getRank();
    assert(lbMaps.size() == ubMaps.size() && "mismatched number of lb and ub maps");

    OpBuilder currentBuilder = rewriter;

    auto scaleValue = CreateProductOfValues(rewriter, loc, baseArrayElementType, adaptor.scaleValues());

    auto constantShapeOpt = GetConstantActiveBlockShape(lbMaps, ubMaps);

    std::optional<VectorizationInfo> vecInfo;
    auto vecInfoLLVMOpt = cacheReduceOp.vectorizationInfo();
    if (vecInfoLLVMOpt.hasValue())
    {
        vecInfo = vecInfoLLVMOpt.getValue().getValue();
    }

    if (constantShapeOpt.has_value())
    {
        auto activeBlockShape = *constantShapeOpt;
        std::optional<v::ExecutionTarget> execTargetOpt = util::ResolveExecutionTarget(cacheReduceOp);
        assert(execTargetOpt.has_value());
        auto execTarget = *execTargetOpt;

        auto [reduceNestOp, reduceScheduleOp, reduceExecPlanOp] = CreateActiveBlockCacheLoopnest(rewriter, loc, memRefType, activeBlockShape, {}, vecInfo, elementByteWidth, execTarget, "reduce", [&](OpBuilder& currentBuilder, const std::vector<mlir::Value>& domainIndices, const std::vector<mlir::Value>& orderedSymbolicIndexOpValues) {
            // The induction variables have been shifted to represent the constant iteration space
            // however, the maps expect they are constructed based on the original mappings so we
            // need to offset each IV by its lower bound map applied to its lower bound operands
            // e.g. affine.for %arg5 = affine_map<(d0) -> (d0)>(%arg4) to affine_map<(d0) -> (d0 + 256)>(%arg4) {
            //      became
            //      affine.for %arg5 = 0 to 256 {
            //      so now we need to do
            //      %lb_resolve = affine.apply affine_map<(d0) -> (d0)>(%arg4)
            //      %real_arg5 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%lb_resolve, %arg5)

            std::vector<mlir::Value> lowerBoundOffsetIVs;
            lowerBoundOffsetIVs.reserve(domainIndices.size());
            assert(lbMaps.size() == domainIndices.size());
            mlir::AffineExpr sumExpr = currentBuilder.getAffineDimExpr(0) + currentBuilder.getAffineDimExpr(1);
            mlir::AffineMap sumMap = mlir::AffineMap::get(2, 0, sumExpr);
            for (unsigned arrayDim = 0; arrayDim < domainIndices.size(); ++arrayDim)
            {
                mlir::Value lbMapApplied = currentBuilder.create<mlir::AffineApplyOp>(loc, lbMaps[arrayDim], lbOperands);
                mlir::Value lbOffsetIV = currentBuilder.create<mlir::AffineApplyOp>(loc, sumMap, mlir::ValueRange{ lbMapApplied, domainIndices[arrayDim] });
                lowerBoundOffsetIVs.push_back(lbOffsetIV);
            }

            mlir::Value loadedCacheValue = CreateLoad(currentBuilder, loc, cache, lowerBoundOffsetIVs);
            auto scaledCacheValue = currentBuilder.create<v::BinOp>(loc, BinaryOpPredicate::MUL, scaleValue, loadedCacheValue);
            mlir::Value currentArrayValue = CreateLoad(currentBuilder, loc, array, lowerBoundOffsetIVs);
            auto accumulatedValue = currentBuilder.create<v::BinOp>(loc, BinaryOpPredicate::ADD, currentArrayValue, scaledCacheValue);
            CreateStore(currentBuilder, loc, accumulatedValue, array, lowerBoundOffsetIVs);
        });

        // Bounds check cache copy loads/stores so we don't introduce
        // a bug by adding a cache copy
        auto copyOrder = reduceScheduleOp.getOrder();
        for (const auto& loopIndex : copyOrder)
        {
            reduceScheduleOp.addLoopAttribute(loopIndex, rewriter.getIdentifier(AccessBoundsCheckAttrName), rewriter.getUnitAttr());
        }
    }
    else
    {
        std::vector<mlir::Value> IVs;
        for (unsigned arrayDim = 0; arrayDim < rank; ++arrayDim)
        {
            auto forOp = mlir::createCanonicalizedAffineForOp(currentBuilder, loc, lbOperands, lbMaps[arrayDim], ubOperands, ubMaps[arrayDim]);
            currentBuilder = mlir::OpBuilder::atBlockTerminator(forOp.getBody());

            // Subscript for the slow memref being copied.
            IVs.push_back(forOp.getInductionVar());
        }

        mlir::Value loadedCacheValue = CreateLoad(currentBuilder, loc, cache, IVs);
        auto scaledCacheValue = currentBuilder.create<v::BinOp>(loc, BinaryOpPredicate::MUL, scaleValue, loadedCacheValue);
        mlir::Value currentArrayValue = CreateLoad(currentBuilder, loc, array, IVs);
        auto accumulatedValue = currentBuilder.create<v::BinOp>(loc, BinaryOpPredicate::ADD, currentArrayValue, scaledCacheValue);
        CreateStore(currentBuilder, loc, accumulatedValue, array, IVs);
    }
    rewriter.eraseOp(cacheReduceOp);

    return success();
}

LogicalResult ActiveElementCacheReduceOpRewrite::matchAndRewrite(ActiveElementCacheReduceOp cacheReduceOp, PatternRewriter& rewriter) const
{
    // Reduce data from the source cache buffer to the destination buffer by iterating over the cache region shape
    // and mapping from cache region indices to the source cache buffer and destination buffer

    auto loc = cacheReduceOp.getLoc();

    ActiveElementCacheReduceOp::Adaptor adaptor{ cacheReduceOp };

    [[maybe_unused]] auto dst = cacheReduceOp.dst();
    assert(dst.getType().isa<MemRefType>());
    auto baseOutputMemRefType = dst.getType().cast<MemRefType>();
    [[maybe_unused]] auto baseOutputShape = baseOutputMemRefType.getShape();
    auto baseOutputElementType = GetInnerElementType(dst);

    auto elementBitWidth = baseOutputMemRefType.getElementTypeBitWidth();
    auto elementByteWidth = elementBitWidth / 8;

    auto cache = cacheReduceOp.srcCache();
    auto cacheMemRefType = cache.getType().cast<MemRefType>();
    [[maybe_unused]] auto cacheElementType = cacheMemRefType.getElementType(); // either something like vector< n x f32 > or f32
    [[maybe_unused]] auto cacheShape = cacheMemRefType.getShape();
    [[maybe_unused]] auto baseCacheElementType = GetInnerElementType(cache); // e.g. f32

    auto cacheRegionIndexRanges = util::ArrayAttrToVector<IndexRange, IndexRangeAttr>(
        cacheReduceOp.cacheRegionRelevantIndexRanges(),
        [](const IndexRangeAttr& indexRangeAttr) {
            return indexRangeAttr.getValue();
        });

    auto cacheRegionBaseIndices = util::ArrayAttrToVector<std::vector<Index>, mlir::ArrayAttr>(
        cacheReduceOp.cacheRegionBaseIndices(),
        util::ConvertArrayAttrToIndexVector);
    assert(cacheRegionIndexRanges.size() == cacheRegionBaseIndices.size());

    // If this op has no volume to operate over due to unswitched boundary conditions, just erase the op and return
    for (const auto& indexRange : cacheRegionIndexRanges)
    {
        if (indexRange.Size() == 0)
        {
            rewriter.eraseOp(cacheReduceOp);
            return success();
        }
    }

    auto scaleValue = CreateProductOfValues(rewriter, loc, baseOutputElementType, adaptor.scaleValues());

    auto [reduceNestOp, reduceScheduleOp, reduceExecPlanOp] = CreateActiveElementCacheLoopnest(rewriter, cacheReduceOp, elementByteWidth, "reduce", [&](OpBuilder& cacheReduceBuilder, const std::vector<mlir::Value>& /*domainIndices*/, const std::vector<mlir::Value>& orderedSymbolicIndexOpValues) {
        std::vector<mlir::Value> combinedRelevantIndices;
        combinedRelevantIndices.insert(
            combinedRelevantIndices.end(),
            adaptor.externalRelevantIndices().begin(),
            adaptor.externalRelevantIndices().end());
        combinedRelevantIndices.insert(combinedRelevantIndices.end(), orderedSymbolicIndexOpValues.begin(), orderedSymbolicIndexOpValues.end());

        auto loadedCacheValue = cacheReduceBuilder.create<AffineLoadOp>(loc, cache, cacheReduceOp.relevantIndicesToSrcCacheMap(), combinedRelevantIndices);
        auto scaledCacheValue = cacheReduceBuilder.create<v::BinOp>(loc, BinaryOpPredicate::MUL, scaleValue, loadedCacheValue);
        auto currentOutputValue = cacheReduceBuilder.create<AffineLoadOp>(loc, dst, cacheReduceOp.relevantIndicesToDstMap(), combinedRelevantIndices);
        auto accumulatedValue = cacheReduceBuilder.create<v::BinOp>(loc, BinaryOpPredicate::ADD, scaledCacheValue, currentOutputValue);
        cacheReduceBuilder.create<AffineStoreOp>(loc, accumulatedValue, dst, cacheReduceOp.relevantIndicesToDstMap(), combinedRelevantIndices);
    });

    // Bounds check cache reduce loads/stores so we don't introduce
    // a bug by adding a cache reduce
    auto reduceOrder = reduceScheduleOp.getOrder();
    for (const auto& loopIndex : reduceOrder)
    {
        reduceScheduleOp.addLoopAttribute(loopIndex, rewriter.getIdentifier(AccessBoundsCheckAttrName), rewriter.getUnitAttr());
    }

    rewriter.eraseOp(cacheReduceOp);

    return success();
}

template <typename CacheOp>
mlir::Operation* GetOpIfActiveBlockTagMatches(CacheOp cacheOp, llvm::StringRef cacheOpTag)
{
    if (cacheOp.activeBlockTag() == cacheOpTag)
    {
        return cacheOp;
    }
    else
    {
        return nullptr;
    }
}

template <typename CacheOp>
mlir::Operation* GetCacheOpPair(CacheOp cacheOp)
{
    // Search for ops in the same block that have the same active block tag

    auto cacheOpTag = cacheOp.activeBlockTag();
    auto parentBlock = cacheOp->getBlock();
    for (auto& op : parentBlock->getOperations())
    {
        if (&op != cacheOp.getOperation())
        {
            mlir::Operation* pairOp = nullptr;
            TypeSwitch<Operation*>(&op)
                .Case([&](MultiCacheCopyOp copyOp) {
                    pairOp = GetOpIfActiveBlockTagMatches(copyOp, cacheOpTag);
                })
                .Case([&](ActiveBlockCacheCopyOp copyOp) {
                    pairOp = GetOpIfActiveBlockTagMatches(copyOp, cacheOpTag);
                })
                .Case([&](ActiveBlockCacheReduceOp reduceOp) {
                    pairOp = GetOpIfActiveBlockTagMatches(reduceOp, cacheOpTag);
                })
                .Case([&](CacheZeroOp zeroOp) {
                    pairOp = GetOpIfActiveBlockTagMatches(zeroOp, cacheOpTag);
                })
                .Default([&](mlir::Operation* op) {
                    // Not a cache op, so nothing to do here
                });
            if (pairOp != nullptr)
            {
                return pairOp;
            }
        }
    }
    return nullptr;
}

template <typename KernelFn>
bool InMemoryLoopnestRecursiveRunner(const std::vector<int64_t>& loopnestShape, const std::vector<int64_t>& stepSizes, size_t currentDim, std::vector<int64_t>& currentIVs, KernelFn&& fn)
{
    // Returns true if the loopnest should continue running, false if it should exit early
    if (currentDim < loopnestShape.size())
    {
        for (int64_t idx = 0; idx < loopnestShape[currentDim]; idx += stepSizes[currentDim])
        {
            currentIVs[currentDim] = idx;
            if (!InMemoryLoopnestRecursiveRunner(loopnestShape, stepSizes, currentDim + 1, currentIVs, fn))
            {
                return false;
            }
        }
        return true;
    }
    else
    {
        // We're inside the innermost loop at this point, so just invoke the given kernel function with the current loop IV values
        // The kernel function should return true if the loopnest should continue running, false if it should exit early
        return fn(currentIVs);
    }
}

template <typename KernelFn>
bool InMemoryLoopnestRunner(const std::vector<int64_t>& loopnestShape, const std::vector<int64_t>& stepSizes, KernelFn&& fn)
{
    // Returns true if the loopnest ran completely, false if it exited early
    std::vector<int64_t> currentIVs(loopnestShape.size(), 0);
    return InMemoryLoopnestRecursiveRunner(loopnestShape, stepSizes, 0, currentIVs, fn);
}

std::vector<int64_t> GetConstantActiveBlockShapeHelper(mlir::ArrayAttr lbMapsArrayAttr, mlir::ArrayAttr ubMapsArrayAttr, mlir::ValueRange lbOperands, mlir::ValueRange ubOperands)
{
    auto lbMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(lbMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });
    auto ubMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(ubMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });

    assert(llvm::all_of(lbMaps, [&](mlir::AffineMap lbMap) {
        return lbMap.getNumInputs() == lbOperands.size();
    }));
    assert(llvm::all_of(ubMaps, [&](mlir::AffineMap ubMap) {
        return ubMap.getNumInputs() == ubOperands.size();
    }));

    assert(lbMaps.size() == ubMaps.size() && "mismatched number of lb and ub maps");

    auto constantShapeOpt = GetConstantActiveBlockShape(lbMaps, ubMaps);
    assert(constantShapeOpt.has_value() && "Only constant active block shapes are supported");
    return *constantShapeOpt;
}

template <typename CacheOp>
std::vector<int64_t> GetConstantActiveBlockShapeHelper(CacheOp cacheOp)
{
    typename CacheOp::Adaptor adaptor{ cacheOp };
    auto lbMapsArrayAttr = adaptor.lbMaps();
    auto ubMapsArrayAttr = adaptor.ubMaps();
    auto lbOperands = adaptor.lbOperands();
    auto ubOperands = adaptor.ubOperands();
    return GetConstantActiveBlockShapeHelper(lbMapsArrayAttr, ubMapsArrayAttr, lbOperands, ubOperands);
}

std::vector<int64_t> GetFullCacheShapeHelper(const std::vector<int64_t>& multiCacheShape,
                                             const std::vector<mlir::Value>& lbOperands,
                                             const std::vector<mlir::Value>& ubOperands,
                                             mlir::ArrayAttr lbMapsArrayAttr,
                                             mlir::ArrayAttr ubMapsArrayAttr)
{
    auto activeBlockShape = GetConstantActiveBlockShapeHelper(lbMapsArrayAttr, ubMapsArrayAttr, lbOperands, ubOperands);
    std::vector<int64_t> combinedCacheShape = multiCacheShape;
    combinedCacheShape.insert(combinedCacheShape.end(), activeBlockShape.begin(), activeBlockShape.end());
    return combinedCacheShape;
}

bool ThriftyCacheAllSingleElementStridesHelper(mlir::PatternRewriter& rewriter,
                                               mlir::OpBuilder& currentBuilder, // Builder positioned inside of the temp multicache loops (if there are any)
                                               mlir::Location loc,
                                               mlir::Value outerArray,
                                               mlir::Value cacheArray,
                                               const std::vector<mlir::Value>& multiCacheIVs,
                                               const std::vector<int64_t>& fullCacheShape,
                                               const std::vector<int64_t>& fullCacheStepSizes,
                                               const std::vector<mlir::Value>& activeBlockExternalSymbols,
                                               mlir::ArrayAttr lbMapsArrayAttr,
                                               mlir::ArrayAttr ubMapsArrayAttr)
{
    mlir::ValueRange lbOperands = activeBlockExternalSymbols;
    [[maybe_unused]] mlir::ValueRange ubOperands = activeBlockExternalSymbols;

    auto lbMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(lbMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });
    auto ubMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(ubMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });

    // Walk the combinedCacheStepSizes dimensions from innermost to outermost and loop over the loads and stores in that order

    // Create temporary op stacks to hold the ops created as part of computing the difference in accesses between iterations.
    // We create two so that one holds the current iteration accesses and one holds the previuos iteration accesses.
    // Then we keep track of them with two pointers so the "current" can become the "previous" with a pointer swap and then
    // one can be cleared out before examining the next iteration
    // Note: here we prefer stacks over other data structures as the accesses may depend on affine apply op computations, so in
    //       general we want to erase the ops in the reverse order they were constructed.
    std::stack<mlir::Operation*> temporaryOpsOne;
    std::stack<mlir::Operation*> temporaryOpsTwo;
    std::stack<mlir::Operation*>* prevTemporaryOps = &temporaryOpsOne;
    std::stack<mlir::Operation*>* currentTemporaryOps = &temporaryOpsTwo;
    auto computeGlobalIndices = [&](const std::vector<int64_t>& activeBlockCurrentIVs, std::stack<mlir::Operation*>* temporaryOps) {
        std::vector<mlir::Value> lowerBoundOffsetIVs;
        lowerBoundOffsetIVs.reserve(activeBlockCurrentIVs.size());
        assert(lbMaps.size() == activeBlockCurrentIVs.size());
        mlir::AffineExpr sumExpr = currentBuilder.getAffineDimExpr(0) + currentBuilder.getAffineDimExpr(1);
        mlir::AffineMap sumMap = mlir::AffineMap::get(2, 0, sumExpr);
        for (unsigned arrayDim = 0; arrayDim < activeBlockCurrentIVs.size(); ++arrayDim)
        {
            mlir::Value lbMapApplied = currentBuilder.create<mlir::AffineApplyOp>(loc, lbMaps[arrayDim], lbOperands);
            mlir::Value constantIV = currentBuilder.create<mlir::ConstantIndexOp>(loc, activeBlockCurrentIVs[arrayDim]);
            mlir::Value lbOffsetIV = currentBuilder.create<mlir::AffineApplyOp>(loc, sumMap, mlir::ValueRange{ lbMapApplied, constantIV });
            lowerBoundOffsetIVs.push_back(lbOffsetIV);

            temporaryOps->push(lbMapApplied.getDefiningOp());
            temporaryOps->push(constantIV.getDefiningOp());
            temporaryOps->push(lbOffsetIV.getDefiningOp());
        }
        return lowerBoundOffsetIVs;
    };

    auto activeBlockDims = fullCacheShape.size() - multiCacheIVs.size();
    std::vector<int64_t> zeroActiveBlockIndices(activeBlockDims, 0);

    auto setMultiCacheIndices = [&](mlir::Operation* op, const std::vector<int64_t>& multiCacheCurrentIVs, std::stack<mlir::Operation*>* temporaryOps) {
        assert(multiCacheIVs.size() == multiCacheCurrentIVs.size());
        for (const auto& [multiCacheCurrentIV, multiCacheIV] : llvm::zip(multiCacheCurrentIVs, multiCacheIVs))
        {
            mlir::Value constantIV = currentBuilder.create<mlir::ConstantIndexOp>(loc, multiCacheCurrentIV);
            op->replaceUsesOfWith(multiCacheIV, constantIV);

            temporaryOps->push(constantIV.getDefiningOp());
        }
    };
    std::vector<int64_t> zeroMultiCacheIndices(multiCacheIVs.size(), 0);

    // For the purposes of this check we don't care which array we're supposed to be loading from or storing to, as we only care about the memory address strides
    // Therefore, just create affine loads for both arrays for simplicity
    std::vector<mlir::Value> initGlobalIndices = computeGlobalIndices(zeroActiveBlockIndices, prevTemporaryOps);
    mlir::AffineLoadOp prevOuterArrayAccessOp = CreateLoad(currentBuilder, loc, outerArray, initGlobalIndices);
    mlir::AffineLoadOp prevCacheArrayAccessOp = CreateLoad(currentBuilder, loc, cacheArray, initGlobalIndices);
    // Set the multicache index constants
    setMultiCacheIndices(prevOuterArrayAccessOp, zeroMultiCacheIndices, prevTemporaryOps);
    setMultiCacheIndices(prevCacheArrayAccessOp, zeroMultiCacheIndices, prevTemporaryOps);

    bool allSingleElementStrides = InMemoryLoopnestRunner(fullCacheShape, fullCacheStepSizes, [&](const std::vector<int64_t>& currentIVs) {
        // Returns true if the loopnest should continue running, false if it should exit early
        if (std::all_of(currentIVs.begin(), currentIVs.end(), [](int64_t idx) { return idx == 0; }))
        {
            // Don't compute anything for the first iteration of the loops, as we're already holding the initial access ops
            return true;
        }
        util::TempOpCleanupGuard prevOpCleanupGuard(prevTemporaryOps, rewriter);
        std::vector<int64_t> multiCacheCurrentIVs(currentIVs.begin(), currentIVs.begin() + multiCacheIVs.size());
        std::vector<int64_t> activeBlockCurrentIVs(currentIVs.begin() + multiCacheIVs.size(), currentIVs.end());

        auto lowerBoundOffsetIVs = computeGlobalIndices(activeBlockCurrentIVs, currentTemporaryOps);
        mlir::AffineLoadOp currentOuterArrayAccessOp = CreateLoad(currentBuilder, loc, outerArray, lowerBoundOffsetIVs);
        mlir::AffineLoadOp currentCacheArrayAccessOp = CreateLoad(currentBuilder, loc, cacheArray, lowerBoundOffsetIVs);
        setMultiCacheIndices(currentOuterArrayAccessOp, multiCacheCurrentIVs, currentTemporaryOps);
        setMultiCacheIndices(currentCacheArrayAccessOp, multiCacheCurrentIVs, currentTemporaryOps);

        // Resolve the position in the memref for each access
        std::vector<mlir::Value> prevOuterArrayIndicesVec(prevOuterArrayAccessOp.indices().begin(), prevOuterArrayAccessOp.indices().end());
        std::vector<mlir::Value> prevCacheArrayIndicesVec(prevCacheArrayAccessOp.indices().begin(), prevCacheArrayAccessOp.indices().end());
        std::vector<mlir::Value> currentOuterArrayIndicesVec(currentOuterArrayAccessOp.indices().begin(), currentOuterArrayAccessOp.indices().end());
        std::vector<mlir::Value> currentCacheArrayIndicesVec(currentCacheArrayAccessOp.indices().begin(), currentCacheArrayAccessOp.indices().end());

        auto prevOuterArrayAccessMapComposition = util::GetIndexToMemoryLocationMap(currentBuilder.getContext(), prevOuterArrayAccessOp);
        auto prevCacheArrayAccessMapComposition = util::GetIndexToMemoryLocationMap(currentBuilder.getContext(), prevCacheArrayAccessOp);
        auto currentOuterArrayAccessMapComposition = util::GetIndexToMemoryLocationMap(currentBuilder.getContext(), currentOuterArrayAccessOp);
        auto currentCacheArrayAccessMapComposition = util::GetIndexToMemoryLocationMap(currentBuilder.getContext(), currentCacheArrayAccessOp);

        auto prevOuterArrayAccess = util::MultiDimAffineApply(currentBuilder, loc, prevOuterArrayAccessMapComposition, prevOuterArrayIndicesVec);
        auto prevCacheArrayAccess = util::MultiDimAffineApply(currentBuilder, loc, prevCacheArrayAccessMapComposition, prevCacheArrayIndicesVec);
        auto currentOuterArrayAccess = util::MultiDimAffineApply(currentBuilder, loc, currentOuterArrayAccessMapComposition, currentOuterArrayIndicesVec);
        auto currentCacheArrayAccess = util::MultiDimAffineApply(currentBuilder, loc, currentCacheArrayAccessMapComposition, currentCacheArrayIndicesVec);

        assert(prevOuterArrayAccess.size() == 1);
        assert(prevCacheArrayAccess.size() == 1);
        assert(currentOuterArrayAccess.size() == 1);
        assert(currentCacheArrayAccess.size() == 1);

        prevTemporaryOps->push(prevOuterArrayAccess[0].getDefiningOp());
        prevTemporaryOps->push(prevCacheArrayAccess[0].getDefiningOp());
        prevTemporaryOps->push(currentOuterArrayAccess[0].getDefiningOp());
        prevTemporaryOps->push(currentCacheArrayAccess[0].getDefiningOp());

        mlir::AffineExpr diffExpr = currentBuilder.getAffineDimExpr(1) - currentBuilder.getAffineDimExpr(0);
        auto outerArrayDiffMap = mlir::AffineMap::get(2, 0, diffExpr);
        auto cacheArrayDiffMap = mlir::AffineMap::get(2, 0, diffExpr);

        mlir::SmallVector<mlir::Value, 4> compareOuterArrayAccesses{ prevOuterArrayAccess[0], currentOuterArrayAccess[0] };
        mlir::SmallVector<mlir::Value, 4> compareCacheArrayAccesses{ prevCacheArrayAccess[0], currentCacheArrayAccess[0] };
        mlir::fullyComposeAffineMapAndOperands(&outerArrayDiffMap, &compareOuterArrayAccesses);
        mlir::fullyComposeAffineMapAndOperands(&cacheArrayDiffMap, &compareCacheArrayAccesses);

        // At this point we don't need the load ops anymore so hold the current accesses as the next iteration's previous accesses
        // and erase the previous access ops
        rewriter.eraseOp(prevOuterArrayAccessOp);
        rewriter.eraseOp(prevCacheArrayAccessOp);
        prevOuterArrayAccessOp = currentOuterArrayAccessOp;
        prevCacheArrayAccessOp = currentCacheArrayAccessOp;
        // Erase the previous temporary ops as we're going along so we don't allocate too much excess memory and leave too many ops around during this procedure

        std::swap(prevTemporaryOps, currentTemporaryOps); // The currentTemporaryOps are the prevTemporaryOps in the next iteration

        assert(outerArrayDiffMap.getNumResults() == 1);
        assert(cacheArrayDiffMap.getNumResults() == 1);

        auto outerArrayResultExpr = outerArrayDiffMap.getResult(0);
        auto cacheArrayResultExpr = cacheArrayDiffMap.getResult(0);
        if (outerArrayResultExpr.isa<mlir::AffineConstantExpr>() && cacheArrayResultExpr.isa<mlir::AffineConstantExpr>())
        {
            auto outerArrayConstExpr = outerArrayResultExpr.dyn_cast<mlir::AffineConstantExpr>();
            auto cacheArrayConstExpr = cacheArrayResultExpr.dyn_cast<mlir::AffineConstantExpr>();
            if (outerArrayConstExpr.getValue() != cacheArrayConstExpr.getValue())
            {
                // The outer array and cache array have a different stride between these two accesses, therefore the cache
                // will not be a strict sub-buffer copy of the outer array
                return false;
            }
            else if (outerArrayConstExpr.getValue() != 1)
            {
                // As a conservative check, additionally only interpret a stride of 1 between the accesses as indicating the cache is a strict subbuffer of the outer array
                return false;
            }
        }
        else
        {
            // One of the strides was non-constant so we can't assert that it is a strict subbuffer
            return false;
        }

        // At this point, both strides were constant 1's, so continue on to the next index
        return true;
    });

    // Do a final cleanup of the ops we created for this check
    rewriter.eraseOp(prevOuterArrayAccessOp);
    rewriter.eraseOp(prevCacheArrayAccessOp);

    while (!prevTemporaryOps->empty())
    {
        auto eraseOp = prevTemporaryOps->top();
        assert(eraseOp->use_empty());
        rewriter.eraseOp(eraseOp);
        prevTemporaryOps->pop();
    }
    assert(temporaryOpsOne.empty());
    assert(temporaryOpsTwo.empty());

    return allSingleElementStrides;
}

std::pair<mlir::Block::iterator, mlir::Block::iterator> GetCacheRegionIterators(MultiCacheCopyOp copyOp)
{
    auto pairOp = GetCacheOpPair(copyOp);

    auto cacheArray = copyOp.cache();
    auto parentBlock = copyOp->getBlock();
    mlir::Block::iterator beginReplace;
    mlir::Block::iterator endReplace;
    if (pairOp)
    {
        auto firstOp = util::GetFirstOp(copyOp, pairOp);
        auto secondOp = copyOp == firstOp ? pairOp : copyOp;
        beginReplace = firstOp->getIterator();
        beginReplace++;
        endReplace = secondOp->getIterator();
    }
    else
    {
        // This op doesn't have a pair op, so if we want to replace all uses of the cache that could be impacted by this op
        // then we need to examine all uses of the cache after this op since a multi-cache copy is only a copy-in cache copy op,
        // but without stepping past other cache ops for this cache.
        // Note: multiple cache ops for the same cache can occur on the same level of the loopnest since separate active block
        // regions or separate trigger regions for the same cache can occur on the same level due to boundary conditions. Since
        // each of these regions should be considered independently, we only deal with our current op and therefore current region
        // in each instance of this lowering
        beginReplace = copyOp->getIterator();
        beginReplace++;
        endReplace = beginReplace;
        for (auto iter = beginReplace; iter != parentBlock->end(); ++iter)
        {
            // Only break on other cache ops
            if (!IsCacheOp(&(*iter)) ||
                (std::find(iter->operand_begin(), iter->operand_end(), cacheArray) == iter->operand_end()))
            {
                // This op doesn't use the cache so continue iterating past this op
                endReplace = iter;
            }
            else
            {
                // Don't advance the endReplace iterator in this case as we always advance it once after the loop
                break;
            }
        }
        // Advance the endIterator as the last op we had it pointing at is the last one to consider replacing the cache
        // usage in, so advancing it now will have it point to the op after the last one we'll make the replacement in
        endReplace++;
    }

    return std::make_pair(beginReplace, endReplace);
}

std::pair<mlir::Block::iterator, mlir::Block::iterator> GetCacheRegionIterators(ActiveBlockCacheCopyOp copyOp)
{
    auto pairOp = GetCacheOpPair(copyOp);
    auto parentBlock = copyOp->getBlock();
    auto cacheArray = copyOp.cache();
    mlir::Block::iterator beginReplace;
    mlir::Block::iterator endReplace;
    if (pairOp)
    {
        auto firstOp = util::GetFirstOp(copyOp, pairOp);
        auto secondOp = copyOp == firstOp ? pairOp : copyOp;
        beginReplace = firstOp->getIterator();
        beginReplace++;
        endReplace = secondOp->getIterator();
    }
    else
    {
        // This op doesn't have a pair op, so if we want to replace all uses of the cache that could be impacted by this op
        // then we need to examine all uses of the cache either before this op if it is a copy-out cache copy op,
        // or after this op if it is a copy-in cache copy op, but without stepping past other cache ops for this cache.
        // Note: multiple cache ops for the same cache can occur on the same level of the loopnest since separate active block
        // regions or separate trigger regions for the same cache can occur on the same level due to boundary conditions. Since
        // each of these regions should be considered independently, we only deal with our current op and therefore current region
        // in each instance of this lowering
        bool copyIn = copyOp.toCache();
        if (copyIn)
        {
            // A copy-in cache copy op without a pair op occurs in a graph like:
            // for ... {
            //     cache_copy(outer array -> cache, copy_in = true)
            //     for ... {
            //         ... // read from cache
            //     }
            //     ...
            //     for ... {
            //         ... // read from cache
            //     }
            //     (end of cache region)
            // }
            // In this case, we need to search forward in the graph until we either reach the end of the block
            // or find another cache op that is using the cache to determine the region of ops that are affected
            // by this cache

            beginReplace = copyOp->getIterator();
            beginReplace++;
            endReplace = beginReplace;
            for (auto iter = beginReplace; iter != parentBlock->end(); ++iter)
            {
                if (!IsCacheOp(&(*iter)) ||
                    (std::find(iter->operand_begin(), iter->operand_end(), cacheArray) == iter->operand_end()))
                {
                    // This op doesn't use the cache so continue iterating past this op
                    endReplace = iter;
                }
                else
                {
                    // Don't advance the endReplace iterator in this case as we always advance it once after the loop
                    break;
                }
            }
            // Advance the endIterator as the last op we had it pointing at is the last one to consider replacing the cache
            // usage in, so advancing it now will have it point to the op after the last one we'll make the replacement in
            endReplace++;
        }
        else
        {
            // A copy-out cache copy op without a pair op occurs in a graph like:
            // for ... {
            //     (beginning of cache region)
            //     for ... {
            //         ... // write to cache
            //     }
            //     ...
            //     for ... {
            //         ... // write to cache
            //     }
            //     cache_copy(cache -> outer array, copy_in = false)
            // }
            // In this case, we need to search backward in the graph until we either reach the beginning of the block
            // or find another cache op that is using the cache to determine the region of ops that are affected
            // by this cache

            endReplace = copyOp->getIterator();
            beginReplace = endReplace;
            if (endReplace != parentBlock->begin())
            {
                for (auto iter = --mlir::Block::iterator(endReplace); iter != --(parentBlock->begin()); --iter)
                {
                    if (!IsCacheOp(&(*iter)) ||
                        (std::find(iter->operand_begin(), iter->operand_end(), cacheArray) == iter->operand_end()))
                    {
                        // This op doesn't use the cache so continue iterating past this op
                        beginReplace = iter;
                    }
                    else
                    {
                        // Don't advance the beginReplace iterator in this case as we always advance it once after the loop
                        break;
                    }
                }
            }
        }
    }

    return std::make_pair(beginReplace, endReplace);
}

std::pair<mlir::Block::iterator, mlir::Block::iterator> GetCacheRegionIterators(ActiveBlockCacheReduceOp reduceOp)
{
    auto pairOp = GetCacheOpPair(reduceOp);
    assert(pairOp != nullptr);
    [[maybe_unused]] auto parentBlock = reduceOp->getBlock();
    mlir::Block::iterator beginReplace = pairOp->getIterator();
    beginReplace++;
    mlir::Block::iterator endReplace = reduceOp->getIterator();

    return std::make_pair(beginReplace, endReplace);
}

template <typename CacheOp>
void EraseThriftyCache(PatternRewriter& rewriter, CacheOp cacheOp, mlir::Value outerArray, mlir::Value cacheArray)
{
    // To do so, we need to remove this cache op and the pair cache op if it exists, and update any uses of this cache within this op's block scope
    // to use the outer array instead of the cache

    auto pairOp = GetCacheOpPair(cacheOp);
    auto loc = cacheOp.getLoc();
    auto parentBlock = cacheOp->getBlock();
    auto [beginReplace, endReplace] = GetCacheRegionIterators(cacheOp);

    parentBlock->walk(beginReplace, endReplace, [&](Operation* op) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        if (std::find(op->operand_begin(), op->operand_end(), cacheArray) != op->operand_end())
        {
            // This op uses the cacheArray, only AffineLoad and AffineStore on cache arrays are supported
            TypeSwitch<Operation*>(op)
                .Case([&](mlir::AffineLoadOp affineLoadOp) {
                    rewriter.setInsertionPoint(affineLoadOp);
                    auto baseArrayPosition = GetBaseArrayPosition(rewriter, loc, affineLoadOp);

                    mlir::AffineLoadOp newLoadOp = CreateLoad(rewriter, loc, outerArray, baseArrayPosition);
                    affineLoadOp.replaceAllUsesWith(newLoadOp.getResult());
                    rewriter.eraseOp(affineLoadOp);
                })
                .Case([&](mlir::AffineStoreOp affineStoreOp) {
                    mlir::AffineStoreOp::Adaptor storeAdaptor{ affineStoreOp };
                    rewriter.setInsertionPoint(affineStoreOp);
                    auto baseArrayPosition = GetBaseArrayPosition(rewriter, loc, affineStoreOp);

                    CreateStore(rewriter, loc, storeAdaptor.value(), outerArray, baseArrayPosition);
                    rewriter.eraseOp(affineStoreOp);
                })
                .Case([&](MultiCacheCopyOp copyOp) {
                    copyOp->replaceUsesOfWith(cacheArray, outerArray);
                })
                .Case([&](ActiveBlockCacheCopyOp copyOp) {
                    copyOp->replaceUsesOfWith(cacheArray, outerArray);
                })
                .Case([&](ActiveBlockCacheReduceOp reduceOp) {
                    reduceOp->replaceUsesOfWith(cacheArray, outerArray);
                })
                .Default([&](Operation* defaultOp) {
                    assert(false && "Usage of mapped op found that doesn't have an op conversion registered!");
                });
        }
    });

    rewriter.eraseOp(cacheOp);
    if (pairOp)
    {
        rewriter.eraseOp(pairOp);
    }
}

LogicalResult ThriftyCacheMultiCopyOpRewrite::matchAndRewrite(MultiCacheCopyOp multiCacheCopyOp, PatternRewriter& rewriter) const
{
    // If this cache op is for a thrifty cache:
    // - Check if there is a pair op for this, e.g. a cache-copy-in paired with a cache-copy-out
    // - Check if the source array and cache array cover the same sequence over the active block. I.e. as we step in the cache region domain both arrays have the same stride for every element
    // - If the source and cache cover a consistent stride-1 region and therefore should be elided according to the thrifty definition,
    //      then erase this cache op and its corresponding pair op (if one exists)
    // If this cache op is not a thrifty cache, then return success and do nothing

    if (!multiCacheCopyOp.thrifty())
    {
        // Not a thrifty cache, so we'll always realize the cache regardless of memory ordering
        return failure();
    }

    MultiCacheCopyOp::Adaptor adaptor{ multiCacheCopyOp };
    auto outerArray = multiCacheCopyOp.array();
    auto cacheArray = multiCacheCopyOp.cache();

    std::optional<v::ExecutionTarget> execTargetOpt = util::ResolveExecutionTarget(multiCacheCopyOp);
    assert(execTargetOpt.has_value());
    auto execTarget = *execTargetOpt;
    if (execTarget == v::ExecutionTarget::GPU && !SameMemorySpace(outerArray, cacheArray))
    {
        // The cache array is in a different memory space than the outer array, so don't elide this cache
        return failure();
    }

    auto loc = multiCacheCopyOp.getLoc();

    MultiCacheLoopInfo multiCacheInfo = CreateMultiCacheLoops(rewriter, multiCacheCopyOp, [&](mlir::OpBuilder& currentBuilder, const MultiCacheLoopInfo& info) {
        auto lbMapsArrayAttr = adaptor.activeBlockLowerBoundMaps();
        auto ubMapsArrayAttr = adaptor.activeBlockUpperBoundMaps();

        auto fullCacheShape = GetFullCacheShapeHelper(info.multiCacheShape, info.activeBlockExternalSymbols, info.activeBlockExternalSymbols, lbMapsArrayAttr, ubMapsArrayAttr);
        if (fullCacheShape.size() != 0 && std::find(fullCacheShape.begin(), fullCacheShape.end(), 0) == fullCacheShape.end())
        {
            assert(fullCacheShape.size() >= info.multiCacheShape.size());
            std::vector<int64_t> activeBlockStepSizes(fullCacheShape.size() - info.multiCacheShape.size(), 1); // TODO : do we have a scenario where the multicache has step sizes > 1 ?
            auto fullCacheStepSizes = info.multiCacheStepSizes;
            fullCacheStepSizes.insert(fullCacheStepSizes.end(), activeBlockStepSizes.begin(), activeBlockStepSizes.end());
            bool allSingleElementStrides = ThriftyCacheAllSingleElementStridesHelper(rewriter,
                                                                                     currentBuilder,
                                                                                     loc,
                                                                                     outerArray,
                                                                                     cacheArray,
                                                                                     info.multiCacheIVs,
                                                                                     fullCacheShape,
                                                                                     fullCacheStepSizes,
                                                                                     info.activeBlockExternalSymbols,
                                                                                     lbMapsArrayAttr,
                                                                                     ubMapsArrayAttr);

            if (allSingleElementStrides)
            {
                // If the accesses into the arrays all had strides of 1, then the cache is a strict subbuffer of the outer array.
                // Since it is a thrifty cache we should therefore elide this cache.
                EraseThriftyCache(rewriter, multiCacheCopyOp, outerArray, cacheArray);
            }
        }
    });
    // Clean up the MultiCacheLoops now that we're done with them
    // Erase innermost-to-outermost loop, so reverse the multiCacheLoops list then erase each element
    std::vector<mlir::AffineForOp> loopsToErase = multiCacheInfo.multiCacheLoops;
    std::reverse(loopsToErase.begin(), loopsToErase.end());
    for (auto& loop : loopsToErase)
    {
        rewriter.eraseOp(loop);
    }

    return success();
}

LogicalResult ThriftyCacheCopyOpRewrite::matchAndRewrite(ActiveBlockCacheCopyOp cacheCopyOp, PatternRewriter& rewriter) const
{
    // If this cache op is for a thrifty cache:
    // - Check if there is a pair op for this, e.g. a cache-copy-in paired with a cache-copy-out
    // - Check if the source array and cache array cover the same sequence over the active block. I.e. as we step in the cache region domain both arrays have the same stride for every element
    // - If the source and cache cover a consistent stride-1 region and therefore should be elided according to the thrifty definition,
    //      then erase this cache op and its corresponding pair op (if one exists)
    // If this cache op is not a thrifty cache, then return success and do nothing

    if (!cacheCopyOp.thrifty())
    {
        // Not a thrifty cache, so we'll always realize the cache regardless of memory ordering
        return failure();
    }

    // Get pair op if it exists
    [[maybe_unused]] auto pairOp = GetCacheOpPair(cacheCopyOp);

    ActiveBlockCacheCopyOp::Adaptor adaptor{ cacheCopyOp };
    auto loc = cacheCopyOp.getLoc();

    auto outerArray = cacheCopyOp.array();
    auto cacheArray = cacheCopyOp.cache();

    std::optional<v::ExecutionTarget> execTargetOpt = util::ResolveExecutionTarget(cacheCopyOp);
    assert(execTargetOpt.has_value());
    auto execTarget = *execTargetOpt;
    if (execTarget == v::ExecutionTarget::GPU && !SameMemorySpace(outerArray, cacheArray))
    {
        // The cache array is in a different memory space than the outer array, so don't elide this cache
        return failure();
    }

    // Check if the cache copy op covers the outer array in the same memory order

    auto lbMapsArrayAttr = adaptor.lbMaps();
    auto ubMapsArrayAttr = adaptor.ubMaps();
    auto lbMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(lbMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });
    auto lbOperands = adaptor.lbOperands();
    auto activeBlockShape = GetConstantActiveBlockShapeHelper(cacheCopyOp);
    std::vector<int64_t> activeBlockStepSizes(activeBlockShape.size(), 1); // TODO : do we have a scenario where the active block has step sizes > 1 ?

    if (activeBlockShape.size() == 0 || std::find(activeBlockShape.begin(), activeBlockShape.end(), 0) != activeBlockShape.end())
    {
        // There's either no active block shape or at least one of the dimensions is 0, resulting in 0 volume, so just skip over this cache op
        return failure();
    }

    std::vector<mlir::Value> lbOperandsVec(lbOperands.begin(), lbOperands.end());

    bool allSingleElementStrides = ThriftyCacheAllSingleElementStridesHelper(rewriter,
                                                                             rewriter,
                                                                             loc,
                                                                             outerArray,
                                                                             cacheArray,
                                                                             std::vector<mlir::Value>{},
                                                                             activeBlockShape,
                                                                             activeBlockStepSizes,
                                                                             lbOperandsVec,
                                                                             lbMapsArrayAttr,
                                                                             ubMapsArrayAttr);

    if (allSingleElementStrides)
    {
        // If the accesses into the arrays all had strides of 1, then the cache is a strict subbuffer of the outer array.
        // Since it is a thrifty cache we should therefore elide this cache.
        EraseThriftyCache(rewriter, cacheCopyOp, outerArray, cacheArray);
    }

    return success();
}

LogicalResult ThriftyCacheReduceOpRewrite::matchAndRewrite(ActiveBlockCacheReduceOp cacheReduceOp, PatternRewriter& rewriter) const
{
    // If this cache op is for a thrifty cache:
    // - Check if there is a pair op for this, e.g. a cache-copy-in or a cache-zero paired with this cache reduce
    // - Check if the source array and cache array cover the same sequence over the active block. I.e. as we step in the cache region domain both arrays have the same stride for every element
    // - If the source and cache cover a consistent stride-1 region and therefore should be elided according to the thrifty definition,
    //      then erase this cache op and its corresponding pair op (if one exists)
    // If this cache op is not a thrifty cache, then return success and do nothing

    if (!cacheReduceOp.thrifty())
    {
        // Not a thrifty cache, so we'll always realize the cache regardless of memory ordering
        return failure();
    }

    // Get pair op if it exists
    auto pairOp = GetCacheOpPair(cacheReduceOp);
    assert(pairOp != nullptr && "ActiveBlockCacheReduceOp must have a pair op");

    ActiveBlockCacheReduceOp::Adaptor adaptor{ cacheReduceOp };
    auto loc = cacheReduceOp.getLoc();

    auto outerArray = cacheReduceOp.array();
    auto cacheArray = cacheReduceOp.cache();

    std::optional<v::ExecutionTarget> execTargetOpt = util::ResolveExecutionTarget(cacheReduceOp);
    assert(execTargetOpt.has_value());
    auto execTarget = *execTargetOpt;
    if (execTarget == v::ExecutionTarget::GPU && !SameMemorySpace(outerArray, cacheArray))
    {
        // The cache array is in a different memory space than the outer array, so don't elide this cache
        return failure();
    }

    // Check if the cache copy op covers the outer array in the same memory order

    auto lbMapsArrayAttr = adaptor.lbMaps();
    auto ubMapsArrayAttr = adaptor.ubMaps();
    auto lbMaps = util::ArrayAttrToVector<mlir::AffineMap, mlir::AffineMapAttr>(lbMapsArrayAttr, [](const mlir::AffineMapAttr& mapAttr) -> mlir::AffineMap {
        return mapAttr.getValue();
    });
    auto lbOperands = adaptor.lbOperands();
    auto activeBlockShape = GetConstantActiveBlockShapeHelper(cacheReduceOp);
    std::vector<int64_t> activeBlockStepSizes(activeBlockShape.size(), 1); // TODO : do we have a scenario where the active block has step sizes > 1 ?
    if (activeBlockShape.size() == 0 || std::find(activeBlockShape.begin(), activeBlockShape.end(), 0) != activeBlockShape.end())
    {
        // There's either no active block shape or at least one of the dimensions is 0, resulting in 0 volume, so just skip over this cache op
        return failure();
    }
    std::vector<mlir::Value> lbOperandsVec(lbOperands.begin(), lbOperands.end());

    bool allSingleElementStrides = ThriftyCacheAllSingleElementStridesHelper(rewriter,
                                                                             rewriter,
                                                                             loc,
                                                                             outerArray,
                                                                             cacheArray,
                                                                             std::vector<mlir::Value>{},
                                                                             activeBlockShape,
                                                                             activeBlockStepSizes,
                                                                             lbOperandsVec,
                                                                             lbMapsArrayAttr,
                                                                             ubMapsArrayAttr);

    if (allSingleElementStrides)
    {
        // If the accesses into the arrays all had strides of 1, then the cache is a strict subbuffer of the outer array.
        // Since it is a thrifty cache we should therefore elide this cache.
        EraseThriftyCache(rewriter, cacheReduceOp, outerArray, cacheArray);
    }

    return success();
}

mlir::Value FindParentAffineForOpIV(mlir::Operation* op, const Index& loopnestIndex)
{
    mlir::AffineForOp currentParentForOp = op->getParentOfType<mlir::AffineForOp>();
    while (currentParentForOp)
    {
        if (auto indexAttr = currentParentForOp->getAttrOfType<IndexAttr>("index"))
        {
            if (indexAttr.getValue() == loopnestIndex)
            {
                return currentParentForOp.getInductionVar();
            }
        }
        currentParentForOp = currentParentForOp->getParentOfType<mlir::AffineForOp>();
    }
    assert(false && "Given loopnest index does not correspond to a parent AffineForOp");
}

llvm::SmallVector<mlir::Value, 4> ResolveParentRelevantScheduleIndices(mlir::Operation* op, const mlir::ValueRange& baseRelevantScheduleIndices)
{
    llvm::SmallVector<mlir::Value, 4> resolvedRelevantScheduleIndices;

    for (auto scheduleIndexValue : baseRelevantScheduleIndices)
    {
        // All of these values should be block arguments or symbolic index ops
        // SymbolicIndexOps corresponding to loops outside of this CacheMappingOp should have already
        // been replaced with block arguments, however SymbolicIndexOps for loops contained within this
        // CacheMappingOp will not have been replaced yet
        if (scheduleIndexValue.isa<mlir::BlockArgument>())
        {
            resolvedRelevantScheduleIndices.push_back(scheduleIndexValue);
        }
        else
        {
            auto definingOp = scheduleIndexValue.getDefiningOp();
            auto symIndexOp = mlir::dyn_cast<SymbolicIndexOp>(definingOp);
            assert(symIndexOp != nullptr);
            resolvedRelevantScheduleIndices.push_back(FindParentAffineForOpIV(op, symIndexOp.index().getValue()));
        }
    }

    return resolvedRelevantScheduleIndices;
}

bool CacheMappingOpsConflict(BeginCacheMappingOp leftOp, BeginCacheMappingOp rightOp)
{
    // Returns true if the two cache mapping ops are for the same fromValue or baseCacheValue and should therefore
    // not overlap
    return leftOp.fromValue() == rightOp.fromValue() ||
           leftOp.baseCacheValue() == rightOp.baseCacheValue() ||
           leftOp.baseInput() == rightOp.baseInput() ||
           leftOp.toValue() == rightOp.toValue();
}

bool OpUsesMappingRelatedBuffers(BeginCacheMappingOp mappingOp, mlir::Operation* op)
{
    auto fromValue = mappingOp.fromValue();
    auto baseCacheValue = mappingOp.baseCacheValue();
    auto baseInput = mappingOp.baseInput();
    auto toValue = mappingOp.toValue();
    bool usesFromValue = std::find(op->operand_begin(), op->operand_end(), fromValue) != op->operand_end();
    bool usesBaseCacheValue = std::find(op->operand_begin(), op->operand_end(), baseCacheValue) != op->operand_end();
    bool usesBaseInput = std::find(op->operand_begin(), op->operand_end(), baseInput) != op->operand_end();
    bool usesToValue = std::find(op->operand_begin(), op->operand_end(), toValue) != op->operand_end();
    return usesFromValue || usesBaseCacheValue || usesBaseInput || usesToValue;
}

LogicalResult AdjustHierarchicalCacheRegionPositionRewrite::matchAndRewrite(BeginCacheRegionOp beginCacheRegionOp, PatternRewriter& rewriter) const
{
    // Examine the block that the BeginCacheRegionOp is in and rearrange BeginCacheRegionOps and their corresponding
    // EndCacheRegionOps such that the lowest cache hierarchy level regions begin first and end last, followed by the next lowest
    // hierarchy levels and so on.
    // This is to ensure that inner hierarchical cache regions are always fully contained within their outer hierarchical
    // cache's region.
    // While moving ops, don't move the regions past any AffineForOps as those subnests may have other cache regions inside of them
    // that we don't want to affect

    // TODO : we should explore using real MLIR blocks and regions for these cache regions so we don't have to worry about this issue as much.
    //        Currently we don't use regions due to how the MLIR MemRefRegion access computation behaves, as it only plays nicely
    //        with nested AffineForOps without any other ops nested in between.

    auto parentBlock = beginCacheRegionOp->getBlock();
    auto blockBegin = parentBlock->begin();
    auto blockEnd = parentBlock->end();
    auto endOp = mlir::dyn_cast<EndCacheRegionOp>(beginCacheRegionOp.getEndOp());

    // First, find the range of BeginCacheRegionOps that we should consider moving this op before or after
    // We need to find all of the BeginCacheRegionOps that both succeed and precede all of the same AffineForOps as this one
    mlir::Block::iterator targetIter(beginCacheRegionOp);
    mlir::Block::iterator startIter(blockBegin);
    mlir::Block::iterator endIter(beginCacheRegionOp);

    // Walk the block ops until we find beginCacheRegionOp. Any time we see an AffineForOp, adjust startIter to be after it
    for (auto opIter = blockBegin; opIter != targetIter; ++opIter)
    {
        if (mlir::isa<mlir::AffineForOp>(&(*opIter)))
        {
            startIter = opIter->getIterator();
            ++startIter;
        }
    }

    // Now walk the block ops from the beginCacheRegionOp onwards until we either reach the end of the block or find an AffineForOp and set the endIter to that position
    bool foundAffineForOp = false;
    for (auto opIter = mlir::Block::iterator(beginCacheRegionOp); opIter != blockEnd; ++opIter)
    {
        endIter = opIter;
        if (mlir::isa<mlir::AffineForOp>(&(*opIter)))
        {
            foundAffineForOp = true;
            break;
        }
    }
    if (!foundAffineForOp)
    {
        endIter = blockEnd;
    }

    // Now that we have the range of ops to consider positioning our beginCacheRegionOp against, we search that range for
    // all of the BeginCacheRegionOps with the same baseInput that have already been positioned and position our beginCacheRegionOp
    // accordingly.
    // Note: if a BeginCacheRegionOp is missing the "hierarchical_positioned" UnitAttr, then this pass hasn't been run on it yet and
    // it may get moved later, so we don't bother considering positioning against those. This also means that the very first BeginCacheRegionOp
    // in a range won't get moved
    for (auto opIter = startIter; opIter != endIter; ++opIter)
    {
        if (auto otherBeginOp = mlir::dyn_cast<BeginCacheRegionOp>(&(*opIter));
            otherBeginOp != nullptr && otherBeginOp != beginCacheRegionOp)
        {
            if (otherBeginOp->hasAttrOfType<mlir::UnitAttr>("hierarchical_positioned") && otherBeginOp.baseInput() == beginCacheRegionOp.baseInput())
            {
                auto otherEndOp = mlir::dyn_cast<EndCacheRegionOp>(otherBeginOp.getEndOp());
                assert(otherBeginOp.cacheHierarchyLevel() != beginCacheRegionOp.cacheHierarchyLevel() && "Two caches for the same input with the same cache hierarchy level are not supported in the same loop region");
                if (otherBeginOp.cacheHierarchyLevel() < beginCacheRegionOp.cacheHierarchyLevel())
                {
                    // The other region's hierarchy is lower, so it is for the outer cache and this beginCacheRegionOp should
                    // come after it, and the corresponding endOp should be before otherEndOp
                    beginCacheRegionOp->moveAfter(otherBeginOp);

                    // Only move the endOp if it is after the otherEndOp. In cases where there are multiple AffineForOps in the block
                    // we don't want to move the end op if we don't have to for safety purposes. The begin ops have already accounted for
                    // this
                    // E.g. if we had a graph like:
                    // begin_0
                    //   begin_1
                    //     affine.for ...
                    //   end_1
                    //   begin_1'
                    //     affine.for ...
                    //   end_1'
                    // end_0
                    // Then we wouldn't want to move end_1 to be before end_0 because that would affect another region
                    if (util::GetFirstOp(endOp, otherEndOp) == otherEndOp)
                    {
                        endOp->moveBefore(otherEndOp);
                    }
                }
                else
                {
                    // The other region's hierarchy is higher, so it is for an inner cache and this beginCacheRegionOp should
                    // come before it, and the corresponding endOp should be after otherEndOp
                    beginCacheRegionOp->moveBefore(otherBeginOp);

                    // Similar to the other case, only move the endOp if it is before the otherEndOp
                    if (util::GetFirstOp(endOp, otherEndOp) == endOp)
                    {
                        endOp->moveAfter(otherEndOp);
                    }
                }
            }
        }
    }
    beginCacheRegionOp->setAttr("hierarchical_positioned", rewriter.getUnitAttr());
    return success();
}

LogicalResult AdjustCacheMappingPositionRewrite::matchAndRewrite(BeginCacheMappingOp beginCacheMappingOp, PatternRewriter& rewriter) const
{
    // Examine the block that the BeginCacheMappingOp is in and move the BeginCacheMappingOp ahead of any ops that precede it except
    // for other BeginCacheMappingOps/EndCacheMappingOps that deal with the same fromValue or baseCacheValue or cache reduce / cache->array copies
    // as those are for a previous mapping.
    // Similarly, move the corresponding EndCacheMappingOp after any ops in the block except for other BeginCacheMappingOps for the same
    // fromValue or baseCacheValue or MultiCacheCopy or array->cache copies as those are for a later mapping.
    // Don't move the mappings past any AffineForOp subnests, as those could contain other caches for our base input that we don't want to mess with

    auto parentBlock = beginCacheMappingOp->getBlock();
    auto blockBegin = parentBlock->begin();
    // If the parent block has a terminator op, then set our blockEnd iterator to point at it, otherwise point at the
    // end iterator of the op list
    auto blockEnd = parentBlock->end();
    if (parentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
    {
        blockEnd = mlir::Block::iterator(&(parentBlock->back()));
    }

    auto endOp = mlir::dyn_cast<EndCacheMappingOp>(beginCacheMappingOp.getEndOp());
    mlir::Block::iterator initBeginPos(beginCacheMappingOp);
    mlir::Block::iterator initEndPos(endOp);

    [[maybe_unused]] auto fromValue = beginCacheMappingOp.fromValue();
    [[maybe_unused]] auto baseCacheValue = beginCacheMappingOp.baseCacheValue();
    [[maybe_unused]] auto toValue = beginCacheMappingOp.toValue();

    // Walk the ops that precede this BeginCacheMappingOp in reverse order to find the new position for the BeginCacheMappingOp
    auto newBeginPos = initBeginPos;
    // Start at the first op before our mapping op
    if (initBeginPos != blockBegin)
    {
        // Only try to move up the beginning of the mapping if it is not already the first op in the block (otherwise --mlir::Block::iterator(initBeginPos) will cycle around to the end of the block)
        for (auto iter = --mlir::Block::iterator(initBeginPos); iter != --mlir::Block::iterator(blockBegin); --iter)
        {
            bool foundConflict = false;
            TypeSwitch<Operation*>(&(*iter))
                .Case([&](BeginCacheMappingOp predecessorBeginOp) {
                    // If this cache mapping region deals with our array, base placeholder cache, or cache
                    foundConflict |= CacheMappingOpsConflict(predecessorBeginOp, beginCacheMappingOp);
                })
                .Case([&](EndCacheMappingOp predecessorEndOp) {
                    // If this cache mapping region deals with our array, base placeholder cache, or cache
                    auto predecessorBeginOp = mlir::dyn_cast<BeginCacheMappingOp>(predecessorEndOp.getBeginOp());
                    foundConflict |= CacheMappingOpsConflict(predecessorBeginOp, beginCacheMappingOp);
                })
                .Case([&](ActiveBlockCacheCopyOp predecessorCopyOp) {
                    // If this copy deals with our array, base placeholder cache, or cache and it is a cache -> array copy, then don't step past it
                    if (!predecessorCopyOp.toCache())
                    {
                        foundConflict |= OpUsesMappingRelatedBuffers(beginCacheMappingOp, predecessorCopyOp);
                    }
                })
                .Case([&](ActiveElementCacheCopyOp predecessorCopyOp) {
                    // If this copy deals with our array, base placeholder cache, or cache then don't step past it
                    foundConflict |= OpUsesMappingRelatedBuffers(beginCacheMappingOp, predecessorCopyOp);
                })
                .Case([&](ActiveBlockCacheReduceOp predecessorReduceOp) {
                    // If this reduce deals with our array, base placeholder cache, or cache then don't step past it
                    foundConflict |= OpUsesMappingRelatedBuffers(beginCacheMappingOp, predecessorReduceOp);
                })
                .Case([&](ActiveElementCacheReduceOp predecessorReduceOp) {
                    // If this reduce deals with our array, base placeholder cache, or cache then don't step past it
                    foundConflict |= OpUsesMappingRelatedBuffers(beginCacheMappingOp, predecessorReduceOp);
                })
                .Case([&](AffineForOp forOp) {
                    foundConflict = true;
                });
            if (foundConflict)
            {
                break;
            }
            else
            {
                newBeginPos = iter;
            }
        }
    }

    // Now find the new position for the end op

    // Walk the ops after this EndCacheMappingOp in order to find the new position for the EndCacheMappingOp
    // Start at the first op after our end op
    auto newEndPos = std::find_if(++mlir::Block::iterator(initEndPos), blockEnd, [&](Operation& op) {
        bool foundConflict = false;
        TypeSwitch<Operation*>(&op)
            .Case([&](BeginCacheMappingOp successorBeginOp) {
                // If this cache mapping region deals with our array, base placeholder cache, or cache
                foundConflict |= CacheMappingOpsConflict(successorBeginOp, beginCacheMappingOp);
            })
            .Case([&](EndCacheMappingOp successorEndOp) {
                // If this cache mapping region deals with our array, base placeholder cache, or cache
                auto successorBeginOp = mlir::dyn_cast<BeginCacheMappingOp>(successorEndOp.getBeginOp());
                foundConflict |= CacheMappingOpsConflict(successorBeginOp, beginCacheMappingOp);
            })
            .Case([&](MultiCacheCopyOp successorCopyOp) {
                // If this copy deals with our array, base placeholder cache, or cache and it is an array -> cache array copy, then don't step past it
                foundConflict |= OpUsesMappingRelatedBuffers(beginCacheMappingOp, successorCopyOp);
            })
            .Case([&](ActiveBlockCacheCopyOp successorCopyOp) {
                // If this copy deals with our array, base placeholder cache, or cache and it is an array -> cache array copy, then don't step past it
                if (successorCopyOp.toCache())
                {
                    foundConflict |= OpUsesMappingRelatedBuffers(beginCacheMappingOp, successorCopyOp);
                }
            })
            .Case([&](ActiveElementCacheCopyOp successorCopyOp) {
                // If this copy deals with our array, base placeholder cache, or cache then don't step past it
                foundConflict |= OpUsesMappingRelatedBuffers(beginCacheMappingOp, successorCopyOp);
            })
            .Case([&](ActiveElementCacheReduceOp successorReduceOp) {
                // If this reduce deals with our array, base placeholder cache, or cache then don't step past it
                foundConflict |= OpUsesMappingRelatedBuffers(beginCacheMappingOp, successorReduceOp);
            })
            .Case([&](AffineForOp forOp) {
                foundConflict = true;
            });
        return foundConflict;
    });

    if (newEndPos != initEndPos)
    {
        // newEndPos points at the earliest successor op that we shouldn't move past
        endOp->moveBefore(parentBlock, newEndPos);
    }

    if (newBeginPos != initBeginPos)
    {
        // newBeginPos points at the earliest predecessor op that we should move past (note: this is different from the newEndPos case)
        beginCacheMappingOp->moveBefore(parentBlock, newBeginPos);
    }
    return success();
}

LogicalResult BeginCacheMappingOpRewrite::matchAndRewrite(BeginCacheMappingOp beginCacheMappingOp, PatternRewriter& rewriter) const
{
    // BeginCacheMappingOp examines the subgraph contained between it and its corresponding EndCacheMappingOp,
    // replacing instances of cacheMappingOp.input() with cacheMappingOp.cache()

    // Ensure the cache mapping ops are resolved in outermost-to-innermost order to handle hierarchical caches appropriately
    // To do this, if there is another BeginCacheMappingOp in any ancestor block or in the parent block preceding this op,
    // then return failure so that the other ones get handled first

    if (!util::IsOutermostOpOfType(beginCacheMappingOp))
    {
        return failure();
    }

    EndCacheMappingOp endOp = mlir::dyn_cast<EndCacheMappingOp>(beginCacheMappingOp.getEndOp());

    auto loc = beginCacheMappingOp.getLoc();

    auto parentBlock = beginCacheMappingOp->getBlock();
    auto parentRegion = parentBlock->getParent();
    [[maybe_unused]] auto parentOp = parentRegion->getParentOp();
    bool isActiveBlockCache = beginCacheMappingOp.activeBlockCache();

    if (IsCacheRegionEmpty(beginCacheMappingOp))
    {
        rewriter.eraseOp(endOp);
        rewriter.eraseOp(beginCacheMappingOp);
        return success();
    }

    Value fromValue = beginCacheMappingOp.fromValue();
    Value baseCacheValue = beginCacheMappingOp.baseCacheValue();
    Value toValue = beginCacheMappingOp.toValue();
    CacheAccessContext toValueAccessContext = beginCacheMappingOp.getToValueAccessContext();

    auto cacheMap = isActiveBlockCache ? toValueAccessContext.accessMaps.inputIndicesToActiveBlockCache : toValueAccessContext.accessMaps.relevantIndicesToActiveElementCache;

    // Map any ops that are referencing the original non-multicache cache value
    parentBlock->walk(++mlir::Block::iterator(beginCacheMappingOp), mlir::Block::iterator(endOp), [&](Operation* op) {
        op->replaceUsesOfWith(baseCacheValue, toValue);
    });

    // Only replace each op once so we don't loop infinitely if we're remapping a cache value to itself
    // with different access maps
    std::set<Operation*> fromValueReplacementOps;

    parentBlock->walk(++mlir::Block::iterator(beginCacheMappingOp), mlir::Block::iterator(endOp), [&](Operation* op) {
        for (auto& operand : op->getOpOperands())
        {
            if (operand.get() == fromValue)
            {
                fromValueReplacementOps.insert(op);
            }
        }
    });

    while (!fromValueReplacementOps.empty())
    {
        auto replacementOpsIter = fromValueReplacementOps.begin();
        Operation* op = *replacementOpsIter;
        TypeSwitch<Operation*>(op)
            .Case([&](mlir::AffineLoadOp loadOp) {
                mlir::AffineLoadOp::Adaptor loadAdaptor{ loadOp };
                rewriter.setInsertionPoint(loadOp);
                if (isActiveBlockCache)
                {
                    auto baseArrayPosition = GetBaseArrayPosition(rewriter, loc, loadOp);
                    mlir::AffineLoadOp newLoadOp = CreateLoad(rewriter, loc, toValue, baseArrayPosition);
                    loadOp.replaceAllUsesWith(newLoadOp.getResult());
                    TransferOrSetAccessAttrs(loadOp, newLoadOp);
                    rewriter.eraseOp(loadOp);
                }
                else
                {
                    llvm::SmallVector<mlir::Value, 4> accessIndices = ResolveParentRelevantScheduleIndices(loadOp, toValueAccessContext.fullRelevantScheduleIndices);
                    rewriter.replaceOpWithNewOp<AffineLoadOp>(loadOp, toValueAccessContext.value, cacheMap, accessIndices);
                }
            })
            .Case([&](mlir::AffineStoreOp storeOp) {
                mlir::AffineStoreOp::Adaptor storeAdaptor{ storeOp };
                rewriter.setInsertionPoint(storeOp);
                if (isActiveBlockCache)
                {
                    auto baseArrayPosition = GetBaseArrayPosition(rewriter, loc, storeOp);
                    auto newStoreOp = CreateStore(rewriter, loc, storeAdaptor.value(), toValue, baseArrayPosition);
                    TransferOrSetAccessAttrs(storeOp, newStoreOp);
                    rewriter.eraseOp(storeOp);
                }
                else
                {
                    llvm::SmallVector<mlir::Value, 4> accessIndices = ResolveParentRelevantScheduleIndices(storeOp, toValueAccessContext.fullRelevantScheduleIndices);
                    rewriter.replaceOpWithNewOp<AffineStoreOp>(storeOp, storeAdaptor.value(), toValueAccessContext.value, cacheMap, accessIndices);
                }
            })
            .Case([&](mlir::memref::LoadOp loadOp) {
                rewriter.setInsertionPoint(loadOp);
                mlir::memref::LoadOp::Adaptor loadAdaptor{ loadOp };
                if (isActiveBlockCache)
                {
                    std::vector<mlir::Value> baseArrayPosition(loadAdaptor.indices().begin(), loadAdaptor.indices().end());
                    mlir::AffineLoadOp newLoadOp = CreateLoad(rewriter, loc, toValue, baseArrayPosition);
                    loadOp.replaceAllUsesWith(newLoadOp.getResult());
                    rewriter.eraseOp(loadOp);
                }
                else
                {
                    llvm::SmallVector<mlir::Value, 4> accessIndices = ResolveParentRelevantScheduleIndices(loadOp, toValueAccessContext.fullRelevantScheduleIndices);
                    rewriter.replaceOpWithNewOp<AffineLoadOp>(loadOp, toValueAccessContext.value, cacheMap, accessIndices);
                }
            })
            .Case([&](memref::StoreOp storeOp) {
                rewriter.setInsertionPoint(storeOp);
                mlir::memref::StoreOp::Adaptor storeAdaptor{ storeOp };
                if (isActiveBlockCache)
                {
                    std::vector<mlir::Value> baseArrayPosition(storeAdaptor.indices().begin(), storeAdaptor.indices().end());
                    CreateStore(rewriter, loc, storeAdaptor.value(), toValue, baseArrayPosition);
                    rewriter.eraseOp(storeOp);
                }
                else
                {
                    llvm::SmallVector<mlir::Value, 4> accessIndices = ResolveParentRelevantScheduleIndices(storeOp, toValueAccessContext.fullRelevantScheduleIndices);
                    rewriter.replaceOpWithNewOp<AffineStoreOp>(storeOp, storeAdaptor.value(), toValueAccessContext.value, cacheMap, accessIndices);
                }
            })
            .Case([&](v::LoadOp loadOp) {
                rewriter.setInsertionPoint(loadOp);
                v::LoadOp::Adaptor loadAdaptor{ loadOp };
                if (isActiveBlockCache)
                {
                    std::vector<mlir::Value> baseArrayPosition(loadAdaptor.indices().begin(), loadAdaptor.indices().end());
                    mlir::AffineLoadOp newLoadOp = CreateLoad(rewriter, loc, toValue, baseArrayPosition);
                    loadOp.replaceAllUsesWith(newLoadOp.getResult());
                    rewriter.eraseOp(loadOp);
                }
                else
                {
                    llvm::SmallVector<mlir::Value, 4> accessIndices = ResolveParentRelevantScheduleIndices(loadOp, toValueAccessContext.fullRelevantScheduleIndices);
                    rewriter.replaceOpWithNewOp<AffineLoadOp>(loadOp, toValueAccessContext.value, cacheMap, accessIndices);
                }
            })
            .Case([&](v::StoreOp storeOp) {
                rewriter.setInsertionPoint(storeOp);
                v::StoreOp::Adaptor storeAdaptor{ storeOp };
                if (isActiveBlockCache)
                {
                    std::vector<mlir::Value> baseArrayPosition(storeAdaptor.indices().begin(), storeAdaptor.indices().end());
                    CreateStore(rewriter, loc, storeAdaptor.value(), toValue, baseArrayPosition);
                    rewriter.eraseOp(storeOp);
                }
                else
                {
                    llvm::SmallVector<mlir::Value, 4> accessIndices = ResolveParentRelevantScheduleIndices(storeOp, toValueAccessContext.fullRelevantScheduleIndices);
                    rewriter.replaceOpWithNewOp<AffineStoreOp>(storeOp, storeAdaptor.value(), toValueAccessContext.value, cacheMap, accessIndices);
                }
            })
            .Case([&](v::MMALoadSyncOp loadOp) {
                v::MMALoadSyncOp::Adaptor loadAdaptor{ loadOp };
                rewriter.setInsertionPoint(loadOp);
                const v::MMAShape mmaShapeType{ static_cast<v::MMAShape>(loadOp.mmaShapeType()) };
                const v::MMAOperandType operandType{ loadOp.operandType() };
                if (isActiveBlockCache)
                {
                    auto baseArrayPosition = GetBaseArrayPosition(rewriter, loc, loadOp);
                    auto newLoadOp = CreateMMALoad(rewriter, loc, loadOp.result().getType(), toValue, mmaShapeType, operandType, baseArrayPosition);
                    loadOp.replaceAllUsesWith(newLoadOp.getResult());
                    TransferOrSetAccessAttrs(loadOp, newLoadOp);
                    rewriter.eraseOp(loadOp);
                }
                else
                {
                    llvm::SmallVector<mlir::Value, 4> accessIndices = ResolveParentRelevantScheduleIndices(loadOp, toValueAccessContext.fullRelevantScheduleIndices);
                    rewriter.replaceOpWithNewOp<v::MMALoadSyncOp>(loadOp, loadOp.result().getType(), toValueAccessContext.value, mmaShapeType, operandType, cacheMap, accessIndices);
                }
            })
            .Case([&](v::MMAStoreSyncOp storeOp) {
                v::MMAStoreSyncOp::Adaptor storeAdaptor{ storeOp };
                rewriter.setInsertionPoint(storeOp);
                const v::MMAShape mmaShapeType{ static_cast<v::MMAShape>(storeOp.mmaShapeType()) };
                if (isActiveBlockCache)
                {
                    auto baseArrayPosition = GetBaseArrayPosition(rewriter, loc, storeOp);
                    auto newStoreOp = CreateMMAStore(rewriter, loc, storeAdaptor.src(), toValue, mmaShapeType, baseArrayPosition);
                    TransferOrSetAccessAttrs(storeOp, newStoreOp);
                    rewriter.eraseOp(storeOp);
                }
                else
                {
                    llvm::SmallVector<mlir::Value, 4> accessIndices = ResolveParentRelevantScheduleIndices(storeOp, toValueAccessContext.fullRelevantScheduleIndices);
                    rewriter.replaceOpWithNewOp<v::MMAStoreSyncOp>(storeOp, storeAdaptor.src(), toValueAccessContext.value, mmaShapeType, cacheMap, accessIndices);
                }
            })
            .Case([&](ActiveElementCacheCopyOp cacheCopyOp) {
                rewriter.setInsertionPoint(cacheCopyOp);
                ActiveElementCacheCopyOp::Adaptor copyAdaptor{ cacheCopyOp };
                auto cacheRegionIndexRanges = util::ArrayAttrToVector<IndexRange, IndexRangeAttr>(copyAdaptor.cacheRegionRelevantIndexRanges(),
                                                                                                  [](const IndexRangeAttr& indexRangeAttr) {
                                                                                                      return indexRangeAttr.getValue();
                                                                                                  });

                auto cacheRegionBaseIndices = util::ArrayAttrToVector<std::vector<Index>, mlir::ArrayAttr>(
                    copyAdaptor.cacheRegionBaseIndices(),
                    util::ConvertArrayAttrToIndexVector);
                if (cacheCopyOp.src() == fromValue)
                {
                    rewriter.replaceOpWithNewOp<ActiveElementCacheCopyOp>(cacheCopyOp,
                                                                          toValue,
                                                                          cacheCopyOp.dst(),
                                                                          copyAdaptor.externalRelevantIndices(), // the cache region is still the same so the external relevant indices are still the same
                                                                          cacheRegionIndexRanges, // cache region is still the same
                                                                          cacheRegionBaseIndices,
                                                                          cacheMap, // source mapping needs to be updated to match the updated source value
                                                                          cacheCopyOp.relevantIndicesToDstMap());
                }
                else if (cacheCopyOp.dst() == fromValue)
                {
                    rewriter.replaceOpWithNewOp<ActiveElementCacheCopyOp>(cacheCopyOp,
                                                                          cacheCopyOp.src(),
                                                                          toValue,
                                                                          copyAdaptor.externalRelevantIndices(), // the cache region is still the same so the external relevant indices are still the same
                                                                          cacheRegionIndexRanges, // cache region is still the same
                                                                          cacheRegionBaseIndices,
                                                                          cacheCopyOp.relevantIndicesToSrcMap(),
                                                                          cacheMap); // source mapping needs to be updated to match the updated source value
                }
                else
                {
                    assert(false && "Cache mapping target found in ActiveElementCacheCopyOp but not as the source or destination");
                }
            })
            .Case([&](ActiveElementCacheReduceOp cacheReduceOp) {
                rewriter.setInsertionPoint(cacheReduceOp);
                ActiveElementCacheReduceOp::Adaptor reduceAdaptor{ cacheReduceOp };
                auto cacheRegionIndexRanges = util::ArrayAttrToVector<IndexRange, IndexRangeAttr>(
                    reduceAdaptor.cacheRegionRelevantIndexRanges(),
                    [](const IndexRangeAttr& indexRangeAttr) {
                        return indexRangeAttr.getValue();
                    });

                auto cacheRegionBaseIndices = util::ArrayAttrToVector<std::vector<Index>, mlir::ArrayAttr>(
                    reduceAdaptor.cacheRegionBaseIndices(),
                    util::ConvertArrayAttrToIndexVector);

                if (cacheReduceOp.srcCache() == fromValue)
                {
                    rewriter.replaceOpWithNewOp<ActiveElementCacheReduceOp>(cacheReduceOp,
                                                                            toValue,
                                                                            cacheReduceOp.dst(),
                                                                            cacheReduceOp.externalRelevantIndices(),
                                                                            cacheRegionIndexRanges,
                                                                            cacheRegionBaseIndices,
                                                                            cacheMap,
                                                                            cacheReduceOp.relevantIndicesToDstMap(),
                                                                            reduceAdaptor.scaleValues());
                }
                else if (cacheReduceOp.dst() == fromValue)
                {
                    rewriter.replaceOpWithNewOp<ActiveElementCacheReduceOp>(cacheReduceOp,
                                                                            cacheReduceOp.srcCache(),
                                                                            toValue,
                                                                            cacheReduceOp.externalRelevantIndices(),
                                                                            cacheRegionIndexRanges,
                                                                            cacheRegionBaseIndices,
                                                                            cacheReduceOp.relevantIndicesToSrcCacheMap(),
                                                                            cacheMap,
                                                                            reduceAdaptor.scaleValues());
                }
                else
                {
                    assert(false && "Cache mapping target found in ActiveElementCacheReduceOp but not as the source or destination");
                }
            })
            .Case([&](BeginCacheMappingOp innerCacheMappingOp) {
                rewriter.setInsertionPoint(innerCacheMappingOp);
                if (innerCacheMappingOp.fromValue() == fromValue)
                {
                    rewriter.replaceOpWithNewOp<BeginCacheMappingOp>(innerCacheMappingOp,
                                                                     toValue,
                                                                     innerCacheMappingOp.baseCacheValue(),
                                                                     innerCacheMappingOp.baseInput(),
                                                                     innerCacheMappingOp.getToValueAccessContext(),
                                                                     innerCacheMappingOp.idAttr().getInt(),
                                                                     innerCacheMappingOp.activeBlockCache());
                }
                else if (innerCacheMappingOp.toValue() == fromValue)
                {
                    rewriter.replaceOpWithNewOp<BeginCacheMappingOp>(innerCacheMappingOp,
                                                                     innerCacheMappingOp.fromValue(),
                                                                     innerCacheMappingOp.baseCacheValue(),
                                                                     innerCacheMappingOp.baseInput(),
                                                                     toValueAccessContext,
                                                                     innerCacheMappingOp.idAttr().getInt(),
                                                                     innerCacheMappingOp.activeBlockCache());
                }
            })
            .Case([&](MultiCacheCopyOp cacheCopyOp) {
                // Nothing to do here, replacing the base cache with the mapping toValue cache is all that is needed
            })
            .Case([&](ActiveBlockCacheCopyOp cacheCopyOp) {
                // Nothing to do here, replacing the base cache with the mapping toValue cache is all that is needed
            })
            .Case([&](ActiveBlockCacheReduceOp cacheReduceOp) {
                // Nothing to do here, replacing the base cache with the mapping toValue cache is all that is needed
            })
            .Case([&](CacheZeroOp cacheZeroOp) {
                // Nothing to do here, replacing the base cache with the mapping toValue cache is all that is needed
            })
            .Default([&](Operation* defaultOp) {
                assert(false && "Usage of mapped op found that doesn't have an op conversion registered!");
                return rewriter.notifyMatchFailure(defaultOp, "Usage of mapped op found that doesn't have an op conversion registered!");
            });

        fromValueReplacementOps.erase(replacementOpsIter);
    }

    rewriter.eraseOp(endOp);
    rewriter.eraseOp(beginCacheMappingOp);
    return success();
}

bool IsValueUsedToAccessArray(mlir::Value array, mlir::Value value)
{
    for (auto& use : value.getUses())
    {
        auto op = use.getOwner();
        if (std::any_of(op->operand_begin(), op->operand_end(), [&](mlir::Value operand) {
                return operand == array;
            }))
        {
            return true;
        }
    }
    return false;
}

mlir::AffineForOp ComputeHoistingDestinationLoop(mlir::Value arrayToBeCached, mlir::AffineForOp parentLoop)
{
    mlir::AffineForOp result;
    while (parentLoop != nullptr)
    {
        auto loopIV = parentLoop.getInductionVar();
        if (IsValueUsedToAccessArray(arrayToBeCached, loopIV))
        {
            break;
        }
        else
        {
            result = parentLoop;
            auto parentOp = parentLoop->getParentOp();
            if (auto affineForOp = mlir::dyn_cast<mlir::AffineForOp>(parentOp))
            {
                parentLoop = affineForOp;
            }
            else
            {
                break;
            }
        }
    }
    return result;
}

std::pair<mlir::Block*, std::vector<mlir::AffineForOp>> GetCacheTriggerLevelBlockAndActiveLevelLoops(BeginCacheRegionOp beginCacheRegionOp)
{
    auto parentBlock = beginCacheRegionOp->getBlock();
    auto endOp = mlir::dyn_cast<EndCacheRegionOp>(beginCacheRegionOp.getEndOp());
    auto cacheLevelIndex = beginCacheRegionOp.cacheIndex().getValue();
    auto triggerLevelIndex = beginCacheRegionOp.triggerIndex().getValue();

    mlir::Block* triggerLevelBlock = nullptr; // Keep track of the trigger level blocks since all trigger level loops need to be in the same block
    std::vector<mlir::AffineForOp> triggerLevelLoops;
    std::vector<mlir::AffineForOp> cacheLevelLoops;
    parentBlock->walk(mlir::Block::iterator(beginCacheRegionOp), mlir::Block::iterator(endOp), [&](mlir::AffineForOp loop) {
        if (auto indexAttr = loop->getAttrOfType<IndexAttr>("index"))
        {
            if (indexAttr.getValue() == cacheLevelIndex)
            {
                cacheLevelLoops.push_back(loop);
            }
            if (indexAttr.getValue() == triggerLevelIndex)
            {
                triggerLevelLoops.push_back(loop);
                auto currentBlock = loop->getBlock();
                if (triggerLevelBlock == nullptr)
                {
                    triggerLevelBlock = currentBlock;
                }
                else
                {
                    assert(triggerLevelBlock == currentBlock && "All trigger level loops must be in the same block");
                }
            }
        }
    });

    return std::make_pair(triggerLevelBlock, cacheLevelLoops);
}

LogicalResult HoistCacheRegionOpsRewrite::matchAndRewrite(BeginCacheRegionOp beginCacheRegionOp, PatternRewriter& rewriter) const
{
    // Hoist cache regions out past loops that don't affect the cache region

    // Hoist the outermost cache region ops before the innermost ones to preserve
    // the relative cache region order

    if (beginCacheRegionOp->hasAttrOfType<mlir::UnitAttr>("hoisted"))
    {
        return success();
    }
    if (!util::IsOutermostOpOfType(beginCacheRegionOp, "hoisted"))
    {
        return failure();
    }

    [[maybe_unused]] auto loc = beginCacheRegionOp.getLoc();

    [[maybe_unused]] auto sourceBlock = beginCacheRegionOp.getOperation()->getBlock();

    EndCacheRegionOp endOp = mlir::dyn_cast<EndCacheRegionOp>(beginCacheRegionOp.getEndOp());

    [[maybe_unused]] mlir::Block* destBlock = beginCacheRegionOp.getOperation()->getBlock();
    mlir::Block::iterator beginIter(beginCacheRegionOp);
    mlir::Block::iterator endIter(endOp);
    auto parentOp = beginCacheRegionOp->getParentOp();
    auto baseArray = beginCacheRegionOp.baseInput();

    // Compute the trigger level hoisting destination
    mlir::AffineForOp newTriggerLoopLevel;
    if (auto parentLoop = mlir::dyn_cast<mlir::AffineForOp>(parentOp))
    {
        newTriggerLoopLevel = ComputeHoistingDestinationLoop(baseArray, parentLoop);
    }

    BeginCacheRegionOp hoistedBeginCacheRegionOp = beginCacheRegionOp;
    if (newTriggerLoopLevel)
    {
        mlir::BlockAndValueMapping mapping;
        rewriter.setInsertionPoint(newTriggerLoopLevel);
        auto clonedBeginOp = rewriter.clone(*beginCacheRegionOp.getOperation(), mapping);
        hoistedBeginCacheRegionOp = mlir::dyn_cast<BeginCacheRegionOp>(clonedBeginOp);
        auto triggerLoopIndexAttr = newTriggerLoopLevel->getAttrOfType<IndexAttr>("index");
        hoistedBeginCacheRegionOp.triggerIndexAttr(triggerLoopIndexAttr);

        rewriter.setInsertionPointAfter(newTriggerLoopLevel);
        [[maybe_unused]] auto clonedEndOp = rewriter.clone(*endOp.getOperation(), mapping);

        rewriter.eraseOp(endOp);
        rewriter.eraseOp(beginCacheRegionOp);
    }
    hoistedBeginCacheRegionOp->setAttr("hoisted", rewriter.getUnitAttr());

    // Compute the active level hoisting destination
    auto [cacheTriggerLevelBlock, cacheActiveLevelLoops] = GetCacheTriggerLevelBlockAndActiveLevelLoops(hoistedBeginCacheRegionOp);

    std::vector<Index> newActiveLevelIndices;
    for (auto& cacheActiveLevelLoop : cacheActiveLevelLoops)
    {
        if (auto parentLoop = mlir::dyn_cast<mlir::AffineForOp>(cacheActiveLevelLoop->getParentOp()))
        {
            mlir::AffineForOp newActiveLevelLoop = ComputeHoistingDestinationLoop(baseArray, parentLoop);
            if (newActiveLevelLoop)
            {
                assert(newActiveLevelLoop->hasAttrOfType<IndexAttr>("index"));
                auto indexAttr = newActiveLevelLoop->getAttrOfType<IndexAttr>("index");
                newActiveLevelIndices.push_back(indexAttr.getValue());
            }
        }
    }
    // We should have either determined that there is no new active level loop, in which case newActiveLevelIndices should
    // be empty, or we should have determined that all of the active level loops should move to the same new index
    assert(newActiveLevelIndices.empty() || newActiveLevelIndices.size() == cacheActiveLevelLoops.size());
    if (!newActiveLevelIndices.empty())
    {
        auto newActiveLevelIndex = newActiveLevelIndices.front();
        assert(std::all_of(newActiveLevelIndices.begin(), newActiveLevelIndices.end(), [&](Index index) {
            return index == newActiveLevelIndex;
        }));
        hoistedBeginCacheRegionOp.cacheIndexAttr(IndexAttr::get(newActiveLevelIndex, rewriter.getContext()));
    }

    return success();
}

mlir::AffineForOp GetOutermostAffineForOp(BeginCacheRegion beginCacheRegionOp)
{
    auto endOp = beginCacheRegionOp.getEndOp();

    mlir::Block::iterator endIter(endOp);
    for (auto iter = mlir::Block::iterator(beginCacheRegionOp); iter != endIter; iter++)
    {
        if (auto forOp = mlir::dyn_cast<mlir::AffineForOp>(*iter))
        {
            return forOp;
        }
    }
    return nullptr;
}

LogicalResult MergeCacheRegionOpsRewrite::matchAndRewrite(BeginCacheRegionOp beginCacheRegionOp, PatternRewriter& rewriter) const
{
    // Merge cache regions that are in the same block and have identical operands

    [[maybe_unused]] auto loc = beginCacheRegionOp.getLoc();

    auto block = beginCacheRegionOp.getOperation()->getBlock();

    EndCacheRegionOp endOp = mlir::dyn_cast<EndCacheRegionOp>(beginCacheRegionOp.getEndOp());

    // Examine ops in the block, and for each BeginCacheRegionOp that has the same operands as this one,
    // merge the cache regions if the outermost loops they wrap don't affect the buffer region being examined.
    // This means that boundary condition loops that produce differently shaped caches stay as separate regions,
    // while boundary condition loops on unrelated indices that don't affect the shape of the cache get merged
    // into a single region

    mlir::Operation* firstBeginOp = nullptr;
    mlir::Operation* lastEndOp = nullptr;
    std::vector<mlir::Operation*> beginOpsForRemoval;
    std::vector<mlir::Operation*> endOpsForRemoval;

    auto baseArray = beginCacheRegionOp.baseInput();
    // If the outermost loop in this cache region is used to index into the base array then the cache regions cannot be merged
    // as they will have different access patterns

    auto baseRegionOutermostAffineForOp = GetOutermostAffineForOp(beginCacheRegionOp);

    for (auto& op : block->getOperations())
    {
        if (auto otherBeginCacheRegionOp = mlir::dyn_cast<BeginCacheRegionOp>(&op))
        {
            if (util::OperationsAreEqual(beginCacheRegionOp, otherBeginCacheRegionOp))
            {
                assert(baseArray == otherBeginCacheRegionOp.baseInput());
                // Check whether the outermost loop IV is used to access the array
                auto otherOutermostAffineForOp = GetOutermostAffineForOp(otherBeginCacheRegionOp);
                if (baseRegionOutermostAffineForOp != otherOutermostAffineForOp)
                {
                    continue;
                }

                auto otherEndOp = otherBeginCacheRegionOp.getEndOp();
                if (firstBeginOp == nullptr)
                {
                    firstBeginOp = otherBeginCacheRegionOp;
                }
                if (&op != beginCacheRegionOp)
                {
                    // If this isn't the begin op we're examining in this pass, but matches it,
                    // then queue up this begin op and its associated end op for removal
                    beginOpsForRemoval.push_back(otherBeginCacheRegionOp);
                    endOpsForRemoval.push_back(otherEndOp);
                }
                lastEndOp = otherEndOp;
            }
        }
    }
    if (beginCacheRegionOp != firstBeginOp)
    {
        beginCacheRegionOp->moveBefore(firstBeginOp);
    }
    if (endOp != lastEndOp)
    {
        endOp->moveAfter(lastEndOp);
    }
    for (auto op : endOpsForRemoval)
    {
        rewriter.eraseOp(op);
    }
    for (auto op : beginOpsForRemoval)
    {
        rewriter.eraseOp(op);
    }

    if (IsCacheRegionEmpty(beginCacheRegionOp) || GetOutermostAffineForOp(beginCacheRegionOp) == nullptr)
    {
        rewriter.eraseOp(endOp);
        rewriter.eraseOp(beginCacheRegionOp);
        return success();
    }

    return success();
}

MakeCacheOp CreateDoubleBufferTempArray(mlir::OpBuilder& builder,
                                        MultiCacheInfo& info,
                                        BeginCacheRegionOp& cacheRegionOp)
{
    mlir::Value cache = info.multiCache.cache();

    auto cacheType = cache.getType();
    assert(cacheType.isa<mlir::MemRefType>());
    auto cacheMemRefType = cacheType.cast<mlir::MemRefType>();

    auto multiCacheShape = info.multiCacheIterationCounts;
    auto fullCacheShape = cacheMemRefType.getShape().vec();
    std::vector<int64_t> activeBlockCacheShape(fullCacheShape.begin() + multiCacheShape.size(), fullCacheShape.end());

    [[maybe_unused]] auto sharedMemSpaceAttr = util::MemorySpaceToAttribute(v::MemorySpace::Shared, builder.getContext());
    auto privateMemSpaceAttr = util::MemorySpaceToAttribute(v::MemorySpace::Private, builder.getContext());
    auto cacheMemSpaceAttr = cacheMemRefType.getMemorySpace();
    auto tempArrayMemSpaceAttrOpt = cacheRegionOp.doubleBufferMemorySpace();
    assert(tempArrayMemSpaceAttrOpt.hasValue() && "Can't create a double buffer cache without a double buffer memory space set");
    auto tempArrayMemSpaceAttr = util::MemorySpaceToAttribute(tempArrayMemSpaceAttrOpt.getValue(), builder.getContext());

    auto cacheAccessMap = info.multiCache.offsetArrayToCacheAccessMap();
    auto multiCacheAccessIndices = util::ConvertArrayAttrToIndexVector(info.multiCache.multiCacheAccessIndices());
    auto cacheOffsetIndices = util::ConvertArrayAttrToIndexVector(info.multiCache.offsetAccessIndices());

    auto tempArrayOffsetIndices = cacheOffsetIndices;
    auto tempArrayMultiCacheAccessIndices = multiCacheAccessIndices;
    auto tempArrayAccessMap = cacheAccessMap;

    std::optional<VectorizationInfo> vecInfo;
    auto vecInfoLLVMOpt = cacheRegionOp.vectorizationInfo();
    if (vecInfoLLVMOpt.hasValue())
    {
        vecInfo = vecInfoLLVMOpt.getValue().getValue();
    }

    auto elementBitWidth = cacheMemRefType.getElementTypeBitWidth();
    auto elementByteWidth = elementBitWidth / 8;

    int64_t vectorSizePerThread = 1;
    if (vecInfo.has_value() && vecInfo->vectorBytes > 0)
    {
        vectorSizePerThread = vecInfo->vectorBytes / elementByteWidth;
    }

    size_t arrayRank = cacheAccessMap.getNumDims() - cacheOffsetIndices.size() - multiCacheAccessIndices.size();

    std::vector<int64_t> tempArrayActiveBlockShape = activeBlockCacheShape;
    std::vector<mlir::AffineMap> tempMemrefMaps = cacheMemRefType.getAffineMaps();

    std::optional<v::ExecutionTarget> execTargetOpt = util::ResolveExecutionTarget(cacheRegionOp);
    auto execTarget = *execTargetOpt;
    auto parentLambda = cacheRegionOp->getParentOfType<v::ValueLambdaOp>();

    if (execTarget == v::ExecutionTarget::GPU)
    {
        // If this is for a GPU target, then our temp array memory space should be set to PRIVATE

        if (cacheMemSpaceAttr != privateMemSpaceAttr && tempArrayMemSpaceAttr == privateMemSpaceAttr)
        {
            // If the double buffer temp array is in the PRIVATE memory space and the cache is not
            // then the different threads will each load a different segment of the cache into their
            // double-buffering temp buffer. To support this, the private memory temp array needs to
            // be shrunk to just hold a single thread's contribution to the cache

            auto launchAttr = parentLambda->getAttrOfType<mlir::ArrayAttr>(parentLambda.getGPULaunchAttrName());
            assert(launchAttr != nullptr);
            auto gpuParams = accera::ir::targets::GPU::FromArrayAttr(launchAttr);
            std::vector<int64_t> blockDimSizes = { gpuParams.block.x, gpuParams.block.y, gpuParams.block.z };

            int64_t activeBlockVolume = std::accumulate(activeBlockCacheShape.begin(), activeBlockCacheShape.end(), 1, std::multiplies<int64_t>());
            int64_t totalWorkPerThread = activeBlockVolume / (blockDimSizes[0] * blockDimSizes[1] * blockDimSizes[2]);
            vectorSizePerThread = std::min(vectorSizePerThread, totalWorkPerThread);
            auto loadsPerThread = activeBlockVolume / (blockDimSizes[0] * blockDimSizes[1] * blockDimSizes[2] * vectorSizePerThread);
            loadsPerThread = std::max((int64_t)1, (int64_t)loadsPerThread);
            tempArrayActiveBlockShape = { loadsPerThread, vectorSizePerThread };

            std::vector<Index> tempArrayIndexPlaceholders;
            tempArrayIndexPlaceholders.emplace_back(ActionsPerThreadIndexName, Index::DefaultID);
            tempArrayIndexPlaceholders.emplace_back(ThreadVectorizationIndexName, Index::DefaultID);
            tempArrayOffsetIndices = tempArrayIndexPlaceholders;

            // Need to create the temp array access expressions such that the ActionsPerThreadIndex and ThreadVectorizationIndex indices
            // are used to index into the array and everything else is ignored.
            // To do this, set the cacheOffsetIndices to be the placeholders (which will need to get fully resolved later once the loopnest is created)
            // and construct the map to only pay attention to those inputs

            [[maybe_unused]] size_t multiCacheIndexPos = 0;
            [[maybe_unused]] size_t cacheOffsetIndexPos = multiCacheAccessIndices.size();
            [[maybe_unused]] size_t multiCacheDimAndPrivateThreadDimCount = multiCacheAccessIndices.size() + 2; // + 2 because there is one ActionPerThread dim and one ThreadVectorization dim
            size_t totalDimCount = multiCacheDimAndPrivateThreadDimCount + arrayRank;

            // Map { multiCacheIndices..., ActionPerThread idx, ThreadVectorization idx, arrayRank global indices... } to { multiCacheIndices..., ActionPerThread idx, ThreadVectorization idx }
            tempArrayAccessMap = util::GetMajorIdentityMap(totalDimCount, multiCacheDimAndPrivateThreadDimCount, builder.getContext());
        }
    }

    auto fullTempArrayShape = tempArrayActiveBlockShape;
    fullTempArrayShape.insert(fullTempArrayShape.begin(), multiCacheShape.begin(), multiCacheShape.end());
    mlir::MemRefType tempArrayType = mlir::MemRefType::get(fullTempArrayShape, cacheMemRefType.getElementType(), tempMemrefMaps, tempArrayMemSpaceAttr);

    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(&parentLambda.body().front());
    auto memorySpaceEnum = util::AttributeToMemorySpace(tempArrayMemSpaceAttr);
    return builder.create<MakeCacheOp>(parentLambda.getLoc(),
                                       tempArrayType,
                                       memorySpaceEnum,
                                       tempArrayAccessMap,
                                       tempArrayOffsetIndices,
                                       tempArrayMultiCacheAccessIndices);
}

void CreateCacheMappingRegionHelper(mlir::PatternRewriter& rewriter,
                                    BeginCacheRegionOp& beginCacheRegionOp,
                                    MultiCacheInfo& multiCacheInfo)
{
    for (auto activeCacheLoopEntry : multiCacheInfo.activeBlockRegionInfos)
    {
        auto cacheLevelLoopOp = activeCacheLoopEntry.first;
        auto cacheLevelLoop = mlir::cast<mlir::AffineForOp>(cacheLevelLoopOp);
        auto& currentActiveBlockRegionInfo = activeCacheLoopEntry.second;

        mlir::Block* cacheLevelBlock = cacheLevelLoop.getOperation()->getBlock();
        mlir::Block::iterator cacheLevelStartOp(cacheLevelLoop);
        mlir::Block::iterator cacheLevelEndOp(cacheLevelLoop);
        cacheLevelEndOp++;

        rewriter.setInsertionPoint(cacheLevelBlock, cacheLevelStartOp);

        // TODO : refactor out CacheAccessContext and simplify this
        currentActiveBlockRegionInfo.cacheAccessContext.externalRelevantScheduleIndices = currentActiveBlockRegionInfo.allCacheExternalSymbols;
        BeginCacheMappingOp cacheMappingOp = rewriter.create<BeginCacheMappingOp>(beginCacheRegionOp.getLoc(),
                                                                                  beginCacheRegionOp.input(),
                                                                                  multiCacheInfo.originalCacheOp,
                                                                                  beginCacheRegionOp.baseInput(),
                                                                                  currentActiveBlockRegionInfo.cacheAccessContext,
                                                                                  beginCacheRegionOp.id(),
                                                                                  beginCacheRegionOp.activeBlockCache());

        rewriter.setInsertionPoint(cacheLevelBlock, cacheLevelEndOp);
        [[maybe_unused]] EndCacheMappingOp endCacheMappingOp = rewriter.create<EndCacheMappingOp>(beginCacheRegionOp.getLoc(), cacheMappingOp.getResult());
    }
}

LogicalResult BeginCacheRegionOpRewrite::matchAndRewrite(BeginCacheRegionOp beginCacheRegionOp, PatternRewriter& rewriter) const
{
    // CacheRegionOp examines the uses of the input value within its region and determines which cache data movements ops are necessary to support that usage
    // Then lowers to those data movement ops, a CacheMappingOp, and some dim size ops

    // The possible prologue cache movements ops are CacheZeroOp and CacheCopyOp
    // The possible epilogue cache movements ops are CacheCopyOp, CacheReduceOp, or not op whatsoever

    // Initially all of the op combinations are:
    // { CacheZeroOp, CacheCopyOp } x { CacheCopyOp, CacheReduceOp, None }
    // Expands to:
    // ( CacheZeroOp, CacheCopyOp )     - Valid scenario
    // ( CacheZeroOp, CacheReduceOp )   - Valid scenario
    // ( CacheZeroOp, None )            - Invalid scenario (any input values are ignored and any data that is written is never stored)
    // ( CacheCopyOp, CacheCopyOp )     - Valid scenario
    // ( CacheCopyOp, CacheReduceOp )   - Invalid scenario (will the accumulation in the CacheReduceOp will lead to duplicated adding of input/accumulated values a number of times)
    // ( CacheCopyOp, None )            - Valid scenario

    // How to determine which valid op combination to use:
    // ( CacheZeroOp, CacheCopyOp )     - If the input value is never read from, but only written to, then use a CacheZeroOp prologue and a CacheCopyOp epilogue
    // ( CacheZeroOp, CacheReduceOp )   - If the input value is read from, but only for the purpose of an accumulation, then use a CacheZeroOp prologue and a CacheReduceOp epilogue
    // ( CacheCopyOp, None )            - If the input value is read from, but never written to, then use a CacheCopyOp prologue and no epilogue op
    // ( CacheCopyOp, CacheCopyOp )     - If we aren't in any of the other scenarios, then default to a CacheCopyOp prologue and a CacheCopyOp epilogue

    EndCacheRegionOp endOp = mlir::dyn_cast<EndCacheRegionOp>(beginCacheRegionOp.getEndOp());

    auto loc = beginCacheRegionOp.getLoc();

    [[maybe_unused]] auto parentBlock = beginCacheRegionOp.getOperation()->getBlock();

    // If the region is empty, just erase this op and move on
    if (IsCacheRegionEmpty(beginCacheRegionOp) || GetOutermostAffineForOp(beginCacheRegionOp) == nullptr)
    {
        rewriter.eraseOp(endOp);
        rewriter.eraseOp(beginCacheRegionOp);
        return success();
    }

    // Lower BeginCacheRegionOps outermost-to-innermost to ensure nested mappings are created in the same
    // nested order as the cache region ops

    if (!util::IsOutermostOpOfType(beginCacheRegionOp))
    {
        return failure();
    }

    mlir::Value baseInput = beginCacheRegionOp.baseInput();

    auto cacheAccessContext = beginCacheRegionOp.getCacheAccessContext();

    // Determine the level of the loopnest that defines the cache active block.
    // If the caching trigger index is equal to the caching index, then that level is
    // the same as where this BeginCacheRegionOp is located, however if the trigger
    // and cache indices are different, then the cache active block will be defined
    // at a deeper level in the loopnest

    // The BeginCacheRegionOp should precede the loop op that is the trigger level loop
    // and in non-multiCache scenarios this is also the cache level
    // In multiCache scenarios, we need to walk deeper into the loopnest to find the cache level loop

    auto [triggerLevelBlock, cacheLevelLoops] = GetCacheTriggerLevelBlockAndActiveLevelLoops(beginCacheRegionOp);
    assert(!cacheLevelLoops.empty() && "Couldn't find cache level loop(s)");
    assert(triggerLevelBlock != nullptr && "Couldn't find trigger level block");

    mlir::Operation* triggerLevelParentOp = triggerLevelBlock->getParentOp();

    // Some cache active blocks can share the same multicache buffer, whereas cache active blocks
    // that have different shapes due to boundary conditions or fusion need separate multicache buffers
    // (as long as they are read-only or accumulate-only cache buffers)

    // Each MultiCacheInfo represents a single multiCache buffer with possibly many cache active block
    // subgraphs that it supports
    std::vector<MultiCacheInfo> multiCacheInfos;
    for (auto& cacheLevelLoop : cacheLevelLoops)
    {
        MultiCacheInfo currentMultiCacheInfo(loc);
        currentMultiCacheInfo.originalCacheOp = mlir::dyn_cast<MakeCacheOp>(beginCacheRegionOp.cache().getDefiningOp());
        currentMultiCacheInfo.activeBlockRegionInfos.emplace(cacheLevelLoop, MultiCacheInfo::ActiveBlockRegionInfo{});
        auto& currentActiveBlockRegionInfo = currentMultiCacheInfo.activeBlockRegionInfos[cacheLevelLoop];
        currentActiveBlockRegionInfo.cacheAccessContext = cacheAccessContext;

        mlir::Block::iterator cacheLevelStartOp(cacheLevelLoop);
        mlir::Block::iterator cacheLevelEndOp(cacheLevelLoop);
        cacheLevelEndOp++;

        auto arrayAccessInfo = ComputeAccessInfoForArrayAtLevel(rewriter, baseInput, cacheLevelStartOp, cacheLevelEndOp, beginCacheRegionOp.activeBlockCache());
        if (!arrayAccessInfo.cacheUsedInRegion)
        {
            // The cache isn't used inside this cacheLevelLoop, so don't bother computing anything else
            continue;
        }
        currentMultiCacheInfo.arrayAccessInfo = arrayAccessInfo;

        if (beginCacheRegionOp.activeBlockCache())
        {
            // Convert the active block region into a series of ranges that can be used to construct cache copy and reduce nests
            // Since Accera deals in statically sizd loopnests, the MemRefRegion::getConstantBoundingSizeAndShape utility will
            // give a constant bounding size and shape

            currentMultiCacheInfo.activeBlockInfo = ConvertMemRefRegionToActiveBlockInfo(rewriter, arrayAccessInfo.activeBlock);
            currentMultiCacheInfo.multiCacheExternalSymbols = currentMultiCacheInfo.activeBlockInfo.externalSymbols;

            // Find the AffineForOps that are external to the active block but inside of the multiCache level (i.e. between this BeginCacheRegionOp and its end op)
            std::vector<mlir::AffineForOp> multiCacheLoops;

            // walk from the cacheLevelLoop upwards so that if there are unswitched boundary cases or fusion subnests that create
            // multiple different cacheLevelLoop's and thus active blocks we get the specific one we're currently examining
            mlir::Operation* currentOp = cacheLevelLoop->getParentOp();
            while (currentOp != triggerLevelParentOp)
            {
                if (auto currentLoop = mlir::dyn_cast<mlir::AffineForOp>(currentOp))
                {
                    auto iv = currentLoop.getInductionVar();
                    if (std::find(currentMultiCacheInfo.activeBlockInfo.externalSymbols.begin(), currentMultiCacheInfo.activeBlockInfo.externalSymbols.end(), iv) != currentMultiCacheInfo.activeBlockInfo.externalSymbols.end())
                    {
                        multiCacheLoops.push_back(currentLoop);

                        // This symbol isn't external to the multiCache layer
                        auto multiCacheExternalSymbolsPos = std::find(currentMultiCacheInfo.multiCacheExternalSymbols.begin(), currentMultiCacheInfo.multiCacheExternalSymbols.end(), iv);
                        assert(multiCacheExternalSymbolsPos != currentMultiCacheInfo.multiCacheExternalSymbols.end());
                        currentMultiCacheInfo.multiCacheExternalSymbols.erase(multiCacheExternalSymbolsPos);
                    }
                }
                else
                {
                    assert(false && "Must be in nested mlir::AffineForOps");
                }
                currentOp = currentOp->getParentOp();
            }

            // Since we walked from the inner loops to the outer loops, our multiCacheLoops need to be reversed to get the shape starting from the outer loops going inwards
            std::reverse(multiCacheLoops.begin(), multiCacheLoops.end());
            currentMultiCacheInfo.multiCacheLoops = multiCacheLoops;

            // Collect the loop shapes that the MultiCacheCopyOp needs to create
            // Since Accera loopnests operate with fixed-size loops, we will assume that the multiCache loops
            // all have constant lower and upper bounds

            // also compute the multiCache buffer shape based on the iteration counts of the multiCache slicing loops
            std::vector<uint64_t> multiCacheIterationCounts;
            for (auto& multiCacheLoop : multiCacheLoops)
            {
                assert(multiCacheLoop.hasConstantBounds() && "AffineForOps for caching must have constant bounds");
                auto lowerBound = multiCacheLoop.getConstantLowerBound();
                auto upperBound = multiCacheLoop.getConstantUpperBound();
                auto step = multiCacheLoop.getStep();

                currentMultiCacheInfo.multiCacheLBMaps.push_back(mlir::AffineMap::get(0, 0, rewriter.getAffineConstantExpr(lowerBound)));
                currentMultiCacheInfo.multiCacheUBMaps.push_back(mlir::AffineMap::get(0, 0, rewriter.getAffineConstantExpr(upperBound)));
                currentMultiCacheInfo.multiCacheStepSizes.push_back(step);
                assert(multiCacheLoop->hasAttrOfType<IndexAttr>("index"));
                currentMultiCacheInfo.multiCacheLoopIndexIds.push_back(multiCacheLoop->getAttrOfType<IndexAttr>("index").getValue());

                // Create an mlir::Value computing the iteration count for this loop so we can use this to index into the multiCache.
                // We need to compute (iv - lowerBound) / step
                // Construct a map that maps (lower_bound_operands..., iv) -> (iv - lowerBoundMap(lower_bound_operands)) / step
                // By extending the number of inputs the lower bound map takes and outputs it produces to pass through the iv:
                // (lower_bound_operands..., iv) -> (lowerBoundMap(lower_bound_operands), iv)
                // then composing it with the offset-and-divide map (d0, d1) -> (d1 - d0) / step
                mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPointToStart(multiCacheLoop.getBody());

                [[maybe_unused]] mlir::Value iterCounter = util::CreateConstantRangeForOpIterationCounter(rewriter, loc, multiCacheLoop);

                auto constantTripCountOpt = mlir::getConstantTripCount(multiCacheLoop);
                assert(constantTripCountOpt.hasValue() && "AffineForOps in Accera loop nests must have constant trip counts");
                currentMultiCacheInfo.multiCacheIterationCounts.push_back(constantTripCountOpt.getValue());
            }

            // The MultiCacheCopyOp will reconstruct the loops that are external to the active block but internal to the multiCache region
            // and feed those induction variables into an ActiveBlockCacheCopyOp
            // To make this work, we construct the MultiCacheCopyOp with the sizes of loops that it needs to build
            // and have it feed the external symbols and the loop IVs it creates into a single map that will re-arrange them into the order that
            // ActiveBlockCacheCopyOp expects them in.
            // E.g. if we have (external_iv_0, multiCache_iv_0, external_iv_1) for ActiveBlockCacheCopy then we need to construct a map
            //      such that (external_ivs..., multiCache_ivs...) is re-arranged to this sequence, which would be the map (d0, d1, d2) -> (d0, d2, d1) in this example

            // the activeBlockInfo externalSymbols may not be in an easily predictable order, but we need the multiCacheCopy to be able to reconstruct that order
            // as the ActiveBlockCacheCopyOp maps will expect the external symbols in that order
            // and at this point we have the multiCacheExternalSymbols in hand, so we know how their positions should map,
            // and we need to determine how the multiCache IVs we will construct in the MultiCacheCopyOp correspond to the remaining active block external symbols
            // For example, suppose we have:
            // - activeBlockInfo.externalSymbols = [ %arg4, %arg7, %arg3, %arg6 ]
            // - multiCacheExternalSymbols = [ %arg3 ]
            // - the multiCacheIVs that will be constructed correspond to [ %arg4, %arg6, %arg7 ]
            // - so the MultiCacheCopyOp will construct the order [ %arg3, %arg4, %arg6, %arg7 ] and we need to map that to [ %arg4, %arg7, %arg3, %arg6 ]
            //      with a (d0, d1, d2, d3) -> (d1, d3, d0, d2) mapping
            // To do this, we need to determine the outermost-to-innermost order of definition of the elements of activeBlockInfo.externalSymbols that are internal to the multiCache
            //      since that is the order the MultiCacheCopyOp will construct them in and use that to construct the mapping between the
            //      construction order and the order that the maps expect them in (the order that activeBlockInfo.externalSymbols is already in)

            if (currentMultiCacheInfo.activeBlockInfo.externalSymbols.empty())
            {
                currentMultiCacheInfo.multiCacheExternalSymbolsPermutationMap = mlir::AffineMap::getMultiDimIdentityMap(0, rewriter.getContext());
            }
            else
            {
                currentMultiCacheInfo.multiCacheExternalSymbolsPermutationMap = ComputeLoopIVDefinitionOrderToCurrentOrderMap(currentMultiCacheInfo.activeBlockInfo.externalSymbols, rewriter.getContext());
            }

            // Determine if this MultiCacheInfo can be merged with an existing one
            auto matchingExistingInfoIter = std::find_if(multiCacheInfos.begin(), multiCacheInfos.end(), [&](const MultiCacheInfo& otherMultiCacheInfo) { return ShouldMergeMultiCacheInfos(currentMultiCacheInfo, otherMultiCacheInfo); });

            if (matchingExistingInfoIter != multiCacheInfos.end())
            {
                // We should merge this cache with an existing multiCache

                assert(matchingExistingInfoIter->activeBlockRegionInfos.count(cacheLevelLoop) == 0);

                // OR-in most of the access flags since if the cache is written/read/used in one of the blocks then the entire multicache is written/read/used
                matchingExistingInfoIter->arrayAccessInfo.valueWritten |= currentMultiCacheInfo.arrayAccessInfo.valueWritten;
                matchingExistingInfoIter->arrayAccessInfo.valueRead |= currentMultiCacheInfo.arrayAccessInfo.valueRead;
                matchingExistingInfoIter->arrayAccessInfo.cacheUsedInRegion |= currentMultiCacheInfo.arrayAccessInfo.cacheUsedInRegion;

                // AND-in the onlyReadsAreAccumulates flag since if this is false for one of the active block regions it is then false for the entire multicache
                matchingExistingInfoIter->arrayAccessInfo.onlyReadsAreAccumulates &= currentMultiCacheInfo.arrayAccessInfo.onlyReadsAreAccumulates;

                // Union the active block regions if the shapes differ but we still need to merge
                bool cachesHaveSameShape = matchingExistingInfoIter->activeBlockInfo.shape.size() == currentMultiCacheInfo.activeBlockInfo.shape.size() &&
                                           std::equal(matchingExistingInfoIter->activeBlockInfo.shape.begin(), matchingExistingInfoIter->activeBlockInfo.shape.end(), currentMultiCacheInfo.activeBlockInfo.shape.begin());

                if (!cachesHaveSameShape)
                {
                    auto unionResult = matchingExistingInfoIter->arrayAccessInfo.activeBlock.unionBoundingBox(currentMultiCacheInfo.arrayAccessInfo.activeBlock);
                    assert(succeeded(unionResult));
                    matchingExistingInfoIter->arrayAccessInfo.activeBlock.cst.removeRedundantConstraints();
                }

                // Append this active block region info to the multiCache info we're merging into
                matchingExistingInfoIter->activeBlockRegionInfos[cacheLevelLoop] = currentMultiCacheInfo.activeBlockRegionInfos[cacheLevelLoop];
            }
            else
            {
                multiCacheInfos.push_back(std::move(currentMultiCacheInfo));
            }
        }
        else
        {
            auto matchingExistingInfoIter = std::find_if(multiCacheInfos.begin(), multiCacheInfos.end(), [&](const MultiCacheInfo& otherMultiCacheInfo) { return ShouldMergeMultiCacheInfos(currentMultiCacheInfo, otherMultiCacheInfo); });
            if (matchingExistingInfoIter != multiCacheInfos.end())
            {
                assert(matchingExistingInfoIter->activeBlockRegionInfos.count(cacheLevelLoop) == 0);
                currentActiveBlockRegionInfo.allCacheExternalSymbols.insert(currentActiveBlockRegionInfo.allCacheExternalSymbols.end(), currentActiveBlockRegionInfo.cacheAccessContext.externalRelevantScheduleIndices.begin(), currentActiveBlockRegionInfo.cacheAccessContext.externalRelevantScheduleIndices.end());
                matchingExistingInfoIter->arrayAccessInfo.valueWritten |= currentMultiCacheInfo.arrayAccessInfo.valueWritten;
                matchingExistingInfoIter->arrayAccessInfo.valueRead |= currentMultiCacheInfo.arrayAccessInfo.valueRead;
                matchingExistingInfoIter->arrayAccessInfo.cacheUsedInRegion |= currentMultiCacheInfo.arrayAccessInfo.cacheUsedInRegion;
                matchingExistingInfoIter->arrayAccessInfo.onlyReadsAreAccumulates &= currentMultiCacheInfo.arrayAccessInfo.onlyReadsAreAccumulates;
                matchingExistingInfoIter->activeBlockRegionInfos[cacheLevelLoop] = currentMultiCacheInfo.activeBlockRegionInfos[cacheLevelLoop];
            }
            else
            {
                auto makeCacheOperation = beginCacheRegionOp.cache().getDefiningOp();
                auto makeCacheOp = mlir::dyn_cast<MakeCacheOp>(makeCacheOperation);
                currentMultiCacheInfo.multiCache = makeCacheOp;
                currentActiveBlockRegionInfo.allCacheExternalSymbols.insert(currentActiveBlockRegionInfo.allCacheExternalSymbols.end(), currentActiveBlockRegionInfo.cacheAccessContext.externalRelevantScheduleIndices.begin(), currentActiveBlockRegionInfo.cacheAccessContext.externalRelevantScheduleIndices.end());
                multiCacheInfos.push_back(std::move(currentMultiCacheInfo));
            }
        }
    }

    // Now that we've merged all the caches we may need to merge, compute the cache buffer shapes
    if (beginCacheRegionOp.activeBlockCache())
    {
        for (auto& currentMultiCacheInfo : multiCacheInfos)
        {
            auto makeCacheOperation = beginCacheRegionOp.cache().getDefiningOp();
            auto makeCacheOp = mlir::dyn_cast<MakeCacheOp>(makeCacheOperation);
            assert(makeCacheOp != nullptr);
            mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointAfter(makeCacheOp);

            auto& tempActiveBlockRegionInfo = currentMultiCacheInfo.activeBlockRegionInfos.begin()->second;

            // Recompute this after any merging we did
            currentMultiCacheInfo.activeBlockInfo = ConvertMemRefRegionToActiveBlockInfo(rewriter, currentMultiCacheInfo.arrayAccessInfo.activeBlock);

            auto shapedMakeCacheOp = UpdateActiveBlockCacheShape(rewriter,
                                                                 makeCacheOp,
                                                                 tempActiveBlockRegionInfo.cacheAccessContext,
                                                                 currentMultiCacheInfo.activeBlockInfo,
                                                                 currentMultiCacheInfo.multiCacheIterationCounts);

            currentMultiCacheInfo.activeBlockToCacheMap = CreateActiveBlockToCacheMap(rewriter, tempActiveBlockRegionInfo.cacheAccessContext);

            size_t activeBlockRank = currentMultiCacheInfo.activeBlockInfo.shape.size();
            std::vector<mlir::Value> multiCacheIVs;
            std::transform(currentMultiCacheInfo.multiCacheLoops.begin(), currentMultiCacheInfo.multiCacheLoops.end(), std::back_inserter(multiCacheIVs), [&](mlir::AffineForOp loop) {
                return loop.getInductionVar();
            });
            auto multiCacheAccessIndices = util::GetIndicesForLoopIVs(multiCacheIVs);
            auto offsetAccessIndices = util::GetIndicesForLoopIVs(currentMultiCacheInfo.activeBlockInfo.externalSymbols);
            currentMultiCacheInfo.multiCache = UpdateActiveBlockCacheAccess(rewriter,
                                                                            shapedMakeCacheOp,
                                                                            activeBlockRank,
                                                                            currentMultiCacheInfo.activeBlockInfo.activeBlockOffsetMap,
                                                                            currentMultiCacheInfo.activeBlockToCacheMap,
                                                                            offsetAccessIndices,
                                                                            multiCacheAccessIndices);

            for (auto& currentActiveBlockRegionInfoEntry : currentMultiCacheInfo.activeBlockRegionInfos)
            {
                auto& currentActiveBlockRegionInfo = currentActiveBlockRegionInfoEntry.second;
                currentActiveBlockRegionInfo.cacheAccessContext.value = currentMultiCacheInfo.multiCache;

                UpdateCacheAccessContextForActiveBlockCache(rewriter,
                                                            currentActiveBlockRegionInfo.cacheAccessContext,
                                                            currentMultiCacheInfo.activeBlockInfo,
                                                            currentMultiCacheInfo.activeBlockToCacheMap,
                                                            currentMultiCacheInfo.multiCacheExternalSymbols);
            }
        }
    }

    for (auto& multiCacheInfo : multiCacheInfos)
    {
        std::string activeBlockTag = "active_block_" + std::to_string(util::GetUniqueId());
        if (multiCacheInfo.arrayAccessInfo.cacheUsedInRegion)
        {
            mlir::Block::iterator cacheRegionStart(beginCacheRegionOp);
            mlir::Block::iterator cacheRegionEnd(endOp);

            // Get the next loop outside of the trigger level loop
            // We can only double-buffer if there is a loop outside of the trigger level loop
            auto triggerLoopParentLoop = util::CastOrGetParentOfType<mlir::AffineForOp>(triggerLevelBlock->getParentOp());
            if (beginCacheRegionOp.doubleBufferCache() && triggerLoopParentLoop != nullptr)
            {
                [[maybe_unused]] bool inputOnlyCache = !multiCacheInfo.arrayAccessInfo.valueWritten;
                assert(inputOnlyCache && "Double buffering is only supported for read-only caches");

                auto doubleBufferTempArray = CreateDoubleBufferTempArray(rewriter, multiCacheInfo, beginCacheRegionOp);

                // Create the 0'th iteration copy just before the triggerLoopParentLoop
                auto parentLoopBlock = triggerLoopParentLoop->getBlock();

                rewriter.setInsertionPoint(parentLoopBlock, triggerLoopParentLoop->getIterator());

                mlir::Value triggerLoopParentIV = triggerLoopParentLoop.getInductionVar();
                [[maybe_unused]] int64_t triggerLoopParentFirstIterIntValue = triggerLoopParentLoop.getConstantLowerBound();
                [[maybe_unused]] int64_t triggerLoopParentStepSize = triggerLoopParentLoop.getStep();

                // Clone the parent loop and wrap it around this ActiveBlockCacheCopyOp for cache access resolution
                // However, limit the range to just a single iteration and remove everything inside the loop body
                auto clonedtriggerLoopParentLoop = dyn_cast<mlir::AffineForOp>(rewriter.clone(*(triggerLoopParentLoop.getOperation())));
                clonedtriggerLoopParentLoop.setConstantUpperBound(triggerLoopParentStepSize);

                // Erase the ops in the cloned loop body and put only the first iteration's cache copy in it followed by an affine.yield
                util::EraseAllOpsInBlock(rewriter, clonedtriggerLoopParentLoop.getLoopBody().front());

                auto loopBuilder = util::MakeBodyBuilder(clonedtriggerLoopParentLoop);

                auto firstIterCopy = loopBuilder.create<MultiCacheCopyOp>(loc,
                                                                          beginCacheRegionOp.input(),
                                                                          multiCacheInfo.multiCache,
                                                                          multiCacheInfo.multiCacheExternalSymbols,
                                                                          rewriter.getAffineMapArrayAttr(multiCacheInfo.multiCacheLBMaps),
                                                                          rewriter.getAffineMapArrayAttr(multiCacheInfo.multiCacheUBMaps),
                                                                          rewriter.getI64ArrayAttr(multiCacheInfo.multiCacheStepSizes),
                                                                          util::ConvertIndexVectorToArrayAttr(multiCacheInfo.multiCacheLoopIndexIds, rewriter.getContext()),
                                                                          rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.lbMaps),
                                                                          rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.ubMaps),
                                                                          multiCacheInfo.multiCacheExternalSymbolsPermutationMap,
                                                                          multiCacheInfo.activeBlockToCacheMap,
                                                                          activeBlockTag,
                                                                          beginCacheRegionOp.thrifty(),
                                                                          true, // toCache
                                                                          beginCacheRegionOp.vectorizationInfoAttr());
                // Re-create the affine yield op at the end of the block that we erased
                loopBuilder.create<mlir::AffineYieldOp>(loc);
                firstIterCopy->replaceUsesOfWith(triggerLoopParentIV, clonedtriggerLoopParentLoop.getInductionVar());

                rewriter.setInsertionPoint(triggerLevelBlock, cacheRegionStart);
                // Create the i+1 iteration copy to the temp buffer

                // Create the prologue cache data moving op
                auto loopStepIncrementExpr = rewriter.getAffineDimExpr(0) + rewriter.getAffineConstantExpr(triggerLoopParentStepSize);
                auto loopStepIncrementMap = mlir::AffineMap::get(1, 0, loopStepIncrementExpr);
                mlir::Value triggerLoopParentNextIterValue = rewriter.create<mlir::AffineApplyOp>(loc, loopStepIncrementMap, mlir::ValueRange{ triggerLoopParentIV });

                // Create an AffineIfOp to guard the cache fills so that it doesn't happen in the final iteration
                // We want to load if triggerLoopParentLoop < parentLoopLastIterInt
                int64_t parentLoopLastIterInt = triggerLoopParentLoop.getConstantUpperBound() - triggerLoopParentLoop.getStep();

                // the inequality will be ((lastIterIVValue - 1) - triggerLoopParentLoopIV >= 0)
                // The -1 is because it's a >= comparison and in the final iteration of the loop we want this check to return false
                mlir::AffineExpr lastIterIntMinusIVExpr = rewriter.getAffineConstantExpr(parentLoopLastIterInt - 1) - rewriter.getAffineDimExpr(0);
                std::vector<mlir::AffineExpr> conditionalLoadConstraintExprs{ lastIterIntMinusIVExpr };
                SmallVector<bool, 4> constraintEqFlags(1, false); // false indicating the checks should be >= 0 inequalities rather than == 0 equalities

                auto nonLastIterCheckSet = mlir::IntegerSet::get(1, 0, conditionalLoadConstraintExprs, constraintEqFlags);

                auto prologueCopyIfOp = rewriter.create<mlir::AffineIfOp>(loc, nonLastIterCheckSet, ValueRange{ triggerLoopParentIV }, false); // false indicating we do not want an "else" region
                auto prologueThenBuilder = prologueCopyIfOp.getThenBodyBuilder();

                MakeDelayedMappingRegion(prologueThenBuilder, triggerLoopParentIV, triggerLoopParentNextIterValue, [&](mlir::OpBuilder& builder) {
                    [[maybe_unused]] auto prologueTempCopy = builder.create<MultiCacheCopyOp>(loc,
                                                                             beginCacheRegionOp.input(),
                                                                             doubleBufferTempArray,
                                                                             multiCacheInfo.multiCacheExternalSymbols,
                                                                             rewriter.getAffineMapArrayAttr(multiCacheInfo.multiCacheLBMaps),
                                                                             rewriter.getAffineMapArrayAttr(multiCacheInfo.multiCacheUBMaps),
                                                                             rewriter.getI64ArrayAttr(multiCacheInfo.multiCacheStepSizes),
                                                                             util::ConvertIndexVectorToArrayAttr(multiCacheInfo.multiCacheLoopIndexIds, rewriter.getContext()),
                                                                             rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.lbMaps),
                                                                             rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.ubMaps),
                                                                             multiCacheInfo.multiCacheExternalSymbolsPermutationMap,
                                                                             multiCacheInfo.activeBlockToCacheMap,
                                                                             activeBlockTag,
                                                                             beginCacheRegionOp.thrifty(),
                                                                             true,
                                                                             beginCacheRegionOp.vectorizationInfoAttr()); // toCache
                });

                // Create mapping ops for each cache active block region associated with this multiCache
                CreateCacheMappingRegionHelper(rewriter, beginCacheRegionOp, multiCacheInfo);

                // Create the i+1 iteration copy from the temp buffer to the cache
                rewriter.setInsertionPoint(triggerLevelBlock, cacheRegionEnd);

                auto epilogueCopyIfOp = rewriter.create<mlir::AffineIfOp>(loc, nonLastIterCheckSet, ValueRange{ triggerLoopParentIV }, false); // false indicating we do not want an "else" region
                auto epilogueThenBuilder = epilogueCopyIfOp.getThenBodyBuilder();

                MakeDelayedMappingRegion(epilogueThenBuilder, triggerLoopParentIV, triggerLoopParentNextIterValue, [&](mlir::OpBuilder& builder) {
                    [[maybe_unused]] auto epilogueTempCopy = builder.create<MultiCacheCopyOp>(loc,
                                                                             multiCacheInfo.multiCache,
                                                                             doubleBufferTempArray,
                                                                             multiCacheInfo.multiCacheExternalSymbols,
                                                                             rewriter.getAffineMapArrayAttr(multiCacheInfo.multiCacheLBMaps),
                                                                             rewriter.getAffineMapArrayAttr(multiCacheInfo.multiCacheUBMaps),
                                                                             rewriter.getI64ArrayAttr(multiCacheInfo.multiCacheStepSizes),
                                                                             util::ConvertIndexVectorToArrayAttr(multiCacheInfo.multiCacheLoopIndexIds, rewriter.getContext()),
                                                                             rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.lbMaps),
                                                                             rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.ubMaps),
                                                                             multiCacheInfo.multiCacheExternalSymbolsPermutationMap,
                                                                             multiCacheInfo.activeBlockToCacheMap,
                                                                             activeBlockTag,
                                                                             beginCacheRegionOp.thrifty(),
                                                                             false, // toCache
                                                                             beginCacheRegionOp.vectorizationInfoAttr());
                });
                // Mark the trigger loop parent loop to unswitch the last iteration so that our affine.if checks
                // are always true in the main loop and always false in the unswitched final iteration
                triggerLoopParentLoop->setAttr(UnswitchSuffixItersName, rewriter.getI64IntegerAttr(1));
            }
            else
            {
                // Non-double-buffering case

                rewriter.setInsertionPoint(triggerLevelBlock, cacheRegionStart);

                // Create the prologue cache data moving op
                if (!multiCacheInfo.arrayAccessInfo.valueRead || multiCacheInfo.arrayAccessInfo.onlyReadsAreAccumulates)
                {
                    rewriter.create<CacheZeroOp>(loc, multiCacheInfo.multiCache, activeBlockTag, beginCacheRegionOp.thrifty());
                }
                else
                {
                    if (beginCacheRegionOp.activeBlockCache())
                    {
                        rewriter.create<MultiCacheCopyOp>(loc,
                                                          beginCacheRegionOp.input(),
                                                          multiCacheInfo.multiCache,
                                                          multiCacheInfo.multiCacheExternalSymbols,
                                                          rewriter.getAffineMapArrayAttr(multiCacheInfo.multiCacheLBMaps),
                                                          rewriter.getAffineMapArrayAttr(multiCacheInfo.multiCacheUBMaps),
                                                          rewriter.getI64ArrayAttr(multiCacheInfo.multiCacheStepSizes),
                                                          util::ConvertIndexVectorToArrayAttr(multiCacheInfo.multiCacheLoopIndexIds, rewriter.getContext()),
                                                          rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.lbMaps),
                                                          rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.ubMaps),
                                                          multiCacheInfo.multiCacheExternalSymbolsPermutationMap,
                                                          multiCacheInfo.activeBlockToCacheMap,
                                                          activeBlockTag,
                                                          beginCacheRegionOp.thrifty(),
                                                          true, // toCache
                                                          beginCacheRegionOp.vectorizationInfoAttr());
                    }
                    else
                    {
                        rewriter.create<ActiveElementCacheCopyOp>(loc, beginCacheRegionOp.input(), cacheAccessContext);
                    }
                }

                // Create mapping ops for each cache active block region associated with this multiCache
                CreateCacheMappingRegionHelper(rewriter, beginCacheRegionOp, multiCacheInfo);

                rewriter.setInsertionPoint(triggerLevelBlock, cacheRegionEnd);

                // Create the epilogue cache data moving op
                // If we never wrote to the value, then don't bother copying data out via any method
                if (multiCacheInfo.arrayAccessInfo.valueWritten)
                {
                    // Note: onlyReadsAreAccumulates defaults to true, but if no reads are seen don't want to use a CacheReduceOp
                    //       so check that reads occurred and that they were all used for accumulates
                    if (multiCacheInfo.arrayAccessInfo.valueRead && multiCacheInfo.arrayAccessInfo.onlyReadsAreAccumulates)
                    {
                        if (beginCacheRegionOp.activeBlockCache())
                        {
                            rewriter.create<ActiveBlockCacheReduceOp>(loc,
                                                                      beginCacheRegionOp.input(),
                                                                      multiCacheInfo.multiCache,
                                                                      multiCacheInfo.activeBlockInfo.externalSymbols,
                                                                      multiCacheInfo.activeBlockInfo.externalSymbols,
                                                                      rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.lbMaps),
                                                                      rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.ubMaps),
                                                                      multiCacheInfo.activeBlockToCacheMap,
                                                                      llvm::None, // scaleValues
                                                                      activeBlockTag,
                                                                      beginCacheRegionOp.thrifty(),
                                                                      beginCacheRegionOp.vectorizationInfoAttr());
                        }
                        else
                        {
                            rewriter.create<ActiveElementCacheReduceOp>(loc, cacheAccessContext, beginCacheRegionOp.input());
                        }
                    }
                    else
                    {
                        if (beginCacheRegionOp.activeBlockCache())
                        {
                            rewriter.create<ActiveBlockCacheCopyOp>(loc,
                                                                    beginCacheRegionOp.input(),
                                                                    multiCacheInfo.multiCache,
                                                                    multiCacheInfo.activeBlockInfo.externalSymbols,
                                                                    multiCacheInfo.activeBlockInfo.externalSymbols,
                                                                    mlir::ValueRange{},
                                                                    rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.lbMaps),
                                                                    rewriter.getAffineMapArrayAttr(multiCacheInfo.activeBlockInfo.ubMaps),
                                                                    multiCacheInfo.activeBlockToCacheMap,
                                                                    false, // toCache : this copy will copy from the cache back to the outer array
                                                                    activeBlockTag,
                                                                    beginCacheRegionOp.thrifty(),
                                                                    false, // skipBarriers : this copy isn't already guarded by barriers, so don't skip them
                                                                    beginCacheRegionOp.vectorizationInfoAttr());
                        }
                        else
                        {
                            rewriter.create<ActiveElementCacheCopyOp>(loc, cacheAccessContext, beginCacheRegionOp.input());
                        }
                    }
                }
            }
        }
    }

    rewriter.eraseOp(endOp);
    rewriter.eraseOp(beginCacheRegionOp);

    return success();
}

LogicalResult MaxElementCacheRegionOpRewrite::matchAndRewrite(BeginMaxElementCacheRegionOp beginMaxElementCacheRegionOp, PatternRewriter& rewriter) const
{
    // Compute where this cache region should be, based on the max element budget, then create a BeginCacheRegionOp at that level and a corresponding EndCacheRegionOp

    auto loc = beginMaxElementCacheRegionOp.getLoc();
    EndCacheRegionOp endOp = mlir::dyn_cast<EndCacheRegionOp>(beginMaxElementCacheRegionOp.getEndOp());

    // If the region is empty, then erase it an move on
    if (IsCacheRegionEmpty(beginMaxElementCacheRegionOp))
    {
        rewriter.eraseOp(endOp);
        rewriter.eraseOp(beginMaxElementCacheRegionOp);
        return success();
    }

    auto input = beginMaxElementCacheRegionOp.input();
    auto cache = beginMaxElementCacheRegionOp.cache();
    auto baseInput = beginMaxElementCacheRegionOp.baseInput();

    int64_t maxElementBudget = beginMaxElementCacheRegionOp.maxElements();

    mlir::Block* newBlock = beginMaxElementCacheRegionOp->getBlock();
    mlir::Block::iterator newBeginPoint(beginMaxElementCacheRegionOp);
    mlir::Block::iterator newEndPoint(endOp);

    ArrayAccessInfo arrayAccessInfo = ComputeAccessInfoForArrayAtLevel(rewriter,
                                                                       baseInput,
                                                                       mlir::Block::iterator(beginMaxElementCacheRegionOp),
                                                                       mlir::Block::iterator(endOp),
                                                                       true /* computeActiveBlock */);
    int64_t initialActiveBlockVolume = GetActiveBlockVolume(arrayAccessInfo.activeBlock);

    if (initialActiveBlockVolume == 0)
    {
        // If the array isn't used in this block, then the active block volume is 0 and we don't need to create a cache region
        rewriter.eraseOp(endOp);
        rewriter.eraseOp(beginMaxElementCacheRegionOp);
        return success();
    }

    mlir::AffineForOp cacheLevelLoop;

    if (initialActiveBlockVolume > maxElementBudget)
    {
        // If the max element budget is so small that even the innermost loop is too much, then create a dummy loop inside of it
        // TODO : make cache regions work as regions so that this dummy loop isn't needed
        mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
        auto originalInnerLoop = GetOutermostAffineForOp(beginMaxElementCacheRegionOp);
        auto& originalInnerBlock = originalInnerLoop->getRegion(0).front();
        rewriter.setInsertionPointToStart(&originalInnerBlock);
        auto newInnermostLoop = rewriter.create<mlir::AffineForOp>(loc, 0, 1);
        // Set an index attr on this loop so it can work with the rest of the caching ops. This doesn't need to be anything in particular,
        // but it must be unique from the other indices and the Index auto-incrementing ID will ensure this.
        Index dummyIndex("internal_dummy_loop");
        newInnermostLoop->setAttr("index", IndexAttr::get(dummyIndex, rewriter.getContext()));
        cacheLevelLoop = newInnermostLoop;

        // Move all of the original innermost loop contents into the new dummy loop body

        auto originalInnerLoopTerminatorOp = originalInnerBlock.getTerminator();
        auto newInnerLoopTerminatorOp = newInnermostLoop.getLoopBody().front().getTerminator();
        auto originalInnerLoopOpsBegin = ++mlir::Block::iterator(newInnermostLoop); // The new loop was inserted at the beginning of the block, so the original ops are all after it
        auto originalInnerLoopOpsEnd = mlir::Block::iterator(originalInnerLoopTerminatorOp);
        mlir::BlockAndValueMapping mapping;
        rewriter.setInsertionPoint(newInnerLoopTerminatorOp);
        std::stack<Operation*> opsToErase;
        for (auto originalOpsIter = originalInnerLoopOpsBegin; originalOpsIter != originalInnerLoopOpsEnd; ++originalOpsIter)
        {
            rewriter.clone(*originalOpsIter, mapping);
            opsToErase.push(&(*originalOpsIter));
        }

        while (!opsToErase.empty())
        {
            auto eraseOp = opsToErase.top();
            if (eraseOp->use_empty())
            {
                rewriter.eraseOp(eraseOp);
            }
            opsToErase.pop();
        }

        newBlock = newInnermostLoop->getBlock();
        newBeginPoint = newInnermostLoop->getIterator();
        newEndPoint = newInnermostLoop->getIterator();
        ++newEndPoint;
    }
    else
    {

        // Find the level to hoist the cache region to
        int64_t nextActiveBlockVolume = initialActiveBlockVolume;
        auto parentOp = GetOutermostAffineForOp(beginMaxElementCacheRegionOp);
        mlir::Block::iterator nextStart(parentOp);
        mlir::Block::iterator nextEnd = ++mlir::Block::iterator(parentOp);
        while (parentOp && nextActiveBlockVolume <= maxElementBudget)
        {
            newBlock = parentOp->getBlock();
            newBeginPoint = nextStart;
            newEndPoint = nextEnd;

            cacheLevelLoop = parentOp;

            parentOp = parentOp->getParentOfType<mlir::AffineForOp>();
            if (parentOp)
            {
                nextStart = mlir::Block::iterator(parentOp);
                nextEnd = ++mlir::Block::iterator(parentOp);
                ArrayAccessInfo nextArrayAccessInfo = ComputeAccessInfoForArrayAtLevel(rewriter, baseInput, nextStart, nextEnd, true /* computeActiveBlock */);
                nextActiveBlockVolume = GetActiveBlockVolume(nextArrayAccessInfo.activeBlock);
            }
        }
    }

    CacheAccessContext cacheAccessContext;
    cacheAccessContext.value = cache;
    cacheAccessContext.accessMaps = CacheAccessMaps::FromAttr(beginMaxElementCacheRegionOp.cacheAccessMaps());
    cacheAccessContext.activeBlockCache = true;
    cacheAccessContext.dimReorderCache = beginMaxElementCacheRegionOp.dimReorderCache();

    auto cacheIndex = cacheLevelLoop->getAttrOfType<IndexAttr>("index").getValue();

    rewriter.setInsertionPoint(newBlock, newEndPoint);
    rewriter.replaceOpWithNewOp<EndCacheRegionOp>(endOp, endOp.regionId());

    rewriter.setInsertionPoint(newBlock, newBeginPoint);
    auto doubleBufferMemorySpaceOpt = beginMaxElementCacheRegionOp.doubleBufferMemorySpace();
    auto doubleBufferMemorySpace = accera::ir::value::MemorySpace::None;
    if (doubleBufferMemorySpaceOpt.hasValue())
    {
        doubleBufferMemorySpace = doubleBufferMemorySpaceOpt.getValue();
    }
    auto newBeginOp = rewriter.create<BeginCacheRegionOp>(loc,
                                                          input,
                                                          cacheAccessContext,
                                                          baseInput,
                                                          cacheIndex, // triggerIndex
                                                          cacheIndex,
                                                          beginMaxElementCacheRegionOp.id(),
                                                          beginMaxElementCacheRegionOp.cacheHierarchyLevel(),
                                                          true, // activeBlockCache
                                                          beginMaxElementCacheRegionOp.dimReorderCache(),
                                                          beginMaxElementCacheRegionOp.thrifty(),
                                                          beginMaxElementCacheRegionOp.doubleBufferCache(),
                                                          doubleBufferMemorySpace,
                                                          GetCacheOpVectorizationInfoOrDefault(beginMaxElementCacheRegionOp));

    // This new cache region op has already been hoisted as high as we want to hoist it
    newBeginOp->setAttr("hoisted", rewriter.getUnitAttr());

    // Replace uses and erase the original BeginCacheRegionOp
    rewriter.replaceOp(beginMaxElementCacheRegionOp, newBeginOp.getResult());

    return success();
}

void VectorizeAffineForOpConversion::vectorizeOpsInBlock(PatternRewriter& rewriter,
                                                         mlir::Block::iterator begin,
                                                         mlir::Block::iterator endPrevSentinel,
                                                         mlir::Value unrollingIV,
                                                         const VectorizationInfo& vectorInfo,
                                                         VectorizedOpMap& vectorizedOps,
                                                         std::vector<BlockAndValueMapping>& laneMappings,
                                                         int64_t step,
                                                         int64_t unrollMax) const
{
    std::stack<Operation*> opsToErase;
    // Note: this loop needs to check std::next(endPrevSentinel) on every iteration since the vectorized ops are being inserted
    //       in the same block that this iterator is traversing, so std::next(endPrevSentinel) is initially the terminator op,
    //       but the new ops get inserted before the terminator op so std::next(endPrevSentinel) will change
    for (auto it = begin; it != std::next(endPrevSentinel); it++)
    {
        Operation* sourceOp = &(*it);

        // If this op can be vectorized, do it
        // Clone the op and then delete it if we were successful in vectorizing it.
        // When cloning, use a BlockAndValueMapping to remap the induction variable
        if (!vectorInfo.unrollOnly && CanVectorizeOp(sourceOp, vectorizedOps, laneMappings, unrollingIV, step, unrollMax))
        {
            auto result = VectorizeOp(rewriter, sourceOp, vectorizedOps, laneMappings, unrollingIV, step, unrollMax);
            if (result.has_value())
            {
                vectorizedOps.Map(sourceOp, *result);
                didVectorizeOp(sourceOp, *result);
            }
        }

        emitVectorizationRemark(sourceOp, "Unrolling op if needed");

        // Unroll the contents of 'forOpToUnroll' by replacing its contents with vectorSize mapped copies of it.
        for (int64_t unrollIdx = 0; unrollIdx < unrollMax; unrollIdx++)
        {
            auto& operandMap = laneMappings[unrollIdx];
            if (unrollIdx == 0)
            {
                opsToErase.push(sourceOp);
            }

            if (IsTerminalOp(sourceOp) && vectorizedOps.Lookup(sourceOp))
            {
                // this op has already been vectorized, and nothing else depends on it, so don't do anything
                emitVectorizationRemark(sourceOp, "Terminal op, vectorized");
            }
            else
            {
                if (IsTerminalOp(sourceOp))
                {
                    emitVectorizationRemark(sourceOp, "Terminal op, not vectorized");
                }

                [[maybe_unused]] auto mappedClonedOp = rewriter.clone(*it, operandMap);
            }
        }
    }

    while (!opsToErase.empty())
    {
        auto eraseOp = opsToErase.top();
        if (eraseOp->use_empty())
        {
            rewriter.eraseOp(eraseOp);
        }
        opsToErase.pop();
    }
}

LogicalResult VectorizeAffineForOpConversion::matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const
{
    if (!HasVectorizationInfo(affineForOp))
    {
        // This isn't an AffineForOp marked for vectorization so just return without modifying it
        return failure();
    }

    auto vectorInfo = GetVectorizationInfo(affineForOp);

    // Enforce some simplifying assumptions about the affine for loop:
    //  - the loop must have a constant trip count
    //  - the loop must have a constant lower bound
    //  - the loop must have a constant upper bound
    // TODO : eventually we'll want to relax these requirements
    Optional<uint64_t> mayBeConstantTripCount = mlir::getConstantTripCount(affineForOp);
    assert(mayBeConstantTripCount.hasValue() && "Vectorized loops must have a constant trip count");
    uint64_t constantTripCount = mayBeConstantTripCount.getValue();
    if (constantTripCount == 0)
    {
        // Discard loops that never run
        rewriter.eraseOp(affineForOp);
        return success();
    }

    auto affineForOpIV = affineForOp.getInductionVar();

    if (affineForOpIV.use_empty())
    {
        // Don't vectorize loops that never uses the induction variable
        return success();
    }

    rewriter.startRootUpdate(affineForOp);

    assert(affineForOp.hasConstantLowerBound() && "Vectorized loops must have a constant lower bound");
    assert(affineForOp.hasConstantUpperBound() && "Vectorized loops must have a constant upper bound");

    // Unroll this AffineForOp and replace the appropriate CacheLoads and CacheStores with VectorizedCacheLoad and VectorizedCacheStore

    // this is a partial port of the meaty bits of mlir::loopUnrollByFactor() from mlir/lib/Transforms/Utils/LoopUtils.cpp
    // but with access to the unroll indices in order to make VectorizedCacheLoad and VectorizedCacheStore
    // and with some more simplifying assumptions and op replacements

    // remove the vectorization attribute from the AffineForOp
    RemoveVectorizationInfo(affineForOp);

    bool erasedBaseLoop = false;
    int64_t step = affineForOp.getStep();

    // Scale the step of loop being unrolled by unroll factor.
    auto numIters = CeilDiv(affineForOp.getConstantUpperBound() - affineForOp.getConstantLowerBound(), affineForOp.getStep());
    int64_t unrollMax = std::min(affineForOp.getConstantUpperBound() - affineForOp.getConstantLowerBound(), numIters);
    affineForOp.setStep(step * numIters);

    // Insert unrolled bodies just before the terminator of the body of 'affineForOp'.
    rewriter.setInsertionPoint(affineForOp.getBody(), affineForOp.getBody()->getTerminator()->getIterator());

    // Keep a pointer to the last non-terminator operation in the original block
    // so that we know what to clone (since we are doing this in-place).
    Block::iterator srcBlockEnd = std::prev(affineForOp.getBody()->end(), 2);

    VectorizedOpMap vectorizedOps;
    std::vector<BlockAndValueMapping> laneMappings(unrollMax);

    if (!affineForOpIV.use_empty())
    {
        // Initialize the mappings with an offset version of the induction variable
        auto loc = affineForOp.getLoc();
        auto inductionVarMap = AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + step * rewriter.getAffineSymbolExpr(0));
        for (int64_t i = 0; i < unrollMax; ++i)
        {
            auto offset = rewriter.create<mlir::ConstantIndexOp>(loc, i);
            auto offsetInductionVar = rewriter.create<AffineApplyOp>(loc, inductionVarMap, ValueRange{ affineForOpIV, offset });

            BlockAndValueMapping& operandMap = laneMappings[i];
            operandMap.map(affineForOpIV, offsetInductionVar);
        }
    }

    vectorizeOpsInBlock(rewriter, affineForOp.getBody()->begin(), srcBlockEnd, affineForOpIV, vectorInfo, vectorizedOps, laneMappings, step, unrollMax);

    if (!erasedBaseLoop)
    {
        (void)util::PromoteIfSingleIteration(rewriter, affineForOp);
    }

    rewriter.finalizeRootUpdate(affineForOp);

    return success();
}

void VectorizeAffineForOpConversion::didVectorizeOp(mlir::Operation* sourceOp, VectorizedOp& vectorizedOp) const
{
    if (printVectorizationDetails)
    {
        auto diagnostic = sourceOp->emitRemark("Vectorized");
        if (vectorizedOp.HasVectorType())
        {
            auto vecResult = vectorizedOp.GetVectorResult();
            if (vecResult && vecResult.getDefiningOp())
            {
                diagnostic << " -- " << vecResult.getDefiningOp();
            }
            else if (auto resultOp = vectorizedOp.GetOp())
            {
                diagnostic << " -- terminal op: " << resultOp;
            }
            else
            {
                diagnostic << " -- terminal op";
            }
        }
    }

    if (!vectorizedOp.HasVectorType())
    {
        // also add to vectorized ops?
        if (printVectorizationDetails)
        {
            sourceOp->emitRemark("Vectorized to a non-vector type");
        }
    }
}

void VectorizeAffineForOpConversion::emitVectorizationRemark(mlir::Operation* sourceOp, const std::string& remark) const
{
    if (printVectorizationDetails)
    {
        sourceOp->emitRemark(remark);
    }
}

// TODO : de-dupe with vectorization
void InPlaceUnrollOpsInBlock(PatternRewriter& rewriter,
                             mlir::Block::iterator begin,
                             mlir::Block::iterator endPrevSentinel,
                             mlir::Value unrollingIV,
                             const InPlaceUnrollInfo& inPlaceUnrollInfo,
                             VectorizedOpMap& vectorizedOps,
                             std::vector<BlockAndValueMapping>& laneMappings,
                             int64_t step,
                             int64_t unrollMax)
{
    std::stack<Operation*> opsToErase;
    // Note: this loop needs to check std::next(endPrevSentinel) on every iteration since the unrolled ops are being inserted
    //       in the same block that this iterator is traversing, so std::next(endPrevSentinel) is initially the terminator op,
    //       but the new ops get inserted before the terminator op so std::next(endPrevSentinel) will change
    for (auto it = begin; it != std::next(endPrevSentinel); it++)
    {
        Operation* sourceOp = &(*it);

        // If this op is an AffineForOp that is also being in-place-unrolled or vectorized, then in-place unroll the ops inside it from the point of view of this unrollingIV without
        // unrolling/vectorizing the AffineForOp itself
        if (auto innerForOp = mlir::dyn_cast<mlir::AffineForOp>(sourceOp); innerForOp && (HasVectorizationInfo(innerForOp) || HasInPlaceUnrollInfo(innerForOp)))
        {
            // In-place unroll the ops inside this loop, but don't unroll the loop terminator
            // Get a sentinel op for where we should stop vectorizing by stepping back from the end of the for loop by stepping back 2 ops from the end of the body:
            // - stepping back 1 op from the end would get us the terminator op, which will move as we insert the new vectorized ops before the terminator
            // - stepping back 2 ops from the end will get us the last original op that should be vectorized, so we can check if the iterator == std::next(innerLoopBlockEndSentinel) to determine when we've gone too far
            Block::iterator innerLoopBlockEndSentinel = std::prev(innerForOp.getBody()->end(), 2);
            mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(innerForOp.getBody(), innerForOp.getBody()->getTerminator()->getIterator());
            InPlaceUnrollOpsInBlock(rewriter, innerForOp.getBody()->begin(), innerLoopBlockEndSentinel, unrollingIV, inPlaceUnrollInfo, vectorizedOps, laneMappings, step, unrollMax);
            continue;
        }

        // Unroll the contents of 'forOpToUnroll' by replacing its contents with vectorSize mapped copies of it.
        for (int64_t unrollIdx = 0; unrollIdx < unrollMax; unrollIdx++)
        {
            auto& operandMap = laneMappings[unrollIdx];
            if (unrollIdx == 0)
            {
                opsToErase.push(sourceOp);
            }

            if (!(IsTerminalOp(sourceOp) && vectorizedOps.Lookup(sourceOp)))
            {
                [[maybe_unused]] auto mappedClonedOp = rewriter.clone(*it, operandMap);
            }
        }
    }

    while (!opsToErase.empty())
    {
        auto eraseOp = opsToErase.top();
        if (eraseOp->use_empty())
        {
            rewriter.eraseOp(eraseOp);
        }
        opsToErase.pop();
    }
}

LogicalResult InPlaceUnrollAffineForOpConversion::matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const
{
    if (!HasInPlaceUnrollInfo(affineForOp))
    {
        // This isn't an AffineForOp marked for in-place unroll so just return without modifying it
        return failure();
    }

    auto inPlaceUnrollInfo = GetInPlaceUnrollInfo(affineForOp);
    bool fullyUnroll = (inPlaceUnrollInfo.loopUnrollFactor == 0);

    // Enforce some simplifying assumptions about the affine for loop:
    //  - the loop must have a constant trip count
    //  - the loop must have a constant lower bound
    //  - the loop must have a constant upper bound
    // TODO : eventually we'll want to relax these requirements
    Optional<uint64_t> mayBeConstantTripCount = mlir::getConstantTripCount(affineForOp);
    assert(mayBeConstantTripCount.hasValue() && "Vectorized loops must have a constant trip count");
    uint64_t constantTripCount = mayBeConstantTripCount.getValue();
    if (constantTripCount == 0)
    {
        // Discard loops that never run
        rewriter.eraseOp(affineForOp);
        return success();
    }
    int64_t loopUnrollFactor = fullyUnroll ? constantTripCount : inPlaceUnrollInfo.loopUnrollFactor;

    auto originalInductionVar = affineForOp.getInductionVar();

    if (originalInductionVar.use_empty())
    {
        // Don't unroll loops that never uses the induction variable
        return success();
    }

    rewriter.startRootUpdate(affineForOp);

    assert(affineForOp.hasConstantLowerBound() && "In-place unrolled loops must have a constant lower bound");
    assert(affineForOp.hasConstantUpperBound() && "In-place unrolled loops must have a constant upper bound");

    // remove the in-place unroll attribute from the AffineForOp
    RemoveInPlaceUnrollInfo(affineForOp);

    std::vector<AffineForOp> forOpsToUnroll;

    bool erasedBaseLoop = false;
    int64_t step = affineForOp.getStep();

    // Generate the cleanup loop if trip count isn't a multiple of loopUnrollFactor
    auto cleanupIterations = constantTripCount % loopUnrollFactor;
    if (cleanupIterations != 0)
    {
        rewriter.setInsertionPoint(affineForOp.getOperation()->getBlock(),
                                   std::next(Block::iterator(affineForOp)));
        auto cleanupForOp = cast<AffineForOp>(rewriter.clone(*affineForOp));

        // Compute lower bound of the cleanup loop. The (new) base loop has (constantTripCount-cleanupIterations) iterations,
        // for a total extent of (constantTripCount-cleanupIterations) * step.
        int64_t originalLowerBound = affineForOp.hasConstantLowerBound() ? affineForOp.getConstantLowerBound() : 0; // handle the case where the base loop doesn't start at 0
        int64_t cleanupLowerBound = originalLowerBound + ((constantTripCount - cleanupIterations) * step);
        cleanupForOp.setConstantLowerBound(cleanupLowerBound);

        // Adjust upper bound of the original loop; this is the same as the lower
        // bound of the cleanup loop.
        affineForOp.setConstantUpperBound(cleanupLowerBound);

        // If the non-cleanup loop now has 0 iterations, erase it, otherwise enqueue it to be unrolled
        Optional<uint64_t> adjustedMayBeConstantTripCount = mlir::getConstantTripCount(affineForOp);
        assert(adjustedMayBeConstantTripCount.hasValue() && "In-place unrolled loops must have a constant trip count");
        uint64_t adjustedConstantTripCount = adjustedMayBeConstantTripCount.getValue();
        if (adjustedConstantTripCount == 0)
        {
            rewriter.eraseOp(affineForOp);
            erasedBaseLoop = true;
        }
        else
        {
            forOpsToUnroll.push_back(affineForOp);
        }

        forOpsToUnroll.push_back(cleanupForOp);
    }
    else
    {
        forOpsToUnroll.push_back(affineForOp);
    }

    for (auto& forOpToUnroll : forOpsToUnroll)
    {
        // Scale the step of loop being unrolled by unroll factor.
        auto numIters = CeilDiv(forOpToUnroll.getConstantUpperBound() - forOpToUnroll.getConstantLowerBound(), forOpToUnroll.getStep());
        auto thisLoopUnrollFactor = std::min(numIters, loopUnrollFactor);
        forOpToUnroll.setStep(step * thisLoopUnrollFactor);

        // Insert unrolled bodies just before the terminator of the body of 'forOpToUnroll'.
        rewriter.setInsertionPoint(forOpToUnroll.getBody(), forOpToUnroll.getBody()->getTerminator()->getIterator());

        // Keep a pointer to the last non-terminator operation in the original block
        // so that we know what to clone (since we are doing this in-place).
        Block::iterator srcBlockEnd = std::prev(forOpToUnroll.getBody()->end(), 2);

        auto forOpToUnrollIV = forOpToUnroll.getInductionVar();

        // Clean up some ops after we've unrolled and mapped everything
        std::stack<Operation*> opsToErase;
        std::map<Operation*, Operation*> cacheLoadReplacementMapping; // TODO : figure out why BlockAndValueMapping isn't handling these cases

        int64_t unrollMax = std::min(forOpToUnroll.getConstantUpperBound() - forOpToUnroll.getConstantLowerBound(), thisLoopUnrollFactor);

        VectorizedOpMap vectorizedOps;
        std::vector<BlockAndValueMapping> laneMappings(unrollMax);

        if (!forOpToUnrollIV.use_empty())
        {
            // Initialize the mappings with an offset version of the induction variable
            auto loc = forOpToUnroll.getLoc();
            auto inductionVarMap = AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + step * rewriter.getAffineSymbolExpr(0));
            for (int64_t i = 0; i < unrollMax; ++i)
            {
                auto offset = rewriter.create<mlir::ConstantIndexOp>(loc, i);
                auto offsetInductionVar = rewriter.create<AffineApplyOp>(loc, inductionVarMap, ValueRange{ forOpToUnrollIV, offset });

                BlockAndValueMapping& operandMap = laneMappings[i];
                operandMap.map(forOpToUnrollIV, offsetInductionVar);
            }
        }

        InPlaceUnrollOpsInBlock(rewriter, forOpToUnroll.getBody()->begin(), srcBlockEnd, forOpToUnrollIV, inPlaceUnrollInfo, vectorizedOps, laneMappings, step, unrollMax);
    }

    if (!erasedBaseLoop)
    {
        (void)util::PromoteIfSingleIteration(rewriter, affineForOp);
    }

    rewriter.finalizeRootUpdate(affineForOp);

    return success();
}

SmallVector<AffineForOp, 4> GetLoopOpsForOperands(mlir::Operation::operand_range operands)
{
    SmallVector<mlir::AffineForOp, 4> operandLoops;
    std::transform(operands.begin(), operands.end(), std::back_inserter(operandLoops), [](mlir::Value operand) {
        return mlir::getForInductionVarOwner(operand);
    });

    return operandLoops;
}

std::optional<GPUIndexDimension> GetGPUIndexForLoop(AffineForOp loop)
{
    if (auto gpuMapAttr = loop->getAttrOfType<StringAttr>("accv_gpu_map"))
    {
        auto attrVal = gpuMapAttr.getValue();
        if (attrVal == "ThreadX")
            return GPUIndexDimension::X;
        else if (attrVal == "BlockX")
            return GPUIndexDimension::X;
        else if (attrVal == "ThreadY")
            return GPUIndexDimension::Y;
        else if (attrVal == "BlockY")
            return GPUIndexDimension::Y;
    }
    return std::nullopt;
}

std::set<GPUIndexDimension> GetGPUIndexDimensionsInExpr(const AffineExpr& expr, ArrayRef<AffineForOp> loops)
{
    std::set<GPUIndexDimension> result;
    expr.walk([&](AffineExpr subExpr) {
        if (auto dimExpr = subExpr.dyn_cast<AffineDimExpr>())
        {
            auto pos = dimExpr.getPosition();
            if (auto indexLoop = loops[pos])
            {
                if (auto gpuIndex = GetGPUIndexForLoop(indexLoop))
                {
                    result.insert(*gpuIndex);
                }
            }
        }
    });
    return result;
}

std::vector<int64_t> GetLoopStepsInExpr(const AffineExpr& expr, ArrayRef<AffineForOp> loops)
{
    std::vector<int64_t> result;
    expr.walk([&](AffineExpr subExpr) {
        if (auto dimExpr = subExpr.dyn_cast<AffineDimExpr>())
        {
            auto pos = dimExpr.getPosition();
            if (auto indexLoop = loops[pos])
            {
                result.push_back(indexLoop.getStep());
            }
            else
            {
                result.push_back(-1);
            }
        }
    });

    // Add outermost loop bounds if there is only 1
    if (result.size() == 1)
    {
        std::vector<AffineExpr> loopDims;
        std::transform(loops.begin(), loops.end(), std::back_inserter(loopDims), [&](mlir::AffineForOp loop) {
            return mlir::getAffineConstantExpr(loop.getConstantUpperBound(), expr.getContext());
        });
        // get extent of expr by replacing all dims with upper loop bounds
        auto boundsExpr = simplifyAffineExpr(expr.replaceDims(loopDims), loops.size(), 0);
        if (boundsExpr.isa<mlir::AffineConstantExpr>())
        {
            result.push_back(boundsExpr.cast<mlir::AffineConstantExpr>().getValue());
        }
        else
        {
            result.push_back(-1);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

template <typename OpT>
std::vector<std::set<GPUIndexDimension>> GetGPUIndexDimensionsInAffineMemOp(OpT op)
{
    std::vector<std::set<GPUIndexDimension>> result;
    auto map = op.getAffineMap();
    auto loops = GetLoopOpsForOperands(op.getMapOperands());
    for (unsigned r = 0; r < map.getNumResults(); ++r)
    {
        auto expr = map.getResult(r);
        auto exprDims = GetGPUIndexDimensionsInExpr(expr, loops);
        result.push_back(exprDims);
    }
    return result;
}

std::optional<GPUIndexDimension> GetSingleGPUDimension(std::set<GPUIndexDimension>& indexDimensions)
{
    if (indexDimensions.size() != 1)
        return std::nullopt;
    return *indexDimensions.begin();
}

std::optional<GPUIndexDimension> GetSingleGPUDimension(const std::vector<std::set<GPUIndexDimension>>& indexDimensions)
{
    std::set<GPUIndexDimension> unionSet;
    for (const auto& dimSet : indexDimensions)
    {
        unionSet.insert(dimSet.begin(), dimSet.end());
    }

    return GetSingleGPUDimension(unionSet);
}

template <typename OpT>
std::vector<std::vector<int64_t>> GetIndexLoopStepSizesInAffineMemOp(OpT op)
{
    std::vector<std::vector<int64_t>> result;
    auto map = op.getAffineMap();
    auto loops = GetLoopOpsForOperands(op.getMapOperands());
    for (unsigned r = 0; r < map.getNumResults(); ++r)
    {
        auto expr = map.getResult(r);
        result.push_back(GetLoopStepsInExpr(expr, loops));
    }
    return result;
}

template <typename OpT1, typename OpT2>
bool AreSameElement(OpT1 memOp1, OpT2 memOp2)
{
    if (memOp1.getAffineMap() != memOp2.getAffineMap())
    {
        return false;
    }
    AffineValueMap memOp2ValueMap(memOp2.getAffineMap(), memOp2.getMapOperands());
    AffineValueMap memOp1ValueMap(memOp1.getAffineMap(), memOp1.getMapOperands());
    AffineValueMap differenceMap;
    AffineValueMap::difference(memOp2ValueMap, memOp1ValueMap, &differenceMap);
    if (!differenceMap.getAffineMap().isConstant())
    {
        return false;
    }
    auto constantResults = differenceMap.getAffineMap().getConstantResults();
    for (unsigned i = 0; i < differenceMap.getNumResults(); ++i)
    {
        if (constantResults[i] != 0)
        {
            return false;
        }
    }
    return true;
}

auto LoadMatrixOpROCM(PatternRewriter& rewriter, Location& loc, AffineLoadOp loadOp, const int64_t warpSize, const v::MMAOp& mmaOp, MMAOperandType kind, const int numPassesInGroup, mlir::Value offset, std::pair<mlir::Value, mlir::Value> rowcol, const int loadAGPUIndexPos, const int loadBGPUIndexPos, GPUIndexDimension gpuDimsC0)
{
    auto ctx = rewriter.getContext();
    AffineMap offsetMap;
    const auto rowThreadblockOffset = rowcol.first;
    const auto colThreadblockOffset = rowcol.second;
    const auto numInElementsPerGroup = numPassesInGroup * mmaOp.getInElementsPerThread(warpSize);
    SmallVector<mlir::Value, 4> loadOpOperands(loadOp.getMapOperands());
    auto elementType = loadOp.getMemRefType().getElementType();
    MemRefType vecTy;
    [[maybe_unused]] auto d0 = rewriter.getAffineDimExpr(0);
    [[maybe_unused]] auto d1 = rewriter.getAffineDimExpr(1);
    auto s0 = rewriter.getAffineSymbolExpr(0);
    auto s1 = rewriter.getAffineSymbolExpr(1);
    switch (kind)
    {
    case MMAOperandType::A:
        vecTy = mlir::MemRefType::get({ numInElementsPerGroup }, elementType);
        if (loadAGPUIndexPos == 0)
            offsetMap = AffineMap::get(2, 2, { s0, d1 + s1 }, ctx);
        else
            offsetMap = AffineMap::get(2, 2, { d0 + s1, s0 }, ctx);
        loadOpOperands.push_back(rowThreadblockOffset);
        loadOpOperands.push_back(offset);
        break;

    case MMAOperandType::B:
        vecTy = mlir::MemRefType::get({ numInElementsPerGroup }, elementType);
        if (loadBGPUIndexPos == 1)
            offsetMap = AffineMap::get(2, 2, { d0 + s1, s0 }, ctx);
        else
            offsetMap = AffineMap::get(2, 2, { s0, d1 + s1 }, ctx);
        loadOpOperands.push_back(colThreadblockOffset);
        loadOpOperands.push_back(offset);
        break;

    case MMAOperandType::Acc:
        // For FP16 output, we need to load C in FP32 mode before passing to MFMA
        if (elementType.isF16())
            elementType = rewriter.getF32Type();

        vecTy = mlir::MemRefType::get({ mmaOp.getOutElementsPerThread(warpSize) / mmaOp.getNumBlocks() }, elementType);
        if (gpuDimsC0 == GPUIndexDimension::Y)
            offsetMap = AffineMap::get(2, 2, { s0, s1 }, ctx);
        else
            offsetMap = AffineMap::get(2, 2, { s1, s0 }, ctx);
        loadOpOperands.push_back(rewriter.create<AddIOp>(loc, rowThreadblockOffset, offset));
        loadOpOperands.push_back(colThreadblockOffset);
        break;

    default:
        llvm::report_fatal_error("Unknown kind of matrix");
    }

    // llvm::dbgs() << "COp with offset " << offsetMap << " with load "
    //              << loadOp.getAffineMap() << " and composed = " << offsetMap.compose(loadOp.getAffineMap()) << "\n";

    std::vector<mlir::Value> ops{ rewriter.create<MMALoadSyncOp>(loc,
                                                                 vecTy,
                                                                 loadOp.memref(),
                                                                 mmaOp.getShapeType(),
                                                                 kind,
                                                                 offsetMap.compose(loadOp.getAffineMap()),
                                                                 loadOpOperands) };
    return ops;
}

auto LoadMatrixOpCUDA(PatternRewriter& rewriter, Location& loc, AffineLoadOp loadOp, const int64_t warpSize, const v::MMAOp& mmaOp, MMAOperandType kind, const int numPassesInGroup, mlir::Value offset, std::pair<mlir::Value, mlir::Value> rowcol, const int loadAGPUIndexPos, const int loadBGPUIndexPos, GPUIndexDimension gpuDimsC0)
{
    std::vector<mlir::Value> ops;
    auto ctx = rewriter.getContext();
    AffineMap offsetMap;
    const auto rowThreadblockOffset = rowcol.first;
    const auto colThreadblockOffset = rowcol.second;
    const auto passIncrements = mmaOp.getPassIncrements(warpSize);
    auto loadOperands = loadOp.getMapOperands();
    auto elementType = loadOp.getMemRefType().getElementType();
    gpu::MMAMatrixType destMemRefType;
    [[maybe_unused]] auto d0 = rewriter.getAffineDimExpr(0);
    [[maybe_unused]] auto d1 = rewriter.getAffineDimExpr(1);
    auto s0 = rewriter.getAffineSymbolExpr(0);
    auto s1 = rewriter.getAffineSymbolExpr(1);
    for (int iPass = 0; iPass < numPassesInGroup; ++iPass, s1 = s1 + passIncrements)
    {
        std::vector<mlir::Value> loadOpOperands(loadOperands.begin(), loadOperands.end());
        switch (kind)
        {
        case MMAOperandType::A:
            destMemRefType = gpu::MMAMatrixType::get({ mmaOp.getM(), mmaOp.getK() }, elementType, "AOp");
            if (loadAGPUIndexPos == 0)
                offsetMap = AffineMap::get(2, 2, { s0, d1 + s1 }, ctx);
            else
                offsetMap = AffineMap::get(2, 2, { d0 + s1, s0 }, ctx);
            loadOpOperands.push_back(rowThreadblockOffset);
            loadOpOperands.push_back(offset);
            break;

        case MMAOperandType::B:
            destMemRefType = gpu::MMAMatrixType::get({ mmaOp.getK(), mmaOp.getN() }, elementType, "BOp");
            if (loadBGPUIndexPos == 1)
                offsetMap = AffineMap::get(2, 2, { d0 + s1, s0 }, ctx);
            else
                offsetMap = AffineMap::get(2, 2, { s0, d1 + s1 }, ctx);
            loadOpOperands.push_back(colThreadblockOffset);
            loadOpOperands.push_back(offset);
            break;

        case MMAOperandType::Acc:
            destMemRefType = gpu::MMAMatrixType::get({ mmaOp.getM(), mmaOp.getN() }, elementType, "COp");
            if (gpuDimsC0 == GPUIndexDimension::Y)
                offsetMap = AffineMap::get(2, 2, { s0, s1 }, ctx);
            else
                offsetMap = AffineMap::get(2, 2, { s1, s0 }, ctx);
            loadOpOperands.push_back(rewriter.create<AddIOp>(loc, rowThreadblockOffset, offset));
            loadOpOperands.push_back(colThreadblockOffset);
            break;

        default:
            llvm::report_fatal_error("Unknown kind of matrix");
        }

        // llvm::dbgs() << "COp with offset " << offsetMap << " with load "
        //              << loadOp.getAffineMap() << " and composed = " << offsetMap.compose(loadOp.getAffineMap()) << "\n";

        int64_t offset;
        SmallVector<int64_t, 2> strides;
        (void)mlir::getStridesAndOffset(loadOp.getMemRefType(), strides, offset);
        mlir::Value castMemref;
        if (strides[1] == 1) // row-major
        {
            castMemref = rewriter.create<memref::CastOp>(loc, canonicalizeStridedLayout(loadOp.getMemRefType()), loadOp.memref());
        }
        else // col-major
        {
            assert(strides[0] == 1);

            // SubgroupMmaLoadMatrixOp only takes as input memref which has identity maps, that is why we need to transpose (metadata only) before passing it
            auto transposeMemref = rewriter.create<memref::TransposeOp>(loc, loadOp.memref(), mlir::AffineMapAttr::get(AffineMap::get(2, 0, { d1, d0 }, ctx)));
            castMemref = rewriter.create<memref::CastOp>(loc, canonicalizeStridedLayout(transposeMemref.getType().cast<MemRefType>()), transposeMemref);
        }

        auto mappedOperands = util::MultiDimAffineApply(rewriter, loc, offsetMap.compose(loadOp.getAffineMap()), loadOpOperands);
        ops.push_back(rewriter.create<gpu::SubgroupMmaLoadMatrixOp>(loc,
                                                                    destMemRefType,
                                                                    castMemref,
                                                                    mappedOperands,
                                                                    rewriter.getIndexAttr(strides[0])));
    }
    return ops;
}

void StoreMatrixOpROCM(PatternRewriter& rewriter, Location& loc, AffineStoreOp storeOp, const v::MMAOp& mmaOp, Value value, const int blockRowOffset, std::pair<mlir::Value, mlir::Value> rowcol, GPUIndexDimension gpuDimsC0)
{
    auto ctx = rewriter.getContext();
    const auto rowThreadblockOffset = rowcol.first;
    const auto colThreadblockOffset = rowcol.second;
    auto rowOffsetSym = rewriter.getAffineSymbolExpr(0);
    auto colOffsetSym = rewriter.getAffineSymbolExpr(1);
    AffineMap offsetMap;
    if (gpuDimsC0 == GPUIndexDimension::Y)
        offsetMap = AffineMap::get(2, 2, { rowOffsetSym, colOffsetSym }, ctx);
    else
        offsetMap = AffineMap::get(2, 2, { colOffsetSym, rowOffsetSym }, ctx);
    SmallVector<mlir::Value, 4> storeOpOperands(storeOp.getMapOperands());
    auto mfma_block_offset = rewriter.create<ConstantIndexOp>(loc, blockRowOffset);
    storeOpOperands.push_back(rewriter.create<AddIOp>(loc, rowThreadblockOffset, mfma_block_offset));
    storeOpOperands.push_back(colThreadblockOffset);
    // llvm::dbgs() << "COpOut with offset " << offsetMap << " with load "
    //              << storeOp.getAffineMap() << " and composed = " << offsetMap.compose(storeOp.getAffineMap()) << "\n";
    rewriter.create<MMAStoreSyncOp>(loc,
                                    value,
                                    storeOp.memref(),
                                    mmaOp.getShapeType(),
                                    offsetMap.compose(storeOp.getAffineMap()),
                                    storeOpOperands);
}

void StoreMatrixOpCUDA(PatternRewriter& rewriter, Location& loc, AffineStoreOp storeOp, const v::MMAOp& mmaOp, Value value, std::pair<mlir::Value, mlir::Value> rowcol, GPUIndexDimension gpuDimsC0)
{
    auto ctx = rewriter.getContext();
    const auto rowThreadblockOffset = rowcol.first;
    const auto colThreadblockOffset = rowcol.second;
    auto rowOffsetSym = rewriter.getAffineSymbolExpr(0);
    auto colOffsetSym = rewriter.getAffineSymbolExpr(1);
    AffineMap offsetMap;
    if (gpuDimsC0 == GPUIndexDimension::Y)
        offsetMap = AffineMap::get(2, 2, { rowOffsetSym, colOffsetSym }, ctx);
    else
        offsetMap = AffineMap::get(2, 2, { colOffsetSym, rowOffsetSym }, ctx);

    auto storeOperands = storeOp.getMapOperands();
    std::vector<mlir::Value> storeOpOperands(storeOperands.begin(), storeOperands.end());
    storeOpOperands.push_back(rowThreadblockOffset);
    storeOpOperands.push_back(colThreadblockOffset);

    int64_t offset;
    SmallVector<int64_t, 2> strides;
    (void)mlir::getStridesAndOffset(storeOp.getMemRefType(), strides, offset);
    mlir::Value castMemref;
    if (strides[1] == 1) // row-major
    {
        castMemref = rewriter.create<memref::CastOp>(loc, canonicalizeStridedLayout(storeOp.getMemRefType()), storeOp.memref());
    }
    else // col-major
    {
        assert(strides[0] == 1);

        // SubgroupMmaStoreMatrixOp only takes as input memref which has identity maps, that is why we need to transpose (metadata only) before passing it
        auto d0 = rewriter.getAffineDimExpr(0);
        auto d1 = rewriter.getAffineDimExpr(1);
        auto transposeMemref = rewriter.create<memref::TransposeOp>(loc, storeOp.memref(), mlir::AffineMapAttr::get(AffineMap::get(2, 0, { d1, d0 }, ctx)));
        castMemref = rewriter.create<memref::CastOp>(loc, canonicalizeStridedLayout(transposeMemref.getType().cast<MemRefType>()), transposeMemref);
    }

    auto mappedOperands = util::MultiDimAffineApply(rewriter, loc, offsetMap.compose(storeOp.getAffineMap()), storeOpOperands);
    // llvm::dbgs() << "COpOut with offset " << offsetMap << " with load "
    //              << storeOp.getAffineMap() << " and composed = " << offsetMap.compose(storeOp.getAffineMap()) << "\n";
    rewriter.create<gpu::SubgroupMmaStoreMatrixOp>(loc,
                                                   value,
                                                   castMemref,
                                                   mappedOperands,
                                                   rewriter.getIndexAttr(strides[0]));
}

std::vector<mlir::Value> LoadMatrixOp(ExecutionRuntime runtime, PatternRewriter& rewriter, Location& loc, AffineLoadOp loadOp, const int64_t warpSize, const v::MMAOp& mmaOp, MMAOperandType kind, const int numPassesInGroup, mlir::Value offset, std::pair<mlir::Value, mlir::Value> rowcol, const int loadAGPUIndexPos, const int loadBGPUIndexPos, GPUIndexDimension gpuDimsC0)
{
    if (runtime == ExecutionRuntime::ROCM)
    {
        return LoadMatrixOpROCM(rewriter, loc, loadOp, warpSize, mmaOp, kind, numPassesInGroup, offset, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC0);
    }

    return LoadMatrixOpCUDA(rewriter, loc, loadOp, warpSize, mmaOp, kind, numPassesInGroup, offset, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC0);
}

void StoreMatrixOp(ExecutionRuntime runtime, PatternRewriter& rewriter, Location& loc, AffineStoreOp storeOp, const v::MMAOp& mmaOp, Value value, int blockRowOffset, std::pair<mlir::Value, mlir::Value> rowcol, GPUIndexDimension gpuDimsC0)
{
    if (runtime == ExecutionRuntime::ROCM)
    {
        return StoreMatrixOpROCM(rewriter, loc, storeOp, mmaOp, value, blockRowOffset, rowcol, gpuDimsC0);
    }

    assert(blockRowOffset == 0);
    return StoreMatrixOpCUDA(rewriter, loc, storeOp, mmaOp, value, rowcol, gpuDimsC0);
}

mlir::Value ComputeMatrixOp(ExecutionRuntime runtime, PatternRewriter& rewriter, Location& loc, mlir::Value aMmaMatrix, mlir::Value bMmaMatrix, mlir::Value cMmaMatrix, const v::MMAOp& mmaOp, const int cbsz, const int abid)
{
    if (runtime == ExecutionRuntime::ROCM)
    {
        return rewriter.create<MMAComputeSyncOp>(loc, cMmaMatrix.getType(), aMmaMatrix, bMmaMatrix, cMmaMatrix, uint32_t(mmaOp.getShapeType()), cbsz, abid, 0);
    }

    return rewriter.create<gpu::SubgroupMmaComputeOp>(loc, cMmaMatrix.getType(), aMmaMatrix, bMmaMatrix, cMmaMatrix);
}

LogicalResult TensorizeAffineForOpConversion::matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const
{
    auto reportMatchFailure = [&](mlir::Operation* op, std::string message) -> LogicalResult {
        llvm::dbgs() << "While processing " << *op << ". " << message << "\n";
        return rewriter.notifyMatchFailure(op, message);
    };

    if (!HasTensorizationInfo(affineForOp))
    {
        // This isn't an AffineForOp marked for tensorization so just return without modifying it
        return success();
    }

    auto tensorizationInfo = GetTensorizationInfo(affineForOp);

    SmallVector<AffineForOp, 4> loops;
    mlir::getPerfectlyNestedLoops(loops, affineForOp);
    if (loops.size() != 3) // there should be 3 loops in the nest
    {
        return failure();
    }
    for (auto& en : llvm::enumerate(loops))
    {
        auto loop = en.value();
        if (!HasTensorizationInfo(loop))
        {
            return failure();
        }
        if (!loop.hasConstantBounds())
        {
            return failure();
        }
        if (loop.getConstantLowerBound() != 0)
        {
            return failure();
        }
        if (loop.getStep() != 1)
        {
            return failure();
        }
    }

    auto innerLoop = loops[2]; // the innermost loop
    auto innerLoopBodyIter = innerLoop.getBody()->begin();
    auto innerLoopBodyEnd = innerLoop.getBody()->end();

    std::stack<Operation*> opsToErase;

    // 1. load from A matrix
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from A Op");
    }
    auto loadAOp = cast<mlir::AffineLoadOp>(*innerLoopBodyIter);

    // Get indexing info for A
    auto aRank = loadAOp.getMemRefType().getRank();
    if (aRank != 2)
    {
        return reportMatchFailure(loadAOp.getOperation(), "A array has rank != 2");
    }
    if (aRank != loadAOp.getAffineMap().getNumResults())
    {
        return reportMatchFailure(loadAOp.getOperation(), "Failed to match the load from A Op");
    }

    // scan loadAOperands and note which ones are loop vars that refer to GPU block/thread IDs (or are affine expressions of them)
    // get each affine expr from the loadMap and scan it for loops which are bound to GPU block/thread IDs
    // One of the result exprs must depend on a single GPU index dimension, and the other must not depend on any
    auto gpuDimsPerDimA = GetGPUIndexDimensionsInAffineMemOp(loadAOp);
    auto maybeGpuDimA = GetSingleGPUDimension(gpuDimsPerDimA);
    if (!maybeGpuDimA)
    {
        return reportMatchFailure(loadAOp.getOperation(), "Failed to match the load from A Op");
    }

    // Keep A's index dimension
    auto gpuDimA = *maybeGpuDimA;
    if (gpuDimA == GPUIndexDimension::Z)
    {
        return reportMatchFailure(loadAOp.getOperation(), "Failed to match: A op uses GPU Z dimension");
    }

    opsToErase.push(loadAOp);

    // 2. load from B matrix
    innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from B Op");
    }
    auto loadBOp = cast<mlir::AffineLoadOp>(innerLoopBodyIter);

    // Get indexing info for B
    auto bRank = loadBOp.getMemRefType().getRank();
    if (bRank != 2)
    {
        return reportMatchFailure(loadBOp.getOperation(), "B array has rank != 2");
    }

    if (bRank != loadBOp.getAffineMap().getNumResults())
    {
        return reportMatchFailure(loadBOp.getOperation(), "Failed to match the load from B Op");
    }

    // scan loadBOperands and note which ones are loop vars that refer to GPU block/thread IDs (or are affine expressions of them)
    // get each affine expr from the loadMap and scan it for loops which are bound to GPU block/thread IDs
    // One of the result exprs must depend on a single GPU index dimension, and the other must not depend on any
    auto gpuDimsPerDimB = GetGPUIndexDimensionsInAffineMemOp(loadBOp);
    auto maybeGpuDimB = GetSingleGPUDimension(gpuDimsPerDimB);
    if (!maybeGpuDimB)
    {
        return reportMatchFailure(loadBOp.getOperation(), "Failed to match the load from B Op");
    }

    // Keep B's index dimension
    auto gpuDimB = *(maybeGpuDimB);
    if (gpuDimB == GPUIndexDimension::Z)
    {
        return reportMatchFailure(loadBOp.getOperation(), "Failed to match: B op uses GPU Z dimension");
    }

    opsToErase.push(loadBOp);

    if (gpuDimA == gpuDimB)
    {
        return reportMatchFailure(loadBOp.getOperation(), "Failed to match the indexing between A and B Ops");
    }

    // Canonicalize load ops: 'A' load op is the one that uses BlockY/ThreadY, and 'B' load op uses BlockX/ThreadX
    if (gpuDimA == GPUIndexDimension::X)
    {
        std::swap(loadAOp, loadBOp);
        std::swap(gpuDimA, gpuDimB); // unnecessary since these vars aren't used any more
        std::swap(gpuDimsPerDimA, gpuDimsPerDimB);
    }

    assert(gpuDimsPerDimA.size() == 2);
    int loadAGPUIndexPos = gpuDimsPerDimA[0].empty() ? 1 : 0;
    assert(gpuDimsPerDimA[1 - loadAGPUIndexPos].empty());
    assert(gpuDimsPerDimB.size() == 2);
    int loadBGPUIndexPos = gpuDimsPerDimB[0].empty() ? 1 : 0;
    assert(gpuDimsPerDimB[1 - loadBGPUIndexPos].empty());

    // 3. muliply A * B
    (void)innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<v::BinOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the binary A*C multiplication op");
    }
    auto mulAB = cast<v::BinOp>(*innerLoopBodyIter);
    if (mulAB.predicate() != v::BinaryOpPredicate::MUL)
    {
        return reportMatchFailure(mulAB, "Failed to match the multiplication op");
    }
    // Check that the operands for the multiply op are in fact the loads from A and B
    if (!((mulAB.lhs() == loadAOp && mulAB.rhs() == loadBOp) || (mulAB.rhs() == loadAOp && mulAB.lhs() == loadBOp)))
    {
        return reportMatchFailure(mulAB, "Failed to match the multiplication operands");
    }
    opsToErase.push(mulAB);

    // 4. load C
    (void)innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from C Op");
    }
    auto loadCOp = cast<mlir::AffineLoadOp>(*innerLoopBodyIter);

    // Get indexing info for C
    auto cType = loadCOp.getMemRefType();
    auto cRank = cType.getRank();
    auto loadCMap = loadCOp.getAffineMap();
    if (cRank != 2)
    {
        return reportMatchFailure(loadCOp.getOperation(), "C array has rank != 2");
    }
    if (cRank != loadCMap.getNumResults())
    {
        return reportMatchFailure(loadCOp.getOperation(), "Failed to match the load from C Op");
    }

    // scan loadCOperands and note which ones are loop vars that refer to GPU block/thread IDs (or are affine expressions of them)
    // get each affine expr from the loadMap and scan it for loops which are bound to GPU block/thread IDs
    // One of the result exprs must depend on a single GPU index dimension, and the other must not depend on any
    auto gpuDimsPerDimC = GetGPUIndexDimensionsInAffineMemOp(loadCOp);

    if (gpuDimsPerDimC.size() != 2 || gpuDimsPerDimC[0].size() != 1 || gpuDimsPerDimC[1].size() != 1)
    {
        return reportMatchFailure(loadCOp.getOperation(), "Failed to match the load from C Op");
    }

    std::vector<GPUIndexDimension> gpuDimsC = { *GetSingleGPUDimension(gpuDimsPerDimC[0]), *GetSingleGPUDimension(gpuDimsPerDimC[1]) };
    if (gpuDimsC[0] == GPUIndexDimension::Z || gpuDimsC[1] == GPUIndexDimension::Z)
    {
        return reportMatchFailure(loadCOp.getOperation(), "Failed to match: C op uses GPU Z dimension");
    }

    opsToErase.push(loadCOp);

    // 5. add A * B + C
    (void)innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<v::BinOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the binary C accumulation op");
    }
    auto accumC = cast<v::BinOp>(*innerLoopBodyIter);
    if (accumC.predicate() != v::BinaryOpPredicate::ADD)
    {
        return reportMatchFailure(accumC, "Failed to match the accumulation op");
    }
    // Check that the operands for the addition op are in fact (A*B) and the load from C
    if (!((accumC.lhs() == mulAB && accumC.rhs() == loadCOp) || (accumC.rhs() == mulAB && accumC.lhs() == loadCOp)))
    {
        return reportMatchFailure(accumC, "Failed to match the accumulation operands");
    }

    opsToErase.push(accumC);

    // 6. store C
    (void)innerLoopBodyIter++;
    auto storeCOp = cast<mlir::AffineStoreOp>(*innerLoopBodyIter);
    if (innerLoopBodyIter == innerLoopBodyEnd || !storeCOp)
    {
        return reportMatchFailure(affineForOp, "Failed to match the store into C");
    }
    // Check that we are in fact storing the (A*B)+C value, and that we're storing back to the same array
    if (storeCOp.getValueToStore() != accumC || storeCOp.getMemRef() != loadCOp.getMemRef())
    {
        return reportMatchFailure(storeCOp, "Failed to match the store into C");
    }
    // Check that we are in fact storing the (A*B)+C value, and that we're storing back to the same place in the array
    if (!AreSameElement(storeCOp, loadCOp))
    {
        return reportMatchFailure(storeCOp, "Failed to match the store into C");
    }

    opsToErase.push(storeCOp);

    (void)innerLoopBodyIter++;

    // for some reason there sometimes is an extra AffineLoadOp / AffineStoreOp pair being redundantly generated, we need to ignore those
    if (innerLoopBodyIter != innerLoopBodyEnd && isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        auto loadOp = cast<mlir::AffineLoadOp>(*innerLoopBodyIter);
        opsToErase.push(loadOp);
        (void)innerLoopBodyIter++;
        if (innerLoopBodyIter != innerLoopBodyEnd && isa<mlir::AffineStoreOp>(*innerLoopBodyIter))
        {
            auto storeOp = cast<mlir::AffineStoreOp>(*innerLoopBodyIter);
            if (!AreSameElement(loadOp, storeOp))
            {
                return reportMatchFailure(storeOp, "Failed to match extraneous load/store");
            }
            opsToErase.push(storeOp);
            (void)innerLoopBodyIter++;
        }
    }

    // Ignore the yield op at the end
    if (innerLoopBodyIter != innerLoopBodyEnd && isa<mlir::AffineYieldOp>(*innerLoopBodyIter))
    {
        (void)innerLoopBodyIter++;
    }
    if (innerLoopBodyIter != innerLoopBodyEnd)
    {
        LLVM_DEBUG(llvm::dbgs() << "While processing " << *innerLoopBodyIter << ". The store into C was not the last instruction\n";
                   llvm::dbgs() << "affine for : " << *affineForOp << "\n";
                   llvm::dbgs() << "current inst " << *innerLoopBodyIter << "\n");
        return rewriter.notifyMatchFailure(&*innerLoopBodyIter, "The store into C was not the last instruction");
    }

    // A * B + C is valid only if A and B are FP32 and C is FP32 or if A and B are FP16 and C is FP32
    if (!((loadAOp.getMemRefType().getElementType().isF32() &&
           loadBOp.getMemRefType().getElementType().isF32() &&
           storeCOp.getMemRefType().getElementType().isF32()) ||
          (loadAOp.getMemRefType().getElementType().isF16() &&
           loadBOp.getMemRefType().getElementType().isF16() &&
           (storeCOp.getMemRefType().getElementType().isF32() || storeCOp.getMemRefType().getElementType().isF16()))))
    {
        return rewriter.notifyMatchFailure(&*innerLoopBodyIter,
                                           "Invalid data types. "
                                           "A * B + C is valid only if A, B, and C are FP32, or if A and B are FP16 and C is FP32");
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(innerLoop.getBody(), innerLoop.getBody()->getTerminator()->getIterator());

    rewriter.startRootUpdate(affineForOp);
    auto loc = innerLoop.getLoc();
    [[maybe_unused]] auto ctx = rewriter.getContext();

    const v::MMAOp mmaOp(tensorizationInfo.dim);

    // Verify the step sizes of the 'i', 'j', and 'k' loop induction vars
    auto stepsA = GetIndexLoopStepSizesInAffineMemOp(loadAOp);
    auto iStepsA = stepsA[loadAGPUIndexPos];
    auto kStepsA = stepsA[1 - loadAGPUIndexPos];
    if (iStepsA[0] != 1 || kStepsA[0] != 1)
    {
        return reportMatchFailure(loadAOp, "Failed to match load A step sizes");
    }

    auto stepsB = GetIndexLoopStepSizesInAffineMemOp(loadBOp);
    auto kStepsB = stepsB[1 - loadBGPUIndexPos];
    auto jStepsB = stepsB[loadBGPUIndexPos];
    if (kStepsB[0] != 1 || jStepsB[0] != 1)
    {
        return reportMatchFailure(loadBOp, "Failed to match load B step sizes");
    }

    auto stepsC = GetIndexLoopStepSizesInAffineMemOp(loadCOp);
    auto iStepsC = stepsC[gpuDimsC[0] == GPUIndexDimension::Y ? 0 : 1];
    auto jStepsC = stepsC[gpuDimsC[1] == GPUIndexDimension::X ? 1 : 0];
    if (iStepsC[0] != 1 || jStepsC[0] != 1)
    {
        return reportMatchFailure(loadCOp, "Failed to match load C step sizes");
    }

    const auto execRuntime = util::ResolveExecutionRuntime(affineForOp).value();
    if (execRuntime == ExecutionRuntime::ROCM && tensorizationInfo.useStaticOffsets)
    {
        std::vector<uint8_t> threadGroupOffsets = mmaOp.getOffsetMap();
        auto&& [memrefType, dataType] = mmaOp.GetMFMAThreadOffsetMapType(rewriter.getIntegerType(8));
        auto dataAttribute = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(threadGroupOffsets));

        // Create a static offset map (unless one already exists)
        if (auto module = util::CastOrGetParentOfType<v::ValueModuleOp>(affineForOp.getOperation()))
        {
            if (auto existingStaticBuffer = util::FindOpWithSymbolName(v::MFMAThreadBufferMapName, module.getOperation()); !existingStaticBuffer)
            {
                [[maybe_unused]] auto globalOp = util::CreateGlobalBufferOp(rewriter, affineForOp, memrefType, v::MFMAThreadBufferMapName, true, dataAttribute, true, false);
            }
        }
    }

    auto warpSizePair = util::ResolveWarpSize(ExecutionRuntime::ROCM).value();
    const auto warpSize = warpSizePair.first * warpSizePair.second;
    auto&& rowcol = mmaOp.GetThreadBlockOffsets(affineForOp, rewriter, loc);

    const auto totalPasses = tensorizationInfo.numTotalPasses;
    const auto numPassesInGroup = tensorizationInfo.numFusedPasses == -1 ? totalPasses : tensorizationInfo.numFusedPasses;
    if (totalPasses % numPassesInGroup != 0)
    {
        return failure();
    }

    const auto numBlocks = mmaOp.getNumBlocks();
    const auto cbsz = numBlocks / 2;
    const auto numPassGroups = totalPasses / numPassesInGroup;
    const auto passGroupIncrement = numPassesInGroup * mmaOp.getPassIncrements(warpSize);
    const auto schedulingPriority = tensorizationInfo.schedulingPolicy;
    const auto blockRowOffsetIncrements = mmaOp.getLeadingDim() / numBlocks;

    if (schedulingPriority == MMASchedulingPolicy::BlockOrder)
    {
        // Process blocks one by one sequentially
        if (numPassGroups == 1)
        {
            auto int0 = rewriter.create<ConstantIndexOp>(loc, 0);
            auto aMmaMatrix = LoadMatrixOp(execRuntime, rewriter, loc, loadAOp, warpSize, mmaOp, MMAOperandType::A, numPassesInGroup, int0, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC[0]);
            auto bMmaMatrix = LoadMatrixOp(execRuntime, rewriter, loc, loadBOp, warpSize, mmaOp, MMAOperandType::B, numPassesInGroup, int0, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC[0]);
            for (int iBlock = 0, blockRowOffset = 0; iBlock < numBlocks; ++iBlock, blockRowOffset += blockRowOffsetIncrements)
            {
                auto offset = rewriter.create<ConstantIndexOp>(loc, blockRowOffset);
                mlir::Value cMmaMatrix = LoadMatrixOp(execRuntime, rewriter, loc, loadCOp, warpSize, mmaOp, MMAOperandType::Acc, 1, offset, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC[0])[0];
                for (auto&& [matA, matB] : llvm::zip(aMmaMatrix, bMmaMatrix))
                {
                    cMmaMatrix = ComputeMatrixOp(execRuntime, rewriter, loc, matA, matB, cMmaMatrix, mmaOp, cbsz, iBlock);
                }
                StoreMatrixOp(execRuntime, rewriter, loc, storeCOp, mmaOp, cMmaMatrix, blockRowOffset, rowcol, gpuDimsC[0]);
            }
        }
        else
        {
            for (int iBlock = 0, blockRowOffset = 0; iBlock < numBlocks; ++iBlock, blockRowOffset += blockRowOffsetIncrements)
            {
                auto offset = rewriter.create<ConstantIndexOp>(loc, blockRowOffset);
                mlir::Value cMmaMatrix = LoadMatrixOp(execRuntime, rewriter, loc, loadCOp, warpSize, mmaOp, MMAOperandType::Acc, 1, offset, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC[0])[0];
                for (int passGroup = 0, passGroupOffset = 0; passGroup < numPassGroups; ++passGroup, passGroupOffset += passGroupIncrement)
                {
                    auto groupOffset = rewriter.create<ConstantIndexOp>(loc, passGroupOffset);
                    auto aMmaMatrix = LoadMatrixOp(execRuntime, rewriter, loc, loadAOp, warpSize, mmaOp, MMAOperandType::A, numPassesInGroup, groupOffset, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC[0]);
                    auto bMmaMatrix = LoadMatrixOp(execRuntime, rewriter, loc, loadBOp, warpSize, mmaOp, MMAOperandType::B, numPassesInGroup, groupOffset, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC[0]);
                    for (auto&& [matA, matB] : llvm::zip(aMmaMatrix, bMmaMatrix))
                    {
                        cMmaMatrix = ComputeMatrixOp(execRuntime, rewriter, loc, matA, matB, cMmaMatrix, mmaOp, cbsz, iBlock);
                    }
                }
                StoreMatrixOp(execRuntime, rewriter, loc, storeCOp, mmaOp, cMmaMatrix, blockRowOffset, rowcol, gpuDimsC[0]);
            }
        }
    }
    else if (schedulingPriority == MMASchedulingPolicy::PassOrder)
    {
        // First load all the data for C for all the blocks
        std::vector<mlir::Value> cMmaMatrix;
        for (int iBlock = 0, blockRowOffset = 0; iBlock < numBlocks; ++iBlock, blockRowOffset += blockRowOffsetIncrements)
        {
            auto offset = rewriter.create<ConstantIndexOp>(loc, blockRowOffset);
            cMmaMatrix.push_back(LoadMatrixOp(execRuntime, rewriter, loc, loadCOp, warpSize, mmaOp, MMAOperandType::Acc, 1, offset, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC[0])[0]);
        }

        // Load A, B and perform matmul
        for (int passGroup = 0, passGroupOffset = 0; passGroup < numPassGroups; ++passGroup, passGroupOffset += passGroupIncrement)
        {
            auto groupOffset = rewriter.create<ConstantIndexOp>(loc, passGroupOffset);
            auto aMmaMatrix = LoadMatrixOp(execRuntime, rewriter, loc, loadAOp, warpSize, mmaOp, MMAOperandType::A, numPassesInGroup, groupOffset, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC[0]);
            auto bMmaMatrix = LoadMatrixOp(execRuntime, rewriter, loc, loadBOp, warpSize, mmaOp, MMAOperandType::B, numPassesInGroup, groupOffset, rowcol, loadAGPUIndexPos, loadBGPUIndexPos, gpuDimsC[0]);
            for (int iBlock = 0; iBlock < numBlocks; ++iBlock)
            {
                for (auto&& [matA, matB] : llvm::zip(aMmaMatrix, bMmaMatrix))
                {
                    cMmaMatrix[iBlock] = ComputeMatrixOp(execRuntime, rewriter, loc, matA, matB, cMmaMatrix[iBlock], mmaOp, cbsz, iBlock);
                }
            }
        }

        // Lastly, store all the data from C for all the blocks
        for (int iBlock = 0, blockRowOffset = 0; iBlock < numBlocks; ++iBlock, blockRowOffset += blockRowOffsetIncrements)
        {
            StoreMatrixOp(execRuntime, rewriter, loc, storeCOp, mmaOp, cMmaMatrix[iBlock], blockRowOffset, rowcol, gpuDimsC[0]);
        }
    }

    while (!opsToErase.empty())
    {
        auto eraseOp = opsToErase.top();
        if (eraseOp->use_empty())
        {
            rewriter.eraseOp(eraseOp);
        }
        opsToErase.pop();
    }

    for (auto loop : loops)
    {
        // change loop step so that the loop runs once
        loop.setConstantUpperBound(1);

        // remove the tensorization annotation
        RemoveTensorizationInfo(loop);
    }
    RemoveTensorizationInfo(affineForOp);
    rewriter.finalizeRootUpdate(affineForOp);

    return success();
}

LogicalResult ParallelizeAffineForOpConversion::matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const
{
    if (!HasParallelizationInfo(affineForOp))
    {
        // This isn't an AffineForOp marked for parallelization so just return without modifying it
        return success();
    }

    assert(affineForOp.hasConstantLowerBound() && "Parallelized loops must have a constant lower bound");
    assert(affineForOp.hasConstantUpperBound() && "Parallelized loops must have a constant upper bound");

    rewriter.startRootUpdate(affineForOp);

    //  Replace affine.for with affine.parallel, tagged with vectorization info
    //  cf. mlir::affineParallelize() in  mlir\lib\Dialect\Affine\Utils\Utils.cpp
    //  lowering path: affine.parallel -> scf.parallel -> omp.parallel + omp.wsloop
    auto newParallelOp = rewriter.create<mlir::AffineParallelOp>(
        affineForOp.getLoc(), /*resultTypes=*/llvm::None, /*reductionKinds=*/llvm::None, llvm::makeArrayRef(affineForOp.getLowerBoundMap()), affineForOp.getLowerBoundOperands(), llvm::makeArrayRef(affineForOp.getUpperBoundMap()), affineForOp.getUpperBoundOperands(), llvm::makeArrayRef(affineForOp.getStep()));

    // Move the loop block to the new op
    rewriter.inlineRegionBefore(affineForOp.region(), newParallelOp.region(), newParallelOp.region().begin());

    // Unpack the parallelization info into OpenMP dialect attributes
    // cf. mlir\lib\Conversion\SCFToOpenMP\SCFToOpenMP.cpp
    auto parallelizationInfo = GetParallelizationInfo(affineForOp);
    newParallelOp->setAttr(mlir::omp::getNumThreadsAttrName(), rewriter.getI64IntegerAttr(parallelizationInfo.numThreads));

    // Valid clause values: llvm\include\llvm\Frontend\OpenMP\OMP.td
    newParallelOp->setAttr(mlir::omp::getScheduleAttrName(), rewriter.getStringAttr(parallelizationInfo.isDynamicPolicy ? "Dynamic" : "Static"));
    newParallelOp->setAttr(mlir::omp::getProcBindAttrName(), rewriter.getStringAttr("close"));

    rewriter.eraseOp(affineForOp);
    rewriter.finalizeRootUpdate(affineForOp);

    return success();
}

LogicalResult CollapseAffineParallelOpsRewrite::matchAndRewrite(AffineParallelOp affineParallelOp, PatternRewriter& rewriter) const
{
    // Find a pair of perfectly nested ops (i.e. affineParallelOp with a child AffineParallelOp):
    // ...
    //  affine.parallel { <-- parent (do the rewrite at this level)
    //      affine.parallel { <-- child
    //          ...
    //     }
    // }
    // This pattern should successively merge the parent + child pairs (in a pre-order manner)
    // until all perfectly nested parallel ops are collapsed

    // First check if this op has a child AffineParallelOp
    AffineParallelOp childOp;
    affineParallelOp.getLoopBody().walk<WalkOrder::PreOrder>([&childOp, affineParallelOp](Operation* op) {
        auto parentParallelOp = dyn_cast<AffineParallelOp>(op->getParentOp());
        if (parentParallelOp == affineParallelOp)
        {
            childOp = dyn_cast<AffineParallelOp>(op);
        }
        return WalkResult::interrupt(); // TODO: instead of walk, is there a more efficient way?
    });

    // cf. isPerfectlyNested in mlir/lib/Transforms/Utils/LoopUtils.cpp
    // this op's body should be just the child op and the terminator.
    auto hasTwoElements = [](Block* block) {
        auto secondOpIt = std::next(block->begin());
        return secondOpIt != block->end() && &*secondOpIt == &block->back();
    };
    if (!childOp || !hasTwoElements(affineParallelOp.getBody()))
    {
        return failure();
    }

    // Merge the current op with its perfectly nested child. For example:
    //   affine.parallel (%arg3) = (0) to (256) step (64) {
    //      affine.parallel (%arg4) = (0) to (256) {
    // Becomes:
    //   affine.parallel (%arg3, %arg4) = (0, 0) to (256, 256) step (64, 1) {
    //
    // Structure of an affine.parallel:
    //   affine.parallel (%arg4) = (0) to (256) {
    //   } {omp.collapse_val = 1 : i64, omp.num_threads = 4 : i64, omp.proc_bind = "close", omp.schedule_val = "Dynamic"}
    //
    //   Region with 1 blocks:
    //     Block with 1 arguments, 0 successors, and 2 operations
    //       visiting op: 'affine.parallel' with 0 operands and 0 results
    //       8 attributes:
    //        - 'lowerBoundsMap' : 'affine_map<() -> (0)>'
    //        - 'omp.collapse_val' : '1 : i64'
    //        - 'omp.num_threads' : '4 : i64'
    //        - 'omp.proc_bind' : '"close"'
    //        - 'omp.schedule_val' : '"Dynamic"'
    //        - 'reductions' : '[]'
    //        - 'steps' : '[1]'
    //        - 'upperBoundsMap' : 'affine_map<() -> (256)>'
    rewriter.startRootUpdate(affineParallelOp);

    auto lbMap = affineParallelOp.getLowerBoundsValueMap();
    auto ubMap = affineParallelOp.getUpperBoundsValueMap();
    auto childLbMap = childOp.getLowerBoundsValueMap();
    auto childUbMap = childOp.getUpperBoundsValueMap();

    auto mergedUbOperands = llvm::to_vector<4>(llvm::concat<const mlir::Value>(ubMap.getOperands(), childUbMap.getOperands()));
    auto mergedLbOperands = llvm::to_vector<4>(llvm::concat<const mlir::Value>(lbMap.getOperands(), childLbMap.getOperands()));

    auto concatPerLoopAffineMaps = [](mlir::AffineMap a, mlir::AffineMap b) -> auto
    {
        // Extract AffineExprs from (possibly) multi-expression AffineMaps, returning a list of single-expression AffineMaps
        // For example: (256, 256), (128) => [(256), (256), (128)]
        //
        // This will result in one AffineMap per upper- and lower-bound, in this format:
        //    affine.parallel(...) = (AffineMap, AffineMap, ...) to (AffineMap, AffineMap, ...) step (...)
        SmallVector<mlir::AffineMap, 4> flattened;
        flattened.reserve(a.getNumResults() + b.getNumResults());
        auto aExprs = a.getResults();
        std::transform(aExprs.begin(), aExprs.end(), std::back_inserter(flattened), [](mlir::AffineExpr expr) {
            assert(expr.isa<mlir::AffineConstantExpr>() && "All expressions must be constant");
            return mlir::AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, expr); // extract the constant expr
        });
        auto bExprs = b.getResults();
        std::transform(bExprs.begin(), bExprs.end(), std::back_inserter(flattened), [](mlir::AffineExpr expr) {
            assert(expr.isa<mlir::AffineConstantExpr>() && "All expressions must be constant");
            return mlir::AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, expr); // extract the constant expr
        });
        return flattened;
    };
    // Instead of concatAffineMaps which creates a single AffineMap with all upper-bound or lower-bound expressions,
    // we need per-loop-index AffineMaps that describe each collapsed loop within affine.parallel
    auto mergedUbMaps = concatPerLoopAffineMaps(ubMap.getAffineMap(), childUbMap.getAffineMap());
    auto mergedLbMaps = concatPerLoopAffineMaps(lbMap.getAffineMap(), childLbMap.getAffineMap());

    auto mergedSteps = affineParallelOp.getSteps();
    auto chidSteps = childOp.getSteps();
    mergedSteps.append(chidSteps.begin(), chidSteps.end());

    // Create a new merged AffineParallelOp (in-place update is not supported), using childOp's location because
    // we will be taking over the childOp's body (the currentOp's body includes the childOp and therefore is not deep enough)
    auto mergedParallelOp = rewriter.create<mlir::AffineParallelOp>(
        childOp.getLoc(), /*resultTypes=*/llvm::None, /*reductionKinds=*/llvm::None, llvm::makeArrayRef(mergedLbMaps), ValueRange(mergedLbOperands), llvm::makeArrayRef(mergedUbMaps), ValueRange(mergedUbOperands), llvm::makeArrayRef(mergedSteps));

    // Insert new parent arguments (before the child block arguments) and replace their uses
    auto bodyBlock = &affineParallelOp.region().front();
    auto childBodyBlock = &childOp.region().front();
    for (auto it = bodyBlock->args_rbegin(); it != bodyBlock->args_rend(); ++it)
    {
        // insert from back to front in the child
        auto newParentInductionVar = childBodyBlock->insertArgument(childBodyBlock->args_begin(), rewriter.getIndexType());
        (*it).replaceAllUsesWith(newParentInductionVar);
    }

    // Move the child loop block (now with the parent arguments) into the new op
    rewriter.inlineRegionBefore(childOp.region(), mergedParallelOp.region(), mergedParallelOp.region().begin());

    // Copy attributes
    mergedParallelOp->setAttr(mlir::omp::getNumThreadsAttrName(), affineParallelOp->getAttr(mlir::omp::getNumThreadsAttrName()));
    mergedParallelOp->setAttr(mlir::omp::getScheduleAttrName(), affineParallelOp->getAttr(mlir::omp::getScheduleAttrName()));
    mergedParallelOp->setAttr(mlir::omp::getProcBindAttrName(), affineParallelOp->getAttr(mlir::omp::getProcBindAttrName()));

    // Merge and set the collapse attribute
    int64_t collapse = (affineParallelOp->hasAttrOfType<IntegerAttr>(mlir::omp::getCollapseAttrName())) ? affineParallelOp->getAttrOfType<IntegerAttr>(mlir::omp::getCollapseAttrName()).getInt() : 1;
    int64_t childCollapse = (childOp->hasAttrOfType<IntegerAttr>(mlir::omp::getCollapseAttrName())) ? childOp->getAttrOfType<IntegerAttr>(mlir::omp::getCollapseAttrName()).getInt() : 1;
    mergedParallelOp->setAttr(mlir::omp::getCollapseAttrName(), rewriter.getI64IntegerAttr(collapse + childCollapse));

    // Hoist the newOp out of the currentOp (it was in childOp's position)
    mergedParallelOp->moveBefore(affineParallelOp);

    // Erase both current and child op (hopefully safe to do during parallel rewrites, as the child op is within the current op's scope)
    rewriter.eraseOp(childOp);
    rewriter.eraseOp(affineParallelOp);

    rewriter.finalizeRootUpdate(affineParallelOp);
    return success();
}

LogicalResult HoistScalingToCacheReduceRewrite::matchAndRewrite(mlir::AffineStoreOp affineStoreOp, PatternRewriter& rewriter) const
{
    // Find if the cache has a CacheReduceOp within the current scope or a parent scope

    // This assumes that there is one relevant CacheReduceOp associated with this AffineStoreOp,
    // however there can be multiple CacheReduceOps in the graph as a whole in separate branches
    // i.e. the graph is assumed to look like:
    // loops... {
    //     loops... {
    //         loop {
    //             loops... {
    //                 AffineStoreOp(myCache, ...)
    //             }
    //         }
    //         CacheReduceOp(myCache, ...)
    //     }
    //     loops... {
    //         loop {
    //             loops... {
    //                 AffineStoreOp(myCache, ...)
    //             }
    //         }
    //         CacheReduceOp(myCache, ...)
    //     }
    //     ...
    // }

    auto loc = affineStoreOp.getLoc();
    auto valueFuncOp = affineStoreOp->getParentOfType<ValueFuncOp>();
    DominanceInfo domInfo(valueFuncOp);

    auto elementType = affineStoreOp.value().getType();

    std::vector<ActiveBlockCacheReduceOp> activeBlockCacheReduceOps = util::getUsesOfType<ActiveBlockCacheReduceOp>(affineStoreOp.memref());
    std::vector<ActiveElementCacheReduceOp> activeElementCacheReduceOps = util::getUsesOfType<ActiveElementCacheReduceOp>(affineStoreOp.memref());
    assert((activeBlockCacheReduceOps.empty() || activeElementCacheReduceOps.empty()) && "At most one type of cache can be active for a store op in a kernel");
    Operation* targetCacheReduceOpOperation = nullptr;
    for (auto& cacheReduceOp : activeBlockCacheReduceOps)
    {
        auto cacheReduceBlock = cacheReduceOp->getBlock();
        auto ancestorOp = cacheReduceBlock->findAncestorOpInBlock(*affineStoreOp.getOperation());
        if (ancestorOp)
        {
            assert(targetCacheReduceOpOperation == nullptr); // Only expect one cache reduce op to be a candidate
            targetCacheReduceOpOperation = cacheReduceOp;
        }
    }
    for (auto& cacheReduceOp : activeElementCacheReduceOps)
    {
        auto cacheReduceBlock = cacheReduceOp->getBlock();
        auto ancestorOp = cacheReduceBlock->findAncestorOpInBlock(*affineStoreOp.getOperation());
        if (ancestorOp)
        {
            assert(targetCacheReduceOpOperation == nullptr); // Only expect one cache reduce op to be a candidate
            targetCacheReduceOpOperation = cacheReduceOp;
        }
    }
    if (!targetCacheReduceOpOperation)
    {
        // No corresponding cache reduce op for this store so nothing to do here
        return failure();
    }

    // We found a corresponding CacheReduceOp, now check:
    // 1) If the value being stored is a simple accumulation sum (i.e. a sum where one operand is a loaded element from the same position as the store)
    // 2) If the value being added to the loaded data is only a product of other values
    // 3) If any of those values in the product dominate the CacheReduceOp, such as a constant, a function argument, or just something defined at a higher loop layer
    // Then:
    // 1) replace the CacheReduceOp with a new CacheReduceOp that has the chosen values appended to its scaling list
    // 2) replace uses of the multiplication result for each chosen value with the multiplication operand that wasn't the chosen value

    // 1) Check if it's a simple accumulation sum
    auto storeValue = affineStoreOp.value();
    Operation* currentOp = storeValue.getDefiningOp();
    auto currentBinOp = dyn_cast_or_null<v::BinOp>(currentOp);
    if (!currentBinOp)
    {
        // Not a simple accumulation sum if the last operation wasn't a binary op
        return failure();
    }
    if (currentBinOp.predicate() != v::BinaryOpPredicate::ADD)
    {
        // Not a simple accumulation sum if the last operation wasn't an add
        return failure();
    }
    // Check if either the lhs or the rhs defining op is an AffineLoadOp from the same location as the store
    auto lhsOp = currentBinOp.lhs().getDefiningOp();
    auto rhsOp = currentBinOp.rhs().getDefiningOp();
    auto lhsLoadOp = dyn_cast_or_null<mlir::AffineLoadOp>(lhsOp);
    auto rhsLoadOp = dyn_cast_or_null<mlir::AffineLoadOp>(rhsOp);
    if (!lhsLoadOp && !rhsLoadOp)
    {
        // Neither are load ops so it's not a simple accumulate
        return failure();
    }
    auto simpleAccumulateLoadOp = lhsLoadOp ? lhsLoadOp : rhsLoadOp;
    mlir::AffineStoreOpAdaptor storeAdaptor{ affineStoreOp };
    mlir::AffineLoadOpAdaptor loadAdaptor{ simpleAccumulateLoadOp };
    auto storeMemref = storeAdaptor.memref();
    auto loadMemref = loadAdaptor.memref();
    auto storeMap = affineStoreOp.getAffineMapAttr().getValue();
    auto loadMap = simpleAccumulateLoadOp.getAffineMapAttr().getValue();
    auto storeIndices = storeAdaptor.indices();
    auto loadIndices = loadAdaptor.indices();
    if (!(storeMemref == loadMemref && storeMap == loadMap && storeIndices == loadIndices))
    {
        // The load and store are targeting different elements so it's not a simple accumulate
        return failure();
    }

    // At this point we know it's a simple accumulation sum, so now
    // 2) check if the value being added is a product of other values
    auto currentVal = lhsLoadOp ? currentBinOp.rhs() : currentBinOp.lhs();
    std::stack<std::pair<Operation*, mlir::Value>> productValuesToFollow;
    productValuesToFollow.push(std::make_pair(currentBinOp, currentVal));
    std::vector<std::pair<Operation*, mlir::Value>> baseProductOpAndOperands;
    while (!productValuesToFollow.empty())
    {
        std::pair<Operation*, mlir::Value> currentValAndBinOpParent = productValuesToFollow.top();
        currentVal = currentValAndBinOpParent.second;
        productValuesToFollow.pop();
        currentBinOp = dyn_cast_or_null<v::BinOp>(currentVal.getDefiningOp());
        if (!currentBinOp)
        {
            // Not a bin op so assume it's the base op for the value
            baseProductOpAndOperands.emplace_back(std::make_pair(currentValAndBinOpParent.first, currentVal));
        }
        else if (currentBinOp.predicate() != v::BinaryOpPredicate::MUL)
        {
            // Not just a product of other values
            return failure();
        }
        else
        {
            // Multiplication bin op, so add the lhs and rhs operands to the stack for examination
            productValuesToFollow.push(std::make_pair(currentBinOp, currentBinOp.lhs()));
            productValuesToFollow.push(std::make_pair(currentBinOp, currentBinOp.rhs()));
        }
    }
    // At this point we know it's a product of other values, so now
    // 3) Check if any of those values dominate the target cache reduce op
    // Remove elements from the vector if the operand defining op doesn't dominate the target cache reduce op
    std::vector<std::pair<Operation*, mlir::Value>> opsAndHoistableOperands;
    for (auto& opAndOperandValue : baseProductOpAndOperands)
    {
        // If this value is a block argument to a block containing the cache reduce op
        // or if this value's defining op dominates the cache reduce op, then it is
        // a hoistable scale value
        bool hoistable = false;
        if (opAndOperandValue.second.isa<mlir::BlockArgument>())
        {
            auto blockArg = opAndOperandValue.second.cast<mlir::BlockArgument>();
            mlir::Block* block = blockArg.getOwner();
            hoistable = block->findAncestorOpInBlock(*targetCacheReduceOpOperation) != nullptr;
        }
        else
        {
            hoistable = domInfo.dominates(opAndOperandValue.second.getDefiningOp(), targetCacheReduceOpOperation);
        }
        if (hoistable)
        {
            opsAndHoistableOperands.push_back(opAndOperandValue);
        }
    }
    // Make a constant 1 value to replace in the product and depend on other optimizations to recognize that 1 * x = x
    mlir::Value constantOne;
    {
        mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(&valueFuncOp.body().front());
        constantOne = rewriter.create<mlir::ConstantOp>(loc, util::GetOneAttr(rewriter, elementType));
    }

    std::vector<mlir::Value> scaleValues;
    for (auto& opAndOperand : opsAndHoistableOperands)
    {
        opAndOperand.first->replaceUsesOfWith(opAndOperand.second, constantOne);
        scaleValues.push_back(opAndOperand.second);
    }
    // If this cache reduce op already has scale values, don't bother setting more
    // In the event of a boundary condition that exists between the cache reduce op
    // and the affine store op, we'll wind up with multiple kernels per cache reduce op
    // and once we've hoisted a set of constants from one kernel, we don't want to
    // duplicate the scaling with another, but we still want to replace values
    // with 1's in the kernel

    if (activeBlockCacheReduceOps.empty())
    {
        auto activeElementCacheReduceOp = dyn_cast<ActiveElementCacheReduceOp>(targetCacheReduceOpOperation);
        ActiveElementCacheReduceOpAdaptor cacheReduceOpAdaptor{ activeElementCacheReduceOp };
        mlir::ValueRange scaleValuesRange = cacheReduceOpAdaptor.scaleValues();
        if (scaleValuesRange.empty())
        {
            mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(activeElementCacheReduceOp);
            auto cacheRegionIndexRanges = util::ArrayAttrToVector<IndexRange, IndexRangeAttr>(activeElementCacheReduceOp.cacheRegionRelevantIndexRanges(),
                                                                                              [](const IndexRangeAttr& indexRangeAttr) {
                                                                                                  return indexRangeAttr.getValue();
                                                                                              });

            auto cacheRegionBaseIndices = util::ArrayAttrToVector<std::vector<Index>, mlir::ArrayAttr>(
                activeElementCacheReduceOp.cacheRegionBaseIndices(),
                util::ConvertArrayAttrToIndexVector);

            rewriter.replaceOpWithNewOp<ActiveElementCacheReduceOp>(activeElementCacheReduceOp,
                                                                    activeElementCacheReduceOp.srcCache(),
                                                                    activeElementCacheReduceOp.dst(),
                                                                    cacheReduceOpAdaptor.externalRelevantIndices(),
                                                                    cacheRegionIndexRanges,
                                                                    cacheRegionBaseIndices,
                                                                    activeElementCacheReduceOp.relevantIndicesToSrcCacheMap(),
                                                                    activeElementCacheReduceOp.relevantIndicesToDstMap(),
                                                                    scaleValues);
        }
    }
    else if (activeElementCacheReduceOps.empty())
    {
        auto activeBlockCacheReduceOp = dyn_cast<ActiveBlockCacheReduceOp>(targetCacheReduceOpOperation);
        ActiveBlockCacheReduceOpAdaptor cacheReduceOpAdaptor{ activeBlockCacheReduceOp };
        mlir::ValueRange scaleValuesRange = cacheReduceOpAdaptor.scaleValues();
        if (scaleValuesRange.empty())
        {
            mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(activeBlockCacheReduceOp);

            rewriter.replaceOpWithNewOp<ActiveBlockCacheReduceOp>(activeBlockCacheReduceOp,
                                                                  activeBlockCacheReduceOp.array(),
                                                                  activeBlockCacheReduceOp.cache(),
                                                                  cacheReduceOpAdaptor.lbOperands(),
                                                                  cacheReduceOpAdaptor.ubOperands(),
                                                                  cacheReduceOpAdaptor.lbMaps(),
                                                                  cacheReduceOpAdaptor.ubMaps(),
                                                                  activeBlockCacheReduceOp.activeBlockToCacheMap(),
                                                                  scaleValues,
                                                                  activeBlockCacheReduceOp.activeBlockTag(),
                                                                  activeBlockCacheReduceOp.thrifty(),
                                                                  activeBlockCacheReduceOp.vectorizationInfoAttr());
        }
    }

    return success();
}

bool AncestorOpContainsAttrOfName(Operation* op, const mlir::StringRef& name)
{
    while (op != nullptr)
    {
        if (op->getAttr(name) != nullptr)
        {
            return true;
        }
        op = op->getParentOp();
    }
    return false;
}

LogicalResult OutOfBoundsLoadRewriteCommon(mlir::AffineLoadOp affineLoadOp, PatternRewriter& rewriter)
{
    if (IsBoundsChecked(affineLoadOp))
    {
        return success();
    }
    auto loc = affineLoadOp.getLoc();
    mlir::AffineLoadOp::Adaptor adaptor{ affineLoadOp };

    if (HasOutOfBoundsAccess(affineLoadOp, loc))
    {
        // This load has a potential out-of-bounds access, so replace it with a conditional load

        auto accessMapAttr = affineLoadOp.getAffineMapAttr();
        auto accessMap = accessMapAttr.getValue();
        auto loadSrc = affineLoadOp.memref();
        auto loadSrcType = loadSrc.getType();
        assert(loadSrcType.isa<mlir::MemRefType>());
        auto memRefType = loadSrcType.cast<mlir::MemRefType>();

        auto loadResultType = affineLoadOp.result().getType();

        std::vector<mlir::AffineExpr> constraintExprs;
        constraintExprs.reserve(accessMap.getNumResults() * 2); // One lower bound and one upper bound check per src dimension
        std::vector<mlir::Value> accessIndices(adaptor.indices().begin(), adaptor.indices().end());
        auto resolvedAccessIndices = util::MultiDimAffineApply(rewriter, loc, accessMap, accessIndices);
        SmallVector<bool, 4> constraintEqFlags(accessMap.getNumResults() * 2, false);
        for (size_t srcDim = 0; srcDim < accessMap.getNumResults(); srcDim++)
        {
            // Lower bound check
            constraintExprs.push_back(rewriter.getAffineDimExpr(srcDim)); // Will check whether this index is >= 0

            // Upper bound check
            constraintExprs.push_back(memRefType.getDimSize(srcDim) - rewriter.getAffineDimExpr(srcDim) - rewriter.getAffineConstantExpr(1)); // Will check whether (this dimSize - this index - 1) >= 0 (note: -1 since we're doing a >= check with 0-based indices)
        }

        std::vector<int64_t> tmpBufferShape{ 1 }; // only one element of type loadResultType
        mlir::MemRefType tmpElementType;
        std::optional<v::ExecutionTarget> execTargetOpt = util::ResolveExecutionTarget(affineLoadOp);
        assert(execTargetOpt.has_value());
        auto execTarget = *execTargetOpt;
        mlir::Value tmpBuffer;
        if (execTarget == v::ExecutionTarget::GPU)
        {
            tmpElementType = mlir::MemRefType::get(tmpBufferShape, loadResultType, {}, static_cast<unsigned>(v::MemorySpace::Private));
            tmpBuffer = rewriter.create<v::AllocOp>(loc, tmpElementType, llvm::None);
        }
        else
        {
            tmpElementType = mlir::MemRefType::get(tmpBufferShape, loadResultType);
            tmpBuffer = rewriter.create<mlir::memref::AllocaOp>(loc, tmpElementType, mlir::ValueRange{}, rewriter.getI64IntegerAttr(AVX2Alignment));
        }

        auto zeroIndex = rewriter.create<mlir::ConstantIndexOp>(loc, 0);

        auto srcBoundsCheckSet = mlir::IntegerSet::get(resolvedAccessIndices.size(), 0, constraintExprs, constraintEqFlags);
        auto ifOp = rewriter.create<mlir::AffineIfOp>(loc, srcBoundsCheckSet, ValueRange{ resolvedAccessIndices }, true); // true indicating we want an "else" region

        auto thenBuilder = ifOp.getThenBodyBuilder();
        auto newLoadOp = thenBuilder.create<mlir::AffineLoadOp>(loc, loadSrc, accessMap, accessIndices);
        SetBoundsChecked(thenBuilder, newLoadOp);

        auto thenStoreOp = thenBuilder.create<mlir::memref::StoreOp>(loc, newLoadOp.getResult(), tmpBuffer, ValueRange{ zeroIndex });
        SetBoundsChecked(thenBuilder, thenStoreOp);

        auto elseBuilder = ifOp.getElseBodyBuilder();
        // TODO : support user-specified padding value rather than always using 0
        auto constantZero = elseBuilder.create<mlir::ConstantOp>(loc, elseBuilder.getZeroAttr(loadResultType));
        auto elseStoreOp = elseBuilder.create<mlir::memref::StoreOp>(loc, constantZero.getResult(), tmpBuffer, ValueRange{ zeroIndex });
        SetBoundsChecked(elseBuilder, elseStoreOp);

        auto tmpSlotLoad = rewriter.create<mlir::memref::LoadOp>(loc, tmpBuffer, ValueRange{ zeroIndex });
        SetBoundsChecked(rewriter, tmpSlotLoad);

        affineLoadOp.replaceAllUsesWith(tmpSlotLoad.getResult());
        rewriter.eraseOp(affineLoadOp);
    }

    return success();
}

LogicalResult OutOfBoundsLoadRewrite::matchAndRewrite(mlir::memref::LoadOp loadOp, PatternRewriter& rewriter) const
{
    // Only check for out-of-bounds-accesses inside of ops that are marked for bounds checking
    if (!AncestorOpContainsAttrOfName(loadOp, AccessBoundsCheckAttrName))
    {
        return success();
    }

    if (IsBoundsChecked(loadOp))
    {
        return success();
    }
    // Convert std.load to affine.load with an identity map
    auto loc = loadOp.getLoc();
    mlir::memref::LoadOp::Adaptor adaptor{ loadOp };
    auto memRefType = adaptor.memref().getType().cast<mlir::MemRefType>();
    auto affineLoadOp = rewriter.create<mlir::AffineLoadOp>(loc, adaptor.memref(), rewriter.getMultiDimIdentityMap(memRefType.getRank()), adaptor.indices());
    loadOp.replaceAllUsesWith(affineLoadOp.getResult());
    auto result = OutOfBoundsLoadRewriteCommon(affineLoadOp, rewriter);
    rewriter.eraseOp(loadOp);
    return result;
}

LogicalResult OutOfBoundsAffineLoadRewrite::matchAndRewrite(mlir::AffineLoadOp affineLoadOp, PatternRewriter& rewriter) const
{
    // Only check for out-of-bounds-accesses inside of ops that are marked for bounds checking
    if (!AncestorOpContainsAttrOfName(affineLoadOp, AccessBoundsCheckAttrName))
    {
        return success();
    }

    return OutOfBoundsLoadRewriteCommon(affineLoadOp, rewriter);
}

LogicalResult OutOfBoundsStoreRewriteCommon(mlir::AffineStoreOp affineStoreOp, PatternRewriter& rewriter)
{
    if (IsBoundsChecked(affineStoreOp))
    {
        return success();
    }

    auto loc = affineStoreOp.getLoc();
    mlir::AffineStoreOp::Adaptor adaptor{ affineStoreOp };

    if (HasOutOfBoundsAccess(affineStoreOp, loc))
    {
        // This store has a potential out-of-bounds access, so replace it with a conditional store

        auto accessMapAttr = affineStoreOp.getAffineMapAttr();
        auto accessMap = accessMapAttr.getValue();
        auto storeDst = affineStoreOp.memref();
        auto storeDstType = storeDst.getType();
        assert(storeDstType.isa<mlir::MemRefType>());
        auto memRefType = storeDstType.cast<mlir::MemRefType>();

        // TODO : de-dupe affine.if constraint code with load case
        std::vector<mlir::AffineExpr> constraintExprs;
        constraintExprs.reserve(accessMap.getNumResults() * 2); // One lower bound and one upper bound check per src dimension
        std::vector<mlir::Value> accessIndices(adaptor.indices().begin(), adaptor.indices().end());
        auto resolvedAccessIndices = util::MultiDimAffineApply(rewriter, loc, accessMap, accessIndices);
        SmallVector<bool, 4> constraintEqFlags(accessMap.getNumResults() * 2, false);
        for (size_t srcDim = 0; srcDim < accessMap.getNumResults(); srcDim++)
        {
            // Lower bound check
            constraintExprs.push_back(rewriter.getAffineDimExpr(srcDim)); // Will check whether this index is >= 0

            // Upper bound check
            constraintExprs.push_back(memRefType.getDimSize(srcDim) - rewriter.getAffineDimExpr(srcDim) - rewriter.getAffineConstantExpr(1)); // Will check whether (this dimSize - this index - 1) >= 0 (note: -1 since we're doing a >= check with 0-based indices)
        }

        auto srcBoundsCheckSet = mlir::IntegerSet::get(resolvedAccessIndices.size(), 0, constraintExprs, constraintEqFlags);
        auto ifOp = rewriter.create<mlir::AffineIfOp>(loc, srcBoundsCheckSet, ValueRange{ resolvedAccessIndices }, true); // true indicating we want an "else" region

        auto thenBuilder = ifOp.getThenBodyBuilder();
        auto newStoreOp = thenBuilder.create<mlir::AffineStoreOp>(loc, affineStoreOp.value(), affineStoreOp.memref(), accessMap, accessIndices);
        SetBoundsChecked(thenBuilder, newStoreOp);

        rewriter.eraseOp(affineStoreOp);
    }

    return success();
}

LogicalResult OutOfBoundsStoreRewrite::matchAndRewrite(mlir::memref::StoreOp storeOp, PatternRewriter& rewriter) const
{
    // Only check for out-of-bounds-accesses inside of ops that are marked for bounds checking
    if (!AncestorOpContainsAttrOfName(storeOp, AccessBoundsCheckAttrName))
    {
        return success();
    }

    if (IsBoundsChecked(storeOp))
    {
        return success();
    }
    // Convert std.store to affine.store with an identity map
    auto loc = storeOp.getLoc();
    mlir::memref::StoreOp::Adaptor adaptor{ storeOp };
    auto memRefType = adaptor.memref().getType().cast<mlir::MemRefType>();
    auto affineStoreOp = rewriter.create<mlir::AffineStoreOp>(loc, adaptor.value(), adaptor.memref(), rewriter.getMultiDimIdentityMap(memRefType.getRank()), adaptor.indices());
    auto result = OutOfBoundsStoreRewriteCommon(affineStoreOp, rewriter);
    rewriter.eraseOp(storeOp);
    return result;
}

LogicalResult OutOfBoundsAffineStoreRewrite::matchAndRewrite(mlir::AffineStoreOp affineStoreOp, PatternRewriter& rewriter) const
{
    // Only check for out-of-bounds-accesses inside of ops that are marked for bounds checking
    if (!AncestorOpContainsAttrOfName(affineStoreOp, AccessBoundsCheckAttrName))
    {
        return success();
    }

    return OutOfBoundsStoreRewriteCommon(affineStoreOp, rewriter);
}

template <typename OpType>
LogicalResult ConvertStoreToAffine(PatternRewriter& rewriter, OpType op)
{
    // Convert store ops to affine store ops with an identity map
    typename OpType::Adaptor adaptor{ op };
    auto memRefType = adaptor.memref().getType().template cast<mlir::MemRefType>();
    rewriter.create<mlir::AffineStoreOp>(op.getLoc(), adaptor.value(), adaptor.memref(), rewriter.getMultiDimIdentityMap(memRefType.getRank()), adaptor.indices());
    rewriter.eraseOp(op);
    return success();
}

template <typename OpType>
LogicalResult ConvertLoadToAffine(PatternRewriter& rewriter, OpType op)
{
    // Convert load ops to affine load ops with an identity map
    typename OpType::Adaptor adaptor{ op };
    auto memRefType = adaptor.memref().getType().template cast<mlir::MemRefType>();
    rewriter.replaceOpWithNewOp<mlir::AffineLoadOp>(op, adaptor.memref(), rewriter.getMultiDimIdentityMap(memRefType.getRank()), adaptor.indices());
    return success();
}

LogicalResult ConvertLoadsToAffineRewrite::matchAndRewrite(mlir::memref::LoadOp loadOp, PatternRewriter& rewriter) const
{
    return ConvertLoadToAffine(rewriter, loadOp);
}

LogicalResult ConvertStoresToAffineRewrite::matchAndRewrite(mlir::memref::StoreOp storeOp, PatternRewriter& rewriter) const
{
    return ConvertStoreToAffine(rewriter, storeOp);
}

LogicalResult ConvertValueLoadsToAffineRewrite::matchAndRewrite(v::LoadOp loadOp, PatternRewriter& rewriter) const
{
    return ConvertLoadToAffine(rewriter, loadOp);
}

LogicalResult ConvertValueStoresToAffineRewrite::matchAndRewrite(v::StoreOp storeOp, PatternRewriter& rewriter) const
{
    return ConvertStoreToAffine(rewriter, storeOp);
}

LogicalResult DelayedMappingRegionOpRewrite::matchAndRewrite(DelayedMappingRegionOp mappingRegionOp, PatternRewriter& rewriter) const
{
    auto fromValue = mappingRegionOp.from();
    auto toValue = mappingRegionOp.to();
    mappingRegionOp.region().walk([&](mlir::Operation* op) {
        op->replaceUsesOfWith(fromValue, toValue);
    });
    util::InlineAllRegionOpsBeforeOp(rewriter, mappingRegionOp.region(), mappingRegionOp);
    rewriter.eraseOp(mappingRegionOp);
    return success();
}

// Returns the second loop, which goes from [n, end), and changes the given loop to go from [begin, n)
mlir::AffineForOp SegmentLoopAtIteration(mlir::AffineForOp forOp, int64_t n)
{
    // To segment the loop into two loops at the n'th iteration
    // 1) Compute the loop IV value at the n'th iteration
    // 2) Clone the loop
    // 3) Update the original loop's end value to be the n'th iteration value
    // 4) Update the cloned loop's begin value to be the n'th iteration value

    // Position a builder in the block containing this forOp just after the loop
    auto iter = forOp->getIterator();
    iter++;
    auto loopParentBlock = forOp->getBlock();
    mlir::OpBuilder builder(loopParentBlock, iter);

    auto constantTripCountOpt = mlir::getConstantTripCount(forOp);

    assert(constantTripCountOpt.hasValue() && "AffineForOps in Accera loop nests must have constant trip counts");
    auto constantTripCount = constantTripCountOpt.getValue();
    if ((int64_t)constantTripCount < n)
    {
        // Can't unswitch more iterations than this loop has, so don't bother unswitching
        return nullptr;
    }

    assert(forOp.hasConstantBounds() && "Only constant-bounded AffineForOps are supported for unswitching");

    auto nthIterValue = forOp.getConstantLowerBound() + (forOp.getStep() * n);

    auto segmentedSecondLoop = mlir::dyn_cast<mlir::AffineForOp>(builder.clone(*(forOp.getOperation())));
    forOp.setConstantUpperBound(nthIterValue);
    segmentedSecondLoop.setConstantLowerBound(nthIterValue);

    return segmentedSecondLoop;
}

LogicalResult LoopUnswitchingOpRewrite::matchAndRewrite(mlir::AffineForOp forOp, PatternRewriter& rewriter) const
{
    if (forOp->hasAttrOfType<IntegerAttr>(UnswitchSuffixItersName) ||
        forOp->hasAttrOfType<IntegerAttr>(UnswitchPrefixItersName))
    {
        // Unswitch the last n iterations first if a suffix unswitch is desired
        if (auto unswitchSuffix = forOp->getAttrOfType<IntegerAttr>(UnswitchSuffixItersName))
        {
            forOp->removeAttr(UnswitchSuffixItersName);

            auto constantTripCountOpt = mlir::getConstantTripCount(forOp);
            assert(constantTripCountOpt.hasValue() && "AffineForOps in Accera loop nests must have constant trip counts");
            auto constantTripCount = constantTripCountOpt.getValue();

            int64_t iter = constantTripCount - unswitchSuffix.getInt();
            [[maybe_unused]] auto secondLoop = SegmentLoopAtIteration(forOp, iter);
        }

        // Now unswitch the first n iterations if a prefix unswitch is desired. Note: requesting both is also supported
        if (auto unswitchPrefix = forOp->getAttrOfType<IntegerAttr>(UnswitchPrefixItersName))
        {
            forOp->removeAttr(UnswitchPrefixItersName);
            [[maybe_unused]] auto secondLoop = SegmentLoopAtIteration(forOp, unswitchPrefix.getInt());
        }
    }
    return success();
}

void ExecutionPlanCacheRegionLoweringPass::runOnOperation()
{
    auto operation = getOperation();
    ConversionTarget target(getContext());

    target.addLegalDialect<ValueDialect,
                           memref::MemRefDialect,
                           mlir::AffineDialect,
                           mlir::StandardOpsDialect,
                           ExecutionPlanDialect>();
    target.addIllegalOp<BeginCacheMappingOp>();
    target.addIllegalOp<EndCacheMappingOp>();
    target.addIllegalOp<BeginCacheRegionOp>();
    target.addIllegalOp<EndCacheRegionOp>();

    OwningRewritePatternList patterns(&getContext());
    accera::transforms::executionPlan::populateExecutionPlanCacheRegionPatterns(patterns);

    (void)applyPatternsAndFoldGreedily(operation, std::move(patterns));
}

void ExecutionPlanVectorizationPass::runOnOperation()
{
    auto operation = getOperation();
    mlir::OpBuilder builder(operation);
    ConversionTarget target(getContext());

    target.addLegalDialect<ValueDialect,
                           memref::MemRefDialect,
                           mlir::AffineDialect,
                           mlir::StandardOpsDialect,
                           ExecutionPlanDialect>();
    target.addDynamicallyLegalOp<AffineForOp>([&](AffineForOp op) {
        // An AffineForOp is legal if it does not have the ExecutionPlan vectorize attributes
        return !HasVectorizationInfo(op);
    });

    OwningRewritePatternList patterns(&getContext());
    accera::transforms::executionPlan::populateExecutionPlanVectorizePatterns(printVecOpDetails, patterns);

    (void)applyPatternsAndFoldGreedily(operation, std::move(patterns));
}

void ExecutionPlanParallelizationPass::runOnOperation()
{
    auto operation = getOperation();
    mlir::OpBuilder builder(operation);
    ConversionTarget target(getContext());

    target.addLegalDialect<ValueDialect,
                           memref::MemRefDialect,
                           mlir::AffineDialect,
                           mlir::StandardOpsDialect,
                           ExecutionPlanDialect>();
    target.addDynamicallyLegalOp<AffineForOp>([&](AffineForOp op) {
        // An AffineForOp is legal if it does not have the ExecutionPlan parallelize attributes
        return !HasParallelizationInfo(op);
    });

    OwningRewritePatternList patterns(&getContext());
    accera::transforms::executionPlan::populateExecutionPlanParallelizePatterns(patterns);

    (void)applyPatternsAndFoldGreedily(operation, std::move(patterns));
}

void ExecutionPlanTensorizationPass::runOnOperation()
{
    auto* ctx = &getContext();
    auto operation = getOperation();

    OwningRewritePatternList patterns(ctx);
    accera::transforms::executionPlan::populateExecutionPlanTensorizePatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(operation, std::move(patterns))))
        return signalPassFailure();
}

void ExecutionPlanMakeCacheLoweringPass::runOnFunction()
{
    ConversionTarget target(getContext());

    target.addLegalDialect<ValueDialect,
                           memref::MemRefDialect,
                           mlir::AffineDialect,
                           mlir::StandardOpsDialect,
                           ExecutionPlanDialect>();
    target.addIllegalOp<MakeCacheOp>();

    OwningRewritePatternList patterns(&getContext());
    accera::transforms::executionPlan::populateExecutionPlanMakeCachePatterns(patterns);

    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
        signalPassFailure();
}

void ExecutionPlanCopyReduceLoweringPass::runOnOperation()
{
    auto operation = getOperation();
    ConversionTarget target(getContext());

    target.addLegalDialect<ValueDialect,
                           LoopNestDialect,
                           memref::MemRefDialect,
                           mlir::AffineDialect,
                           mlir::StandardOpsDialect,
                           ExecutionPlanDialect>();
    target.addIllegalOp<ActiveElementCacheCopyOp>();
    target.addIllegalOp<ActiveBlockCacheCopyOp>();
    target.addIllegalOp<ActiveElementCacheReduceOp>();
    target.addIllegalOp<ActiveBlockCacheReduceOp>();
    target.addIllegalOp<CacheZeroOp>();

    OwningRewritePatternList patterns(&getContext());
    accera::transforms::executionPlan::populateExecutionPlanCopyReducePatterns(patterns);

    (void)applyPatternsAndFoldGreedily(operation, std::move(patterns));
}

void ExecutionPlanScaleHoistingPass::runOnFunction()
{
    ConversionTarget target(getContext());

    OwningRewritePatternList patterns(&getContext());
    accera::transforms::executionPlan::populateExecutionPlanScaleHoistingPatterns(patterns);

    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
        signalPassFailure();
}

void OutOfBoundsAccessHandlingPass::runOnFunction()
{
    ConversionTarget target(getContext());

    OwningRewritePatternList patterns(&getContext());
    accera::transforms::executionPlan::populateOutOfBoundsAccessHandlingPatterns(patterns);

    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
        signalPassFailure();
}

namespace accera::transforms::executionPlan
{

std::unique_ptr<Pass> createExecutionPlanMakeCachePass()
{
    return std::make_unique<ExecutionPlanMakeCacheLoweringPass>();
}

std::unique_ptr<Pass> createExecutionPlanCopyReducePass()
{
    return std::make_unique<ExecutionPlanCopyReduceLoweringPass>();
}

std::unique_ptr<Pass> createExecutionPlanCacheRegionLoweringPass()
{
    return std::make_unique<ExecutionPlanCacheRegionLoweringPass>();
}

std::unique_ptr<Pass> createExecutionPlanVectorizationPass()
{
    return std::make_unique<ExecutionPlanVectorizationPass>();
}

std::unique_ptr<Pass> createExecutionPlanParallelizationPass()
{
    return std::make_unique<ExecutionPlanParallelizationPass>();
}

std::unique_ptr<mlir::Pass> createExecutionPlanTensorizationPass()
{
    return std::make_unique<ExecutionPlanTensorizationPass>();
}

std::unique_ptr<mlir::Pass> createExecutionPlanScaleHoistingPass()
{
    return std::make_unique<ExecutionPlanScaleHoistingPass>();
}

std::unique_ptr<mlir::Pass> createOutOfBoundsAccessHandlingPass()
{
    return std::make_unique<OutOfBoundsAccessHandlingPass>();
}

void populateExecutionPlanMakeCachePatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<MakeCacheOpLowering>(patterns.getContext());
}

void populateExecutionPlanThriftyCachePatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<ThriftyCacheMultiCopyOpRewrite>(patterns.getContext());
    patterns.insert<ThriftyCacheCopyOpRewrite>(patterns.getContext());
    patterns.insert<ThriftyCacheReduceOpRewrite>(patterns.getContext());
}

void populateExecutionPlanMultiCachePatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<MultiCacheCopyOpRewrite>(patterns.getContext());
}

void populateExecutionPlanCopyReducePatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<ActiveElementCacheCopyOpRewrite,
                    ActiveBlockCacheCopyOpRewrite,
                    ActiveElementCacheReduceOpRewrite,
                    ActiveBlockCacheReduceOpRewrite,
                    CacheZeroOpRewrite>(patterns.getContext());
}

void populateExecutionPlanDelayedMappingPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<DelayedMappingRegionOpRewrite>(patterns.getContext());
}

void populateExecutionPlanLoopUnswitchingPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<LoopUnswitchingOpRewrite>(patterns.getContext());
}

void populateExecutionPlanMaxElementCacheRegionPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<MaxElementCacheRegionOpRewrite>(patterns.getContext());
}

void populateExecutionPlanCacheRegionHoistingPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<HoistCacheRegionOpsRewrite>(patterns.getContext());
}

void populateExecutionPlanCacheRegionMergingPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<MergeCacheRegionOpsRewrite>(patterns.getContext());
}

void populateExecutionPlanCacheRegionPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<BeginCacheRegionOpRewrite>(patterns.getContext());
}

void populateExecutionPlanCacheMappingPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<BeginCacheMappingOpRewrite>(patterns.getContext());
}

void populateExecutionPlanAdjustHierarchicalCacheRegionPositionPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<AdjustHierarchicalCacheRegionPositionRewrite>(patterns.getContext());
}

void populateExecutionPlanAdjustCacheMappingPositionPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<AdjustCacheMappingPositionRewrite>(patterns.getContext());
}

void populateExecutionPlanVectorizePatterns(bool printVectorizationDetails, mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<VectorizeAffineForOpConversion,
                    InPlaceUnrollAffineForOpConversion>(patterns.getContext(), printVectorizationDetails);
}

void populateExecutionPlanTensorizePatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<TensorizeAffineForOpConversion>(patterns.getContext());
}

void populateExecutionPlanParallelizePatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<ParallelizeAffineForOpConversion,
                    CollapseAffineParallelOpsRewrite>(patterns.getContext());
}

void populateExecutionPlanScaleHoistingPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<HoistScalingToCacheReduceRewrite>(patterns.getContext());
}

void populateOutOfBoundsAccessHandlingPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<OutOfBoundsLoadRewrite,
                    OutOfBoundsStoreRewrite,
                    OutOfBoundsAffineLoadRewrite,
                    OutOfBoundsAffineStoreRewrite>(patterns.getContext());
}

void populateConvergeLoadStoresPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<ConvertLoadsToAffineRewrite,
                    ConvertStoresToAffineRewrite,
                    ConvertValueLoadsToAffineRewrite,
                    ConvertValueStoresToAffineRewrite>(patterns.getContext());
}

} // namespace accera::transforms::executionPlan
