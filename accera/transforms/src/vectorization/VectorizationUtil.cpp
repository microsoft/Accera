////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vectorization/VectorizationUtil.h"
#include "vectorization/VectorizedOp.h"

#include <ir/include/IRUtil.h>
#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/value/ValueDialect.h>

#include <utilities/include/TypeTraits.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Utils/VectorUtils.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <stack>
#include <stdexcept>

using namespace mlir;

using namespace accera::utilities;
namespace v = accera::ir::value;

#define DEBUG_TYPE "vectorization-util"

// TODO : plumb through a sufficient target enum / bitmap so we can dynamically enable/disable vpmaddwd and other pattern matchers
#define MATCH_VPMADDWD_INTRINSIC 1

namespace accera::transforms
{

std::optional<VectorizedOp> VectorizedOpMap::Lookup(mlir::Operation* op) const
{
    return Lookup(static_cast<void*>(op));
}

std::optional<VectorizedOp> VectorizedOpMap::Lookup(mlir::Value value) const
{
    return Lookup(value.getAsOpaquePointer());
}

std::optional<VectorizedOp> VectorizedOpMap::Lookup(void* value) const
{
    if (auto it = _vectorizedOps.find(value); it != _vectorizedOps.end())
    {
        return it->second;
    }
    else
    {
        return std::nullopt;
    }
}

void VectorizedOpMap::Map(mlir::Operation* op, VectorizedOp vectorizedOp)
{
    Map(static_cast<void*>(op), vectorizedOp);
}

void VectorizedOpMap::Map(mlir::Value value, VectorizedOp vectorizedOp)
{
    auto ptr = value.getAsOpaquePointer();
    Map(ptr, vectorizedOp);
}

void VectorizedOpMap::Map(void* value, VectorizedOp vectorizedOp)
{
    _vectorizedOps[value] = vectorizedOp;
}

bool VectorizedOpMap::HasMapping(mlir::Operation* op) const
{
    return HasMapping(static_cast<void*>(op));
}

bool VectorizedOpMap::HasMapping(mlir::Value value) const
{
    auto ptr = value.getAsOpaquePointer();
    return HasMapping(ptr);
}

bool VectorizedOpMap::HasMapping(void* value) const
{
    return _vectorizedOps.find(value) != _vectorizedOps.end();
}

bool CanVectorizeOp(mlir::Operation* op,
                    const VectorizedOpMap& vectorizedOps,
                    std::vector<mlir::BlockAndValueMapping>& laneMappings,
                    mlir::Value inductionVar,
                    int64_t step,
                    int64_t vectorSize)
{
    // If this op doesn't depend on the induction var and produces a result, it should be trivially vectorizable (by broadcasting it)
    if (op->getNumResults() == 1 && (!inductionVar || !ir::util::hasRecursiveUseOfOp(inductionVar, op)))
    {
        return true;
    }

    auto result =
        mlir::TypeSwitch<mlir::Operation*, bool>(op)
            .Case([](mlir::memref::AllocaOp) { return true; })
            .Case([](mlir::arith::ConstantOp) { return true; })
            .Case([](mlir::memref::LoadOp) { return true; })
            .Case([](mlir::memref::StoreOp) { return true; })
            .Case([](mlir::AffineLoadOp) { return true; })
            .Case([](mlir::AffineStoreOp) { return true; })
            .Case([](mlir::SelectOp) { return true; })
            .Case([](mlir::arith::ShLIOp) { return true; })
            .Case([](mlir::arith::FPToSIOp) { return true; })
            .Case([](mlir::arith::ExtSIOp) { return true; })
            .Case([](mlir::math::AbsOp) { return true; })
            // .Case([&](mlir::AffineApplyOp) { return true; }) // TODO: either enable or remove this
            .Case([](mlir::math::ExpOp) { return true; })
            .Case([](v::CastOp) { return true; })
            .Case([vectorSize](v::RoundOp) { return v::RoundOp::SupportsVectorization(vectorSize); })
            .Case([](v::BitcastOp) { return true; })
            .Case([](v::BinOp) { return true; })
            .Case([](v::CmpOp) { return true; })
            .Case([](v::ReferenceGlobalOp) { return true; })
            .Default([&](mlir::Operation* defaultOp) {
                return false;
            });
    return result;
}

std::optional<VectorizedOp> GetVectorizedPredecessor(mlir::PatternRewriter& rewriter,
                                                     mlir::Operation* pred,
                                                     const VectorizedOpMap& vectorizedOps,
                                                     std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                     mlir::Value inductionVar,
                                                     int64_t step,
                                                     int64_t vectorSize)
{
    assert(pred);

    if (auto vecOp = vectorizedOps.Lookup(pred))
    {
        if (vecOp->HasVectorType())
        {
            return vecOp->GetVectorResult().getDefiningOp();
        }
        else
        {
            return vecOp;
        }
    }

    if (CanVectorizeOp(pred, vectorizedOps, laneMappings, inductionVar, step, vectorSize))
    {
        auto vecPred = VectorizeOp(rewriter, pred, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
        if (!vecPred.has_value())
        {
            return std::nullopt;
        }
        if (vecPred->HasVectorType())
        {
            return vecPred->GetVectorResult().getDefiningOp();
        }
        else
        {
            return vecPred;
        }
    }

    return std::nullopt;
}

std::optional<VectorizedOp> GetVectorizedPredecessor(mlir::PatternRewriter& rewriter,
                                                     mlir::Value pred,
                                                     const VectorizedOpMap& vectorizedOps,
                                                     std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                     mlir::Value inductionVar,
                                                     int64_t step,
                                                     int64_t vectorSize)
{
    if (auto vecOp = vectorizedOps.Lookup(pred))
    {
        if (vecOp->HasVectorType())
        {
            return vecOp->GetVectorResult();
        }
        else
        {
            return vecOp;
        }
    }

    if (auto predOp = pred.getDefiningOp())
    {
        return GetVectorizedPredecessor(rewriter, predOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    }

    return std::nullopt;
}

std::optional<VectorizedOp> VectorizeGenericOp(mlir::PatternRewriter& rewriter,
                                               mlir::Operation* op,
                                               const VectorizedOpMap& vectorizedOps,
                                               std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                               mlir::Value inductionVar,
                                               int64_t step,
                                               int64_t vectorSize)
{
    if (op == nullptr || op->getNumResults() != 1)
    {
        return std::nullopt;
    }

    auto loc = op->getLoc();
    auto opResult = op->getResult(0);
    auto elementType = opResult.getType();
    auto vectorType = mlir::VectorType::get({ vectorSize }, elementType);

    auto result = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, opResult);

    return result.getOperation();
}

std::optional<VectorizedOp> VectorizeAllocaOp(mlir::PatternRewriter& rewriter,
                                              mlir::memref::AllocaOp op,
                                              const VectorizedOpMap& vectorizedOps,
                                              std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                              mlir::Value inductionVar,
                                              int64_t step,
                                              int64_t vectorSize)
{
    // Just create many copies of the AllocaOp
    // TODO : figure out when replacing the allocation with a vector allocation is a valid choice

    [[maybe_unused]] auto loc = op.getLoc();
    std::vector<mlir::Value> result;
    for (int64_t i = 0; i < vectorSize; ++i)
    {
        auto allocaOp = rewriter.clone(*op.getOperation(), laneMappings[i]);
        result.push_back(allocaOp->getResult(0));
    }

    return result;
}

std::optional<mlir::Operation*> VectorizeConstantOp(mlir::PatternRewriter& rewriter,
                                                    mlir::arith::ConstantOp op,
                                                    const VectorizedOpMap& vectorizedOps,
                                                    std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                    mlir::Value inductionVar,
                                                    int64_t step,
                                                    int64_t vectorSize)
{
    auto loc = op.getLoc();
    auto constType = op.getType();
    auto vectorType = mlir::VectorType::get({ vectorSize }, constType);
    auto constVec = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, op.getResult());
    return constVec;
}

// TODO de-dupe some internals with GetConstantStrideBetweenUnrolledAccesses
template <typename LhsOpType, typename RhsOpType>
std::optional<int64_t> GetConstantStrideBetweenAccesses(mlir::PatternRewriter& rewriter,
                                                        LhsOpType lhsAccessOp,
                                                        RhsOpType rhsAccessOp)
{
    std::stack<mlir::Operation*> tempOps;
    ir::util::TempOpCleanupGuard tempOpGuard(&tempOps, rewriter);

    auto lhsAccessMapComposition = ir::util::GetIndexToMemoryLocationMap(rewriter.getContext(), lhsAccessOp);
    auto rhsAccessMapComposition = ir::util::GetIndexToMemoryLocationMap(rewriter.getContext(), rhsAccessOp);

    // For dynamically shaped memrefs, currently we only handle identity-mapped memrefs,
    // general dynamic memref support will come later.
    auto lhsMemRefType = lhsAccessOp.memref().getType().template cast<mlir::MemRefType>();
    if (!lhsMemRefType.hasStaticShape())
    {
        if (!ir::util::HasIdentityLayout(lhsAccessOp.memref()))
        {
            return std::nullopt;
        }
    }

    auto rhsMemRefType = rhsAccessOp.memref().getType().template cast<mlir::MemRefType>();
    if (!rhsMemRefType.hasStaticShape())
    {
        if (!ir::util::HasIdentityLayout(rhsAccessOp.memref()))
        {
            return std::nullopt;
        }
    }

    // Re-check if there is no static shape and collect the symbols now that we know we won't be returning std::nullopt
    // because ir::util::GetIdentityMemrefStrideSymbols() does a non-trivial amount of work that me may as well not waste
    std::vector<mlir::Value> lhsStrideSymbols;
    std::vector<mlir::Value> rhsStrideSymbols;
    if (!lhsMemRefType.hasStaticShape())
    {
        lhsStrideSymbols = ir::util::GetIdentityMemrefStrideSymbols(rewriter, lhsAccessOp.getLoc(), lhsAccessOp.memref());
    }
    if (!rhsMemRefType.hasStaticShape())
    {
        rhsStrideSymbols = ir::util::GetIdentityMemrefStrideSymbols(rewriter, rhsAccessOp.getLoc(), rhsAccessOp.memref());
    }

    std::vector<mlir::Value> lhsIndicesVec(lhsAccessOp.indices().begin(), lhsAccessOp.indices().end());
    std::vector<mlir::Value> rhsIndicesVec(rhsAccessOp.indices().begin(), rhsAccessOp.indices().end());

    // Append any dynamic stride symbols since we're dealing with a flattened layout map
    lhsIndicesVec.insert(lhsIndicesVec.end(), lhsStrideSymbols.begin(), lhsStrideSymbols.end());
    rhsIndicesVec.insert(rhsIndicesVec.end(), rhsStrideSymbols.begin(), rhsStrideSymbols.end());

    auto lhsAccess = ir::util::MultiDimAffineApply(rewriter, lhsAccessOp.getLoc(), lhsAccessMapComposition, lhsIndicesVec);
    auto rhsAccess = ir::util::MultiDimAffineApply(rewriter, rhsAccessOp.getLoc(), rhsAccessMapComposition, rhsIndicesVec);
    assert(lhsAccess.size() == 1);
    assert(rhsAccess.size() == 1);
    tempOps.push(lhsAccess[0].getDefiningOp());
    tempOps.push(rhsAccess[0].getDefiningOp());

    mlir::AffineExpr diffExpr = rewriter.getAffineDimExpr(1) - rewriter.getAffineDimExpr(0);
    auto diffMap = mlir::AffineMap::get(2, 0, diffExpr);

    mlir::SmallVector<mlir::Value, 4> compareAccesses{ lhsAccess[0], rhsAccess[0] };
    mlir::fullyComposeAffineMapAndOperands(&diffMap, &compareAccesses);

    assert(diffMap.getNumResults() == 1);
    auto resultExpr = diffMap.getResult(0);
    if (resultExpr.isa<mlir::AffineConstantExpr>())
    {
        auto constExpr = resultExpr.dyn_cast<mlir::AffineConstantExpr>();
        return constExpr.getValue();
    }

    // There isn't a constant difference between memory accesses
    return std::nullopt;
}

template <typename OpType>
std::optional<int64_t> GetConstantStrideBetweenUnrolledAccesses(mlir::PatternRewriter& rewriter,
                                                                OpType op,
                                                                std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                                int64_t vectorSize)
{
    // Create some unrolled clones in-memory and see whether they are accessing memory-sequential elements in the MemRef
    std::stack<mlir::Operation*> tempOps;
    ir::util::TempOpCleanupGuard tempOpGuard(&tempOps, rewriter);

    auto loc = op.getLoc();
    std::vector<OpType> temporaryClones;
    temporaryClones.reserve(vectorSize);
    for (int64_t i = 0; i < vectorSize; ++i)
    {
        auto newTempOp = mlir::dyn_cast<OpType>(rewriter.clone(*op.getOperation(), laneMappings[i]));
        tempOps.push(newTempOp); // Useful for automatic cleanup
        temporaryClones.push_back(newTempOp); // Needed for ordered comparison
    }

    // Check if the temporary clones are all accessing sequential memory
    auto accessMapComposition = ir::util::GetIndexToMemoryLocationMap(rewriter.getContext(), op);

    // For dynamically shaped memrefs, currently we only handle identity-mapped memrefs,
    // general dynamic memref support will come later.
    auto memRefType = op.memref().getType().template cast<mlir::MemRefType>();
    std::vector<mlir::Value> strideSymbols;
    if (!memRefType.hasStaticShape())
    {
        if (!ir::util::HasIdentityLayout(op.memref()))
        {
            return std::nullopt;
        }
        strideSymbols = ir::util::GetIdentityMemrefStrideSymbols(rewriter, loc, op.memref());
    }

    std::optional<int64_t> stride = std::nullopt;
    for (int64_t unrollIdx = 1; unrollIdx < vectorSize; ++unrollIdx)
    {
        std::vector<mlir::Value> prevIndicesVec(temporaryClones[unrollIdx - 1].indices().begin(), temporaryClones[unrollIdx - 1].indices().end());
        std::vector<mlir::Value> currentIndicesVec(temporaryClones[unrollIdx].indices().begin(), temporaryClones[unrollIdx].indices().end());

        // Append any dynamic stride symbols since we're dealing with a flattened layout map
        prevIndicesVec.insert(prevIndicesVec.end(), strideSymbols.begin(), strideSymbols.end());
        currentIndicesVec.insert(currentIndicesVec.end(), strideSymbols.begin(), strideSymbols.end());

        auto prevAccess = ir::util::MultiDimAffineApply(rewriter, loc, accessMapComposition, prevIndicesVec);
        auto currentAccess = ir::util::MultiDimAffineApply(rewriter, loc, accessMapComposition, currentIndicesVec);
        assert(prevAccess.size() == 1);
        assert(currentAccess.size() == 1);
        tempOps.push(prevAccess[0].getDefiningOp());
        tempOps.push(currentAccess[0].getDefiningOp());

        mlir::AffineExpr diffExpr = rewriter.getAffineDimExpr(1) - rewriter.getAffineDimExpr(0);
        auto diffMap = mlir::AffineMap::get(2, 0, diffExpr);

        mlir::SmallVector<mlir::Value, 4> compareAccesses{ prevAccess[0], currentAccess[0] };
        mlir::fullyComposeAffineMapAndOperands(&diffMap, &compareAccesses);

        assert(diffMap.getNumResults() == 1);
        auto resultExpr = diffMap.getResult(0);
        if (resultExpr.isa<mlir::AffineConstantExpr>())
        {
            auto constExpr = resultExpr.dyn_cast<mlir::AffineConstantExpr>();
            if (!stride.has_value())
            {
                stride = constExpr.getValue();
            }
            else if (constExpr.getValue() != *stride)
            {
                // The strides aren't consistent
                return std::nullopt;
            }
        }
        else
        {
            // There isn't a constant difference between sequential op memory accesses
            return std::nullopt;
        }
    }

    return stride;
}

template <typename OpType>
bool DoesUnrolledAccessHaveStride(mlir::PatternRewriter& rewriter,
                                  OpType op,
                                  std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                  int64_t vectorSize,
                                  int64_t stride)
{
    auto strideOpt = GetConstantStrideBetweenUnrolledAccesses(rewriter, op, laneMappings, vectorSize);
    return strideOpt.has_value() && *strideOpt == stride;
}

template <typename OpType>
bool IsUnrolledAccessSequential(mlir::PatternRewriter& rewriter,
                                OpType op,
                                std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                int64_t vectorSize)
{
    return DoesUnrolledAccessHaveStride(rewriter, op, laneMappings, vectorSize, 1 /* stride */);
}

template <typename OpType>
bool IsUnrolledAccessConstant(mlir::PatternRewriter& rewriter,
                              OpType op,
                              std::vector<mlir::BlockAndValueMapping>& laneMappings,
                              int64_t vectorSize)
{
    return DoesUnrolledAccessHaveStride(rewriter, op, laneMappings, vectorSize, 0 /* stride */);
}

mlir::Value FlattenMemRefCast(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value memref)
{
    auto type = memref.getType();
    assert(type.isa<mlir::MemRefType>());
    auto memRefType = type.cast<mlir::MemRefType>();
    auto elementType = memRefType.getElementType();

    if (memRefType.hasStaticShape())
    {
        auto volume = memRefType.getNumElements();
        std::vector<int64_t> flattenedSizes{ volume };
        std::vector<int64_t> flattenedStrides{ 1 };
        mlir::MemRefType flattenedType = mlir::MemRefType::get(flattenedSizes, elementType, { mlir::AffineMap::getMultiDimIdentityMap(1, builder.getContext()) }, memRefType.getMemorySpace());
        return builder.create<mlir::memref::ReinterpretCastOp>(memref.getLoc(), flattenedType, memref, 0 /* offset */, flattenedSizes, flattenedStrides);
    }
    else
    {
        assert(ir::util::HasIdentityLayout(memref) && "Only identity memref maps are currently supported for dynamically sized memrefs");
        auto shapeValueMap = ir::util::GetMemrefShapeValueMap(memref);
        auto shapeAffineApplyOps = ir::util::AffineValueMapToAffineApplyOps(builder, memref.getLoc(), shapeValueMap);
        std::vector<mlir::Value> shapeValues(shapeAffineApplyOps.begin(), shapeAffineApplyOps.end());
        mlir::AffineExpr productExpr = mlir::getAffineConstantExpr(1, builder.getContext());
        for (unsigned symIdx = 0; symIdx < shapeValues.size(); ++symIdx)
        {
            productExpr = productExpr * mlir::getAffineSymbolExpr(symIdx, builder.getContext());
        }
        auto productMap = mlir::AffineMap::get(0, shapeValues.size(), productExpr);
        mlir::Value volume = builder.create<mlir::AffineApplyOp>(loc, productMap, shapeValues);
        mlir::Value flattenedStride = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
        std::vector<mlir::Value> flattenedSizes{ volume };
        std::vector<mlir::Value> flattenedStrides{ flattenedStride };
        mlir::Value zeroOffset = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::MemRefType flattenedType = mlir::MemRefType::get({ mlir::ShapedType::kDynamicSize }, elementType, { mlir::AffineMap::getMultiDimIdentityMap(1, builder.getContext()) }, memRefType.getMemorySpace());
        return builder.create<mlir::memref::ReinterpretCastOp>(memref.getLoc(), flattenedType, memref, zeroOffset, flattenedSizes, flattenedStrides);
    }
}

template <typename OpTy>
std::pair<mlir::Value, mlir::Value> FlattenAccess(mlir::OpBuilder& builder, OpTy accessOp, const std::vector<mlir::Value>& indices)
{
    auto loc = accessOp->getLoc();
    auto flatCastMemref = FlattenMemRefCast(builder, loc, accessOp.memref());
    auto flattenMap = ir::util::GetIndexToMemoryLocationMap(builder.getContext(), accessOp);
    std::vector<mlir::Value> strideSymbols;
    if (ir::util::HasIdentityLayout(accessOp.memref()))
    {
        strideSymbols = ir::util::GetIdentityMemrefStrideSymbols(builder, loc, accessOp.memref());
    }
    std::vector<mlir::Value> indicesAndStrideSymbols = indices;
    indicesAndStrideSymbols.insert(indicesAndStrideSymbols.end(), strideSymbols.begin(), strideSymbols.end());
    auto flatPosition = builder.create<mlir::AffineApplyOp>(loc, flattenMap, indicesAndStrideSymbols);
    return std::make_pair(flatCastMemref, flatPosition);
}

std::optional<VectorizedOp> VectorizeLoadOp(mlir::PatternRewriter& rewriter,
                                            mlir::memref::LoadOp op,
                                            const VectorizedOpMap& vectorizedOps,
                                            std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                            mlir::Value inductionVar,
                                            int64_t step,
                                            int64_t vectorSize)
{
    auto loc = op.getLoc();
    auto memRefType = op.getMemRefType();
    auto elementType = memRefType.getElementType();
    auto vectorType = mlir::VectorType::get({ vectorSize }, elementType);

    mlir::memref::LoadOpAdaptor adaptor{ op };
    std::vector<mlir::Value> indices(adaptor.indices().begin(), adaptor.indices().end());

    mlir::Value result;
    if (IsUnrolledAccessSequential(rewriter, op, laneMappings, vectorSize))
    {
        // We know these reads are sequential, but mlir::vector::LoadOp only operates on memrefs where the minor
        // dimension has unit stride, so cast the memref to a flat buffer and load from that shape
        auto [flatCastMemref, flattenedPosition] = FlattenAccess(rewriter, op, indices);
        result = rewriter.create<mlir::vector::LoadOp>(op.getLoc(), vectorType, flatCastMemref, mlir::ValueRange{ flattenedPosition });
    }
    else
    {
        // Fall back to many loads and stores into a vector
        auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, elementType, rewriter.getZeroAttr(elementType));
        result = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, zero);
        for (int64_t i = 0; i < vectorSize; ++i)
        {
            auto elementLoad = rewriter.clone(*op.getOperation(), laneMappings[i]);
            result = rewriter.create<mlir::vector::InsertElementOp>(loc, elementLoad->getResult(0), result, rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
        }
    }
    return result;
}

std::optional<VectorizedOp> VectorizeStoreOp(mlir::PatternRewriter& rewriter,
                                             mlir::memref::StoreOp op,
                                             const VectorizedOpMap& vectorizedOps,
                                             std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                             mlir::Value inductionVar,
                                             int64_t step,
                                             int64_t vectorSize)
{
    // Get (vector) value to store from map
    mlir::memref::StoreOpAdaptor adaptor{ op };
    auto scalarValue = op.getValueToStore();
    auto vecOp = vectorizedOps.Lookup(scalarValue.getDefiningOp());
    if (!vecOp)
    {
        return std::nullopt;
    }

    [[maybe_unused]] auto loc = op.getLoc();
    auto memRefType = op.getMemRefType();
    [[maybe_unused]] auto elementType = memRefType.getElementType();

    // can't directly store a vector into a non-vector memref, so we'll have to extract each element and store them individually
    auto vectorizedValueToStore = vecOp->GetVectorResult();

    std::vector<mlir::Value> indices(adaptor.indices().begin(), adaptor.indices().end());

    if (IsUnrolledAccessSequential(rewriter, op, laneMappings, vectorSize))
    {
        // We know these reads are sequential, but mlir::vector::StoreOp only operates on memrefs where the minor
        // dimension has unit stride, so cast the memref to a flat buffer and load from that shape
        auto [flatCastMemref, flattenedPosition] = FlattenAccess(rewriter, op, indices);
        mlir::Operation* storeOp = rewriter.create<mlir::vector::StoreOp>(op.getLoc(), vectorizedValueToStore, flatCastMemref, mlir::ValueRange{ flattenedPosition });
        return storeOp;
    }
    else
    {
        std::vector<mlir::Operation*> storeOps;
        for (int64_t i = 0; i < vectorSize; ++i)
        {
            auto offset = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), i);
            auto element = rewriter.create<mlir::vector::ExtractElementOp>(op.getLoc(), vectorizedValueToStore, offset);
            auto elementStore = rewriter.clone(*op.getOperation(), laneMappings[i]);
            elementStore->setOperand(0, element);
            storeOps.push_back(elementStore);
        }
        return storeOps;
    }
}

mlir::vector::LoadOp VectorizeAffineLoadOpHelper(mlir::PatternRewriter& rewriter,
                                                 mlir::AffineLoadOp op,
                                                 int64_t vectorSize)
{
    auto memRefType = op.getMemRefType();
    auto elementType = memRefType.getElementType();
    auto vectorType = mlir::VectorType::get({ vectorSize }, elementType);
    mlir::AffineLoadOpAdaptor adaptor{ op };
    std::vector<mlir::Value> indices(adaptor.indices().begin(), adaptor.indices().end());

    auto [flatCastMemRef, flattenedPos] = FlattenAccess(rewriter, op, indices);
    return rewriter.create<mlir::vector::LoadOp>(op.getLoc(), vectorType, flatCastMemRef, mlir::ValueRange{ flattenedPos });
}

mlir::vector::StoreOp VectorizeAffineStoreOpHelper(mlir::PatternRewriter& rewriter,
                                                   mlir::AffineStoreOp op,
                                                   mlir::Value vecValToStore,
                                                   int64_t vectorSize)
{
    mlir::AffineStoreOpAdaptor adaptor{ op };
    std::vector<mlir::Value> indices(adaptor.indices().begin(), adaptor.indices().end());

    auto [flatCastMemRef, flattenedPos] = FlattenAccess(rewriter, op, indices);
    return rewriter.create<mlir::vector::StoreOp>(op.getLoc(), vecValToStore, flatCastMemRef, mlir::ValueRange{ flattenedPos });
}

mlir::vector::StoreOp VectorizeAffineStoreOpHelper(mlir::PatternRewriter& rewriter,
                                                   mlir::AffineStoreOp op,
                                                   mlir::BlockAndValueMapping valueMapping,
                                                   int64_t vectorSize)
{
    auto scalarStoreVal = op.getValueToStore();
    assert(valueMapping.contains(scalarStoreVal));
    return VectorizeAffineStoreOpHelper(rewriter, op, valueMapping.lookup(scalarStoreVal), vectorSize);
}

std::optional<VectorizedOp> VectorizeAffineLoadOp(mlir::PatternRewriter& rewriter,
                                                  mlir::AffineLoadOp op,
                                                  const VectorizedOpMap& vectorizedOps,
                                                  std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                  mlir::Value inductionVar,
                                                  int64_t step,
                                                  int64_t vectorSize)
{
    auto loc = op.getLoc();
    auto memRefType = op.getMemRefType();
    auto elementType = memRefType.getElementType();
    auto vectorType = mlir::VectorType::get({ vectorSize }, elementType);

    mlir::AffineLoadOpAdaptor adaptor{ op };
    std::vector<mlir::Value> baseIndices(adaptor.indices().begin(), adaptor.indices().end());

    mlir::Value result;
    auto strideOpt = GetConstantStrideBetweenUnrolledAccesses(rewriter, op, laneMappings, vectorSize);
    if (strideOpt.has_value())
    {
        int64_t stride = *strideOpt;
        if (stride == 1)
        {
            // We know these reads are sequential, but mlir::vector::LoadOp only operates on memrefs where the minor
            // dimension has unit stride, so cast the memref to a flat buffer and load from that shape
            auto [flatCastMemref, flattenedPosition] = FlattenAccess(rewriter, op, baseIndices);
            result = rewriter.create<mlir::vector::LoadOp>(op.getLoc(), vectorType, flatCastMemref, mlir::ValueRange{ flattenedPosition });
            return result;
        }
        else if (stride == 0)
        {
            // Broadcast a single loaded element
            auto clonedLoadOp = mlir::dyn_cast<AffineLoadOp>(rewriter.clone(*op.getOperation())); // The original op will likely get discarded as part of successful vectorization
            result = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, clonedLoadOp.getResult());
            return result;
        }
    }
    // Fall back to many loads and stores into a vector
    auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, elementType, rewriter.getZeroAttr(elementType));
    result = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, zero);
    for (int64_t i = 0; i < vectorSize; ++i)
    {
        auto elementLoad = rewriter.clone(*op.getOperation(), laneMappings[i]);
        result = rewriter.create<mlir::vector::InsertElementOp>(loc, elementLoad->getResult(0), result, rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
    }
    return result;
}

std::optional<VectorizedOp> VectorizeAffineStoreOp(mlir::PatternRewriter& rewriter,
                                                   mlir::AffineStoreOp op,
                                                   const VectorizedOpMap& vectorizedOps,
                                                   std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                   mlir::Value inductionVar,
                                                   int64_t step,
                                                   int64_t vectorSize)
{
    [[maybe_unused]] auto loc = op.getLoc();

    // Get (vector) value to store from map
    mlir::AffineStoreOpAdaptor adaptor{ op };
    auto scalarValue = op.getValueToStore();
    auto scalarValueDefOp = scalarValue.getDefiningOp();
    auto vecOp = vectorizedOps.Lookup(scalarValueDefOp);
    if (!vecOp)
    {
        if (mlir::isa<mlir::ConstantOp>(scalarValueDefOp))
        {
            // If it's a constant being stored, just broadcast it to a vector and store that
            auto vectorType = mlir::VectorType::get({ vectorSize }, scalarValue.getType());
            mlir::Value broadcastVal = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, scalarValue);
            vecOp = VectorizedOp(broadcastVal);
        }
        else
        {
            return std::nullopt;
        }
    }

    auto memRefType = op.getMemRefType();
    [[maybe_unused]] auto elementType = memRefType.getElementType();

    // can't directly store a vector into a non-vector memref, so we'll have to extract each element and store them individually
    auto vectorizedValueToStore = vecOp->GetVectorResult();

    std::vector<mlir::Value> baseIndices(adaptor.indices().begin(), adaptor.indices().end());

    if (IsUnrolledAccessSequential(rewriter, op, laneMappings, vectorSize))
    {
        // We know these reads are sequential, but mlir::vector::StoreOp only operates on memrefs where the minor
        // dimension has unit stride, so cast the memref to a flat buffer and load from that shape
        auto [flatCastMemref, flattenedPosition] = FlattenAccess(rewriter, op, baseIndices);
        mlir::Operation* storeOp = rewriter.create<mlir::vector::StoreOp>(op.getLoc(), vectorizedValueToStore, flatCastMemref, mlir::ValueRange{ flattenedPosition });
        return storeOp;
    }
    else
    {
        std::vector<mlir::Operation*> storeOps;
        for (int64_t i = 0; i < vectorSize; ++i)
        {
            auto offset = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), i);
            auto element = rewriter.create<mlir::vector::ExtractElementOp>(op.getLoc(), vectorizedValueToStore, offset);
            auto elementStore = rewriter.clone(*op.getOperation(), laneMappings[i]);
            elementStore->setOperand(0, element);
            storeOps.push_back(elementStore);
        }
        return storeOps;
    }
}

std::optional<VectorizedOp> VectorizeAffineApplyOp(mlir::PatternRewriter& rewriter,
                                                   mlir::AffineApplyOp op,
                                                   const VectorizedOpMap& vectorizedOps,
                                                   std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                   mlir::Value inductionVar,
                                                   int64_t step,
                                                   int64_t vectorSize)
{
    auto loc = op.getLoc();
    std::vector<mlir::Value> result;
    auto inductionVarMap = mlir::AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + step * rewriter.getAffineSymbolExpr(0));
    for (int64_t i = 0; i < vectorSize; ++i)
    {
        // TODO: make a helper function for this indices-array-modification code
        auto offset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
        auto offsetInductionVar = rewriter.create<mlir::AffineApplyOp>(loc, inductionVarMap, mlir::ValueRange{ inductionVar, offset });

        mlir::BlockAndValueMapping& operandMap = laneMappings[i];
        operandMap.map(inductionVar, offsetInductionVar);
        auto elementOp = rewriter.clone(*op.getOperation(), operandMap);
        result.push_back(elementOp->getResult(0));
    }

    return result;
}

std::optional<mlir::Operation*> VectorizeSelectOp(mlir::PatternRewriter& rewriter,
                                                  mlir::SelectOp op,
                                                  const VectorizedOpMap& vectorizedOps,
                                                  std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                  mlir::Value inductionVar,
                                                  int64_t step,
                                                  int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto scalarCond = op.getCondition();
    auto scalarTrueVal = op.getTrueValue();
    auto scalarFalseVal = op.getFalseValue();
    auto cond = GetVectorizedPredecessor(rewriter, scalarCond, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    auto trueVal = GetVectorizedPredecessor(rewriter, scalarTrueVal, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    auto falseVal = GetVectorizedPredecessor(rewriter, scalarFalseVal, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!cond || !trueVal || !falseVal)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto result = rewriter.create<mlir::SelectOp>(loc, cond->GetVectorResult(), trueVal->GetVectorResult(), falseVal->GetVectorResult());
    return result;
}

std::optional<mlir::Operation*> VectorizeShiftLeftOp(mlir::PatternRewriter& rewriter,
                                                     mlir::arith::ShLIOp op,
                                                     const VectorizedOpMap& vectorizedOps,
                                                     std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                     mlir::Value inductionVar,
                                                     int64_t step,
                                                     int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto lhs = GetVectorizedPredecessor(rewriter, op.getLhs(), vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    auto rhs = GetVectorizedPredecessor(rewriter, op.getRhs(), vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!lhs || !rhs)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto result = rewriter.create<mlir::arith::ShLIOp>(loc, lhs->GetVectorResult(), rhs->GetVectorResult());
    return result;
}

// TODO : de-dupe with cast and other simple vectorizable ops
std::optional<mlir::Operation*> VectorizeAccRoundOp(mlir::PatternRewriter& rewriter,
                                                    v::RoundOp op,
                                                    const VectorizedOpMap& vectorizedOps,
                                                    std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                    mlir::Value inductionVar,
                                                    int64_t step,
                                                    int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto inputOp = op.val();
    auto input = GetVectorizedPredecessor(rewriter, inputOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!input)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto scalarResultType = op.getResult().getType();
    auto resultType = mlir::VectorType::get({ vectorSize }, scalarResultType);
    auto result = rewriter.create<v::RoundOp>(loc, resultType, input->GetVectorResult());
    return result;
}

std::optional<mlir::Operation*> VectorizeAccCastOp(mlir::PatternRewriter& rewriter,
                                                   v::CastOp op,
                                                   const VectorizedOpMap& vectorizedOps,
                                                   std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                   mlir::Value inductionVar,
                                                   int64_t step,
                                                   int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto inputOp = op.source();
    auto input = GetVectorizedPredecessor(rewriter, inputOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!input)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto scalarResultType = op.getResult().getType();
    auto resultType = mlir::VectorType::get({ vectorSize }, scalarResultType);
    auto result = rewriter.create<v::CastOp>(loc, resultType, input->GetVectorResult());
    return result;
}

std::optional<mlir::Operation*> VectorizeFPToSIOp(mlir::PatternRewriter& rewriter,
                                                  mlir::arith::FPToSIOp op,
                                                  const VectorizedOpMap& vectorizedOps,
                                                  std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                  mlir::Value inductionVar,
                                                  int64_t step,
                                                  int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto inputOp = op.getIn();
    auto input = GetVectorizedPredecessor(rewriter, inputOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!input)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto scalarResultType = op.getResult().getType();
    auto resultType = mlir::VectorType::get({ vectorSize }, scalarResultType);
    auto result = rewriter.create<mlir::arith::FPToSIOp>(loc, resultType, input->GetVectorResult());
    return result;
}

std::optional<mlir::Operation*> VectorizeSignExtendIOp(mlir::PatternRewriter& rewriter,
                                                       mlir::arith::ExtSIOp op,
                                                       const VectorizedOpMap& vectorizedOps,
                                                       std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                       mlir::Value inductionVar,
                                                       int64_t step,
                                                       int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto inputOp = op.getIn();
    auto input = GetVectorizedPredecessor(rewriter, inputOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!input)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto scalarResultType = op.getResult().getType();
    auto resultType = mlir::VectorType::get({ vectorSize }, scalarResultType);
    auto result = rewriter.create<mlir::arith::ExtSIOp>(loc, resultType, input->GetVectorResult());
    return result;
}

std::optional<mlir::Operation*> VectorizeAbsFOp(mlir::PatternRewriter& rewriter,
                                                mlir::math::AbsOp op,
                                                const VectorizedOpMap& vectorizedOps,
                                                std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                mlir::Value inductionVar,
                                                int64_t step,
                                                int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto inputOp = op.getOperand();
    auto input = GetVectorizedPredecessor(rewriter, inputOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!input)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto result = rewriter.create<mlir::math::AbsOp>(loc, input->GetVectorResult());
    return result;
}

std::optional<mlir::Operation*> VectorizeExpOp(mlir::PatternRewriter& rewriter,
                                               mlir::math::ExpOp op,
                                               const VectorizedOpMap& vectorizedOps,
                                               std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                               mlir::Value inductionVar,
                                               int64_t step,
                                               int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto inputOp = op.getOperand();
    auto input = GetVectorizedPredecessor(rewriter, inputOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!input)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto result = rewriter.create<mlir::math::ExpOp>(loc, input->GetVectorResult());
    return result;
}

std::optional<VectorizedOp> VectorizeBinOp(mlir::PatternRewriter& rewriter,
                                           v::BinOp op,
                                           const VectorizedOpMap& vectorizedOps,
                                           std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                           mlir::Value inductionVar,
                                           int64_t step,
                                           int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto lhs = GetVectorizedPredecessor(rewriter, op.lhs(), vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    auto rhs = GetVectorizedPredecessor(rewriter, op.rhs(), vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!lhs || !rhs)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto predicate = op.getPredicate();

    assert(lhs->HasVectorType() == rhs->HasVectorType()); // TODO : do we need to support the case where one operand is a vector and the other is a series of unrolled values?
    if (lhs->HasVectorType())
    {
        mlir::Value result;
        auto vectorTy = lhs->GetVectorResult().getType();
        if (vectorSize == 8)
        {
            // Special-case max and min for better codegen
            if (predicate == v::BinaryOpPredicate::MAX)
            {
                result = rewriter.create<v::vmaxps>(loc, vectorTy, lhs->GetVectorResult(), rhs->GetVectorResult());
                return result;
            }
            else if (predicate == v::BinaryOpPredicate::MIN)
            {
                result = rewriter.create<v::vminps>(loc, vectorTy, lhs->GetVectorResult(), rhs->GetVectorResult());
                return result;
            }
        }
        result = rewriter.create<v::BinOp>(loc, predicate, lhs->GetVectorResult(), rhs->GetVectorResult());
        return result;
    }
    else
    {
        auto lhsVec = lhs->GetScalarResults();
        auto rhsVec = rhs->GetScalarResults();
        std::vector<mlir::Value> results;
        for (int64_t unrollIdx = 0; unrollIdx < vectorSize; ++unrollIdx)
        {
            mlir::Value current = rewriter.create<v::BinOp>(loc, predicate, lhsVec[unrollIdx], rhsVec[unrollIdx]);
            results.push_back(current);
        }
        return results;
    }
}

std::optional<VectorizedOp> VectorizeCmpOp(mlir::PatternRewriter& rewriter,
                                           v::CmpOp op,
                                           const VectorizedOpMap& vectorizedOps,
                                           std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                           mlir::Value inductionVar,
                                           int64_t step,
                                           int64_t vectorSize)
{
    // Comparisons that are used for If checks can't be vectorized, only unrolled
    // however comparisons that are used for Select checks can be vectorized.

    auto scfIfOps = ir::util::getRecursiveUsesOfType<mlir::scf::IfOp>(op);
    auto rcvIfOps = ir::util::getRecursiveUsesOfType<v::IfOp>(op);
    if (!scfIfOps.empty() || !rcvIfOps.empty())
    {
        // Get (scalar) arguments from map
        std::vector<mlir::Operation*> cmpOps;
        for (int64_t i = 0; i < vectorSize; ++i)
        {
            auto cmpOp = rewriter.clone(*op.getOperation(), laneMappings[i]);
            cmpOps.push_back(cmpOp);
        }
        return cmpOps;
    }
    else
    {
        // Get (vector) arguments from map
        auto lhs = GetVectorizedPredecessor(rewriter, op.lhs(), vectorizedOps, laneMappings, inductionVar, step, vectorSize);
        auto rhs = GetVectorizedPredecessor(rewriter, op.rhs(), vectorizedOps, laneMappings, inductionVar, step, vectorSize);
        if (!lhs || !rhs)
        {
            return std::nullopt;
        }

        auto loc = op.getLoc();
        auto predicate = op.getPredicate();
        mlir::Value result = rewriter.create<v::CmpOp>(loc, predicate, lhs->GetVectorResult(), rhs->GetVectorResult());
        return result;
    }
}

std::optional<mlir::Operation*> VectorizeBitcastOp(mlir::PatternRewriter& rewriter,
                                                   v::BitcastOp op,
                                                   const VectorizedOpMap& vectorizedOps,
                                                   std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                   mlir::Value inductionVar,
                                                   int64_t step,
                                                   int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto scalarInputOp = op.input().getDefiningOp();
    auto input = GetVectorizedPredecessor(rewriter, scalarInputOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!input)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto scalarResultType = op.result().getType();
    auto resultType = mlir::VectorType::get({ vectorSize }, scalarResultType);
    auto result = rewriter.create<v::BitcastOp>(loc, resultType, input->GetVectorResult());
    return result;
}

std::optional<mlir::Operation*> VectorizeReferenceGlobalOp(mlir::PatternRewriter& rewriter,
                                                           v::ReferenceGlobalOp op,
                                                           const VectorizedOpMap& vectorizedOps,
                                                           std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                           mlir::Value inductionVar,
                                                           int64_t step,
                                                           int64_t vectorSize)
{
    // This is just a passthrough -- just clone it
    auto clonedOp = rewriter.clone(*op);
    return clonedOp;
}

std::optional<VectorizedOp> VectorizeOp(mlir::PatternRewriter& rewriter,
                                        mlir::Operation* op,
                                        const VectorizedOpMap& vectorizedOps,
                                        std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                        mlir::Value inductionVar,
                                        int64_t step,
                                        int64_t vectorSize)
{
    namespace memref = mlir::memref;
    namespace math = mlir::math;
    auto resultOp =
        mlir::TypeSwitch<mlir::Operation*, std::optional<VectorizedOp>>(op)
            .Case([&](memref::AllocaOp allocaOp) {
                return VectorizeAllocaOp(rewriter, allocaOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::arith::ConstantOp constantOp) {
                return VectorizeConstantOp(rewriter, constantOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](memref::LoadOp loadOp) {
                return VectorizeLoadOp(rewriter, loadOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](memref::StoreOp storeOp) {
                return VectorizeStoreOp(rewriter, storeOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::AffineLoadOp affineLoadOp) {
                return VectorizeAffineLoadOp(rewriter, affineLoadOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::AffineStoreOp affineStoreOp) {
                return VectorizeAffineStoreOp(rewriter, affineStoreOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::AffineApplyOp affineApplyOp) {
                return VectorizeAffineApplyOp(rewriter, affineApplyOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::SelectOp selectOp) {
                return VectorizeSelectOp(rewriter, selectOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::arith::ShLIOp shiftLeftOp) {
                return VectorizeShiftLeftOp(rewriter, shiftLeftOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::arith::FPToSIOp castOp) {
                return VectorizeFPToSIOp(rewriter, castOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::arith::ExtSIOp castOp) {
                return VectorizeSignExtendIOp(rewriter, castOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::math::AbsOp absOp) {
                return VectorizeAbsFOp(rewriter, absOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::math::ExpOp expOp) {
                return VectorizeExpOp(rewriter, expOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](v::BinOp binOp) {
                return VectorizeBinOp(rewriter, binOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](v::CmpOp cmpOp) {
                return VectorizeCmpOp(rewriter, cmpOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](v::CastOp castOp) {
                return VectorizeAccCastOp(rewriter, castOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](v::BitcastOp bitcastOp) {
                return VectorizeBitcastOp(rewriter, bitcastOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](v::RoundOp roundOp) {
                return VectorizeAccRoundOp(rewriter, roundOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](v::ReferenceGlobalOp refGlobalOp) {
                return VectorizeReferenceGlobalOp(rewriter, refGlobalOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Default([&](mlir::Operation* defaultOp) -> std::optional<VectorizedOp> {
                if (op->getNumResults() > 0)
                {
                    if (!inductionVar || !ir::util::hasRecursiveUseOfOp(inductionVar, op))
                    {
                        return VectorizeGenericOp(rewriter, defaultOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
                    }
                }

                op->emitError("Trying to vectorize an un-vectorizable op");
                llvm_unreachable("unexpected");
                return {};
            });

    return resultOp;
}

// TODO : support multi-dim vector reductions
mlir::LogicalResult vectorizeHorizontalReduction(mlir::AffineForOp affineForOp, mlir::PatternRewriter& rewriter)
{
    // Try to match a pattern like:
    // for indices
    // for i:
    //     x = load(input[..., i]) : memref<?? x M, T1> -> T1
    //     y = load(output[...]) : memref<??, T1> (doesn't depend on i) -> T1
    //     z = x + y
    //     store(z, output[...]) : (same position as load)

    // And replace it with:
    // flat_input = reinterpret_cast input to flat
    // flat_output = reinterpret_cast output to flat
    // x = vector_load(flat_input, flatten_input_pos(..., i)) : vector<M x T1>
    // y = affine_load(output[...]) : T1
    // z = vector.reduction "add"
    // affine_store(z, output[...])

    // Note: the 'add' operation above can also be many other ops
    // See enum values from  <llvm-project>/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td
    // e.g. add, mul, minui, minsi, minf, maxui, maxsi, maxf, and, or, xor

    // Also allow for the loaded values to be cast before the sum

    // So we need to check for the:
    //  - this affine for op is the innermost loop
    //  - the loop has constant bounds (TODO: relax this check)
    // And the ops in the loop are:
    //  - loop-sequential load
    //  - loop-constant load from location Y
    //  - BinOp of the loaded values
    //  - store BinOp result to location Y

    // Implement the matcher
    auto reportMatchFailure = [&](mlir::Operation* op, std::string message) -> LogicalResult {
        llvm::dbgs() << "[vectorizeHorizontalReduction] While processing " << *op << ". " << message << "\n";
        return rewriter.notifyMatchFailure(op, message);
    };

    std::stack<mlir::Operation*> matchedOps;
    std::stack<mlir::Operation*> tempOps;
    ir::util::TempOpCleanupGuard(&tempOps, rewriter);

    SmallVector<AffineForOp, 2> loops;
    mlir::getPerfectlyNestedLoops(loops, affineForOp);
    if (loops.size() != 1) // there should be exactly 1 loops in the nest being vectorized
    {
        return failure();
    }

    // TODO : support dynamic loops that operate over contiguous memory
    if (!affineForOp.hasConstantBounds() || affineForOp.getConstantLowerBound() != 0)
    {
        return failure();
    }

    int64_t begin = affineForOp.getConstantLowerBound();
    int64_t end = affineForOp.getConstantUpperBound();
    int64_t step = affineForOp.getStep();
    int64_t numIters = (end - begin) / step;
    auto inductionVar = affineForOp.getInductionVar();

    int64_t unrollMax = std::min(numIters, (end - begin));
    auto vectorSize = unrollMax;

    // iterate on loop body from begin to end to match the ops list
    auto loopBodyIter = affineForOp.getBody()->begin();
    auto loopBodyEnd = affineForOp.getBody()->end();

    // 1. load from lhs array
    if (loopBodyIter == loopBodyEnd || !isa<mlir::AffineLoadOp>(*loopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the lhs load op");
    }

    auto lhsLoadOp = cast<mlir::AffineLoadOp>(*loopBodyIter++);
    auto lhsLoadVal = lhsLoadOp.getResult(); // Keep the laoded val separate from the current lhs val for mapping later
    auto lhsVal = lhsLoadVal;
    matchedOps.push(lhsLoadOp);

    // Set up sequential mappings for the loop
    std::vector<mlir::BlockAndValueMapping> laneMappings(unrollMax);
    for (int64_t idx = begin; idx < end; idx += step)
    {
        auto offsetMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) + (idx * step));
        auto offsetInductionVar = rewriter.create<AffineApplyOp>(lhsLoadOp.getLoc(), offsetMap, ValueRange{ inductionVar });
        tempOps.push(offsetInductionVar);
        laneMappings[idx].map(inductionVar, offsetInductionVar);
    }

    bool lhsLoadIsLoopSequential = IsUnrolledAccessSequential(rewriter, lhsLoadOp, laneMappings, unrollMax);
    bool lhsLoadIsLoopConstant = IsUnrolledAccessConstant(rewriter, lhsLoadOp, laneMappings, unrollMax);

    // 1a. (optional) cast
    v::CastOp lhsLoadCastOp;
    mlir::Type lhsCastType;
    if (isa<v::CastOp>(*loopBodyIter))
    {
        lhsLoadCastOp = cast<v::CastOp>(*loopBodyIter++);
        if (lhsLoadCastOp.source() != lhsVal)
        {
            return reportMatchFailure(affineForOp, "Cast after lhs load isn't casting the loaded value");
        }
        auto castedValue = lhsLoadCastOp.result();
        lhsCastType = castedValue.getType();
        lhsVal = castedValue;
        matchedOps.push(lhsLoadCastOp);
    }

    // 2. load from rhs array
    if (loopBodyIter == loopBodyEnd || !isa<mlir::AffineLoadOp>(*loopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the rhs load op");
    }

    auto rhsLoadOp = cast<mlir::AffineLoadOp>(*loopBodyIter++);
    auto rhsLoadVal = rhsLoadOp.getResult();
    auto rhsVal = rhsLoadVal;
    matchedOps.push(rhsLoadOp);

    bool rhsLoadIsLoopSequential = IsUnrolledAccessSequential(rewriter, rhsLoadOp, laneMappings, unrollMax);
    bool rhsLoadIsLoopConstant = IsUnrolledAccessConstant(rewriter, rhsLoadOp, laneMappings, unrollMax);

    // 2a. (optional) cast
    v::CastOp rhsLoadCastOp(nullptr);
    mlir::Type rhsCastType;
    if (isa<v::CastOp>(*loopBodyIter))
    {
        rhsLoadCastOp = cast<v::CastOp>(*loopBodyIter++);
        if (rhsLoadCastOp.source() != rhsVal)
        {
            return reportMatchFailure(affineForOp, "Cast after rhs load isn't casting the loaded value");
        }
        auto castedValue = rhsLoadCastOp.result();
        rhsCastType = castedValue.getType();
        rhsVal = castedValue;
        matchedOps.push(rhsLoadCastOp);
    }

    // 3. bin op
    if (loopBodyIter == loopBodyEnd || !isa<v::BinOp>(*loopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the bin op");
    }
    auto binOp = cast<v::BinOp>(*loopBodyIter++);
    auto binOpVal = binOp.getResult();
    bool lhsRhsLineUp = (binOp.lhs() == lhsVal) && (binOp.rhs() == rhsVal);
    bool lhsRhsSwap = (binOp.lhs() == rhsVal) && (binOp.rhs() == lhsVal);
    if (!lhsRhsLineUp && !lhsRhsSwap)
    {
        return reportMatchFailure(affineForOp, "Bin op isn't using loaded lhs and rhs values");
    }
    matchedOps.push(binOp);

    auto elementType = binOpVal.getType();

    // Get the bin op combining kind and verify that it has a vector reduction counterpart
    mlir::vector::CombiningKind reductionKind;
    // TODO : support AND, OR, MIN, MAX, and XOR as accera bin ops (accera has LOGICAL_AND and LOGICAL_OR, can those be used here?)
    switch (binOp.getPredicate())
    {
    case v::BinaryOpPredicate::ADD:
        reductionKind = mlir::vector::CombiningKind::ADD;
        break;
    case v::BinaryOpPredicate::MUL:
        reductionKind = mlir::vector::CombiningKind::MUL;
        break;
    case v::BinaryOpPredicate::MAX:
        if (elementType.isIntOrFloat())
        {
            if (elementType.isIntOrIndex())
            {
                if (elementType.isUnsignedInteger())
                {
                    reductionKind = mlir::vector::CombiningKind::MAXUI;
                }
                else
                {
                    reductionKind = mlir::vector::CombiningKind::MAXSI;
                }
            }
            else
            {
                reductionKind = mlir::vector::CombiningKind::MAXF;
            }
        }
        else
        {
            return reportMatchFailure(binOp, "'Max' bin op with the given element type cannot be turned into a vector reduction");
        }
        break;
    case v::BinaryOpPredicate::MIN:
        if (elementType.isIntOrFloat())
        {
            if (elementType.isIntOrIndex())
            {
                if (elementType.isUnsignedInteger())
                {
                    reductionKind = mlir::vector::CombiningKind::MINUI;
                }
                else
                {
                    reductionKind = mlir::vector::CombiningKind::MINSI;
                }
            }
            else
            {
                reductionKind = mlir::vector::CombiningKind::MINF;
            }
        }
        else
        {
            return reportMatchFailure(binOp, "'Min' bin op with the given element type cannot be turned into a vector reduction");
        }
        break;
    default:
        return reportMatchFailure(binOp, "Bin op predicate type cannot be turned into a vector reduction");
    }

    // 4. store to output array
    if (loopBodyIter == loopBodyEnd || !isa<mlir::AffineStoreOp>(*loopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the store op");
    }

    auto storeOp = cast<mlir::AffineStoreOp>(*loopBodyIter++);
    auto storeMemRefType = storeOp.getMemRefType();
    auto storeElementType = storeMemRefType.getElementType();
    auto storedVal = storeOp.value();
    matchedOps.push(storeOp);

    // Check that the value being stored is the result of the BinOp
    if (storedVal != binOpVal)
    {
        return reportMatchFailure(storeOp, "Store op isn't storing the result of the bin op");
    }

    // Check that store is constant wrt to the loop
    bool storeIsLoopConstant = IsUnrolledAccessConstant(rewriter, storeOp, laneMappings, unrollMax);
    if (!storeIsLoopConstant)
    {
        return reportMatchFailure(storeOp, "Store op isn't constant wrt the loop being vectorized");
    }

    // Check which load is sequential wrt the loop and which is constant and which one is being stored to

    mlir::AffineLoadOp outputLoadOp;
    if (storeOp.getMemRef() == lhsLoadOp.getMemRef())
    {
        if (!lhsLoadIsLoopConstant)
        {
            return reportMatchFailure(lhsLoadOp, "LHS load op isn't constant wrt the loop being vectorized but is the same memref being stored to");
        }
        if (!rhsLoadIsLoopSequential)
        {
            return reportMatchFailure(rhsLoadOp, "RHS load op isn't sequential when LHS load is constant");
        }
        outputLoadOp = lhsLoadOp;
    }
    else if (storeOp.getMemRef() == rhsLoadOp.getMemRef())
    {
        if (!rhsLoadIsLoopConstant)
        {
            return reportMatchFailure(rhsLoadOp, "RHS load op isn't constant wrt the loop being vectorized but is the same memref being stored to");
        }
        if (!lhsLoadIsLoopSequential)
        {
            return reportMatchFailure(lhsLoadOp, "LHS load op isn't sequential when RHS load is constant");
        }
        outputLoadOp = rhsLoadOp;
    }
    else
    {
        return reportMatchFailure(storeOp, "Store op isn't storing to the same memref as either load");
    }

    // Check that the output load and store are at the same position

    auto strideOpt = GetConstantStrideBetweenAccesses(rewriter, outputLoadOp, storeOp);
    if (!strideOpt.has_value() || *strideOpt != 0)
    {
        return reportMatchFailure(storeOp, "Output load and store ops aren't at the same location");
    }

    // At this point we've verified:
    //  - this affine for op is the innermost loop
    //  - the loop has constant bounds
    // And the ops in the loop are:
    //  - loop-sequential load
    //  - loop-constant load from location Y
    //  - BinOp of the loaded values
    //  - store BinOp result to location Y

    // Check that all that remains are optionally redundant load-stores and the yield op
    
    // match the final pair of redundant load and store ops
    if (loopBodyIter != loopBodyEnd && isa<mlir::AffineLoadOp>(*loopBodyIter))
    {
        auto loadOp = cast<mlir::AffineLoadOp>(*loopBodyIter++);
        matchedOps.push(loadOp);
        if (loopBodyIter != loopBodyEnd && isa<mlir::AffineStoreOp>(*loopBodyIter))
        {
            auto storeOp = cast<mlir::AffineStoreOp>(*loopBodyIter++);
            if (storeOp.getMemRef() != loadOp.getMemRef())
            {
                return reportMatchFailure(storeOp, "Extraneous load/store aren't to the same memref");
            }
            
            auto strideOpt = GetConstantStrideBetweenAccesses(rewriter, loadOp, storeOp);
            if (!strideOpt.has_value() || *strideOpt != 0)
            {
                return reportMatchFailure(storeOp, "Extraneous load/store aren't to the same location");
            }

            matchedOps.push(storeOp);
        }
        else
        {
            return reportMatchFailure(loadOp, "Failed to match extraneous store");
        }
    }

    // Ignore the yield op at the end
    if (loopBodyIter != loopBodyEnd && isa<mlir::AffineYieldOp>(*loopBodyIter))
    {
        (void)loopBodyIter++;
    }

    if (loopBodyIter != loopBodyEnd)
    {
        LLVM_DEBUG(llvm::dbgs() << "Found additional instructions after the store");
        return failure();
    }

    // Set the insertion point to the end of the loop (just before the terminator)
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(affineForOp.getBody(), affineForOp.getBody()->getTerminator()->getIterator());

    // Now replace the matched ops with the vector load and reduction sequence
    mlir::BlockAndValueMapping mappings;

    // LHS Load
    mlir::Value vecLhsVal;
    if (lhsLoadIsLoopSequential)
    {
        auto lhsLoadVecOp = VectorizeAffineLoadOpHelper(rewriter, lhsLoadOp, vectorSize);
        vecLhsVal = lhsLoadVecOp.getResult();
        mappings.map(lhsLoadVal, vecLhsVal);
    }
    else
    {
        vecLhsVal = mlir::cast<mlir::AffineLoadOp>(rewriter.clone(*lhsLoadOp.getOperation(), mappings));
    }
    mappings.map(lhsLoadVal, vecLhsVal);

    // Optional cast
    if (lhsLoadCastOp)
    {
        // Create a vector cast
        auto castVecType = mlir::VectorType::get({ vectorSize }, lhsCastType);
        vecLhsVal = rewriter.create<v::CastOp>(lhsLoadCastOp.getLoc(), vecLhsVal, castVecType);
    }
    mappings.map(lhsVal, vecLhsVal);

    // RHS Load
    mlir::Value vecRhsVal;
    if (rhsLoadIsLoopSequential)
    {
        auto rhsLoadVecOp = VectorizeAffineLoadOpHelper(rewriter, rhsLoadOp, vectorSize);
        vecRhsVal = rhsLoadVecOp.getResult();
        mappings.map(rhsLoadVal, vecRhsVal);
    }
    else
    {
        vecRhsVal = mlir::cast<mlir::AffineLoadOp>(rewriter.clone(*rhsLoadOp.getOperation(), mappings));
    }
    mappings.map(rhsLoadVal, vecRhsVal);

    // Optional cast
    if (rhsLoadCastOp)
    {
        // Create a vector cast
        auto castVecType = mlir::VectorType::get({ vectorSize }, rhsCastType);
        vecRhsVal = rewriter.create<v::CastOp>(rhsLoadCastOp.getLoc(), vecRhsVal, castVecType);
    }
    mappings.map(rhsVal, vecRhsVal);

    // Now create the appropriate vector reduce given the bin op type and apply it to either the LHS vector val or RHS vector val, whichever is the loaded vector
    auto vectorValToReduce = lhsLoadIsLoopSequential ? vecLhsVal : vecRhsVal;
    auto reduceOp = rewriter.create<mlir::vector::ReductionOp>(binOp.getLoc(), storeElementType, mlir::vector::stringifyEnum(reductionKind), vectorValToReduce, mlir::ValueRange{} /* optional accumulate values */);
    
    mlir::Value reducedVal = reduceOp.getResult();
    auto scalarValThatWasReduced = lhsLoadIsLoopSequential ? lhsVal : rhsVal;
    mappings.map(scalarValThatWasReduced, reducedVal);

    // Now we're left with two scalars, since we've reduced one vector to a scalar and the other value was a scalar to begin with.
    // Clone the original bin op now that we've vector reduced either the LHS or RHS side and are left with 2 vectors
    // At this point, in our mappings we've replaces the original lhsVal and rhsVal with either their cloned scalar versions,
    // or the result of the vector reduce
    auto finalBinOp = mlir::cast<v::BinOp>(rewriter.clone(*binOp.getOperation(), mappings));
    mappings.map(binOp, finalBinOp);

    // Clone the final store op
    rewriter.clone(*storeOp.getOperation(), mappings);

    // Set the step size for the vectorized loops such that they each have a single iteration and will later get simplified away while replacing any IV usage with their begin value
    affineForOp.setStep(step * numIters);

    // Erase the original non-vectorized ops
    ir::util::EraseOps(matchedOps, rewriter);

    return mlir::success();
}

// TODO : de-dupe with part of vectorizeInt16Matmul matcher
mlir::LogicalResult vectorizeSequentialCast(mlir::AffineForOp affineForOp, mlir::PatternRewriter& rewriter)
{
    // Try to match a pattern like:
    // for jj:
    //     for kk:
    //         x = load(input[..., jj, kk]) : memref<...x M x N, T1>
    //         y = cast(x, T2) : T2
    //         store(y, output[..., jj, kk]) : memref<...x M x N, T2>

    // And replace it with:
    // flat_input = reinterpret_cast input to flat
    // flat_output = reinterpret_cast output to flat
    // x = vector_load(flat_input, flatten_input_pos(..., jj, kk)) : vector<(M*N)xT1>
    // y = cast(x, T2) : vector<(M*N)xT2>
    // vector_store(y, flat_output, flatten_output_pos(..., jj, kk))

    // So we need to check:
    //  - there are 2 nested loops (TODO : generalize this)
    //  - the loops have constant bounds (TODO: relax this check)
    //  - the innermost loop contains a sequential load
    //  - the innermost loop contains a cast of the loaded value
    //  - the innermost loop contains a sequential store of the cast value
    //  - there are no other ops in the innermost loop (other than a loop terminator op)

    // Implement the matcher
    auto reportMatchFailure = [&](mlir::Operation* op, std::string message) -> LogicalResult {
        llvm::dbgs() << "[vectorizeSequentialCast] While processing " << *op << ". " << message << "\n";
        return rewriter.notifyMatchFailure(op, message);
    };

    std::stack<mlir::Operation*> matchedOps;
    std::stack<mlir::Operation*> tempOps;
    ir::util::TempOpCleanupGuard(&tempOps, rewriter);

    // Match j and k loop
    SmallVector<AffineForOp, 2> loops;
    mlir::getPerfectlyNestedLoops(loops, affineForOp);
    if (loops.size() != 2) // there should be exactly 2 loops in the nest
    {
        return failure();
    }

    // TODO : support dynamic loops that operate over contiguous memory
    for (auto& loop : loops)
    {
        if (!loop.hasConstantBounds() || loop.getConstantLowerBound() != 0)
        {
            return failure();
        }
    }

    auto outerLoop = loops.front(); // jj loop
    int64_t jj_begin = outerLoop.getConstantLowerBound();
    int64_t jj_end = outerLoop.getConstantUpperBound();
    int64_t jj_step = outerLoop.getStep();
    int64_t jj_numIters = (jj_end - jj_begin) / jj_step;
    auto jj_inductionVar = outerLoop.getInductionVar();

    auto innerLoop = loops.back(); // the innermost loop, kk
    int64_t kk_begin = innerLoop.getConstantLowerBound();
    int64_t kk_end = innerLoop.getConstantUpperBound();
    int64_t kk_step = innerLoop.getStep();
    int64_t kk_numIters = (kk_end - kk_begin) / kk_step;
    auto kk_inductionVar = innerLoop.getInductionVar();

    // iterate on loop body from begin to end to match the ops list
    auto innerLoopBodyIter = innerLoop.getBody()->begin();
    auto innerLoopBodyEnd = innerLoop.getBody()->end();

    // 1. load from input array
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the input load op");
    }

    auto loadOp = cast<mlir::AffineLoadOp>(*innerLoopBodyIter);
    auto loadedVal = loadOp.getResult();
    matchedOps.push(loadOp);

    // 2. cast loaded input value
    innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<v::CastOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the cast op");
    }

    auto castOp = cast<v::CastOp>(*innerLoopBodyIter);
    auto castedValue = castOp.result();
    auto castResultType = castedValue.getType();
    matchedOps.push(castOp);

    if (castOp.source() != loadedVal)
    {
        return reportMatchFailure(affineForOp, "Cast op isn't casting the loaded value");
    }

    // 3. store cast value
    innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineStoreOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the store op");
    }

    auto storeOp = cast<mlir::AffineStoreOp>(*innerLoopBodyIter);
    matchedOps.push(storeOp);

    if (storeOp.value() != castedValue)
    {
        return reportMatchFailure(affineForOp, "Store op isn't storing the cast value");
    }

    // Ignore the yield op at the end
    innerLoopBodyIter++;
    if (innerLoopBodyIter != innerLoopBodyEnd && isa<mlir::AffineYieldOp>(*innerLoopBodyIter))
    {
        (void)innerLoopBodyIter++;
    }

    if (innerLoopBodyIter != innerLoopBodyEnd)
    {
        LLVM_DEBUG(llvm::dbgs() << "Found additional instructions after the store");
        return failure();
    }

    // Check if the input loads and output writes are sequential
    int64_t unrollMax_jj = std::min(jj_numIters, (jj_end - jj_begin));
    int64_t unrollMax_kk = std::min(kk_numIters, (kk_end - kk_begin));

    // create lanemappings for jj * kk
    std::vector<mlir::BlockAndValueMapping> laneMappings(unrollMax_kk * unrollMax_jj);
    auto loadLoc = loadOp.getLoc();

    for (int64_t jj_idx = jj_begin; jj_idx < jj_end; jj_idx += jj_step)
    {
        auto jjOffsetMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) + (jj_idx * jj_step));
        auto offsetInductionVar_jj = rewriter.create<AffineApplyOp>(loadLoc, jjOffsetMap, ValueRange{ jj_inductionVar });
        tempOps.push(offsetInductionVar_jj);
        for (int64_t kk_idx = kk_begin; kk_idx < kk_end; kk_idx += kk_step)
        {
            auto kkOffsetMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) + (kk_idx * kk_step));
            auto offsetInductionVar_kk = rewriter.create<AffineApplyOp>(loadLoc, kkOffsetMap, ValueRange{ kk_inductionVar });
            tempOps.push(offsetInductionVar_kk);
            BlockAndValueMapping& operandMap = laneMappings[jj_idx * unrollMax_kk + kk_idx];
            operandMap.map(kk_inductionVar, offsetInductionVar_kk);
            operandMap.map(jj_inductionVar, offsetInductionVar_jj);
        }
    }

    int64_t vectorSize = unrollMax_jj * unrollMax_kk;

    if (!IsUnrolledAccessSequential(rewriter, loadOp, laneMappings, vectorSize))
    {
        return reportMatchFailure(loadOp, "Failed: isUnrolledAcessSequential for load op");
    }
    if (!IsUnrolledAccessSequential(rewriter, storeOp, laneMappings, vectorSize))
    {
        return reportMatchFailure(storeOp, "Failed: isUnrolledAcessSequential for store op");
    }

    // At this point we know:
    //  - there are 2 nested loops
    //  - the loops have constant bounds
    //  - the innermost loop contains a load that is sequential wrt the 2 loops
    //  - the innermost loop contains a cast of the loaded value
    //  - the innermost loop contains a store of the cast value that is sequential wrt the 2 loops
    //  - there are no other ops in the innermost loop (other than a loop terminator op)

    // So now we can create the new vectorized version of the loops

    // Set the insertion point to the end of the inner loop (just before the terminator)
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(innerLoop.getBody(), innerLoop.getBody()->getTerminator()->getIterator());

    // 1. create vector load of the input
    auto inputMemRefType = loadOp.getMemRefType();
    auto inputElementType = inputMemRefType.getElementType();
    auto inputVectorType = mlir::VectorType::get({ vectorSize }, inputElementType);
    mlir::AffineLoadOpAdaptor loadAdaptor{ loadOp };
    std::vector<mlir::Value> loadIndices(loadAdaptor.indices().begin(), loadAdaptor.indices().end());

    auto [flatCastInputMemRef, flattenedInputPos] = FlattenAccess(rewriter, loadOp, loadIndices);
    auto loadVecOp = rewriter.create<mlir::vector::LoadOp>(loadOp.getLoc(), inputVectorType, flatCastInputMemRef, mlir::ValueRange{ flattenedInputPos });

    // 2. create a cast op of the loaded vector
    auto castResultVecType = mlir::VectorType::get({ vectorSize }, castResultType);
    mlir::Value castVecVal = rewriter.create<v::CastOp>(castOp.getLoc(), loadVecOp, castResultVecType);

    // 3. create a vector store op of the casted value
    mlir::AffineStoreOpAdaptor storeAdaptor{ storeOp };
    std::vector<mlir::Value> storeIndices(storeAdaptor.indices().begin(), storeAdaptor.indices().end());

    auto [flatCastOutputMemRef, flattenedOutputPos] = FlattenAccess(rewriter, storeOp, storeIndices);
    rewriter.create<mlir::vector::StoreOp>(storeOp.getLoc(), castVecVal, flatCastOutputMemRef, mlir::ValueRange{ flattenedOutputPos });

    // Set the step size for the vectorized loops such that they each have a single iteration and will later get simplified away while replacing any IV usage with their begin value
    outerLoop.setStep(jj_step * jj_numIters);
    innerLoop.setStep(kk_step * kk_numIters);

    // Erase the original non-vectorized ops
    ir::util::EraseOps(matchedOps, rewriter);

    return mlir::success();
}

mlir::LogicalResult vectorizeTwoRowInterleavedPack(mlir::AffineForOp affineForOp,
                                                   mlir::PatternRewriter& rewriter)
{
    // TODO : generalize this beyond 2 rows

    // Try to match a pattern like:
    // for jj:
    //     for kk = 0 ... 2:
    //         x = load(input[..., kk, jj]) : memref<...x N x M>
    //         store(x, output[..., jj, kk]) : memref<...x M x N>

    // And replace it with:
    // flat_input = reinterpret_cast input to flat
    // loaded_vec_0 = vector_load(flat_input, flatten_input_pos(..., 0, i))  // vector<MxT1>
    // loaded_vec_1 = vector_load(flat_input, flatten_input_pos(..., 1, i))  // vector<MxT1>
    // interleaved = vector.shuffle loaded_vec_0, loaded_vec_1 [0, M, 1, M+1, 2, M+2, ...]
    // flat_output = reinterpret_cast output to flat
    // vector_store(interleaved, flat_output, flatten_output_pos(..., 0, 0))

    // So we need to check:
    //  - there are 2 nested loops (TODO : generalize this)
    //  - the loops have constant bounds (TODO: relax this check)
    //  - the innermost loop contains a load that is sequential wrt the outer loop
    //  - the innermost loop contains a store that is sequential wrt both loops
    //  - there are no other ops in the innermost loop (other than a loop terminator op)

    // Implement the matcher
    auto reportMatchFailure = [&](mlir::Operation* op, std::string message) -> LogicalResult {
        llvm::dbgs() << "[vectorizeTwoRowInterleavedPack] While processing " << *op << ". " << message << "\n";
        return rewriter.notifyMatchFailure(op, message);
    };

    std::stack<mlir::Operation*> matchedOps;
    std::stack<mlir::Operation*> tempOps;
    ir::util::TempOpCleanupGuard(&tempOps, rewriter);

    // Match j and k loop
    SmallVector<AffineForOp, 2> loops;
    mlir::getPerfectlyNestedLoops(loops, affineForOp);
    if (loops.size() != 2) // there should be exactly 2 loops in the nest
    {
        return failure();
    }

    // TODO : support dynamic loops that operate over contiguous memory
    for (auto& loop : loops)
    {
        if (!loop.hasConstantBounds() || loop.getConstantLowerBound() != 0)
        {
            return failure();
        }
    }

    auto outerLoop = loops.front(); // jj loop
    int64_t jj_begin = outerLoop.getConstantLowerBound();
    int64_t jj_end = outerLoop.getConstantUpperBound();
    int64_t jj_step = outerLoop.getStep();
    int64_t jj_numIters = (jj_end - jj_begin) / jj_step;
    auto jj_inductionVar = outerLoop.getInductionVar();

    auto innerLoop = loops.back(); // the innermost loop, kk
    int64_t kk_begin = innerLoop.getConstantLowerBound();
    int64_t kk_end = innerLoop.getConstantUpperBound();
    int64_t kk_step = innerLoop.getStep();
    int64_t kk_numIters = (kk_end - kk_begin) / kk_step;
    if (kk_numIters != 2)
        return failure();
    auto kk_inductionVar = innerLoop.getInductionVar();

    int64_t unrollMax_jj = std::min(jj_numIters, (jj_end - jj_begin));
    int64_t unrollMax_kk = std::min(kk_numIters, (kk_end - kk_begin));

    // iterate on loop body from begin to end to match the ops list
    auto innerLoopBodyIter = innerLoop.getBody()->begin();
    auto innerLoopBodyEnd = innerLoop.getBody()->end();

    // 1. load from input array
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the input load op");
    }

    auto loadOp = cast<mlir::AffineLoadOp>(*innerLoopBodyIter);
    auto loadLoc = loadOp.getLoc();
    auto loadedVal = loadOp.getResult();
    matchedOps.push(loadOp);

    // 2. store value
    innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineStoreOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the store op");
    }

    auto storeOp = cast<mlir::AffineStoreOp>(*innerLoopBodyIter);
    matchedOps.push(storeOp);

    if (storeOp.value() != loadedVal)
    {
        return reportMatchFailure(affineForOp, "Store op isn't storing the loaded value");
    }

    // Ignore the yield op at the end
    innerLoopBodyIter++;
    if (innerLoopBodyIter != innerLoopBodyEnd && isa<mlir::AffineYieldOp>(*innerLoopBodyIter))
    {
        (void)innerLoopBodyIter++;
    }

    if (innerLoopBodyIter != innerLoopBodyEnd)
    {
        LLVM_DEBUG(llvm::dbgs() << "Found additional instructions after the store");
        return failure();
    }

    // Create two sets of lane mappings: one just for jj and one for jj and kk together

    // create lanemappings for jj
    std::vector<mlir::BlockAndValueMapping> jj_laneMappings(unrollMax_jj);

    // create lanemappings for jj and kk
    std::vector<mlir::BlockAndValueMapping> jj_kk_laneMappings(unrollMax_kk * unrollMax_jj);

    for (int64_t jj_idx = jj_begin; jj_idx < jj_end; jj_idx += jj_step)
    {
        auto jjOffsetMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) + (jj_idx * jj_step));
        auto offsetInductionVar_jj = rewriter.create<AffineApplyOp>(loadLoc, jjOffsetMap, ValueRange{ jj_inductionVar });
        tempOps.push(offsetInductionVar_jj);
        BlockAndValueMapping& jj_operandMap = jj_laneMappings[jj_idx];
        jj_operandMap.map(jj_inductionVar, offsetInductionVar_jj);
        for (int64_t kk_idx = kk_begin; kk_idx < kk_end; kk_idx += kk_step)
        {
            auto kkOffsetMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) + (kk_idx * kk_step));
            auto offsetInductionVar_kk = rewriter.create<AffineApplyOp>(loadLoc, kkOffsetMap, ValueRange{ kk_inductionVar });
            tempOps.push(offsetInductionVar_kk);
            BlockAndValueMapping& jj_kk_operandMap = jj_kk_laneMappings[jj_idx * unrollMax_kk + kk_idx];
            jj_kk_operandMap.map(kk_inductionVar, offsetInductionVar_kk);
            jj_kk_operandMap.map(jj_inductionVar, offsetInductionVar_jj);
        }
    }

    // Check if the input load is sequential wrt the jj loop
    int64_t inputVectorSize = unrollMax_jj;
    if (!IsUnrolledAccessSequential(rewriter, loadOp, jj_laneMappings, inputVectorSize))
    {
        return reportMatchFailure(loadOp, "Failed: isUnrolledAcessSequential for load op");
    }

    // Check if the output store is sequential wrt the jj and kk loops
    int64_t outputVectorSize = unrollMax_jj * unrollMax_kk;
    if (!IsUnrolledAccessSequential(rewriter, storeOp, jj_kk_laneMappings, outputVectorSize))
    {
        return reportMatchFailure(storeOp, "Failed: isUnrolledAcessSequential for store op");
    }

    // At this point we know:
    //  - there are 2 nested loops, the inner of which has 2 iterations
    //  - the loops have constant bounds
    //  - the innermost loop contains a load that is sequential wrt the outer loop
    //  - the innermost loop contains a store of the loaded value that is sequential wrt the 2 loops
    //  - there are no other ops in the innermost loop (other than a loop terminator op)

    // So now we can create the new vectorized version of the loops

    // Set the insertion point to the end of the inner loop (just before the terminator)
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(innerLoop.getBody(), innerLoop.getBody()->getTerminator()->getIterator());

    // 1. create vector load of the input rows
    auto inputMemRefType = loadOp.getMemRefType();
    auto inputElementType = inputMemRefType.getElementType();
    auto inputVectorType = mlir::VectorType::get({ inputVectorSize }, inputElementType);

    std::vector<mlir::Value> loadedVecs;
    // Clone the load op for each iteration of the kk loop and vectorize each of those loads wrt the jj loop
    for (int64_t kk_idx = kk_begin; kk_idx < kk_end; kk_idx += kk_step)
    {
        auto unrolledInductionVar_kk = rewriter.create<mlir::arith::ConstantIndexOp>(loadLoc, kk_idx);
        tempOps.push(unrolledInductionVar_kk);
        mlir::BlockAndValueMapping kIterMapping;
        kIterMapping.map(kk_inductionVar, unrolledInductionVar_kk);
        auto clonedLoadOp = mlir::cast<mlir::AffineLoadOp>(rewriter.clone(*(loadOp.getOperation()), kIterMapping));
        tempOps.push(clonedLoadOp);

        mlir::AffineLoadOpAdaptor loadAdaptor{ clonedLoadOp };
        std::vector<mlir::Value> loadIndices(loadAdaptor.indices().begin(), loadAdaptor.indices().end());

        auto [flatCastInputMemRef, flattenedInputPos] = FlattenAccess(rewriter, clonedLoadOp, loadIndices);
        mlir::Value loadedVec = rewriter.create<mlir::vector::LoadOp>(loadOp.getLoc(), inputVectorType, flatCastInputMemRef, mlir::ValueRange{ flattenedInputPos });
        loadedVecs.push_back(loadedVec);
    }
    assert(loadedVecs.size() == 2); // Eventually we could relax this, but vector.shuffle ops require precisely 2 vectors, so if we relax this we need to create a sequence of shuffles

    // 2. create a vector.shuffle op to interleave the input rows
    std::vector<int64_t> interleaveMask;
    interleaveMask.reserve(outputVectorSize);
    for (unsigned colIdx = 0; colIdx < unrollMax_jj; ++colIdx)
    {
        // The vector.shuffle mask should be like { 0, N, 1, N+1, 2, N+2, ... } where the jj loop has N iterations
        interleaveMask.push_back(colIdx);
        interleaveMask.push_back(colIdx + unrollMax_jj);
    }

    auto outputMemRefType = storeOp.getMemRefType();
    auto outputElementType = outputMemRefType.getElementType();
    auto outputVectorType = mlir::VectorType::get({ outputVectorSize }, outputElementType);
    auto shuffledRowsOp = rewriter.create<mlir::vector::ShuffleOp>(loadLoc, outputVectorType, loadedVecs[0], loadedVecs[1], rewriter.getI64ArrayAttr(interleaveMask));

    // 3. create a vector store op of the interleaved rows
    mlir::AffineStoreOpAdaptor storeAdaptor{ storeOp };
    std::vector<mlir::Value> storeIndices(storeAdaptor.indices().begin(), storeAdaptor.indices().end());

    auto [flatCastOutputMemRef, flattenedOutputPos] = FlattenAccess(rewriter, storeOp, storeIndices);
    rewriter.create<mlir::vector::StoreOp>(storeOp.getLoc(), shuffledRowsOp, flatCastOutputMemRef, mlir::ValueRange{ flattenedOutputPos });

    // Set the step size for the vectorized loops such that they each have a single iteration and will later get simplified away while replacing any IV usage with their begin value
    outerLoop.setStep(jj_step * jj_numIters);
    innerLoop.setStep(kk_step * kk_numIters);

    // Erase the original non-vectorized ops
    ir::util::EraseOps(matchedOps, rewriter);

    return mlir::success();
}

mlir::LogicalResult vectorizeInt16MatMul(mlir::AffineForOp affineForOp,
                                         mlir::PatternRewriter& rewriter)
{
    // Implement the matcher
    auto reportMatchFailure = [&](mlir::Operation* op, std::string message) -> LogicalResult {
        llvm::dbgs() << "[vectorizeInt16MatMul] While processing " << *op << ". " << message << "\n";
        return rewriter.notifyMatchFailure(op, message);
    };

    auto avx2Support = ir::util::ModuleSupportsTargetDeviceFeature(affineForOp, "avx2");
    auto avx512Support = ir::util::ModuleSupportsTargetDeviceFeature(affineForOp, "avx512");
    if (!avx2Support && !avx512Support)
    {
        // the vpmaddwd instruction is only supported on machines with the AVX2 or AVX512 instruction set extensions
        return reportMatchFailure(affineForOp, "Target device does not support vpmaddwd instruction");
    }

    std::vector<mlir::Type> supportedBaseInputElementTypes { rewriter.getIntegerType(8), rewriter.getIntegerType(8, false /* isSigned */), rewriter.getIntegerType(16) };
    std::vector<mlir::Type> supportedCastInputElementTypes { rewriter.getIntegerType(16), rewriter.getIntegerType(32) };
    auto isInputTypeSupported = [&supportedBaseInputElementTypes, &supportedCastInputElementTypes](const mlir::Type& type, bool baseInputType) {
        if (baseInputType)
            return std::find(supportedBaseInputElementTypes.begin(), supportedBaseInputElementTypes.end(), type) != supportedBaseInputElementTypes.end();
        else
            return std::find(supportedCastInputElementTypes.begin(), supportedCastInputElementTypes.end(), type) != supportedCastInputElementTypes.end();
    };
    auto inputTypeNeedsCast = [&supportedCastInputElementTypes](const mlir::Type& type) {
        return std::find(supportedCastInputElementTypes.begin(), supportedCastInputElementTypes.end(), type) == supportedCastInputElementTypes.end();
    };

    std::stack<Operation*> matchedOps;
    std::stack<mlir::Operation*> tempOps;

    // Match jj and kk loop in int16 matmul for vectorization rewrite rules
    SmallVector<AffineForOp, 2> loops;
    mlir::getPerfectlyNestedLoops(loops, affineForOp);
    if (loops.size() != 2) // there should be exactly 2 loops in the nest
    {
        return failure();
    }

    for (auto& loop : loops)
    {
        if (!loop.hasConstantBounds() || loop.getConstantLowerBound() != 0)
        {
            return failure();
        }
    }

    // order of nested loops we are looking for is
    // jj {0 to 8} followed by kk {0 to 2}
    auto outerLoop = loops.front(); // jj loop
    int64_t jj_begin = outerLoop.getConstantLowerBound();
    int64_t jj_end = outerLoop.getConstantUpperBound();
    int64_t jj_step = outerLoop.getStep();
    int64_t jj_numIters = (jj_end - jj_begin) / jj_step;
    bool supported_jj_numIters = (jj_numIters == 8) || (jj_numIters == 16 && avx512Support);
    if (!supported_jj_numIters)
        return failure();
    auto jj_inductionVar = outerLoop.getInductionVar();

    auto innerLoop = loops.back(); // the innermost loop, kk
    int64_t kk_begin = innerLoop.getConstantLowerBound();
    int64_t kk_end = innerLoop.getConstantUpperBound();
    int64_t kk_step = innerLoop.getStep();
    int64_t kk_numIters = (kk_end - kk_begin) / kk_step;
    if (kk_numIters != 2)
        return failure();
    auto kk_inductionVar = innerLoop.getInductionVar();

    // get unroll max for jj and kk
    int64_t unrollMax_jj = std::min(jj_numIters, (jj_end - jj_begin));
    int64_t unrollMax_kk = std::min(kk_numIters, (kk_end - kk_begin));
    int64_t vectorSize = unrollMax_jj * unrollMax_kk;

    // create IV map for jj and kk
    auto inductionVarMap_jj = AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + jj_step * rewriter.getAffineSymbolExpr(0));
    auto inductionVarMap_kk = AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + kk_step * rewriter.getAffineSymbolExpr(0));

    // create lanemappings for jj, kk, and jj * kk
    std::vector<mlir::BlockAndValueMapping> laneMappings_jj(unrollMax_jj);
    std::vector<mlir::BlockAndValueMapping> laneMappings_kk(unrollMax_kk);
    std::vector<mlir::BlockAndValueMapping> laneMappings_jj_kk(unrollMax_kk * unrollMax_jj);

    for (int64_t jj_idx = jj_begin; jj_idx < jj_end; jj_idx += jj_step)
    {
        auto offset_jj = rewriter.create<arith::ConstantIndexOp>(outerLoop.getLoc(), jj_idx);
        auto offsetInductionVar_jj = rewriter.create<AffineApplyOp>(outerLoop.getLoc(), inductionVarMap_jj, ValueRange{ jj_inductionVar, offset_jj });
        tempOps.push(offset_jj);
        tempOps.push(offsetInductionVar_jj);
        laneMappings_jj[jj_idx].map(jj_inductionVar, offsetInductionVar_jj);
        for (int64_t kk_idx = kk_begin; kk_idx < kk_end; kk_idx += kk_step)
        {
            auto offset_kk = rewriter.create<arith::ConstantIndexOp>(innerLoop.getLoc(), kk_idx);
            auto offsetInductionVar_kk = rewriter.create<AffineApplyOp>(innerLoop.getLoc(), inductionVarMap_kk, ValueRange{ kk_inductionVar, offset_kk });
            tempOps.push(offset_kk);
            tempOps.push(offsetInductionVar_kk);
            laneMappings_jj_kk[jj_idx * unrollMax_kk + kk_idx].map(kk_inductionVar, offsetInductionVar_kk);
            laneMappings_jj_kk[jj_idx * unrollMax_kk + kk_idx].map(jj_inductionVar, offsetInductionVar_jj);
            if (jj_idx == jj_begin)
            {
                // Only map for the first iter of jj
                laneMappings_kk[kk_idx].map(kk_inductionVar, offsetInductionVar_kk);
            }
        }
    }

    // iterate on loop body from begin to end to match the ops list
    auto innerLoopBodyIter = innerLoop.getBody()->begin();
    auto innerLoopBodyEnd = innerLoop.getBody()->end();

    // TODO: ensure we're storing the updated C value back into the same location (disallow C[m,n] = C[i,j] + A[i,k] * B[k,j])

    // TODO : de-dupe between first and second cases

    // 1. load from first matrix
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from the first array");
    }
    auto firstLoad = cast<mlir::AffineLoadOp>(*innerLoopBodyIter++);
    auto firstElementType = firstLoad.getMemRefType().getElementType();
    matchedOps.push(firstLoad);

    // 1a. Optionally allow casting the A value to an int16 if it is not an int16 already
    bool castFirstLoad = false;
    mlir::Value firstLoadVal = firstLoad.getResult();

    if (!isInputTypeSupported(firstElementType, true /* baseInput */))
    {
        return reportMatchFailure(affineForOp, "First load array element type is not a supported type");
    }

    // Check if there's a cast after the load
    if (innerLoopBodyIter != innerLoopBodyEnd && isa<v::CastOp>(*innerLoopBodyIter))
    {
        castFirstLoad = true;
        auto castOp = cast<v::CastOp>(*innerLoopBodyIter++);
        firstLoadVal = castOp.result();
        auto castResultType = firstLoadVal.getType();
        matchedOps.push(castOp);
        if (!isInputTypeSupported(castResultType, false /* baseInput = false because this is a cast */))
        {
            return reportMatchFailure(affineForOp, "First load element is cast to an unsupported type");
        }
    }
    else if (inputTypeNeedsCast(firstElementType))
    {
        return reportMatchFailure(affineForOp, "First load element is not cast to supported type");
    }

    // 2. load from second matrix
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from the second array");
    }
    auto secondLoad = cast<mlir::AffineLoadOp>(innerLoopBodyIter++);
    auto secondElementType = secondLoad.getMemRefType().getElementType();
    matchedOps.push(secondLoad);

    // 2a. Optionally allow casting the B value to an int16 if it is not an int16 already
    bool castSecondLoad = false;
    mlir::Value secondLoadVal = secondLoad.getResult();

    if (!isInputTypeSupported(secondElementType, true /* baseInput */))
    {
        return reportMatchFailure(affineForOp, "Second load array element type is not a supported type");
    }

    // Check if there's a cast after the load
    if (innerLoopBodyIter != innerLoopBodyEnd && isa<v::CastOp>(*innerLoopBodyIter))
    {
        castSecondLoad = true;
        auto castOp = cast<v::CastOp>(*innerLoopBodyIter++);
        secondLoadVal = castOp.result();
        auto castResultType = secondLoadVal.getType();
        matchedOps.push(castOp);
        if (!isInputTypeSupported(castResultType, false /* baseInput = false because this is a cast */))
        {
            return reportMatchFailure(affineForOp, "Second load element is cast to an unsupported type");
        }
    }
    else if (inputTypeNeedsCast(secondElementType))
    {
        return reportMatchFailure(affineForOp, "Second load element is not cast to supported type");
    }

    // If a load is sequential wrt the inner loop and constant wrt the outer loop, then we want to load the elements and broadcast them to fill a 16-element buffer
    // If a load is sequential wrt both loops, then we simply want to load the data

    bool broadcastFirstLoad = IsUnrolledAccessSequential(rewriter, firstLoad, laneMappings_kk, unrollMax_kk) && IsUnrolledAccessConstant(rewriter, firstLoad, laneMappings_jj, unrollMax_jj);
    bool broadcastSecondLoad = IsUnrolledAccessSequential(rewriter, secondLoad, laneMappings_kk, unrollMax_kk) && IsUnrolledAccessConstant(rewriter, secondLoad, laneMappings_jj, unrollMax_jj);

    int64_t firstLoadVecSize = vectorSize;
    int64_t secondLoadVecSize = vectorSize;

    // 3. muliply A * B
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<v::BinOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the binary A*B multiplication op");
    }
    auto mulAB = cast<v::BinOp>(*innerLoopBodyIter++);
    if (mulAB.predicate() != v::BinaryOpPredicate::MUL)
    {
        return reportMatchFailure(mulAB, "Failed to match the multiplication op");
    }
    // Check that the operands for the multiply op are in fact the loads from A and B
    if (!((mulAB.lhs() == firstLoadVal && mulAB.rhs() == secondLoadVal) || (mulAB.rhs() == firstLoadVal && mulAB.lhs() == secondLoadVal)))
    {
        return reportMatchFailure(mulAB, "Failed to match the multiplication operands");
    }
    matchedOps.push(mulAB);
    auto mulABVal = mulAB.getResult();
    auto mulABValType = mulABVal.getType();

    // 4. sign-extend / cast result of A * B if it is not int32
    if (mulABValType != rewriter.getIntegerType(32))
    {
        if (innerLoopBodyIter == innerLoopBodyEnd || !isa<v::CastOp>(*innerLoopBodyIter))
        {
            return reportMatchFailure(affineForOp, "Failed to match the sign extend op");
        }
        auto castMulABOp = cast<v::CastOp>(*innerLoopBodyIter++);
        matchedOps.push(castMulABOp);
        mulABVal = castMulABOp.getResult();
    }
    // TODO: match the type of `from` and `to` operand of sign extend op

    // 5. load from C matrix
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from C Op");
    }
    auto loadCOp = cast<mlir::AffineLoadOp>(innerLoopBodyIter++);
    auto elementBitWidthC = loadCOp.getMemRefType().getElementTypeBitWidth();
    if (elementBitWidthC != 32)
    {
        return failure();
    }
    if (!IsUnrolledAccessSequential(rewriter, loadCOp, laneMappings_jj, vectorSize / 2))
    {
        return reportMatchFailure(loadCOp, "Failed: isUnrolledAcessSequential for C load");
    }

    matchedOps.push(loadCOp);

    // 6. add C + (A * B)
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<v::BinOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the binary add op");
    }
    auto accOp = cast<v::BinOp>(*innerLoopBodyIter++);
    if (accOp.predicate() != v::BinaryOpPredicate::ADD)
    {
        return reportMatchFailure(accOp, "Failed to match the addition op");
    }
    // Check that the operands for the add op are load from C, and multiplication result of A and B
    if (!((accOp.lhs() == loadCOp && accOp.rhs() == mulABVal) || (accOp.rhs() == loadCOp && accOp.lhs() == mulABVal)))
    {
        return reportMatchFailure(accOp, "Failed to match the accumulation operands");
    }
    matchedOps.push(accOp);

    // 7. store result of accumulation op to cache of C matrix
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineStoreOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the store into C");
    }
    auto storeCOp = cast<mlir::AffineStoreOp>(*innerLoopBodyIter++);
    // Check that we are in fact storing the (A*B)+C value, and that we're storing back to the same array
    if (storeCOp.getValueToStore() != accOp || storeCOp.getMemRef() != loadCOp.getMemRef())
    {
        return reportMatchFailure(storeCOp, "Failed to match the store into C");
    }
    if (!IsUnrolledAccessSequential(rewriter, storeCOp, laneMappings_jj, vectorSize / 2))
    {
        return reportMatchFailure(loadCOp, "Failed: isUnrolledAcessSequential for C store");
    }
    matchedOps.push(storeCOp);

    // 8. match the final pair of redundant load and store ops
    // for some reason there sometimes is an extra AffineLoadOp / AffineStoreOp pair being redundantly generated, we need to ignore those
    if (innerLoopBodyIter != innerLoopBodyEnd && isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        auto loadOp = cast<mlir::AffineLoadOp>(*innerLoopBodyIter++);
        matchedOps.push(loadOp);
        if (innerLoopBodyIter != innerLoopBodyEnd && isa<mlir::AffineStoreOp>(*innerLoopBodyIter))
        {
            auto storeOp = cast<mlir::AffineStoreOp>(*innerLoopBodyIter++);
            if (storeOp.getMemRef() != loadOp.getMemRef())
            {
                return reportMatchFailure(storeOp, "Failed to match extraneous load/store");
            }
            matchedOps.push(storeOp);
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
        return failure();
    }

    auto memRefTypeC = loadCOp.getMemRefType();
    auto elementTypeC = memRefTypeC.getElementType();
    auto vectorTypeC = mlir::VectorType::get({ vectorSize / 2 }, elementTypeC);
    mlir::AffineLoadOpAdaptor adaptorC{ loadCOp };
    std::vector<mlir::Value> baseIndicesC(adaptorC.indices().begin(), adaptorC.indices().end());

    mlir::Value loadCVecOp;
    // Set the insertion point to the end of the inner loop (just before the terminator)
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(innerLoop.getBody(), innerLoop.getBody()->getTerminator()->getIterator());

    // Now replace the loop with a sequence of operations:

    // module @Int16Test1Module {
    //   func @Int16Test1(%arg0: vector<16xi16>, %arg1: vector<16xi16>, %arg2: memref<vector<8xi32>>) {
    //     (load A)
    //     (load B)
    //     %0 = vector.shuffle %arg0, %arg0 [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi16>, vector<16xi16>
    //     %1 = vector.shuffle %arg0, %arg0 [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi16>, vector<16xi16>
    //     %2 = vector.shuffle %arg1, %arg1 [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi16>, vector<16xi16>
    //     %3 = vector.shuffle %arg1, %arg1 [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi16>, vector<16xi16>
    //     %4 = arith.extsi %0 : vector<8xi16> to vector<8xi32>
    //     %5 = arith.extsi %2 : vector<8xi16> to vector<8xi32>
    //     %6 = arith.extsi %1 : vector<8xi16> to vector<8xi32>
    //     %7 = arith.extsi %3 : vector<8xi16> to vector<8xi32>
    //     %8 = arith.muli %4, %5 : vector<8xi32>
    //     %9 = arith.muli %6, %7 : vector<8xi32>
    //     %10 = arith.addi %8, %9 : vector<8xi32>
    //     memref.store %10, %arg2[] : memref<vector<8xi32>>
    //     return
    //   }
    // }

    // Match the order of indices for A, B, C

    // Match the stride access counts for B to ensure that they are contiguous in memory

    // Implement the rewriter by stiching together a list of vector instructions, vector of 16 elements in this case
    // 1. create vector.load A
    auto i16Type = rewriter.getIntegerType(16);
    auto i32Type = rewriter.getIntegerType(32);
    auto fullVecType = mlir::VectorType::get({ vectorSize }, i16Type);
    std::vector<int64_t> altElemsVec; // { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 }
    altElemsVec.reserve(vectorSize);
    for (int64_t i = 0; i < vectorSize; ++i)
    {
        altElemsVec.push_back(i % 2);
    }
    auto altElemsMask = rewriter.getI64ArrayAttr(altElemsVec);

    auto halfVecType = mlir::VectorType::get({ vectorSize / 2 }, i16Type);
    std::vector<int64_t> oddMaskVec; // { 1, 3, 5, 7, 9, 11, 13, 15 }
    std::vector<int64_t> evenMaskVec; // { 0, 2, 4, 6, 8, 10, 12, 14 }
    oddMaskVec.reserve(vectorSize / 2);
    evenMaskVec.reserve(vectorSize / 2);
    for (int64_t i = 0; i < vectorSize / 2; ++i)
    {
        oddMaskVec.push_back(i*2 + 1);
        evenMaskVec.push_back(i*2);
    }
    auto oddMask = rewriter.getI64ArrayAttr(oddMaskVec);
    auto evenMask = rewriter.getI64ArrayAttr(evenMaskVec);

    auto loadCastBroadcastExtractVec = [&](mlir::AffineLoadOp loadOp, int64_t loadVecSize, mlir::Type loadElementType, bool cast, bool broadcast) -> std::tuple<mlir::Value, mlir::Value, mlir::Value> {
        auto loadOpVectorType = mlir::VectorType::get({ loadVecSize }, loadElementType);
        mlir::AffineLoadOpAdaptor loadOpAdaptor{ loadOp };
        std::vector<mlir::Value> loadOpIndices(loadOpAdaptor.indices().begin(), loadOpAdaptor.indices().end());
        auto [flatCastMemRef, flattenedPos] = FlattenAccess(rewriter, loadOp, loadOpIndices);
        mlir::Value loadVecVal = rewriter.create<mlir::vector::LoadOp>(loadOp.getLoc(), loadOpVectorType, flatCastMemRef, mlir::ValueRange{ flattenedPos });
        if (cast)
        {
            // 1a. sign-extend loaded vector values
            auto castLoadVecType = mlir::VectorType::get({ loadVecSize }, i16Type);
            loadVecVal = rewriter.create<v::CastOp>(loadOp.getLoc(), loadVecVal, castLoadVecType);
        }
        if (broadcast)
        {
            // 1b. create vector.shuffle op for first load: alternate between A[0,0] and A[0,1]
            loadVecVal = rewriter.create<mlir::vector::ShuffleOp>(loadOp.getLoc(), fullVecType, loadVecVal, loadVecVal, altElemsMask);
        }

        // 2. Now extract the odds and evens
        mlir::Value oddShuffleVal = rewriter.create<mlir::vector::ShuffleOp>(loadOp.getLoc(), halfVecType, loadVecVal, loadVecVal, oddMask);
        mlir::Value evenShuffleVal = rewriter.create<mlir::vector::ShuffleOp>(loadOp.getLoc(), halfVecType, loadVecVal, loadVecVal, evenMask);

        return { loadVecVal, oddShuffleVal, evenShuffleVal };
    };


    // If there's only one broadcasted load, make sure it happens first for better vpmaddwd matching
    mlir::Value firstLoadVec;
    mlir::Value firstLoadOdds;
    mlir::Value firstLoadEvens;
    mlir::Value secondLoadVec;
    mlir::Value secondLoadOdds;
    mlir::Value secondLoadEvens;

    if (broadcastFirstLoad == broadcastSecondLoad || broadcastFirstLoad)
    {
        auto [firstLoadVecVal, firstLoadOddVal, firstLoadEvenVal] = loadCastBroadcastExtractVec(firstLoad, firstLoadVecSize, firstElementType, castFirstLoad, broadcastFirstLoad);
        auto [secondLoadVecVal, secondLoadOddVal, secondLoadEvenVal] = loadCastBroadcastExtractVec(secondLoad, secondLoadVecSize, secondElementType, castSecondLoad, broadcastSecondLoad);
        firstLoadVec = firstLoadVecVal;
        firstLoadOdds = firstLoadOddVal;
        firstLoadEvens = firstLoadEvenVal;
        secondLoadVec = secondLoadVecVal;
        secondLoadOdds = secondLoadOddVal;
        secondLoadEvens = secondLoadEvenVal;
    }
    else
    {
        // broadcastFirstLoad == false and broadcastSecondLoad == true
        auto [firstLoadVecVal, firstLoadOddVal, firstLoadEvenVal] = loadCastBroadcastExtractVec(secondLoad, secondLoadVecSize, secondElementType, castSecondLoad, broadcastSecondLoad);
        auto [secondLoadVecVal, secondLoadOddVal, secondLoadEvenVal] = loadCastBroadcastExtractVec(firstLoad, firstLoadVecSize, firstElementType, castFirstLoad, broadcastFirstLoad);
        firstLoadVec = firstLoadVecVal;
        firstLoadOdds = firstLoadOddVal;
        firstLoadEvens = firstLoadEvenVal;
        secondLoadVec = secondLoadVecVal;
        secondLoadOdds = secondLoadOddVal;
        secondLoadEvens = secondLoadEvenVal;
    }

    auto bigVecType = mlir::VectorType::get({ vectorSize / 2 }, i32Type);

    // TODO : plumb this from the DSL
#if MATCH_VPMADDWD_INTRINSIC
    // (3-5). Create results using vpmaddwd intrinsic
    auto accumOp = rewriter.create<v::vpmaddwd>(outerLoop.getLoc(), bigVecType, firstLoadVec, secondLoadVec);
#else
    // 3. Sign-extend all ops for further arithmetic operations
    // auto i32Type = rewriter.getIntegerType(32);
    auto sextA_oddOp = rewriter.create<mlir::arith::ExtSIOp>(rewriter.getUnknownLoc(), firstLoadOdds, bigVecType);
    auto sextA_evenOp = rewriter.create<mlir::arith::ExtSIOp>(rewriter.getUnknownLoc(), firstLoadEvens, bigVecType);
    auto sextB_oddOp = rewriter.create<mlir::arith::ExtSIOp>(rewriter.getUnknownLoc(), secondLoadOdds, bigVecType);
    auto sextB_evenOp = rewriter.create<mlir::arith::ExtSIOp>(rewriter.getUnknownLoc(), secondLoadEvens, bigVecType);

    // 4. binOp.mul for sign-extended even shuffled elements of A and B
    // A[00] * B[0], A[00] * B[2], A[00] * B[4] ...
    auto vecMulAB_even = rewriter.create<mlir::arith::MulIOp>(mulAB.getLoc(), sextA_evenOp, sextB_evenOp);
    // A[01] * B[1], A[01] * B[3], A[01] * B[5] ...
    auto vecMulAB_odd = rewriter.create<mlir::arith::MulIOp>(mulAB.getLoc(), sextA_oddOp, sextB_oddOp);

    // 5. Add odd/even sign-extended results
    auto accumOp = rewriter.create<mlir::arith::AddIOp>(rewriter.getUnknownLoc(), vecMulAB_even, vecMulAB_odd);
#endif

    // 6. Vectorize affine.load of C
    auto [flatCastMemRefC, flattenedPosC] = FlattenAccess(rewriter, loadCOp, baseIndicesC);
    loadCVecOp = rewriter.create<mlir::vector::LoadOp>(loadCOp.getLoc(), vectorTypeC, flatCastMemRefC, mlir::ValueRange{ flattenedPosC });

    // 7. Add accumOp to vecLoadC
    auto finalAccOp = rewriter.create<mlir::arith::AddIOp>(accOp.getLoc(), loadCVecOp, accumOp);

    // 8. store final accumulated result to vectorized C
    mlir::AffineStoreOpAdaptor adaptorStoreC{ storeCOp };
    std::vector<mlir::Value> baseIndicesStoreC(adaptorStoreC.indices().begin(), adaptorStoreC.indices().end());

    mlir::vector::StoreOp storeCVecOp;
    auto [flatCastMemRefStoreC, flattenedPosStoreC] = FlattenAccess(rewriter, storeCOp, baseIndicesStoreC);

    rewriter.create<mlir::vector::StoreOp>(storeCOp.getLoc(), finalAccOp.getResult(), flatCastMemRefStoreC, mlir::ValueRange{ flattenedPosStoreC });

    // Set the step size for the vectorized loops to be the vector size in that dimension
    outerLoop.setStep(jj_step * jj_numIters);
    innerLoop.setStep(kk_step * kk_numIters);

    ir::util::EraseOps(matchedOps, rewriter);

    return mlir::success();
}

mlir::LogicalResult vectorizeMaskedLoadStore(mlir::AffineForOp loopOp,
                                            mlir::PatternRewriter& rewriter)
{
    auto reportMatchFailure = [&](mlir::Operation* op, std::string message) -> LogicalResult {
        llvm::dbgs() << "[vectorizeMaskedLoadStore] While processing " << *op << ". " << message << "\n";
        return rewriter.notifyMatchFailure(op, message);
    };
    std::stack<Operation*> matchedOps;

    // Set the insertion point to the end of the loop (just before the terminator)
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(loopOp.getBody(), loopOp.getBody()->getTerminator()->getIterator());


    if (!loopOp.hasConstantBounds() || loopOp.getConstantLowerBound() != 0)
    {
        return reportMatchFailure(loopOp, "Failed: loop op either doesn't have constant bounds or lower bound is not equal to zero");
    }

    int64_t iter_begin = loopOp.getConstantLowerBound();
    int64_t iter_end = loopOp.getConstantUpperBound();
    int64_t iter_step = loopOp.getStep();
    int64_t numIters = (iter_end - iter_begin) / iter_step;
    int64_t unrollMax = std::min(numIters, (iter_end - iter_begin));
    auto inductionVar = loopOp.getInductionVar();

    // iterate on loop body from begin to end to match the ops list
    auto loopBodyStart = loopOp.getBody()->begin();
    auto loopBodyEnd = loopOp.getBody()->end();

    // 1. accera cmp op to compare index with a constant value
    if (loopBodyStart == loopBodyEnd || !isa<v::CmpOp>(*loopBodyStart))
    {
        return reportMatchFailure(loopOp, "Failed to match the accv compare op");
    }

    auto cmpOp = cast<v::CmpOp>(*loopBodyStart);
    auto cmpOpResult = cmpOp.result();
    matchedOps.push(cmpOp);
   
    loopBodyStart++; 

    // 2. match scf.if op 
    if (loopBodyStart == loopBodyEnd || !isa<mlir::scf::IfOp>(*loopBodyStart))
    {
        return reportMatchFailure(loopOp, "Failed to match the scf.if op");
    }
    auto ifOp = cast<mlir::scf::IfOp>(*loopBodyStart);
    matchedOps.push(ifOp);

    // get then and else block of scf.if op
    auto thenBlock = ifOp.thenBlock();
    auto elseBlock = ifOp.elseBlock();

    // match ops in then block
    auto thenOpsIter = thenBlock->getOperations().begin();
    auto thenOpsEnd = thenBlock->getOperations().end();

    if (thenOpsIter == thenOpsEnd || !isa<mlir::AffineLoadOp>(thenOpsIter))
    {
        return reportMatchFailure(ifOp, "Failed to match the load op in then block");
    }

    auto loadOp = cast<mlir::AffineLoadOp>(thenOpsIter++);
    matchedOps.push(loadOp);

    // Optionally allow casting the load value
    mlir::Value loadVal = loadOp.getResult();
    v::CastOp thenCastOp;

    // Check if there's a cast after the load
    if (thenOpsIter != thenOpsEnd && isa<v::CastOp>(thenOpsIter))
    {
        thenCastOp = cast<v::CastOp>(thenOpsIter++);
        matchedOps.push(thenCastOp);
        loadVal = thenCastOp.result();
    }

    if (thenOpsIter == thenOpsEnd || !isa<mlir::AffineStoreOp>(thenOpsIter))
    {
        return reportMatchFailure(ifOp, "Failed to match the store op in then block");
    }
    auto storeOp = cast<mlir::AffineStoreOp>(thenOpsIter++);
    matchedOps.push(storeOp);

    // match ops in else block
    auto storeElemType = storeOp.getMemRefType().getElementType();
    mlir::Value paddingOpValue = rewriter.create<mlir::arith::ConstantOp>(loopOp.getLoc(), rewriter.getZeroAttr(storeElemType));
    v::CastOp elseCastOp;

    if (elseBlock != nullptr)
    {
        auto elseOpsIter = elseBlock->getOperations().begin();
        auto elseOpsEnd = elseBlock->getOperations().end();

        // Optionally check if there is a cast of a zero constant value before it is stored
        if (elseOpsIter != elseOpsEnd && isa<v::CastOp>(elseOpsIter))
        {
            elseCastOp = cast<v::CastOp>(elseOpsIter++);
            mlir::Value castSource = elseCastOp.source();

            // check if source of cast op is a constant
            if (!castSource.getDefiningOp<mlir::arith::ConstantOp>())
            {
                return reportMatchFailure(elseCastOp, "Failed: source of cast op is not a cosntant");
            }

            matchedOps.push(elseCastOp);
        }

        if (elseOpsIter == elseOpsEnd || !isa<mlir::AffineStoreOp>(elseOpsIter))
        {
            return reportMatchFailure(ifOp, "Failed to match the store op in else block");
        }
        auto paddingOp = cast<mlir::AffineStoreOp>(elseOpsIter++);
        mlir::Value paddingSource = paddingOp.getValueToStore();

        if (!paddingSource.getDefiningOp<mlir::arith::ConstantOp>() &&
            !paddingSource.getDefiningOp<v::CastOp>())
        {
            return reportMatchFailure(paddingOp, "Failed: source of affine.store op is neither a cosntant op nor a cast op");
        }

        matchedOps.push(paddingOp);
        paddingOpValue = paddingOp.value();
    }

    // match successful, start rewriting here
    // unroll cmp ops (create lanemappings)
    std::vector<mlir::BlockAndValueMapping> laneMappings(unrollMax);
    for (int64_t idx = iter_begin; idx < iter_end; idx += iter_step)
    {
        auto offsetMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) + (idx * iter_step));
        auto offsetInductionVar = rewriter.create<AffineApplyOp>(cmpOp.getLoc(), offsetMap, ValueRange{ inductionVar });
        laneMappings[idx].map(inductionVar, offsetInductionVar);
    }

    // create a vector<i1> cmpOp results = maskValue
    auto cmpOpLoc = cmpOp.getLoc();
    auto cmpOpType = cmpOpResult.getType();
    auto vectorType = mlir::VectorType::get({ unrollMax }, cmpOpType);

    auto zero = rewriter.create<mlir::arith::ConstantOp>(cmpOpLoc, cmpOpType, rewriter.getZeroAttr(cmpOpType));
    mlir::Value mask = rewriter.create<mlir::vector::BroadcastOp>(cmpOpLoc, vectorType, zero);
    for (int64_t i = 0; i < unrollMax; ++i)
    {
        auto elementCmp = rewriter.clone(*cmpOp.getOperation(), laneMappings[i]);
        mlir::Value elementCmpResult = elementCmp->getResult(0);
        mask = rewriter.create<mlir::vector::InsertElementOp>(cmpOpLoc, elementCmpResult, mask, rewriter.create<mlir::arith::ConstantIndexOp>(cmpOpLoc, i));
    }

    if (!IsUnrolledAccessSequential(rewriter, loadOp, laneMappings, unrollMax))
    {
        return reportMatchFailure(loadOp, "Failed: isUnrolledAcessSequential for load op in then block");
    }

    // create a cast op for source of transfer read's padding value
    if (elseCastOp)
    {
        // clone elseCastOp
        auto cloneElseCastOp = rewriter.clone(*elseCastOp.getOperation());
        paddingOpValue = cloneElseCastOp->getResult(0);
    }

    auto loadLoc = loadOp.getLoc();
    auto loadMemRefType = loadOp.getMemRefType();
    auto loadElementType = loadMemRefType.getElementType();
    auto loadVectorType = mlir::VectorType::get({ unrollMax }, loadElementType);

    // type of padding op value may not be the same as what is required by transfer read op
    // so, we need a cast here always.
    auto finalPaddingOpValue = rewriter.create<v::CastOp>(loopOp.getLoc(), paddingOpValue, loadElementType);
 
    // create transferRead op with mask value
    mlir::AffineLoadOpAdaptor adaptor{ loadOp };
    std::vector<mlir::Value> indices(adaptor.indices().begin(), adaptor.indices().end());
    auto [flatCastMemref, flattenedPosition] = FlattenAccess(rewriter, loadOp, indices);

    // create a default identity map for mapping 1:1 dimension
    mlir::AffineMap permutationMap = mlir::AffineMap::getMinorIdentityMap(1, 1, rewriter.getContext());
    llvm::SmallVector<bool, 1> inbound_init;
    inbound_init.push_back(false);
    auto inbounds = rewriter.getBoolArrayAttr(inbound_init);

    mlir::Value valueToStore = rewriter.create<mlir::vector::TransferReadOp>(loadLoc, loadVectorType, flatCastMemref, mlir::ValueRange{ flattenedPosition }, permutationMap, finalPaddingOpValue, mask, inbounds);

    // optional cast op
    if (thenCastOp) // then cast op
    {
        // Create a cast to hold vector of values
        auto castVecType = mlir::VectorType::get({ unrollMax }, thenCastOp.getType());
        valueToStore = rewriter.create<v::CastOp>(loopOp.getLoc(), valueToStore, castVecType);
    }

    // create vector store op
    mlir::AffineStoreOpAdaptor adaptorStore{ storeOp };
    std::vector<mlir::Value> baseIndicesStore(adaptorStore.indices().begin(), adaptorStore.indices().end());

    mlir::vector::StoreOp storeVecOp;
    auto [flatCastMemRefStore, flattenedPosStore] = FlattenAccess(rewriter, storeOp, baseIndicesStore);


    rewriter.create<mlir::vector::StoreOp>(storeOp.getLoc(), valueToStore, flatCastMemRefStore, mlir::ValueRange{ flattenedPosStore });

    // Set the step size for vectorized loop
    loopOp.setStep(iter_step * numIters);

    ir::util::EraseOps(matchedOps, rewriter);

    return mlir::success();
}

mlir::LogicalResult TryVectorizeKnownSubgraph(mlir::AffineForOp affineForOp,
                                              mlir::PatternRewriter& rewriter)
{
    // TODO : convert these to rewrite pattern structs with benefit weights
    if (succeeded(vectorizeHorizontalReduction(affineForOp, rewriter)))
        return success();
    if (succeeded(vectorizeSequentialCast(affineForOp, rewriter)))
        return success();
    if (succeeded(vectorizeTwoRowInterleavedPack(affineForOp, rewriter)))
        return success();
    if (succeeded(vectorizeInt16MatMul(affineForOp, rewriter)))
        return success();
    if (succeeded(vectorizeMaskedLoadStore(affineForOp, rewriter)))
        return success();
    return failure();
}

} // namespace accera::transforms
