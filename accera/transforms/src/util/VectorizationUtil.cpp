////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "util/VectorizationUtil.h"
#include "util/VectorizedOp.h"

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

template <typename OpType>
bool IsUnrolledAccessSequential(mlir::PatternRewriter& rewriter,
                                OpType op,
                                std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                int64_t vectorSize)
{
    // Create some unrolled clones in-memory and see whether they are accessing memory-sequential elements in the MemRef
    auto loc = op.getLoc();
    std::vector<OpType> temporaryClones;
    temporaryClones.reserve(vectorSize);
    for (int64_t i = 0; i < vectorSize; ++i)
    {
        temporaryClones.push_back(mlir::dyn_cast<OpType>(rewriter.clone(*op.getOperation(), laneMappings[i])));
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
            return false;
        }
        strideSymbols = ir::util::GetIdentityMemrefStrideSymbols(rewriter, loc, op.memref());
    }

    bool sequential = true;
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

        mlir::AffineExpr diffExpr = rewriter.getAffineDimExpr(1) - rewriter.getAffineDimExpr(0);
        auto diffMap = mlir::AffineMap::get(2, 0, diffExpr);

        mlir::SmallVector<mlir::Value, 4> compareAccesses{ prevAccess[0], currentAccess[0] };
        mlir::fullyComposeAffineMapAndOperands(&diffMap, &compareAccesses);

        assert(diffMap.getNumResults() == 1);
        auto resultExpr = diffMap.getResult(0);
        if (resultExpr.isa<mlir::AffineConstantExpr>())
        {
            auto constExpr = resultExpr.dyn_cast<mlir::AffineConstantExpr>();
            if (constExpr.getValue() != 1)
            {
                // There is a constant difference between sequential op memory accesses
                // but the stride is not 1, so the memory isn't contiguous and therefore
                // it's not safe to replace all of the memory ops with a single vector op
                sequential = false;
                break;
            }
        }
        else
        {
            // There isn't a constant difference between sequential op memory accesses
            // so it's not necessarily safe to convert all of the memory ops into a single
            // vector op
            sequential = false;
            break;
        }
    }

    // Clean up the temporary clones
    for (auto& clone : temporaryClones)
    {
        rewriter.eraseOp(clone);
    }
    return sequential;
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
    if (IsUnrolledAccessSequential(rewriter, op, laneMappings, vectorSize))
    {
        // We know these reads are sequential, but mlir::vector::LoadOp only operates on memrefs where the minor
        // dimension has unit stride, so cast the memref to a flat buffer and load from that shape
        auto [flatCastMemref, flattenedPosition] = FlattenAccess(rewriter, op, baseIndices);
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

std::optional<VectorizedOp> VectorizeAffineStoreOp(mlir::PatternRewriter& rewriter,
                                                   mlir::AffineStoreOp op,
                                                   const VectorizedOpMap& vectorizedOps,
                                                   std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                   mlir::Value inductionVar,
                                                   int64_t step,
                                                   int64_t vectorSize)
{
    // Get (vector) value to store from map
    mlir::AffineStoreOpAdaptor adaptor{ op };
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
        mlir::Value result = rewriter.create<v::BinOp>(loc, predicate, lhs->GetVectorResult(), rhs->GetVectorResult());
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
            .Case([&](mlir::math::ExpOp expOp) {
                return VectorizeExpOp(rewriter, expOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](v::BinOp binOp) {
                return VectorizeBinOp(rewriter, binOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](v::CmpOp cmpOp) {
                return VectorizeCmpOp(rewriter, cmpOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](v::BitcastOp bitcastOp) {
                return VectorizeBitcastOp(rewriter, bitcastOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
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

mlir::LogicalResult vectorizeInt16MatMul(mlir::AffineForOp affineForOp,
                                         mlir::PatternRewriter& rewriter)
{
    // Implement the matcher
    auto reportMatchFailure = [&](mlir::Operation* op, std::string message) -> LogicalResult {
        llvm::dbgs() << "While processing " << *op << ". " << message << "\n";
        return rewriter.notifyMatchFailure(op, message);
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
    if (jj_numIters != 8)
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

    // iterate on loop body from begin to end to match the ops list
    auto innerLoopBodyIter = innerLoop.getBody()->begin();
    auto innerLoopBodyEnd = innerLoop.getBody()->end();

    // TODO: deal with case where we load B before A (allow C[i,j] += B[k,j] * A[i,k])
    // TODO: ensure we're storing the updated C value back into the same location (disallow C[m,n] = C[i,j] + A[i,k] * B[k,j])

    // 1. load from A matrix
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from A Op");
    }
    auto loadAOp = cast<mlir::AffineLoadOp>(*innerLoopBodyIter);
    auto elementBitWidthA = loadAOp.getMemRefType().getElementTypeBitWidth();
    if (elementBitWidthA != 16)
    {
        return failure();
    }
    matchedOps.push(loadAOp);

    // verify load from A looks like A[*,kk] or A[kk,*]
    int loadA_kIndex = -1;
    for (auto en : llvm::enumerate(loadAOp.indices()))
    {
        auto i = en.value();
        if (i == kk_inductionVar)
        {
            if (loadA_kIndex != -1)
            {
                return reportMatchFailure(affineForOp, "Failed to match the load from A Op (too many 'k' indicies)");
            }
            loadA_kIndex = en.index();
        }
    }

    if (loadA_kIndex == -1)
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from A Op (no 'k' index)");
    }

    // 2. load from B matrix
    innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from B Op");
    }
    auto loadBOp = cast<mlir::AffineLoadOp>(innerLoopBodyIter);
    auto elementBitWidthB = loadBOp.getMemRefType().getElementTypeBitWidth();
    if (elementBitWidthB != 16)
    {
        return failure();
    }
    matchedOps.push(loadBOp);

    // verify load from B looks like B[kk,jj] or B[jj,kk]
    int loadB_kIndex = -1;
    int loadB_jIndex = -1;
    for (auto en : llvm::enumerate(loadBOp.indices()))
    {
        auto i = en.value();
        if (i == kk_inductionVar)
        {
            if (loadB_kIndex != -1)
            {
                return reportMatchFailure(affineForOp, "Failed to match the load from B Op (too many 'k' indicies)");
            }
            loadB_kIndex = en.index();
        }
        else if (i == jj_inductionVar)
        {
            if (loadB_jIndex != -1)
            {
                return reportMatchFailure(affineForOp, "Failed to match the load from B Op (too many 'j' indicies)");
            }
            loadB_jIndex = en.index();
        }
    }

    if (loadB_kIndex == -1)
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from B Op (no 'k' index)");
    }

    if (loadB_jIndex == -1)
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from B Op (no 'j' index)");
    }

    // 3. muliply A * B
    innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<v::BinOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the binary A*B multiplication op");
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
    matchedOps.push(mulAB);

    // 4. sign-extend / cast result of A * B
    innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<v::CastOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the sign extend op");
    }
    auto castMulABOp = cast<v::CastOp>(*innerLoopBodyIter);
    matchedOps.push(castMulABOp);
    // TODO: match the type of `from` and `to` operand of sign extend op

    // 5. load from C matrix
    innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the load from C Op");
    }
    auto loadCOp = cast<mlir::AffineLoadOp>(innerLoopBodyIter);
    auto elementBitWidthC = loadCOp.getMemRefType().getElementTypeBitWidth();
    if (elementBitWidthC != 32)
    {
        return failure();
    }
    matchedOps.push(loadCOp);

    // 6. add C + (A * B)
    innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<v::BinOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the binary add op");
    }
    auto accOp = cast<v::BinOp>(*innerLoopBodyIter);
    if (accOp.predicate() != v::BinaryOpPredicate::ADD)
    {
        return reportMatchFailure(accOp, "Failed to match the addition op");
    }
    // Check that the operands for the add op are load from C, and multiplication result of A and B
    if (!((accOp.lhs() == loadCOp && accOp.rhs() == castMulABOp) || (accOp.rhs() == loadCOp && accOp.lhs() == castMulABOp)))
    {
        return reportMatchFailure(accOp, "Failed to match the accumulation operands");
    }
    matchedOps.push(accOp);

    // 7. store result of accumulation op to cache of C matrix
    innerLoopBodyIter++;
    if (innerLoopBodyIter == innerLoopBodyEnd || !isa<mlir::AffineStoreOp>(*innerLoopBodyIter))
    {
        return reportMatchFailure(affineForOp, "Failed to match the store into C");
    }
    auto storeCOp = cast<mlir::AffineStoreOp>(*innerLoopBodyIter);
    // Check that we are in fact storing the (A*B)+C value, and that we're storing back to the same array
    if (storeCOp.getValueToStore() != accOp || storeCOp.getMemRef() != loadCOp.getMemRef())
    {
        return reportMatchFailure(storeCOp, "Failed to match the store into C");
    }
    matchedOps.push(storeCOp);

    // 8. match the final pair of redundant load and store ops
    (void)innerLoopBodyIter++;
    // for some reason there sometimes is an extra AffineLoadOp / AffineStoreOp pair being redundantly generated, we need to ignore those
    if (innerLoopBodyIter != innerLoopBodyEnd && isa<mlir::AffineLoadOp>(*innerLoopBodyIter))
    {
        auto loadOp = cast<mlir::AffineLoadOp>(*innerLoopBodyIter);
        matchedOps.push(loadOp);
        (void)innerLoopBodyIter++;
        if (innerLoopBodyIter != innerLoopBodyEnd && isa<mlir::AffineStoreOp>(*innerLoopBodyIter))
        {
            auto storeOp = cast<mlir::AffineStoreOp>(*innerLoopBodyIter);
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

    // Instantiate a TempOpCleanupGuard so that all the matched ops will get cleaned up
    ir::util::TempOpCleanupGuard matchedOpsGuard(&matchedOps, rewriter);
    //ir::util::TempOpCleanupGuard tempOpsGuard(&tempOps, rewriter);

    // Check if elements of B are sequential
    // get unroll max for jj and kk
    int64_t unrollMax_jj = std::min(jj_numIters, (jj_end - jj_begin));
    int64_t unrollMax_kk = std::min(kk_numIters, (kk_end - kk_begin));

    // create IV map for jj and kk
    auto inductionVarMap_jj = AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + jj_step * rewriter.getAffineSymbolExpr(0));
    auto inductionVarMap_kk = AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + kk_step * rewriter.getAffineSymbolExpr(0));

    // create lanemappings for jj * kk
    std::vector<mlir::BlockAndValueMapping> laneMappings(unrollMax_kk * unrollMax_jj);
    auto locB = loadBOp.getLoc();
    
    for (int64_t jj_idx = jj_begin; jj_idx < jj_end; jj_idx += jj_step)
    {
        auto offset_jj = rewriter.create<arith::ConstantIndexOp>(locB, jj_idx);
        auto offsetInductionVar_jj = rewriter.create<AffineApplyOp>(locB, inductionVarMap_jj, ValueRange{ jj_inductionVar, offset_jj });
        tempOps.push(offset_jj);
        tempOps.push(offsetInductionVar_jj);
        for (int64_t kk_idx = kk_begin; kk_idx < kk_end; kk_idx += kk_step)
        {
            auto offset_kk = rewriter.create<arith::ConstantIndexOp>(locB, kk_idx);
            auto offsetInductionVar_kk = rewriter.create<AffineApplyOp>(locB, inductionVarMap_kk, ValueRange{ kk_inductionVar, offset_kk });
            tempOps.push(offset_kk);
            tempOps.push(offsetInductionVar_kk);
            BlockAndValueMapping& operandMap = laneMappings[jj_idx * unrollMax_kk + kk_idx];
            operandMap.map(kk_inductionVar, offsetInductionVar_kk);
            operandMap.map(jj_inductionVar, offsetInductionVar_jj);
        }
    }

    int64_t vectorSize = 16;
    auto memRefTypeB = loadBOp.getMemRefType();
    auto elementTypeB = memRefTypeB.getElementType();
    auto vectorTypeB = mlir::VectorType::get({ vectorSize }, elementTypeB);
    mlir::AffineLoadOpAdaptor adaptorB{ loadBOp };
    std::vector<mlir::Value> baseIndicesB(adaptorB.indices().begin(), adaptorB.indices().end());

    mlir::Value loadBVecOp;
    if (!IsUnrolledAccessSequential(rewriter, loadBOp, laneMappings, vectorSize))
    {
        return reportMatchFailure(loadBOp, "Failed: isUnrolledAcessSequential for B");
    }

    // Check if elements of output array, Y are sequential
    // create lanemappings for jj
    std::vector<mlir::BlockAndValueMapping> laneMappingsC(unrollMax_jj);
    auto loc_loadCOp = loadCOp.getLoc();
    for (int64_t jj_idx = 0; jj_idx < unrollMax_jj; ++jj_idx)
    {
        auto offset_jj = rewriter.create<arith::ConstantIndexOp>(loc_loadCOp, jj_idx);
        auto offsetInductionVar_jj = rewriter.create<AffineApplyOp>(loc_loadCOp, inductionVarMap_jj, ValueRange{ jj_inductionVar, offset_jj });
        tempOps.push(offset_jj);
        tempOps.push(offsetInductionVar_jj);
        BlockAndValueMapping& operandMapC = laneMappingsC[jj_idx];
        operandMapC.map(jj_inductionVar, offsetInductionVar_jj);
    }

    auto memRefTypeC = loadCOp.getMemRefType();
    auto elementTypeC = memRefTypeC.getElementType();
    auto vectorTypeC = mlir::VectorType::get({ vectorSize / 2 }, elementTypeC);
    mlir::AffineLoadOpAdaptor adaptorC{ loadCOp };
    std::vector<mlir::Value> baseIndicesC(adaptorC.indices().begin(), adaptorC.indices().end());

    mlir::Value loadCVecOp;
    if (!IsUnrolledAccessSequential(rewriter, loadCOp, laneMappingsC, vectorSize / 2))
    {
        return reportMatchFailure(loadCOp, "Failed: isUnrolledAcessSequential for C");
    }


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
    auto memRefType = loadAOp.getMemRefType();
    auto elementType = memRefType.getElementType();
    auto vectorType = mlir::VectorType::get({ vectorSize }, elementType);
    mlir::AffineLoadOpAdaptor adaptorA{ loadAOp };
    std::vector<mlir::Value> baseIndicesA(adaptorA.indices().begin(), adaptorA.indices().end());
    // Ignoring the sequential access check for elements of A because that's not required.

    auto [flatCastMemRef, flattenedPos] = FlattenAccess(rewriter, loadAOp, baseIndicesA);
    auto loadAVecOp = rewriter.create<mlir::vector::LoadOp>(loadAOp.getLoc(), vectorType, flatCastMemRef, mlir::ValueRange{ flattenedPos });

    // 2. create vector.shuffle op for A: alternate between A[0,0] and A[0,1]
    auto locA = loadAOp.getLoc();
    auto i16Type = rewriter.getIntegerType(16);
    auto vecType = mlir::VectorType::get({ vectorSize }, i16Type);
    auto altElemsMask = rewriter.getI64ArrayAttr({ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });

    auto halfVecType = mlir::VectorType::get({ vectorSize / 2 }, i16Type);
    auto oddMask = rewriter.getI64ArrayAttr({ 1, 3, 5, 7, 9, 11, 13, 15 });
    auto evenMask = rewriter.getI64ArrayAttr({ 0, 2, 4, 6, 8, 10, 12, 14 });

    auto shuffledAOp = rewriter.create<mlir::vector::ShuffleOp>(locA, vecType, loadAVecOp, loadAVecOp, altElemsMask);

    // 3. create vector shuffle op for A to pick odd and even elements separately
    auto vecLoadA_oddShuffleOp = rewriter.create<mlir::vector::ShuffleOp>(locA, halfVecType, shuffledAOp, shuffledAOp, oddMask);
    auto vecLoadA_evenShuffleOp = rewriter.create<mlir::vector::ShuffleOp>(locA, halfVecType, shuffledAOp, shuffledAOp, evenMask);

    // 4. create vector load op for B
    if (IsUnrolledAccessSequential(rewriter, loadBOp, laneMappings, vectorSize))
    {
        auto [flatCastMemRefB, flattenedPosB] = FlattenAccess(rewriter, loadBOp, baseIndicesB);
        loadBVecOp = rewriter.create<mlir::vector::LoadOp>(loadBOp.getLoc(), vectorTypeB, flatCastMemRefB, mlir::ValueRange{ flattenedPosB });
    }
    else
    {
        return failure();
    }

    // 5. create shuffled ops (odd and even) for loadBVecOp
    auto vecLoadB_oddShuffleOp = rewriter.create<mlir::vector::ShuffleOp>(locB, halfVecType, loadBVecOp, loadBVecOp, oddMask);
    auto vecLoadB_evenShuffleOp = rewriter.create<mlir::vector::ShuffleOp>(locB, halfVecType, loadBVecOp, loadBVecOp, evenMask);

    // 6. Sign-extend all ops for further arithmetic operations
    auto i32Type = rewriter.getIntegerType(32);
    auto bigVecType = mlir::VectorType::get({ vectorSize / 2 }, i32Type);
    auto sextA_oddOp = rewriter.create<mlir::arith::ExtSIOp>(rewriter.getUnknownLoc(), vecLoadA_oddShuffleOp, bigVecType);
    auto sextA_evenOp = rewriter.create<mlir::arith::ExtSIOp>(rewriter.getUnknownLoc(), vecLoadA_evenShuffleOp, bigVecType);
    auto sextB_oddOp = rewriter.create<mlir::arith::ExtSIOp>(rewriter.getUnknownLoc(), vecLoadB_oddShuffleOp, bigVecType);
    auto sextB_evenOp = rewriter.create<mlir::arith::ExtSIOp>(rewriter.getUnknownLoc(), vecLoadB_evenShuffleOp, bigVecType);

    // 7. binOp.mul for sign-extended even shuffled elements of A and B
    // A[00] * B[0], A[00] * B[2], A[00] * B[4] ...
    auto vecMulAB_even = rewriter.create<mlir::arith::MulIOp>(mulAB.getLoc(), sextA_evenOp, sextB_evenOp);
    // A[01] * B[1], A[01] * B[3], A[01] * B[5] ...
    auto vecMulAB_odd = rewriter.create<mlir::arith::MulIOp>(mulAB.getLoc(), sextA_oddOp, sextB_oddOp);

    // 8. Add odd/even sign-extended results
    auto accABOp = rewriter.create<mlir::arith::AddIOp>(rewriter.getUnknownLoc(), vecMulAB_even, vecMulAB_odd);

    // 9. Vectorize affine.load of C
    if (IsUnrolledAccessSequential(rewriter, loadCOp, laneMappingsC, vectorSize / 2))
    {
        // TODO: substitute 0 for jj here
        auto [flatCastMemRefC, flattenedPosC] = FlattenAccess(rewriter, loadCOp, baseIndicesC);
        loadCVecOp = rewriter.create<mlir::vector::LoadOp>(loadCOp.getLoc(), vectorTypeC, flatCastMemRefC, mlir::ValueRange{ flattenedPosC });
    }
    else
    {
        return failure();
    }

    // 10. Add accABOp to vecLoadC
    auto finalAccOp = rewriter.create<mlir::arith::AddIOp>(accOp.getLoc(), loadCVecOp, accABOp);

    // 11. store final accumulated result to vectorized C
    // Verify again if the memory access is sequential and then vectorize the store op
    std::vector<mlir::BlockAndValueMapping> laneMappingsStoreC(unrollMax_jj);
    auto loc_storeCOp = storeCOp.getLoc();
    for (int64_t jj_idx = 0; jj_idx < unrollMax_jj; ++jj_idx)
    {
        auto offset_jj = rewriter.create<arith::ConstantIndexOp>(loc_storeCOp, jj_idx);
        auto offsetInductionVar_jj = rewriter.create<AffineApplyOp>(loc_storeCOp, inductionVarMap_jj, ValueRange{ jj_inductionVar, offset_jj });
        tempOps.push(offset_jj);
        tempOps.push(offsetInductionVar_jj);
        BlockAndValueMapping& operandMapStoreC = laneMappingsStoreC[jj_idx];
        operandMapStoreC.map(jj_inductionVar, offsetInductionVar_jj);
    }

    mlir::AffineStoreOpAdaptor adaptorStoreC{ storeCOp };
    std::vector<mlir::Value> baseIndicesStoreC(adaptorStoreC.indices().begin(), adaptorStoreC.indices().end());

    mlir::vector::StoreOp storeCVecOp;
    if (IsUnrolledAccessSequential(rewriter, storeCOp, laneMappingsStoreC, vectorSize / 2))
    {
        auto [flatCastMemRefStoreC, flattenedPosStoreC] = FlattenAccess(rewriter, storeCOp, baseIndicesStoreC);
        storeCVecOp = rewriter.create<mlir::vector::StoreOp>(storeCOp.getLoc(), finalAccOp.getResult(), flatCastMemRefStoreC, mlir::ValueRange{ flattenedPosStoreC });
    }
    else
    {
        return failure();
    }

    // Set the step size for the vectorized loops to be the vector size in that dimension
    outerLoop.setStep(jj_step * jj_numIters);
    innerLoop.setStep(kk_step * kk_numIters);
    
    return mlir::success();
}

} // namespace accera::transforms
