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
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/Dialect/Vector/VectorUtils.h>

#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>
#include <map>
#include <stdexcept>

using namespace accera::utilities;
namespace v = accera::ir::value;

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
            .Case([](mlir::ConstantOp) { return true; })
            .Case([](mlir::memref::LoadOp) { return true; })
            .Case([](mlir::memref::StoreOp) { return true; })
            .Case([](mlir::AffineLoadOp) { return true; })
            .Case([](mlir::AffineStoreOp) { return true; })
            .Case([](mlir::SelectOp) { return true; })
            .Case([](mlir::ShiftLeftOp) { return true; })
            .Case([](mlir::FPToSIOp) { return true; })
            .Case([](mlir::AbsFOp) { return true; })
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

    auto loc = op.getLoc();
    std::vector<mlir::Value> result;
    for (int64_t i = 0; i < vectorSize; ++i)
    {
        auto allocaOp = rewriter.clone(*op.getOperation(), laneMappings[i]);
        result.push_back(allocaOp->getResult(0));
    }

    return result;
}

std::optional<mlir::Operation*> VectorizeConstantOp(mlir::PatternRewriter& rewriter,
                                                    mlir::ConstantOp op,
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

    bool sequential = true;
    for (size_t unrollIdx = 1; unrollIdx < vectorSize; ++unrollIdx)
    {
        std::vector<mlir::Value> prevIndicesVec(temporaryClones[unrollIdx - 1].indices().begin(), temporaryClones[unrollIdx - 1].indices().end());
        std::vector<mlir::Value> currentIndicesVec(temporaryClones[unrollIdx].indices().begin(), temporaryClones[unrollIdx].indices().end());
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

mlir::Value FlattenMemRefCast(mlir::OpBuilder& builder, mlir::Value memref)
{
    auto type = memref.getType();
    assert(type.isa<mlir::MemRefType>());
    auto memRefType = type.cast<mlir::MemRefType>();
    auto elementType = memRefType.getElementType();
    auto volume = memRefType.getNumElements();

    std::vector<int64_t> flattenedSizes{ volume };
    std::vector<int64_t> flattenedStrides{ 1 };
    mlir::MemRefType flattenedType = mlir::MemRefType::get(flattenedSizes, elementType, { mlir::AffineMap::getMultiDimIdentityMap(1, builder.getContext()) }, memRefType.getMemorySpace());
    return builder.create<mlir::memref::ReinterpretCastOp>(memref.getLoc(), flattenedType, memref, 0 /* offset */, flattenedSizes, flattenedStrides);
}

template <typename OpTy>
std::pair<mlir::Value, mlir::Value> FlattenAccess(mlir::OpBuilder& builder, OpTy accessOp, const std::vector<mlir::Value>& indices)
{
    auto loc = accessOp->getLoc();
    auto flatCastMemref = FlattenMemRefCast(builder, accessOp.memref());
    auto flattenMap = ir::util::GetIndexToMemoryLocationMap(builder.getContext(), accessOp);
    auto flatPosition = builder.create<mlir::AffineApplyOp>(loc, flattenMap, indices);
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
        auto zero = rewriter.create<mlir::ConstantOp>(loc, elementType, rewriter.getZeroAttr(elementType));
        result = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, zero);
        for (int64_t i = 0; i < vectorSize; ++i)
        {
            auto elementLoad = rewriter.clone(*op.getOperation(), laneMappings[i]);
            result = rewriter.create<mlir::vector::InsertElementOp>(loc, elementLoad->getResult(0), result, i);
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
            auto element = rewriter.create<mlir::vector::ExtractElementOp>(op.getLoc(), vectorizedValueToStore, i);
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
        auto zero = rewriter.create<mlir::ConstantOp>(loc, elementType, rewriter.getZeroAttr(elementType));
        result = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, zero);
        for (int64_t i = 0; i < vectorSize; ++i)
        {
            auto elementLoad = rewriter.clone(*op.getOperation(), laneMappings[i]);
            result = rewriter.create<mlir::vector::InsertElementOp>(loc, elementLoad->getResult(0), result, i);
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
            auto element = rewriter.create<mlir::vector::ExtractElementOp>(op.getLoc(), vectorizedValueToStore, i);
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
        auto offset = rewriter.create<mlir::ConstantIndexOp>(loc, i);
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
                                                     mlir::ShiftLeftOp op,
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
    auto result = rewriter.create<mlir::ShiftLeftOp>(loc, lhs->GetVectorResult(), rhs->GetVectorResult());
    return result;
}

std::optional<mlir::Operation*> VectorizeFPToSIOp(mlir::PatternRewriter& rewriter,
                                                  mlir::FPToSIOp op,
                                                  const VectorizedOpMap& vectorizedOps,
                                                  std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                  mlir::Value inductionVar,
                                                  int64_t step,
                                                  int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto inputOp = op.in();
    auto input = GetVectorizedPredecessor(rewriter, inputOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!input)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto scalarResultType = op.getResult().getType();
    auto resultType = mlir::VectorType::get({ vectorSize }, scalarResultType);
    auto result = rewriter.create<mlir::FPToSIOp>(loc, resultType, input->GetVectorResult());
    return result;
}

std::optional<mlir::Operation*> VectorizeAbsFOp(mlir::PatternRewriter& rewriter,
                                                mlir::AbsFOp op,
                                                const VectorizedOpMap& vectorizedOps,
                                                std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                                mlir::Value inductionVar,
                                                int64_t step,
                                                int64_t vectorSize)
{
    // Get (vector) arguments from map
    auto inputOp = op.operand();
    auto input = GetVectorizedPredecessor(rewriter, inputOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
    if (!input)
    {
        return std::nullopt;
    }

    auto loc = op.getLoc();
    auto result = rewriter.create<mlir::AbsFOp>(loc, input->GetVectorResult());
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
    auto inputOp = op.operand();
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
        for (size_t unrollIdx = 0; unrollIdx < vectorSize; ++unrollIdx)
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
            .Case([&](mlir::ConstantOp constantOp) {
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
            .Case([&](mlir::ShiftLeftOp shiftLeftOp) {
                return VectorizeShiftLeftOp(rewriter, shiftLeftOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::FPToSIOp castOp) {
                return VectorizeFPToSIOp(rewriter, castOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](mlir::AbsFOp absOp) {
                return VectorizeAbsFOp(rewriter, absOp, vectorizedOps, laneMappings, inductionVar, step, vectorSize);
            })
            .Case([&](math::ExpOp expOp) {
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

} // namespace accera::transforms
