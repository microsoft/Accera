////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

#include "llvm/ADT/DenseMap.h"

#ifndef __ACCERA__
#include "mlir/Dialect/Argo/IR/ArgoOps.h"
#include "mlir/Dialect/Argo/IR/ArgoTypes.h"
#include "mlir/Dialect/Argo/IR/FoldedIntrinsics.h"
#include "mlir/Dialect/Argo/Utils/Utils.h"
#else
#include "argo/ArgoOps.h"
#include "argo/ArgoTypes.h"
#include "argo/Utils.h"
#include "argo/FoldedIntrinsics.h"
#endif // !__ACCERA__

using namespace llvm;
using namespace mlir;
using namespace mlir::argo;
using namespace mlir::argo::intrinsics;
using namespace mlir::scf;

// global utilities
bool mlir::isConstantZero(Value v)
{
    return isa_and_nonnull<arith::ConstantIndexOp>(v.getDefiningOp()) &&
           cast<arith::ConstantIndexOp>(v.getDefiningOp()).value() == 0;
}

Value mlir::createCeilDivIndex(OpBuilder& b, Location loc, Value lhs, Value rhs, bool useAffine, OperationFolder* folder)
{
    // when use affine, we disable folder due to conflict
    // TODO after resolving, bring back folder in possible
    if (useAffine)
    {
        auto ctx = b.getContext();

        AffineExpr x0 = mlir::getAffineDimExpr(0, ctx);
        AffineExpr x1 = mlir::getAffineDimExpr(1, ctx);
        AffineExpr out = x0.ceilDiv(x1);
        SmallVector<AffineExpr, 4> result;
        result.push_back(out);
        auto ceilMap = AffineMap::get(2, 0, result, ctx);

        SmallVector<Value, 4> operands = { lhs, rhs };
        return b.create<AffineApplyOp>(loc, ceilMap, operands);
    }

    Value rhs_sub_one =
        folded_std_subi(b, loc, folder, rhs, folded_std_constant_index(b, loc, folder, 1));
    Value sum = folded_std_addi(b, loc, folder, lhs, rhs_sub_one);
    Value res = folded_std_diviu(b, loc, folder, sum, rhs);
    return res;
}

int64_t mlir::getConstantIndex(Value v, int64_t dynVal)
{
    if (arith::ConstantIndexOp cOp =
            dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp()))
    {
        return cOp.value();
    }

    // handles possible constant-related ops
    // support DimOp for now
    if (auto dimOp = dyn_cast_or_null<memref::DimOp>(v.getDefiningOp()))
    {
        if (Optional<int64_t> index = dimOp.getConstantIndex())
        {
            return getMemrefOrTensorShape(dimOp.getOperand(0), index.getValue());
        }
    }
    return dynVal;
}

SmallVector<int64_t, 4> mlir::getConstantIndices(ArrayRef<Value> values,
                                                 int64_t dynVal)
{
    SmallVector<int64_t, 4> res;
    for (auto v : values)
    {
        res.push_back(getConstantIndex(v, dynVal));
    }
    return res;
}

// simplify a trivial affineMap to a constant if possible
Optional<Value> mlir::getSimplifiedAffineMap(OpBuilder& b,
                                             Location loc,
                                             AffineMap map,
                                             ValueRange mapOperands)
{
    // simpify 1D for now
    if (map.getNumResults() == 1 && map.getNumInputs() <= 1)
    {
        AffineExpr expr = map.getResult(0);
        if (map.getNumInputs() == 0)
        {
            if (auto val = expr.dyn_cast<AffineConstantExpr>())
            {
                Value value = b.create<arith::ConstantIndexOp>(loc, val.getValue());
                return value;
            }
        }
        else
        {
            // getNumInputs == 1
            if (expr.dyn_cast<AffineDimExpr>() || expr.dyn_cast<AffineSymbolExpr>())
            {
                return mapOperands[0];
            }
        }
    }

    return llvm::None;
}

SmallVector<unsigned, 4> mlir::findIndicesOfNonZeros(ArrayRef<int64_t> vec)
{
    SmallVector<unsigned, 4> res;
    for (unsigned i = 0; i < vec.size(); ++i)
    {
        if (vec[i] != 0)
        {
            res.push_back(i);
        }
    }
    return res;
}

SmallVector<int64_t, 4> mlir::createOneHot(unsigned size, unsigned offset)
{
    assert(offset < size && "offset should be smaller than size");
    SmallVector<int64_t, 4> res(size, 0);
    res[offset] = 1;
    return res;
}

SmallVector<int64_t, 4> mlir::createSequence(unsigned size)
{
    SmallVector<int64_t, 4> res;
    res.reserve(size);
    for (int64_t i = 0; i < size; ++i)
    {
        res.push_back(i);
    }
    return res;
}

SmallVector<unsigned, 4> mlir::findOperandOffsets(Operation* op, Value value)
{
    SmallVector<unsigned, 4> res;
    for (unsigned i = 0; i < op->getNumOperands(); ++i)
    {
        if (value == op->getOperand(i))
        {
            res.push_back(i);
        }
    }
    return res;
}

Optional<unsigned> mlir::findFirstOperandOffset(Operation* op, Value value)
{
    for (unsigned i = 0; i < op->getNumOperands(); ++i)
    {
        if (value == op->getOperand(i))
        {
            return i;
        }
    }
    return llvm::None;
}

// get referenceIndexingMap of ArgoOp
SmallVector<AffineMap, 8> mlir::getReferenceIndexingMaps(argo::ArgoOp argoOp)
{
    auto mapsRange = argoOp.indexing_maps().getAsRange<AffineMapAttr>();
    auto maps = llvm::to_vector<8>(
        llvm::map_range(mapsRange, [](AffineMapAttr a) { return a.getValue(); }));
    return maps;
}

bool mlir::hasAttrs(ArrayRef<NamedAttribute> attrs,
                    ArrayRef<StringRef> filterAttrs)
{
    // If there are no attributes, then there is nothing to be done.
    if (attrs.empty())
    {
        return false;
    }

    // Filter any attributes in filterAttrs
    SmallVector<NamedAttribute, 8> filteredAttrs(
        llvm::make_filter_range(attrs, [&](NamedAttribute attr) {
            return llvm::is_contained(filterAttrs, attr.getName().strref());
        }));

    return !filteredAttrs.empty();
}

bool mlir::hasAllAttrs(ArrayRef<NamedAttribute> attrs,
                       ArrayRef<StringRef> filterAttrs)
{
    // If there are no attributes, then there is nothing to be done.
    if (attrs.empty())
    {
        return false;
    }

    // Filter any attributes in filterAttrs
    SmallVector<NamedAttribute, 8> filteredAttrs(
        llvm::make_filter_range(attrs, [&](NamedAttribute attr) {
            return llvm::is_contained(filterAttrs, attr.getName().strref());
        }));

    return filteredAttrs.size() == filterAttrs.size();
}

void mlir::removeAttrs(Operation* op, ArrayRef<StringRef> attrs)
{
    for (auto attr : attrs)
    {
        op->removeAttr(attr);
    }
}

// get memSpace
Optional<unsigned> mlir::getMemSpace(Value value)
{
    if (auto memrefType = value.getType().dyn_cast_or_null<MemRefType>())
    {
        return memrefType.getMemorySpaceAsInt();
    }
    return llvm::None;
}

bool mlir::isMemSpace(Value value, ArrayRef<unsigned> spaces)
{
    auto maybeSpace = getMemSpace(value);
    if (!maybeSpace.hasValue())
    {
        return false;
    }

    unsigned space = maybeSpace.getValue();
    return llvm::is_contained(spaces, space);
}

// return true, if value is a func's arg
bool mlir::isFunctionArgument(FuncOp f, Value value)
{

    for (Value arg : f.getArguments())
    {
        if (value == arg)
        {
            return true;
        }
    }
    return false;
}

bool mlir::isDeadAllocDeallocPair(memref::DeallocOp op)
{
    Value view = op.getOperand();
    if (view.hasOneUse())
    {
        return isa_and_nonnull<memref::AllocOp>(view.getDefiningOp());
    }
    return false;
}

Operation* mlir::getCommonParentOp(ArrayRef<Operation*> ops)
{

    if (ops.size() == 0)
    {
        return nullptr;
    }

    auto currentParent = ops.front()->getParentOp();
    if (ops.size() == 1 || currentParent == nullptr)
    {
        return currentParent;
    }

    SmallDenseMap<Operation*, unsigned> parentsOfFirst;

    Operation* commonParent = currentParent;
    unsigned dist = 0;
    while (currentParent != nullptr)
    {
        parentsOfFirst.insert({ currentParent, dist++ });
        currentParent = currentParent->getParentOp();
    }

    unsigned curretDist = 0;
    for (auto op : ops)
    {
        currentParent = op->getParentOp();
        if (currentParent == nullptr)
        {
            return currentParent;
        }

        while (currentParent != nullptr)
        {
            if (parentsOfFirst.count(currentParent) > 0)
            {
                if (parentsOfFirst[currentParent] > curretDist)
                {
                    commonParent = currentParent;
                    curretDist = parentsOfFirst[currentParent];
                }
                break;
            }
            currentParent = currentParent->getParentOp();
        }
    }

    return commonParent;
}

scf::ForOp mlir::cloneAndReplaceSCFForOp(OpBuilder& b, scf::ForOp old, Value lowerBound, Value upperBound, Value step, ValueRange iterArgs)
{

    // create a new ForOp
    scf::ForOp newForOp =
        b.create<scf::ForOp>(old.getLoc(), lowerBound, upperBound, step, iterArgs, [&](OpBuilder& nestedBuilder, Location nestedLoc, Value iv, ValueRange args) {

        });

    // move the loop body
    SmallVector<Operation*, 8> opsToMove;
    auto& oldLoopBody = old.getLoopBody();
    for (auto& block : oldLoopBody)
    {
        for (auto& op : block)
        {
            opsToMove.push_back(&op);
        }
    }

    Region& newLoopBody = newForOp.getLoopBody();
    Block& newBlock = newLoopBody.front();
    for (auto* op : opsToMove)
    {
        op->moveBefore(&newBlock, newBlock.end());
    }

    // replace the induction
    Value oldInd = old.getInductionVar();
    Value newInd = newForOp.getInductionVar();
    oldInd.replaceAllUsesWith(newInd);

    // clone all attr
    newForOp->setAttrs(old->getAttrs());

    return newForOp;
}

/// Given a list of subview ranges, extract individual values for lower, upper
/// bounds and steps and put them into the corresponding vectors.
static void unpackRanges(ArrayRef<Range> ranges, SmallVectorImpl<Value>& lbs, SmallVectorImpl<Value>& ubs, SmallVectorImpl<Value>& steps)
{
    for (Range range : ranges)
    {
        lbs.emplace_back(range.offset);
        ubs.emplace_back(range.size);
        steps.emplace_back(range.stride);
    }
}

void mlir::argo::loopNestBuilder(OpBuilder& b, Location loc, ArrayRef<Range> loopRanges, function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuilderFn, LoopDialectType ldType)
{
    SmallVector<Value, 4> lbs, ubs, steps;
    unpackRanges(loopRanges, lbs, ubs, steps);
    if (ldType == LoopDialectType::kSCFFor)
    {
        buildLoopNest(b, loc, lbs, ubs, steps, bodyBuilderFn);
    }
    else
    {
        SmallVector<int64_t, 4> constSteps = getConstantIndices(steps);

        buildAffineLoopNest(b, loc, lbs, ubs, constSteps, bodyBuilderFn);
    }
}

Value mlir::getLoopLikeOpLowerBound(OpBuilder& b, Location loc, LoopLikeOpInterface looplike)
{
    Operation* op = looplike.getOperation();
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op))
    {
        return forOp.getLowerBound();
    }

    if (AffineForOp forOp = dyn_cast<AffineForOp>(op))
    {
        AffineBound lb = forOp.getLowerBound();
        auto maybeValue = getSimplifiedAffineMap(b, loc, lb.getMap(), lb.getOperands());

        if (maybeValue.hasValue())
        {
            return maybeValue.getValue();
        }

        // default
        return b.create<AffineApplyOp>(loc, lb.getMap(), lb.getOperands());
    }

    llvm_unreachable("unsupported looplike");
}

Value mlir::getLoopLikeOpUpperBound(OpBuilder& b, Location loc, LoopLikeOpInterface looplike)
{
    Operation* op = looplike.getOperation();
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op))
    {
        return forOp.getUpperBound();
    }

    if (AffineForOp forOp = dyn_cast<AffineForOp>(op))
    {
        AffineBound lb = forOp.getUpperBound();
        auto maybeValue = getSimplifiedAffineMap(b, loc, lb.getMap(), lb.getOperands());

        if (maybeValue.hasValue())
        {
            return maybeValue.getValue();
        }

        // default
        return b.create<AffineApplyOp>(loc, lb.getMap(), lb.getOperands());
    }

    llvm_unreachable("unsupported looplike");
}

Value mlir::getLoopLikeOpInductionVar(LoopLikeOpInterface looplike)
{
    Operation* op = looplike.getOperation();
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op))
    {
        return forOp.getInductionVar();
    }

    if (AffineForOp forOp = dyn_cast<AffineForOp>(op))
    {
        return forOp.getInductionVar();
    }

    llvm_unreachable("unsupported looplike");
}

Block* mlir::getLoopLikeOpBody(LoopLikeOpInterface looplike)
{
    Operation* op = looplike.getOperation();
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op))
    {
        return forOp.getBody();
    }

    if (AffineForOp forOp = dyn_cast<AffineForOp>(op))
    {
        return forOp.getBody();
    }

    llvm_unreachable("unsupported looplike");
}

static Value emitOrFoldComposedAffineApply(OpBuilder& b, Location loc, AffineMap map, ArrayRef<Value> operandsRef, OperationFolder* folder)
{
    SmallVector<Value, 4> operands(operandsRef.begin(), operandsRef.end());
    canonicalizeMapAndOperands(&map, &operands);
    return folded_affine_apply(b, loc, folder, map, operands);
    // return affine_apply(map, operands);
}

SmallVector<Value, 4> mlir::argo::applyMapToValues(OpBuilder& b, Location loc, AffineMap map, ArrayRef<Value> values, OperationFolder* folder)
{
    SmallVector<Value, 4> res;
    res.reserve(map.getNumResults());
    unsigned numDims = map.getNumDims();
    unsigned numSym = map.getNumSymbols();

    // For each `expr` in `map`, applies the `expr` to the values extracted from
    // ranges. If the resulting application can be folded into a Value, the
    // folding occurs eagerly. Otherwise, an affine.apply operation is emitted.

    for (auto expr : map.getResults())
    {
        AffineMap mapNew = AffineMap::get(numDims, numSym, expr);
        res.push_back(
            emitOrFoldComposedAffineApply(b, loc, mapNew, values, folder));
    }
    return res;
}

/// Returns all the operands of `argoOp` that are not views.
/// Asserts that these operands are value types to allow transformations like
/// tiling to just use the values when cloning `argoOp`.
SmallVector<Value, 4> mlir::argo::getAssumedNonViewOperands(ArgoOp argoOp)
{
    auto* op = argoOp.getOperation();
    unsigned numViews = argoOp.getNumInputsAndOutputBuffers();
    unsigned nOperands = op->getNumOperands() - numViews;
    SmallVector<Value, 4> res;
    res.reserve(nOperands);
    for (unsigned i = 0; i < nOperands; ++i)
    {
        res.push_back(op->getOperand(numViews + i));
        auto t = res.back().getType();
        (void)t;
        assert((t.isIntOrIndexOrFloat() || t.isa<VectorType>()) &&
               "expected scalar or vector type");
    }
    return res;
}

Value mlir::argo::AllocFromView(OpBuilder& b, Location loc, Value value, OperationFolder* folder, int64_t memorySpace)
{

    unsigned space = 0;
    if (auto memrefType = value.getType().dyn_cast_or_null<MemRefType>())
    {
        space = memorySpace >= 0 ? memorySpace : memrefType.getMemorySpaceAsInt();
    }

    if (auto subviewOp = dyn_cast_or_null<memref::SubViewOp>(value.getDefiningOp()))
    {
        auto memrefType = subviewOp.getType().cast<MemRefType>();
        const auto& shape = memrefType.getShape();
        SmallVector<Value, 4> shapeValue;

        unsigned idx = 0;
        for (auto& range : subviewOp.getOrCreateRanges(b, loc))
        {
            if (shape[idx++] == argo::kDynamicSize)
            {
                shapeValue.push_back(range.size);
            }
        }

        auto type =
            MemRefType::get(shape, subviewOp.getType().getElementType(), {}, space);
        return b.create<memref::AllocOp>(loc, type, shapeValue);
    }
    else if (auto memrefType = value.getType().dyn_cast_or_null<MemRefType>())
    {
        const auto& shape = memrefType.getShape();
        auto type = MemRefType::get(shape, memrefType.getElementType(), memrefType.getLayout().getAffineMap(), space);

        SmallVector<Value, 4> shapeValue;
        shapeValue.reserve(shape.size());
        for (unsigned idx = 0, n = shape.size(); idx < n; ++idx)
        {
            if (shape[idx] == argo::kDynamicSize)
            {
                shapeValue.push_back(folded_std_dim(b, loc, folder, value, idx));
            }
        }
        return b.create<memref::AllocOp>(loc, type, shapeValue);
    }

    assert("Value is not a view");
    return Value();
}

// return true if we can convert srcTy to dstTy
bool mlir::argo::isConvertibleTypes(Type srcTy, Type dstTy)
{
    return (srcTy.isa<FloatType>() && dstTy.isSignlessInteger()) ||
           (srcTy.isSignlessInteger() && dstTy.isa<FloatType>()) ||
           (srcTy.isa<FloatType>() && dstTy.isa<FloatType>()) ||
           (srcTy.isSignlessInteger() && srcTy.isSignlessInteger());
}
