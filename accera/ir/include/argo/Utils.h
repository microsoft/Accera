////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MLIR_DIALECT_ARGO_UTILS_H_
#define MLIR_DIALECT_ARGO_UTILS_H_

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "ArgoOps.h"

namespace mlir
{

namespace argo {
    namespace {
            constexpr int64_t kDynamicSize = -1;
    }
}

class AffineExpr;
class AffineMap;
class OperationFolder;

// utilities
bool isConstantZero(Value v);
Value createCeilDivIndex(OpBuilder& b, Location loc, Value lhs, Value rhs, bool useAffine = true, OperationFolder* folder = nullptr);
int64_t getConstantIndex(Value, int64_t dynVal = argo::kDynamicSize);
SmallVector<int64_t, 4>
getConstantIndices(ArrayRef<Value> values,
                   int64_t dynVal = argo::kDynamicSize);

// simplify a trivial affineMap to a constant if possible
Optional<Value> getSimplifiedAffineMap(OpBuilder& b, Location loc, AffineMap map, ValueRange mapOperands);

// todo remove inline
inline int64_t getMemrefOrTensorShape(Value memrefOrTensor, unsigned index)
{
    Type type = memrefOrTensor.getType();
    int64_t dim = argo::kDynamicSize;
    if (auto memref = type.dyn_cast<MemRefType>())
    {
        dim = memref.getShape()[index];
    }
    else if (auto tensor = type.dyn_cast<TensorType>())
    {
        dim = tensor.getShape()[index];
    }
    return dim;
}

// Find indices of non-zero value in vec
// E.g. vec = {0, 1, 2, 0, -1}  => {1, 2, 4}
SmallVector<unsigned, 4> findIndicesOfNonZeros(ArrayRef<int64_t> vec);

// Create a vec with a size `size` containing only 1 at vec[offset] = 1, rest
// are 0.
SmallVector<int64_t, 4> createOneHot(unsigned size, unsigned offset);

// Create a vec with a size `size`, and v[i] = i for 0 <= i < size
SmallVector<int64_t, 4> createSequence(unsigned size);

// Find offsets of operands of an op that is equal to value
// E.g. op(other0, value, other1, value) => {1, 3}
SmallVector<unsigned, 4> findOperandOffsets(Operation* op, Value value);

// Find first offset of operands of an op that is equal to value
// E.g. op(other0, value, other1, value) => 1
Optional<unsigned> findFirstOperandOffset(Operation* op, Value value);

// return true, if `attrs` has at least an attr in `filterAttrs`
bool hasAttrs(ArrayRef<NamedAttribute> attrs, ArrayRef<StringRef> filterAttrs);

// return true, if `attrs` has all attr in `filterAttrs`
bool hasAllAttrs(ArrayRef<NamedAttribute> attrs,
                 ArrayRef<StringRef> filterAttrs);

void removeAttrs(Operation* op, ArrayRef<StringRef> attrs);

// get referenceIndexingMap of ArgoOp
SmallVector<AffineMap, 8> getReferenceIndexingMaps(argo::ArgoOp);

// get memSpace
Optional<unsigned> getMemSpace(Value value);

// is memSpace
bool isMemSpace(Value, ArrayRef<unsigned>);

// return true, if value is a func's arg
bool isFunctionArgument(FuncOp f, Value value);

bool isDeadAllocDeallocPair(memref::DeallocOp op);

// return the common parent of a list of ops
Operation* getCommonParentOp(ArrayRef<Operation*> ops);

// clone an existing scf::ForOp with modified parameters
scf::ForOp cloneAndReplaceSCFForOp(OpBuilder& b, scf::ForOp old, Value lowerBound, Value upperBound, Value step, ValueRange iterArgs);

namespace argo
{
    enum class LoopDialectType
    {
        kSCFFor = 0,
        kAffineFor = 1,
    };

    void loopNestBuilder(OpBuilder& b,
                         Location loc,
                         ArrayRef<Range> loopRanges,
                         function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuilderFn,
                         LoopDialectType ldType = LoopDialectType::kAffineFor);

} // namespace argo

// Helper functions for retrieving common members from LoopLikeOpInterface.
// Currently, only scf::ForOp and AffineForOp are supported.
Value getLoopLikeOpLowerBound(OpBuilder& b, Location loc, LoopLikeOpInterface looplike);
Value getLoopLikeOpUpperBound(OpBuilder& b, Location loc, LoopLikeOpInterface looplike);
Value getLoopLikeOpInductionVar(LoopLikeOpInterface looplike);
Block* getLoopLikeOpBody(LoopLikeOpInterface looplike);

namespace argo
{
    /// Returns the linearized list of all view dimensions in an argoOp. Applying
    /// the inverse, concatenated loopToOperandRangeMaps to this list allows the
    /// derivation of loop ranges for any argoOp.
    template <typename ConcreteOp>
    SmallVector<Value, 8> getViewSizes(OpBuilder& builder, ConcreteOp argoOp)
    {
        auto loc = argoOp.getLoc();
        SmallVector<Value, 8> res;
        SmallVector<unsigned, 4> ranks;

        for (auto v : argoOp.getInputsAndOutputBuffers())
        {
            MemRefType t = v.getType().template cast<MemRefType>();
            auto shape = t.getShape();
            ranks.push_back(t.getRank());
            for (unsigned i = 0; i < t.getRank(); ++i)
            {
                if (mlir::ShapedType::isDynamic(shape[i]))
                {
                    res.push_back(builder.create<memref::DimOp>(loc, v, i));
                }
                else
                {
                    res.push_back(builder.create<ConstantIndexOp>(loc, shape[i]));
                }
            }
        }

        // handle generic ops
        auto attr = argoOp.template getAttrOfType<IntegerAttr>("symbol_source");
        if (attr)
        {
            // Find the correct position for inserting values for symbols.
            unsigned numSymb = ranks[attr.getInt()], symbolsPos = 0;
            for (unsigned idx = 0; idx < attr.getInt(); idx++)
                symbolsPos += ranks[idx];

            // Append the end of the value list that corresponds to the
            // values mapping to symbols. Since inside concatinated map symbols are
            // repeated we have to repeat the sizes as well.

            // Reserve is mandatory to avoid a potential undefined behavior with
            // pushing back to smallvector from itself.
            res.reserve(res.size() + ranks.size() * numSymb);
            for (unsigned idx = 0, s = ranks.size(); idx < s; ++idx)
                for (unsigned idx2 = 0; idx2 < numSymb; ++idx2)
                    res.push_back(res[symbolsPos + idx2]);
        }

        return res;
    }

    /// Returns the values obtained by applying `map` to the list of values.
    /// When non-null, the optional pointer `folder` is used to call into the
    /// `createAndFold` builder method. If `folder` is null, the regular `create`
    /// method is called.
    SmallVector<Value, 4> applyMapToValues(OpBuilder& b, Location loc, AffineMap map, ArrayRef<Value> values, OperationFolder* folder = nullptr);

    // Returns all the operands of `argoOp` that are not views.
    /// Asserts that these operands are value types to allow transformations like
    /// tiling to just use the values when cloning `argoOp`.
    llvm::SmallVector<Value, 4> getAssumedNonViewOperands(ArgoOp argoOp);

    // Alloc the same Memref as a Subview
    mlir::Value AllocFromView(OpBuilder& b, Location loc, Value view, OperationFolder* folder, int64_t memorySpace = -1);
    // return true if we can convert srcTy to dstTy
    bool isConvertibleTypes(Type srcTy, Type dstTy);

} // namespace argo
} // namespace mlir

#endif // MLIR_DIALECT_ARGO_UTILS_H_
