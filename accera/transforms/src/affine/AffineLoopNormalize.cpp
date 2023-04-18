////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "affine/AffineLoopNormalize.h"

#include "AcceraPasses.h"

#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <memory>
#include <vector>

namespace
{

// mlir::normalizeAffineFor has a bug where AffineForOp map bounds are treated as having the same operands.
// To work around this bug, adjust the lower and upper bound maps to have the same operands and adjust
// the upper bound map's indexing accordingly

// E.g. given
//      affine.for %arg5 = affine_map<()[s0] -> (-s0)>()[%arg4] to affine_map<()[s0, s1] -> (s0 - s1)>()[%arg0, %arg3]
//          without adjusting, mlir::normalizeAffineFor would turn this into:
//      affine.for %arg5 = 0 to affine_map<()[s0, s1, s2] -> (s0 - s1 + s0)>()[%arg0, %0, %1]
//          this term should be s2 now, though it *was* s0 before ------^
// So to work around this bug, we change the for loop maps to be:
//      affine.for %arg5 = affine_map<()[s0, s1, s2] -> (-s0)>()[%arg4, %arg0, %arg3] to affine_map<()[s0, s1, s2] -> (s1 - s2)>()[%arg4, %arg0, %arg3]
// Where we simply concatenate operands for each map together, and adjust the indexing used by the upper bound map to account
// for however many dims and symbols the lower bound map had initially
void workaroundModifyAffineForOp(mlir::AffineForOp& op)
{
    mlir::AffineBound lb = op.getLowerBound();
    mlir::AffineBound ub = op.getUpperBound();
    mlir::AffineMap origLbMap = lb.getMap();
    mlir::AffineMap origUbMap = ub.getMap();
    auto lbDimCount = origLbMap.getNumDims();
    auto lbSymCount = origLbMap.getNumSymbols();
    auto ubDimCount = origUbMap.getNumDims();
    auto ubSymCount = origUbMap.getNumSymbols();

    // Adjust the number of dims and syms to the LB map, but otherwise it doesn't need to change
    mlir::MutableAffineMap mutableLbMap(origLbMap);
    mutableLbMap.setNumDims(lbDimCount + ubDimCount);
    mutableLbMap.setNumSymbols(lbSymCount + ubSymCount);
    mlir::AffineMap newLbMap = mutableLbMap.getAffineMap();

    // Adjust the number of dims and syms to the UB map, and also shift its expressions by the number of lb dims and syms
    mlir::AffineMap shiftedUbMap = origUbMap.shiftDims(lbDimCount).shiftSymbols(lbSymCount);
    mlir::MutableAffineMap mutableShiftedUbMap(shiftedUbMap);
    mutableShiftedUbMap.setNumDims(lbDimCount + ubDimCount);
    mutableShiftedUbMap.setNumSymbols(lbSymCount + ubSymCount);
    mlir::AffineMap newUbMap = mutableShiftedUbMap.getAffineMap();

    // interleave [ lbDims..., ubDims..., lbSyms..., ubSyms... ] because dim operands occur before symbol operands when applying a map
    std::vector<mlir::Value> combinedOperands;
    combinedOperands.reserve(ub.getNumOperands() + lb.getNumOperands());
    combinedOperands.insert(combinedOperands.end(), lb.operandBegin(), lb.operandBegin() + lbDimCount);
    combinedOperands.insert(combinedOperands.end(), ub.operandBegin(), ub.operandBegin() + ubDimCount);
    combinedOperands.insert(combinedOperands.end(), lb.operandBegin() + lbDimCount, lb.operandEnd());
    combinedOperands.insert(combinedOperands.end(), ub.operandBegin() + ubDimCount, ub.operandEnd());

    // Now we have our new maps and operands, so adjust the given for op and return
    op.setLowerBound(combinedOperands, newLbMap);
    op.setUpperBound(combinedOperands, newUbMap);
}

struct AcceraAffineLoopNormalizePass : public accera::transforms::AcceraAffineLoopNormalizeBase<AcceraAffineLoopNormalizePass>
{
    // This pass and these patterns only differs from the builtin mlir AffineLoopNormalize pass in that it adjusts the maps on AffineForOps
    // before calling into the mlir affine loop normalize function in order to work around a bug in that implementation
    void runOnOperation() final
    {
        auto op = getOperation();

        // See <llvm-project>\mlir\lib\Dialect\Affine\Transforms\AffineLoopNormalize.cpp
        op->walk([](mlir::AffineForOp affineForOp) {
            workaroundModifyAffineForOp(affineForOp);
            (void)mlir::normalizeAffineFor(affineForOp);
        });
    }
};

} // namespace

namespace accera::transforms::affine
{
std::unique_ptr<mlir::Pass> createAcceraAffineLoopNormalizePass()
{
    return std::make_unique<AcceraAffineLoopNormalizePass>();
}
} // namespace accera::transforms::affine
