////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "util/RangeValueUtilities.h"

#include "AcceraPasses.h"

#include <ir/include/IRUtil.h>

#include <llvm/IR/GlobalValue.h>
#include <mlir/Analysis/DataFlowAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/FoldUtils.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/IR/ConstantRange.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_os_ostream.h>

#include <algorithm>

#define DEBUG_TYPE "value-optimize"

using namespace mlir;

using namespace accera::ir;
using namespace accera::transforms;
using namespace accera::ir::value;
using namespace accera::ir::util;

using llvm::CmpInst;
using llvm::ConstantRange;
using llvm::Instruction;

namespace
{
int dimIndexToInteger(llvm::StringRef dim)
{
    return ::llvm::StringSwitch<int>(dim)
        .Case("x", 0)
        .Case("y", 1)
        .Case("z", 2)
        .Default(-1);
}

} // namespace

namespace accera::ir::util
{

RangeValue::RangeValue()
{
    range = ConstantRange::getFull(maxBitWidth);
}
RangeValue::RangeValue(const ConstantRange& range_) :
    range(range_)
{
}
RangeValue::RangeValue(int64_t min_, int64_t max_)
{
    range = ConstantRange::getNonEmpty(APInt(maxBitWidth, min_, true), APInt(maxBitWidth, max_ + 1, true));
}
RangeValue::RangeValue(APInt min_, APInt max_)
{
    if (min_.isSingleWord() && max_.isSingleWord())
    {
        range = ConstantRange::getNonEmpty(
            APInt(maxBitWidth, min_.getSExtValue(), true),
            APInt(maxBitWidth, max_.getSExtValue(), true) + 1);
    }
    else
    {
        // is not an int64_t, then the range is not valid
        range = ConstantRange::getFull(maxBitWidth);
    }
}

RangeValue RangeValue::binaryOp(Instruction::BinaryOps op, const RangeValue& other) const
{
    return range.binaryOp(op, other.range);
}

bool RangeValue::icmp(CmpInst::Predicate op, const RangeValue& other) const
{
    return range.icmp(op, other.range);
}

bool RangeValue::operator==(const RangeValue& other) const
{
    return range == other.range;
}

bool RangeValue::contains(APInt value) const
{
    return range.contains(value);
}

bool RangeValue::isFullSet() const
{
    return range.isFullSet();
}

bool RangeValue::isConstant() const
{
    return !range.isFullSet() && (range.getLower() + 1 == range.getUpper());
}

DictionaryAttr RangeValue::asAttr(MLIRContext* ctx) const
{
    mlir::NamedAttrList entries;
    entries.set("lower_bound", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), range.getLower()));
    entries.set("upper_bound", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), range.getUpper()));
    return DictionaryAttr::get(ctx, entries);
}

RangeValueAnalysis::RangeValueAnalysis(Operation* rootOp)
{
    llvm::SmallPtrSet<Operation*, 16> worklist;
    rootOp->walk([&](Operation* op) {
        if (!op->hasTrait<OpTrait::SymbolTable>())
        {
            worklist.insert(op);
        }
    });

    while (!worklist.empty())
    {
        auto nextOp = llvm::find_if(worklist, [&, this](Operation* op) {
            return allOperandsHaveRanges(op);
        });
        if (nextOp == worklist.end())
            break;
        Operation* op = *nextOp;
        worklist.erase(op);

        auto range = resolveRangeValue(op);
        mlir::TypeSwitch<Operation*>(op)
            .Case([&](scf::ForOp op) { rangeMap.insert({ op.getInductionVar(), range }); })
            .Case([&](AffineForOp op) { rangeMap.insert({ op.getInductionVar(), range }); })
            .Default([&](Operation* op) {
                for (auto res : op->getResults())
                {
                    rangeMap.insert({ res, range });
                }
            });
    }
}

bool RangeValueAnalysis::hasRange(Value value) const
{
    return rangeMap.find(value) != rangeMap.end();
}

RangeValue RangeValueAnalysis::getRange(Value value) const
{
    if (!hasRange(value))
    {
        return RangeValue();
    }
    auto it = rangeMap.find(value);
    assert(it != rangeMap.end());
    return it->second;
}

bool RangeValueAnalysis::allOperandsHaveRanges(Operation* op)
{
    return llvm::all_of(op->getOperands(), [&, this](Value operand) {
        return rangeMap.find(operand) != rangeMap.end();
    });
}

SmallVector<RangeValue, 3> RangeValueAnalysis::resolveOperands(Operation* op)
{
    SmallVector<RangeValue, 3> operands;
    transform(op->getOperands(), std::back_inserter(operands), [&](Value operand) {
        if (hasRange(operand))
        {
            return rangeMap[operand];
        }
        return RangeValue();
    });
    return operands;
}

RangeValue RangeValueAnalysis::resolveRangeValue(ConstantOp op)
{
    auto attr = op.getValue();
    if (auto value = attr.dyn_cast<IntegerAttr>())
    {
        return RangeValue(value.getValue(), value.getValue());
    }
    return RangeValue();
}
RangeValue RangeValueAnalysis::resolveRangeValue(ConstantIndexOp op)
{
    auto value = op.getValue();
    return RangeValue(value, value);
}
RangeValue RangeValueAnalysis::resolveRangeValue(ConstantIntOp op)
{
    auto value = op.getValue();
    return RangeValue(value, value);
}
RangeValue RangeValueAnalysis::resolveRangeValue(IndexCastOp op)
{
    auto val = op.in();
    if (auto defOp = val.getDefiningOp())
    {
        return resolveRangeValue(defOp);
    }
    // otherwise this is a BlockArgument which conservatively we assume has no range
    return RangeValue();
}
RangeValue RangeValueAnalysis::resolveRangeValue(gpu::ThreadIdOp op)
{
    auto gpuMod = op->getParentOfType<gpu::GPUFuncOp>();
    if (!gpuMod)
    {
        return RangeValue();
    }
    auto blockIdxAttr = gpuMod->getAttrOfType<ArrayAttr>("blockSize");
    auto blockDimIdx = dimIndexToInteger(op.dimension());
    if (!blockIdxAttr || blockDimIdx == -1)
    {
        return RangeValue();
    }
    auto upperBound = blockIdxAttr.getValue()[blockDimIdx].cast<IntegerAttr>().getInt();
    return RangeValue(0, upperBound);
}
RangeValue RangeValueAnalysis::resolveRangeValue(gpu::BlockIdOp op)
{
    auto gpuMod = op->getParentOfType<gpu::GPUFuncOp>();
    if (!gpuMod)
    {
        return RangeValue();
    }
    auto gridIdxAttr = gpuMod->getAttrOfType<ArrayAttr>("gridSize");
    auto gridDimIdx = dimIndexToInteger(op.dimension());
    if (!gridIdxAttr || gridDimIdx == -1)
    {
        return RangeValue();
    }
    auto upperBound = gridIdxAttr.getValue()[gridDimIdx].cast<IntegerAttr>().getInt();
    return RangeValue(0, upperBound);
}
RangeValue RangeValueAnalysis::resolveRangeValue(Instruction::BinaryOps binOp, mlir::Operation* op)
{
    auto operands = resolveOperands(op);
    return operands[0].binaryOp(binOp, operands[1]);
}
RangeValue RangeValueAnalysis::resolveRangeValue(AffineForOp op)
{
    return op.hasConstantBounds() ? RangeValue(op.getConstantLowerBound(), op.getConstantUpperBound()) : RangeValue();
}
RangeValue RangeValueAnalysis::resolveRangeValue(scf::ForOp op)
{
    assert(op.getNumInductionVars() == 1);
    RangeValue lowerBound = resolveRangeValue(op.lowerBound().getDefiningOp());
    RangeValue upperBound = resolveRangeValue(op.upperBound().getDefiningOp());
    return lowerBound.isConstant() && upperBound.isConstant() ? RangeValue(lowerBound.range.getLower(), upperBound.range.getUpper() - 1) : RangeValue();
}
RangeValue RangeValueAnalysis::resolveRangeValue(mlir::Operation* op)
{
    return mlir::TypeSwitch<mlir::Operation*, RangeValue>(op)
        .Case([&](ConstantOp op) { return resolveRangeValue(op); })
        .Case([&](ConstantIndexOp op) { return resolveRangeValue(op); })
        .Case([&](ConstantIntOp op) { return resolveRangeValue(op); })
        .Case([&](IndexCastOp op) { return resolveRangeValue(op); })
        .Case([&](gpu::ThreadIdOp op) { return resolveRangeValue(op); })
        .Case([&](gpu::BlockIdOp op) { return resolveRangeValue(op); })
        .Case([&](AddIOp op) { return resolveRangeValue(Instruction::BinaryOps::Add, op); })
        .Case([&](SubIOp op) { return resolveRangeValue(Instruction::BinaryOps::Sub, op); })
        .Case([&](MulIOp op) { return resolveRangeValue(Instruction::BinaryOps::Mul, op); })
        .Case([&](SignedRemIOp op) { return resolveRangeValue(Instruction::BinaryOps::SRem, op); })
        .Case([&](UnsignedRemIOp op) { return resolveRangeValue(Instruction::BinaryOps::URem, op); })
        .Case([&](SignedDivIOp op) { return resolveRangeValue(Instruction::BinaryOps::SDiv, op); })
        .Case([&](UnsignedDivIOp op) { return resolveRangeValue(Instruction::BinaryOps::UDiv, op); })
        .Case([&](scf::ForOp op) { return resolveRangeValue(op); })
        .Case([&](AffineForOp op) { return resolveRangeValue(op); })
        .Default([&](mlir::Operation*) { return RangeValue(); });
}

} // namespace accera::ir::util