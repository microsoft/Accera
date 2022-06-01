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

RangeValue resolveThreadIdRange(Operation* op, llvm::StringRef dimId)
{
    auto upperBound = GetBlockDimSize(op, dimId.str());
    return RangeValue(0, upperBound - 1); // -1 because RangeValue will add 1 to the upper bound and the thread id never takes on the upperBound value
}

RangeValue resolveBlockIdRange(Operation* op, llvm::StringRef dimId)
{
    auto upperBound = GetGridDimSize(op, dimId.str());
    return RangeValue(0, upperBound - 1); // -1 because RangeValue will add 1 to the upper bound and the block id never takes on the upperBound value
}

RangeValue resolveBlockDimRange(Operation* op, llvm::StringRef dimId)
{
    auto upperBound = GetBlockDimSize(op, dimId.str());
    return RangeValue(upperBound, upperBound);
}

RangeValue resolveGridDimRange(Operation* op, llvm::StringRef dimId)
{
    auto upperBound = GetGridDimSize(op, dimId.str());
    return RangeValue(upperBound, upperBound);
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

RangeValueAnalysis::RangeValueAnalysis(mlir::Operation* rootOp)
{
    rootOp->walk([&](mlir::Operation* op) {
        if (!op->hasTrait<OpTrait::SymbolTable>())
        {
            addOperation(op);
        }
    });
}

RangeValueAnalysis::RangeValueAnalysis(const std::vector<mlir::Operation*>& ops)
{
    for (auto op : ops)
    {
        addOperation(op);
    }
}

bool RangeValueAnalysis::hasRange(Value value) const
{
    return _rangeMap.find(value) != _rangeMap.end();
}

RangeValue RangeValueAnalysis::getRange(Value value) const
{
    if (!hasRange(value))
    {
        return RangeValue();
    }
    auto it = _rangeMap.find(value);
    assert(it != _rangeMap.end());
    return it->second;
}

RangeValue RangeValueAnalysis::addOperation(mlir::Operation* op)
{
    if (op->getNumResults() > 1)
    {
        // Only operations with 0 or 1 results can have their ranges tracked successfully currently
        return RangeValue();
    }
    // Don't re-add ops we already have
    bool allResultsTracked = op->getNumResults() > 0;
    RangeValue existingRV;
    for (auto res : op->getResults())
    {
        if (hasRange(res))
        {
            existingRV = getRange(res);
        }
        else
        {
            allResultsTracked = false;
        }
    }
    if (allResultsTracked)
    {
        return existingRV;
    }

    // Ensure this op's operands are part of this analysis before resolving this op range
    for (auto operand : op->getOperands())
    {
        if (auto definingOp = GetDefiningOpOrForLoop(operand))
        {
            addOperation(definingOp);
        }
    }

    auto range = resolveRangeValue(op);
    mlir::TypeSwitch<Operation*>(op)
        .Case([&](scf::ForOp op) { _rangeMap.insert({ op.getInductionVar(), range }); })
        .Case([&](AffineForOp op) { _rangeMap.insert({ op.getInductionVar(), range }); })
        .Default([&](Operation* op) {
            for (auto res : op->getResults())
            {
                _rangeMap.insert({ res, range });
            }
        });
    return range;
}

bool RangeValueAnalysis::allOperandsHaveRanges(Operation* op)
{
    return llvm::all_of(op->getOperands(), [&, this](Value operand) {
        return _rangeMap.find(operand) != _rangeMap.end();
    });
}

SmallVector<RangeValue, 3> RangeValueAnalysis::resolveOperands(Operation* op)
{
    SmallVector<RangeValue, 3> operands;
    transform(op->getOperands(), std::back_inserter(operands), [&](Value operand) {
        if (hasRange(operand))
        {
            return _rangeMap[operand];
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
    if (hasRange(val))
    {
        return getRange(val);
    }
    else if (auto defOp = val.getDefiningOp())
    {
        return resolveRangeValue(defOp);
    }
    // otherwise this is a BlockArgument which conservatively we assume has no range
    return RangeValue();
}

RangeValue RangeValueAnalysis::resolveRangeValue(gpu::ThreadIdOp op)
{
    return resolveThreadIdRange(op, op.dimension());
}

RangeValue RangeValueAnalysis::resolveRangeValue(ROCDL::ThreadIdXOp op)
{
    return resolveThreadIdRange(op, "x");
}

RangeValue RangeValueAnalysis::resolveRangeValue(ROCDL::ThreadIdYOp op)
{
    return resolveThreadIdRange(op, "y");
}

RangeValue RangeValueAnalysis::resolveRangeValue(ROCDL::ThreadIdZOp op)
{
    return resolveThreadIdRange(op, "z");
}

RangeValue RangeValueAnalysis::resolveRangeValue(gpu::BlockIdOp op)
{
    return resolveBlockIdRange(op, op.dimension());
}

RangeValue RangeValueAnalysis::resolveRangeValue(ROCDL::BlockIdXOp op)
{
    return resolveBlockIdRange(op, "x");
}

RangeValue RangeValueAnalysis::resolveRangeValue(ROCDL::BlockIdYOp op)
{
    return resolveBlockIdRange(op, "y");
}

RangeValue RangeValueAnalysis::resolveRangeValue(ROCDL::BlockIdZOp op)
{
    return resolveBlockIdRange(op, "z");
}

RangeValue RangeValueAnalysis::resolveRangeValue(mlir::gpu::BlockDimOp op)
{
    return resolveBlockDimRange(op, op.dimension());
}

RangeValue RangeValueAnalysis::resolveRangeValue(mlir::ROCDL::BlockDimXOp op)
{
    return resolveBlockDimRange(op, "x");
}

RangeValue RangeValueAnalysis::resolveRangeValue(mlir::ROCDL::BlockDimYOp op)
{
    return resolveBlockDimRange(op, "y");
}

RangeValue RangeValueAnalysis::resolveRangeValue(mlir::ROCDL::BlockDimZOp op)
{
    return resolveBlockDimRange(op, "z");
}

RangeValue RangeValueAnalysis::resolveRangeValue(mlir::gpu::GridDimOp op)
{
    return resolveGridDimRange(op, op.dimension());
}

RangeValue RangeValueAnalysis::resolveRangeValue(mlir::ROCDL::GridDimXOp op)
{
    return resolveGridDimRange(op, "x");
}
RangeValue RangeValueAnalysis::resolveRangeValue(mlir::ROCDL::GridDimYOp op)
{
    return resolveGridDimRange(op, "y");
}
RangeValue RangeValueAnalysis::resolveRangeValue(mlir::ROCDL::GridDimZOp op)
{
    return resolveGridDimRange(op, "z");
}

RangeValue RangeValueAnalysis::resolveRangeValue(Instruction::BinaryOps binOp, mlir::Operation* op)
{
    auto operands = resolveOperands(op);
    return operands[0].binaryOp(binOp, operands[1]);
}
RangeValue RangeValueAnalysis::resolveRangeValue(AffineForOp op)
{
    return op.hasConstantBounds() ? RangeValue(op.getConstantLowerBound(), op.getConstantUpperBound() - op.getStep()) : RangeValue();
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
        .Case([&](ROCDL::ThreadIdXOp op) { return resolveRangeValue(op); })
        .Case([&](ROCDL::ThreadIdYOp op) { return resolveRangeValue(op); })
        .Case([&](ROCDL::ThreadIdZOp op) { return resolveRangeValue(op); })
        .Case([&](gpu::BlockIdOp op) { return resolveRangeValue(op); })
        .Case([&](ROCDL::BlockIdXOp op) { return resolveRangeValue(op); })
        .Case([&](ROCDL::BlockIdYOp op) { return resolveRangeValue(op); })
        .Case([&](ROCDL::BlockIdZOp op) { return resolveRangeValue(op); })
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