////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/IRUtil.h>

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
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_os_ostream.h>

#include <algorithm>

#define DEBUG_TYPE "value-optimize"

using namespace mlir;

using namespace accera::ir;
using namespace accera::transforms;
using namespace accera::ir::value;

namespace
{
struct RangeValue
{
    static constexpr int maxBitWidth = 64;
    APInt min;
    APInt max;
    RangeValue()
    {
        min = negInf();
        max = inf();
    }
    RangeValue(int64_t min_, int64_t max_)
    {
        min = APInt(maxBitWidth, min_, true);
        max = APInt(maxBitWidth, max_, true);
    }
    RangeValue(APInt min, APInt max) :
        min(min), max(max) {}

    RangeValue(DictionaryAttr dict)
    {
        auto lowerBoundAttr = dict.getAs<IntegerAttr>("lower_bound");
        auto upperBoundAttr = dict.getAs<IntegerAttr>("upper_bound");
        if (lowerBoundAttr)
        {
            min = lowerBoundAttr.getValue();
        }
        else
        {
            min = negInf();
        }
        if (upperBoundAttr)
        {
            max = upperBoundAttr.getValue();
        }
        else
        {
            max = inf();
        }
    }

    // [a,b] + [c,d] = [a+c,b+d]
    RangeValue operator+(const RangeValue& other) const
    {
        APInt lowerBound, upperBound;
        if (min.eq(negInf()) || other.min.eq(negInf()))
        {
            lowerBound = negInf();
        }
        else
        {
            bool overflows = false;
            lowerBound = min.sadd_ov(other.min, overflows);
            if (overflows)
            {
                lowerBound = negInf();
            }
        }
        if (max.eq(inf()) || other.max.eq(inf()))
        {
            upperBound = inf();
        }
        else
        {
            bool overflows = false;
            upperBound = max.sadd_ov(other.max, overflows);
            if (overflows)
            {
                upperBound = inf();
            }
        }
        return RangeValue(lowerBound, upperBound);
    }

    // [a,b] - [c,d] = [a-d,b-c]
    RangeValue operator-(const RangeValue& other) const
    {
        APInt lowerBound, upperBound;
        if (min.eq(negInf()) || other.max.eq(inf()))
        {
            lowerBound = negInf();
        }
        else
        {
            bool overflows = false;
            lowerBound = min.ssub_ov(other.max, overflows);
            if (overflows)
            {
                lowerBound = negInf();
            }
        }
        if (max.eq(inf()) || other.min.eq(negInf()))
        {
            upperBound = inf();
        }
        else
        {
            bool overflows = false;
            upperBound = max.ssub_ov(other.min, overflows);
            if (overflows)
            {
                upperBound = inf();
            }
        }
        return RangeValue(lowerBound, upperBound);
    }

    // [a, b] * [c, d] = [Min(a*c, a*d, b*c, b*d), Max(a*c, a*d, b*c, b*d)]
    RangeValue operator*(const RangeValue& other) const
    {
        APInt acMin, acMax, adMin, adMax, bcMin, bcMax, bdMin, bdMax;
#define MUL(targetMin, targetMax, a, b)     \
    {                                       \
        bool overflows = false;             \
        auto tmp = a.smul_ov(b, overflows); \
        if (overflows)                      \
        {                                   \
            targetMin = inf();              \
            targetMax = negInf();           \
        }                                   \
        else                                \
        {                                   \
            targetMin = tmp;                \
            targetMax = tmp;                \
        }                                   \
    }
        MUL(acMin, acMax, min, other.min);
        MUL(adMin, adMax, min, other.max);
        MUL(bcMin, bcMax, max, other.min);
        MUL(bdMin, bdMax, max, other.max);

#define APMIN(a, b) (a.slt(b) ? a : b)
#define APMAX(a, b) (a.sgt(b) ? a : b)

        APInt lowerbound = APMIN(APMIN(acMin, adMin), APMIN(bcMin, bdMin));
        APInt upperbound = APMAX(APMAX(acMax, adMax), APMAX(bcMax, bdMax));

#undef APMIN
#undef APMAX
#undef MUL
        return RangeValue(lowerbound, upperbound);
    }

    RangeValue join(const RangeValue& other) const
    {
        if (isUnBounded())
        {
            return other;
        }
        if (other.isUnBounded())
        {
            return *this;
        }
        return RangeValue(min.slt(other.min) ? min : other.min,
                          max.sgt(other.max) ? max : other.max);
    }
    bool operator==(const RangeValue& other) const
    {
        return min.eq(other.min) && max.eq(other.max);
    }

    // [a,b] < [c,d] iff b < c
    bool operator<(const RangeValue& other) const
    {
        return max.slt(other.min);
    }
    // [a,b] <= [c,d] iff b <= c
    bool operator<=(const RangeValue& other) const
    {
        return max.sle(other.min);
    }
    // [a,b] > [c,d] iff a > d
    bool operator>(const RangeValue& other) const
    {
        return min.sgt(other.max);
    }
    // [a,b] >= [c,d] iff a >= d
    bool operator>=(const RangeValue& other) const
    {
        return min.sge(other.min);
    }

    bool contains(APInt value) const
    {
        if (value == inf() || value == negInf())
        {
            return false;
        }
        return value.sge(min) && value.sle(max);
    }

    bool isBounded() const
    {
        return !isUnBounded();
    }
    bool isUnBounded() const
    {
        return min.eq(negInf()) || max.eq(inf());
    }
    bool isConstant() const
    {
        return isBounded() && min.eq(max);
    }
    APInt getConstant() const
    {
        assert(isConstant());
        return min;
    }
    static APInt negInf()
    {
        return APInt::getSignedMinValue(maxBitWidth);
    }
    static APInt inf()
    {
        return APInt::getSignedMaxValue(maxBitWidth);
    }
    DictionaryAttr asAttr(MLIRContext* ctx) const
    {
        mlir::NamedAttrList entries;
        entries.set("lower_bound", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), min));
        entries.set("upper_bound", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), max));
        return DictionaryAttr::get(ctx, entries);
    }
};

inline raw_ostream& operator<<(raw_ostream& os, RangeValue value)
{
    os << "[" << value.min << ", " << value.max << "]";
    return os;
}

struct RangeValueAnalysis
{
    RangeValueAnalysis(Operation* rootOp)
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

    bool hasRange(Value value) const
    {
        return rangeMap.find(value) != rangeMap.end();
    }

    RangeValue getRange(Value value) const
    {
        if (!hasRange(value))
        {
            return RangeValue();
        }
        auto it = rangeMap.find(value);
        assert(it != rangeMap.end());
        return it->second;
    }

private:
    DenseMap<Value, RangeValue> rangeMap;

    bool allOperandsHaveRanges(Operation* op)
    {
        return llvm::all_of(op->getOperands(), [&, this](Value operand) {
            return rangeMap.find(operand) != rangeMap.end();
        });
    }

    SmallVector<RangeValue, 3> resolveOperands(Operation* op)
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

    int dimIndexToInteger(llvm::StringRef dim)
    {
        return ::llvm::StringSwitch<int>(dim)
            .Case("x", 0)
            .Case("y", 1)
            .Case("z", 2)
            .Default(-1);
    }

    RangeValue resolveRangeValue(Operation* op)
    {
        return mlir::TypeSwitch<Operation*, RangeValue>(op)
            .Case([&](ConstantOp op) { return resolveRangeValue(op); })
            .Case([&](ConstantIndexOp op) { return resolveRangeValue(op); })
            .Case([&](ConstantIntOp op) { return resolveRangeValue(op); })
            .Case([&](gpu::ThreadIdOp op) { return resolveRangeValue(op); })
            .Case([&](gpu::BlockIdOp op) { return resolveRangeValue(op); })
            .Case([&](AddIOp op) { return resolveRangeValue(op); })
            .Case([&](SubIOp op) { return resolveRangeValue(op); })
            .Case([&](MulIOp op) { return resolveRangeValue(op); })
            .Case([&](scf::ForOp op) { return resolveRangeValue(op); })
            .Case([&](AffineForOp op) { return resolveRangeValue(op); })
            .Default([&](Operation*) { return RangeValue(); });
    }

    RangeValue resolveRangeValue(ConstantOp op)
    {
        auto attr = op.getValue();
        if (auto value = attr.dyn_cast<IntegerAttr>())
        {
            return RangeValue(value.getValue(), value.getValue());
        }
        return RangeValue();
    }
    RangeValue resolveRangeValue(ConstantIndexOp op)
    {
        auto value = op.getValue();
        return RangeValue(value, value);
    }
    RangeValue resolveRangeValue(ConstantIntOp op)
    {
        auto value = op.getValue();
        return RangeValue(value, value);
    }
    RangeValue resolveRangeValue(gpu::ThreadIdOp op)
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
    RangeValue resolveRangeValue(gpu::BlockIdOp op)
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
    RangeValue resolveRangeValue(AddIOp op)
    {
        auto operands = resolveOperands(op);
        return operands[0] + operands[1];
    }
    RangeValue resolveRangeValue(SubIOp op)
    {
        auto operands = resolveOperands(op);
        return operands[0] - operands[1];
    }
    RangeValue resolveRangeValue(MulIOp op)
    {
        auto operands = resolveOperands(op);
        return operands[0] * operands[1];
    }
    RangeValue resolveLoopBounds(AffineForOp op)
    {
        return op.hasConstantBounds() ? RangeValue(op.getConstantLowerBound(), op.getConstantUpperBound()) : RangeValue();
    }
    RangeValue resolveRangeValue(scf::ForOp op)
    {
        assert(op.getNumInductionVars() == 1);
        RangeValue lowerBound = resolveRangeValue(op.lowerBound().getDefiningOp());
        RangeValue upperBound = resolveRangeValue(op.upperBound().getDefiningOp());
        return lowerBound.isConstant() && upperBound.isConstant() ? RangeValue(lowerBound.min, upperBound.max) : RangeValue();
    }
};
struct RangeValueOptimizePass : public ConvertRangeValueOptimizeBase<RangeValueOptimizePass>
{
    void runOnOperation() final
    {
        rangeValue = &getAnalysis<RangeValueAnalysis>();

        // now we use them to classify the comparison operation
        auto ctx = &getContext();
        OpBuilder builder(ctx);
        Type i1Ty = builder.getI1Type();
        getOperation()->walk([&](CmpIOp op) {
            auto classification = classifyCmpIOp(op);
            if (classification != CmpIOpClassification::Unknown)
            {
                builder.setInsertionPoint(op);
                Value val = builder.create<ConstantOp>(op->getLoc(), i1Ty, builder.getBoolAttr(classification == CmpIOpClassification::AlwaysTrue));
                op.replaceAllUsesWith(val);
                op.erase();
            } 
        });
    }

private:
    enum CmpIOpClassification : int
    {
        Unknown,
        AlwaysFalse,
        AlwaysTrue
    };

    CmpIOpClassification classifyCmpIOp(CmpIOp op)
    {

        auto predicate = op.getPredicate();
        auto lhs = op.lhs();
        auto rhs = op.rhs();
        if (!rangeValue->hasRange(lhs) || !rangeValue->hasRange(rhs))
        {
            return CmpIOpClassification::Unknown;
        }
        auto lhsRange = rangeValue->getRange(lhs);
        auto rhsRange = rangeValue->getRange(rhs);
        if (lhsRange.isUnBounded() || rhsRange.isUnBounded())
        {
            return CmpIOpClassification::Unknown;
        } 

        switch (predicate)
        {
        case CmpIPredicate::slt:
            if (lhsRange < rhsRange)
            {
                return CmpIOpClassification::AlwaysTrue;
            }
            else if (lhsRange >= rhsRange)
            {
                return CmpIOpClassification::AlwaysFalse;
            }
            break;
        case CmpIPredicate::sle:
            if (lhsRange <= rhsRange)
            {
                return CmpIOpClassification::AlwaysTrue;
            }
            else if (lhsRange > rhsRange)
            {
                return CmpIOpClassification::AlwaysFalse;
            }
            break;
        case CmpIPredicate::sgt:
            if (lhsRange > rhsRange)
            {
                return CmpIOpClassification::AlwaysTrue;
            }
            else if (lhsRange <= rhsRange)
            {
                return CmpIOpClassification::AlwaysFalse;
            }
            break;
        case CmpIPredicate::sge:
            if (lhsRange >= rhsRange)
            {
                return CmpIOpClassification::AlwaysTrue;
            }
            else if (lhsRange < rhsRange)
            {
                return CmpIOpClassification::AlwaysFalse;
            }
            break;
        default:
            break;
        }

        return CmpIOpClassification::Unknown;
    }
    RangeValueAnalysis* rangeValue = nullptr;
};

} // namespace

namespace accera::transforms::value
{

std::unique_ptr<mlir::Pass> createRangeValueOptimizePass()
{
    return std::make_unique<RangeValueOptimizePass>();
}

} // namespace accera::transforms::value
