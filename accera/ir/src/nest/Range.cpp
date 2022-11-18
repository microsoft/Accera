////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/Range.h"
#include "nest/LoopNestAttributes.h"
#include "nest/LoopNestOps.h"
#include "IRUtil.h"

#include <utilities/include/MathUtil.h>
#include <utilities/include/TypeTraits.h>
#include <value/ValueDialect.h>

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Visitors.h>

#include <iostream>
#include <ostream>

using namespace accera::ir::value;
using namespace accera::utilities;

namespace
{
    int64_t GetValueMapSingleConstant(mlir::AffineValueMap& valueMap)
    {
        mlir::AffineMap simplifiedMap = valueMap.getAffineMap();
        auto operandsTmp = valueMap.getOperands();
        llvm::SmallVector<mlir::Value, 2> operands(operandsTmp.begin(), operandsTmp.end());
        mlir::fullyComposeAffineMapAndOperands(&simplifiedMap, &operands);
        mlir::canonicalizeMapAndOperands(&simplifiedMap, &operands);
        if (simplifiedMap.isSingleConstant())
        {
            return simplifiedMap.getSingleConstantResult();
        }
        return mlir::ShapedType::kDynamicSize;
    }
}

namespace accera::ir
{
namespace loopnest
{
    Range::Range(int64_t begin, int64_t end, int64_t increment) :
        _begin(begin),
        _end(end),
        _increment(increment)
    {}

    Range::Range(int64_t begin, mlir::Value end, int64_t increment) :
        _begin(begin),
        _increment(increment)
    {
        if (end.isa<mlir::BlockArgument>())
        {
            _end = end;
            return;
        }

        auto op = end.getDefiningOp();
        assert(op);

        mlir::TypeSwitch<mlir::Operation*>(op)
            .Case<DimSizeOp>([&](DimSizeOp dimSizeOp) {
                auto index = dimSizeOp.dimensionIndex();
                _end = index.getValue();
            })
            .Case<mlir::arith::ConstantOp>([&](mlir::arith::ConstantOp constantOp) {
                auto constantAttr = constantOp.getValue();
                assert(constantAttr.isa<mlir::IntegerAttr>() && "Range end must be an integer constant");
                auto constantVal = constantAttr.cast<mlir::IntegerAttr>().getInt();
                _end = static_cast<int64_t>(constantVal);
            })
            .Case<CastOp>([&](CastOp castOp) {
                _end = castOp.result();
            })
            .Default([&](Operation* op) {
                if (op->getNumResults() == 1)
                {
                    _end = op->getResult(0);
                }
                else
                {
                    assert(false && "Unsupported Range end Value");
                }
            });
    }

    Range::Range(int64_t begin, Index endIndex, int64_t increment) :
        _begin(begin),
        _end(endIndex),
        _increment(increment)
    {}

    Range::Range(int64_t begin, OperandIndex endIndex, int64_t increment) :
        _begin(begin),
        _end(endIndex),
        _increment(increment)
    {}

    Range::Range(int64_t begin, std::string endSymbol, int64_t increment) :
        _begin(begin),
        _end(endSymbol),
        _increment(increment)
    {}

    Range::Range(mlir::AffineValueMap begin, mlir::AffineValueMap end, int64_t increment) :
        _begin(begin),
        _end(end),
        _increment(increment)
    {}

    int64_t Range::Begin() const
    {
        return std::visit(
            VariantVisitor{
                [](int64_t beginVal) -> int64_t {
                    return beginVal;
                },
                [](mlir::AffineValueMap beginValueMap) -> int64_t {
                    return GetValueMapSingleConstant(beginValueMap);
                },
                [](auto&& beginVal) -> int64_t {
                    assert(false && "Unsupported begin value type");
                    return -1;
                } },
            _begin);
    }

    mlir::AffineValueMap Range::ValueMapBegin() const
    {
        return std::visit(
            VariantVisitor{
                [](int64_t beginMap) -> mlir::AffineValueMap {
                    assert(false && "Calling VariableBegin() on a constant range begin");
                    return {};
                },
                [](mlir::AffineValueMap beginMap) -> mlir::AffineValueMap {
                    return beginMap;
                },
                [](auto&& beginMap) -> mlir::AffineValueMap {
                    assert(false && "Unsupported begin value type");
                    return {};
                } },
            _begin);
    }

    int64_t Range::End() const
    {
        return std::visit(
            VariantVisitor{
                [](int64_t endVal) -> int64_t {
                    return endVal;
                },
                [](Index endIndex) -> int64_t {
                    assert(false && "Range must be resolved before requesting End()");
                    return 0;
                },
                [](OperandIndex endIndex) -> int64_t {
                    assert(false && "Range must be resolved before requesting End()");
                    return 0;
                },
                [](mlir::Value endIndex) -> int64_t {
                    return mlir::ShapedType::kDynamicSize;
                },
                [](std::string endIndex) -> int64_t {
                    return mlir::ShapedType::kDynamicSize;
                },
                [](mlir::AffineValueMap endValueMap) -> int64_t {
                    return GetValueMapSingleConstant(endValueMap);
                },
                [](auto&& endVal) -> int64_t {
                    assert(false && "Unsupported end value type");
                    return -1;
                } },
            _end);
    }

    mlir::AffineValueMap Range::ValueMapEnd() const
    {
        return std::visit(
            VariantVisitor{
                [](int64_t endDynIdx) -> mlir::AffineValueMap {
                    assert(false && "Calling ValueMapEnd() on a constant range");
                    return {};
                },
                [](Index endDynIdx) -> mlir::AffineValueMap {
                    assert(false && "Calling ValueMapEnd() on an Index range");
                    return {};
                },
                [](OperandIndex endDynIdx) -> mlir::AffineValueMap {
                    assert(false && "Calling ValueMapEnd() on an OperandIndex range");
                    return {};
                },
                [](mlir::Value endDynIdx) -> mlir::AffineValueMap {
                    assert(false && "Calling ValueMapEnd() on a mlir::Value range");
                    return {};
                },
                [](mlir::AffineValueMap endMap) -> mlir::AffineValueMap {
                    return endMap;
                },
                [](auto&& endDynIdx) -> mlir::AffineValueMap {
                    assert(false && "Unsupported end value type");
                    return {};
                } },
            _end);
    }

    mlir::Value Range::VariableEnd() const
    {
        return std::visit(
            VariantVisitor{
                [](int64_t endDynIdx) -> mlir::Value {
                    assert(false && "Calling VariableEnd() on a constant range");
                    return {};
                },
                [](Index endDynIdx) -> mlir::Value {
                    assert(false && "Calling VariableEnd() on an Index range");
                    return {};
                },
                [](OperandIndex endDynIdx) -> mlir::Value {
                    assert(false && "Calling VariableEnd() on an OperandIndex range");
                    return {};
                },
                [](mlir::Value endDynIdx) -> mlir::Value {
                    return endDynIdx;
                },
                [](mlir::AffineValueMap endMap) -> mlir::Value {
                    assert(false && "Calling VariableEnd() on a ValueMap range");
                    return {};
                },
                [](auto&& endDynIdx) -> mlir::Value {
                    assert(false && "Unsupported end value type");
                    return {};
                } },
            _end);
    }

    std::string Range::SymbolNameEnd() const
    {
        return std::visit(
            VariantVisitor{
                [](int64_t endDynIdx) -> std::string {
                    assert(false && "Calling SymbolNameEnd() on a constant range");
                    return {};
                },
                [](Index endDynIdx) -> std::string {
                    assert(false && "Calling SymbolNameEnd() on an Index range");
                    return {};
                },
                [](OperandIndex endDynIdx) -> std::string {
                    assert(false && "Calling SymbolNameEnd() on an OperandIndex range");
                    return {};
                },
                [](mlir::Value endDynIdx) -> std::string {
                    assert(false && "Calling SymbolNameEnd() on an mlir::Value range");
                    return {};
                },
                [](std::string endDynIdx) -> std::string {
                    return endDynIdx;
                },
                [](mlir::AffineValueMap endMap) -> std::string {
                    assert(false && "Calling SymbolNameEnd() on a ValueMap range");
                    return {};
                },
                [](auto&& endDynIdx) -> std::string {
                    assert(false && "Unsupported end value type");
                    return {};
                } },
            _end);
    }

    Index Range::EndIndex() const
    {
        return std::visit(
            VariantVisitor{
                [](int64_t endVal) -> Index {
                    assert(false && "Calling EndIndex() on a constant range");
                    return {};
                },
                [](Index endVal) -> Index {
                    return endVal;
                },
                [](OperandIndex endIndex) -> Index {
                    assert(false && "Calling EndIndex() on an OperandIndex range");
                    return {};
                },
                [](mlir::Value endIndex) -> Index {
                    assert(false && "Calling EndIndex() on a variable range");
                    return {};
                },
                [](mlir::AffineValueMap endMap) -> Index {
                    assert(false && "Calling EndIndex() on a ValueMap range");
                    return {};
                },
                [](auto&& endVal) -> Index {
                    assert(false && "Unsupported end value type");
                    return {};
                } },
            _end);
    }

    OperandIndex Range::EndOperandIndex() const
    {
        return std::visit(
            VariantVisitor{
                [](int64_t endOpIdx) -> OperandIndex {
                    assert(false && "Calling EndOperandIndex() on a constant range");
                    return {};
                },
                [](Index endOpIdx) -> OperandIndex {
                    assert(false && "Calling EndOperandIndex() on an Index range");
                    return {};
                },
                [](OperandIndex endOpIdx) -> OperandIndex {
                    return endOpIdx;
                },
                [](mlir::Value endOpIdx) -> OperandIndex {
                    assert(false && "Calling EndOperandIndex() on a variable range");
                    return {};
                },
                [](mlir::AffineValueMap endMap) -> OperandIndex {
                    assert(false && "Calling EndOperandIndex() on a ValueMap range");
                    return {};
                },
                [](auto&& endOpIdx) -> OperandIndex {
                    assert(false && "Unsupported end value type");
                    return {};
                } },
            _end);
    }

    bool Range::HasConstantBegin() const
    {
        return std::holds_alternative<int64_t>(_begin);
    }

    bool Range::HasValueMapBegin() const
    {
        return std::holds_alternative<mlir::AffineValueMap>(_begin);
    }

    bool Range::HasConstantEnd() const
    {
        return std::holds_alternative<int64_t>(_end);
    }

    bool Range::HasIndexEnd() const
    {
        return std::holds_alternative<Index>(_end);
    }

    bool Range::HasOperandIndexEnd() const
    {
        return std::holds_alternative<OperandIndex>(_end);
    }

    bool Range::HasVariableEnd() const
    {
        return std::holds_alternative<mlir::Value>(_end);
    }

    bool Range::HasSymbolNameEnd() const
    {
        return std::holds_alternative<std::string>(_end);
    }

    bool Range::HasValueMapEnd() const
    {
        return std::holds_alternative<mlir::AffineValueMap>(_end);
    }

    int64_t Range::Size() const
    {
        return End() - Begin();
    }

    int64_t Range::Increment() const
    {
        return _increment;
    }

    int64_t Range::NumIterations() const
    {
        return CeilDiv(End() - Begin(), Increment());
    }

    int64_t Range::LastIterationBegin() const
    {
        auto result = End() - (Size() % Increment());
        if (result == End()) // not a boundary
        {
            result = End() - Increment();
        }
        return result;
    }

    std::ostream& operator<<(std::ostream& os, const Range& r)
    {
        os << "[" << r.Begin() << ",";
        if (r.HasVariableEnd())
        {
            auto arg = r.VariableEnd().dyn_cast<mlir::BlockArgument>();
            os << "arg" << arg.getArgNumber();
        }
        else
        {
            os << r.End();
        }
        os << ":" << r.Increment() << ")";
        return os;
    }

    bool operator==(const Range& i1, const Range& i2)
    {
        if (i1.Increment() != i2.Increment())
        {
            return false;
        }
        if (i1.HasValueMapBegin() && i2.HasValueMapBegin())
        {
            if (!util::AffineValueMapsEqual(i1.ValueMapBegin(), i2.ValueMapBegin()))
            {
                return false;
            }
        }
        else if (i1.HasConstantBegin() && i2.HasConstantBegin())
        {
            if (i1.Begin() != i2.Begin())
            {
                return false;
            }
        }

        if (i1.HasConstantEnd() && i2.HasConstantEnd())
        {
            return (i1.End() == i2.End());
        }
        else if (i1.HasIndexEnd() && i2.HasIndexEnd())
        {
            // Both i1 and i2 are unresolved Index values, now they're only equal if they have the same index
            return i1.EndIndex() == i2.EndIndex();
        }
        else if (i1.HasOperandIndexEnd() && i2.HasOperandIndexEnd())
        {
            // Both i1 and i2 are unresolved OperandIndex values, now they're only equal if they have the same index
            return i1.EndOperandIndex() == i2.EndOperandIndex();
        }
        else if (i1.HasVariableEnd() && i2.HasVariableEnd())
        {
            // Both i1 and i2 have variable end values, now they're only equal if they have the same mlir::Value
            return i1.VariableEnd() == i2.VariableEnd();
        }
        else if (i1.HasSymbolNameEnd() && i2.HasSymbolNameEnd())
        {
            // Both i1 and i2 have variable end values, now they're only equal if they have the same std::string
            return i1.SymbolNameEnd() == i2.SymbolNameEnd();
        }
        else if (i1.HasValueMapEnd() && i2.HasValueMapEnd())
        {
            return util::AffineValueMapsEqual(i1.ValueMapEnd(), i2.ValueMapEnd());
        }
        else
        {
            // Can't determine at this time if a constant is equal to an un-resolved value
            return false;
        }
    }

    bool operator!=(const Range& i1, const Range& i2)
    {
        return !(i1 == i2);
    }

    bool operator<(const Range& i1, const Range& i2)
    {
        if (i1.Begin() != i2.Begin())
        {
            return i1.Begin() < i2.Begin();
        }
        else if (i1.HasConstantEnd() && i2.HasConstantEnd())
        {
            return i1.End() < i2.End();
        }
        else if (i1.HasIndexEnd() && i2.HasIndexEnd())
        {
            return i1.EndIndex() < i2.EndIndex();
        }
        else if (i1.HasOperandIndexEnd() && i2.HasOperandIndexEnd())
        {
            return i1.EndOperandIndex() < i2.EndOperandIndex();
        }
        else
        {
            // if only one is resolved, then examine the increment
            return i1.Increment() < i2.Increment();
        }
    }

    bool operator<=(const Range& i1, const Range& i2)
    {
        return i1 < i2 || i1 == i2;
    }

    bool Intersects(const Range& a, const Range& b)
    {
        // std::cout << "Checking intersection of ranges " << a << " and " << b << std::endl;

        int64_t aIter = a.NumIterations();
        int64_t bIter = b.NumIterations();

        if (aIter == 0 || bIter == 0)
        {
            return false;
        }
        auto aLast = a.Begin() + (aIter - 1) * a.Increment();
        auto bLast = b.Begin() + (bIter - 1) * b.Increment();

        return aLast >= b.Begin() && a.Begin() <= bLast;
    }

} // namespace loopnest
} // namespace accera::ir
