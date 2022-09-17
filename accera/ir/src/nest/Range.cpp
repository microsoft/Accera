////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/Range.h"
#include "nest/LoopNestAttributes.h"
#include "nest/LoopNestOps.h"

#include <utilities/include/MathUtil.h>
#include <utilities/include/TypeTraits.h>
#include <value/ValueDialect.h>

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Visitors.h>

#include <iostream>
#include <ostream>

using namespace accera::ir::value;
using namespace accera::utilities;

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

    int64_t Range::Begin() const
    {
        return _begin;
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
                [](auto&& endVal) -> int64_t {
                    assert(false && "Unsupported end value type");
                    return -1;
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
                [](auto&& endOpIdx) -> OperandIndex {
                    assert(false && "Unsupported end value type");
                    return {};
                } },
            _end);
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
        if (i1.HasConstantEnd() && i2.HasConstantEnd())
        {
            return (i1.Begin() == i2.Begin()) && (i1.End() == i2.End()) && (i1.Increment() == i2.Increment());
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
