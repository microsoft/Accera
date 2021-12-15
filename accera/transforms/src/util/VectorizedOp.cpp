////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "util/VectorizedOp.h"

#include <utilities/include/TypeTraits.h>

#include <mlir/Dialect/Vector/VectorOps.h>

#include <algorithm>
#include <stdexcept>

using namespace accera::utilities;

namespace accera::transforms
{
VectorizedOp::VectorizedOp(const std::vector<mlir::Operation*>& scalarOps)
{
    if (scalarOps.empty())
    {
        _result = std::vector<mlir::Value>();
    }
    else if (scalarOps[0]->getNumResults() == 1)
    {
        std::vector<mlir::Value> results;
        std::transform(scalarOps.begin(), scalarOps.end(), std::back_inserter(results), [](mlir::Operation* op) { return op->getResult(0); });
        _result = results;
    }
    else
    {
        _result = scalarOps;
    }
}

VectorizedOp::VectorizedOp(mlir::Operation* vectorOp)
{
    if (vectorOp && vectorOp->getNumResults() == 1)
    {
        _result = vectorOp->getResult(0);
    }
    else if (vectorOp && mlir::isa<mlir::vector::TransferReadOp>(vectorOp))
    {
        _result = vectorOp->getResult(0);
    }
    else
    {
        _result = vectorOp;
    }
}

bool VectorizedOp::HasVectorType() const
{
    return std::holds_alternative<mlir::Value>(_result) || std::holds_alternative<mlir::Operation*>(_result);
}

int64_t VectorizedOp::GetNumResults()
{
    return std::visit(
        VariantVisitor{
            [](const std::vector<mlir::Value>& values) -> int64_t { return 1; },
            [](const std::vector<mlir::Operation*>& ops) -> int64_t { return ops[0]->getNumResults(); },
            [](const mlir::Value& values) -> int64_t { return 1; },
            [](mlir::Operation* op) -> int64_t { return op->getNumResults(); } },
        _result);
}

std::vector<mlir::Value> VectorizedOp::GetScalarResults() const
{
    return std::visit(
        VariantVisitor{
            [](const std::vector<mlir::Value>& values) -> std::vector<mlir::Value> { return values; },
            [](const std::vector<mlir::Operation*>& ops) -> std::vector<mlir::Value> {
                throw std::runtime_error("Operations don't return a single result value");
            },
            [](auto&& rest) -> std::vector<mlir::Value> {
                throw std::runtime_error("Has vector results");
            } },
        _result);
}

mlir::Value VectorizedOp::GetVectorResult() const
{
    return std::visit(
        VariantVisitor{
            [](const mlir::Value& value) -> mlir::Value { return value; },
            [](mlir::Operation* op) -> mlir::Value {
                if (op->getNumResults() == 1)
                {
                    return op->getResult(0);
                }
                else if (op->getNumResults() == 0)
                {
                    return {};
                }
                else if (mlir::isa<mlir::vector::TransferReadOp>(op)) // special case
                {
                    return op->getResult(0);
                }
                else
                {
                    throw std::runtime_error("Result has zero or more than 1 result");
                }
            },
            [](auto&& arg) -> mlir::Value {
                throw std::runtime_error("Not a vector-valued result");
            },
        },
        _result);
}

mlir::Operation* VectorizedOp::GetOp()
{
    return std::visit(
        VariantVisitor{
            [](const mlir::Value& value) -> mlir::Operation* { return value.getDefiningOp(); },
            [](mlir::Operation* op) -> mlir::Operation* { return op; },
            [](auto&& rest) -> mlir::Operation* {
                throw std::runtime_error("Doesn't have vector op");
            } },
        _result);
}
} // namespace accera::transforms
