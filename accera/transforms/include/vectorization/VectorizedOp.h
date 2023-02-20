////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <cstddef>
#include <variant>
#include <vector>

namespace accera::transforms
{
// VectorizedOp represents the result of an op that has been vectorized. It can be realized as either a
// single vector-valued result, or a list of scalar-values results (for cases where a vector-valued result
// isn't possible -- for instance, when the underlying scalar type is `index`)
class VectorizedOp
{
public:
    VectorizedOp() :
        _result(mlir::Value{ nullptr }) {}

    VectorizedOp(const std::vector<mlir::Value>& scalarValues) :
        _result(scalarValues) {}

    VectorizedOp(const std::vector<mlir::Operation*>& scalarOps);

    VectorizedOp(mlir::Value vectorValue) :
        _result(vectorValue) {}

    VectorizedOp(mlir::Operation* vectorOp);

    VectorizedOp(std::nullptr_t) :
        VectorizedOp() {}

    bool HasVectorType() const;
    int64_t GetNumResults();
    std::vector<mlir::Value> GetScalarResults() const;
    mlir::Value GetVectorResult() const;
    mlir::Operation* GetOp();

private:
    std::variant<mlir::Value, // if the original op was vectorized into a vector-valued Value that's not the result of an operation, or an operation with 1 result
                 mlir::Operation*, // if the original op was vectorized into an op with either 0 or > 1 results
                 std::vector<mlir::Value>, // if the original op was vectorized into a collection of scalar-value Values that aren't the result of an operation, or a set of operations with 1 result
                 std::vector<mlir::Operation*> // if the original op was vectorized into a collection of scalar ops with either 0 or > 1 results
                 >
        _result;
};

} // namespace accera::transforms
