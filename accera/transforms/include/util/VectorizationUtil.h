////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "VectorizedOp.h"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

#include <cstddef>
#include <map>
#include <optional>
#include <variant>
#include <vector>

namespace accera::transforms
{

class VectorizedOpMap
{
public:
    VectorizedOpMap() = default;
    std::optional<VectorizedOp> Lookup(mlir::Operation* op) const;
    std::optional<VectorizedOp> Lookup(mlir::Value value) const;
    void Map(mlir::Operation* op, VectorizedOp vectorizedOp);
    void Map(mlir::Value value, VectorizedOp vectorizedOp);

    bool HasMapping(mlir::Operation* op) const;
    bool HasMapping(mlir::Value value) const;

private:
    std::optional<VectorizedOp> Lookup(void* value) const;
    void Map(void* value, VectorizedOp vectorizedOp);
    bool HasMapping(void* value) const;

    std::map<void*, VectorizedOp> _vectorizedOps;
};

std::optional<VectorizedOp> VectorizeOp(mlir::PatternRewriter& rewriter,
                                        mlir::Operation* op,
                                        const VectorizedOpMap& vectorizedOps,
                                        std::vector<mlir::BlockAndValueMapping>& laneMappings,
                                        mlir::Value inductionVar,
                                        int64_t step,
                                        int64_t vectorSize);

bool CanVectorizeOp(mlir::Operation* op,
                    const VectorizedOpMap& vectorizedOps,
                    std::vector<mlir::BlockAndValueMapping>& laneMappings,
                    mlir::Value inductionVar,
                    int64_t step,
                    int64_t vectorSize);

} // namespace accera::transforms
