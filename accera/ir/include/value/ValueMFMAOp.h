////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ValueAttributes.h"
#include "ValueEnums.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/TypeSupport.h>

namespace accera::ir::value
{
using llvm::ArrayRef;
using llvm::StringRef;

using mlir::LogicalResult;
using mlir::Type;

constexpr auto MFMAThreadBufferMapName = "threadOffsetsMFMA";

enum class MMAShapeType
{
    T4x16x64,
    T2x32x64,
    T4x4x32,
    T2x2x16,
    Invalid
};

enum class MMAOperandType
{
    A,
    B,
    Acc
};

class MMAOp
{
public:
    MMAOp(const std::array<int64_t, 3>& shape);
    MMAOp(MMAShapeType shape);

    int64_t getLeadingDim() const;

    MMAShapeType getShapeType() const;

    int64_t getThreadTileSize() const;

    bool isValidShape() const;

    int64_t getNumBlocks() const;

    int64_t getTileFactor() const;

    std::pair<int, int> getTileShape() const;

    std::vector<uint8_t> getOffsetMap() const;
    std::array<int64_t, 2> getOffsetMapSize() const;
    std::pair<mlir::MemRefType, mlir::RankedTensorType> GetMFMAThreadOffsetMapType(mlir::IntegerType mlirElemType) const;

    std::pair<mlir::Value, mlir::Value> GetThreadBlockOffsets(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Location& loc) const;

private:
    MMAShapeType shape;
};

} // namespace accera::ir::value
