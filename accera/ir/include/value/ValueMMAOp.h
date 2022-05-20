////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Abdul Dakkak, Kern Handa, Captain Jack Sparrow
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

enum class MMAShape
{
    // The shapes below refer to the dimensions of the matmul operation
    // they perform. The B{N} refers to the number of blocks the operation
    // is divided into. e.g. M64xM64xK1_B2 performs the 64x64x1 matmul using
    // 2 separate instruction, each computing half of the output data.
    M64xN64xK1_B4,
    M64xM64xK1_B2,
    M32xN32xK2_B1,
    M16xN16xK4_B1,

    M64xN64xK4_B4,
    M64xN64xK4_B2,
    M32xN32xK8_B1,
    M16xN16xK16_B1
};

enum class MMAOperandType
{
    A,
    B,
    Acc
};

enum class MMASchedulingPolicy
{
    BlockOrder,
    PassOrder
};

class MMAOp
{
public:
    MMAOp(MMAShape shape);

    int64_t getLeadingDim() const;

    MMAShape getShapeType() const;
    int getM() const { return m; }
    int getN() const { return n; }
    int getK() const { return k; }

    int64_t getInElementsPerThread(int64_t warpSize) const;
    int64_t getOutElementsPerThread(int64_t warpSize) const;

    int64_t getPassIncrements(int64_t warpSize) const;

    int64_t getNumBlocks() const;

    std::pair<int, int> getTileShape(int warpSizeX, int warpSizeY) const;

    std::vector<uint8_t> getOffsetMap() const;
    std::array<int64_t, 2> getOffsetMapSize() const;
    std::pair<mlir::MemRefType, mlir::RankedTensorType> GetMFMAThreadOffsetMapType(mlir::IntegerType mlirElemType) const;

    std::pair<mlir::Value, mlir::Value> GetThreadBlockOffsets(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Location& loc) const;

private:
    MMAShape shape;
    int m{};
    int n{};
    int k{};
    int blocks{};

    int64_t getTileFactor() const;
};

} // namespace accera::ir::value
