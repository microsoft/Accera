////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <llvm/IR/GlobalValue.h>
#include <mlir/Analysis/DataFlowAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
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

#include <llvm/ADT/APInt.h>
#include <llvm/IR/ConstantRange.h>
#include <llvm/Support/raw_os_ostream.h>

namespace accera::ir::util
{

struct RangeValue
{
    static constexpr int maxBitWidth = 64;
    llvm::ConstantRange range = llvm::ConstantRange::getFull(maxBitWidth);

    RangeValue();
    RangeValue(const llvm::ConstantRange& range_);
    RangeValue(int64_t min_, int64_t max_);
    RangeValue(mlir::APInt min_, mlir::APInt max_);

    RangeValue binaryOp(llvm::Instruction::BinaryOps op, const RangeValue& other) const;
    bool icmp(llvm::CmpInst::Predicate op, const RangeValue& other) const;
    bool contains(mlir::APInt value) const;
    bool isFullSet() const;
    bool isConstant() const;
    mlir::DictionaryAttr asAttr(mlir::MLIRContext* ctx) const;
    bool operator==(const RangeValue& other) const;
};

class RangeValueAnalysis
{
public:
    RangeValueAnalysis() = default;
    RangeValueAnalysis(mlir::Operation* rootOp);
    RangeValueAnalysis(const std::vector<mlir::Operation*>& ops);

    bool hasRange(mlir::Value value) const;
    RangeValue getRange(mlir::Value value) const;
    RangeValue addOperation(mlir::Operation* op);

private:
    mlir::DenseMap<mlir::Value, RangeValue> _rangeMap;

    bool allOperandsHaveRanges(mlir::Operation* op);
    llvm::SmallVector<RangeValue, 3> resolveOperands(mlir::Operation* op);
    RangeValue resolveRangeValue(mlir::ConstantOp op);
    RangeValue resolveRangeValue(mlir::ConstantIndexOp op);
    RangeValue resolveRangeValue(mlir::ConstantIntOp op);
    RangeValue resolveRangeValue(mlir::IndexCastOp op);
    RangeValue resolveRangeValue(mlir::gpu::ThreadIdOp op);
    RangeValue resolveRangeValue(mlir::ROCDL::ThreadIdXOp op);
    RangeValue resolveRangeValue(mlir::ROCDL::ThreadIdYOp op);
    RangeValue resolveRangeValue(mlir::ROCDL::ThreadIdZOp op);
    RangeValue resolveRangeValue(mlir::gpu::BlockIdOp op);
    RangeValue resolveRangeValue(mlir::ROCDL::BlockIdXOp op);
    RangeValue resolveRangeValue(mlir::ROCDL::BlockIdYOp op);
    RangeValue resolveRangeValue(mlir::ROCDL::BlockIdZOp op);
    RangeValue resolveRangeValue(llvm::Instruction::BinaryOps binOp, mlir::Operation* op);
    RangeValue resolveRangeValue(mlir::AffineForOp op);
    RangeValue resolveRangeValue(mlir::scf::ForOp op);
    RangeValue resolveRangeValue(mlir::Operation* op);
};

} // namespace accera::ir::util

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, accera::ir::util::RangeValue value)
{
    os << value.range;
    return os;
}
