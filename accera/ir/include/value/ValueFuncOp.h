////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ValueEnums.h"
#include "ValueAttributes.h"

#include <llvm/Support/PointerLikeTypeTraits.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>

namespace accera::ir::value
{

class ValueFuncOp : public mlir::Op<ValueFuncOp, mlir::OpTrait::SymbolTable, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::OneRegion, mlir::OpTrait::IsIsolatedFromAbove, mlir::FunctionOpInterface::Trait, mlir::OpTrait::AutomaticAllocationScope, mlir::OpTrait::AffineScope, mlir::CallableOpInterface::Trait, mlir::SymbolOpInterface::Trait>
{
public:
    struct ExternalFuncTag
    {};

    using Op::Op;
    using Op::print;

    static StringRef getOperationName() { return "accv.func"; }
    static ArrayRef<StringRef> getAttributeNames() { return {}; }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& result, mlir::StringRef name, mlir::FunctionType type, ExecutionTarget target);
    static void build(mlir::OpBuilder& builder, mlir::OperationState& result, mlir::StringRef name, mlir::FunctionType type, ExecutionTarget target, ExternalFuncTag);

    /// Operation hooks.
    static ParseResult parse(OpAsmParser& parser, OperationState& result);
    void print(OpAsmPrinter& p);
    LogicalResult verify();

    /// Erase a single argument at `argIndex`.
    void eraseArgument(unsigned argIndex) { eraseArguments({ argIndex }); }
    /// Erases the arguments listed in `argIndices`.
    /// `argIndices` is allowed to have duplicates and can be in any order.
    void eraseArguments(ArrayRef<unsigned> argIndices);

    //===--------------------------------------------------------------------===//
    // CallableOpInterface
    //===--------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    Region* getCallableRegion();

    /// Returns the results types that the callable region produces when executed.
    ArrayRef<Type> getCallableResults();

    static StringRef getExecTargetAttrName() { return "exec_target"; }
    static StringRef getGPULaunchAttrName() { return "gpu_launch"; }

    mlir::StringAttr sym_nameAttr()
    {
        return (*this)->getAttrOfType<::mlir::StringAttr>("sym_name");
    }

    llvm::StringRef sym_name()
    {
        auto attr = sym_nameAttr();
        return attr.getValue();
    }

    ExecutionTargetAttr exec_targetAttr()
    {
        return (*this)->getAttrOfType<ExecutionTargetAttr>(getExecTargetAttrName());
    }

    ExecutionTarget exec_target()
    {
        return exec_targetAttr().getValue();
    }

    mlir::Region& body()
    {
        return this->getOperation()->getRegion(0);
    }

    /// Returns the type of this function.
    /// FIXME: We should drive this via the ODS `type` param.
    FunctionType getType() { 
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    /// Returns the argument types of this function. This is a hook for FunctionOpInterface.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function. This is a hook for FunctionOpInterface.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

    /// Hook for FunctionOpInterface verifier.
    LogicalResult verifyType();
};

ValueFuncOp CreateRawPointerAPIWrapperFunction(mlir::OpBuilder& builder, ValueFuncOp functionToWrap, mlir::StringRef wrapperFnName);

} // namespace accera::ir::value

namespace llvm
{

// Functions hash just like pointers.
template <>
struct DenseMapInfo<::accera::ir::value::ValueFuncOp>
{
    static ::accera::ir::value::ValueFuncOp getEmptyKey()
    {
        auto pointer = llvm::DenseMapInfo<void*>::getEmptyKey();
        return ::accera::ir::value::ValueFuncOp::getFromOpaquePointer(pointer);
    }
    static ::accera::ir::value::ValueFuncOp getTombstoneKey()
    {
        auto pointer = llvm::DenseMapInfo<void*>::getTombstoneKey();
        return ::accera::ir::value::ValueFuncOp::getFromOpaquePointer(pointer);
    }
    static unsigned getHashValue(::accera::ir::value::ValueFuncOp val)
    {
        return hash_value(val.getAsOpaquePointer());
    }
    static bool isEqual(::accera::ir::value::ValueFuncOp LHS, ::accera::ir::value::ValueFuncOp RHS) { return LHS == RHS; }
};

/// Allow stealing the low bits of FuncOp.
template <>
struct PointerLikeTypeTraits<::accera::ir::value::ValueFuncOp>
{
public:
    static inline void* getAsVoidPointer(::accera::ir::value::ValueFuncOp I)
    {
        return const_cast<void*>(I.getAsOpaquePointer());
    }
    static inline ::accera::ir::value::ValueFuncOp getFromVoidPointer(void* P)
    {
        return ::accera::ir::value::ValueFuncOp::getFromOpaquePointer(P);
    }
    static constexpr int NumLowBitsAvailable = 3;
};

} // namespace llvm
