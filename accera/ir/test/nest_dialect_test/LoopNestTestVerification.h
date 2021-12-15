////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <functional>

namespace mlir
{
class FuncOp;
class OwningModuleRef;
class OpBuilder;
}

mlir::OpBuilder& GetTestBuilder();

//
// Test function verifiers
//
bool VerifyGenerate(mlir::OwningModuleRef& module, mlir::FuncOp& fn);
bool VerifyParse(mlir::OwningModuleRef& module, mlir::FuncOp& fn);
bool VerifyLowerToValue(mlir::OwningModuleRef& module, mlir::FuncOp& fn);
bool VerifyLowerToStd(mlir::OwningModuleRef& module, mlir::FuncOp& fn);
bool VerifyLowerToLLVM(mlir::OwningModuleRef& module, mlir::FuncOp& fnOp);
bool VerifyTranslateToLLVMIR(mlir::OwningModuleRef& module, mlir::FuncOp& fnOp);
bool VerifyJIT(mlir::OwningModuleRef& module, mlir::FuncOp& fnOp);
