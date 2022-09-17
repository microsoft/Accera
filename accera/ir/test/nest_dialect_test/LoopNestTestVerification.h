////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <functional>

namespace mlir
{
class FuncOp;
class OpBuilder;

template <typename OpTy>
class OwningOpRef;
class ModuleOp;
}

//
// Test function verifiers
//
bool VerifyGenerate(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn);
bool VerifyParse(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn);
bool VerifyLowerToValue(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn);
bool VerifyLowerToStd(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn);
bool VerifyLowerToLLVM(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp);
bool VerifyTranslateToLLVMIR(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp);
bool VerifyJIT(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp);
