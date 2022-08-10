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

mlir::OpBuilder& GetTestBuilder();

//
// Test function verifiers
//
bool VerifyGenerate(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn, std::string outputFile="");
bool VerifyParse(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn, std::string outputFile="");
bool VerifyLowerToValue(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn, std::string outputFile="");
bool VerifyLowerToStd(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn, std::string outputFile="");
bool VerifyLowerToLLVM(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp, std::string outputFile="");
bool VerifyTranslateToLLVMIR(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp, bool optimize, std::string outputFile="");
bool VerifyJIT(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp, bool optimize);

