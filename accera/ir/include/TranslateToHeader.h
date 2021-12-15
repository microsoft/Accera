////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>

#include <iosfwd>
#include <string>
#include <vector>

namespace mlir
{
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace accera
{
namespace ir
{
    mlir::LogicalResult TranslateToHeader(mlir::ModuleOp, std::ostream& os);
    mlir::LogicalResult TranslateToHeader(mlir::ModuleOp, llvm::raw_ostream& os);
    mlir::LogicalResult TranslateToHeader(std::vector<mlir::ModuleOp>& modules, const std::string& libraryName, llvm::raw_ostream& os);
} // namespace ir
} // namespace accera
