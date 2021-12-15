////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class ModuleOp;
class Pass;
template <typename OpT>
class OperationPass;
} // namespace mlir

namespace accera::transforms::value
{

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createFunctionPointerResolutionPass();
} // namespace accera::transforms::value
