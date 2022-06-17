////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace mlir
{
class ModuleOp;

template <typename OpT>
class OperationPass;

} // namespace mlir

namespace accera::transforms
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createEmitDebugFunctionPass();
} // namespace accera::transforms
