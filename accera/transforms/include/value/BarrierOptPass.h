////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class Pass;
} // namespace mlir

namespace accera::transforms::value
{
std::unique_ptr<mlir::Pass> createBarrierOptPass();
} // namespace accera::transforms::value
