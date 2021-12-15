////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace mlir
{
class DialectRegistry;
}

namespace accera::ir
{

mlir::DialectRegistry& GetDialectRegistry();

} // namespace accera::ir
