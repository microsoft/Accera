////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace mlir
{

class DialectRegistry;
class MLIRContext;

} // namespace mlir

namespace accera::transforms::intrinsics
{

/// Register the Intrinsic dialect and the translation from it to the LLVM IR
/// in the given registry;
void registerIntrinsicsDialectTranslation(mlir::DialectRegistry& registry);

/// Register the Intrinsic dialect and the translation from it in the registry
/// associated with the given context.
void registerIntrinsicsDialectTranslation(mlir::MLIRContext& context);

} // namespace accera::transforms::intrinsics
