////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ir/include/intrinsics/AcceraIntrinsicsDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;

#include "intrinsics/AcceraIntrinsicsDialect.cpp.inc"

namespace accera::ir::intrinsics
{

void AcceraIntrinsicsDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "intrinsics/AcceraIntrinsics.cpp.inc"
        >();
}

} // namespace accera::ir::intrinsics

#define GET_OP_CLASSES
#include "intrinsics/AcceraIntrinsics.cpp.inc"
