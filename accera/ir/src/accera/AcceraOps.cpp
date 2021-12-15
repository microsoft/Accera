////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "accera/AcceraOps.h"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/StringSwitch.h>

namespace accera::ir::rc
{
void AcceraDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "accera/AcceraOps.cpp.inc"
        >();
}
} // namespace accera::ir::rc

using namespace llvm;
using namespace mlir;
using namespace accera::ir;
using namespace accera::ir::rc;

// TableGen'd op method definitions
#define GET_OP_CLASSES
#include "accera/AcceraOps.cpp.inc"
