////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ACCERA_VALUE_CONVERSION
#define ACCERA_VALUE_CONVERSION

include "ir/include/value/ValueOps.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "mlir/Dialect/MemRef/IR/MemRefOps.td"

// These patterns are here to work around a bug with linalg.view operations that correspond to a single array element.
// The issue is that a store operation (either accv.StoreOp or (standard) StoreOp) with a (single-element) accv.OffsetOp
// eventually gets lowered to a store into a linalg.view, which fails when storing a single scalar value.
def : Pat<(rcv_GetElementOp (rcv_OffsetOp $source, $args)), (LoadOp $source, $args)>;
def : Pat<(rcv_StoreOp $val, (rcv_OffsetOp $source, $args), $indices), (MemRef_StoreOp $val, $source, $args)>;
def : Pat<(MemRef_StoreOp $val, (rcv_OffsetOp $source, $args), $indices), (MemRef_StoreOp $val, $source, $args)>;

#endif // ACCERA_VALUE_CONVERSION
