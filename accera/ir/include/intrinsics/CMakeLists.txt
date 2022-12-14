####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

add_mlir_dialect(AcceraIntrinsics accintr)

set(LLVM_TARGET_DEFINITIONS AcceraIntrinsics.td)
mlir_tablegen(AcceraIntrinsicsConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(AcceraIntrinsicsConversionsIncGen)
