####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(LLVM_TARGET_DEFINITIONS AcceraOps.td)
mlir_tablegen(AcceraOps.h.inc -gen-op-decls)
mlir_tablegen(AcceraOps.cpp.inc -gen-op-defs)
mlir_tablegen(AcceraDialect.h.inc -gen-dialect-decls -dialect=accera)
add_public_tablegen_target(AcceraOpsIncGen)

add_accera_ir_doc(AcceraOps -gen-dialect-doc AcceraDialect references/markdown/)
