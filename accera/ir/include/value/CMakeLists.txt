####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(LLVM_TARGET_DEFINITIONS ValueOps.td)
mlir_tablegen(ValueOps.h.inc -gen-op-decls)
mlir_tablegen(ValueOps.cpp.inc -gen-op-defs)
mlir_tablegen(ValueOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ValueOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ValueDialect.h.inc -gen-dialect-decls -dialect=accv)
add_public_tablegen_target(ValueOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ValueAttrs.td)
mlir_tablegen(ValueAttrs.h.inc -gen-struct-attr-decls)
mlir_tablegen(ValueAttrs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(ValueAttrsIncGen)

add_accera_ir_doc(ValueOps -gen-dialect-doc ValueDialect references/markdown/)
add_accera_ir_doc(ValueAttrs -gen-attr-interface-docs ValueAttributes references/markdown/)
