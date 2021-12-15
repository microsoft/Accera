####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(LLVM_TARGET_DEFINITIONS LoopNestInterfaces.td)
mlir_tablegen(LoopNestInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LoopNestInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(LoopNestInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS LoopNestExportedInterfaces.td)
mlir_tablegen(LoopNestExportedInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LoopNestExportedInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(LoopNestExportedInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS LoopNestAttrs.td)
mlir_tablegen(LoopNestAttrs.h.inc -gen-struct-attr-decls)
mlir_tablegen(LoopNestAttrs.cpp.inc -gen-struct-attr-defs)
mlir_tablegen(LoopNestEnums.h.inc -gen-enum-decls)
mlir_tablegen(LoopNestEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(LoopNestAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS LoopNestOps.td)
mlir_tablegen(LoopNestOps.h.inc -gen-op-decls)
mlir_tablegen(LoopNestOps.cpp.inc -gen-op-defs)
mlir_tablegen(LoopNestDialect.h.inc -gen-dialect-decls -dialect=accln)
add_public_tablegen_target(LoopNestOpsIncGen)

add_accera_ir_doc(LoopNestOps -gen-dialect-doc LoopNestDialect references/markdown/)
add_accera_ir_doc(LoopNestInterfaces -gen-op-interface-docs LoopNestInterfaces references/markdown/)
add_accera_ir_doc(LoopNestExportedInterfaces -gen-op-interface-docs LoopNestExportedInterfaces references/markdown/)
add_accera_ir_doc(LoopNestAttrs -gen-attr-interface-docs LoopNestAttributes references/markdown/)
