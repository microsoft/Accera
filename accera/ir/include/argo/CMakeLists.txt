####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

include_directories(${ACCERA_LIBRARIES_DIR})

set(LLVM_TARGET_DEFINITIONS ArgoOps.td)
mlir_tablegen(ArgoOps.h.inc -gen-op-decls)
mlir_tablegen(ArgoOps.cpp.inc -gen-op-defs)
mlir_tablegen(ArgoDialect.h.inc -gen-dialect-decls -dialect=argo)
mlir_tablegen(ArgoDialect.cpp.inc -gen-dialect-defs -dialect=argo)
add_public_tablegen_target(ArgoOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ArgoStructuredOps.td)
mlir_tablegen(ArgoStructuredOps.h.inc -gen-op-decls)
mlir_tablegen(ArgoStructuredOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(ArgoStructuredOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ArgoStructuredOpsInterface.td)
mlir_tablegen(ArgoStructuredOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ArgoStructuredOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ArgoStructuredOpsInterfaceIncGen)
