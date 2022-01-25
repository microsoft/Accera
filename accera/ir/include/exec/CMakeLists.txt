include_directories(${ACCERA_LIBRARIES_DIR})

set(LLVM_TARGET_DEFINITIONS ExecutionPlanInterfaces.td)
mlir_tablegen(ExecutionPlanInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ExecutionPlanInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ExecutionPlanInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ExecutionPlanOps.td)
mlir_tablegen(ExecutionPlanOps.h.inc -gen-op-decls)
mlir_tablegen(ExecutionPlanOps.cpp.inc -gen-op-defs)
mlir_tablegen(ExecutionPlanDialect.h.inc -gen-dialect-decls -dialect=accxp)
mlir_tablegen(ExecutionPlanDialect.cpp.inc -gen-dialect-defs -dialect=accxp)
add_public_tablegen_target(ExecutionPlanOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ExecutionPlanAttrs.td)
mlir_tablegen(ExecutionPlanAttrs.h.inc -gen-struct-attr-decls)
mlir_tablegen(ExecutionPlanAttrs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(ExecutionPlanAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS ExecutionPlanAttrs.td)
mlir_tablegen(ExecutionPlanEnums.h.inc -gen-enum-decls)
mlir_tablegen(ExecutionPlanEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(ExecutionPlanEnumsIncGen)

add_accera_ir_doc(ExecutionPlanOps -gen-dialect-doc ExecutionPlanDialect references/markdown/)
add_accera_ir_doc(ExecutionPlanAttrs -gen-attr-interface-docs ExecutionPlanAttributes references/markdown/)
