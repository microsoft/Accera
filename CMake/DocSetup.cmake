####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

# Borrowed from MLIR

# <command> can be one of:
# -gen-attr-interface-docs
# -gen-dialect-doc
# -gen-op-doc
# -gen-op-interface-docs
# -gen-pass-doc
# -gen-type-interface-docs

# This needs to be inside ${SPHINX_SOURCE}
set(ACCERA_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs/sphinx_source/ir)

function(add_accera_ir_doc doc_filename command output_file output_directory)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  tablegen(MLIR ${output_file}.md ${command} "-I${MLIR_MAIN_INCLUDE_DIR}" "-I${MLIR_INCLUDE_DIR}")
  set(GEN_DOC_FILE ${ACCERA_TABLEGEN_OUTPUT_DIR}/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(tablegen-ir-docs ${output_file}DocGen)
endfunction()
