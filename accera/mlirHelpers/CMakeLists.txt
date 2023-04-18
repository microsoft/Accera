####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
set(library_name mlirHelpers)

set(src
    src/MLIRExecutionEngine.cpp
    src/ConvertToLLVM.cpp
    src/TranslateToLLVMIR.cpp
)

set(include
    include/MLIRExecutionEngine.h
    include/ConvertToLLVM.h
    include/TranslateToLLVMIR.h
)

source_group("src" FILES ${src})
source_group("include" FILES ${include})

add_library(${library_name} ${src} ${include})
target_include_directories(
  ${library_name} PRIVATE include ${ACCERA_LIBRARIES_DIR}
)
target_include_directories(${library_name} SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})

# This is supposed to be overriden on the command line As of LLVM 8.0.1, the
# possible values within the list are: AArch64 AMDGPU ARM BPF Hexagon Lanai Mips
# MSP430 NVPTX PowerPC Sparc SystemZ WebAssembly X86 XCore
set(LLVM_EMITTER_TARGETS
    "X86;ARM;AMDGPU"
    CACHE
      STRING
      "List of LLVM emitter targets to support. Default is \"X86;ARM;AMDGPU\". Specify 'ALL' to support all targets"
)
if(LLVM_EMITTER_TARGETS STREQUAL "ALL")
  set(LLVM_EMITTER_TARGETS_FINAL ${LLVM_ALL_TARGETS})
else()
  set(LLVM_EMITTER_TARGETS_FINAL ${LLVM_EMITTER_TARGETS})
endif()

set(emitter_targets_content "")
set(llvm_emitter_target_libs)
foreach(LLVM_EMITTER_TARGET ${LLVM_EMITTER_TARGETS_FINAL})
  if(NOT
     ${LLVM_EMITTER_TARGET}
     IN_LIST
     LLVM_ALL_TARGETS
  )
    message(
      FATAL_ERROR
        "Unrecognized LLVM emitter target: ${LLVM_EMITTER_TARGET}.\n\nTargets must be one of: ${LLVM_ALL_TARGETS}"
    )
  endif()
  set(emitter_targets_content
      "${emitter_targets_content}    EMITTER_TARGET_ACTION(${LLVM_EMITTER_TARGET}) \\\n"
  )
  set(llvm_emitter_target_libs
      ${llvm_emitter_target_libs}
      LLVM${LLVM_EMITTER_TARGET}CodeGen
      LLVM${LLVM_EMITTER_TARGET}AsmParser
      LLVM${LLVM_EMITTER_TARGET}Disassembler
      # LLVM${LLVM_EMITTER_TARGET}AsmPrinter
      LLVM${LLVM_EMITTER_TARGET}Desc
      LLVM${LLVM_EMITTER_TARGET}Info
  )
endforeach(LLVM_EMITTER_TARGET LLVM_EMITTER_TARGETS)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(
  ${library_name}
  PUBLIC LLVMMCJIT
         LLVMOrcJIT
         ${llvm_emitter_target_libs}
         ${dialect_libs}
         ${conversion_libs}
         ir
         MLIRStandardToLLVM
         MLIRSCFToControlFlow
         MLIRControlFlowToLLVM
         MLIRAffineToStandard
         MLIRAffineTransforms
         MLIRAffineUtils
         MLIRExecutionEngine
         MLIRLinalgToLLVM
         MLIRLinalgTransforms
         MLIRSupport
         MLIRIR
         MLIRAnalysis
         MLIRLinalg
         MLIRLLVMIR
         MLIRParser
         MLIRPass
         MLIRTargetLLVMIRExport
         MLIRTransforms
)

#
# Install headers and library
#
InstallAcceraHeaders(
  INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
               ${CMAKE_CURRENT_LIST_DIR}/include
)
InstallAcceraLibrary(${library_name})
