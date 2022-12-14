####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

# setup for using LLVM and MLIR
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(src)

set(util_name acc-translate)

set(target_src
    src/Target/Cpp/AcceraDialectCppPrinter.cpp
    src/Target/Cpp/AffineDialectCppPrinter.cpp
    src/Target/Cpp/CppPrinter.cpp
    src/Target/Cpp/CppPrinterUtils.cpp
    src/Target/Cpp/GpuDialectCppPrinter.cpp
    src/Target/Cpp/RocDLDialectCppPrinter.cpp
    src/Target/Cpp/ScfDialectCppPrinter.cpp
    src/Target/Cpp/StdDialectCppPrinter.cpp
    src/Target/Cpp/TranslateToCpp.cpp
    src/Target/Cpp/AMDGPU.cpp
    src/Target/Cpp/VectorDialectCppPrinter.cpp
    src/Target/Cpp/LLVMDialectCppPrinter.cpp
    src/Target/LLVMIR/IntrinsicToLLVMIRTranslation.cpp
    )

set(target_include
    src/Target/Cpp/AcceraDialectCppPrinter.h
    src/Target/Cpp/AffineDialectCppPrinter.h
    src/Target/Cpp/CppPrinter.h
    src/Target/Cpp/CppPrinterUtils.h
    src/Target/Cpp/GpuDialectCppPrinter.h
    src/Target/Cpp/RocDLDialectCppPrinter.h
    src/Target/Cpp/ScfDialectCppPrinter.h
    src/Target/Cpp/StdDialectCppPrinter.h
    src/Target/Cpp/TranslateToCpp.h
    src/Target/Cpp/AMDGPU.h
    src/Target/Cpp/VectorDialectCppPrinter.h
    src/Target/Cpp/LLVMDialectCppPrinter.h
    src/Target/LLVMIR/IntrinsicToLLVMIRTranslation.h
    )


set(util_src ${target_src} src/AcceraTranslateMain.cpp)
set(util_include ${target_include})

source_group("src" FILES ${util_src})
source_group("include" FILES ${util_include})

add_executable(${util_name} ${util_src} ${util_include})
target_include_directories(${util_name} PRIVATE ${ACCERA_ROOT}/accera)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

target_link_libraries(
  ${util_name}
  PRIVATE MLIROptLib
          utilities
          ir
          transforms
          value
          mlirHelpers
          ${translation_libs}
          ${dialect_libs}
)
copy_shared_libraries(${util_name})

set_property(TARGET ${util_name} PROPERTY FOLDER "accera")

# binplace
set_property(TARGET ${util_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY "${ACCERA_TOOLS_DIR}")
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES}) # multi configuration
  string(TOUPPER "${CONFIG}" CONFIG_UPPER)
  set_property(TARGET ${util_name}
    PROPERTY
    RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${ACCERA_TOOLS_DIR}")
endforeach(CONFIG ${CMAKE_CONFIGURATION_TYPES})

#
# Install acc-translate binary
#
InstallAcceraPyRuntimeLibrary(${util_name} accera-gpu "accera/bin")
