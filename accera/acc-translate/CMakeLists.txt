####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

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
    )

set(util_src ${target_src} src/AcceraTranslateMain.cpp)
set(util_include ${target_include})

source_group("src" FILES ${util_src})
source_group("include" FILES ${util_include})

add_executable(${util_name} ${util_src} ${util_include})
target_include_directories(${util_name} PRIVATE ${ACCERA_ROOT}/accera)

target_link_libraries(
  ${util_name}
  PRIVATE MLIROptLib
          utilities
          ir
          transforms
          value
          mlirHelpers
)
copy_shared_libraries(${util_name})

set_property(TARGET ${util_name} PROPERTY FOLDER "accera")

#
# Install acc-translate binary
#
InstallAcceraPyRuntimeLibrary(${util_name} accera-gpu "accera/bin")
