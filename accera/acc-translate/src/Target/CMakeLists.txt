####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

add_llvm_library(ArgoTargetCppTranslation
  Cpp/AffineDialectCppPrinter.cpp
  Cpp/AcceraDialectCppPrinter.cpp
  Cpp/CppPrinter.cpp
  Cpp/CppPrinterUtils.cpp
  Cpp/GpuDialectCppPrinter.cpp
  Cpp/RocDLDialectCppPrinter.cpp
  Cpp/ScfDialectCppPrinter.cpp
  Cpp/StdDialectCppPrinter.cpp
  Cpp/TranslateToCpp.cpp

  ADDITIONAL_HEADER_DIRS
  ${ARGO_SRC_INCLUDE_DIR}/mlir/Target

  LINK_LIBS PUBLIC
  MLIRArgo
  LLVMSupport
  )
