####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

include_guard()

# vcpkg targets
find_package(LLVM CONFIG REQUIRED)
find_package(LLD CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using LLDConfig.cmake in ${LLD_CMAKE_DIR}")
message(STATUS "Using MLIRConfig.cmake in ${MLIR_CMAKE_DIR}")
