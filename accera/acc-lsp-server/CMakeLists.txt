####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(ACCERA_TOOLS_DIR "${CMAKE_BINARY_DIR}/bin")

set(util_name acc-lsp-server)

set(util_src src/acc-lsp-server.cpp)

source_group("src" FILES ${util_src}) 

add_executable(${util_name} ${util_src})
target_include_directories(${util_name} PRIVATE ${ACCERA_ROOT}/accera)

target_link_libraries(
  ${util_name}
  PRIVATE MLIRLspServerLib
          utilities
          ir
          transforms
          value
          mlirHelpers
)
copy_shared_libraries(${util_name})

#
# Install acc-lsp-server binary
#
InstallAcceraRuntime(${util_name})
