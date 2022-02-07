####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(util_name acc-opt)

set(util_src src/ACCOptMain.cpp)
set(util_include)

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

# binplace
set_property(TARGET ${util_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY "${ACCERA_TOOLS_DIR}")
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES}) # multi configuration
  string(TOUPPER "${CONFIG}" CONFIG_UPPER)
  set_property(TARGET ${util_name}
    PROPERTY
    RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${ACCERA_TOOLS_DIR}")
endforeach(CONFIG ${CMAKE_CONFIGURATION_TYPES})

add_subdirectory(test)

#
# Install acc-opt binary
#
InstallAcceraPyRuntimeLibrary(${util_name} accera-compilers "accera/bin")
