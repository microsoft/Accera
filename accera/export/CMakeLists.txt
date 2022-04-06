####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
if(NOT "${LLVM_SETUP_VARIANT}" STREQUAL "Conan")
  return()
endif()

# Generate a list of CMake library targets so that other CMake projects can
# link against them.
set(ACCERA_INSTALL_PACKAGE_DIR lib/cmake/accera)
set(ACCERA_CMAKE_BUILDDIR "${CMAKE_BINARY_DIR}/${ACCERA_INSTALL_PACKAGE_DIR}")

get_property(ACCERA_EXPORTED_LIBS GLOBAL PROPERTY ACCERA_EXPORTED_LIBS)
get_property(ACCERA_RUNTIME_LIBS GLOBAL PROPERTY ACCERA_RUNTIME_LIBS)
set(ALL_EXPORTED_TARGETS ${ACCERA_EXPORTED_LIBS} ${ACCERA_RUNTIME_LIBS})
export(TARGETS ${ALL_EXPORTED_TARGETS} FILE ${ACCERA_CMAKE_BUILDDIR}/AcceraTargets.cmake)

# Set variables for text-replacement in AcceraConfig.cmake.in -> AcceraConfig.cmake

# Generate AcceraConfig.cmake for the build tree.
set(ACCERA_CONFIG_CMAKE_DIR "${ACCERA_CMAKE_BUILDDIR}")
set(ACCERA_CONFIG_EXPORTS_FILE "\${ACCERA_CMAKE_DIR}/AcceraTargets.cmake") # AcceraConfig.cmake must set ACCERA_CMAKE_DIR before including ACCERA_CONFIG_EXPORTS_FILE
set(ACCERA_CONFIG_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AcceraConfig.cmake.in
  ${ACCERA_CMAKE_BUILDDIR}/AcceraConfig.cmake
  @ONLY)
set(ACCERA_CONFIG_CMAKE_DIR)
set(ACCERA_CONFIG_EXPORTS_FILE)
set(ACCERA_CONFIG_INCLUDE_DIRS)

# Generate AcceraConfig.cmake for the install tree.
set(ACCERA_CONFIG_CODE "
# Compute the installation prefix from this AcceraConfig.cmake file location.
get_filename_component(ACCERA_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${ACCERA_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(ACCERA_CONFIG_CODE "${ACCERA_CONFIG_CODE}
get_filename_component(ACCERA_INSTALL_PREFIX \"\${ACCERA_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(ACCERA_CONFIG_CMAKE_DIR "\${ACCERA_INSTALL_PREFIX}/${ACCERA_INSTALL_PACKAGE_DIR}")
set(ACCERA_CONFIG_EXPORTS_FILE "\${ACCERA_CMAKE_DIR}/AcceraTargets.cmake")
set(ACCERA_CONFIG_INCLUDE_DIRS
  "\${ACCERA_INSTALL_PREFIX}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AcceraConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AcceraConfig.cmake
  @ONLY)
set(ACCERA_CONFIG_CODE)
set(ACCERA_CONFIG_CMAKE_DIR)
set(ACCERA_CONFIG_EXPORTS_FILE)
set(ACCERA_CONFIG_INCLUDE_DIRS)

install(EXPORT AcceraTargets DESTINATION ${ACCERA_INSTALL_PACKAGE_DIR}
        COMPONENT accera-cmake-exports)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AcceraConfig.cmake
  DESTINATION ${ACCERA_INSTALL_PACKAGE_DIR}
  COMPONENT accera-cmake-exports)

# Install cmake files to enable generator projects to get LLVM/MLIR, etc
install(FILES
  ${CMAKE_SOURCE_DIR}/CMake/LLVMSetup.cmake
  ${CMAKE_SOURCE_DIR}/CMake/LLVMSetup${LLVM_SETUP_VARIANT}.cmake
  ${CMAKE_SOURCE_DIR}/CMake/MKLSetup.cmake
  ${CMAKE_SOURCE_DIR}/CMake/OpenBLASSetup.cmake
  DESTINATION ${ACCERA_INSTALL_PACKAGE_DIR}
  COMPONENT accera-cmake-exports)
