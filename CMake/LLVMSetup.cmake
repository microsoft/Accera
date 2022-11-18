####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Centralized place to define LLVM variables that we can leverage in components
# with dependencies on the accera emitters libraries
####################################################################################################
#
# Gets the following variables:
#
# LLVM_SETUP_VARIANT: An optional environment variable or CMake define
# that specifies the LLVM build source:
#   LLVM_SETUP_VARIANT="Default" - uses vcpkg to acquire LLVM
#                                  Pre-requisite: `vcpkg install accera-llvm` or
#                                  `vcpkg install accera-llvm:x64-windows`
#
#   LLVM_SETUP_VARIANT="Conan"   - uses Conan to acquire LLVM
#                                  (for internal use only)
#
# Sets the following variables:
#
# General information: LLVM_FOUND LLVM_PACKAGE_VERSION
#
# Settings for compiling against LLVM libraries: LLVM_DEFINITIONS
# LLVM_COMPILE_OPTIONS LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIRS LLVM_LIBS
#
# Info about how LLVM was built: LLVM_ENABLE_ASSERTIONS LLVM_ENABLE_EH
# LLVM_ENABLE_RTTI
#
# Location of the executable tools: LLVM_TOOLS_BINARY_DIR
#
# Misc: LLVM_CMAKE_DIR

# Include guard so we don't try to find or download LLVM more than once
include_guard()

set(LLVM_SETUP_VARIANT "Default" CACHE STRING "Source for LLVM binaries")
if(DEFINED ENV{LLVM_SETUP_VARIANT})
  set(LLVM_SETUP_VARIANT $ENV{LLVM_SETUP_VARIANT} CACHE STRING "" FORCE)
endif()

message(STATUS "Using LLVMSetup${LLVM_SETUP_VARIANT}.cmake")

include(LLVMSetup${LLVM_SETUP_VARIANT})

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${LLD_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS})

# CMake seems to have a problem with adding a list of definitions when
# generating VS2017 projects. It ends up creating a define that is comprised of
# multiple preprocessor defines.
foreach(DEFINITION ${LLVM_DEFINITIONS})
  add_definitions(${DEFINITION})
endforeach()

if(TARGET_TRIPLE MATCHES ".*-msvc$")
  message(STATUS "Detected LLVM for MSVC")
  string(TOUPPER "${LLVM_BUILD_TYPE}" LLVM_BUILD_TYPE_UPPER)
  include(CMakePrintHelpers)
  if(NOT LLVM_BUILD_TYPE_UPPER)
    message(
      WARNING "Unable to determine LLVM build type. Defaulting to Release"
    )
    set(LLVM_BUILD_TYPE_UPPER "RELEASE")
  endif(NOT LLVM_BUILD_TYPE_UPPER)
  set(LLVM_MD_OPTION "/${LLVM_USE_CRT_${LLVM_BUILD_TYPE_UPPER}}")
  cmake_print_variables(LLVM_BUILD_TYPE LLVM_BUILD_TYPE_UPPER
                        LLVM_USE_CRT_${LLVM_BUILD_TYPE_UPPER} LLVM_MD_OPTION)
  unset(LLVM_BUILD_TYPE_UPPER)
else()
  message(STATUS "LLVM Triple: ${TARGET_TRIPLE}")
endif(TARGET_TRIPLE MATCHES ".*-msvc$")

if(LLVM_CUSTOM_PATH)
  set(FIND_LLVM_EXTRA_OPT "NO_CMAKE_PATH;NO_SYSTEM_ENVIRONMENT_PATH")
else()
  set_property(TARGET intrinsics_gen PROPERTY FOLDER "cmake_macros")
endif(LLVM_CUSTOM_PATH)

find_program(LLC_EXECUTABLE llc HINTS ${LLVM_TOOLS_BINARY_DIR} ${FIND_LLVM_EXTRA_OPT})
find_program(OPT_EXECUTABLE opt HINTS ${LLVM_TOOLS_BINARY_DIR} ${FIND_LLVM_EXTRA_OPT})

if((NOT OPT_EXECUTABLE) OR (NOT LLC_EXECUTABLE))
  message(ERROR "LLVM not found, please check that LLVM is installed.")
  return()
endif()

find_program(
    MLIR_TRANSLATE_EXECUTABLE mlir-translate HINTS ${LLVM_TOOLS_BINARY_DIR} ${FIND_LLVM_EXTRA_OPT}
)
if(NOT MLIR_TRANSLATE_EXECUTABLE)
  message(ERROR
      "mlir-translate not found, please check that MLIR is installed."
      )
  return()
endif()

find_program(LLVM_SYMBOLIZER llvm-symbolizer HINTS ${LLVM_TOOLS_BINARY_DIR} ${FIND_LLVM_EXTRA_OPT})
find_program(FILECHECK FileCheck HINTS ${LLVM_TOOLS_BINARY_DIR} ${FIND_LLVM_EXTRA_OPT})