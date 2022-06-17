####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

if (ACCERA_DISABLE_LIT_TESTS)
  return()
endif()

set(test_name value_mlir_test)

get_target_property(FileCheck_LOCATION FileCheck LOCATION)
get_filename_component(
  MLIR_TOOLS_DIR
  "${FileCheck_LOCATION}"
  DIRECTORY
  CACHE
)
message(STATUS "MLIR_TOOLS_DIR: " ${MLIR_TOOLS_DIR})

set(ACCERA_LIT_TEST_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ACCERA_LIT_TEST_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

if(NOT
   "${LLVM_EXTERNAL_LIT}"
   STREQUAL
   ""
)
  if(NOT EXISTS ${LLVM_EXTERNAL_LIT})
    unset(LLVM_EXTERNAL_LIT CACHE)
    unset(LLVM_EXTERNAL_LIT PARENT_SCOPE)
  endif()
endif()
if("${LLVM_EXTERNAL_LIT}" STREQUAL "")
  find_program(
    LLVM_EXTERNAL_LIT
    NAMES lit
          lit.py
          llvm-lit
          llvm-lit.py
    DOC "Path to LLVM's lit utility"
  )
endif()

# setup for configure_lit_site_cfg, add_lit_testsuite
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
include(AddLLVM)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

configure_file(process_tests.py.in ${CMAKE_CURRENT_BINARY_DIR}/process_tests.py @ONLY)
add_executable(${test_name} value_mlir_test.cpp)
target_link_libraries(${test_name} PRIVATE value Catch2::Catch2WithMain)
catch_discover_tests(${test_name})

set(TEST_DEPENDS
    FileCheck
    count
    not
    mlir-opt
    mlir-translate
    acc-opt
    ${test_name}
)

add_lit_testsuite(
 check-all "Running the lit regression tests" ${CMAKE_CURRENT_BINARY_DIR}
 DEPENDS ${TEST_DEPENDS}
)

# ensure that lit tests can find this binary
set_property(TARGET ${test_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY "${ACCERA_TOOLS_DIR}")
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES}) # multi configuration
  string(TOUPPER "${CONFIG}" CONFIG_UPPER)
  set_property(TARGET ${test_name}
    PROPERTY
    RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${ACCERA_TOOLS_DIR}")
endforeach(CONFIG ${CMAKE_CONFIGURATION_TYPES})

add_lit_testsuites(ACCERA ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})

set_target_properties(${test_name} check-all PROPERTIES FOLDER "tests")
