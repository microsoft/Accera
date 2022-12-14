####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

include(FetchContent)

set(PYBIND_VERSION "2.10.1" CACHE STRING "Version string to use for pybind11")

set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/v${PYBIND_VERSION}.tar.gz
)

FetchContent_GetProperties(pybind11)

set(Python3_FIND_REGISTRY LAST)
find_package(Python3 COMPONENTS Interpreter Development)

if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
