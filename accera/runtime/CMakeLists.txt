####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(src src/Random.cpp)

set(include include/Random.h)

set(library_name runtime)

add_library(${library_name} ${src} ${include})
target_include_directories(
  ${library_name} PRIVATE include)

#
# Install headers and library
#
InstallAcceraHeaders(
  INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
               ${CMAKE_CURRENT_LIST_DIR}/include
)
InstallAcceraLibrary(${library_name})
