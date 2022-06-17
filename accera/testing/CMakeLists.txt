####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(library_name testing)

set(src
    src/testing.cpp
)
set(include
    include/testing.h
)

source_group("src" FILES ${src})
source_group("include" FILES ${include})

add_library(${library_name} ${src} ${include})
target_include_directories(${library_name} PRIVATE include ${ACCERA_LIBRARIES_DIR})
target_link_libraries(${library_name} utilities)

set_property(TARGET ${library_name} PROPERTY FOLDER "accera")

add_library(CatchWrapper OBJECT EXCLUDE_FROM_ALL src/CatchMain.cpp)
target_link_libraries(CatchWrapper PUBLIC Catch2::Catch2WithMain)
