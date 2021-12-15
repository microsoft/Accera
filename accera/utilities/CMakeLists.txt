####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(library_name utilities)

set(src
    src/Boolean.cpp
    src/Files.cpp
    src/Logger.cpp
    src/MemoryLayout.cpp
    src/OutputStreamImpostor.cpp
    src/PropertyBag.cpp
    src/StringUtil.cpp
    src/TypeName.cpp
    src/UniqueId.cpp
)

set(include
    include/Boolean.h
    include/EnumFlagHelpers.h
    include/Exception.h
    include/Files.h
    include/FunctionUtils.h
    include/Hash.h
    include/Logger.h
    include/MemoryLayout.h
    include/OutputStreamImpostor.h
    include/PropertyBag.h
    include/StringUtil.h
    include/TunableParameters.h
    include/TupleUtils.h
    include/TypeAliases.h
    include/TypeName.h
    include/TypeTraits.h
    include/UniqueId.h
    include/ZipIterator.h
)

source_group("src" FILES ${src})
source_group("include" FILES ${include})

add_library(${library_name} ${src} ${include})
target_include_directories(
  ${library_name} PRIVATE include ${ACCERA_LIBRARIES_DIR}
)
target_link_libraries(${library_name} Threads::Threads)

set_property(TARGET ${library_name} PROPERTY FOLDER "accera")

#
# test projects
#

#
# main tests
#
set(test_name ${library_name}_test)

set(test_src
#    test/src/Files_test.cpp
    test/src/FunctionUtils_test.cpp
    test/src/Hash_test.cpp
    test/src/MemoryLayout_test.cpp
    test/src/PropertyBag_test.cpp
    test/src/TunableParameters_test.cpp
    test/src/TypeName_test.cpp
)

source_group("src" FILES ${test_src})

add_executable(${test_name} ${test_src})
target_link_libraries(${test_name} PRIVATE utilities CatchWrapper)
catch_discover_tests(${test_name} )

#
# Install headers and library
#
InstallAcceraHeaders(
  INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
               ${CMAKE_CURRENT_LIST_DIR}/include
)
InstallAcceraLibrary(${library_name})
