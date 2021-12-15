####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Searches for LLVM's OpenMP library
#####################################################################################################

# Note: find_package(OpenMP) is not used here because it presumes the GNU libgomp library
set(LLVMOpenMP_NAMES libomp.so.5 libomp.dylib libomp.lib)

find_library(LLVMOpenMP_LIBRARY
    NAMES ${LLVMOpenMP_NAMES}
)

if((NOT LLVMOpenMP_LIBRARY) AND (MSVC_TOOLSET_VERSION GREATER_EQUAL 142))
    # Fallback: Visual Studio 2019 (Update 10+) includes this lib, so even if
    # cmake cannot find it, cl.exe should
    message(WARNING "LLVM OpenMP library not found with CMake intrinsics, fallback to well-known library")
    set(LLVMOpenMP_LIBRARY "libomp.lib")
endif()

if(LLVMOpenMP_LIBRARY)
    message(STATUS "Using LLVM OpenMP library: ${LLVMOpenMP_LIBRARY}")
else()
    message(WARNING "LLVM OpenMP library not found.")
endif()