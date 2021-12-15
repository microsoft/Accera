####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Centralized place to define MKL variables
####################################################################################################

# Sets the following variables:
#
# General information:
# BLAS_FOUND
#
# Settings for compiling against MKL libraries:
# BLAS_INCLUDE_DIRS
# BLAS_LIBS
# BLAS_DLL_DIR
#
# Using FindBLAS module:
# find_package(BLAS)
# if(BLAS_FOUND)
#     message(STATUS "Blas libraries: ${BLAS_LIBRARIES}")
#     message(STATUS "Blas linker flags: ${BLAS_LINKER_FLAGS}")
#     message(STATUS "Blas vendor: ${BLA_VENDOR}")
#
# Variables defined by FindBLAS module that we don't set:
#     BLAS_LIBRARIES
#     BLAS_LINKER_FLAGS
#     BLA_VENDOR

# Include guard so we don't try to find or download BLAS more than once
# Use the same BLASSetup_included flag as OpenBLASSetup so we only build with either OpenBLAS or MKL, but not both
if(BLASSetup_included)
    return()
endif()
set(BLASSetup_included true)

# Set policy saying to use newish IN_LIST operator
cmake_policy(SET CMP0057 NEW)

macro(get_processor_mapping _result _processor_generation)
    if(DEFINED processor_map_${_processor_generation})
        set(${_result} ${processor_map_${_processor_generation}})
    else()
        set(${_result} ${_processor_generation})
    endif()
endmacro()

if (NOT WIN32)
    set(fallback_mkl_path /opt/intel/mkl/)
endif()

if (DEFINED ENV{MKLROOT})
    message(STATUS "Using MKL from MKLROOT environment variable: $ENV{MKLROOT}")
    set(mkl_include_path $ENV{MKLROOT}/include)
    set(mkl_lib_path $ENV{MKLROOT}/lib/intel64)
else()
    if (fallback_mkl_path)
        message(WARNING "Environment variable MKLROOT not found, attempting to use fallback default path ${fallback_mkl_path}. Try running mklvars from your mkl path to set the environment variables")
        set(mkl_include_path $ENV{MKLROOT}/include)
        set(mkl_lib_path $ENV{MKLROOT}/lib/intel64)
    else()
        message(ERROR " Environment variable MKLROOT not found and no fallback default path available. Try running mklvars from your mkl path to set the environment variables or manually setting your MKLROOT environment variable")
        set(mkl_include_path )
        set(mkl_lib_path )
    endif()
endif()

set(BLAS_INCLUDE_SEARCH_PATHS ${mkl_include_path})
set(BLAS_LIB_SEARCH_PATHS ${mkl_lib_path})

if(WIN32)
    set(BLAS_LIB_NAMES libmkl_intel_ilp64.lib libmkl_sequential.lib libmkl_core.lib)
else()
    set(BLAS_LIB_NAMES libmkl_intel_ilp64.so libmkl_sequential.so libmkl_core.so libmkl_intel_ilp64 libmkl_sequential libmkl_core)
endif()

set(BLA_VENDOR "Intel10_64lp_seq")
find_package(BLAS REQUIRED)
find_path (BLAS_INCLUDE_DIRS mkl.h ${mkl_include_path})

if(BLAS_FOUND)
    message(STATUS "Blas libraries: ${BLAS_LIBRARIES}")
    message(STATUS "Blas linker flags: ${BLAS_LINKER_FLAGS}")
    message(STATUS "Blas include directories: ${BLAS_INCLUDE_DIRS}")
    message(STATUS "Blas Vendor: ${BLA_VENDOR}")
    set(BLAS_LIBS ${BLAS_LIBRARIES})
endif()


find_path(BLAS_INCLUDE_DIRS mkl_cblas.h
    PATHS ${BLAS_INCLUDE_SEARCH_PATHS} ${BLAS_INCLUDE_DIRS}
    NO_DEFAULT_PATH
)

find_library(BLAS_LIBS
    NAMES ${BLAS_LIB_NAMES}
    PATHS ${BLAS_LIB_SEARCH_PATHS}
    NO_DEFAULT_PATH
)

if(BLAS_LIBS AND BLAS_INCLUDE_DIRS)
    message(STATUS "Using BLAS include path: ${BLAS_INCLUDE_DIRS}")
    message(STATUS "Using BLAS library: ${BLAS_LIBS}")
    message(STATUS "Using BLAS DLLs: ${BLAS_DLLS}")
    set(BLAS_FOUND "YES")
    add_compile_definitions(USE_MKL=1)
    add_compile_definitions(BLAS_INCLUDE_HEADER="mkl_cblas.h")
else()
    message(STATUS "BLAS library not found")
    set(BLAS_INCLUDE_DIRS "")
    set(BLAS_LIBS "")
    set(BLAS_FOUND "NO")
endif()
