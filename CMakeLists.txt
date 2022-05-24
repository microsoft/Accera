####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# Error on non-existent dependency in add_dependencies.
cmake_policy(SET CMP0046 NEW)
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif(POLICY CMP0116)

if(POLICY CMP0116)
  # Resolve relative DEPFILE paths in add_custom_command.
  cmake_policy(SET CMP0116 NEW)
endif()

set(ACCERA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${ACCERA_ROOT}/CMake")
include(CompilerCache)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake
    CACHE STRING "vcpkg toolchain file")

project(ACCERA CXX)

include(GetGitVersion)
get_git_version(ACCERA_VERSION)
message(STATUS "Accera version ${ACCERA_VERSION}")

option(ACCERA_BUILD_DOCS "Build documentation" OFF)
option(ACCERA_DISABLE_LIT_TESTS "Disable Accera tests that use lit/FileCheck" OFF)
option(STRICT_MODE "Build with 'warnings as errors'" OFF)
option(USE_MKL "Build with Intel MKL" OFF)

option(USE_LIBCXX "Build with libc++ if using the Clang compiler" OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  if(USE_LIBCXX OR (CMAKE_HOST_SYSTEM_NAME STREQUAL Darwin))
    add_compile_options(-stdlib=libc++)
    link_libraries(-lc++ -lc++abi)
  endif(USE_LIBCXX OR (CMAKE_HOST_SYSTEM_NAME STREQUAL Darwin))
endif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)

# Try to create a compilation database, which is useful to have when working
# with clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT ACCERA_EXTERNAL_DIR)
  set(ACCERA_EXTERNAL_DIR "${ACCERA_ROOT}/external"
      CACHE STRING "Directory to install external dependencies"
  )
endif(NOT ACCERA_EXTERNAL_DIR)

if(NOT ACCERA_TOOLS_DIR)
  set(ACCERA_TOOLS_DIR "${CMAKE_BINARY_DIR}/bin"
      CACHE STRING "Directory to place executables")
endif(NOT ACCERA_TOOLS_DIR)
message(STATUS "ACCERA_TOOLS_DIR: " ${ACCERA_TOOLS_DIR})

# Turn on ability to create folders to organize projects
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set C++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apply -fPIC where applicable to the platform
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find the platform-specific way of working with threads
if(NOT MSVC)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  set(CMAKE_THREAD_PREFER_PTHREAD ON)
endif(NOT MSVC)
find_package(Threads REQUIRED)

if(STRICT_MODE)
  message("-- Turning on strict mode with warnings as errors")
endif()

# Turn on ctest tests
enable_testing()

# Set up global variables to help find NuGet projects
set(PACKAGE_ROOT ${ACCERA_EXTERNAL_DIR})

# Set up install location in build directory
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

include(BuildTargetSetup)

if(USE_MKL)
  include(MKLSetup)
else()
  include(OpenBLASSetup)
endif()

include(Catch2Setup)
include(LLVMSetup) # controls the LLVM_SETUP_VARIANT variable
include(VulkanSetup)
include(CopySharedLibraries)
include(InstallUtilities)
include(AddPrecompiledHeader)
include(DocSetup)

# Prerequisites: vcpkg install tomlplusplus or vcpkg install tomlplusplus:x64-windows
find_package(tomlplusplus CONFIG REQUIRED)

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if(MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")


set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
if(MSVC)
  # Set Visual Studio-specific options
  add_definitions(-DUNICODE -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
  add_compile_options(/utf-8)
  add_compile_options(/MP)
  add_compile_options(/bigobj)
  add_compile_options(/W4)
  add_compile_options(/permissive-)
  if(STRICT_MODE)
    add_compile_options(/WX)
  endif()
  # the debug linker needs to match how LLVM was built (either /MD or /MDd)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${LLVM_MD_OPTION}")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${LLVM_MD_OPTION}")
  if (MSVC_VERSION GREATER 1900)
    # Set options for Control Flow Guard and Qspectre
    add_compile_options(/guard:cf)
    add_compile_options(/DYNAMICBASE)
    add_compile_options(/Qspectre)
    # Enable SDL checks
    add_compile_options(/sdl)
    # Enable Control-flow Enforcement Technology (CET) Shadow Stack mitigation
    add_link_options("/CETCOMPAT")
  endif()
else()
  # Set Clang/GCC-specific options
  add_compile_options(-Wall)
  if(STRICT_MODE)
    add_compile_options(-Werror)
  endif()
  add_compile_options(-Wsign-compare)
  add_compile_options(-Wno-missing-braces)
  add_compile_options(-Wno-shadow)
  add_compile_options(-Wmissing-field-initializers)
  add_compile_options(-Wno-unknown-pragmas)
  add_compile_options(-Wno-comment)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb3 -O0")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -ggdb3 -O0")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ggdb3")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -ggdb3")
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    if(CMAKE_BUILD_TYPE STREQUAL Debug)
    # Set options for Control Flow Integrity
      add_compile_options(-fsanitize=cfi)
    endif(CMAKE_BUILD_TYPE STREQUAL Debug)

    add_compile_options(-Wno-backslash-newline-escape)
    add_compile_options(-Wno-self-assign)
    add_compile_options(-fcolor-diagnostics)
    # Exit after the first 2 errors are reported
    add_compile_options(-ferror-limit=2)
  else() # GCC
    add_compile_options(-Wno-ignored-attributes)
    add_compile_options(-fdiagnostics-color=always)
    add_compile_options(-Wl,dynamicbase)

    if(NOT ${OSX_NATIVE_ARCH} STREQUAL "arm64")
      # Set options for Control Flow Integrity (not supported on macos/arm64)
      add_compile_options(-fcf-protection)
    endif()
  endif()
endif()

# Print helpful message if LLVM not found on Windows
if(NOT LLVM_FOUND AND WIN32)
  message(FATAL_ERROR "LLVM not found.")
endif()

add_definitions(-D__ACCERA__)

# Create custom target for tablegen-generated documentation
add_custom_target(tablegen-ir-docs)

# Include cmake projects for libraries and executables
add_subdirectory(accera)

add_subdirectory(tools)

# Add user directories to Accera build if requested
if(EXISTS "${ACCERA_ROOT}/user")
  # Add root user directory if it has a CMakeLists.txt file and
  # INCLUDE_IN_ACCERA_BUILD.txt file
  if(EXISTS"${ACCERA_ROOT}/user/CMakeLists.txt"
     AND EXISTS "${ACCERA_ROOT}/user/INCLUDE_IN_ACCERA_BUILD.txt"
  )
    message(STATUS "Adding user directory to Accera build")
    add_subdirectory(user)
  endif()

  # Now add all child directories that have CMakeLists.txt files and
  # INCLUDE_IN_ACCERA_BUILD.txt file
  file(GLOB children RELATIVE "${ACCERA_ROOT}/user" "${ACCERA_ROOT}/user/*")
  foreach(child ${children})
    if(IS_DIRECTORY "${ACCERA_ROOT}/user/${child}"
       AND EXISTS "${ACCERA_ROOT}/user/${child}/CMakeLists.txt"
       AND EXISTS
           "${ACCERA_ROOT}/user/${child}/INCLUDE_IN_ACCERA_BUILD.txt"
    )
      message(STATUS "Adding user directory ${child} to Accera build")
      add_subdirectory("user/${child}")
    endif()
  endforeach()
endif()

option(BUILD_PYTHON_BINDINGS "Enable building of Python3 bindings" ON)
if (BUILD_PYTHON_BINDINGS)
    add_subdirectory(accera/python)
    install(
            DIRECTORY accera/hat/scripts/ # trailing slash is important!
            COMPONENT accera
            DESTINATION "accera/hat"
    )
endif (BUILD_PYTHON_BINDINGS)
