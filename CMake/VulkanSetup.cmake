####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Centralized place to try to find the Vulkan SDK, installed from https://vulkan.lunarg.com/
####################################################################################################

# At first try "FindVulkan" from:
# https://cmake.org/cmake/help/v3.7/module/FindVulkan.html
if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
  find_package(Vulkan)
endif()

# If Vulkan is not found try a path specified by VULKAN_SDK.
if (NOT Vulkan_FOUND)
  if ("$ENV{VULKAN_SDK}" STREQUAL "")
    message(WARNING "Vulkan SDK was not found with CMake intrinsics and environment variable VULKAN_SDK is not set")
  else()
    find_library(Vulkan_LIBRARY vulkan HINTS "$ENV{VULKAN_SDK}/lib" REQUIRED)
    if (Vulkan_LIBRARY)
      set(Vulkan_FOUND ON)
      set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/include")
      message(STATUS "Found Vulkan: " ${Vulkan_LIBRARY})
    endif()
  endif()
endif()
