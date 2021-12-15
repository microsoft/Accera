####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(lib_name acc-vulkan-runtime-wrappers)

if(NOT Vulkan_FOUND)
  message(WARNING "Vulkan library not found, ${lib_name} won't be available to build")
else()
  message(STATUS "Vulkan library found at ${Vulkan_LIBRARY}, ${lib_name} will be an available build target")

  set(src
    src/vulkan-runtime-wrappers.cpp
    src/VulkanRuntime.cpp
  )
  set(include
    include/VulkanRuntime.h
  )
  add_library(${lib_name} SHARED
    ${src}
    ${include}
  )

  target_include_directories(${lib_name}
    PRIVATE
    ${Vulkan_INCLUDE_DIR}
    include
  )

  target_link_libraries(${lib_name}
    PUBLIC
    ${Vulkan_LIBRARY}
  )

  InstallAcceraHeaders(
    INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
                ${CMAKE_CURRENT_LIST_DIR}/include
  )
  InstallAcceraCppRuntimeLibrary(${lib_name})
  InstallAcceraPyRuntimeLibrary(${lib_name} accera-gpu "accera")

  set(shared_lib_dir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  if(MSVC)
    set(shared_lib_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif()
  set(RC_VULKAN_RUNTIME_WRAPPER_PATH ${shared_lib_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}${lib_name}${CMAKE_SHARED_LIBRARY_SUFFIX})

  file(RELATIVE_PATH RC_VULKAN_RUNTIME_WRAPPER_REL_PATH ${CMAKE_BINARY_DIR} ${RC_VULKAN_RUNTIME_WRAPPER_PATH})
  set_property(GLOBAL PROPERTY RC_VULKAN_RUNTIME_WRAPPER_REL_PATH ${RC_VULKAN_RUNTIME_WRAPPER_REL_PATH})
endif(NOT Vulkan_FOUND)
