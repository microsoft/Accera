####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Inspects the latest git tag and sets a variable with the simplified version (vN.N.N)
# cf. llvm/third-party/benchmark/cmake/GetGitVersion.cmake
####################################################################################################

include_guard()

find_package(Git QUIET)

function(get_git_version var)
  if(GIT_EXECUTABLE)
      execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --match "v[0-9]*.[0-9]*.[0-9]*" --abbrev=0
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
          RESULT_VARIABLE status
          OUTPUT_VARIABLE GIT_DESCRIBE_VERSION
          ERROR_QUIET)
      if(status)
          set(GIT_DESCRIBE_VERSION "v0.0.0")
      endif()
      
      string(STRIP ${GIT_DESCRIBE_VERSION} GIT_VERSION)
  else()
      set(GIT_VERSION "0.0.0")
  endif()

  set(${var} ${GIT_VERSION} PARENT_SCOPE)
endfunction()
