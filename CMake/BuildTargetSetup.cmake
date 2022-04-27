####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

if(APPLE)
  # cf. https://discourse.cmake.org/t/how-to-determine-which-architectures-are-available-apple-m1/2401/10
  # on macOS "uname -m" returns the architecture (x86_64 or arm64)
  execute_process(
      COMMAND uname -m
      RESULT_VARIABLE result
      OUTPUT_VARIABLE OSX_NATIVE_ARCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()