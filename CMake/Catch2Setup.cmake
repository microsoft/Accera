####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

include_guard()

# Prerequisites: vcpkg install catch2 or vcpkg install catch2:x64-windows
find_package(Catch2 CONFIG REQUIRED)

include(Catch)
include(ParseAndAddCatchTests)
