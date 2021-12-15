#!/bin/bash
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Note that the line endings in this file should be LF instead of CRLF in order to run correctly.
####################################################################################################
set -x -e

/opt/vcpkg/bootstrap-vcpkg.sh
/opt/vcpkg/vcpkg install catch2 tomlplusplus --overlay-ports=/opt/ports
/opt/vcpkg/vcpkg install accera-llvm --overlay-ports=/opt/ports
