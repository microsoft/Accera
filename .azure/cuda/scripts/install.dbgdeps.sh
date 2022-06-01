#!/bin/bash
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Note that the line endings in this file should be LF instead of CRLF in order to run correctly.
####################################################################################################
set -x -e

# dependencies for debugging only
apt-get update && apt-get install --no-install-recommends \
    llvm-12 \
  && rm -rf /var/lib/apt/lists/*
