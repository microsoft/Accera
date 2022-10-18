#!/bin/bash
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Note that the line endings in this file should be LF instead of CRLF in order to run correctly.
####################################################################################################
set -x -e

# Install Accera CI dependencies
apt-get update && apt-get install --no-install-recommends \
    ccache \
    g++-10 \
    libomp-dev \
    libunwind-dev \
    libvulkan-dev \
    ninja-build \
    pkg-config \
    zip \
  && rm -rf /var/lib/apt/lists/*

update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# TODO: support different versions of python if needed
python -m pip install -r requirements.txt
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1