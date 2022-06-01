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
    curl \
    ccache \
    libomp-dev \
    libunwind-dev \
    libvulkan-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    tar \
    unzip \
    zip \
  && rm -rf /var/lib/apt/lists/*

update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# TODO: support different versions of python if needed
python -m pip install -r requirements.txt
python -m pip install cuda-python

# install more recent cmake (Ubuntu 20.04 only bundles cmake 3.16)
curl -LO https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1-linux-x86_64.tar.gz \
  && tar zxvf cmake-3.23.1-linux-x86_64.tar.gz \
  && mv cmake-3.23.1-linux-x86_64 /usr/local/ \
  && ln -s /usr/local/cmake-3.23.1-linux-x86_64/bin/cmake /usr/bin/cmake \
  && rm -rf cmake-3.23.1-Linux-x86_64.tar.gz