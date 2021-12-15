#!/bin/sh
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Note that the line endings in this file should be LF instead of CRLF in order to run correctly.
####################################################################################################
set -x -e

# Install pre-requisites for building Accera
yum -y install \
    ninja-build \
    ccache \
    zip

# Install Python pre-requisites for building Accera
PYTHON_EXES=("/opt/python/cp37-cp37m/bin/python3.7" "/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9")

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} -m pip install -r requirements.txt
done

# Choose a default Python for general use
ln -s /opt/python/cp39-cp39/bin/python3.9 /usr/bin/python3