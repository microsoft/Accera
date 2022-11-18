#!/bin/bash
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
set -x -e

CUDAVER=11.8.0-devel-ubuntu20.04

SCRIPT_DIR=$(dirname $(readlink -f "$0"))
ACCERA_ROOT=${SCRIPT_DIR}/../../
CWD=$(pwd)

# Running from the repository root, so that the Dockerfile can
# access all repo files from its build context (e.g. requirements.txt)
cd ${ACCERA_ROOT}
sudo docker build . \
    --build-arg CUDAVER=${CUDAVER} \
    --tag acceracontainers.azurecr.io/cuda-linuxagent:${CUDAVER} \
    --file .azure/cuda/Dockerfile

sudo docker tag acceracontainers.azurecr.io/cuda-linuxagent:${CUDAVER} \
    acceracontainers.azurecr.io/cuda-linuxagent:latest
cd ${CWD}