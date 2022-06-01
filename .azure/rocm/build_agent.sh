#!/bin/bash
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
set -x -e

ROCMVER=5.1.1-ub20

SCRIPT_DIR=$(dirname $(readlink -f "$0"))
ACCERA_ROOT=${SCRIPT_DIR}/../../
CWD=$(pwd)

# Running from the repository root, so that the Dockerfile can
# access all repo files from its build context (e.g. requirements.txt)
cd ${ACCERA_ROOT}
sudo docker build . \
    --build-arg ROCMVER=${ROCMVER} \
    --tag acceracontainers.azurecr.io/rocm-linuxagent:${ROCMVER} \
    --file .azure/rocm/Dockerfile

sudo docker tag acceracontainers.azurecr.io/rocm-linuxagent:${ROCMVER} \
    acceracontainers.azurecr.io/rocm-linuxagent:latest
cd ${CWD}