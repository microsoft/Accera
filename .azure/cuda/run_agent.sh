#!/bin/bash
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# To run this, use bash: bash ./run_agent.sh
####################################################################################################
set -x -e

VARS=(AZP_URL AZP_TOKEN ACR_REPO)
for var in "${VARS[@]}"; do
    if [[ (-z "${!var}") ]]; then
        echo "${var} is not set"
        exit
    fi
done

CUDAVER=11.7.0-devel-ubuntu20.04
IMAGE=${ACR_REPO}/cuda-linuxagent:${CUDAVER}
POOL=LinuxNVGPUPool

SCRIPT_DIR=$(dirname $(readlink -f "$0"))
ACCERA_ROOT=${SCRIPT_DIR}/../../

#
# Debugging Example:
#
# Uncomment this block below to launch a debug container:
# It maps the volume to our local start.sh so we can modify and run manually to debug
# The Accera repository is mapped in case we need to debug build configuration issues
#
# sudo -E docker run \
#  --cap-add=SYS_ADMIN \
#  -v ${SCRIPT_DIR}/scripts:/azp \
#  -v ${ACCERA_ROOT}:/code \
#  -it --entrypoint /bin/bash \
#  --gpus all \
#  -e AZP_URL \
#  -e AZP_TOKEN \
#  -e AZP_POOL=${POOL} \
#  -e AZP_AGENT_NAME=$(hostname)-${CUDAVER} \
#  -e TARGETARCH=linux-x64 \
#  ${IMAGE}

sudo -E docker run -d \
 --cap-add=SYS_ADMIN \
 --restart=always \
 --gpus all \
 -e AZP_URL \
 -e AZP_TOKEN \
 -e AZP_POOL=${POOL} \
 -e AZP_AGENT_NAME=$(hostname)-${CUDAVER} \
 -e TARGETARCH=linux-x64 \
 ${IMAGE}
