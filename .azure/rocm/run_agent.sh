#!/bin/bash
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
set -x -e

VARS=(AZP_URL AZP_TOKEN ACR_REPO ACR_USER ACR_SECRET)
for var in "${VARS[@]}"; do
    if [[ (-z "${!var}") ]]; then
        echo "${var} is not set"
        exit
    fi
done

ROCMVER=5.1.1-ub20
IMAGE=${ACR_REPO}/rocm-linuxagent:${ROCMVER}
POOL=LinuxAMDGPUPool

SCRIPT_DIR=$(dirname $(readlink -f "$0"))
ACCERA_ROOT=${SCRIPT_DIR}/../../

sudo docker login -u ${ACR_USER} -p ${ACR_SECRET} ${ACR_REPO}

#
# Debugging Example:
#
# Uncomment this block below to launch a debug container:
# It maps the volume to our local start.sh so we can modify and run manually to debug
# The Accera and hat repositories are mapped in case we need to debug build configuration issues
# The hat repository should be in the same root as the Accera repository
#
# sudo -E docker run \
#  -v ${SCRIPT_DIR}/scripts:/azp \
#  -v ${ACCERA_ROOT}:/code \
#  -v ${ACCERA_ROOT}../hat:/hat \
#  -it --entrypoint /bin/bash \
#  --ipc=host --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
#  -e AZP_URL \
#  -e AZP_TOKEN \
#  -e AZP_POOL=${POOL} \
#  -e AZP_AGENT_NAME=$(hostname)-${ROCMVER} \
#  -e TARGETARCH=linux-x64 \
#  ${IMAGE}

sudo -E docker run -d \
 --restart=always \
 --ipc=host --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
 -e AZP_URL \
 -e AZP_TOKEN \
 -e AZP_POOL=${POOL} \
 -e AZP_AGENT_NAME=$(hostname)-${ROCMVER} \
 -e TARGETARCH=linux-x64 \
 ${IMAGE}
