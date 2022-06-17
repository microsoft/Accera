####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Usage: call docker build from the root of this repository
#  docker build -f .azure/cuda/Dockerfile . -t registry_name/cuda-linuxagent:latest
####################################################################################################

ARG CUDAVER=11.6.2-devel-ubuntu20.04

# cf: nvidia/cuda:${CUDAVER}
FROM acceracontainers.azurecr.io/nvidia/cuda:${CUDAVER}

ARG CUDAVER
RUN echo "CUDA Version: " ${CUDAVER}

ADD .azure/cuda/scripts /tmp/scripts
ADD requirements.txt /tmp/scripts/requirements.txt

WORKDIR /tmp/scripts
RUN sh /tmp/scripts/install.azp.sh
RUN sh /tmp/scripts/install.builddeps.sh
RUN sh /tmp/scripts/install.dbgdeps.sh
RUN rm -rf /tmp/scripts

ENV CC=gcc-10 \
    CXX=g++-10 \
    CMAKE_C_COMPILER=gcc-10 \
    CMAKE_CXX_COMPILER=g++-10

# Start the build agent
# Can be 'linux-x64', 'linux-arm64', 'linux-arm', 'rhel.6-x64'.
ENV TARGETARCH=linux-x64
WORKDIR /azp
COPY .azure/cuda/scripts/start.sh .
RUN chmod +x start.sh

ENTRYPOINT ["./start.sh"]