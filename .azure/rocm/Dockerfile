####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Usage: call docker build from the root of this repository
#  docker build -f .azure/rocm/Dockerfile . -t registry_name/rocm-linuxagent:latest
####################################################################################################

# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG ROCMVER=5.1.1-ub20

# cf: amddcgpuce/rocm:${ROCMVER}
FROM acceracontainers.azurecr.io/rocm:${ROCMVER}

ARG ROCMVER
RUN echo "ROCm Version: " ${ROCMVER}

ADD .azure/rocm/scripts /tmp/scripts
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
COPY .azure/rocm/scripts/start.sh .
RUN chmod +x start.sh

ENTRYPOINT ["./start.sh"]