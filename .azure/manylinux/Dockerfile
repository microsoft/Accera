####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Usage: call docker build from the root of this repository
#  docker build -f .azure\manylinux\Dockerfile . -t registry_name/accera-llvm-manylinux2014:latest
####################################################################################################

# cf: quay.io/pypa/manylinux2014_x86_64:2022-04-24-d28e73e
# cf. https://quay.io/repository/pypa/manylinux2010_x86_64?tab=tags
FROM acceracontainers.azurecr.io/pypa/manylinux2014_x86_64:2022-04-24-d28e73e

ADD .azure/manylinux/scripts /tmp/scripts
ADD requirements.txt /tmp/scripts/requirements.txt

WORKDIR /tmp/scripts
RUN sh /tmp/scripts/install.manylinux.sh
RUN sh /tmp/scripts/install.vulkanloader.sh

ADD external/vcpkg /opt/vcpkg
ADD external/llvm /opt/ports
WORKDIR /opt/vcpkg
RUN sh /tmp/scripts/install.vcpkg.sh
RUN rm -rf /tmp/scripts