#!/bin/bash
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Note that the line endings in this file should be LF instead of CRLF in order to run correctly.
####################################################################################################
set -x -e

SRC_PATH=`pwd`

SDK_VERSION=`curl -L https://vulkan.lunarg.com/sdk/latest/linux.txt`
INSTALL_PATH=/opt/vulkansdk
mkdir ${INSTALL_PATH}

git clone https://github.com/KhronosGroup/Vulkan-Loader.git
cd Vulkan-Loader
git checkout tags/sdk-${SDK_VERSION}

mkdir build
cd build
python3 ../scripts/update_deps.py
cmake -C helper.cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
    -DBUILD_WSI_WAYLAND_SUPPORT=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_WSI_XLIB_SUPPORT=OFF ..
cmake --build . --config Release --target install

cp -rf ${SRC_PATH}/Vulkan-Loader/build/Vulkan-Headers/build/install/include ${INSTALL_PATH}
export VULKAN_SDK=${INSTALL_PATH}