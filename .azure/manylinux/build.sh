#!/bin/bash
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
set -x -e

ACCERA_ROOT=`pwd`
export VCPKG_TOOLCHAIN=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake
export ACCERA_PACKAGE_FOR_CI=1
export CMAKE_BUILD_PARALLEL_LEVEL=4
export VULKAN_SDK=/opt/vulkansdk

PYTHON_EXE=$1
DEST="${ACCERA_ROOT}/dist"

cd "${ACCERA_ROOT}"
${PYTHON_EXE} -m pip install -r requirements.txt
${PYTHON_EXE} setup.py build bdist_wheel -d ${DEST}

# Build the subpackages
cd "${ACCERA_ROOT}/accera/python/compilers"
${PYTHON_EXE} setup.py build bdist_wheel -d ${DEST}
cd "${ACCERA_ROOT}/accera/python/llvm"
${PYTHON_EXE} setup.py build bdist_wheel -d ${DEST}
cd "${ACCERA_ROOT}/accera/python/gpu"
${PYTHON_EXE} setup.py build bdist_wheel -d ${DEST}

${PYTHON_EXE} -m pip install auditwheel
cd "${DEST}"
${PYTHON_EXE} -m auditwheel repair accera-*
${PYTHON_EXE} -m auditwheel repair accera_compilers*
${PYTHON_EXE} -m auditwheel repair accera_llvm*

LD_LIBRARY_PATH="${VULKAN_SDK}/lib64" ${PYTHON_EXE} -m auditwheel repair accera_gpu*