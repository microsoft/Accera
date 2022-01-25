#!/bin/sh
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
set -e

# Build script for the Accera Python package
ACCERA_ROOT=`pwd`

# Ensure that submodules are cloned
git submodule init
git submodule update

# Install dependencies
# Linux: apt get install pkg-config
pip install -r requirements.txt
cd external/vcpkg
./bootstrap-vcpkg.sh
./vcpkg install catch2 tomlplusplus --overlay-ports=../llvm

if [ -z "${LLVM_SETUP_VARIANT}" ] && [ -f "$ACCERA_ROOT/CMake/LLVMSetupConan.cmake" ]; then
    echo Using LLVM from Conan
    export LLVM_SETUP_VARIANT=Conan
else 
    echo Using LLVM from vcpkg
    export LLVM_SETUP_VARIANT=Default

    # Uncomment these lines below to build a debug version (will include release as well, due to vcpkg quirks)
    # export VCPKG_BUILD_TYPE=debug
    # export VCPKG_KEEP_ENV_VARS=LLVM_BUILD_TYPE

    # Install LLVM (takes a couple of hours and ~20GB of space)
    ./vcpkg install accera-llvm --overlay-ports=../llvm
fi

# Build the accera package
cd "$ACCERA_ROOT"
python setup.py build bdist_wheel

# Build the subpackages
cd "$ACCERA_ROOT/accera/python/compilers"
python setup.py build bdist_wheel -d "$ACCERA_ROOT/dist"
cd "$ACCERA_ROOT/accera/python/llvm"
python setup.py build bdist_wheel -d "$ACCERA_ROOT/dist"
cd "$ACCERA_ROOT/accera/python/gpu"
python setup.py build bdist_wheel -d "$ACCERA_ROOT/dist"
cd "$ACCERA_ROOT"

echo Complete. Packages are in the 'dist' folder.