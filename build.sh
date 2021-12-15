#!/bin/sh
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

if [ -f "$ACCERA_ROOT/CMake/LLVMSetupConan.cmake" ]; then
    echo Using LLVM from Conan
    export LLVM_SETUP_VARIANT=Conan
else 
    echo Using LLVM from vcpkg
    export LLVM_SETUP_VARIANT=Default
    # Install LLVM (takes a couple of hours and ~20GB of space)
    ./vcpkg install accera-llvm --overlay-ports=../llvm
fi

# Build the accera package
cd "$ACCERA_ROOT"
python setup.py clean --all build bdist_wheel

# Build the subpackages
cd "$ACCERA_ROOT/accera/python/compilers"
python setup.py build bdist_wheel -d "$ACCERA_ROOT/dist"
cd "$ACCERA_ROOT/accera/python/llvm"
python setup.py build bdist_wheel -d "$ACCERA_ROOT/dist"
cd "$ACCERA_ROOT/accera/python/gpu"
python setup.py build bdist_wheel -d "$ACCERA_ROOT/dist"
cd "$ACCERA_ROOT"

echo Complete. Packages are in the 'dist' folder.