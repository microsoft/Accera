@echo off
REM ####################################################################################################
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License. See LICENSE in the project root for license information.
REM Build script for the Accera Python package
REM ####################################################################################################

setlocal

set ACCERA_ROOT=%~dp0

REM Ensure that submodules are cloned
git submodule update --init --recursive
git pull --recurse-submodules

REM Install dependencies
pip install -r requirements.txt
cd external\vcpkg
call bootstrap-vcpkg.bat
vcpkg install catch2:x64-windows tomlplusplus:x64-windows --overlay-ports=..\llvm

if exist "%ACCERA_ROOT%\CMake\LLVMSetupConan.cmake" (
    echo Using LLVM from Conan
    set LLVM_SETUP_VARIANT=Conan
) else (
    echo Using LLVM from vcpkg
    set LLVM_SETUP_VARIANT=Default
    REM Install LLVM (takes a couple of hours and ~20GB of space)
    vcpkg install accera-llvm:x64-windows --overlay-ports=..\llvm
)

REM Build the accera package
cd "%ACCERA_ROOT%"
python setup.py build bdist_wheel

REM Build the subpackages
cd "%ACCERA_ROOT%\accera\python\compilers"
python setup.py build bdist_wheel -d "%ACCERA_ROOT%\dist"
cd "%ACCERA_ROOT%\accera\python\llvm"
python setup.py build bdist_wheel -d "%ACCERA_ROOT%\dist"
cd "%ACCERA_ROOT%\accera\python\gpu"
python setup.py build bdist_wheel -d "%ACCERA_ROOT%\dist"
cd "%ACCERA_ROOT%"

echo Complete. Packages are in the '%ACCERA_ROOT%\dist' folder.