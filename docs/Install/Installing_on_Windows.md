[//]: # (Project: Accera)

## Installing on Windows

### Install Dependencies

#### Visual Studio

Accera requires a C++ compiler that supports C++ 17. You can download [Visual Studio 2019 Enterprise Edition](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202019) or [Visual Studio 2022 Community Edition](https://visualstudio.microsoft.com/vs/). Install Update 10 or later which includes the LLVM OpenMP libraries only for VS 2019.

Select *Desktop Development with C++*.

Accera requires [Spectre-mitigated libraries](https://docs.microsoft.com/en-us/cpp/build/reference/qspectre?view=msvc-160):
1. Go to *Indivudual Components*
2. Type in "Spectre" in the search box
3. Select the latest version of the MSVC libraries, e.g. *MSVC v142 - VS 2019 C++ x64/x86 Spectre-mitigated libs (Latest)* (your actual version may vary)

#### CMake

Accera requires [CMake](https://cmake.org/) 3.14 or newer.  A version of CMake that satisfies this requirement is included with Visual Studio 2019  and Visual Studio 2022.

#### Python

Accera's packages require Python 3.7 64-bit or newer, plus a version of `pip` that supports 64-bit packages (`win_amd64`). One way to obtain this is to download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Download "Miniconda3 Windows 64-bit".

###### Optional: Create a conda environment
After installing Miniconda, you can optionally create an environment to manage different Python versions.

From an "Anaconda Prompt", create and then activate an environment for Python 3.7 (or a newer version if you prefer). Make sure to activate an environment from other applications as well that you use for development of Accera.

```shell
conda create -n py37 python=3.7
conda activate py37
```

### Clone Accera

Visual Studio 2019 and 2022 include a version of `git`. To use it, launch Visual Studio 2019 or 2022, and select `Clone a repository`.

Repository location:

```
https://intelligentdevices.visualstudio.com/ELL/_git/Accera
```

### Build and install Accera

From a command line that has Python in the path, such as an Anaconda Command Prompt, run the `build.bat` script to install dependencies and build the Accera Python package. Replace `<path_to_accera>` with the path to the cloned Accera repository.

```shell
cd <path_to_accera>
build.bat
```

Update or install the resulting `.whl` file from the `dist` sudirectory. The `--find-links` option tells pip to look at the `dist` subdirectory for the dependent packages.
 The whl filename depends on your Python version, your OS and your CPU architecture e.g.
```shell
pip install -U dist\accera-0.0.1-cp37-cp37m-win_amd64.whl --find-links=dist
```

### Build and install using CMake

Accera can also be built using CMake (intended for expert users).

#### Install dependencies

```shell
cd <path_to_accera>
git submodule init
git submodule update
external\vcpkg\bootstrap-vcpkg.bat
external\vcpkg\vcpkg install catch2:x64-windows tomlplusplus:x64-windows accera-llvm:x64-windows --overlay-ports=external\llvm
```

The last command typically takes a few hours to build and then install Accera's fork of LLVM. We recommend you reserve at least 20GB of disk space for the LLVM build.

#### Configure CMake

```shell
cd <path_to_accera>
mkdir build
cd build

# For Visual Studio 2019:
cmake .. -DCMAKE_BUILD_TYPE=Release -G"Visual Studio 16 2019" -Ax64

# For Visual Studio 2022:
cmake .. -DCMAKE_BUILD_TYPE=Release -G"Visual Studio 17 2022" -Ax64
```

#### Build and run tests

```shell
cmake --build . --config Release -- /m
ctest -C Release
```

#### Install

```
cmake --build . --config Release --target install -- /m
```

#### Use MSVC Compiler from Command Line
For benchmarking Accera, [benchmark\_hat\_package script](https://github.com/microsoft/Accera/tree/main/accera/benchmark-hat-package) use Microsoft C++ Compiler (cl.exe). However, it is
not included in the `PATH` environment variable. Run the following command on the
Anaconda Prompt or Command Prompt to get an access to `cl.exe`.

- Locate and then call the `vcvarsall.bat` file from your **Microsoft Visual Studio 2019** install location, selecting the `x64` configuration:
```
> call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
```
The output should look like (the actual version may vary for your install):
```
**********************************************************************
** Visual Studio 2019 Developer Command Prompt v16.11.7
** Copyright (c) 2021 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x64'
```
- Invoke `cl.exe` to test if the MSVC compiler is in the `PATH` (your installed version may vary):
```
> cl
Microsoft (R) C/C++ Optimizing Compiler Version 19.29.30137 for x64
```
Please check the details on `vcvarsall.bat` [here](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170).

**NOTE:** If you are using **Visual Studio 2022**, locate and then call the `vcvarsall.bat` file from your Microsoft Visual
Studio 2022 install location, selecting the `x64`cofniguration:
```
> call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```
The output should look like (the actual version may vary for your install):
```
**********************************************************************
** Visual Studio 2022 Developer Command Prompt v17.0.1
** Copyright (c) 2021 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x64'
```
Invoke `cl.exe` to test if the MSVC compiler is in the `PATH` (your installed version mayvary):
```
> cl
Microsoft (R) C/C++ Optimizing Compiler Version 19.30.30705 for x64
```

## Troubleshooting
- Activate Conda for Python 3.7 or any other version that you are working with in an application that you are using for setting up Accera.
- Update Microsoft Visual Studio from Microsoft Visual Studio Installer application if there are any errors similar to the ones shown below. Make sure to launch the Command Prompt again and follow the build/install steps for Accera from the beginning.
```
C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\14.20.27508\include\xtree(1164): error C2672: 'operator __surrogate_func': no matching overloaded function found [C:\accera\build\libraries\ir\ir.vcxproj]

C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\14.20.27508\include\xtree(1164): error C2893: Failed to specialize function template 'unknown-type std::less<void>::operator ()(_Ty1 &&,_Ty2 &&) noexcept(<expr>) const' [C:\accera\build\libraries\ir\ir.vcxproj]
```
