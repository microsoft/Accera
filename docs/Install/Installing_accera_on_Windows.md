[//]: # (Project: Accera)

## Installing on Windows

### Install dependencies

#### Visual Studio

Accera's generated code requires a C++ compiler. You can download [Visual Studio 2019 Enterprise Edition](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202019) for free, and select *Desktop development with C++* during installation. If using parallelization, ensure that you have Update 10 or later installed, which contains the LLVM OpenMP libraries.

#### Python

Accera's packages require Python 3.7 64-bit or newer, plus a version of `pip` that supports 64-bit packages (`win_amd64`). One way to obtain this is to download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Download "Miniconda3 Windows 64-bit".


###### Optional: Create a conda environment
After installing Miniconda, you can optionally to create an environment to manage different Python versions.

From an "Anaconda Prompt", create and then activate an environment for Python 3.7 (or a newer version if you prefer):

```shell
conda create -n py37 python=3.7
conda activate py37
```

### Install Accera

The 'accera` Python package can be installed from PyPI:

```shell
pip install accera
```

### Use Microsoft Visual C++ Compiler from Command Line
For benchmarking Accera, [benchmark\_hat\_package script](../../accera/benchmark-hat-package/README.md) use Microsoft C++ (MSVC) Compiler (`cl.exe`). However, it is not included in the `PATH` environment variable. Run the following command on
Anaconda Prompt or Command Prompt to get an access to `cl.exe`.

- Locate and then call the `vcvarsall.bat` file from your Microsoft Visual Studio 2019 install location, selecting the `x64` configuration:
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
- You can now invoke `cl.exe` to test if the MSVC compiler is in the `PATH` (your installed version may vary):
```
> cl
Microsoft (R) C/C++ Optimizing Compiler Version 19.29.30137 for x64
```
Please check the details on `vcvarsall.bat` [here](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170).
