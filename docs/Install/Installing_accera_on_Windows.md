[//]: # (Project: Accera)

## Installing on Windows

### Install dependencies

#### Visual Studio

Accera's generated code requires a C++ compiler. Download [Visual Studio 2019 Enterprise Edition](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202019) or [Visual Studio 2022 Community Edition](https://visualstudio.microsoft.com/vs/), and select *Desktop development with C++* during installation. 

If you've selected VS 2019 and would like to use parallelization, ensure that Update 10 or later is installed. Both VS 2019 Update 10 or later and VS 2022 will include the LLVM OpenMP libraries.

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

The `accera` Python package can be installed from PyPI:

```shell
pip install accera
```

### Use MSVC Compiler from Command Line
For benchmarking Accera, the [benchmark\_hat\_package script](https://github.com/microsoft/Accera/tree/main/accera/benchmark-hat-package) uses the Microsoft C++ (MSVC) Compiler (`cl.exe`) to build HAT packages.

You can add `cl.exe` to the `PATH` environment variable for your command prompt by following these steps:

1. Locate and then call the `vcvarsall.bat` file from your Microsoft Visual Studio install location, selecting the `x64` configuration:

    #### Visual Studio 2019:
    ```shell
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
    ```

    The output should look like (the actual version may vary for your install):
    ```
    **********************************************************************
    ** Visual Studio 2019 Developer Command Prompt v16.11.7
    ** Copyright (c) 2021 Microsoft Corporation
    **********************************************************************
    [vcvarsall.bat] Environment initialized for: 'x64'
    ```

    #### Visual Studio 2022:

    ```shell
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    ```

    The output should look like (the actual version may vary for your install):
    ```
    **********************************************************************
    ** Visual Studio 2022 Developer Command Prompt v17.0.1
    ** Copyright (c) 2021 Microsoft Corporation
    **********************************************************************
    [vcvarsall.bat] Environment initialized for: 'x64'
    ```

2. Invoke `cl.exe` to test if the MSVC compiler is in the `PATH` (your installed version may vary):

    #### Visual Studio 2019:
    ```shell
    cl
    ```

    The output should look like (the actual version may vary for your install):
    ```
    Microsoft (R) C/C++ Optimizing Compiler Version 19.29.30137 for x64
    ```

    #### Visual Studio 2022:
    ```shell
    cl    
    ```

    The output should look like (the actual version may vary for your install):
    ```
    Microsoft (R) C/C++ Optimizing Compiler Version 19.30.30705 for x64
    ```

You can find documentation on `vcvarsall.bat` [here](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170).