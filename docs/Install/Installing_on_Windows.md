[//]: # (Project: Accera)

## Installing on Windows

### Install Dependencies

#### Visual Studio

Accera requires a C++ compiler that supports C++ 17. You can download [Visual Studio 2019 Enterprise Edition](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202019) for free. Install Update 10 or later which includes the LLVM OpenMP libraries.

Select *Desktop Development with C++*.

Accera requires [Spectre-mitigated libraries](https://docs.microsoft.com/en-us/cpp/build/reference/qspectre?view=msvc-160):
1. Go to *Indivudual Components*
2. Type in "Spectre" in the search box
3. Select the latest version of the MSVC libraries, e.g. *MSVC v142 - VS 2019 C++ x64/x86 Spectre-mitigated libs (Latest)* (your actual version may vary)

#### CMake

Accera requires [CMake](https://cmake.org/) 3.14 or newer. A version of CMake that satisfies this requirement is included with Visual Studio 2019.

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

Visual Studio 2019 includes a version of `git`. To use it, launch Visual Studio 2019, and select `Clone a repository`.

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

Accera can also be built using CMake. This is for expert users only.

#### Configure CMake

```shell
cd <path_to_accera>
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -G"Visual Studio 16 2019" -Ax64 [-DLLVM_SETUP_VARIANT=Default]
```

#### Build and run tests

```shell
cmake --build . --config Release -- /m
ctest -C Release
```

#### Install

```shell
cmake --build . --config Release --target install -- /m
```