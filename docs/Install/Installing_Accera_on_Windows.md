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
