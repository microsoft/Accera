[//]: # (Project: Accera)
[//]: # (Version: v1.2.3)

## Installing on MacOS

### Install Dependencies

Accera requires the following tools and libraries:

* A C++ compiler that supports C++ 17, such as `clang`, which is bundled in XCode
* CMake 3.14 or newer
* Python 3.7 or newer
* Ninja
* Ccache
* LLVM OpenMP 5, if using parallelization

Homebrew is a package manager that makes it easy to install the prerequesits. Homebrew can be downloaded and installed by:

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

If you already have Homebrew installed, update it to the latest version by typing:

```
brew update
```

Install the dependencies:

```
brew install cmake python@3.7 ninja-build ccache libomp
```

#### Clang

Select the `clang` compiler from XCode:

```
xcode-select --install
```

### Clone Accera

A version of [git](https://git-scm.com/download) should already be included in XCode.

Clone the git repository:

```
git clone --recurse-submodules https://github.com/microsoft/Accera
```

### Build and install Accera

Run the `build.sh` script to install dependencies and build the Accera Python package (replace `<path_to_accera>` with the path to the cloned Accera repository).

```shell
cd <path_to_accera>
sh ./build.sh
```

Update or install the resulting `.whl` file from the `dist` sudirectory. The name depends on your Python version, your OS and your CPU architecture e.g.
```shell
pip install -U ./dist/accera-0.0.1-cp37-cp37-macosx_10_15_x86_64.whl --find-links=dist
```

### Build and install using CMake

Accera can also be built using CMake (intended for expert users).

#### Install dependencies

```shell
cd <path_to_accera>
git submodule init
git submodule update
./external/vcpkg/bootstrap-vcpkg.sh
./external/vcpkg/vcpkg install catch2 tomlplusplus accera-llvm --overlay-ports=external/llvm
```

The last command typically takes a few hours to build and then install Accera's fork of LLVM. We recommend you reserve at least 20GB of disk space for the LLVM build.

#### Configure CMake

```shell
cd <path_to_accera>
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
```

#### Build and run tests

```shell
cmake --build . --config Release
ctest -C Release
```

#### Install

```shell
cmake --build . --config Release --target install
```
