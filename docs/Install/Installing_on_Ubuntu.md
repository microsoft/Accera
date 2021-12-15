[//]: # (Project: Accera)

## Installing on Ubuntu

### Install Dependencies

Accera requires the following tools and libraries:
* A C++ compiler that supports C++ 17, such as *GCC 8*
* CMake 3.14 or newer
* Python 3.7 or newer
* Ninja
* Ccache
* LLVM OpenMP 5, if using parallelization

```shell
sudo apt update -y
sudo apt-get install -y gcc-8 g++-8 cmake python3.7 ninja-build ccache libomp-11-dev pkg-config
```

Some Ubuntu distributions will install an older version of CMake. Check the version of cmake using `cmake --version`, and [download](https://cmake.org/download/) a newer version if older than 3.14.

### Clone Accera

#### Install [git](https://git-scm.com/download) if you don't already have it:

```
sudo apt-get install -y git
```

#### Clone the git repository:

```shell
git clone --recurse-submodules https://intelligentdevices.visualstudio.com/ELL/_git/Accera
```

### Build and install Accera

Run the `build.sh` script to install dependencies and build the Accera Python package (replace `<path_to_accera>` with the path to the cloned Accera repository).

```shell
cd <path_to_accera>
sh ./build.sh
```

Update or install the resulting `.whl` files from the `dist` sudirectory. The `--find-links` option tells pip to look at the `dist` subdirectory for the dependent packages. 
The name depends on your Python version, your OS and your CPU architecture e.g.
```shell
pip install -U ./dist/accera-0.0.1-cp37-cp37m-linux_x86_64.whl --find-links=dist
```

### Build and install using CMake

Accera can also be built using CMake (intended for expert users).

#### Configure CMake:

```shell
cd <path_to_accera>
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja [-DLLVM_SETUP_VARIANT=Default]
```

#### Build and run tests:

```shell
cmake --build . --config Release
ctest -C Release
```

#### Install:

```shell
cmake --build . --config Release --target install
```