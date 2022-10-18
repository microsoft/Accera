[//]: # (Project: Accera)
[//]: # (Version: v1.2.11)

## Installing on MacOS

### Install dependencies

Accera requires the following tools and libraries for building the generated code:

* A C++ compiler, such as `clang`, which is bundled in XCode
* Python 3.7 or newer
* OpenMP 5, if using parallelization

Homebrew is a package manager that makes it easy to install the prerequisites. Homebrew can be downloaded and installed by:

```shell
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

If you already have Homebrew installed, update it to the latest version by typing:

```shell
brew update
```

Install the dependencies:

```shell
brew install cmake python@3.7
```

Install the optional dependency if using parallelization:

```shell
brew install libomp
```

#### Clang

Select the `clang` compiler from XCode:

```shell
xcode-select --install
```

### Install Accera

The `accera` Python package can be installed from PyPI:

```shell
pip install accera
```


