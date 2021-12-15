[//]: # (Project: Accera)

## Installing on Ubuntu

### Install dependencies

Accera requires the following tools and libraries for building the generated code:
* A C++ compiler, such as *GCC 8*
* Python 3.7 or newer
* OpenMP 5, if using parallelization

Ubuntu 20.04 is recommended. A quick way to start is to use a fresh Docker container for Ubuntu 20.04:

```shell
docker run -v $PWD:/code -it --entrypoint "/bin/bash" ubuntu:focal
```

Install Accera's dependencies:

```shell
apt update
apt-get install gcc-8 g++-8 python3 python3-pip libncurses5
```

Install the optional dependency if using parallelization:

```shell
apt-get install libomp-11-dev
```

### Install Accera

The 'accera` Python package is distributed as part of the Azure Artifacts feed. To consume the Python package from the command line, we will use pip (19.2+) and the Azure Artifacts keyring.

Before proceeding, ensure that you have obtained your Personal Access Token (PAT) from https://intelligentdevices.pkgs.visualstudio.com. You will be prompted to enter the PAT later on when installing the package.

Upgrade pip and install the Azure Artifacts keyring package:

```shell
python3 -m pip install --upgrade pip
pip3 install keyring artifacts-keyring
```

Install the package:
```shell
pip3 install -U accera --index-url https://intelligentdevices.pkgs.visualstudio.com/_packaging/Robopy/pypi/simple/
```
