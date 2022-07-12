[//]: # (Project: Accera)
[//]: # (Version: v1.2.7)

## Installing on Ubuntu

### Install dependencies

Accera requires the following tools and libraries for building the generated code:

* A C++ compiler, such as *GCC 8*
* Python 3.7 or newer
* OpenMP 5, if using parallelization

Ubuntu 20.04 is recommended. A quick way to start is to use a new Docker container for Ubuntu 20.04:

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

The `accera` Python package can be installed from PyPI:

```shell
pip install accera
```


