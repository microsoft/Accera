![Accera logo](https://microsoft.github.io/Accera/assets/Accera_darktext.png)
<div style="margin-bottom:30px"></div>

<a href="https://pypi.org/project/accera/"><img src="https://badge.fury.io/py/accera.svg" alt="PyPI package version"/></a> <a href="https://pypi.org/project/accera/"><img src="https://img.shields.io/pypi/pyversions/accera" alt="Python versions"/></a> ![MIT License](https://img.shields.io/pypi/l/accera)

# Welcome to Accera

Accera is a compiler that enables you to experiment with loop optimizations without hand-writing Assembly code. Accera is available as a Python library and supports cross-compiling to a wide range of [processor targets](https://github.com/microsoft/Accera/blob/main/accera/python/accera/Targets.py).

Writing highly optimized compute-intensive code in a traditional programming language is a difficult and time-consuming process. It requires special engineering skills, such as fluency in Assembly language and a deep understanding of computer architecture. Manually optimizing the simplest numerical algorithms already requires a significant engineering effort. Moreover, highly optimized numerical code is prone to bugs, is often hard to read and maintain, and needs to be reimplemented every time a new target architecture is introduced. Accera aims to solve these problems.

Accera has three goals:

* Performance: generate the fastest implementation of any compute-intensive algorithm.
* Readability: do so without sacrificing code readability and maintainability.
* Writability: a user-friendly programming model, designed for agility.


## Install

To install for Linux, macOS, or Windows (requires Python 3.7-3.9):

```shell
pip install accera
```

See the [Install Instructions](https://microsoft.github.io/Accera/Install/) for more details on installing pre-built Python 3 packages and how to build Accera from source.


### Quickstart

In this example, we will:

* Implement matrix multiplication with a ReLU activation (matmul + ReLU), commonly used in in machine learning algorithms
  * Generate two implementations: a naive algorithm and one with loop transformations
* Compare the timings of both implementations

#### Run in your browser

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/microsoft/Accera/main?labpath=docs%2Fdemos%2Fquickstart.ipynb)

No installation is required. This will launch a Jupyter notebook with the quickstart example running in the cloud.

#### Run on your machine

1. Create a Python 3 script called `quickstart.py`:

    ```python
    import accera as acc

    # define placeholder inputs/output
    A = acc.Array(role=acc.Array.Role.INPUT, shape=(512, 512))
    B = acc.Array(role=acc.Array.Role.INPUT, shape=(512, 512))
    C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(512, 512))

    # implement the logic for matmul and relu
    matmul = acc.Nest(shape=(512, 512, 512))
    i1, j1, k1 = matmul.get_indices()
    @matmul.iteration_logic
    def _():
        C[i1, j1] += A[i1, k1] * B[k1, j1]

    relu = acc.Nest(shape=(512, 512))
    i2, j2 = relu.get_indices()
    @relu.iteration_logic
    def _():
        C[i2, j2] = acc.max(C[i2, j2], 0.0)

    package = acc.Package()

    # fuse the i and j indices of matmul and relu, add to the package
    schedule = acc.fuse(matmul.create_schedule(), relu.create_schedule(), partial=2)
    package.add(schedule, args=(A, B, C), base_name="matmul_relu_fusion_naive")

    # transform the schedule, add to the package
    f, i, j, k = schedule.get_indices()
    ii, jj = schedule.tile((i, j), (16, 16)) # loop tiling
    schedule.reorder(j, i, f, k, jj, ii) # loop reordering
    plan = schedule.create_plan()
    plan.unroll(ii) # loop unrolling
    package.add(plan, args=(A, B, C), base_name="matmul_relu_fusion_transformed")

    # build a dynamically-linked package (a .dll or .so) that exports both functions
    print(package.build(name="hello_accera", format=acc.Package.Format.HAT_DYNAMIC))
    ```

2. Ensure that you have a compiler in your PATH:

    * Windows: Install Microsoft Visual Studio and run `vcvars64.bat` to setup the command prompt
    * Linux/macOS: Install gcc

    Don't have a compiler handy? We recommend trying Accera in your browser instead [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/microsoft/Accera/main?labpath=docs%2Fdemos%2Fquickstart.ipynb).


3. Install Accera:

    ```shell
    pip install accera
    ```

4. Generate the library that implements two versions of matmul + ReLU:

    ```shell
    python quickstart.py
    ```

5. To consume and compare the library functions, create a file called `benchmark.py` in the same location:

    ```python
    import hatlib as hat
    import numpy as np

    # load the package
    hat_package = hat.load("hello_accera.hat")

    # call one of the functions with test inputs
    A_test = np.random.rand(512, 512).astype(np.float32)
    B_test = np.random.rand(512, 512).astype(np.float32)
    C_test = np.zeros((512, 512)).astype(np.float32)
    C_numpy = np.maximum(C_test + A_test @ B_test, 0.0)

    matmul_relu = hat_package["matmul_relu_fusion_transformed"]
    matmul_relu(A_test, B_test, C_test)

    # check correctness
    np.testing.assert_allclose(C_test, C_numpy, atol=1e-3)

    # benchmark all functions
    hat.run_benchmark("hello_accera.hat", batch_size=5, min_time_in_sec=5)
    ```

6. Run the benchmark to get the timing results:

    ```shell
    python benchmark.py
    ```

#### Next Steps

The [Manual](https://microsoft.github.io/Accera/Manual/00%20Introduction/) is a good place to start for an introduction to the Accera Python programming model.

In particular, the [schedule transformations](https://microsoft.github.io/Accera/Manual/03%20Schedules/#schedule-transformations) describe how you can experiment with different loop transformations with just a few lines of Python.

Finally, the `.hat` format is just a C header file containing metadata. Learn more about the [HAT format](https://github.com/microsoft/hat) and [benchmarking](https://github.com/microsoft/hat/tree/main/tools).


## How it works

In a nutshell, Accera takes the Python code that defines the loop schedule and algorithm and converts it into [MLIR](https://mlir.llvm.org/) intermediate representation (IR). Accera's compiler then takes this IR through a series of MLIR pipelines to perform transformations. The result is a binary library with a C header file. The library implements the algorithms that are defined in Python, and is compatible with the target.

To peek into the stages of IR transformation that Accera does, try replacing `format=acc.Package.Format.HAT_DYNAMIC` with `format=acc.Package.Format.MLIR_DYNAMIC` in `quickstart.py`, re-run the script, and search the `_tmp` subfolder for the intermediate `*.mlir` files. We plan to document these IR constructs in the future.

## Documentation

Get to know Accera's concepts and Python constructs in the [Documentation](https://microsoft.github.io/Accera/) page.

## Tutorials

More step-by-step examples are available on the [Tutorials](https://microsoft.github.io/Accera/Tutorials) page. We're working on more examples and tutorials soon.

## Contributions

Accera is a research platform-in-progress. We would love your contributions, feedback, questions, and feature requests! Please file a [Github issue](https://github.com/microsoft/Accera/issues/new) or send us a pull request. Please review the [Microsoft Code of Conduct](https://opensource.microsoft.com/codeofconduct/) to learn more.

## Credits

Accera is built using several open source libraries, including: [LLVM](https://llvm.org/), [toml++](https://marzer.github.io/tomlplusplus/), [tomlkit](https://github.com/sdispater/tomlkit), [vcpkg](https://vcpkg.io/en/index.html), [pyyaml](https://pyyaml.org/), and [HAT](https://github.com/microsoft/hat). For testing, we also use [numpy](https://github.com/numpy/numpy) and [catch2](https://github.com/catchorg/Catch2).

## License

This project is released under the [MIT License](https://github.com/microsoft/Accera/blob/main/LICENSE).