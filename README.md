![Accera logo](https://microsoft.github.io/Accera/assets/Accera_darktext.png)
<div style="margin-bottom:30px"></div>

<a href="https://pypi.org/project/accera/"><img src="https://badge.fury.io/py/accera.svg" alt="PyPI package version"/></a> <a href="https://pypi.org/project/accera/"><img src="https://img.shields.io/pypi/pyversions/accera" alt="Python versions"/></a> ![MIT License](https://img.shields.io/pypi/l/accera)

Accera is a programming model, a domain-specific programming language embedded in Python (eDSL), and an optimizing cross-compiler for compute-intensive code. Accera currently supports CPU and GPU targets and focuses on optimization of nested for-loops.

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

#### Try Accera in your browser

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/microsoft/Accera/HEAD?labpath=docs%2Fdemos%2Fbinder%2Fquickstart.ipynb)

No installation required.

#### Run Accera on your local machine

In this quickstart example, you will:

* Implement a simple `hello_accera` function that performs basic matrix multiplication with a ReLU activation
* Build a [HAT](https://github.com/microsoft/hat) package with a dynamic (shared) library that exports this function
* Call the `hello_accera` function in the dynamic library with some NumPy arrays, and checks against a NumPy implementation

1. Create a Python 3 script called `quickstart.py`

```python
import accera as acc
import hatlib as hat
import numpy as np

A = acc.Array(role=acc.Array.Role.INPUT, shape=(16, 16))
B = acc.Array(role=acc.Array.Role.INPUT, shape=(16, 16))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(16, 16))

matmul = acc.Nest(shape=(16, 16, 16))
i1, j1, k1 = matmul.get_indices()

@matmul.iteration_logic
def _():
    C[i1, j1] += A[i1, k1] * B[k1, j1]

relu = acc.Nest(shape=(16, 16))
i2, j2 = relu.get_indices()

@relu.iteration_logic
def _():
    C[i2, j2] = acc.max(C[i2, j2], 0.0)

matmul_schedule = matmul.create_schedule()
relu_schedule = relu.create_schedule()

# fuse the first 2 indices of matmul and relu
schedule = acc.fuse(matmul_schedule, relu_schedule, partial=2)

package = acc.Package()
package.add(schedule, args=(A, B, C), base_name="hello_accera")

# build a dynamically-linked HAT package
package.build(name="mypackage", format=acc.Package.Format.HAT_DYNAMIC)

# load the package and call the function with random test input
hat_package = hat.load("mypackage.hat")
hello_accera = hat_package["hello_accera"]

A_test = np.random.rand(16, 16).astype(np.float32)
B_test = np.random.rand(16, 16).astype(np.float32)
C_test = np.zeros((16, 16)).astype(np.float32)

# compute using NumPy as a comparison
C_np = np.maximum(C_test + A_test @ B_test, 0)

hello_accera(A_test, B_test, C_test)

# compare the result with NumPy
np.testing.assert_allclose(C_test, C_np)
print(C_test)
print(C_np)
```

2. Ensure that you have a compiler in your PATH:

    * Windows: Install Microsoft Visual Studio and run `vcvars64.bat` to setup the command prompt
    * Linux/macOS: Install gcc

3. Install Accera:

```shell
pip install accera
```

4. Run the Python script:

```python
python quickstart.py
```

#### Next Steps

The function can be optimized using [schedule transformations](https://microsoft.github.io/Accera/Manual/03%20Schedules/#schedule-transformations). The [Manual](https://microsoft.github.io/Accera/Manual/00%20Introduction/) is a good place to start for an introduction to the Accera programming model.

## Documentation
Get to know Accera by reading the [Documentation](https://microsoft.github.io/Accera/).

You can find more step-by-step examples in the [Tutorials](https://microsoft.github.io/Accera/Tutorials).
