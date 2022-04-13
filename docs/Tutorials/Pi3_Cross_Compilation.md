[//]: # (Project: Accera)
[//]: # (Version: v1.2.3)

# Cross Compiling for the Raspberry Pi 3

By the end of this tutorial, you will learn how to:

* Cross compile a simple Matrix Multiplication (MatMul) function for execution on a Raspberry Pi 3
* Produce a [HAT](https://github.com/microsoft/hat) package containing the MatMul function that can be called on Pi 3 target
* Call the function from C or C++ code on a Raspberry Pi 3

## Prerequisites

* This tutorial assumes you already have Accera installed. If not, you can find the instructions in [here](../Install/README.md)
* You should also be familiar with writing Python and C++
* You have access to a Raspberry Pi 3 device

## A naive MatMul algorithm

Let's consider the example of multiplying matrices A and B, and adding the result into matrix C. In NumPy syntax, this can be expressed as:

```
C += A @ B
```

A naive algorithm for matrix multiplication typically contains 3 nested for loops. Expressed in Python, this could like:

```
# A.shape = (M, K), B.shape = (K, N), C.shape = (M, N)

for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i, j] += A[i, k] * B[k, j]
```

### Accera Python DSL

We will now walk through a naive Matrix Multiplication (MatMul) using Accera. In order to cross compile on the host for a different target, instead of using the default target that is the host machine, we will specify a target that represents a Raspberry Pi 3.

Create an empty file called `hello_matmul_pi3_generator.py`. First we'll import Accera's module.

```python
import accera as acc
```

Define some matrix sizes. A will be M by K, B will be K by N, and C will be M by N.

```python
# Define our matrix sizes
M = 128
N = 256
K = 256
```

Write a Python function that receives arrays `A`, `B` and `C`. These are our input and input/output matrices.

```python
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, K))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(K, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))
```

Here, we will use the `Nest` class to define our 3-layered nested for loop. The range indices are `M`, `N`, and `K`, with the outermost loop (`M`) listed first. We can get the loop nest indices in order to perform the computation.

```python
# Define the loop nest
nest = acc.Nest(shape=(M, N, K))

# Get the loop nest indices
i, j, k = nest.get_indices()
```

Next we define the logic of each iteration of the loop nest:
```python
# Define the loop nest logic
@nest.iteration_logic
def _():
    C[i, j] += A[i, k] * B[k, j]
```

We have finished defining the logic of MatMul, and let's define the schedule which controls how the logic is executed. To do this, we first create the schedule from the nest:

```python
sched = nest.create_schedule()
```

At this point, `sched` represents the default schedule for our algorithm. We can also perform some basic transformations on this schedule. For example, the following lines of code will split the `k` index in blocks of 4 (so `k`, `k+4`, `k+8`, and so on).

```python
# Split the k loop into blocks of 4, effectively doing this
# (assuming K is divisible by 4):
#
# for i in range(M):
#    for j in range(N):
#        # Split k into two loops
#        for k in range(0, K, 4):
#            for kk in range(4):
#                C[i, j] += A[i, k + kk] * B[k + kk, j]
#
# If k is not divisible by 4, Accera will take care of the boundary
# case for you.
kk = sched.split(k, 4)
```

The split index is now `k` and `kk`.

The next step is to create a plan from the schedule. For instance, we can use this plan to unroll the innermost loop.

```python
# Create a plan, specify the target to be a Raspberry Pi 3
pi3 = acc.Target(acc.Target.Model.RASPBERRY_PI_3B)
plan = sched.create_plan(pi3)

# Unroll kk, effectively doing this
# (assuming K is divisible by 4):
#
# for i in range(M):
#    for j in range(N):
#        for k in range(0, K, 4):
#            # Unrolled kk
#            C[i, j] += A[i, k + 0] * B[k + 0, j]
#            C[i, j] += A[i, k + 1] * B[k + 1, j]
#            C[i, j] += A[i, k + 2] * B[k + 2, j]
#            C[i, j] += A[i, k + 3] * B[k + 3, j]
#
# If k is not divisible by 4, Accera will take care of the boundary
# case for you.
plan.unroll(kk)
```

Use the plan to add a callable function named `hello_matmul_pi3_py` to a HAT package.

```python
# Create a package and add a function to the package based on the plan
package = acc.Package()
package.add(plan, args=(A, B, C), base_name="hello_matmul_pi3_py")
```

Finally, we build the statically-linked HAT package for the Raspbian platform:
```python
# Build the HAT package
package.build(name="hello_matmul_pi3", format=acc.Package.Format.HAT_STATIC, platform=acc.Package.Platform.RASPBIAN)
```
By now, you should have all the code necessary to generate your Accera MatMul function that can be called on a Raspberry Pi 3 target. You can also find the complete Python script [here](cross_compilation_pi3/hello_matmul_pi3_generator.py).

### Generate HAT package

Next, we run the generator script to produce a HAT package for the Raspberry Pi 3 target.

#### Windows/MacOS

```shell
python hello_matmul_pi3_generator.py
```

#### Ubuntu

```shell
python3 hello_matmul_pi3_generator.py
```

After this runs, you should see a header file `hello_matmul_pi3.HAT` and an object file `hello_matmul_pi3.o` in the ELF format. The `.HAT` file format is described [here](https://github.com/microsoft/HAT). Together with the `obj`, this is part of a HAT package.


### Runner code

We will now walk through how to call our MatMul implementation from the HAT package on the Raspberry Pi 3.

Create a file called `hello_matmul_pi3_runner.cpp` with the code below. You can also find it [here](cross_compilation_pi3/hello_matmul_pi3_runner.cpp).

```cpp
#include <stdio.h>
#include <algorithm>

// Include the HAT file that declares our MatMul function
#include "hello_matmul_p3.HAT"

#define M 128
#define N 256
#define K 256

int main(int argc, const char** argv)
{
    // Prepare our matrices
    float A[M*K];
    float B[K*N];
    float C[M*N];

    // Fill with data
    std::fill_n(A, M*K, 2.0f);
    std::fill_n(B, K*N, 3.0f);
    std::fill_n(C, M*N, 0.42f);

    printf("Calling MatMul M=%d, K=%d, N=%d\n", M, K, N);
    hello_matmul_py(A, B, C);

    printf("Result (first few elements): ");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f ", C[i]);
    }
    printf("\n");
    return 0;
}
```

The code above creates the `A`, `B`, and `C` matrices, and calls the function `hello_matmul_pi3_py` to perform MatMul.

Now that we have written the code, we will compile and link it with the HAT package to create an executable. Save the file to your working directory, in the same location as `hello_matmul_pi3_generator.py` and the generated `*.HAT` and `*.o` files.

### Build and run

#### On the Raspberry Pi 3 device

For this step, you’ll be working with your Raspberry Pi device. If your Pi device is accessible over the network, copy `hello_matmul_pi3_runner.cpp`, `hello_matmul_pi3.HAT`, and `hello_matmul_pi3.o` using the Unix scp tool or the Windows WinSCP tool [here](https://winscp.net/eng/index.php)., otherwise use a USB thumb drive to transfer files manually. You do not need to copy the other generated files and folders.

You will need *gcc*, it is often installed by default on Raspberry Pi 3 systems, but to confirm type:

```shell
sudo apt-get install -y gcc
```

This has been verified with "Raspbian GNU/Linux 9 (stretch)" and gcc<4:6.3.0-4>, and should work with subsequent versions.
Then you can run the following commands to build and run.

```shell
gcc hello_matmul_pi3_runner.cpp hello_matmul_pi3.o -o hello_matmul_pi3_runner
./hello_matmul_pi3_runner
```

The output should look like:

```
Calling MatMul M=128, K=256, N=256
Result (first few elements): 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922
```

You can now experiment with the generated MatMul function with your own inputs. Simply modify `hello_matmul_pi3_runner.cpp` on the Raspberry Pi 3 and recompile with the existing HAT package.
