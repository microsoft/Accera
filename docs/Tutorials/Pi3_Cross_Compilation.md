[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Cross Compiling for the Raspberry Pi 3

By the end of this tutorial, you will learn how to:

* Cross compile a simple Matrix Multiplication (MatMul) function for execution on a Raspberry Pi 3.
* Produce a [HAT](https://github.com/microsoft/hat) package containing the MatMul function that can be called on the Pi 3 target.
* Call the function on a Raspberry Pi 3 from C/C++ code.

## Prerequisites

* You should have Accera installed. If not, you can find the instructions in [here](../Install/README.md).
* Be familiar with writing Python and C++ code.
* Have access to a Raspberry Pi 3 device.

## A naive MatMul algorithm

Consider the example of multiplying matrices A and B and adding the result into matrix C. In NumPy syntax, this can be expressed as:

```
C += A @ B
```

A naive algorithm for matrix multiplication typically contains 3 nested for-loops. In Python, this can be expressed as:

```
# A.shape = (M, K), B.shape = (K, N), C.shape = (M, N)

for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i, j] += A[i, k] * B[k, j]
```


### Accera Python DSL

Let's walk through a naïve Matrix Multiplication (MatMul) using Accera. Instead of using the default target, i.e., the host machine, we specify a target representing a Raspberry Pi 3 to cross-compile the host for a different target.

Create an empty file called `hello_matmul_pi3_generator.py`. First, we import Accera's module:

```python
import accera as acc
```

Define some matrix sizes, where A's shape is M by K, B's is K by N, and C's, M by N. 

```python
# Define our matrix sizes
M = 128
N = 256
K = 256
```

Write a Python function that receives `A`, `B`, and `C` arrays. These are our input and input/output matrices.

```python
A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, K))
B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(K, N))
C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))
```

We now use the `Nest` class to define our 3-layered nested for-loop. The range indices are `M`, `N`, and `K`, with the outermost loop (`M`) listed first. We can get the loop nest indices to perform the computation.

```python
# Define the loop nest
nest = acc.Nest(shape=(M, N, K))

# Get the loop nest indices
i, j, k = nest.get_indices()
```

Next, we define the logic for every iteration of the loop nest:
```python
# Define the loop nest logic
@nest.iteration_logic
def _():
    C[i, j] += A[i, k] * B[k, j]
```

We have finished defining the logic of MatMul. Let's now define the schedule which controls the execution of logic. For this, we first create the schedule from the nest:

```python
sched = nest.create_schedule()
```

At this point, `sched` represents the default schedule for our algorithm. We can also perform some basic transformations on this schedule. For example, the following lines of code split the `k` index into blocks of 4 ( `k`, `k+4`, `k+8`, and so on).

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
After following the above steps, you should now have all the code necessary to generate your Accera MatMul function that can be called on a Raspberry Pi 3 target. You can find the complete Python script [here](cross_compilation_pi3/hello_matmul_pi3_generator.py).

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

After we run the script, there should be a header file `hello_matmul_pi3.hat` and an object file `hello_matmul_pi3.o` in the ELF format. The `.hat` file format is described [here](https://github.com/microsoft/HAT). Collectively, we call the `.hat file` and `object file` a "HAT package".


### Runner code

Let's now see how we can call our MatMul implementation from the HAT package on the Raspberry Pi 3.

Create a file called `hello_matmul_pi3_runner.cpp` with the code below. You can find it [here](cross_compilation_pi3/hello_matmul_pi3_runner.cpp).

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

The above code creates the `A`, `B`, and `C` matrices and calls the function `hello_matmul_pi3_py` to perform MatMul.

Now that we have written the code, we compile and link it with the HAT package to create an executable file. Save this file to your working directory, in the exact location as `hello_matmul_pi3_generator.py,` and the generated `*.hat` and `*.o` files.


### Build and run

#### On the Raspberry Pi 3 device

For this step, you'll be working with your Raspberry Pi device. If your Pi device is accessible over the network, copy `hello_matmul_pi3_runner.cpp`, `hello_matmul_pi3.hat`, and `hello_matmul_pi3.o` using the Unix scp tool or the Windows WinSCP tool [here](https://winscp.net/eng/index.php)., otherwise use a USB thumb drive to transfer files manually. You do not need to copy the other generated files and folders.

You also need *gcc*. Although it is often installed by default on Raspberry Pi 3 systems, type this for confirmation:

```shell
sudo apt-get install -y gcc
```

This has been verified with "Raspbian GNU/Linux 9 (stretch)" and gcc<4:6.3.0-4> and should work with subsequent versions.
Now, you can run the following commands to build and run.

```shell
gcc hello_matmul_pi3_runner.cpp hello_matmul_pi3.o -o hello_matmul_pi3_runner
./hello_matmul_pi3_runner
```

The output should look like:

```
Calling MatMul M=128, K=256, N=256
Result (first few elements): 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922
```

You can now experiment with the generated MatMul function with your own inputs. To try different inputs, you can modify `hello_matmul_pi3_runner.cpp` on the Raspberry Pi 3 and recompile it with the existing HAT package.
