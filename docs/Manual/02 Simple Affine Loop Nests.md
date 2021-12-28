[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Section 2: Simple affine loop nests
In this section we introduce loop nests and define the types of loops nests that appear in the Accera programming model.

## Affine loop nests
Many important compute-intensive workloads can be expressed using nested for-loops. An algorithm that can be defined using nested for-loops is called a *loop nest*. Accera is restricted to the class of *affine loop nests*. A loop nest is *affine* if the indices of the elements accessed on each iteration are an affine function of the loop iterator variables. For example, the following loop nest is affine:
```python
for i in range(M):
    for j in range(N):
        C[2*i+2, j+2] += A[3*i, j] + B[j, i]
```
because `2*i+2`, `j+2`, `3*i`, `j` and `i` are all affine functions of the iterator variables `i` and `j`.

On the other hand, the following loop nest is not affine:
```python
for i in range(M):
    for j in range(N):
        C[i*i, j] += A[i*i, j] + B[i*j, i]
```
because `i*i` and `i*j` are quadratic (non-affine) functions of `i` and `j`.

## Simple affine loops nests, a.k.a. simple nests
An important subclass of affine loop nests is the class of  *simple affine loop nests*, or just *simple nests* for short. An affine loop nest is *simple* if it satisfies the following properties:
1. The loops are *perfectly nested*: all the computation is entirely contained within the deepest loop.
2. All the loops are *normalized*: each loop starts at 0, increments by 1, and ends at a compile-time constant size.
3. The loop iterations are *order invariant*: the logic doesn't change if the loop iterations are executed in a different sequential order.
4. *No conditional exit*: the loop doesn't contain *break* or *continue* commands

The matrix-matrix multiplication example given in the introduction is an example of a simple nest. Another example is *2-dimensional convolution*, which is the fundamental operation in convolutional neural networks, and can be written in Python as:
```python
# Convolve M x N data matrix A with S x T filter matrix B and add output to matrix C
for i in range(M):
    for j in range(N):
        for k in range(S):
            for l in range(T):
                C[i, j] += A[i + k, j + l] * B[k, l]
```

While Accera supports arbitrary affine loop nests, the programmer defines the logic of their algorithm using simple nests. More complex nests are obtained by applying schedule transformations (see [Section 3](03%20Schedules.md)) or by fusing multiple schedules (see [Section 4](04%20Fusing.md)).

## Defining the loop nest logic
The programmer's goal is to create a highly optimized target-specific implementation of an affine loop nest. The first step towards this goal is to define the logic of one or more simple nests. The logic is a target-independent pseudo-code of a simple nest, written without considering performance. For example, the following code defines the logic of the matrix-matrix multiplication loop nest:

```python
# Import accera
import accera as acc

# Define matrix sizes
M = 16
N = 10
S = 11

A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, S))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(S, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

# Define a simple affine loop nest and name its loops i, j, k
nest = acc.Nest(shape=(M, N, S))
i, j, k = nest.get_indices()

# Define the logic of each iteration in the nest
@nest.iteration_logic
def _():
    C[i,j] += A[i,k] * B[k,j]
```
We start by defining the arrays that participate in the computation: `A` and `B` are input arrays and `C` is an input/output array. Next, we initialize `nest` to be an empty skeleton of a loop nest, with nested loops of sizes `M`, `N`, `S`. These loops are logical -- think of them as pseudo-code loops -- they do not define the execution order of the iterations. The index variables that correspond to the three loops are named `i, j, k` respectively.

[comment]: # (* MISSING: the iteration spaces defined above have a compile-time shape. How do we handle run-time shapes? Can all the iteration space dimensions be runtime variables? )

The last part of the example sets the iteration logic to `C[i, j] += A[i, k] * B[k, j]`. Note that this iteration logic follows an affine memory access pattern. The syntax in the example makes use of Python decorators and is shorthand for the more explicit syntax:
```python
def logic_fn():
    C[i, j] += A[i, k] * B[k, j]

nest.iteration_logic(logic_fn)
```

## Supported operations
The iteration logic can include the following operations (assuming `accera` was imported as `rp`):

### Assignment operators


| Operation | Types (Operands must be of same type)  | Description  |
|----------|----------|--------------|
| `a = b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Assigns the value of scalar *b* to scalar *a* |

__Not yet implemented:__ unsigned types (`acc.ScalarType.uint8/16/32/64`)

### Arithmetic operators

| Operation | Types (Operands must be of same type)  | Description  |
|----------|----------|--------------|
| `a + b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns the sum of scalars *a* and *b* |
| `a - b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns the difference between scalars *a* and *b* |
| `a * b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns the product of scalars *a* and *b* |
| `a / b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns the quotient of scalars *a* and *b*. If the operands are integers, an integer division result is returned |
| `a ** b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns the *b*'th power of scalar *a* |
| `a // b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns the floor of the quotient of scalars *a* and *b* |
| `a % b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns the signed remainder after dividing scalar *a* by scalar *b* |
| `-a` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns the additive inverse of scalar *a* |

__Not yet implemented:__ unsigned types (`acc.ScalarType.uint8/16/32/64`)

Comment: Accera also supports the corresponding compound-assignment operators, such as `a += b`, `a -= b`, etc.

### Relational operators

| Operation | Types (Operands must be of same type) | Description  |
|----------|----------|--------------|
| `a == b` | `acc.ScalarType.bool, acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns True if scalar *a* equals scalar *b*, else False |
| `a != b` | `acc.ScalarType.bool, acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns True if scalar *a* is not equal to scalar *b*, else False |
| `a < b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns True if scalar *a* is strictly smaller than scalar *b*, else False |
| `a <= b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns True if scalar *a* is smaller than or equal to scalar *b*, else False |
| `a > b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns True if scalar *a* is strictly greater than scalar *b*, else False |
| `a >= b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns True if scalar *a* is greater than or equal to scalar *b*, else False |

__Not yet implemented:__ unsigned types (`acc.ScalarType.uint8/16/32/64`)

### Logical operators

| Operation  | Types (Operands must be of same type) | Description  |
|----------|----------|--------------|
| `acc.logical_and(a, b)` | `acc.ScalarType.bool, acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns True if scalars *a* and *b* are non-zero, else False |
| `acc.logical_or(a, b)` | `acc.ScalarType.bool, acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns True if either scalar *a* or scalar *b* are non-zero, else False |
| `acc.logical_not(a)` | `acc.ScalarType.bool, acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns True if *a* is zero, else False |

__Not yet implemented:__ unsigned types (`acc.ScalarType.uint8/16/32/64`)

### Bitwise operators

| Operation  | Types (Operands must be of same type) | Description  |
|----------|----------|--------------|
| `a & b` | `acc.ScalarType.int8/16/32/64` | Returns the bitwise AND of the bits in scalars *a* and *b* |
| `a \| b` | `acc.ScalarType.int8/16/32/64` | Returns the bitwise OR of the bits in scalars *a* and *b* |
| `a ^ b` | `acc.ScalarType.int8/16/32/64` | Returns the bitwise XOR of the bits in scalars *a* and *b* |
| `~a` | `acc.ScalarType.int8/16/32/64` | Returns the bitwise inverse of the bits in scalar *a* |
| `a << b` | `acc.ScalarType.int8/16/32/64` | Returns scalar *a* whose bitwise representation is shifted left by *b* bits |
| `a >> b` | `acc.ScalarType.int8/16/32/64` | Returns scalar *a* whose bitwise representation is shifted right by *b* bits |

Comment: Accera also supports the corresponding compound-assignment operators, such as `a &= b`, `a |= b`, etc.

__Not yet implemented:__ unsigned types (`acc.ScalarType.uint8/16/32/64`)

### Intrinsics

| Operation  | Types (Operands must be of same type) | Description  |
|----------|----------|--------------|
| `acc.abs(a)` | `acc.ScalarType.float32/64` | Returns the absolute value of scalar *a* |
| `acc.max(a, b)` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns the larger of the two scalars *a* and *b* |
| `acc.min(a, b)` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.float32/64` | Returns the smaller of the two scalars *a* and *b* |
| `acc.ceil(a)` | `acc.ScalarType.float32/64` | Returns the value of scalar *a* rounded up to the nearest integer as an int64 type |
| `acc.floor(a)` | `acc.ScalarType.float32/64` | Returns the value of scalar *a* rounded down to the nearest integer as an int64 type |
| `acc.sqrt(a)` | `acc.ScalarType.float32/64` | Returns the square root of scalar *a* |
| `acc.exp(a)` | `acc.ScalarType.float32/64` | Returns the exponential *e* raised to the scalar *a* |
| `acc.log(a)` | `acc.ScalarType.float32/64` | Returns the natural logarithm (base *e*) of scalar *a* |
| `acc.log10(a)` | `acc.ScalarType.float32/64` | Returns the common logarithm (base 10) of scalar *a* |
| `acc.log2(a)` | `acc.ScalarType.float32/64` | Returns the binary logarithm (base 2) of scalar *a* |
| `acc.sin(a)` | `acc.ScalarType.float32/64` | Returns the sine of scalar *a*, where *a* is in radians |
| `acc.cos(a)` | `acc.ScalarType.float32/64` | Returns the cosine of scalar *a*, where *a* is in radians |
| `acc.tan(a)` | `acc.ScalarType.float32/64` | Returns the tangent of scalar *a*, where *a* is in radians |
| `acc.sinh(a)` | `acc.ScalarType.float32/64` | Returns the hyperbolic sine of scalar *a*, where *a* is in radians |
| `acc.cosh(a)` | `acc.ScalarType.float32/64` | Returns the hyperbolic cosine of scalar *a*, where *a* is in radians |
| `acc.tanh(a)` | `acc.ScalarType.float32/64` | Returns the hyperbolic tangent of scalar *a*, where *a* is in radians |

__Not yet implemented:__ unsigned types (`acc.ScalarType.uint8/16/32/64`)

## Accera program stages
We take a step back to describe the stages of a Accera program:
* `Nest`: A nest captures the logic of a simple nest, without any optimizations or implementation details.
* `Schedule`: A `Nest` is used to create a schedule. The schedule controls the order in which the nest iterations are visited. Multiple schedules can be fused into a single schedule, which may no longer represent a simple nest.
* `ActionPlan`: A `Schedule` is used to create an action plan. An action plan controls the implementation details that are specific to a specific target platform (e.g., data caching strategy, vectorization, assignment of arrays and caches to different types of memory).
* `Package`: An `ActionPlan` is used to create a function in a function package. The package is then compiled and emitted.

Once a package is emitted, the Accera functions contained in it can be called from external client code. This external code is typically not written using Accera.

Accera currently supports the following package formats:
* [HAT](https://github.com/microsoft/hat), which is a schematized version of a standard C library. The external client code can be written in C or C++ and linked with the HAT package.
* [MLIR](https://mlir.llvm.org), which uses standard MLIR dialects. The external code must also be in MLIR.

Overall, to build and emit `nest` (defined above), we would write:

```python
# create a default schedule from the nest
schedule = nest.create_schedule()

# create a default action plan from the schedule
plan = schedule.create_action_plan()

# create a HAT package. Create a function in the package based on the action plan
package = acc.Package()
package.add_function(plan, args=(A, B, C), base_name="simple_matmul")

# build the HAT package
package.build(format=acc.Package.Format.HAT, name="linear_algebra")
```

It may not be immediately clear why so many stages are needed just to compile a simple nest. The importance of each step will hopefully become clear once we describe the stages in detail.

In the example above, The call to `package.add_function` takes three arguments: the first is the action plan that defines the function's implementation; the second is the order of the input and input/output arrays in the function signature; and the third is a base name for the function. The full name of the function is the base name followed by an automatically-generated unique identifier. For example, the function in the example could appear in the package as `simple_matmul_8f24bef5`. The automatically-generated suffix ensures that each function in the package has a unique name. More details on function packages can be found in [Section 10](10%20Packages.md).

## Convenience syntax
For convenience, Accera also provides shortcuts to avoid unneeded verbosity. Specifically, we can create a function in a package directly from a nest, as follows:
```python
package.add_function(nest, args=(A, B, C), base_name="simple_matmul")
```
The abbreviated syntax makes it seem like a callable function is generated directly from `nest`, but what actually happens behind the scenes is that `nest` creates a default schedule, which creates a default action plan, which is added as a function in the package. Accera has a similar convenience syntax to create a function from a schedule:
```python
package.add_function(schedule, args=(A, B, C), base_name="simple_matmul")
```
and to create an action plan directly from a nest:
```python
plan = nest.create_action_plan()
```


<div style="page-break-after: always;"></div>
