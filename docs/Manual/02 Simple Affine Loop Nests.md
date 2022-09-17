[//]: # (Project: Accera)
[//]: # (Version: v1.2.9)

# Section 2: Simple affine loop nests
This section introduces *loop nests* and their different types that are provided in Accera programming model.

## Affine loop nests
Many important compute-intensive workloads can be expressed using nested for-loops. An algorithm that can be defined using nested for-loops is called a *loop nest*. Accera only supports the class of *affine loop nests*. A loop nest is *affine* if the indices of the elements accessed on each iteration are an affine function of the loop iterator variables. For example, the following loop nest is affine:
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
*Simple Affine Loop Nests*, hereinafter referred to as *simple nests*, is an important subclass of affine loop nests that satisfies the following properties:
1. The loops are *perfectly nested*: all the computation is entirely contained within the deepest loop.
2. All the loops are *normalized*: each loop starts at 0, increments by 1, and ends at a compile-time constant size.
3. The loop iterations are *order invariant*: the logic doesn't change if the loop iterations are executed in a different sequential order.
4. *No conditional exit*: the loop doesn't contain *break* or *continue* commands.


The matrix-matrix multiplication example given in the introduction is an example of a simple nest. Another example is *2-dimensional convolution*, which is the fundamental operation in convolutional neural networks, and can be written in Python as:
```python
# Convolve M x N data matrix A with S x T filter matrix B and add output to matrix C
for i in range(M):
    for j in range(N):
        for k in range(S):
            for l in range(T):
                C[i, j] += A[i + k, j + l] * B[k, l]
```

While Accera supports arbitrary affine loop nests, the programmer defines the logic of their algorithms using simple nests. More complex nests are obtained by applying schedule transformations (see [Section 3](<03%20Schedules.md>)) or by fusing multiple schedules (see [Section 4](<04%20Fusing.md>)).

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

The last part of the example sets the iteration logic to `C[i, j] += A[i, k] * B[k, j]`. Note that this iteration logic follows an affine memory access pattern. The syntax in the example makes use of Python decorators and is shorthand for the more explicit syntax:
```python
def logic_fn():
    C[i, j] += A[i, k] * B[k, j]

nest.iteration_logic(logic_fn)
```

The iteration spaces above have compile-time shapes. We can define runtime shapes by replacing any or all of the constant matrix sizes `M`, `N`, and `S` with an `acc.Dimension` placeholder:

```python

M = acc.create_dimensions() # replace M with a runtime dimension
N = 10 # a compile-time dimension
S = 11

A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, S))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(S, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

# Define a simple affine loop nest and name its loops i, j, k
nest = acc.Nest(shape=(M, N, S))

```

The iteration space dimensions will now be runtime variables that need to be provided to the function (more on this later).

## Supported operations
The iteration logic can include the following operations (assuming `accera` was imported as `acc`):

### Assignment operators


| Operation | Types (Operands must be of same type)  | Description  |
|----------|----------|--------------|
| `a = b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Assigns the value of scalar *b* to scalar *a* |

### Arithmetic operators

| Operation | Types (Operands must be of same type)  | Description  |
|----------|----------|--------------|
| `a + b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the sum of scalars *a* and *b* |
| `a - b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the difference between scalars *a* and *b* |
| `a * b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the product of scalars *a* and *b* |
| `a / b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the quotient of scalars *a* and *b*. If the operands are integers, an integer division result is returned |
| `a ** b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the *b*'th power of scalar *a* |
| `a // b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the floor of the quotient of scalars *a* and *b* |
| `a % b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the signed remainder after dividing scalar *a* by scalar *b* |
| `-a` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the additive inverse of scalar *a* |

Comment: Accera also supports the corresponding compound-assignment operators, such as `a += b`, `a -= b`, etc.

### Relational operators

| Operation | Types (Operands must be of same type) | Description  |
|----------|----------|--------------|
| `a == b` | `acc.ScalarType.bool, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if scalar *a* equals scalar *b*, else False |
| `a != b` | `acc.ScalarType.bool, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if scalar *a* is not equal to scalar *b*, else False |
| `a < b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if scalar *a* is strictly smaller than scalar *b*, else False |
| `a <= b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if scalar *a* is smaller than or equal to scalar *b*, else False |
| `a > b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if scalar *a* is strictly greater than scalar *b*, else False |
| `a >= b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if scalar *a* is greater than or equal to scalar *b*, else False |

### Logical operators

| Operation  | Types (Operands must be of same type) | Description  |
|----------|----------|--------------|
| `acc.logical_and(a, b)` | `acc.ScalarType.bool, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if scalars *a* and *b* are non-zero, else False |
| `acc.logical_or(a, b)` | `acc.ScalarType.bool, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if either scalar *a* or scalar *b* are non-zero, else False |
| `acc.logical_not(a)` | `acc.ScalarType.bool, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if *a* is zero, else False |

### Bitwise operators

| Operation  | Types (Operands must be of same type) | Description  |
|----------|----------|--------------|
| `a & b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns the bitwise AND of the bits in scalars *a* and *b* |
| `a \| b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns the bitwise OR of the bits in scalars *a* and *b* |
| `a ^ b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns the bitwise XOR of the bits in scalars *a* and *b* |
| `~a` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns the bitwise inverse of the bits in scalar *a* |
| `a << b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns scalar *a* whose bitwise representation is shifted left by *b* bits |
| `a >> b` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns scalar *a* whose bitwise representation is shifted right by *b* bits |

Comment: Accera also supports the corresponding compound-assignment operators, such as `a &= b`, `a |= b`, etc.

### Intrinsics

| Operation  | Types (Operands must be of same type) | Description  |
|----------|----------|--------------|
| `acc.abs(a)` | `acc.ScalarType.float16/32/64` | Returns the absolute value of scalar *a* |
| `acc.max(a, b)` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the larger of the two scalars *a* and *b* |
| `acc.min(a, b)` | `acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the smaller of the two scalars *a* and *b* |
| `acc.ceil(a)` | `acc.ScalarType.float16/32/64` | Returns the value of scalar *a* rounded up to the nearest integer as an int64 type |
| `acc.floor(a)` | `acc.ScalarType.float16/32/64` | Returns the value of scalar *a* rounded down to the nearest integer as an int64 type |
| `acc.sqrt(a)` | `acc.ScalarType.float16/32/64` | Returns the square root of scalar *a* |
| `acc.exp(a)` | `acc.ScalarType.float16/32/64` | Returns the exponential *e* raised to the scalar *a* |
| `acc.log(a)` | `acc.ScalarType.float16/32/64` | Returns the natural logarithm (base *e*) of scalar *a* |
| `acc.log10(a)` | `acc.ScalarType.float16/32/64` | Returns the common logarithm (base 10) of scalar *a* |
| `acc.log2(a)` | `acc.ScalarType.float16/32/64` | Returns the binary logarithm (base 2) of scalar *a* |
| `acc.sin(a)` | `acc.ScalarType.float16/32/64` | Returns the sine of scalar *a*, where *a* is in radians |
| `acc.cos(a)` | `acc.ScalarType.float16/32/64` | Returns the cosine of scalar *a*, where *a* is in radians |
| `acc.tan(a)` | `acc.ScalarType.float16/32/64` | Returns the tangent of scalar *a*, where *a* is in radians |
| `acc.sinh(a)` | `acc.ScalarType.float16/32/64` | Returns the hyperbolic sine of scalar *a*, where *a* is in radians |
| `acc.cosh(a)` | `acc.ScalarType.float16/32/64` | Returns the hyperbolic cosine of scalar *a*, where *a* is in radians |
| `acc.tanh(a)` | `acc.ScalarType.float16/32/64` | Returns the hyperbolic tangent of scalar *a*, where *a* is in radians |

### Implicit type casting

Accera operators require operands to be the same type. Computations that use multiple types can take advantage of Accera's implicit type casting support when converting from smaller-sized types to larger-sized types.

To do implicit casting, simply assign a source type to its implicitly-castable destination type. No additional casting operation is needed for converting between these types.

| Source types | Destination type (implicitly-castable) |
| ------------ | -------------------------------------- |
| `acc.ScalarType.bool`, `acc.ScalarType.uint8` | `acc.ScalarType.int8` |
| `acc.ScalarType.bool`,  `acc.ScalarType.int8` | `acc.ScalarType.uint8` |
| `acc.ScalarType.bool`, `acc.ScalarType.int8`, `acc.ScalarType.uint8`, `acc.ScalarType.uint16` | `acc.ScalarType.int16` |
| `acc.ScalarType.bool`, `acc.ScalarType.int8`, `acc.ScalarType.uint8`, `acc.ScalarType.int16` | `acc.ScalarType.uint16` |
| `acc.ScalarType.bool`, `acc.ScalarType.int8`, `acc.ScalarType.uint8`, `acc.ScalarType.int16`, `acc.ScalarType.uint16`, `acc.ScalarType.uint32` | `acc.ScalarType.int32` |
| `acc.ScalarType.bool`, `acc.ScalarType.int8`, `acc.ScalarType.uint8`, `acc.ScalarType.int16`, `acc.ScalarType.uint16`, `acc.ScalarType.int32` | `acc.ScalarType.uint32` |
| `acc.ScalarType.bool`, `acc.ScalarType.int8`, `acc.ScalarType.uint8`, `acc.ScalarType.int16`, `acc.ScalarType.uint16`, `acc.ScalarType.int32`, `acc.ScalarType.uint32`, `acc.ScalarType.uint64` | `acc.ScalarType.int64` |
| `acc.ScalarType.bool`, `acc.ScalarType.int8`, `acc.ScalarType.uint8`, `acc.ScalarType.int16`, `acc.ScalarType.uint16`, `acc.ScalarType.int32`, `acc.ScalarType.uint32`, `acc.ScalarType.int64` | `acc.ScalarType.uint64` |
| `acc.ScalarType.bool`, `acc.ScalarType.int8`, `acc.ScalarType.uint8`, `acc.ScalarType.int16`, `acc.ScalarType.uint16` | `acc.ScalarType.float16` |
| `acc.ScalarType.bool`, `acc.ScalarType.int8`, `acc.ScalarType.uint8`, `acc.ScalarType.int16`, `acc.ScalarType.uint16` | `acc.ScalarType.bfloat16` |
| `acc.ScalarType.bool`, `acc.ScalarType.int8`, `acc.ScalarType.uint8`, `acc.ScalarType.int16`, `acc.ScalarType.uint16`, `acc.ScalarType.int32`, `acc.ScalarType.uint32`, `acc.ScalarType.int64`, `acc.ScalarType.float16`, `acc.ScalarType.bfloat16` | `acc.ScalarType.float32` |
| `acc.ScalarType.bool`, `acc.ScalarType.int8`, `acc.ScalarType.uint8`, `acc.ScalarType.int16`, `acc.ScalarType.uint16`, `acc.ScalarType.int32`, `acc.ScalarType.uint32`, `acc.ScalarType.int64`, `acc.ScalarType.float16`, `acc.ScalarType.bfloat16`, `acc.ScalarType.float32` | `acc.ScalarType.float64` | 

[comment]: # (bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64 | index)

To override the casting behavior above, or cast a larger-sized type to a smaller-sized type, use the `acc.cast` operation.

Comment: implicit casting of constants may result in truncation.

[comment]: # (MISSING: examples for constant implicit casting that cause unexpected truncation)

## Accera program stages
Let’s take a step back to describe the stages of Accera program:

* `Nest`: A nest captures the logic of a simple nest, without any optimizations or implementation details.
* `Schedule`: A `Nest` is used to create a schedule. The schedule controls the order in which the nest iterations are visited. Multiple schedules can be fused into a single schedule, which may no longer represent a simple nest.
* `Plan`: A `Schedule` is used to create a plan. A plan controls the implementation details that are specific for a target platform (e.g., data caching strategy, vectorization, assignment of arrays and caches to different types of memory).
* `Package`: A `Plan` is used to create a function in a function package. The package is then compiled and emitted.

Once a package is emitted, the Accera functions contained in it can be called from external client code. This external code is typically not written using Accera.

Accera currently supports the following package formats:

* [HAT](https://github.com/microsoft/hat), which is a schematized version of a standard C library. The external client code can be written in C or C++ and linked with the HAT package.
* [MLIR](https://mlir.llvm.org), which uses standard MLIR dialects. The external code must also be in MLIR.

Overall, to build and emit `nest` (defined above), we would write:

```python
# create a default schedule from the nest
schedule = nest.create_schedule()

# create a default plan from the schedule
plan = schedule.create_plan()

# create a HAT package. Create a function in the package based on the plan
package = acc.Package()
package.add(plan, args=(A, B, C), base_name="simple_matmul")

# build the HAT package
package.build(format=acc.Package.Format.HAT_DYNAMIC, name="linear_algebra")
```

It may not be immediately clear why so many stages are needed just to compile a simple nest. Therefore, let’s discuss each stage in detail to understand their importance. 

In the example above, the call to `package.add` takes three arguments: the first is the plan that defines the function's implementation; the second is the order of the input and input/output arrays in the function signature; and the third is a base name for the function. The full name of the function is the base name followed by an automatically-generated unique identifier. For example, the function in the example could appear in the package as `simple_matmul_8f24bef5`. The automatically-generated suffix ensures that each function in the package has a unique name. More details on function packages can be found in [Section 10](<10%20Packages.md>).

The Array shapes above are known at compile-time. If one or all of the shapes are known at runtime, we provide dimensions as arguments to the function:

```python
M, N, S = acc.create_dimensions() # runtime dimensions

A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, S))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(S, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

...

# create a default schedule from the nest
schedule = nest.create_schedule()

# create a default plan from the schedule
plan = schedule.create_plan()

# create a HAT package. Create a function in the package based on the plan, with
# the dimensions as additional arguments (in any order)
package = acc.Package()
package.add(plan, args=(M, N, S, A, B, C), base_name="simple_matmul_runtime_shapes")

# build the HAT package
package.build(format=acc.Package.Format.HAT_DYNAMIC, name="linear_algebra")
```


## Convenience syntax
For convenience, Accera also provides shortcuts to avoid unnecessary verbosity. Specifically, we can create a function in a package directly from a nest, as follows:
```python
package.add(nest, args=(A, B, C), base_name="simple_matmul")
```
The abbreviated syntax makes it seem like a callable function is generated directly from `nest`. However, what actually happens behind the scenes is that `nest` creates a default schedule, which creates a default plan, which is added as a function in the package. Accera has a similar convenience syntax to create a function from a schedule:
```python
package.add(schedule, args=(A, B, C), base_name="simple_matmul")
```
and to create a plan directly from a nest:
```python
plan = nest.create_plan()
```


<div style="page-break-after: always;"></div>
