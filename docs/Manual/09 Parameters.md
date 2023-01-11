[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Section 9: Parameters

Accera's parameters are placeholders that get replaced with concrete values when adding a function to a package. A parameter can be used in a `Nest`, a `Schedule`, or a `Plan`.

## Parameterized nests
Recall that a `Nest` represents the loop-nest logic. We can parameterize the nest's shape and iteration logic. For example, consider the following parameterized version of matrix multiplication:

```python
# Create parameters
P0, P1, P2, P3 = acc.create_parameters()

A = acc.Array(role=acc.Array.Role.INPUT, shape=(P0, P2))
B = acc.Array(role=acc.Array.Role.INPUT, shape=(P2, P1))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(P0, P1))

# Define a simple nest
nest = acc.Nest(shape=(P0, P1, P2))
i, j, k = nest.get_indices()

# Define the loop nest logic and add it to the nest
@nest.iteration_logic
def _():
    C[i, j] += P3 * A[i, k] * B[k, j]

# create a package
package = acc.Package()

# Use the templated nest to add two different functions to the package
package.add(nest, args=(A, B, C), parameters={P0:16, P1:16, P2:16, P3:1.0}, base_name="matmul_16_16_16_1")
package.add(nest, args=(A, B, C), parameters={P0:32, P1:32, P2:32, P3:2.0}, base_name="matmul_32_32_32_2")
```
In the above scenario, the shape of the nest is parameterized by (`P0`, `P1`, `P2`) and its iteration logic includes the parameter `P3`. The nest is used twice with different settings of these parameters to create two separate functions in the package.

## Parameterized schedules and plans
Parameters can also appear in schedules and plans. For example, we can add the following code snippet:

```python
P4, P5 = acc.create_parameters()

# Create a parameterized schedule
schedule = nest.create_schedule()
ii = schedule.split(i, size=P4)

# Create a parameterized plan
plan = schedule.create_plan()
plan.cache(A, level=P5)

# Add another function to the package
package.add(plan, args=(A, B, C), parameters={P0:16, P1:16, P2:16, P3:1.0, P4:4, P5:2}, base_name="alternative_matmul_16_16_16")
```

## Supported operations
Accera's parameters support the basic arithmetic operations and other relational/bitwise/intrinsics operations. For example, we can add the following code snippet instead:

```python
fma_unit_count, vector_size, P5 = acc.create_parameters()

# Create a parameterized schedule
schedule = nest.create_schedule()
ii = schedule.split(i, size=fma_unit_count * vector_size)
iii = schedule.split(ii, size=vector_size)

# Create a parameterized plan
plan = schedule.create_plan()
plan.cache(A, level=P5)

# Add another function to the package
package.add(plan, args=(A, B, C), parameters={P0:16, P1:16, P2:16, P3:1.0, fma_unit_count:4, vector_size:16, P5:2}, base_name="alternative_matmul_16_16_16")
```

The supported operations include the following operations:


### Arithmetic operators

| Operation | Types | Description  |
|----------|----------|--------------|
| `a + b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the sum of parameters (or parameter and scalar) *a* and *b* |
| `a - b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the difference between parameters (or parameter and scalar) *a* and *b* |
| `a * b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the product of parameters (or parameter and scalar) *a* and *b* |
| `a / b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the quotient of parameters (or parameter and scalar) *a* and *b* |
| `a ** b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the *b*'th power of parameter *a* |
| `a // b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the floor of the quotient of parameters (or parameter and scalar) *a* and *b* |
| `a % b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the signed remainder after dividing parameter *a* by parameter or scalar *b* |
| `-a` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns the additive inverse of parameter *a* |


### Comparison Operations

| Operation | Types | Description  |
|----------|----------|--------------|
| `a == b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if parameter or scalar *a* equals parameter or scalar *b*, else False |
| `a != b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if parameter or scalar *a* is not equal to parameter or scalar *b*, else False |
| `a < b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if parameter or scalar *a* is strictly smaller than parameter or scalar *b*, else False |
| `a <= b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if parameter or scalar *a* is smaller than or equal to parameter or scalar *b*, else False |
| `a > b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if parameter or scalar *a* is strictly greater than parameter or scalar *b*, else False |
| `a >= b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64, acc.ScalarType.float16/32/64` | Returns True if parameter or scalar *a* is greater than or equal to parameter or scalar *b*, else False |

### Bitwise operators

| Operation  | Types | Description  |
|----------|----------|--------------|
| `a & b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns the bitwise AND of the bits in parameters (or parameter and scalar) *a* and *b* |
| `a \| b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns the bitwise OR of the bits in parameters (or parameter and scalar) *a* and *b* |
| `a ^ b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns the bitwise XOR of the bits in parameters (or parameter and scalar) *a* and *b* |
| `~a` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns the bitwise inverse of the bits in parameter *a* |
| `a << b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns parameter *a* whose bitwise representation is shifted left by *b* bits |
| `a >> b` | `acc.DelayedParameter, acc.ScalarType.int8/16/32/64, acc.ScalarType.uint8/16/32/64` | Returns parameter *a* whose bitwise representation is shifted right by *b* bits |


### Intrinsics

| Operation  | Types | Description  |
|----------|----------|--------------|
| `acc.abs(a)` | `acc.ScalarType.float16/32/64` | Returns the absolute value of parameter *a* |


## Tuple parameter values
Parameters can be used as placeholders for tuples, specifically for tuples of indices. For example, assume that we want to parameterize the order of the iteration space dimensions. We can then write:
```python
P6 = acc.create_parameters()
schedule.reorder(order=P6)
```
Later, we can set the value of `P6` to the index tuple `(j,k,i)`.

## Create parameters from an entire parameter grid
Consider the parameterized nest defined above. Rather than setting a specific value for each parameter, imagine that we have a set of different values for each parameter. For example, consider that we want `P0` to have a value in set `{8, 16}`, `P1` in `{16, 32}`, `P2` to be always `16`, and `P3` in `{1,2}`. We can define the *parameter grid* with this data, which lists all the valid parameter combinations. In our case, this grid includes the following parameter settings:
```python
{P0:8, P1:16, P2:16, P3:1.0}
{P0:8, P1:16, P2:16, P3:2.0}
{P0:8, P1:32, P2:16, P3:1.0}
{P0:8, P1:32, P2:16, P3:2.0}
{P0:16, P1:16, P2:16, P3:1.0}
{P0:16, P1:16, P2:16, P3:2.0}
{P0:16, P1:32, P2:16, P3:1.0}
{P0:16, P1:32, P2:16, P3:2.0}
```

Accera provides an easy way to add all the functions that correspond to the parameter grid at once:
```python
parameters = create_parameter_grid(parameter_choices={P0:[8,16], P1:[16,32], P2:[16], P3:[1.0,2.0])
package.add(nest, args=(A, B, C), base_name="matmul", parameters)
```
In this case, `package.add` generates a function eight times, once for each parameter combination in the grid.  Other than `nest`, `package.add` can alternatively accept a `Schedule` (if we are performing schedule transformations), or a `Plan` (if we are setting target-specific options). All eight functions share the same base name. However, Accera automatically adds a unique suffix to each function name to prevent duplication. This pattern allows optional filtering by inspecting the generated parameter values list before calling `package.add`.

You can define a lambda or function to filter out combinations from the parameter grid. The arguments to the filter are the values of a parameter combination, and it should return `True` if the combination should be included, and `False` otherwise:
```python
parameters = create_parameter_grid(parameter_choices={P0:[8,16], P1:[16,32], P2:[16], P3:[1.0,2.0]}, filter_func=lambda p0, p1, p2, p3: p2 < p1 and 4 * (p0 * p3 + p1 * p2 + p1 * p3 + p2 * p3) / 1024 < 256)
```

To limit the size of the parameter grid (and therefore the number of functions generated) to at most 5:
```python
parameters = create_parameter_grid(parameter_choices={P0:[8,16], P1:[16,32], P2:[16], P3:[1.0,2.0]}, sample=5)
```

If the parameter is a loop order which is a list or tuple of indices, `create_parameter_grid` can generate all the permutations of loop order. Furthermore, you can pass in a filter function to filter out invalid loop orders:
```python
parameters = create_parameter_grid({P0:(i, j, k, ii, jj, kk)}, filter_func = lambda *p : schedule.is_valid_loop_order(p[0][0]))

```

`Schedule.is_valid_loop_order()` is a pre-defined filter function that determines if a given loop order is valid for that schedule.

Note that the order of the list or tuple of indices provided to `create_parameter_grid` does not matter.

To filter parameters with more complicated logic, you can define your own filter function that wraps `Schedule.is_valid_loop_order()`:

```python
def my_filter(parameters_choice):
    P1, P2, P3, P4, P5, loop_order = parameters_choice

    return P1 > P2 \
        and P3 > P4 \
        and P1 * P5 < P3 \
        and P2 * P5 < P4 \
        and schedule.is_valid_loop_order(loop_order)

 parameters = acc.create_parameter_grid({
        P1: [64, 128, 256],
        P2: [32, 128], 
        P3: [16, 32, 128],
        P4: [8, 64],
        P5: [4],
        loop_order: (i, j, k, ii, jj, kk)
    }, my_filter)

```

<div style="page-break-after: always;"></div>
