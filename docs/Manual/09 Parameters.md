[//]: # (Project: Accera)
[//]: # (Version: v1.2.1)

# Section 9: Parameters

Accera's parameters are placeholders that get replaced with concrete values when adding a function to a package. A parameter can be used in a `Nest`, a `Schedule`, or a `Plan`.

## Parameterized nests
Recall that a `Nest` represents the loop-nest logic. We can parameterize the nest's shape and iteration logic. For example, consider the following parameterized version of matrix multiplication:

```python
# Create parameters
P0, P1, P2, P3 = acc.create_parameters(4)

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
P4, P5 = acc.create_parameters(2)

# Create a parameterized schedule
schedule = nest.create_schedule()
ii = schedule.split(i, size=P4)

# Create a parameterized plan
plan = schedule.create_plan()
plan.cache(A, level=P5)

# Add another function to the package
package.add(plan, args=(A, B, C), parameters={P0:16, P1:16, P2:16, P3:1.0, P4:4, P5:2}, base_name="alternative_matmul_16_16_16")
```

## Tuple parameter values
Parameters can be used as placeholders for tuples, specifically for tuples of indices. For example, assume that we want to parameterize the order of the iteration space dimensions. We can then write:
```python
P6 = acc.create_parameters(1)
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
parameters = create_parameter_grid(parameter_choices={P0:[8,16], P1:[16,32], P2:[16], P3:[1.0,2.0]})
package.add(nest, args=(A, B, C), base_name="matmul", parameters)
```
In this case, `package.add` generates a function eight times, once for each parameter combination in the grid.  Other than `nest`, `package.add` can alternatively accept a `Schedule` (if we are performing schedule transformations), or a `Plan` (if we are setting target-specific options). All eight functions share the same base name. However, Accera automatically adds a unique suffix to each function name to prevent duplication. This pattern allows optional filtering by inspecting the generated parameter values list before calling `package.add`.

You can define a lambda or function to filter out combinations from the parameter grid. The arguments to the filter are the values of a parameter combination, and it should return True if the combination should be included, and False otherwise:
```python
parameters = create_parameter_grid(parameter_choices={P0:[8,16], P1:[16,32], P2:[16], P3:[1.0,2.0]}, filter_func=lambda p0, p1, p2, p3: p2 < p1 and 4 * (p0 * p3 + p1 * p2 + p1 * p3 + p2 * p3) / 1024 < 256)
```

You can limit the size of the parameter grid and therefore the number of functions generated:
```python
parameters = create_parameter_grid(parameter_choices={P0:[8,16], P1:[16,32], P2:[16], P3:[1.0,2.0]}, sample=5)
```
<div style="page-break-after: always;"></div>
