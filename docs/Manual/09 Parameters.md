[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Section 9: Parameters

Accera parameters are placeholders that get replaced with concrete values when we add a function to a package. A parameter can be used in a `Nest`, a `Schedule`, or an `ActionPlan`.

## Parameterized nests
Recall that a `Nest` represents the loop-nest logic. We can parameterize the nest shape and the iteration logic. For example, consider the following parameterized version of matrix multiplication:

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
package.add_function(nest, args=(A, B, C), parameters={P0:16, P1:16, P2:16, P3:1.0}, base_name="matmul_16_16_16_1")
package.add_function(nest, args=(A, B, C), parameters={P0:32, P1:32, P2:32, P3:2.0}, base_name="matmul_32_32_32_2")
```
In this example, the shape of the nest is parameterized by (`P0`, `P1`, `P2`) and its iteration logic includes the parameter `P3`. The nest is used twice, with different settings of these parameters, to create two separate functions in the package.

## Parameterized schedules and action plans
Parameters can also appear in schedules and action plans. For example, we could add the following to the code above:
```python
P4, P5 = acc.create_parameters(2)

# Create a parameterized schedule
schedule = nest.create_schedule()
ii = schedule.split(i, size=P4)

# Create a parameterized action plan
plan = schedule.create_action_plan()
plan.cache(A, level=P5)

# Add another function to the package
package.add_function(plan, args=(A, B, C), parameters={P0:16, P1:16, P2:16, P3:1.0, P4:4, P5:2}, base_name="alternative_matmul_16_16_16")
```

## Tuple parameter values
Parameters can be used as placeholders for tuples, and specifically tuples of indices. For example, say that we wanted to parameterize the order of the iteration space dimensions. We could write:
```python
P6 = acc.create_parameters(1)
schedule.reorder(order=P6)
```
Later, we could set the value of `P6` to the index tuple `(j,k,i)`.

## Get parameters from an entire parameter grid
Consider the parameterized nest defined above. Rather than setting each parameter to one specific value, imagine that we had a set of different values for each parameter. For example, say that we wanted `P0` to take a value in the set `{8, 16}`, `P1` in `{16, 32}`, `P2` would always equal 16, and `P3` would take a value in `{1,2}`. This list of all valid parameter combinations is called a *parameter grid*. In this case, the grid includes the following parameter settings:
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

Accera provides an easy way to add all of the functions that correspond to the parameter grid at once.
```python
parameters = get_parameters_from_grid(parameter_grid={P0:[8,16], P1:[16,32], P2:[16], P3:[1.0,2.0]})
package.add_functions(nest, args=(A, B, C), base_name="matmul", parameters)
```
add_functions simply calls `package.add_function` eight times, once for each parameter combination in the grid. Instead of `nest`, this function could take a schedule or an action plan. All eight functions share the same base name, and Accera automatically adds a unique suffix to each function name to prevent duplicates.
This pattern allows you to optionally perform filtering by inspecting the list of generated parameter values before calling add_functions.


<div style="page-break-after: always;"></div>
