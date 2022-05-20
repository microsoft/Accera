[//]: # (Project: Accera)
[//]: # (Version: v1.2.4)

# Accera v1.2.4 Reference

## `accera.create_parameter_grid(parameter_choices, filter_func, sample, seed)`
Create a parameter grid from a dictionary that maps each parameter to its possible values.

## Arguments

argument | description | type/default
--- | --- | ---
`parameter_choices` | A dictionary that maps each parameter to its possible values | dictionary
`filter_func` | A callable to filter parameter_choices which returns a bool to indicate whether a given parameter combination should be included in the grid | Callable
`sample` | A number to limit the size of the parameter grid. The grid is randomly sampled. | integer
`seed` | The seed value for random sampling.  | integer

## Returns
List of dictionary

## Examples

Create a parameter grid from a dictionary that maps each parameter to its possible values:

```python
parameters = acc.create_parameter_grid(parameter_choices={P0:[8,16], P1:[16,32], P2:[16], P3:[1.0,2.0]})
package.add(nest, args=(A, B, C), base_name="matmul", parameters)
```

Define a lambda or function to filter out combinations from the parameter grid. The arguments to the filter are the values of a parameter combination. The filter function should return True if the combination should be included, and False otherwise:

```python
parameters = acc.create_parameter_grid(parameter_choices={P0:[8,16], P1:[16,32], P2:[16], P3:[1.0,2.0]}, filter_func=lambda p0, p1, p2, p3: p2 < p1 and 4 * (p0 * p3 + p1 * p2 + p1 * p3 + p2 * p3) / 1024 < 256)
```

Parameter grids can result in a large number of possible combinations. We can limit the number of combinations by random sampling:

```python
parameters = acc.create_parameter_grid(parameter_choices={P0:[8,16], P1:[16,32], P2:[16], P3:[1.0,2.0]}, sample=5)
```

<div style="page-break-after: always;"></div>


