[//]: # (Project: Accera)
[//]: # (Version: v1.2.8)

# Accera v1.2.8 Reference

## `accera.Package.add(source, args[, base_name, parameters])`
Adds one or more functions to the package.

## Arguments

argument | description | type
--- | --- | ---
`source` | The source which defines the function's implementation. | `Nest` or `Schedule` or `Plan`
`args` | The order of external-scope arrays, scalars, and dimensions used in the function signature. | tuple of `Array`, `Scalar`, or `Dim`
`base_name` | A base name for the function. The full name for the function will be the base name followed by an automatically-generated unique identifier. | string
`parameters` | A value for each parameter if the function's implementation is parameterized. See [Parameters](<../../../Manual/09%20Parameters.md>). A list of dictionaries can also be provided, in which case, multiple functions are generated.| `Parameter` to value dictionary or a list of `Parameter` to value dictionaries.

## Examples

Adding a function defined by an `Plan`:

```python
package.add(plan, args=(A, B, C), base_name="simple_matmul")
```

Convenience syntax to add a function defined by a `Schedule`. A default `Plan` will be created automatically:

```python
package.add(schedule, args=(A, B, C), base_name="simple_matmul")
```

Convenience syntax to add a function defined by a `Nest`. A default `Schedule` and `Plan` will be created internally:

```python
package.add(nest, args=(A, B, C), base_name="simple_matmul")
```

Adding a function with concrete values specified for its parameters (`P0`, `P1`, `P2`, `P3`).

```python
package.add(nest, args=(A, B, C), parameters={P0:16, P1:16, P2:16, P3:1}, base_name="matmul_16_16_16_1")
```

Adding a function with runtime dimension sizes `M`, `N`, `K` and arrays `A`, `B`, and `C`:

```python
package.add(nest, args=(M, N, K, A, B, C), base_name="matmul_M_N_K")
```


<div style="page-break-after: always;"></div>


