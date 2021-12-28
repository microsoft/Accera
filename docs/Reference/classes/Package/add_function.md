[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Accera 1.2.0 Reference

## `accera.Package.add_function(source, args[, base_name, parameters])`
Adds a function to the package.

## Arguments
argument | description | type/default
--- | --- | ---
`source` | The source which defines the function's implementation. | `Nest` or `Schedule` or `ActionPlan`
`args` | The order of external-scope arrays to use in the function signature. | tuple of `Array`
`base_name` | A base name for the function. The full name for the function will be the base name followed by an automatically-generated unique identifier. | string
`parameters` | A value for each parameter if the function's implementation is parameterized. See [Parameters](../../../Manual/09%20Parameters.md). | `Parameter` to value dictionary

## Examples

Adding a function defined by an `ActionPlan`:

```python
package.add_function(plan, args=(A, B, C), base_name="simple_matmul")
```

Convenience syntax to add a function defined by a `Schedule`. A default `ActionPlan` will be created automatically:

```python
package.add_function(schedule, args=(A, B, C), base_name="simple_matmul")
```

Convenience syntax to add a function defined by a `Nest`. A default `Schedule` and `ActionPlan` will be created internally:

```python
package.add_function(nest, args=(A, B, C), base_name="simple_matmul")
```

Adding a function with concrete values specified for its parameters (`P0`, `P1`, `P2`, `P3`).

```python
package.add_function(nest, args=(A, B, C), parameters={P0:16, P1:16, P2:16, P3:1}, base_name="matmul_16_16_16_1")
```

<div style="page-break-after: always;"></div>
