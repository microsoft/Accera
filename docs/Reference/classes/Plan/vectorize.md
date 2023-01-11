[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Accera v1.2 Reference

## `accera.Plan.vectorize(index)`
Only available for targets that have SIMD registers and support vector instructions. Marks a dimension of the iteration-space for vectorization.

## Arguments

argument | description | type/default
--- | --- | ---
`index` | The index to vectorize. | `Index`

## Examples

Mark the dimension `ii` for vectorized execution:

```python
plan.vectorize(index=ii)
```

<div style="page-break-after: always;"></div>
