[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Accera 1.2.0 Reference

## `accera.ActionPlan.cache(source[, index, trigger_index, layout, level, trigger_level, max_elements, thrifty, type])`
Adds a caching strategy to an action plan.

## Arguments

argument | description | type/default
--- | --- | ---
`source` | The array or cache from which this cache is copied. | `Array` or `Cache`
`index` | The index used to determine the cache level. Specify one and only one of `index`, `level`, `max_elements`. | `Index`
`trigger_index` | The index used to determine what level to fill the cache at. `trigger_index` can't come after `index` in the schedule order, and will default to `index` if not specified. Specify at most one of `trigger_index` or `trigger_level`. | `Index`
`layout` | The affine memory map, if different from the source. | [`accera.Layout`](<../Array/Layout.md>)
`level` | The key-slice level to cache (the number of wildcard dimensions in a key-slice). Specify one and only one of `index`, `level`, `max_elements`. | positive integer
`trigger_level` | The key-slice level to fill the cache at. `trigger_level` can't be smaller than `level`, and will default to `level` if not specified. Specify at most one of `trigger_index` or `trigger_level`. | positive integer
`max_elements` | The maximum elements to include in the cached region. Specify one and only one of `index`, `level`, `max_elements`. | positive integer
`thrifty` | Use thrifty caching (copy data into a cache only if the cached data differs from the original active block).  | True or False
`location` | The type of memory used to store the cache. | `MemoryType`

## Returns
A `Cache` handle that represents the created cache.

## Examples

Create a cache of array `A` at level 2.
```python
AA = plan.cache(A, level=2)
```

Create a cache of array `A` with the `Array.Layout.FIRST_MAJOR` layout:
```python
AA = plan.cache(A, level=2, layout=acc.Array.Layout.FIRST_MAJOR)
```

Create a cache of array `A` for dimension `j`:
```python
AA = plan.cache(A, index=j)
```

__Not yet implemented:__ Create a cache of array `A` for the largest active block that does not exceed 1024 elements:
```python
AA = plan.cache(A, max_elements=1024)
```

Create a level 2 cache of array `A` from its level 4 cache:
```python
AA = plan.cache(A, level=4)
AAA = plan.cache(AA, level=2)
```

__Not yet implemented:__ Create a cache of array `A` at index `i` in GPU shared memory:
```python
v100 = Target(model=acc.Target.Models.NVIDIA_TESLA_V100)
AA = plan.cache(A, i, location=v100.MemoryType.SHARED)
```

<div style="page-break-after: always;"></div>
