[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Accera 1.2.0 Reference

## `accera.Schedule.tile(indices, sizes)`
The `tile` transformation is a convenience syntax that takes a tuple of indices and a tuple of sizes, and splits each index by the corresponding size. The indices involved in the split are then reordered such that all the outer indices precede all of the inner indices.

## Arguments

argument | description | type/default
--- | --- | ---
`indices` | The indices to tile | tuple of `Index`
`sizes` | The tile sizes | tuple of non-negative integers

## Returns
Tuple of `Index` representing the new inner dimensions

## Examples

Tile the `i`, `j`, and `k` dimensions by 8, 2, and 3 respectively

```python
ii, jj, kk = schedule.tile((i, j, k), (8, 2, 3))
```

<div style="page-break-after: always;"></div>
