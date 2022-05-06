[//]: # (Project: Accera)
[//]: # (Version: v1.2.4)

# Accera v1.2.4 Reference

## `accera.Schedule.tile(shape)`
The `tile` transformation is a convenience syntax that takes a tuple of indices and a tuple of sizes, and splits each index by the corresponding size. The indices involved in the split are then ordered such that all the outer indices precede all of their respective inner indices.

## Arguments

argument | description | type/default
--- | --- | ---
`shape` | Mapping of indices to tile sizes | dict of `Index` and non-negative integers

## Returns
Tuple of `Index` representing the new inner dimensions.

## Examples

Tile the `i`, `j`, and `k` dimensions by 8, 2, and 3, respectively.

```python
ii, jj, kk = schedule.tile({
    i: 8,
    j: 2,
    k: 3
})
```

<div style="page-break-after: always;"></div>


