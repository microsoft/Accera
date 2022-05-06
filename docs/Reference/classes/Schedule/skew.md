[//]: # (Project: Accera)
[//]: # (Version: v1.2.4)

# Accera v1.2.4 Reference

## `accera.Schedule.skew(index, reference_index [, unroll_loops_smaller_than])`
Transforms a dimension with respect to a reference dimension into a parallelogram by padding with empty elements.

## Arguments

argument | description | type/default
--- | --- | ---
`index` | The dimension to skew | `Index`
`reference_index` | The reference dimension | `Index`
`unroll_loops_smaller_than` | Unroll loops that are smaller than this range (non-inclusive) | non-negative integer

## Examples

Skew dimension `i` with respect to dimension `j`:

```python
schedule.skew(i, j)
```

Skew dimension `j` with respect to dimension `i`, and unroll if the resulting loops are smaller than 3:

```python
schedule.skew(j, i, unroll_loops_smaller_than=3)
```

<div style="page-break-after: always;"></div>


