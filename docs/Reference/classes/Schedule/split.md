[//]: # (Project: Accera)
[//]: # (Version: v1.2.4)

# Accera v1.2.4 Reference

## `accera.Schedule.split(index, size)`
The `split` transformation takes a dimension `i` and a `size`, modifies `i`, and creates a new dimension `ii`.

Assume that the original size of dimension `i` was *n*: The `split` transformation splits dimension `i` into *ceil(n/size)* parts of size `size`, arranges each of those parts along dimension `ii`, and stacks the *ceil(n/size)* parts along dimension `i`.

If the split size does not divide the dimension size, empty elements are added such that the split size does divide the dimension size.

## Arguments

argument | description | type/default
--- | --- | ---
`index` | The dimension to split | `Index`
`size` | The split size | non-negative integer

## Returns
`Index` for the new inner dimension

## Examples

Split the `i` dimension by 5, creating a new dimension `ii`:

```python
ii = schedule.split(j, 5)
```

<div style="page-break-after: always;"></div>


