[//]: # (Project: Accera)
[//]: # (Version: v1.2.3)

# Accera v1.2.3 Reference

## `accera.fuse(schedules[, *args, partial])`
The `fuse` operation combines multiple iteration spaces into a single "fused" iteration space. The fused iteration space represents the union of the work in the original spaces.

In cases where it doesn't make sense to fuse all of the iteration space dimensions, we can choose to fuse a prefix of the dimensions and leave the rest unfused.

## Arguments

argument | description | type/default
--- | --- | ---
`schedules` | Either the schedules to fuse if performing partial fusing, or the first schedule to fuse if fusing all dimensions | tuple of `Schedule` or `Schedule` |
`*args` | Optional variable arguments containing subsequent schedules to fuse | variable `Schedule` arguments
`partial` | The number of dimensions to fuse. If not specified, all dimensions will be fused | non-negative integer

## Returns
The fused `Schedule`

## Examples

Full fusing of same-shaped iteration spaces:

```python
# Fuse all dimensions of schedule0 and schedule1
schedule = acc.fuse(schedule0, schedule1)
f, i, j = schedule.get_indices()

# Reorder the indices so that the fused dimension is the innermost
schedule.reorder(i, j, f)
```

Partial iteration space fusing:

```python
# Fuse the first two dimensions of schedule0 and schedule1
schedule = acc.fuse((schedule0, schedule1), partial=2)
f, i, j, k = schedule.get_indices()

# Reorder the indices to interleave the schedules
schedule.reorder(i, j, f, k)
```


<div style="page-break-after: always;"></div>
