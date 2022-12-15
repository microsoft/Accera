[//]: # (Project: Accera)
[//]: # (Version: v1.2.14)

# Accera v1.2.14 Reference

## `accera.Schedule.reorder(order, *args)`
The `reorder` transformation sets the order of the indices in the schedule.

These orders are not allowed:
1. The *outer dimension* created by a `split` transformation must always precede the corresponding *inner dimension*.
2. The *fusing dimension* created by a `fuse` operation must always precede any *unfused dimensions*.

## Arguments

argument | description | type/default
--- | --- | ---
`order` | Either the order of indices to set or the outermost index if using variable arguments | tuple of `Index` or `Index`.
`*args` | Optional variable arguments containing subsequent indices to set | variable `Index` arguments

## Examples

Reorder a schedule by moving the `k` dimension to the outermost loop:

```python
schedule.reorder(k, i, j)
```

Using a tuple to reorder a schedule. This overloaded form is better suited for parameters:

```python
schedule.reorder(order=(k, i, j))
```


<div style="page-break-after: always;"></div>


