[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Accera 1.2.0 Reference

## `accera.get_parameters_from_grid(parameter_grid)`
Get parameters combinations from parameter gid.

## Arguments

argument | description | type/default
--- | --- | ---
`parameter_grid` | A set of different values for each parameter, which will be used to generate a list of all valid parameter combinations | dictionary

## Returns
List of dictionary

## Examples

Get parameters combinations from a parameter grid:

```python
parameter_grid = {p1:[1, 2, 3], p2:[4], p3:[5,6]}
parameters = acc.get_parameters_from_grid(parameter_grid)
```

<div style="page-break-after: always;"></div>
