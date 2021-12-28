[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Accera 1.2.0 Reference

## `accera.create_parameters(number)`
Creates placeholder parameters.

## Arguments

argument | description | type/default
--- | --- | ---
`number` | number of parameters to create | positive integer

## Returns
Tuple of `Parameter`

## Examples

Create 3 parameters `m`, `n`, `k` and use them to parameterize the nest shape:

```python
m, n, k = acc.create_parameters(3)
nest = acc.Nest(shape=(m, n, k))
```


<div style="page-break-after: always;"></div>
