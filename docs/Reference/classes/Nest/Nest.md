[//]: # (Project: Accera)
[//]: # (Version: v1.2.11)

# Accera v1.2.11 Reference

## `accera.Nest(shape)`
Creates an affine loop nest.

## Arguments

argument | description | type/default
--- | --- | ---
`shape` | The shape of the iteration space | tuple of positive integers

## Examples

Create a nest with 3 nested for-loops of sizes 16, 10, and 11:

```python
nest = acc.Nest(shape=(16, 10, 11))
```

<div style="page-break-after: always;"></div>
