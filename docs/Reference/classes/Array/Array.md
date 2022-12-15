[//]: # (Project: Accera)
[//]: # (Version: v1.2.14)

# Accera v1.2.14 Reference

## `accera.Array(role[, data, element_type, layout, offset, shape])`
Constructs an array.

## Arguments

argument | description | type/default
--- | --- | ---
`role` | The role of the array determines if the array scope is internal or external, if the array is mutable or immutable, and if the array memory is dynamically allocated. | [`accera.Array.Role`](<Role.md>)
`data` | The contents of a constant array. Required for `accera.Array.Role.CONST` arrays but should not be specified for other roles. | Python buffer or `numpy.ndarray`.
`element_type` | The array element type. | [`accera.ScalarType`](<../../enumerations/ScalarType.md>), default: `accera.ScalarType.float32`.
`layout` | The affine memory map. | [`accera.Array.Layout`](<Layout.md>), or tuple of (integers or [`accera.Dimension`](<../Dimension/Dimension.md>)), default: `accera.Array.Layout.FIRST_MAJOR`.
`offset` | The offset of the affine memory map | integer (positive, zero, or negative), default: 0.
`shape` | The array shape. Required for roles other than `accera.Array.Role.CONST`, should not be specified for `accera.Array.Role.CONST`. | tuple of (integers or [`accera.Dimension`](<../Dimension/Dimension.md>)).

## Examples

Construct an input array:
```python
import accera as acc
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(10, 20))  # the default layout is acc.Array.Layout.FIRST_MAJOR
```

Construct an input array with an explicit standard layout:
```python
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(10, 20), layout=acc.Array.Layout.LAST_MAJOR)
```

Construct an input array with an explicit affine memory map:
```python
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(10, 20), layout=(1, 10))
```

Construct an input array with an infinite (undefined) major dimension:
```python
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(10, acc.inf), layout=acc.Array.Layout.LAST_MAJOR)
```

Construct a input array with both runtime and compile-time dimension sizes:
```python
M = acc.create_dimensions()
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, 20))
```

Construct an input/output array:
```python
A = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(10, 20))
```

Construct an input/output array with runtime input dimension sizes:
```python
M, N = acc.create_dimensions()
A = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))
```

Construct an output array with runtime output dimension sizes:
```python
M, N = acc.create_dimensions(role=acc.Dimension.Role.OUTPUT)
A = acc.Array(role=acc.Array.Role.OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))
```

Construct an output array with explicit affine memory map:
```python
M, N = acc.create_dimensions(role=acc.Dimension.Role.OUTPUT)
A = acc.Array(role=acc.Array.Role.OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N), layout=(1, M))
```


Construct a constant array:
```python
D = np.random.rand(10, 16)
A = acc.Array(role=acc.Array.Role.CONST, data=D)
```

Construct a constant array with an explicit element type and layout, which does not necessarily match the input data:
```python
D = np.random.rand(10, 16)
A = acc.Array(role=acc.Array.Role.CONST, element_type=acc.ScalarType.float32, layout=acc.Array.Layout.LAST_MAJOR, data=D)
```

Construct a temporary array:
```python
A = acc.Array(role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, shape=(10, 20), layout=acc.Array.Layout.LAST_MAJOR)
```

<div style="page-break-after: always;"></div>


