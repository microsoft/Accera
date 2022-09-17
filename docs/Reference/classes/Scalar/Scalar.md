[//]: # (Project: Accera)
[//]: # (Version: v1.2.9)

# Accera v1.2.9 Reference

## `accera.Scalar([element_type, value])`
Constructs a scalar that holds a number.

## Arguments

argument | description | type/default
--- | --- | ---
`element_type` | The element type. | [`accera.ScalarType`](<../../enumerations/ScalarType.md>), default: `accera.ScalarType.float32`.
`value` | An optional value. | A number.

## Examples

Construct a float32 scalar:
```python
import accera as acc

X = acc.Scalar()
```

Construct a float32 scalar and initialize it:
```python
Pi = acc.Scalar(value=3.14)
```

Construct integer scalars and perform arithmetic operations on them:
```python
X = acc.Scalar(element_type=acc.ScalarType.int32)
Y = acc.Scalar(element_type=acc.ScalarType.int32)
Y.value = x + 2
```

<div style="page-break-after: always;"></div>


