[//]: # (Project: Accera)
[//]: # (Version: v1.2.13)

# Accera v1.2.13 Reference
## `accera.Dimension([role, value])`
Constructs a runtime dimension size with optional initialization.

Note: This constructor is meant for advanced use cases that involve Python generator expressions. For the simplified syntax to create dimensions, see [create_dimensions](../../functions/create_dimensions.md).

## Arguments

argument | description | type/default
--- | --- | ---
`role` | The role of the dimension determines if it is mutable or immutable. | [`accera.Dimension.Role`](<Role.md>). default: `accera.Dimension.Role.INPUT`. Must be set to `accera.Dimension.Role.OUTPUT` if used for an `accera.Array.Role.OUTPUT` `Array`.
`value` | The optional value to initialize the dimension. Only applies to mutable dimensions (`accera.Dimension.Role.OUTPUT`) | integer or `Dimension`

## Returns
`Dimension`

## Examples

Construct an output array with runtime dimensions using Python tuple comprehension over an input shape:
```python
import accera as acc

# input_shape is a tuple or list of acc.Dimensions or integers
output_shape = tuple(acc.Dimension(role=acc.Dimension.Role.OUTPUT, value=i) for i in input_shape)
A = acc.Array(role=acc.Array.Role.OUTPUT, element_type=acc.ScalarType.float32, shape=output_shape)
```

<div style="page-break-after: always;"></div>