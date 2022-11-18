[//]: # (Project: Accera)
[//]: # (Version: v1.2.12)

# Accera v1.2.12 Reference

## `accera.create_dimensions([role])`
Creates placeholder dimensions of the specified role. These represent runtime `Array` and `Nest` dimensions.

There are two roles for runtime dimensions:

* `accera.Dimension.Role.INPUT` - immutable dimension that is provided by an input parameter to an Accera function
* `accera.Dimension.Role.OUTPUT` - mutable dimension that is set within an Accera function

A third type of dimension, the compile-time dimension, is not covered here because it is just a constant.

## Arguments

argument | description | type/default
--- | --- | ---
`role` | The role of the dimension determines if it is mutable or immutable. | [`accera.Dimension.Role`](<../classes/Dimension/Role.md>). default: `accera.Dimension.Role.INPUT`. Must be set to `accera.Dimension.Role.OUTPUT` if intended for an `accera.Array.Role.OUTPUT` `Array`.

## Returns
Tuple of `Dimension`

## Examples

Construct an input array with runtime input dimensions:
```python
import accera as acc
M, K = acc.create_dimensions()
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, K))
```

Construct a input/output array using a combination of runtime and compile-time dimensions, respectively:
```python
M = acc.create_dimensions()
A = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, 20))
```

Adding a function for an input/output array with runtime input dimensions:
```python
M, N = acc.create_dimensions()
A = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

nest = acc.Nest(M, N)
...

package = acc.Package()
package.add(nest, args=(A, M, N), base_name="myFunc")
```

Construct a output array with runtime (mutable) output dimensions.
```python
M, N = acc.create_dimensions(role=acc.Dimension.Role.OUTPUT)
A = acc.Array(role=acc.Array.Role.OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))
```

Assign the value of a runtime input dimension to a runtime output dimension:
```python
M = acc.create_dimensions()
N = acc.create_dimensions(role=acc.Dimension.Role.OUTPUT)

N.value = M
```

Assign the value of a runtime input dimension to a runtime output dimension:
```python
M = acc.create_dimensions()
N = acc.create_dimensions(role=acc.Dimension.Role.OUTPUT)

N.value = M
```

Assign an integer value to a runtime output dimension:
```python
N = acc.create_dimensions(role=acc.Dimension.Role.OUTPUT)
N.value = 100
```

Assign a value to a runtime output dimension using an expression of runtime input dimensions:
```python
M, K = acc.create_dimensions()
N = acc.create_dimensions(role=acc.Dimension.Role.OUTPUT)

N.value = M + K + 1
```


<div style="page-break-after: always;"></div>