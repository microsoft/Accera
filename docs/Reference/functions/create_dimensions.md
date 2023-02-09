[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Accera v1.2 Reference

## `accera.create_dimensions([role])`
Creates placeholder dimensions of the specified role. These typically represent runtime `Array` and `Nest` dimensions.

There are two roles for runtime dimensions:

* `accera.Role.INPUT` - immutable dimension that is provided by an input parameter to an Accera function
* `accera.Role.OUTPUT` - mutable dimension that is set within an Accera function

A third type of dimension, the compile-time dimension, is not covered here because it is just a constant.

## Arguments

argument | description | type/default
--- | --- | ---
`role` | The role of the dimension determines if it is mutable or immutable. | [`accera.Role`](<../enumerations/Role.md>). default: `accera.Role.INPUT`. Must be set to `accera.Role.OUTPUT` if intended for an `accera.Role.OUTPUT` `Array`.

## Returns
Tuple of `Dimension`

## Examples

Construct an input array with runtime input dimensions:
```python
import accera as acc
M, K = acc.create_dimensions()
A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, K))
```

Construct a input/output array using a combination of runtime and compile-time dimensions, respectively:
```python
M = acc.create_dimensions()
A = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, 20))
```

Adding a function for an input/output array with runtime input dimensions:
```python
M, N = acc.create_dimensions()
A = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

nest = acc.Nest(M, N)
...

package = acc.Package()
package.add(nest, args=(A, M, N), base_name="myFunc")
```

Construct a output array with runtime (mutable) output dimensions.
```python
M, N = acc.create_dimensions(role=acc.Role.OUTPUT)
A = acc.Array(role=acc.Role.OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))
```

Assign the value of a runtime input dimension to a runtime output dimension:
```python
M = acc.create_dimensions()
N = acc.create_dimensions(role=acc.Role.OUTPUT)

N.value = M
```

Assign the value of a runtime input dimension to a runtime output dimension:
```python
M = acc.create_dimensions()
N = acc.create_dimensions(role=acc.Role.OUTPUT)

N.value = M
```

Assign an integer value to a runtime output dimension:
```python
N = acc.create_dimensions(role=acc.Role.OUTPUT)
N.value = 100
```

Assign a value to a runtime output dimension using an expression of runtime input dimensions:
```python
M, K = acc.create_dimensions()
N = acc.create_dimensions(role=acc.Role.OUTPUT)

N.value = M + K + 1
```


<div style="page-break-after: always;"></div>