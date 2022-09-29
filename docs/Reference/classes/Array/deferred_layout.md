[//]: # (Project: Accera)
[//]: # (Version: v1.2.10)

# Accera v1.2.10 Reference

## `accera.Array.deferred_layout(cache)`
Specifies the layout for a `Array.Role.CONST` array based on a `Cache`. For more details, see [Deferred layout of constant arrays](<../../../Manual/08%20Deferred%20Layout%20of%20Constant%20Arrays.md>)

## Arguments

argument | description | type/default
--- | --- | ---
`cache` | The cache that defines the layout to set. | `accera.Cache`

## Examples

Create a constant 16x16 array without specifying a layout. Later on, define its layout based on a cache:

```python
import numpy as np
import accera as acc

matrix = np.random.rand(16, 16)

# Create a constant array with a deferred layout
A = acc.Array(role=acc.Array.Role.CONST, data=matrix, layout=acc.Array.Layout.DEFERRED)
B = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=matrix.shape)

nest = Nest(shape=matrix.shape)
i, j = nest.get_indices()

@nest.iteration_logic
def_():
    B[i, j] += A[i, j]

plan = nest.create_plan()

# create a cache for the constant array
AA = plan.cache(A, i, layout=acc.Array.Layout.FIRST_MAJOR, thrifty=True)

# update the constant array's layout based on the cache
A.deferred_layout(cache=AA)
```


<div style="page-break-after: always;"></div>
