[//]: # (Project: Accera)
[//]: # (Version: v1.2.10)

# Accera v1.2.10 Reference

## `accera.Array.sub_array(offsets, shape[, strides])`
Creates a sub-array of a specific shape from an array. The sub-array is created from elements at specified offsets and strides into the original array.

## Arguments

argument | description | type/default
--- | --- | ---
`offsets` | The offsets into the original array. | `Tuple[int]`
`shape` | The size of the sub-array. | `Tuple[int]`
`strides` | (Optional) The strides in the original array used to create the sub-array. | `Tuple[int]`

## Examples

Create a sub-array of size 2x3 from an array of size 5x5 at an offset of {1, 1} and a stride of {2, 1}:

```python
import numpy as np
import accera as acc

N = 5
subArrayNumRows = 2
subArrayNumCols = 3

matrix = np.random.rand(N, N)
Arr = Array(role=Array.Role.INPUT, data=matrix)

# Zero out a sub array of size [2, 3] such that the resulting array looks like this:
# xxxxx
# x000x
# xxxxx
# x000x
# xxxxx

nest = Nest(shape=(subArrayNumRows, subArrayNumCols))
i, j = nest.get_indices()

@nest.iteration_logic
def _():
    SubArr = Arr.sub_array([1, 1], [subArrayNumRows, subArrayNumCols], [2, 1])
    SubArr[i, j] = 0.0

schedule = nest.create_schedule()
```


<div style="page-break-after: always;"></div>
