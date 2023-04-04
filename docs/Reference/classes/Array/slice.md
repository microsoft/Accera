[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Accera v1.2 Reference

## `accera.Array.slice(sliced_dims, sliced_offsets)`
Creates a sliced view of reduced rank from an array. The view is created from elements at specified offsets of the sliced dimensions of the original array.

## Arguments

argument | description | type/default
--- | --- | ---
`sliced_dims` | The dimension indices of the original array to slice on. | `Tuple[int]`
`sliced_offsets` | The offsets of the corresponding dliced dimensions. | `Tuple[Scalar]`

## Examples

Clear a slice of size 5 from an array of size 5x5 at dimension 0 with offset 2:

```python
import numpy as np
import accera as acc

N = 5
slice_dim = 0
slice_offset = 2

matrix = np.random.rand(N, N)
Arr = Array(role=Role.INPUT, data=matrix)

# Zero out a slice of size [5] such that the resulting array looks like this:
# xxxxx
# xxxxx
# 00000
# xxxxx
# xxxxx

nest = Nest(shape=(N,))
i, = nest.get_indices()

@nest.iteration_logic
def _():
    SliceArr = Arr.slice([slice_dim], [slice_offset])
    SliceArr[i] = 0.0

schedule = nest.create_schedule()
```


<div style="page-break-after: always;"></div>
