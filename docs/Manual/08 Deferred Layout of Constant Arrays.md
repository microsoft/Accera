[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Section 8: Deferred layout of constant arrays
Let's revisit the memory layout of constant arrays. As explained in [Section 1](<01%20Arrays%20and%20Scalars.md>), the contents of constant arrays are known at compile-time, and these contents are immutable. Accera stores constant arrays in a non-standard memory layout optimized for a particular plan. In some cases, storing multiple copies of each array element may even prove advantageous (e.g., storing a matrix in row-major and column-major layouts).

## Deferred layout based on a cache
Accera's cache strategy creates local copies of an array's active blocks. The constant array can be arranged based on the defined cache. Specifically, the array is stored by serializing the active blocks consecutively. If the caching strategy is `thrifty=True`, the active blocks are ready to use without copying the data.   

To define an array layout based on a cache, Accera DSL has to overcome the chicken-and-egg paradox. While on the one hand, arrays need to be defined even before the nest logic. On the other hand, array layout depends on a cache, which is defined only as a part of a plan. In Accera, we overcome this situation by splitting the array definition into two parts. Though we still define the constant array upfront, we avoid committing to a specific layout. 
```python
import accera as acc
import numpy as np

matrix = np.random.rand(16, 16)
A = acc.Array(role=acc.Array.Role.CONST, data=matrix, layout=acc.Array.Layout.DEFERRED)
```
Now we define the nest logic, the schedule, and the plan. Consider that we define a plan named `plan` and use this plan to define a cache `A` based on dimension `i`:
```python
AA = plan.cache(A, i, layout=acc.Array.Layout.FIRST_MAJOR, thrifty=True)
```
We can now use the cache `AA` to determine the layout of the original array `A`:
```python
A.deferred_layout(cache=AA)
```


<div style="page-break-after: always;"></div>
