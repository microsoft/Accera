[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Section 8: Deferred layout of constant arrays
We revisit the topic of memory layout of constant arrays. As mentioned in [Section 1](01%20Arrays.md), the contents of constant arrays are known at compile time, they are immutable, and they are not externally visible. This allows Accera to store them using a non-standard layout, optimized for a specific action plan. In some cases, there may even be a benefit to storing multiple copies of each array element (e.g., storing a matrix in both row-major and column-major layouts).

## Deferred layout based on a cache
A Accera cache strategy makes local copies of an array's active blocks. The constant array can be laid out based on a defined cache. Namely, the array is stored by serializing its active blocks one-after-the-other. If the caching strategy is `thrifty=True`, no data needs to be copied at runtime and the active blocks are ready to use.

To define an array layout based on a cache, note that the Accera DSL has to overcome a chicken-and-egg situation: on one hand, arrays need to be defined upfront, even before the nest logic; on the other hand, the array layout depends on a cache, which is only defined as part of the action plan. We overcome this problem by splitting the array definition into two parts. We still define the constant array upfront, but avoid committing to a specific layout:
```python
import accera as acc
import numpy as np

matrix = np.random.rand(16, 16)
A = acc.Array(role=acc.Array.Role.CONST, data=matrix, layout=acc.Array.Layout.DEFERRED)
```
Then we proceed to define the nest logic, the schedule, and the action plan. Imagine that we define an action plan named `plan` and use that plan to define a cache for `A` based on dimension `i`:
```python
AA = plan.cache(A, i, layout=acc.Array.Layout.FIRST_MAJOR, thrifty=True)
```
We can now use the cache `AA` to determine the layout of the original array `A`:
```python
A.deferred_layout(cache=AA)
```


<div style="page-break-after: always;"></div>
