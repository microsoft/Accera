[//]: # (Project: Accera)
[//]: # (Version: v1.2.1)

# Section 6: Plans - Caching
In the previous sections, we defined the logic and then scheduled its iterations. Now, let's move on to completing the implementation with target-specific options.

First, we create a plan from the schedule:
```python
plan = schedule.create_plan()
```
The Accera programming model allows us to create multiple plans from a single schedule. More importantly, we can modify individual plans without changing the schedule. We can manually specify the target platform by calling `create_plan` that takes a `target` argument. The default value of this `target` argument is `acc.Target.HOST`, which refers to the current host computer.

In this section, we discuss how to add data caching strategies to a plan.

## Key slices

Recall that a [slice](<03%20Schedules.md>) is a set of iteration space elements that match a coordinate template with wildcards, such as `(1, *, 3)`. A *key-slice* is a slice with only right-aligned wildcards, such as `(1, 2, *)` and `(3, *, *)`. The *level* of a key-slice is determined by the number of wildcards in its definition. For example, `(1, 2, *)` is a level 1 key-slice and `(3, *, *)` is a level 2 key-slice.

Note that the `key-slices` are changed by reordering the dimensions of the iteration space. However, it is always true that the entire *d*-dimensional iteration space is a level *d* key-slice and each individual element is a level zero key-slice. For a total of *d+1* different key-slices, each iteration belongs to one key-slice from each level, zero to *d*. When the schedule is executed, the key-slices containing the current iteration are called the *current key-slices*.

In the Accera programming model, key-slices are significant because they partition the iteration space into sets of consecutive iterations. Therefore, they can describe the phases of computation at different levels of granularity. The term *key-slice* suggests using them to *key* (trigger) different actions. Specifically, each time the current level-*l* key slice changes, we use this event to trigger a cache update.

As mentioned above, a key-slice can be identified by its level. Another way to specify a key-slice is to take advantage of the iteration space dimensions being named in the order. To specify a key-slice for a dimension, replace it and subsequent dimensions with wildcard symbols. For example, if the names of the iteration space dimensions are `(i, j, k)`, then a key-slice that corresponds to the dimension `j` is one of `(0, *, *)`, `(1, *, *)`, etc. Both ways of specifying a key-slice are useful and Accera uses them interchangeably.

## Active elements and active blocks
A loop nest operates on the data that is stored in arrays. Each key-slice can access a subset of the array elements, which we call the *active elements* that correspond to that specific key-slice. Since the current iteration belongs to key-slices at different levels, we need to define corresponding sets of active elements at different levels.

More precisely, array `A` elements that are read from or written to by the iterations of the current level *l* key-slice are called the level *l* active elements of `A`.  This set of elements does not necessarily take the shape of a block. Therefore, the *level l active block* of `A` can be defined as the smallest block of elements that contains all of the level *l* active elements in `A`. Accera uses active blocks to define caching strategies.

Just like we can specify a key-slice using a dimension, we can also refer to the active block that corresponds to a specific dimension. For example, if the names of the iteration space dimensions are `(i, j, k)` and the current iteration is one of the iterations for which `i=3`, then the active block in `A` that corresponds to dimension `j` is the block that includes all the elements touched by the key-slice `(3, *, *)`.

## Caches
An Accera cache is a local copy of an active block. A cache is contiguous in memory and its memory layout may differ from the layout of the original array. The loop nest iterations operate on the cache elements instead of the original array elements.

The contents of the active block are copied into the cache at the start of the corresponding key-slice. If the array is mutable (namely, an input/output array or a temporary array), the cache contents are copied back into the original array at the end of the key-slice. 

### Caching by level
To define a cache for a given array, all we need is to specify the desired level.For example:
```python
AA = plan.cache(A, level=2)
```
The return value `AA` is a handle that can be used to refer to the cache in subsequent operations. We can choose the cache layout, just as we did when we defined the original array.
```python
AA = plan.cache(A, level=2, layout=acc.Array.Layout.FIRST_MAJOR)
```

### Caching by dimension
As mentioned above, we can specify an active block using a dimension. We use this to define a cache as follows: 
```python
AA = plan.cache(A, index=j)
```

### Caching by element budget
Note that the current active blocks of an array are nested, and their sizes are monotonic (nondecreasing) in their level. Therefore, we can also select the largest active block that does not exceed a certain number of elements: 
```python
AA = plan.cache(A, max_elements=1024)
```


## __Not yet implemented:__ Thrifty caching
By default, Accera caching strategies are *thrifty* in the sense that the data is physically copied into an allocated cache only if the cached data somehow differs from the original active block. Therefore, if the original active block is already in the correct memory layout and resides contiguous in memory. Accera skips the caching steps and uses the original array instead. Note that a physical copy is created on a GPU if the cache is supposed to be allocated a different type of memory than the original array (e.g., the array is in global memory, but the cache is supposed to be in shared memory).

For example, assume that `A` is a two-dimensional array and its active block at the chosen level is always one of its rows. If `A` is row-major, the rows are already stored contiguously. Additionally, the data in the active block and the data to be copied to cache are identical: both are contiguous and share the same memory layout. In this case, there is no benefit in using cache over the original array. The thrifty caching strategy will skip the caching steps and use the original array instead.

On the other hand, if `A` is column-major, its rows are not stored contiguously. In this case, copying the active row into a contiguous temporary location could be computationally advantageous. Therefore, the thrifty caching strategy would create the cache and populate it with the data. 


Thrifty caching can be turned off using the optional argument `thrifty=False`. If turned off, a physical copy is always created. 

[comment]: # (MISSING:)
[comment]: # (* A concept of disjoint active blocks. This is critical for temp arrays and the question of which part of the array do we actually store in RAM)
[comment]: # (* The idea of double buffering - this becomes complex for mutable caches and consecutive active blocks that overlap. This can lead to a cache coherence issue.)

## Hierarchical caching
Caches can be composed hierarchically. Namely, a high-level key-slice can trigger a copy from the original array into a big cache, and a lower level key-slice can be used to trigger a copy from the big cache into a smaller cache.

For example,
```python
AA = plan.cache(A, level=4)
AAA = plan.cache(AA, level=2)
```

## Multicaching
While caches are defined with a key-slice `level`, a higher-level key slice `trigger_level` can be specified as the trigger key-slice for copying multiple successive active blocks of elements to a local copy. These copied active blocks have their layouts defined as usual, and only the trigger level for copying them has been changed. Since active blocks are not mutually exclusive, this can result in the same element being copied into multiple locations as separate caches. Therefore, a `trigger_level` may only be specified on an `INPUT` or `CONST` array as Accera does not support multicache write coherence.

For example,
```python
AA = plan.cache(A, level=2, trigger_level=4)
```

## __Not yet implemented:__ Mapping caches to specific types of memory
Some target platforms have different types of memory that can hold Accera caches. In the case of a GPU target, caches can be located in *global or shared memory*. Following Python code can be used to specify the location of a cache:
```python
AA = plan.cache(A, level=4, location=v100.MemoryType.SHARED)
```


<div style="page-break-after: always;"></div>
