[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Section 6: Action plans - Caching
In the previous sections, we defined the logic and then scheduled its iterations. The next step is to complete the implementation with target-specific options.

First, we create an action plan from the schedule:
```python
plan = schedule.create_action_plan()
```
We can create multiple action plans from a schedule and modify each one without changing the schedule. To manually specify the target platform, the call to `create_action_plan` can take a `target` argument. The default value of this argument is `acc.Target.HOST`, which represents the current host computer.

In this section, we discuss how to add data caching strategies to an action plan.

## Key slices

Recall that a [slice](03%20Schedules.md) is a set of iteration space elements that match a coordinate template with wildcards, such as (1, \*, 3). A *key-slice* is a slice whose wildcards are right-aligned, such as (1, 2, \*) and (3, \*, \*). The *level* of a key-slice is the number of wildcards in its definition, so for example, (1, 2, \*) is a level 1 key-slice and (3, \*, \*) is a level 2 key slice.

Note that reordering the dimensions of an iteration space changes which slices are key-slices. However, it is always true that the entire *d*-dimensional iteration space is a level *d* key-slice and each individual element is a level zero key-slice. Each iteration belongs to one key-slice from each level, from zero to *d*, for a total of *d+1* different key-slices. When the schedule is executed, the key-slices that contain the current iteration are called the *current key-slices*.

The significance of key-slices is that they partition the iteration space into sets of consecutive iterations. Therefore, they can be used to describe phases of the computation, at different levels of granularity. The term *key-slice* suggests that we will use them to *key* different actions. Specifically, each time the current level-*l* key slice changes, we use this event to trigger a cache update.

As mentioned above, we can identify a specific current key-slice by noting its level. Another way to specify a current key-slice is to take advantage of the fact that the iteration space dimensions are named and ordered, and to specify the first dimension that is replaced by a wildcard in the key-slice definition. For example, if the names of the iteration space dimensions are `(i, j, k)`, then the current key-slice that corresponds to the dimension `j` is one of `(0, *, *)`, `(1, *, *)`, etc. Both ways of specifying a current key-slice are useful and Accera uses them interchangeably.

## Active elements and active blocks
A loop nest operates on data stored in arrays. Each key-slice touches a subset of the array elements, which we call the *active elements* that correspond to that key-slice. Since the current iteration belongs to key-slices at different levels, we also define corresponding sets of active elements at different levels.

More precisely, the elements of an array `A` that are touched (read from or written to) by the iterations of the current level *l* key-slice are called the level *l* active elements of `A`. This set of elements does not necessarily take the shape of a block. Therefore, we define the *level l active block* of `A` as the smallest block of elements that contains all of the level *l* active elements in `A`. Accera uses active blocks to define caching strategies.

Just like we can specify a current key-slice using a dimension, we can also refer to the active block that corresponds to a dimension. For example, if the names of the iteration space dimensions are `(i, j, k)` and the current iteration is one of the iterations for which `i=3` then the active block in `A` that corresponds to dimension `j` is the block that includes all the elements touched by the key-slice `(3, *, *)`.

## Caches
A Accera cache is a local copy of an active block. A cache is contiguous in memory and its memory layout may be different from the layout of the original array. The loop nest iterations operate on the elements of the cache instead of the original array elements.

The contents of the active block are copied into the cache at the beginning of the corresponding key-slice. If the array is mutable (namely, if it is an input/output array or a temporary array), the contents of the cache are also copied back into the original array at the end of the key-slice.

### Caching by level
To define a cache for a given array, all we need to do is specify the desired level. For example:
```python
AA = plan.cache(A, level=2)
```
The return value `AA` is a handle that can be used to refer to the cache in subsequent operations. We can choose the cache layout, just as we did when we defined the original array.
```python
AA = plan.cache(A, level=2, layout=acc.Array.Layout.FIRST_MAJOR)
```

### Caching by dimension
As mentioned above, we can also specify an active block using a dimension. We use this to define a cache as follows:
```python
AA = plan.cache(A, index=j)
```

### Caching by element budget
Note that the current active blocks of an array are nested, and their size is monotonic (nondecreasing) in their level. Therefore, we can also choose the largest active block that does not exceed a certain budget of elements.
```python
AA = plan.cache(A, max_elements=1024)
```


## __Not yet implemented:__ Thrifty caching
By default, Accera caching strategies are *thrifty*, which means that data is physically copied into an allocated cache only if the cached data somehow differs from the original active block. In other words, if the original active block happens to already be contiguous in memory and in the correct memory layout, then Accera skips the caching step and instead just uses the original array. Note that on a GPU, if the cache is supposed to be allocated in a different type of memory than the original array (e.g., the array is in global memory but the cache is supposed to be in shared memory) then a physical copy is created.

For example, say that `A` is a two-dimensional array and its active block at the chosen level is always one of its rows. If `A` is row-major, its rows are already stored contiguously and the data in the active block is identical to the data that would be copied into the cache: both are contiguous and both share the same layout. Since the two are identical, there is no benefit to using the cache over using the original array. The thrifty caching strategy skips the caching step and instead uses the data in the original array. On the other hand, if `A` is column-major, its rows are not stored contiguously. Copying the active row into a contiguous temporary location could be computationally advantageous. In this case, the thrifty caching strategy would create the cache and populate it with data.

Thrifty caching can be turned off using the optional argument `thrifty=False`. When thrifty caching is turned off, a physical copy is always created.

[comment]: # (MISSING:)
[comment]: # (* A concept of disjoint active blocks. This is critical for temp arrays and the question of which part of the array do we actually store in RAM)
[comment]: # (* The idea of double buffering - this becomes complex for mutable caches and consecutive active blocks that overlap. This can lead to a cache coherence issue.)

## __Not yet implemented:__ Hierarchical caching
Caches can be composed hierarchically. Namely, a high-level key-slice can trigger a copy from the original array into a big cache, and a lower level key-slice can be used to trigger a copy from the big cache into a smaller cache.

For example,
```python
AA = plan.cache(A, level=4)
AAA = plan.cache(AA, level=2)
```

## Multicaching
Caches are defined with a key-slice `level`, and a higher-level key slice `trigger_level` can be specified as the trigger key-slice for copying multiple successive active blocks of elements to a local copy. These copied active blocks each have their layouts defined as usual, only the trigger level for copying them has been changed. Note that since active blocks are not mutually exclusive, this can result in the same element being copied into multiple locations in the local copy, however they will be in separate caches. Because of this, a `trigger_level` may only be specified on an `INPUT` or `CONST` array as Accera does not perform multicache write coherence.

For example,
```python
AA = plan.cache(A, level=2, trigger_level=4)
```

## __Not yet implemented:__ Mapping caches to specific types of memory
Some target platforms have different types of memory that can hold Accera caches. For example, on a GPU target, caches can be located in *global memory* or *shared memory*. To explicitly choose the location of the cache, we write
```python
AA = plan.cache(A, level=4, location=v100.MemoryType.SHARED)
```


<div style="page-break-after: always;"></div>
