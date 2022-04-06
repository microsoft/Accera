[//]: # (Project: Accera)
[//]: # (Version: v1.2.3)

# Section 3: Schedules
We begin with `nest` from [Section 2](<02%20Simple%20Affine%20Loop%20Nests.md>) which captures the logic of matrix-matrix multiplication. We use `nest` to create a `Schedule` that controls the execution order of the nest's iterations. Schedules are target-independent in the sense that the same schedule can be used to emit code for multiple target platforms.

We create a default schedule as follows:
```python
schedule = nest.create_schedule()
```

The default schedule is equivalent to the following straightforward for-loop version of the loop nest:
```python
for i in range(16):
    for j in range(10):
        for k in range(11):
            C[i, j] += A[i, k] * B[k, j]
```
In other words, each of the logical pseudo-code loops in `nest` becomes an actual for-loop in the default schedule.

We can now transform this schedule in various ways. However, these transformations do not change the underlying logic defined in `nest` and merely change the order of the loop iterations. We can even generate as many independent schedules as we want by calling `nest.create_schedule()`.

## Iteration spaces: a geometric representation of schedules
In the Accera programming model, a schedule is geometrically interpreted as a multi-dimensional discrete hypercube called the *iteration space* of the nest. The elements of the iteration space represent the individual iterations of the loop nest. Initially, the dimensions of the iteration space correspond to the logical loops defined in `nest`. For example, the default iteration space for the matrix-matrix multiplication nest forms a three-dimensional discrete hypercube, whose shape is (16, 10, 11).

### How does an iteration space imply an order over the iterations?
The dimensions of the iteration space are ordered, and this order corresponds to the original order of the logical loops in `nest` by default. In fact, the order over the dimensions induces a lexicographic sequence over the individual elements of the iteration space.

This geometric interpretation of schedules helps us visualize how different transformations modify them. While some transformations merely rearrange the elements of the iteration space, others increase its dimensions, and some even pad the space with empty (no-op) elements. The transformed iteration space defines a new lexicographic order over the individual iterations.

Comment: It is important not to confuse arrays, like `A`, `B`, `C`, with iteration spaces, like `schedule`. A possible source of confusion could be that both arrays and iteration spaces have a multidimensional rectilinear structure (i.e., they both look like hypercubes). However, arrays and iteration spaces are fundamentally different. Arrays are data structures whose elements are scalars. Iteration spaces are abstract geometric representations of schedules and their elements represent individual iterations of a loop nest. Transformations apply to iteration spaces, not to arrays.

Comment: Accera's geometric interpretation of schedules resembles the *iteration domain polyhedron*, which is the cornerstone of the [polyhedral model](https://en.wikipedia.org/wiki/Polytope_model) of compiler optimization. However, unlike polyhedrons, Accera iteration spaces are not embedded in a continuous space and cannot be manipulated by algebraic transformations. Accera iteration spaces always remain rectilinear and are inherently discrete objects.

### Iteration space slices
*Iteration space slices* is an abstract concept that affects different aspects of the Accera programming model. Since the iteration space dimensions are ordered, each element of the iteration space can be identified by a vector of coordinates.  For example, the vector (5, 6, 7) identifies the iteration at position 5 along the first dimension, position 6 along the second dimension, and position 7 along the third dimension. If one or more coordinates are replaced with the *wildcard* symbol `\*`, we get an *iteration space slice*, which is a set of iterations obtained by replacing the wildcard with all possible values. For example, (\*, \*, 5) represents a slice containing all the elements with 5 as their last coordinate. The *dimension of a slice* equals the number of wildcards in its definition.

### Loops, indices, and dimensions
When we defined `nest`, we used variables such as `i`, `j`, and `k` to name the loops in the loop-nest. When we described the default schedule using equivalent for-loops, `i`, `j`, and `k` became the index variables of those loops. Now, when we represent a schedule as an iteration space, these variables are used as the names of the corresponding iteration space dimensions. From here on, we move seamlessly between these different representations and use the terms *loop*, *index*, and *dimension* interchangeably.

## Schedule transformations
Iteration space transformations change the shape of the iteration space, possibly by adding dimensions or padding the space with empty elements.

The iterations space always retains its rectilinear (hypercube) shape. In some cases, Accera transformations must pad the iteration space with empty elements to avoid reaching a jagged iteration space structure.

### `reorder`
```python
# Reorder the indices.
schedule.reorder(k, i, j)
```

The `reorder` transformation sets the order of indices in the schedule. From the iteration space point-of-view, `reorder` performs a pivot rotation of the iteration space, which orients its dimensions in a specified order. Since the iteration space elements are executed in lexicographic order, pivoting the iteration space is equivalent to reordering the loops.

For example, we can write:
```python
schedule.reorder(k, i, j)
```
After this transformation, `schedule` becomes equivalent to the Python code:
```python
for k in range(11):
    for i in range(16):
        for j in range(10):
            C[i, j] += A[i, k] * B[k, j]
```

However, some orders are not allowed. Describing these restrictions in full will require concepts that are yet to be introduced. Therefore, we are stating these restrictions here and will discuss them later in the upcoming sections. The restrictions are:
1. The *inner dimension* created by a `split` transformation (see below) must be ordered later than its corresponding *outer dimension*.
2. The *fusing dimension* created by a `fuse` operation (see [Section 4](<04%20Fusing.md>)) must always precede any *unfused dimensions*.


Also note that `reorder` can also have the following overloaded form:
```python
schedule.reorder(order=(k, i, j))
```
This form is better suited for use with parameters (see [Section 9](<09%20Parameters.md>)).

### `split`
```python
# Splits dimension i into equally-sized parts, orients those parts along a new dimension ii, and stacks those parts along dimension i
ii = schedule.split(i, size)
```

From the iteration space point-of-view, the `split` transformation takes a dimension `i` and a `size`, modifies `i`, and creates a new dimension `ii`. Assume that the original size of dimension `i` was *n*: The `split` transformation splits the dimension `i` into *ceil(n/size)* parts of size `size`, orients each of these parts along dimension `ii`, and stacks the *ceil(n/size)* parts along the dimension `i`. If the split size does not divide the dimension size, empty elements are added such that the split size does divide the dimension size. As a result of the split, the size of `i` becomes *ceil(n/size)*, the size of the new dimension `ii` equals `size`, and the iteration space remains rectilinear.

In loop terms, `ii = split(i, size)` splits loop `i` into two loops: an inner loop `ii` and an outer loop, which inherits the original name `i`. Note that the outer loop always precedes the corresponding inner loop in the loop ordering.

For example, starting from `nest` defined in [Section 2](<02%20Simple%20Affine%20Loop%20Nests.md>), we could write:
```python
schedule = nest.create_schedule()
jj = schedule.split(j, 5)
```
The resulting iteration space has a shape of (16,2,5,11) and corresponds to the following python code:
```python
for i in range(16):
    for j in range(0, 10, 5):
        for jj in range(5):
            for k in range(11):
                C[i, j+jj] += A[i, k] * B[k, j+jj]
```
Note that loop `j` is no longer normalized (it has a stride of 5 rather than 1), which means that the nest is no longer a simple nest. As mentioned in the previous section, `Nest` objects always represent simple nests, but `Schedule` objects can represent more complex affine loop nests.

After performing a split, both the outer index and the inner index can be split again. For example,
```python
schedule = nest.create_schedule()
ii = schedule.split(i,4)
iii = schedule.split(i,2)
iiii = schedule.split(ii,2)
```
After the first split, the iteration space has the shape (4, 4, 10, 11). After the second split, the shape becomes (2, 2, 4, 10, 11). Finally, the shape becomes (2, 2, 2, 2, 10, 11) after the third split. The transformed schedule corresponds to the following Python code:
```python
for i in range(0, 16, 8):
    for iii in range(0, 8, 4):
        for ii in range(0, 4, 2):
            for iiii in range(2):
                for j in range(10):
                    for k in range(11):
                        C[i+ii+iii+iiii, j] += A[i+ii+iii+iiii, k] * B[k, j]
```

The split does not necessarily need to divide the dimension size. For example, consider the following code:
```python
schedule = nest.create_schedule()
kk = schedule.split(k, 4)  # original size of dimension k was 11
```
From the iteration space point-of-view, this code splits dimension `k` into three parts of size 4, where the last part is padded with empty (no-op) elements. Before the transformation, the iteration space shape is (16, 10, 11), and after the transformation, the shape is (16, 10, 3, 4) (so, 160 empty elements were added).

In loop form, the transformed iteration space corresponds to the following Python code:
```python
for i in range(16):
    for j in range(10):
        for k in range(0, 11, 4):
            for kk in range(4):
                if k+kk < 11:
                    C[i, j] += A[i, k+kk] * B[k+kk, j]
```
Note that Accera optimizes away costly `if` statements by *unswitching* the loops, which results in code that looks more like this:
```python
for i in range(16):
    for j in range(10):
        for k in range(0, 8, 4):
            for kk in range(4):
                C[i, j] += A[i, k+kk] * B[k+kk, j]
        # loop unswitching: handle the last iteration of the k loop separately
        for kk in range(3):
            C[i, j] += A[i, 8+kk] * B[8+kk, j]
```

#### Meaningless splits
Next, we will describe Accera’s behavior in a few degenerate cases. If the split size equals the dimension size, the transformation simply renames the split dimension. For example,
```python
schedule = nest.create_schedule()
kk = schedule.split(k, 11) # original size of dimension k was 11
```
After the split, the size of `k` becomes 1 and the size of `kk` is `11`. The new shape of the iteration space is (16, 10, 1, 11). The dimension `k` becomes meaningless and therefore the schedule is basically unchanged.

If the split size exceeds the dimension size, Accera will treat it as if the split size doesn't divide the dimension size. This special case is handled by adding empty elements. For example,
```python
schedule = nest.create_schedule()
kk = schedule.split(k, 13)  # original size of dimension k was 11
```
After the split, the size of `k` becomes 1 and the size of `kk`, 13. The new shape of the iteration space is (16, 10, 1, 13), which means that 320 empty elements were added. These empty elements are removed during code generation, which means that the schedule is basically unchanged.

Finally, note that `kk = schedule.split(k, 1)` simply adds a meaningless new dimension `kk` of size 1, and again, the schedule is unchanged.

### Convenience syntax: `tile`
The `tile` transformation is a convenience syntax and does not provide any unique functionality. Consider the following code
```python
schedule = nest.create_schedule()
ii, jj, kk = schedule.tile({
    i: 8,
    j: 2,
    k: 3
})
```
The `tile` transformation above is shorthand for the following sequence of transformations:
```python
ii = schedule.split(i, 8)
jj = schedule.split(j, 2)
kk = schedule.split(k, 3)
```

It will result in a sequence of indices that are ordered as:
```
(i, ii, j, jj, k, kk)
```
In other words, the `tile` transformation takes a tuple of indices and a tuple of sizes, splitting each index by the corresponding size. The indices involved in the split are then ordered such that each of the outer indices (parent index) precedes its inner indices (child index). On the other hand, indices that did not participate in the transformation retain their relative positions.

### `skew`
```python
# Skew dimension i with respect to dimension j.
schedule.skew(i, j)
```

The `skew` transformation is the easiest to explain for a two-dimensional iteration space of shape *(N, M)*. Skewing dimension `i` (the row dimension) with respect to `j` (the column dimension) modifies the iteration space column-by-column: column `j` gets *j* empty elements added to its start and *M-j-1* empty elements to its end. As a result, each column grows from size *N* to size *N+M-1*. Geometrically, the original iteration space elements take the form of a 45-degree parallelogram, embedded within a bounding rectangle of shape *(N+M-1, M)*. The element that used to be at coordinate *(i, j)* moves to coordinate *(i+j, j)*.

Similarly, skewing `j` with respect to `i` adds empty elements at the beginning and end of each row, resulting in an iteration space of shape *(N, N+M-1)*. In higher dimensions, we simply apply the two-dimensional skew transformation independently to each two-dimensional slice along the two specified dimensions.

To demonstrate the importance of this transformation, consider convolving a 10-element vector with a 3-element filter. The loop logic for this operation is defined as follows:
```python
import accera as acc

N = 10  # input size
K = 3  # filter size
M = N - K + 1  # output size = 8

A = acc.Array(role=acc.Array.Role.INPUT, shape=(N,))
B = acc.Array(role=acc.Array.Role.INPUT, shape=(K,))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M,))

nest = acc.Nest(shape=(M, K))
i, j = nest.get_indices()

@nest.iteration_logic
def _():
    C[i] += A[i+j] * B[j]

schedule = nest.create_schedule()
```
`schedule` corresponds to an iteration space of shape (8,3), where the first dimension corresponds to the 8 elements of the output vector. This schedule calculates the outputs one by one: first `C[0]`, then `C[1]`, etc.

Here is the equivalent Python code:
```python
for i in range(8):
    for j in range(3):
        C[i] += A[i+j] * B[j]
```

Now, say that we apply the `skew` transformation as follows:
```python
schedule.skew(i, j)
```
This transformation results in an iteration space of shape (10, 3), where the first dimension corresponds to the 10 elements of the input. This transformed schedule processes the input elements one-by-one: it extracts all the information from `A[0]` (`A[0]` is only used in the calculation of `C[0]`), then moves on to `A[1]` (which contributes to both `C[0]` and `C[1]`), and so on.

In this example, the default schedule achieves memory locality with respect to array `C`, whereas the skewed schedule achieves memory locality with respect to array `A`.

In loop form, the transformed iteration space corresponds to the following Python code:

```python
for i in range(10):
    for j in range(3):
        if (i-j) >= 0 and (i-j) < 8:
            C[i-j] += A[i] * B[j]
```

Behind the scenes, *unswitching* the loops results in code that looks more like this:

```python
# triangle of height 2, width 3
for j in range(1):
    C[0-j] += A[0] * B[j]
for j in range(2):
    C[1-j] += A[1] * B[j]

# rectangle of shape (6, 3)
for i in range(2, 8):
    for j in range(3):
        C[i-j] += A[i] * B[j]

# upside-down triangle of height 2, width 3
for j in range(2):
    C[6+j] += A[8] * B[2-j]
for j in range(1):
    C[7+j] += A[9] * B[2-j]
```

Finally, note that some loops have small sizes that can be replaced by unrolls. To enable the unrolling of these small loops, we can use this optional parameter:

```python
schedule.skew(i, j, unroll_loops_smaller_than=3)
```

This will unroll all loops that are smaller than 3, which include the `range(2)` and `range(1)` loops in the example above.

### `pad`
```python
# Adds empty elements to the beginning of dimension i.
schedule.pad(i, size)
```

The `pad` transformation pads the beginning of dimension `i` with empty elements. This operation is meaningless by itself, but can be useful when used with splitting or fusing.

## Order-invariant schedules and safety
A schedule is *order-invariant* if its underlying logic doesn't depend on the execution order of its iterations. For example, schedules created from a single `Nest` (via `create_schedule()`) are order-invariant. All of the schedules discussed so far have been order-invariant.

A schedule is *safe* if its underlying logic is guaranteed to remain intact regardless of the applied transformations. Not all schedules are safe, but order-invariant schedules are. This is because the transformations introduced in this section only change the execution order of iterations without adding or removing any work.

In [Section 4](<04%20Fusing.md>), we introduce fused schedules, which are not order-invariant, but may still be safe.


<div style="page-break-after: always;"></div>
