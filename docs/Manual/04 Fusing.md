[//]: # (Project: Accera)
[//]: # (Version: v1.2.1)

# Section 4: Fusing
Multiple schedules can be combined into a single schedule using the `fuse` operation. The fused schedule represents the union of the work in the original schedules. The fused schedule can be transformed using any of the transformations presented in [Section 3](<03%20Schedules.md>).

## Full fusing
```python
import accera as acc

# Fuse three schedules to create a fused schedule
schedule = acc.fuse(schedule0, schedule1, ...)
```

*Full fusing* is the most straightforward form of fusing, where each dimension is fused with the corresponding dimension from the other schedules.

### Full fusing of same-shaped iteration spaces
First, consider the simplest case, where we fuse schedules whose iteration spaces have identical shapes. The fused schedule `schedule` gets a new dimension, called the *fusing dimension*, which did not exist in the original schedules. By default, the fusing dimension is the first dimension in the fused schedule and its size equals the number of schedules that were fused. The first slice along the fusing dimension contains a copy of `schedule0`, the second slice contains a copy `schedule1`, and so on. Since the fusing dimension is the first dimension, the fused schedule is logically equivalent to fully executing `schedule0`, followed by `schedule1`, and so on. To interleave the original schedules, we apply additional transformations to the fused schedule.

As a concrete example, imagine that we want to shift and then scale each of the elements of a matrix, or in other words, perform the equivalent of the Python code:
```python
C = (C + A) * B
```
where all three matrices are 16 by 16.

One way to do this without fusing is to simply write:
```python
A = acc.Array(role=acc.Array.Role.INPUT, shape=(16, 16))
B = acc.Array(role=acc.Array.Role.INPUT, shape=(16, 16))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(16, 16))

# Create nest_simple and schedule_simple
nest_simple = acc.Nest(shape=(16, 16))
i, j = nest_simple.get_indices()

@nest_simple.iteration_logic
def _():
    C[i,j] = (C[i,j] + A[i,j]) * B[i,j]

schedule_simple = nest_simple.create_schedule()
```
Note that each iteration in `schedule_simple` operates on all three arrays at once. Imagine that there is some computational advantage to operating on only two arrays at a time. For example, say that operating on all three arrays simultaneously creates excessive pressure on the computer's memory caches, which could hurt performance.

Therefore, we may want to first compute `C += A` and only then compute `C *= B`. Better yet, we may want to compute `C` in 4&times;4 blocks: first computing `C[0:4, 0:4] += A[0:4, 0:4]`; next computing `C[0:4, 0:4] *= B[0:4, 0:4]`; then moving on to the next block and computing `C[4:8, 0:4] += A[4:8, 0:4]`, and so on. Fusing gives us the flexibility to explore all of these possibilities, and more.

First, we define two separate nests, one for the logic `C += A` and one for the logic `C *= B`, and obtain their corresponding default schedules:
```python
# Create nest0 and schedule0
nest0 = acc.Nest(shape=(16, 16))
i0, j0 = nest0.get_indices()

@nest0.iteration_logic
def _():
    C[i0, j0] += A[i0, j0]

schedule0 = nest0.create_schedule()

# Create nest1 and schedule1
nest1 = acc.Nest(shape=(16, 16))
i1, j1 = nest1.get_indices()

@nest1.iteration_logic
def _():
    C[i1, j1] *= B[i1, j1]

schedule1 = nest1.create_schedule()
```

Before fusing, both `schedule0` and `schedule1` have a shape of (16, 16). Next we fuse them:
```python
# Create a fused schedule
schedule = acc.fuse(schedule0, schedule1)
f, i, j = schedule.get_indices()
```
Fusing does not change `schedule0` or `schedule1` but rather creates a new fused schedule named `schedule`, whose shape is (2, 16, 16). The first dimension in `schedule` is the so-called fusing dimension `f`, its slice (0, \*, \*) contains a copy of `schedule0`, and its slice (1, \*, \*) contains a copy of `schedule1`.

In loop form, `schedule` is now equivalent to the following Python code:
```python
# f = 0
for i in range(16):
    for j in range(16):
        C[i, j] += A[i, j]
# f = 1
for i in range(16):
    for j in range(16):
        C[i, j] *= B[i, j]
```
Not much has happened yet: executing `schedule` as-is is equivalent to executing `schedule0` and then executing `schedule1`. However, this can be changed by transforming the fused schedule. For example, we can recover `schedule_simple` by reordering the indices as follows:
```python
schedule.reorder(i, j, f)
```
The fusing dimension moves from the first position to the last position. Now, `schedule` is equivalent to the following Python code:
```python
for i in range(16):
    for j in range(16):
        # f = 0
        C[i, j] += A[i, j]
        # f = 1
        C[i, j] *= B[i, j]
```

We also discussed computing the output block-by-block: first computing `C[0:4, 0:4] += A[0:4, 0:4]`, then computing `C[0:4, 0:4] *= B[0:4, 0:4]`, and so on. This can be accomplished with the following sequence of transformations
```python
ii, jj = schedule.tile((i, j), (4, 4))
schedule.reorder(i, j, f, ii, jj)
```
The resulting `schedule` is equivalent to the Python code:
```python
for i in range(0, 16, 4):
    for j in range(0, 16, 4):
        # f = 0
        for ii in range(4):
            for jj in range(4):
                C[i+ii, j+jj] += A[i+ii, j+jj]
        # f = 1
        for ii in range(4):
            for jj in range(4):
                C[i+ii, j+jj] *= B[i+ii, j+jj]
```

### Constraint 1: the fusing dimension is executed sequentially
The fusing dimension has a special constraint, which does not apply to other dimensions. Specifically, the fusing dimension cannot be parallelized, vectorized, or tensorized (see [Section 7](<07%20Plans%20-%20Vectorization%20and%20Parallelization.md>) ) and it must be executed sequentially. This constraint enables the safety guarantee discussed below.

### Safety
The fused schedule (before applying any subsequent transformations) is always logically equivalent to executing the original schedules one-by-one. However, is it safe? Recall that a schedule is considered safe if its underlying logic is guaranteed not to change, regardless of how we transform it. The safety of a fully fused schedule depends on the circumstances:

Accera guarantees that the order of the fused schedules is preserved *for each value of the fused dimensions*, regardless of how the fused schedule is transformed. For example, in the example above, the fused dimensions are `i` and `j`. Therefore, for any concrete value of `i` and `j`, the corresponding operation from `schedule0` is guaranteed to execute before the corresponding operation from `schedule1`, regardless of how the fused schedule is transformed. More specifically, for each `i` and `j`, the operation `C[i, j] += A[i, j]` is guaranteed to execute before the operation `C[i, j] *= B[i, j]`, no matter how we transform the fused schedule. Since those are the only operations that touch `C[i,j]`, the Accera guarantee is sufficient and we conclude that fused schedule is safe. With this guarantee, the programmer can apply any sequence of transformations without worrying about the resulting implementation's correctness.

However, note that not every fusing operation creates a safe schedule. For example, imagine that we had fused `schedule0` and `schedule1` differently:
```python
# Reorder schedule1 before fusing
schedule1.reorder(j1, i1)
# Fuse schedule0 with the reordered schedule1
schedule_t = acc.fuse(schedule0, schedule1)
f, a, b = schedule_t.get_indices()
```
In this unnatural example, `i0` and `j1` are fused and named `a`, and `i1` and `j0` are fused and named `b`. As before, Accera guarantees that for each value of `a` and `b` the operation `C[a, b] += A[a, b]` is executed before `C[b, a] *= B[b, a]`. As noted above, the fusing operation itself preserves logical equivalence, but if we proceed to transform the fused schedule as follows,
```python
schedule_t.reorder(a, b, f)
```
the logic actually changes. To see this, note that the resulting schedule is equivalent to the following Python code:
```python
for a in range(16):
    for b in range(16):
        C[a, b] += A[a, b]
        C[b, a] *= B[b, a]
```
In particular, this code sets `C[1,0]` to `C[1,0] * B[1,0] + A[1,0]`, whereas the original fused logic set `C[1,0]` to `(C[1,0] + A[1,0]) * B[1,0] `. We conclude that `schedule_t` is certainly not safe. If the programmer creates an unsafe schedule, they take upon themselves the responsibility of maintaining logical equivalence.

### Fusing iteration spaces with different shapes
If the iterations spaces have different shapes, Accera matches their shapes by padding them appropriately with empty cells.

## Partial fusing
In many cases, instead of fusing all of the dimensions, we only need to fuse some of the dimensions, leaving the rest unfused. To fuse the first *s* dimensions, we use the syntax
```python
# Fuse the first s dimensions of three schedules
schedule = acc.fuse((schedule0, schedule1, ...), partial=s)
```
The order of the dimensions in the fused schedule is as follows: first the fusing dimension `f`, then the *s* fused dimensions, then the unfused dimensions of `schedule0`, `schedule1`, etc.

We can easily calculate the number of dimensions in the fused schedule. For example, if we fuse the first *s* dimensions of a *d0*-dimensional space `schedule0` and a *d1*-dimensional space `schedule1`, the fused iteration space will have *s* fused dimensions, *d0 + d1 - 2s* unfused dimensions, and the special fusing dimension `f`, for a total of *d0 + d1 - s + 1* dimensions.

As before, the `fuse` operation uses padding to ensure that the fused iteration space is not jagged in any direction. For example, say that `schedule0` is 4-dimensional, `schedule1` is 3-dimensional, and we partially fuse their first 2 dimensions:
```python
schedule = acc.fuse((schedule0, schedule1), partial=2)
f, i, j, k, l, m = schedule.get_indices()
```
The first dimension is the fusing dimensions `f`, whose size is 2. Next come the fused dimensions `i` and `j`. Then, `k` and `l` are the two unfused dimensions from `schedule0` and `m` is the unfused dimension from `schedule1`. The slice (0, \*, \*, \*, \*, 0) contains a copy of `schedule0`, the slice (1, \*, \*, 0, 0, \*) contains a copy of `schedule1`, and the rest of `schedule` is padded with empty elements. Note that full fusing is a special case of partial fusing, where `s` is the larger of the dimensions of `schedule0` and `schedule1`.

### Constraint 2: the fusing dimension always precedes unfused dimensions
Partial fusing introduces a second constraint on the fusing dimension. Namely, the fusing dimension must precede all of the unfused dimensions in the dimension order. This constraint also applies to dimensions that are derived from the fusing dimension and from the unfused dimensions via splitting.

### Safety
The safety guarantees for partial-fusing are a natural extension of the guarantees for full fusing. Accera guarantees that the order of the fused schedules is preserved *for each value of the fused dimensions*, regardless of how the fused schedule is transformed. In other words, for each concrete value of the fused dimensions, all the corresponding work in `schedule0` (across all of its unfused dimensions) is performed before any of the corresponding work in `schedule1` (across all of its unfused dimensions), and this holds no matter how we transform the fused schedule. When fusing, the programmer needs to consider whether this property implies safety - we will show how this can be done in each of the examples below.

### Partial fusing example: fully-connected neural layer with activation
Consider applying an element-wise operation, such as the ReLU function from the field of AI, to the result of a matrix-matrix multiplication. In the language of neural networks, this is called a fully connected layer with a ReLU activation. The function `relu(x)` is simply `max(x,0)`.

Specifically, imagine that we have an element-wise operator `relu` and we want to implement the equivalent of the Python code:
```python
C = relu(C + A @ B)
```
where `A` has a shape of (16, 11), `B` has a shape of (11, 10), and `C` has a shape of (16, 10). We define two nests, one for `C += A @ B` and the other for `C = relu(C)`, and obtain their corresponding default schedules:
```python
# Create nest0 and schedule0
nest0 = acc.Nest(shape=(16, 10, 11))
i0, j0, k0 = nest0.get_indices()

@nest0.iteration_logic
def _():
    C[i0, j0] += A[i0, k0] * B[k0, j0]

schedule0 = nest0.create_schedule()

# Create nest1 and schedule1
nest1 = acc.Nest(shape=(16, 10))
i1, j1 = nest1.get_indices()

@nest1.iteration_logic
def _():
    C[i1, j1] = acc.max(C[i1, j1], 0)

schedule1 = nest1.create_schedule()
```
In both `schedule0` and `schedule1`, the first dimension corresponds to the rows of `C` and the second dimension corresponds to the columns of `C`. In addition, `schedule0` has a third dimension that `schedule1` does not have. Therefore, we fuse the first two dimensions of the iteration spaces and leave the third dimension of `schedule0` unfused.
```python
schedule = acc.fuse((schedule0, schedule1), partial=2)
f, i, j, k0 = schedule.get_indices()
```
The fused iteration space `schedule` has a shape of (2, 16, 10, 11), its slice (0, \*, \*, \*) contains a copy of `schedule0`, its slice (1, \*, \*, 0) contains a copy of `schedule1`, and the rest of its elements are filled with padding. Note that the code above overwrites the index `k0`: originally, `k0` was an index of `schedule0`, but now it is the corresponding unfused index in `schedule`. This is a stylistic choice and we could have chosen a different name.

Is `schedule` safe? Recall that Accera guarantees that for each value of `i` and `j`, the corresponding work in `schedule0` (which is `C[i,j] += A[i,k0] * B[k0,j]` for all values of `k0`) is executed before the corresponding work in `schedule1` (which is `C[i,j] = max(C[i,j], 0)`), and this holds regardless of how the fused schedule is transformed. Since these are the only operations that touch `C[i,j]` and the `ReLU` operation is always executed last, this confirms that `schedule` is safe, and from this point forward we can focus all of our attention on optimizing performance without worrying about correctness.

Executing `schedule` as-is is equivalent to executing `schedule0` in its entirety and then executing `schedule1`. If we want to interleave the two schedules and perform `relu` immediately after calculating each element of the matrix product, we reorder the dimensions such that `i` and `j` preceded `f`:
```python
schedule.reorder(i, j, f, k0)
```
The resulting schedule is now equivalent to the following Python code:

```python
for i in range(16):
    for j in range(10):
        # f = 0
        for k0 in range(11):
                C[i,j] += A[i,k0] * B[k0,j]
        # f = 1
        C[i,j] = max(C[i,j], 0)
```

### Partial fusing example: multiplying three matrices
Consider fusing two matrix-matrix multiplications, to get matrix-matrix-matrix multiplication. Specifically, say that our goal is to calculate the equivalent of the Python code:
```python
E += A @ B @ D
```
where `A` is of shape (16, 11), `B` is of shape (11, 10), `D` is of shape (10, 7), and `E` is of shape (16, 7).

We start by defining the arrays. In addition to `A`, `B`, `D`, and `E`, we define a temporary array `C` to store the intermediate result of `A@B`.
```python
A = acc.Array(role=acc.Array.Role.INPUT, shape=(16, 11))
B = acc.Array(role=acc.Array.Role.INPUT, shape=(11, 10))
C = acc.Array(role=acc.Array.Role.TEMP, shape=(16, 10))
D = acc.Array(role=acc.Array.Role.INPUT, shape=(10, 7))
E = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(16, 7))
```
Note that `C` is declared with a role of `TEMP`. Recall that temporary arrays are mutable and initialized with zeros. Moreover, temporary arrays are logical objects, which may not actually exist in memory during the entire computation.

Next, define a simple nest to compute `C += A @ B` and another simple nest to compute `E += C @ D`.
```python
# Create nest0 and schedule0 for C = A @ B
nest0 = acc.Nest(shape=(16, 10, 11))
i0, j0, k0 = nest0.get_indices()

@nest0.iteration_logic
def _():
    C[i0, j0] += A[i0, k0] * B[k0, j0]

schedule0 = nest0.create_schedule()

# Create nest1 and schedule1 E += C @ D
nest1 = acc.Nest(shape=(16, 7, 10))
i1, j1, k1 = nest1.get_indices()

@nest1.iteration_logic
def _():
    E[i1, j1] += C[i1, k1] * D[k1, j1]

schedule1 = nest1.create_schedule()
```
The temporary array `C` is used to store the output of `schedule0`, and is then used again as one of the inputs of `schedule1`. Dimensions `i0` and `j0` correspond to the rows and columns of `C` in `schedule0`. Dimensions `i1` and `k1` correspond to the rows and columns of `C` in `schedule1`. Therefore, we fuse `i0` with `i1` and `j0` with `k1`. To do this, we need to correctly line-up the dimensions of the two iteration spaces and perform partial fusing.
```python
schedule1.reorder(i1, k1, j1)
schedule = acc.fuse((schedule0, schedule1), partial=2)
f, i, j, k0, j1 = schedule.get_indices()
```
The fused iteration space has a shape of (2, 16, 10, 11, 7): `f` is the fusing dimension, `i` is the result of fusing `i0` and `i1`, `j` is the result of fusing `j0` and `k1`, `k0` is the unfused dimension from `schedule0`, and `j1` is the unfused dimension from `schedule1`. The slice (0, \*, \*, \*, 0) contains a copy of `schedule0` and the slice (1, \*, \*, 0, \*) contains a copy of `schedule1`. The rest of the iteration space is padded with empty elements.

Is `schedule` safe? Again, recall that Accera guarantees that for each value of `i` and `j`, all of the corresponding work in `schedule0` (which is `C[i, j] += A[i, k0] * B[k0, j]` for all values of `k0`) is executed before any of the corresponding work from `schedule1` (which is `E[i, j1] += C[i, j] * D[j, j1]` for all values of `j1`). In other words, each element of `C` is fully computed before it is used. This confirms that `schedule` is safe.

Initially, the fused schedule is equivalent to the following Python code:
```python
# f = 0
for i in range(0, 16):
    for j in range(0, 10):
        for k0 in range(11):
            C[i, j] += A[i, k0] * B[k0, j]
# f = 1
for i in range(0, 16):
    for j in range(0, 10):
        for j1 in range(7):
            E[i, j1] += C[i, j] * D[j, j1]
```

We can now manipulate the fused schedule in various ways. For example, we can do all the work to create one element of `C` and then immediately do all the work that uses that element, before moving on to the next element.
```python
schedule.reorder(i, j, f, k0, j1)
```
This schedule is equivalent to the following Python code:
```python
for i in range(0, 16):
    for j in range(0, 10):
        # f = 0, create C[i, j]
        for k0 in range(11):
            C[i, j] += A[i, k0] * B[k0, j]
        # f = 1, use C[i, j]
        for j1 in range(7):
            E[i, j1] += C[i, j] * D[j, j1]
```

The advantage of this schedule is that only one element of `C` is active at any time in the computation. Accera can reuse the same memory location to store the active element of `C`, instead of storing all of `C` in physical memory,

Similarly, we can compute a 4&times;2 block of `C`, do all the work that uses that block, and then move on to the next block:
```python
ii, jj = schedule.tile((i, j), (4, 2))
schedule.reorder(i, j, f, ii, jj, k0, j1)
```
This schedule is equivalent to the following:
```python
for i in range(0, 16, 4):
    for j in range(0, 10, 2):
        # f = 0
        for ii in range(4):
            for jj in range(2):
                for k0 in range(11):
                    C[i+ii, j+jj] += A[i+ii, k0] * B[k0, j+jj]
        # f = 1
        for ii in range(4):
            for jj in range(2):
                for j1 in range(7):
                    E[i+ii, j1] += C[i+ii, j+jj] * D[j+jj, j1]
```

<!-- TODO: A more in-depth analysis of three-matrix multiplication can be found in [this case study](<../Case%20Studies/Three-matrix%20multiplication%20-%20part%201.md>).
-->


<div style="page-break-after: always;"></div>
