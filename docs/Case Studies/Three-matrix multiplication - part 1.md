[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Case study - Three-matrix multiplication (part 1)
In [Section 4 of the Accera manual](../Manual/04 Fusing.md) we introduced fusing and showed how to fuse two matrix-matrix multiplications to get a three-matrix multiplication operation, the equivalent of `E += A @ B @ D` in Python. That example used a temporary array `C`, which held the results of `A @ B`, and we briefly discussed the strategy of computing `C` block-by-block, to avoid storing all of `C` in memory. We now revisit that example with a few changes.

First, assume that `A`, `D`, and `E` have a shape of (256, 32), `B` has a shape of (32, 256), and `C` has a shape of (256,256). Setting all of the dimension sizes to be powers of 2 simplifies our exposition. Moreover, the advantages of fusing are accentuated when the temporary array `C` is significantly larger than the others.

## Target hardware characteristics
For concreteness, assume that the target hardware has the following characteristics:
* It is a single-core CPU.
* It supports vector instructions with a width of 8 elements.
* It has 24KB of hardware cache memory available for us to use.

## Basic setup
The example begins as follows:
```python
import accera as acc

# Define the arrays
A = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 32))
B = acc.Array(role=acc.Array.Role.INPUT, shape=(32, 256))
C = acc.Array(role=acc.Array.Role.TEMP, shape=(256, 256))
D = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 32))
E = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(256, 32))

# Create nest0 and schedule0 for C = A @ B
nest0 = acc.Nest(shape=(256, 256, 32))
i0, j0, k0 = nest0.get_indices()

@nest0.iteration_logic
def _():
    C[i0, j0] += A[i0, k0] * B[k0, j0]

schedule0 = nest0.create_schedule()

# Create nest1 and schedule1 E += C @ D
nest1 = acc.Nest(shape=(256, 32, 256))
i1, j1, k1 = nest1.get_indices()

@nest1.iteration_logic
def _():
    E[i1, j1] += C[i1, k1] * D[k1, j1]

schedule1 = nest1.create_schedule()
```

Indices `i0` and `i1` both iterate over the rows of `C`, while indices `j0` and `k1` both iterate over the columns of `C`. Therefore, we partially fuse `i0` with `i1` and `j0` with `k1`:
```python
schedule0.reorder(i0, j0, k0) # redundant operation, included for clarity
schedule1.reorder(i1, k1, j1)
schedule = acc.fuse((schedule0, schedule1), partial=2)
f, i, j, k0, j1 = schedule.get_indices()
```

The index `f` corresponds to the fusing dimension, which ensures that each element of `C` is fully calculated before it is used. `i` and `j` correspond to the rows and columns of `C` respectively. `k0` corresponds to iterations that create the elements of `C` whereas `j1` corresponds to operations that use the elements of `C`. To create and then use `C` block-by-block, we tile dimensions `i` and `j` according to the desired block shape. Instead of choosing the block shape now, we use parameters `m` and `n`, which we set later on.
```python
m, n = acc.create_parameters(2)
ii, jj = schedule.tile((i,j), (m,n))
```

## Index order
Next, we order the iteration-space dimensions as follows:
```python
schedule.reorder(i, j, f, jj, k0, j1, ii)
```
Assuming that `m, n` are chosen to be powers of 2, the resulting schedule is equivalent to the following Python code:
```python
for i in range(0, 256, m):
    for j in range(0, 256, n):
        # f = 0, C[i:i+m, j:j+n] = A[i:i+m, :] @ B[:, j:j+n]
        for jj in range(n):
            for k0 in range(32):
                for ii in range(m):
                    C[i+ii, j+jj] += A[i+ii, k0] * B[k0, j+jj]
        # f = 1, E[i:i+m, :] += C[i:i+m, j:j+n] @ D[j:j+n, :]
        for jj in range(n):
            for j1 in range(32):
                for ii in range(m):
                    E[i+ii, j1] += C[i+ii, j+jj] * D[j+jj, j1]
```

## Vectorization and caching
We consider vectorization and caching together because vectorization requires a particular data layout, which can be provided by choosing the correct cache attributes. Note that the top `ii` loop performs the equivalent of
```python
C[i:i+m, j+jj] += A[i:i+m, k0] * B[k0, j+jj]
```
while the bottom `ii` loop performs the equivalent of
```python
E[i:i+m, j1] += C[i:i+m, j+jj] * D[j+jj, j1]
```
Both of these operations add a scaled column-vector to another column-vector. To vectorize these operations, `m` must equal the vector width, which in our case is 8. We must also ensure that the array elements that participate in each vector operation are contiguous in memory, by creating column-major (a.k.a., `LAST_MAJOR`) caches for `C`, `A`, and `E`. Assuming that all of the caches are keyed on index `f`, we add the instructions:
```Python
plan = schedule.create_action_plan()
plan.cache(A, index=f, layout=acc.Array.Layout.LAST_MAJOR)
plan.cache(B, index=f)
plan.cache(C, index=f, layout=acc.Array.Layout.LAST_MAJOR)
plan.cache(D, index=f)
plan.cache(E, index=f, layout=acc.Array.Layout.LAST_MAJOR)
```
We can now vectorize the inner-most index:
```python
plan.vectorize(ii)
```
The result is equivalent to the following Python code:
```python
for i in range(0, 256, m):
    for j in range(0, 256, n):
        # cache A[i:i+m, :], B[:, j:j+n], C[i:i+m, j:j+n], D[j:j+n, :], E[i:i+m, :]
        for jj in range(n):
            for k0 in range(32):
                C[i:i+m, j+jj] += A[i:i+m, k0] * B[k0, j+jj] # vectorized index ii
        for jj in range(n):
            for j1 in range(32):
                E[i:i+m, j1] += C[i:i+m, j+jj] * D[j+jj, j1] # vectorized index ii
```
We have already chosen `m`=8 and it remains to set `n`. The number of cached elements is as follows:

array | cached elements
------|----------------
`A`   | 32 &middot; `m` = 256
`B`   | 32 &middot; `n`
`C`   | `m` &middot; `n` = 8 &middot; `n`
`D`   | 32 &middot; `n`
`E`   | 32 &middot; `m` = 256
total | 64 &middot; (`m` + `n`) + `m` &middot; `n` = 512 + 72 &middot; `n`

If we set `n`=64, the total number of elements comes out to 5120, or roughly 20KB, which fits in the specified cache budget. Alternatively, we could empirically tune `n` by generating a grid of possible values and measuring their speed on the target platform.

## TODO
* Try swapping the order of `i` and `j`
* Try all permutations of `jj, k0, j1`
* Why are all the caches keyed on `f`?
* Try forcing the caches of `B` and `E` to specific layouts.
* Add optimizations from the standard 2-matrix multiplication
