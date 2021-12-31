[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Case study - Three-matrix multiplication (part 2)
In [part 1](<Three-matrix%20multiplication%20-%20part%201.md>) we described a simple schedule for three-matrix multiplication. Next, we present an alternative schedule. Assume that all of the arrays are defined as in part 1.

## Defining the iteration-space dimensions
For completeness, we repeat the definitions of `schedule0` and `schedule1`:
```python
nest0 = acc.Nest(shape=(256, 256, 32))
i0, j0, k0 = nest0.get_indices()

@nest0.iteration_logic
def _():
    C[i0, j0] += A[i0, k0] * B[k0, j0]

schedule0 = nest0.create_schedule()

nest1 = acc.Nest(shape=(256, 32, 256))
i1, j1, k1 = nest1.get_indices()

@nest1.iteration_logic
def _():
    E[i1, j1] += C[i1, k1] * D[k1, j1]

schedule1 = nest1.create_schedule()
```
Instead of rushing to fuse these two schedules, as we did in part 1, we first split each one of them separately.
```python
n = acc.create_parameters(1)
jj0 = schedule0.split(j0, n)
kk1 = schedule1.split(k1, n)
```
Now, we fuse the two schedules as follows:
```python
schedule0.reorder(i0, j0, jj0, k0) # redundant operation, included for clarity
schedule1.reorder(i1, k1, j1, kk1)
schedule = acc.fuse((schedule0, schedule1), partial=2)
f, i, j, jj0, k0, j1, kk1 = schedule.get_indices()
```
After fusing, we perform additional splits and set the order of the indices:
```python
m, s = acc.create_parameters(2)
ii, jj1 = schedule.tile((i, j1), (m, s))
```
Compare the current schedule to the one presented in part 1. In part 1, we first fused indices `j0` and `k1` into a shared index `j` and then split `j` to create `jj`; here we first split `j0` and `k1` to create `jj0` and `kk1` respectively, and then left these indices unfused. Additionally, here we split index `j1` to create `jj1`.

## Setting the order
We set the index order as follows.
```python
schedule.reorder(i, j, f, j1, ii, k0, jj0, kk1, jj1)
```
Assuming that `m, n, s` are all chosen to be powers of 2, the resulting schedule is equivalent to the following Python code:
```python
for i in range(0, 256, m):
    for j in range(0, 256, n):
        # f = 0
        for ii in range(m):
            for k0 in range(32):
                for jj0 in range(n):
                    C[i+ii, j+jj0] += A[i+ii, k0] * B[k0, j+jj0]
        # f = 1
        for j1 in range(0, 32, s):
            for ii in range(m):
                for kk1 in range(n):
                    for jj1 in range(s):
                        E[i+ii, j1+jj1] += C[i+ii, j+kk1] * D[j+kk1, j1+jj1]
```

## Vectorization and caching
As in part 1, we consider vectorization and caching together. Note that the `jj0` loop performs the equivalent of
```python
C[i+ii, j:j+n] += A[i+ii, k0] * B[k0, j:j+n]
```
and the `jj1` loop performs the equivalent of
```python
E[i+ii, j1:j1+s] += C[i+ii, j+kk1] * D[j+kk1, j1:j1+s]
```
Both of these operations add a scaled row-vector to another row-vector. To vectorize these operations, both `n` and `s` must equal the vector width, which is 8. To ensure that the array elements that participate in each vector operation are contiguous in memory, the caches of `C, B, E, D` must all be row-major (a.k.a., `FIRST_MAJOR`). Assuming that all of the caches are keyed on index `f`, we add the instructions:
```Python
plan = schedule.create_action_plan()
plan.cache(A, index=f)
plan.cache(B, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
plan.cache(C, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
plan.cache(D, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
plan.cache(E, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
```
We can now vectorize the inner-most indices:
```python
plan.vectorize(jj0)
plan.vectorize(jj1)
```
The result is equivalent to the following Python code:
```python
for i in range(0, 256, m):
    for j in range(0, 256, n):
        # cache A[i:i+m, :], B[:, j:j+n], C[i:i+m, j:j+n], D[j:j+n, :], E[i:i+m, :]
        for ii in range(m):
            for k0 in range(32):
                C[i+ii, j:j+n] += A[i+ii, k0] * B[k0, j:j+n] # vectorized index jj0
        for j1 in range(0, 32, s):
            for ii in range(m):
                for kk1 in range(n):
                    E[i+ii, j1:j1+s] += C[i+ii, j+kk1] * D[j+kk1, j1:j1+s] # vectorized index jj1
```
Compare the above to the schedule from part 1. There, the vectorized operations operated on columns of `A`, `C`, and `E`, while here they operate on rows of `B`, `C`, `D`, and `E`.

We have already chosen `n` = `s` = 8 and it remains to set `m`. As in part 1, the number of cached elements is
64 &middot; (`m` + `n`) + `m` &middot; `n`, which in this case equals 72 &middot; `m` + 512. If we set `m`=64, the total number of elements comes out to 5120, or roughly 20KB, which fits in the cache budget.

### TODO

* Try swapping the order of `i` and `j`
* Try swapping `ii` and `k0`.
* Try permuting `j1, ii, kk1`.
* Try forcing `A`'s cache to a specific layout.
* Why are all the caches keyed on `f`?
* Add optimizations from the standard 2-matrix multiplication
* Say more about how the strategy in part 2 compares to the strategy in part 1. How does it depend on the memory layout of the original arrays? How does it depend on the shapes of the original arrays?
