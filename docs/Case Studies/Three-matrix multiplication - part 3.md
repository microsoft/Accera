[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Case study - Three-matrix multiplication (part 3)
We now present a third scheduling strategy for three-matrix multiplication, which is different from the ones in [part 1](<Three-matrix%20multiplication%20-%20part%201.md>) and [part 2](<Three-matrix%20multiplication%20-%20part%202.md>).

## Defining the iteration-space dimensions
Repeat the construction from [part 2](<Three-matrix%20multiplication%20-%20part%202.md>), up until the point where the two schedules are fused:
```python
schedule = acc.fuse((schedule0, schedule1), partial=2)
f, i, j, jj0, k0, j1, kk1 = schedule.get_indices()
```
From here, the two scheduling strategies diverge. We split the fused schedule and order its indices as follows:
```python
m, t = acc.create_parameters(2)
ii, kk0 = schedule.tile((i, k0), (m, t))
schedule.reorder(i, j, f, ii, jj0, k0, kk0, j1, kk1)
```
Assuming that `m, n, t` are all chosen to be powers of 2, the resulting schedule is equivalent to the following Python code:
```python
for i in range(0, 256, m):
    for j in range(0, 256, n):
        # f = 0
        for ii in range(m):
            for jj0 in range(n):
                for k0 in range(0, 32, t):
                    for kk0 in range(t):
                        C[i+ii, j+jj0] += A[i+ii, k0+kk0] * B[k0+kk0, j+jj0]
        # f = 1
        for ii in range(m):
            for j1 in range(32):
                for kk1 in range(n):
                    E[i+ii, j1] += C[i+ii, j+kk1] * D[j+kk1, j1]
```

## Vectorization and caching
Once again, we consider vectorization and caching together. Note that the `kk0` loop performs the equivalent of
```python
C[i+ii, j+jj0] += A[i+ii, k0:k0+t] * B[k0:k0+t, j+jj0]
```
and the `kk1` loop performs the equivalent of
```python
E[i+ii, j1] += C[i+ii, j:j+n] * D[j:j+n, j1]
```
Both of these operations perform a dot product and then add the scalar result to another scalar. To vectorize these operation, both `n` and `t` must equal the vector width, which is 8. To ensure that the array elements that participate in each vector operation are contiguous in memory, the caches of `A` and `C` must be row-major (a.k.a., `FIRST_MAJOR`) while the caches of `B` and `D` must be column-major (a.k.a., `LAST_MAJOR`). Assuming that all of the caches are keyed on index `f`, we add the instructions:
```Python
plan = schedule.create_action_plan()
plan.cache(A, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
plan.cache(B, index=f, layout=acc.Array.Layout.LAST_MAJOR)
plan.cache(C, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
plan.cache(D, index=f, layout=acc.Array.Layout.LAST_MAJOR)
plan.cache(E, index=f)
```
We now vectorize the inner-most indices:
```python
plan.vectorize(kk0)
plan.vectorize(kk1)
```
The result is equivalent to the following Python code:
```python
for i in range(0, 256, m):
    for j in range(0, 256, n):
        # cache A[i:i+m, :], B[:, j:j+n], C[i:i+m, j:j+n], D[j:j+n, :], E[i:i+m, :]
        for ii in range(m):
            for jj0 in range(n):
                for k0 in range(0, 32, t):
                    C[i+ii, j+jj0] += A[i+ii, k0:k0+t] * B[k0:k0+t, j+jj0] # vectorized index kk0
        for ii in range(m):
            for j1 in range(32):
                E[i+ii, j1] += C[i+ii, j:j+n] * D[j:j+n, j1] # vectorized index kk1
```
Compare the above to the schedules from parts 1 and 2. Here, the vectorized operations operate on rows of `A` and `C` and on columns of `B` and `D`. Moreover, the vectorized operation is a dot product, rather than a scaled vector addition.

We have already chosen `n` = `t` = 8 and it remains to set `m`. The number of cached elements is identical to our calculation in part 2. Therefore, if we set `m`=64, the total number of elements again fits in our cache budget.

### TODO
* Try swapping the order of `i` and `j`
* Try all permutations of `ii, jj0, k0`.
* Try swapping `ii1` and `j1`.
* Try forcing `E`'s cache to a specific layout.
* Why are all the caches keyed on `f`?
* Add optimizations from the standard 2-matrix multiplication
* How to choose between the different schedules proposed on parts 1,2,3? Maybe just emit all of them and measure their performance empirically.

## Summary
We presented three different schedule strategies, each with multiple variants. It is hard to guess which one will be the best.

The first two schedules reduce the computation to a micro-kernel that performs scaled vector addition (on either row or column vectors), while the third reduces the computation to a dot-product. The two types of micro-kernels may have different computational requirements on each target platform.
