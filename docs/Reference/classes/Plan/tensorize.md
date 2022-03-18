[//]: # (Project: Accera)
[//]: # (Version: v1.2.1)

# Accera v1.2.1 Reference

## `accera.Plan.tensorize(indices)`
Only available for targets that have native matrix multiplication instruction (tensor core) support. Marks the dimensions of the iteration-space for tensorization. Only perfectly nested loops of the following form can be tensorized:


```python
for i in range(M):
    for k in range(N):
        for j in range(K):
            C[i, j] += A[i, k] * B[k, j]
```

## Arguments

argument | description | type/default
--- | --- | ---
`indices` | The iteration space dimensions to tensorize. | tuple of `accera.Index`

## Examples

Mark the dimensions `ii`, `jj`, and `kk` for tensorization execution:

```python
plan.tensorize(indices=(ii,jj,kk))
```

<div style="page-break-after: always;"></div>
