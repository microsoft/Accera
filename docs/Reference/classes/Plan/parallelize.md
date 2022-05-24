[//]: # (Project: Accera)
[//]: # (Version: v1.2.5)

# Accera v1.2.5 Reference

## `accera.Plan.parallelize(indices[, pin, policy])`

Executes one or more loops in parallel on multiple cores or processors.

Only available for targets with multiple cores or processors.

## Arguments

argument | description | type/default
--- | --- | ---
`indices` | The iteration-space dimensions to run in parallel. To assign multiple threads to an index, first split that index, then parallelize its split indices. <br/> Unsplit indices will be assigned one thread each, split indices will be assigned threads based on the number of split blocks. This is limited by the number of threads supported by the target. | tuple of `accera.Index`
`pin` | Pin the computation to a subset of cores or processors. | tuple of target-specific identifiers
`policy` | The scheduling policy to apply ("dynamic" or "static"). | string. Defaults to "static".

## Examples

Parallelize the `i`, `j`, and `k` dimensions using 3 threads:

```python
plan.parallelize(indices=(i, j, k))
```

Parallelize the `i` dimension using 4 threads:

```python
N = 1024 # shape of the i dimension
num_threads = 4

# divide the shape by num_threads to get the block size per thread
block_size = N//num_threads

ii = schedule.split(i, size=block_size)
plan.parallelize(indices=i)
```

__Not yet implemented:__ Parallelize the `i`, `j`, and `k` dimensions by pinning them to specific cores on an Intel Xeon E5:

```python
plan.parallelize(indices=(i, j, k), pin=(xeonE5.cores[0], xeonE5.cores[1], xeonE5.cores[2]))
```

Apply a dynamic scheduling policy, which uses a queue to partition the work across multiple cores:

```python
plan.parallelize(indices=(i, j, k), policy="dynamic")
```

<div style="page-break-after: always;"></div>


