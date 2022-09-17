[//]: # (Project: Accera)
[//]: # (Version: v1.2.9)

# Accera v1.2.9 Reference

## `accera.Plan.parallelize(indices[, pin, policy, max_threads])`

Executes one or more loops in parallel on multiple cores or processors.

Only available for targets with multiple cores or processors.

## Arguments

argument | description | type/default
--- | --- | ---
`indices` | The iteration-space dimensions to run in parallel. To assign multiple threads to an index, first split that index, then parallelize its split indices. <br/> Unsplit indices will be assigned one thread each, split indices will be assigned threads based on the number of split blocks. This is limited by the number of threads supported by the target. | tuple of `accera.Index`
`pin` | Pin the computation to a subset of cores or processors. | tuple of target-specific identifiers
`policy` | The scheduling policy to apply ("dynamic" or "static"). | string. Defaults to "static".
`max_threads` | The maximum number of threads to use when distributing the workload. The actual number of threads used is the lowest value among (a) `max_threads`, (b) the number of threads supported by the target and (c) the number of iterations in the domain as specified by `indices`. | int. Defaults to None.

## Examples

### Parallelize the `i`, `j`, and `k` dimensions using default number of threads:

```python
nest = Nest(shape=(2, 3, 4))
i, j, k = nest.get_indices()
plan.parallelize(indices=(i, j, k)) # This will use 2 x 3 x 4 = 24 threads
```

### Parallelize the `i` dimension after splitting using default number of threads:

```python
nest = Nest(shape=(20,))
schedule = nest.create_schedule()
i = schedule.get_indices()
ii = schedule.split(i, 4)
plan.parallelize(indices=i) # This will use 20 / 4 = 5 threads
```

### Parallelize the `i`, `j`, and `k` dimensions using thread limit:

```python
nest = Nest(shape=(2, 3, 4))
i, j, k = nest.get_indices()
plan.parallelize(indices=(i, j, k), max_threads=4) # This will use 4 threads
```

### Parallelize the `i` dimension with thread limit set higher than the number of iterations:

```python
nest = Nest(shape=(2, 3, 4))
i, j, k = nest.get_indices()
plan.parallelize(indices=i, max_threads=4) # This will use 2 threads since 'i' has only 2 iterations
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


