[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Accera 1.2.0 Reference

## `accera.ActionPlan.kernelize(unroll_indices[, vectorize_indices])`
A convenience method for a sequence of `unroll` instructions followed by a possible sequence of `vectorize` instructions

## Arguments
argument | description | type/default
--- | --- | ---
`unroll_indices` | The iteration-space dimensions to unroll | tuple of `accera.Index`
`vectorize_indices` | The optional iteration-space dimensions to vectorize | `accera.Index` or tuple of `accera.Index`

## Examples

Unroll `i` and `k`, and then vectorize `j`:

```python
schedule.reorder(i, k, j)
plan = schedule.create_action_plan()
plan.kernelize(unroll_indices=(i, k), vectorize_indices=j)
```

Another example is to Unroll `i`, and then vectorize `j` and `k`:

```python
schedule.reorder(i, j, k)
plan = schedule.create_action_plan()
plan.kernelize(unroll_indices=(i,), vectorize_indices=(j, k))
```

<div style="page-break-after: always;"></div>
