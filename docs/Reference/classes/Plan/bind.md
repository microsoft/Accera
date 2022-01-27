[//]: # (Project: Accera)
[//]: # (Version: 1.2)

# Accera 1.2 Reference

## `accera.Plan.bind(indices, grid)`
Only available for targets that can execute a grid of work (such as GPUs). The `bind` function binds dimensions of the iteration space to axes of the target-specific grid (such as `v100.GridUnit.BLOCK_X`, `v100.GridUnit.THREAD_X` on an Nvidia GPU).

## Arguments

argument | description | type/default
--- | --- | ---
`indices` | The iteration-space dimensions to bind. | tuple of `Index`
`grid` | The respective target-specific grid axes to bind with. | tuple of target-specific identifiers

## Examples

Mark the `i`, `j`, and `k` indices to execute on an NVidia V100's `BLOCK_X`, `THREAD_X`, and `THREAD_Y` grid axes respectively.

```python
v100 = acc.Target(Target.Model.NVIDIA_V100)
plan.bind(indices=(i, j, k), grid=(v100.GridUnit.BLOCK_X, v100.GridUnit.THREAD_X, v100.GridUnit.THREAD_Y))
```

<div style="page-break-after: always;"></div>
