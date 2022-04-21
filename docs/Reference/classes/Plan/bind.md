[//]: # (Project: Accera)
[//]: # (Version: v1.2.3)

# Accera v1.2.3 Reference

## `accera.Plan.bind(mapping)`
Only available for targets that can execute a grid of work (such as GPUs). The `bind` function binds dimensions of the iteration space to axes of the target-specific grid (such as `v100.GridUnit.BLOCK_X`, `v100.GridUnit.THREAD_X` on Nvidia GPU).

## Arguments

argument | description | type/default
--- | --- | ---
`mapping` | Mapping of indices to GPU thread or block identifiers. | dict of `Index` to target-specific identifiers

## Examples

Mark the `i`, `j`, and `k` indices to execute on NVidia V100's `BLOCK_X`, `THREAD_X`, and `THREAD_Y` grid axes, respectively.

```python
v100 = acc.Target(Target.Model.NVIDIA_V100)
plan.bind({
    i: v100.GridUnit.BLOCK_X,
    j: v100.GridUnit.THREAD_X,
    k: v100.GridUnit.THREAD_Y
})
```

<div style="page-break-after: always;"></div>


