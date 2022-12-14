[//]: # (Project: Accera)
[//]: # (Version: v1.2.13)

# Accera v1.2.13 Reference

## `accera.Plan.bind(mapping)`
Only available for targets that can execute a grid of work (such as GPUs). The `bind` function binds dimensions of the iteration space to axes of the target-specific grid (such as `v100.GridUnit.BLOCK_X`, `v100.GridUnit.THREAD_X` or `v100.GridUnit.WARP_X` on an Nvidia GPU).

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

In some cases, e.g. with tensorization where it might be non-trivial to assign threads to their respective data, it might be simpler to bind iteration space indices to warps (Nvidia) or waves (AMD) in the x and y dimensions rather than threads. This also abstracts the computation at a level higher than individual threads where, instead of each thread performing calculation independently we consider a group of threads (warp) working to solve a bigger computational problem collaboratively (as we often see in warp synchronous primitives like the CUDA's WMMA api). For example,

```python
v100 = acc.Target(Target.Model.NVIDIA_V100)
plan.bind({
    i: v100.GridUnit.BLOCK_X,
    j: v100.GridUnit.BLOCK_Y,
    ii: v100.GridUnit.WARP_Y,
    jj: v100.GridUnit.WARP_X
})
```

in this case, we assign a *warp/wave* of threads to each unique combination of the (`ii`, `jj`) in the iteration space. The spatial arrangement of the warps to their data is defined by the ranges assigned to these individual indices. For example, if both `ii` and `jj` are ranges [0, 32) with step size of 16, we will have a total of 4 warps (2 in the x-dimension and 2 in the y-dimension) covering a 32x32 data region.

<div style="page-break-after: always;"></div>


