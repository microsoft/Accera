[//]: # (Project: Accera)
[//]: # (Version: v1.2.12)

# Section 5: Targets
Accera is a cross compiler, which means that it can generate executable code for different target platforms. A target is described using the `Target` class. Accera already supports many different targets, for example:
```python
import accera as acc

corei9 = acc.Target(Target.Model.INTEL_7960X, num_threads=44)
```
or
```python
v100 = acc.Target(Target.Model.NVIDIA_V100)
```

We can also define custom targets:
```python
my_target = acc.Target(name="Custom processor", category=acc.Target.Category.CPU, architecture=acc.Target.Architecture.X86_64, family="Broadwell", extensions=["MMX", "SSE", "SSE2", "SSE3", "SSSE3", "SSE4", "SSE4.1", "SSE4.2", "AVX", "AVX2", "FMA3"], num_cores=22, num_threads=44, frequency_GHz=3.2, turbo_frequency_GHz=3.8, cache_sizes=[32, 256, 56320], cache_lines=[64, 64, 64])
```

One benefit of targets is that they provide a standard way of accessing useful constants. For example, we may want to split an iteration space dimension by the number of elements that fit in a vector register.
```python
schedule.split(i, size=corei9.vector_bytes/4)
```
We may tile the iteration space for GPU targets based on input shapes and available resources like shared memory. If you are not sure of what to use, try starting with the default:
```python
# find block_x and block_y in powers of two, such that block_x*block_y=v100.default_block_size.
block_x = pow(2, math.log2(v100.default_block_size)//2)
block_y = v100.default_block_size // block_x
ii, jj = schedule.tile({
    i: block_x,
    j: block_y
})
```

<div style="page-break-after: always;"></div>
