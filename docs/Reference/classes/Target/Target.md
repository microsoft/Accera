[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Accera 1.2.0 Reference

## `accera.Target([architecture, cache_lines, cache_sizes, category, extensions, family, frequency_GHz, model, name, num_cores, num_threads, turbo_frequency_GHz])`

Defines the capabilities of a target processor.

## Arguments
argument | description | type/default
--- | --- | ---
`architecture` | The processor architecture | accera.Target.Architecture
`cache_lines` | Cache lines | list of positive integers
`cache_sizes` | Cache sizes (bytes) | list of positive integers
`category` | The processor category | accera.Target.Category
`extensions` | Supported processor extensions | list of extension codes
`family` | The processor family | string
`frequency_GHz` | The processor frequency (GHz) | positive number
`model` | The processor model | accera.Target.Model
`name` | The processor name | string
`num_cores` | Number of cores | positive integer
`num_threads` | Number of threads | positive integer
`turbo_frequency_GHz` | Turbo frequency (GHz) | positive number

## Examples

Create a custom CPU target:
```python
cpu_target = acc.Target(name="Custom processor", category=acc.Target.Category.CPU, architecture=acc.Target.Architecture.X86_64, num_cores=10)
```

Craete a known CPU target representing an Intel Core i9:

```python
corei9 = acc.Target(model=acc.Target.Model.INTEL_CORE_I9, num_threads=44)
```

Create a custom GPU target:

```python
gpu_target = acc.Target(name="Custom GPU processor", category=acc.Target.Category.GPU, default_block_size=16)
```

Craete a known GPU target representing an NVidia Tesla v100:

```python
v100 = acc.Target(model=acc.Target.Model.NVIDIA_TESLA_V100)
```


<div style="page-break-after: always;"></div>
