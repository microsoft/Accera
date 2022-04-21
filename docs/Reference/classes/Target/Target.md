[//]: # (Project: Accera)
[//]: # (Version: v1.2.30)

# Accera v1.2.3 Reference

## `accera.Target([architecture, cache_lines, cache_sizes, category, extensions, family, frequency_GHz, model, name, num_cores, num_threads, turbo_frequency_GHz])`

Defines the capabilities of a target processor.

## Arguments

argument | description | type/default
--- | --- | ---
`known_name` | A name of a device known to Accera | string \| accera.Target.Model / "HOST".
`architecture` | The processor architecture | accera.Target.Architecture
`cache_lines` | Cache lines (kilobytes) | list of positive integers
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
`vector_bytes` | Bytes per vector register | positive number
`vector_registers` | total number of SIMD registers | positive number

## Known device names

Accera provides a pre-defined list of known target through the [`accera.Target.Models`](<Model.md>) enumeration.

These known targets provide typical hardware settings and may not fit your specific hardware characteristics exactly. If your target matches closely with (but not exactly to) one of these targets, you can always start with a known target and update the properties accordingly.

## Examples

Let's have a look at some examples to understand how to define a CPU target in Accera.

Create a custom CPU target:
```python
cpu_target = acc.Target(name="Custom processor", category=acc.Target.Category.CPU, architecture=acc.Target.Architecture.X86_64, num_cores=10)
```

We further create a known CPU target and can selectively override fields.

```python
gen10 = acc.Target(
                known_name="Intel 7940X",
                category=acc.Target.Category.CPU,
                extensions=["SSE4.1", "SSE4.2", "AVX2"])
```

In this example, we created a target device of a known CPU but overrode the
extensions to remove AVX512 support.

You can use this example as a starting point to define any other Intel Core Processor. Their specifications are listed in the table above.

Craete a pre-defined GPU target representing an NVidia Tesla v100 processor:

```python
v100 = acc.Target(model=acc.Target.Model.NVIDIA_TESLA_V100)
```

Here is another example to create a custom GPU target:

```python
gpu_target = acc.Target(name="Custom GPU processor", category=acc.Target.Category.GPU, default_block_size=16)
```

## Additional Notes on Instruction Set Extensions
The details on extensions are important to identify the number of vector registers and vector bytes of each SIMD register supported by a processor. These values may help you determine if you are leveraging the vector units of the underlying hardware to its best capabilities.

### AVX
Advanced Vector Extensions (AVX) promotes legacy 128-bit SIMD instructions that operate on XMM registers to use a vector-extension (VEX) prefix and operate on 256-bit YMM registers.

Intel AVX introduced support for 256-bit wide SIMD registers (YMM0-YMM7 in operating modes that are 32-bit or less, YMM0-YMM15 in 64-bit mode). For Accera, 64-bit mode is the default, and we do not add this as an argument to define
a target. The lower 128-bits of the YMM registers are aliased to the respective 128-bit XMM registers.
In Intel AVX, there are 256-bit wide vector registers, 16 XMM registers, and 16 YMM registers to support an extension of 128-bits.

### AVX512
AVX-512 is a further extension offering 32 ZMM registers, and each SIMD register is 512 bits (64 bytes) wide.

### SSE4 Extension
There are 16 XMM registers (XMM0 to XMM15), each 128-bit wide. In 64-bit mode, eight additional XMM registers are accessible. Registers XMM8-XMM15 are accessed by using REX prefixes.

<div style="page-break-after: always;"></div>


