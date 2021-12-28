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
`vector_bytes` | Bytes per vector register | positive number
`vector_registers` | total number of SIMD registers | positive number

## Supported CPU Targets

Accera supports various CPU target architectures ranging from Intel Core Processors (7th Generation to 12th Generation),
Raspberry Pi3, Raspberry Pi Zero. We also support GPU target architectures, such as NVIDIA Tesla V100.

For CPU target architecture, the detailed list of Intel processors is mentioned below.
- Intel Core Processors

| Generation  | Intel Core Processor | Instruction Set Extensions             | Family       | Vector Registers | Vector Bytes |
| :---------  | :------------------- | :------------------------------------- | :--------------------------| :--| :----------- |
| 7th Gen     | Only Intel Core i7   | Intel SSE4.1, Intel SSE4.2, Intel AVX2 | Skylake-X                  | 16 | 32           |
| 8th Gen     | Intel Core i9 and i7 | Intel SSE4.1, Intel SSE4.2, Intel AVX2 | Coffee Lake                | 16 | 32           |
| 9th Gen     | Intel Core i9 and i7 | Intel SSE4.1, Intel SSE4.2, Intel AVX2 | Coffee Lake                | 16 | 32           |
| 10th Gen    | Intel Core i9 and i7 | Intel SSE4.1, Intel SSE4.2, Intel AVX2 | Comet Lake                 | 16 | 32           |
| 11th Gen    | Intel Core i9 and i7 | Intel SSE4.1, Intel SSE4.2, Intel AVX2, Intel AVX-512 | Rocket Lake | 32 | 64           |
| 12th Gen    | Intel Core i9 and i7 | Intel SSE4.1, Intel SSE4.2, Intel AVX2 | Alder Lake                 | 16 | 32           |

- Intel Xeon Processors

| Processor              | Instruction Set Extensions              | Additional Comments |
| :-----------           | :---------                              | :------------------ |
| Intel Xeon E Processor |  Intel SSE4.1, Intel SSE4.2, Intel AVX2 | All products launched in this family support this subset of instruction set extensions. However, the latest prodduct launched in Intel Xeon E processor in 2021 also supports AVX-512 extension.|


There are other kinds of Intel Xeon Processors, such as Intel Xeon W, Intel Xeon E5 V2, Intel Xeon E5 V3, and
Intel Xeon E5 V4 which support different kinds of instruction set extensions.
Processors of the Intel Xeon family support a wide variety of instruction set extensions that are specific to the processor model.
For example, Server and Desktop models of Intel Xeon E5 V2 support AVX while Intel Xeon E5 V3 and Intel Xeon E5 V4 support AVX2.
To create targets for a given Intel Xeon processor, you will need to consult its documentation
[here](https://ark.intel.com/content/www/us/en/ark/products/96901/intel-xeon-processor-e52699r-v4-55m-cache-2-20-ghz.html)
and define a target specifically for it.

## Examples

- Define CPU Targets
Let's take a look at some examples to understand how to define a CPU target in Accera.
Create a custom CPU target:
```python
cpu_target = acc.Target(name="Custom processor", category=acc.Target.Category.CPU, architecture=acc.Target.Architecture.X86_64, num_cores=10)
```

We further craete a known CPU target representing a 10th Generation Intel Core Processor

```python
gen10 = acc.Target(
		model=acc.Target.Model.INTEL_CORE_GENERATION_10,
                category=acc.Target.Category.CPU,
                architecture=acc.Target.Architecture.X86_64,
                family="Comet Lake",
                vector_bytes=32,
                vector_registers=16,
                extensions=["SSE4.1", "SSE4.2", "AVX2"],
                _device_name="gen10")
```
You can use this example as a starting point to define any other Intel Core Processor and the specifications
of them are listed in the table above.

- Define GPU Targets
Here is another example to create a custom GPU target:

```python
gpu_target = acc.Target(name="Custom GPU processor", category=acc.Target.Category.GPU, default_block_size=16)
```

Craete a known GPU target representing an NVidia Tesla v100:

```python
v100 = acc.Target(model=acc.Target.Model.NVIDIA_TESLA_V100)
```

## Additional Notes on Instruction Set Extensions
The details on extensions are important to identify the number of vector registers and vector bytes of each SIMD
register supported by a processor. These values may help you determine if you are leveraging
the vector units of an underlying hardware to its best capabilities.

### AVX
Advanced Vector Extensions (AVX) promotes legacy 128-bit SIMD instructions that operate on XMM
registers to use a vector-extension (VEX) prefix and operate on 256-bit YMM registers.


Intel AVX introduced support for 256-bit wide SIMD registers (YMM0-YMM7 in operating modes that are 32-bit or
less, YMM0-YMM15 in 64-bit mode). For Accera, 64-bit mode is the default and we do not add this as an argument to define
a target. The lower 128-bits of the YMM registers are aliased to the respective 128-bit XMM registers.
In Intel AVX, there are 256-bit wide vector registers, 16 XMM registers and 16 YMM registers to support an extension of 128-bits.

### AVX512
AVX-512 is a further extension offering 32 ZMM registers and each SIMD register is 512 bits (64 bytes) wide.

### SSE4 Extension
There are 16 XMM registers (XMM0 to XMM15) and each is 128-bit wide. In 64-bit mode, eight additional
XMM registers are accessible. Registers XMM8-XMM15 are accessed by using REX prefixes.

<div style="page-break-after: always;"></div>
