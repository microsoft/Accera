[//]: # (Project: Accera)
[//]: # (Version: <<VERSION>>)

# Section 7: Plans - Vectorization and Parallelization
The plan includes operations and optimizations that control instruction pipelining, vectorized SIMD instructions, and parallelization.

## `unroll`
By default, each dimension of the iteration space is implemented as a for-loop. The `unroll` instruction marks a dimension for *unrolling* rather than looping. For example, imagine that we have the following nest, which multiplies the entires of an array by a constant:
```python
import accera as acc

my_target = acc.Target(type=acc.Target.Category.CPU)

nest = acc.Nest(shape=(3,5))
i, j = nest.get_indices()

@nest.iteration_logic
def _():
    A[i, j] *= 2.0

plan = nest.create_plan(my_target)
```
If we build `plan` as is, the resulting implementation would be equivalent to the following Python code:
```python
for i in range(3):
    for j in range(5):
        A[i, j] *= 2.0
```
If we add the instruction `plan.unroll(index=j)`, the resulting implementation becomes equivalent to:
```python
for i in range(3):
    A[i, 0] *= 2.0
    A[i, 1] *= 2.0
    A[i, 2] *= 2.0
    A[i, 3] *= 2.0
    A[i, 4] *= 2.0
```
If, instead of unrolling `j`, we add the instruction `plan.unroll(index=i)`, the resulting implementation becomes equivalent to:
```python
for j in range(5):
    A[0, j] *= 2.0
for j in range(5):
    A[1, j] *= 2.0
for j in range(5):
    A[2, j] *= 2.0
```
And, of course, we could also unroll both dimensions, removing for-loops completely.

## `vectorize`
Many modern target platforms support SIMD vector instructions. SIMD instructions perform the same operation on an entire vector of elements, all at once. By default, each dimension of an iteration space becomes a for-loop, but the `vectorize` instruction labels a dimension for vectorized execution, rather than for-looping.

For example, assume that the host supports 256-bit vector instructions, which means that its vector instructions operate on 8 floating-point elements at once. Imagine that we already have arrays `A`, `B`, and `C`, and that we write the following code:
```python
nest = acc.Nest(shape=(64,))
i = nest.get_indices()

@nest.iteration_logic
def _():
    C[i] = A[i] * B[i]

schedule = nest.create_schedule()
ii = schedule.split(i, 8)

plan = nest.create_plan()
plan.vectorize(index=ii)
```
The dimension marked for vectorization is of size 8, which is a supported vector size on the specific target platform. Therefore, the resulting binary will contain something like:
```
  00000001400010B0: C5 FC 10 0C 11     vmovups     ymm1,ymmword ptr [rcx+rdx]
  00000001400010B5: C5 F4 59 0A        vmulps      ymm1,ymm1,ymmword ptr [rdx]
  00000001400010B9: C4 C1 7C 11 0C 10  vmovups     ymmword ptr [r8+rdx],ymm1
  00000001400010BF: 48 8D 52 20        lea         rdx,[rdx+20h]
  00000001400010C3: 48 83 E8 01        sub         rax,1
  00000001400010C7: 75 E7              jne         00000001400010B0
```
Note how the multiplication instruction *vmulps* and the memory move instruction *vmovups* deal with 8 32-bit floating point values at a time.

Different targets support different vector instructions, with different vector sizes. The following table includes iteration logic that vectorizes correctly on most targets with vectorization support, such as Intel Haswell, Broadwell or newer, and ARM v7/A32. Other examples of iteration logic may or may not vectorize correctly. Variables prefixed with *v* are vector types, and those prefixed with *s* are scalar types.

| Vector pseudocode | Equivalent to | Supported types |
|---------|---------------|---------|
| `v1 += s0 * v0` | `for i in range(vector_size):` <br>&emsp; `v1[i] += s0 * v0[i]` | float32 |
| `v1 += v0 * s0` | `for i in range(vector_size):` <br>&emsp; `v1[i] += v0[i] * s0` | float32 |
| `v1 += v0 / s0` | `for i in range(vector_size):` <br>&emsp; `v1[i] += v0[i] / s0` | float32 |
| `v1 -= s0 * v0` | `for i in range(vector_size):` <br>&emsp; `v1[i] -= s0 * v0[i]` | float32 |
| `v1 -= v0 * s0` | `for i in range(vector_size):` <br>&emsp; `v1[i] -= v0[i] * s0` | float32 |
| `v1 -= v0 / s0` | `for i in range(vector_size):` <br>&emsp; `v1[i] -= v0[i] / s0` | float32 |
| `v2 += v0 * v1` | `for i in range(vector_size):` <br>&emsp; `v2[i] += v0[i] * v1[i]` | float32 |
| vector inner (dot) product: `s0 += dot(v0, v1)` | `for i in range(vector_size):` <br>&emsp; `s0 += v0[i] * v1[i]` | float32 |
| `v2 = v0 + v1` | `for i in range(vector_size):` <br>&emsp; `v2[i] = v0[i] + v1[i]` | int8/16/32/64, float32 |
| `v2 = v0 - v1` | `for i in range(vector_size):` <br>&emsp; `v2[i] = v0[i] - v1[i]` | int8/16/32/64, float32 |
| `v2 = v0 * v1`  | `for i in range(vector_size):` <br>&emsp; `v2[i] = v0[i] * v1[i]` | int8/16/32/64, float32 |
| `v2 = v0 / v1` | `for i in range(vector_size):` <br>&emsp; `v2[i] = v0[i] / v1[i]` | float32 |
| `v1 = abs(v[0])` | `for i in range(vector_size):` <br>&emsp; `v1[i] = abs(v0[i])` | int8/16/32/64, float32 |
| `v2 = (v0 == v1)` | `for i in range(vector_size):` <br>&emsp; `v2[i] = 0XF..F if v0[i] == v1[i] else 0` | int8/16/32/64, float32 |
| `v2 = (v0 > v1)` | `for i in range(vector_size):` <br>&emsp; `v2[i] = 0XF..F if v0[i] > v1[i] else 0` | int8/16/32/64, float32 |
| `v2 = (v0 >= v1)` | `for i in range(vector_size):` <br>&emsp; `v2[i] = 0XF..F if v0[i] >= v1[i] else 0` | int8/16/32/64, float32 |
| `v2 = (v0 < v1)` | `for i in range(vector_size):` <br>&emsp; `v2[i] = 0XF..F if v0[i] < v1[i] else 0` | int8/16/32/64, float32 |
| `v2 = (v0 <= v1)` | `for i in range(vector_size):` <br>&emsp; `v2[i] = 0XF..F if v0[i] <= v1[i] else 0` | int8/16/32/64, float32 |
| `v1 = v0 << s0` | `for i in range(vector_size):` <br>&emsp; `v1[i] = v0[i] << s0` | int16/32/64, float32 |
| `v1 = v0 >> s0` | `for i in range(vector_size):` <br>&emsp; `v1[i] = v0[i] >> s0` | int16/32/64, float32 |
| `s0 = sum(v0)` | `for i in range(vector_size):` <br>&emsp; `s0 += v0[i]` | int8/16/32/64, float32 |
| `s0 = max(v0 + v1)` | `for i in range(vector_size):` <br>&emsp; `s0 = max(v0[i] + v1[i], s0)` | int/8int16/32/64, float32 |
| `s0 = max(v0 - v1)` | `for i in range(vector_size):` <br>&emsp; `s0 = max(v0[i] - v1[i], s0)` | int/8int16/32/64, float32 |

In addition, Accera can perform vectorized load and store operations to/from vector registers and memory if the memory locations are contiguous.

To vectorize dimension `i`, the number of active elements that corresponds to dimension `i` must exactly match the vector instruction width of the target processor. For example, if the target processor has vector instructions that operate on either 4 or 8 floating point elements at once, then the number of active elements can be either 4 or 8. Additionally, those active elements must occupy adjacent memory locations (they cannot be spread out).

## Convenience syntax: `kernelize`
The `kernelize` instruction is a convenience syntax and does not provide any unique functionality. Specifically, `kernelize` is equivalent to a sequence of `unroll` instructions, followed by an optional `vectorize` instruction.

A typical Accera design pattern is to break a loop-nest into tiles and then apply an optimized kernel to each tile. For example, imagine that the loop nest multiplies two 256&times;256 matrices and the kernel is a highly optimized procedure for multiplying 4&times;4 matrices. In the future, Accera will introduce different ways to write highly optimized kernels, but currently, it only supports *automatic kernelization* using the `kernelize` instruction. As mentioned above, `kernelize` is shorthand for unrolling and vectorizing. These instructions structure the code in a way that makes it easy for downstream compiler heuristics to automatically generate kernels.

Consider, once again, the matrix multiplication example we saw previously in [Section 2](<02%20Simple%20Affine%20Loop%20Nests.md>).
Imagine we declare the schedule and reorder as follows:

```python
schedule = nest.create_schedule()
schedule.reorder(i, k, j)
```
Notice that `i, k, j` are the last three dimensions in the iteration space and the resulting implementation becomes equivalent to:

```python
for i in range(M):
    for k in range(S):
        for j in range(N):
            C[i, j] += A[i, k] * B[k, j]
```

The instruction:
```python
plan.kernelize(unroll_indices=(i, k), vectorize_indices=j)
```
is just shorthand for
```python
plan.unroll(i)
plan.unroll(k)
plan.vectorize(j)
```
Applying this sequence of instructions allows the compiler to automatically create an optimized kernel from loops `i, k, j`.

For simplicity, let's assume that the matrix sizes, defined by M, N, S are M=3, N=4, S=2.

After applying `kernelize`, the schedule is equivalent to the following Python code:
```python
C[0,0:4] += A[0,0] * B[0,0:4] # vectorized
C[0,0:4] += A[0,1] * B[1,0:4] # vectorized
C[1,0:4] += A[1,0] * B[0,0:4] # vectorized
C[1,0:4] += A[1,1] * B[1,0:4] # vectorized
C[2,0:4] += A[2,0] * B[0,0:4] # vectorized
C[2,0:4] += A[2,1] * B[1,0:4] # vectorized
```

This would result in the following vectorized instructions on an Intel Haswell CPU:
```
  0000000000000200: C4 C1 78 10 00     vmovups     xmm0,xmmword ptr [r8]
  0000000000000205: C4 E2 79 18 09     vbroadcastss xmm1,dword ptr [rcx]
  000000000000020A: C5 F8 10 12        vmovups     xmm2,xmmword ptr [rdx]
  000000000000020E: C4 E2 69 A8 C8     vfmadd213ps xmm1,xmm2,xmm0
  0000000000000213: C5 F8 10 5A 10     vmovups     xmm3,xmmword ptr [rdx+10h]
  0000000000000218: C4 E2 79 18 61 04  vbroadcastss xmm4,dword ptr [rcx+4]
  000000000000021E: C4 E2 61 A8 E1     vfmadd213ps xmm4,xmm3,xmm1
  0000000000000223: C4 E2 79 18 49 08  vbroadcastss xmm1,dword ptr [rcx+8]
  0000000000000229: C4 E2 69 A8 C8     vfmadd213ps xmm1,xmm2,xmm0
  000000000000022E: C4 E2 79 18 69 0C  vbroadcastss xmm5,dword ptr [rcx+0Ch]
  0000000000000234: C4 E2 61 A8 E9     vfmadd213ps xmm5,xmm3,xmm1
  0000000000000239: C4 E2 79 18 49 10  vbroadcastss xmm1,dword ptr [rcx+10h]
  000000000000023F: C4 E2 69 A8 C8     vfmadd213ps xmm1,xmm2,xmm0
  0000000000000244: C4 E2 79 18 41 14  vbroadcastss xmm0,dword ptr [rcx+14h]
  000000000000024A: C4 E2 61 A8 C1     vfmadd213ps xmm0,xmm3,xmm1
  000000000000024F: C4 C1 58 58 09     vaddps      xmm1,xmm4,xmmword ptr [r9]
  0000000000000254: C4 C1 50 58 51 10  vaddps      xmm2,xmm5,xmmword ptr [r9+10h]
  000000000000025A: C4 C1 78 58 41 20  vaddps      xmm0,xmm0,xmmword ptr [r9+20h]
```

## `parallelize`
The `parallelize` instruction performs one or more loops in parallel on multiple cores.

```python
xeonPlat = acc.Target("Intel 9221", num_threads=16)
plan = schedule.create_plan(xeonPlat)
plan.parallelize(indices=(i,j,k))
```
Specifying multiple dimensions is equivalent to the `collapse` argument in OpenMP. Therefore, the dimensions must be contiguous in the iteration space dimension order.

### Static scheduling policy
A static scheduling strategy is invoked by setting the argument `policy="static"` in the call to `parallelize`. If *n* iterations are parallelized across *c* cores, static scheduling partitions the work into *c* fixed parts, some of size *floor(n/c)* and some of size *ceil(n/c)*, and executes each part on a different core.

### Dynamic scheduling policy
A dynamic scheduling strategy is invoked by setting the argument `policy="dynamic"` in the call to `parallelize`. Dynamic scheduling creates a single queue of work that is shared across the different cores.

### __Not yet implemented:__ Pinning to specific cores
The `pin` argument allows the parallel work to be pinned to specific cores.

[comment]: # (* MISSING: multithreading and caching: how does parallelization affect caching? Separate cache per thread?)

## `bind`
Some target platforms, such as GPUs, are specifically designed to execute nested loops. They can take an entire grid of work and schedule its execution on multiple cores. On a GPU, this grid is broken up into multiple blocks, where each block contains multiple threads. Block iterators and thread iterators are identified by special variables in the `Target` object. To take advantage of a target platform's ability to execute grids, we must bind dimensions of the iteration space with these special iterator variables.

For example,
```python
v100 = acc.Target("Tesla V100")
plan.bind(indices=(i, j, k), grid=(v100.GridUnit.BLOCK_X, v100.GridUnit.THREAD_X, v100.GridUnit.THREAD_Y))
```


<div style="page-break-after: always;"></div>
