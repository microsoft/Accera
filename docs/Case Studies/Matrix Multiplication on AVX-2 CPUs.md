[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Case study - Float32 Matrix Multiplication on AVX-2 FMA3 CPUs
We first introduced matrix multiplication in [Section 0](../Manual/00%20Introduction.md) of the manual with a definition of `C += A @ B`. Recall that for an `M`&times;`N` matrix `C`, an `M`&times;`S` matrix `A`, and a `S`&times;`N` matrix `B`, the logic for matrix multiplication can be expressed in Python as:
```python
# C += A @ B
for i in range(M):
    for j in range(N):
        for k in range(S):
            C[i, j] += A[i, k] * B[k, j]
```

In [Section 2](../Manual/02%20Simple%20Affine%20Loop%20Nests.md) of the manual we provided a sample Accera implementation of matrix multiplication. That example used a default schedule and action plan, which will compute the correct result but will not produce an efficient implementation.

In this case study, we will focus on how to construct a Accera schedule and action plan to optimize matrix multiplication performance for `float32` element types on 64-bit CPUs with AVX-2 and FMA3 features. Many of these ideas will be valid on other CPUs, however some of the details will vary.

To have a sufficiently general case that is also concrete, for this case study we assume `M = 1020, N = S = 1024`, and that matrices `A`, `B`, and `C` all have `FIRST_MAJOR` layout in memory (see [Section 1](../Manual/01%20Arrays.md) of the manual for a discussion of array layout). We will also discuss how to augment some scheduling choices to suit other sizes.


## Target Hardware Characteristics
For concreteness, we assume that our target hardware has the following characteristics:
* It is a single-core CPU with AVX-2 and FMA3 instruction set extensions.
* Intel Haswell/Broadwell-era common hardware cache sizes such as:
    * L1 cache size of `32KB` (per core)
    * L2 cache size of `256KB` (per core)
    * L3 cache size of at least `12MB`. Note: this is large enough to contain `(3 * 1024 * 1024)` 4-byte float elements, which is greater than the size of all of our `A`, `B`, and `C` matrices combined for this case study, so we will ignore the L3 cache as we will rarely overflow elements from it.
    * A CPU with two 256-bit FMA units. Most Intel Broadwell and later era chips and similar era AMD chips have two. To find how many FMA units your chip has, you can consult the optimization guide for your chip ( [Intel](https://software.intel.com/content/www/us/en/develop/articles/intel-sdm.html#optimization), [AMD](https://developer.amd.com/resources/developer-guides-manuals/) ). Alternatively you may examine the throughput or the port documentation for `vfmadd*` instructions for your chip [here](https://uops.info/table.html), the throughput is the average cycles per instruction, if your chip's average throughput for a `vfmadd` is `0.5` then it indicates you have 2 FMA units.

## Basic Setup
The accera logic functions remains unchanged from [Section 2](../Manual/02%20Simple%20Affine%20Loop%20Nests.md). We include it here for completeness:
```python
import accera as acc

# Define matrix sizes
M = 1020
N = 1024
S = 1024

A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, S))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(S, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

# Define a simple affine loop nest and name its loops i, j, k
nest = acc.Nest(shape=(M, N, S))
i, j, k = nest.get_indices()

# Define the logic of each iteration in the nest
@nest.iteration_logic
def _():
    C[i,j] += A[i,k] * B[k,j]
```

At this point, our implementation is equivalent to the following Python code:
```python
for i in range(1020):
    for j in range(1024):
        for k in range(1024):
            C[i,j] += A[i,k] * B[k,j]
```

## Improving Hardware Cache Usage

Due to matrix multiplication having a 3-dimensional iteration space and operating over 2-dimensional arrays, it will revisit elements in each matrix a number of times. Specifically it will revisit each element of the `C` matrix `S` times, each element of the `A` matrix `N` times, and each element of the `B` matrix `M` times.

While this data reuse cannot be avoided, we can make changes to reduce the performance penalty of revisiting elements. Once an element has been loaded, it will reside in the various hardware caches on our system, and reusing an element that is already in a hardware cache will be faster than using an element that was never in the hardware cache or has since been evicted.

### Iteration Space Tiling
To achieve this reuse, we *tile* the iteration space, using the `tile()` API described in [Section 3](../Manual/03%20Schedules.md) of the manual, to break the iteration space into smaller blocks in which we will reuse data elements multiple times before moving on.

We will use Accera Parameters, defined in [Section 9](../Manual/09%20Parameters.md) of the manual, to set our tile split sizes for each iteration space dimension and determine their values later based on hardware characteristics.

Let's add this tiling to our Accera implementation as a series of schedule splits:
```python
# Using nest and iteration logic declared earlier
schedule = nest.create_schedule()

# Tile splits
m_tile_size, n_tile_size, s_tile_size = acc.create_parameters(3)

ii, jj, kk = schedule.tile((i, j, k), (m_tile_size, n_tile_size, s_tile_size))
```

At this point, our implementation is equivalent to the following Python code:
```python
for i in range(0, 1020, m_tile_size):
    for j in range(0, 1024, n_tile_size):
        for k in range(0, 1024, s_tile_size):
            for ii in range(0, m_tile_size, 1):
                for jj in range(0, n_tile_size, 1):
                    for kk in range(0, s_tile_size, 1):
                        C[(i + ii), (j + jj)] += A[(i + ii), (k + kk)] * B[(k + kk), (j + jj)]
```

### Tile Sizes

For this case study, we want to have the `B` matrix tile data fill half of our L2 cache and leave the remaining half of the L2 cache for `A` and `C` tile data. Since we assumed a `256KB` L2 cache, this means we want to examine `32768` elements of `B` (`128KB = 32768 * 4 (bytes per float)`), so we will choose a `128`&times;`256` (`128*256 = 32768`) tile size for `B` as these sizes evenly divide the `B` matrix size `1024`&times;`1024` so these splits will not produce a leftover section that is less efficient to compute. Arbitrarily, we set `n_tile_size = 256` and `s_tile_size = 128`, however these values can be swapped with minimal difference in performance. We will not set the `m_tile_size` yet, but since we've chosen the `B` submatrix tile to be half of the L2 cache, we know we have the constraint that (size of `C` submatrix tile + size of `A` submatrix tile) <= remaining L2 cache space, which means `((m_tile_size` &times; `n_tile_size) + (m_tile_size` &times; `s_tile_size)) <= 32768` elements.

### Tile Ordering

Since the `B` submatrix tiles are the largest tiles we examine, we would like to maximize how much we reuse them before moving on. Therefore in addition to tiling, we will also reorder our schedule such that the `j` and `k` indices precede the `i` index, so that `j` and `k` iterate less often so we change `B` tiles less frequently at the expense of changing `A` and `C` tiles more frequently. So instead of an `(i, j, k, ii, jj, kk)` order, we should prefer a `(j, k, i, ii, jj, kk)` order.

What order should the indices inside the tiles `(..., ii, jj, kk)` be in? We know for each each `(j, k, i)` iteration subdomain tile we will be using more than half of our `256KB` L2 cache so we will overflow our `32KB` L1 cache repeatedly. If we kept the default inner tile order of `(ii, jj, kk)`, then for each `ii` iteration we examine a `m_tile_size`&times;`256` submatrix of our `C` tile, a `m_tile_size`&times;`128` submatrix of our `A` tile, and still the entire `128`&times;`256` `B` tile (`128KB`), so we'll overflow our L1 cache within that loop. Instead we order those inner tile dimensions as `(jj, kk, ii)` so that within the `jj` loop we examine a `m_tile_size`&times;`1` submatrix of our `C` tile, the full `m_tile_size`&times;`128` tile of `A`, and a `128`&times;`1` submatrix of our `B` tile (`512` bytes). We aim to not overflow our L1 cache during this loop, so our `m_tile_size` value has another constraint: `(4 bytes per float)*(m_tile_size`&times;`1 + m_tile_size`&times;`128 + 128`&times;`1) <= 32KB`, simplified: `m_tile_size <= 62` (rounded down).

Therefore with the order `(j, k, i, jj, kk, ii)` we get L2 cache reuse within each iteration of the `i` loop and L1 cache reuse within each iteration of the `kk` loop.

Let's add this reorder to our Accera implementation:
```python
# Using nest, iteration logic, and splits declared earlier
schedule.reorder(j, k, i, jj, kk, ii)
```

At this point, our implementation is equivalent to the following Python code:
```python
for j in range(0, 1024, n_tile_size):
    for k in range(0, 1024, s_tile_size):
        for i in range(0, 1020, m_tile_size):
            for jj in range(0, n_tile_size, 1):
                for kk in range(0, s_tile_size, 1):
                    for ii in range(0, m_tile_size, 1):
                        C[(i + ii), (j + jj)] += A[(i + ii), (k + kk)] * B[(k + kk), (j + jj)]
```

## Kernel Design

The "kernel" of our implementation is the innermost part of the loopnest which performs the computation, and it may consist of many of the innermost splits of the nest. Optimizing this kernel code will matter far more than any other optimization in our loopnest as the code spends the majority of its time in this innermost section.

### Register Size and Data Layout Considerations
With Accera, we don't need to specifically plan out the exact machine code we want to achieve, but having the capabilities of the hardware in mind will be beneficial for generating performant code. For the AVX-2 / FMA3 CPUs we're targeting in this case study, there are 16 &times; 256-bit vector registers which can perform data-parallel computation when we structure our kernels appropriately. Since these registers are 256 bits wide, they can hold 8 32-bit floats, so we will want to structure our innermost loop so that the innermost 8 iterations can map data to positions in these registers. This innermost loop will be *vectorized*, as discussed in [Section 7](../Manual/07%20Action%20plans%20-%20Vectorization%20and%20Parallelization.md) of the manual.

Consider what happens to the matrix multiplication logic if each of the 3 iteration domain dimensions are selected to be the inner split dimension of size 8. Accera vectorization will mark the loop as a data-parallel loop, unroll it, and attempt to vectorize instructions inside of it, so it is helpful to consider the unrolled version of the loop and what the data flow will look like:

If we select the `M` dimension, then we will try to act on elements from 8 different rows of the `A` matrix, elements from 8 different rows of the `C` matrix, and the same element from the `B` matrix
```
C[  i,j] += A[  i,k] * B[k,j]
C[i+1,j] += A[i+1,k] * B[k,j]
...
C[i+7,j] += A[i+7,k] * B[k,j]
```
Note that since `A` and `C` have `FIRST_MAJOR` layout, elements `A[i,k]` and `A[i+1,k]` are far apart in memory, likewise for `C[i,k]` and `C[i+1,k]`, so loading this data into vector registers will be slow, so this is likely not a good candidate.

If we select the `N` dimension, then similar to before we consider the following inner loop:
```
C[i,  j] += A[i,k] * B[k,  j]
C[i,j+1] += A[i,k] * B[k,j+1]
...
C[i,j+7] += A[i,k] * B[k,j+7]
```
Note that since `B` and `C` have `FIRST_MAJOR` layout, elements `B[k,j]` and `B[k,j+1]` are sequential in memory, likewise for `C[i,j]` and `C[i,j+1]`, so loading this data into vector registers will be fast, so this is potentially a good candidate.

If we select the `S` dimension, then similar to before we consider the following inner loop:
```
C[i,j] += A[i,  k] * B[  k,j]
C[i,j] += A[i,k+1] * B[k+1,j]
...
C[i,j] += A[i,k+7] * B[k+7,j]
```
Note since `A` and `B` have `FIRST_MAJOR` layout, loading the `A` elements will be efficient but loading the `B` elements will be slow as discussed before. Furthermore, this loop is not data-parallel as we are writing to the same element `C[i,j]` every time and running a reduction step across our result vector of elements is not going to be automatic or efficient. Therefore this is not going to be a good candidate.

Based on the above, we choose our innermost loop to be in the `N` dimension and be of size 8 so our data fits nicely within the registers our hardware has, and we will vectorize this loop.

Adding this split, reorder, and vectorize to our implementation:
```python
# Using nest, iteration logic, and splits declared earlier
vector_size = acc.create_parameters(1)

jjj = schedule.split(jj, vector_size)

schedule.reorder(j, k, i, jj, kk, ii, jjj)

plan = schedule.create_action_plan()

# Vectorize the innermost loop
plan.vectorize(jjj)
```

### Accounting for Instruction-Level Parallelism

As mentioned in the target hardware characteristics for this case study, we assume that we have two 256-bit FMA (fused multiply add) units on our CPU. With two such units in our hardware, we can have two vector fma instructions running simultaneously. If we design our kernel so that these two instructions appear sequentially, the hardware will see that it can issue these simultaneously by scheduling each instruction to a different unit within the same cycle. Our inner logic function performs a multiplication (`A[i,k]*B[k,j]`) followed by an addition (`C[i,j] += ...`), so when we vectorize our inner loop, we will get a single vector fused-multiply-add instruction. To get two of these instructions successively, we want to unroll the next innermost loop by some factor.

First, we need to determine on which dimension our next innermost loop will be. Since our innermost loop computes `C[i,j] += A[i,k] * B[k,j], ... C[i,j+7] += A[i,k] * B[k,j+7]`, unrolling our next innermost loop will unroll this computation in one of the 3 dimensions. As before, we consider what we would be computing if we were to unroll each dimension:

If we select the `M` dimension, then we will be computing:
```
C[  i,j] += A[  i,k] * B[k,j], ... C[  i,j+7] += A[  i,k] * B[k,j+7]
C[i+1,j] += A[i+1,k] * B[k,j], ... C[i+1,j+7] += A[i+1,k] * B[k,j+7]
```
Note that the same element of `B` is used in all of these results, so we can get good register reuse from our loaded `B` values. However, since `A` and `C` have `FIRST_MAJOR` layout, elements `A[i,k]` and `A[i+1,k]` are far apart in memory, likewise for `C`, so we will not get much benefit from hardware prefetching. This is potentially a good candidate since we can get good data reuse.

If we select the `N` dimension, then we will be computing:
```
C[i,  j] += A[i,k] * B[k,  j], ... C[i, j+7] += A[i,k] * B[k, j+7]
C[i,j+8] += A[i,k] * B[k,j+8], ... C[i,j+15] += A[i,k] * B[k,j+15]
```
Note that the same element of `A` is used in all of these results, so we can get good register reuse from our loaded `A` value, and since `B` and `C` have `FIRST_MAJOR` layout, elements `B[k,j+8], ..., B[k,j+15]` immediately follow elements `B[k,j], ..., B[k,j+7]` in memory, likewise for `C`, so our hardware prefetcher will likely have already loaded those values into our hardware caches. So this is a good candidate, and slightly better than `M`.

If we select the `S` dimension, then we will be computing:
```
C[i,j] += A[i,  k] * B[  k,j], ... C[i,j+7] += A[i,  k] * B[  k,j+7]
C[i,j] += A[i,k+1] * B[k+1,j], ... C[i,j+7] += A[i,k+1] * B[k+1,j+7]
```
Note that both of these are writing to the same elements `C[i,j], ..., C[i,j+7]`, which does not produce a data parallel issue like we saw before since we are not trying to vectorize this loop, however this means that the result of the first vector FMA is required in order to complete the second vector FMA, so we will be forcing our CPU to stall waiting for a result and we won't be getting the instruction-level parallelism we desire.

Based on the above, we choose our next innermost loop to also be in the `N` dimension and be of size 16 so that we can unroll it and get two successive vector FMAs and maximize instruction-level parallelism.

Updating our implementation with this split and reorder:
```python
# Using nest, iteration logic, and splits declared earlier
# Updating the splits and reorders from before
vector_size, n_kernel_split_size = acc.create_parameters(2)

jjj = schedule.split(jj, n_kernel_split_size)
jjjj = schedule.split(jjj, vector_size)

schedule.reorder(j, k, i, jj, kk, ii, jjj, jjjj)

plan = schedule.create_action_plan()

# Unroll the non-vectorized kernel loops
plan.unroll(jjj)

# Vectorize the innermost loop
plan.vectorize(jjjj)
```

### Optimizing Register Use

As we've seen, we want our innermost loop to be in the `N` dimension, and our next innermost loop to also be in the `N` dimension. While it would be tempting to continue going in the `N` dimension further, it would not produce a performant kernel. As mentioned before, the AVX-2 / FMA3 systems we're targeting have 16 of these 256-bit registers, and in the instruction-level parallel kernel loops we've discussed we're already using 5 of them: one to hold the values `B[k,j], ..., B[k,j+7]`, one to hold the values `B[k,j+8], ..., B[k,j+15]`, one to hold the value `A[i,k]` (which gets broadcast into each of the 8 register slots), one to hold the values `C[i,j], ..., C[i,j+7]`, and one to hold the values `C[i,j+8], ..., C[i,j+15]`.

If we continued in the `N` dimension, we would need one more register per 8 elements of `C`, and potentially two more registers per 16 elements of `B`, though we could reuse the registers from the first 16 elements of `B` after we've run our vector FMA instructions. So if we reused the registers, we would need to load 2 vectors from `B` for every 2 vector FMAs we run, and we could use all of our remaining registers for holding different 8-element chunks of `C` data, for a total of 13 registers holding 8 elements from `C`, or `13 * 8 = 104` elements of `C`. This would work, however having 104 sequential elements of `C` is not going to divide evenly into our `N` tile split of 256, or any other potential split size that would evenly divide our full `N` dimension of 1024. If we don't have any further loops within the tile in the `N` dimension, then our current `N` inner loop size of 16 will evenly divide our tile split of 256. Furthermore, having 2 loads for every 2 vector FMAs is slower than other alternatives, and anything that affects the performance of our kernel will have a magnified effect on our overall performance.

If we aren't going to make our next innermost loop in the `N` dimension, then we'll consider `M` and `S`.

If we select the `M` dimension, then we will be computing:
```
C[i,  j] += A[i,k] * B[k,  j], ... C[i, j+7] += A[i,k] * B[k, j+7]
C[i,j+8] += A[i,k] * B[k,j+8], ... C[i,j+15] += A[i,k] * B[k,j+15]

C[i+1,  j] += A[i+1,k] * B[k,  j], ... C[i+1, j+7] += A[i+1,k] * B[k, j+7]
C[i+1,j+8] += A[i+1,k] * B[k,j+8], ... C[i+1,j+15] += A[i+1,k] * B[k,j+15]
...
```
Note that this is using the same elements from `B` each iteration, so we can reuse the registers holding that data. It's using a different element from `A` each time, so we can overwrite the register holding `A` data with a newly broadcast element from `A`, and 1 broadcast is faster than the 2 loads we were considering for the `N` dimension.

If we select the `S` dimension, then we will be computing:
```
C[i,  j] += A[i,k] * B[k,  j], ... C[i, j+7] += A[i,k] * B[k, j+7]
C[i,j+8] += A[i,k] * B[k,j+8], ... C[i,j+15] += A[i,k] * B[k,j+15]

C[i,  j] += A[i,k+1] * B[k+1,  j], ... C[i, j+7] += A[i,k+1] * B[k+1, j+7]
C[i,j+8] += A[i,k+1] * B[k+1,j+8], ... C[i,j+15] += A[i,k+1] * B[k+1,j+15]
...
```
Note that this is using different elements from `B` each iteration and a different element from `A` each iteration, so we are getting no register reuse whatsoever and must perform a broadcast and two loads before every vector FMA pair. Therefore this is a worse option than unrolling in the `M` dimension.

Based on the above, we choose our next innermost loop to be in the `M` dimension. How many of these loops can we run in our kernel? As mentioned earlier we are using 3 registers for `A` and `B` data, so we have up to 13 for `C` data, but we know we're using 2 registers for `C` data in the `N` dimension, so we can have at most 6 iterations of the `M` dimension in our kernel, using 6 * 2 = 12 of the registers for accumulation. We don't need any other splits in the `M` dimension, so we will choose `m_tile_size` to be 6 to match this kernel. Note: this satisfies the constraints identified earlier, that `m_tile_size <= 62`, `m_tile_size < n_tile_size = 256`, and `m_tile_size < s_tile_size = 128`

Note that our Accera implementation does not change now, all we've done is determine the range of our `ii` loop, which is already in the appropriate position in the schedule.

### Optimizing Kernel Instruction Pipelining

We have one final schedule change to make to improve kernel speed. When the kernel gets compiled down to assembly code, the innermost loop that will still be represented as a loop in machine code is the `kk` loop. Since we're going to run this kernel many times in a row, we can partially unroll this loop to get better pipelining of instructions as there will be less loop management code and jump instructions being run, and the CPU instruction lookahead can operate better and over a larger section of code.

More concretely, on the CPUs we are focusing on in this case study, the latency of the vector FMA instructions is 5 cycles, which means if we chain 5 such instructions per FMA unit we can maximally saturate the instruction pipeline for that unit. With two FMA units, this means we want 10 vector FMA instructions issued close together, which is possible within a single kernel (which runs 12 vector FMAs), but if we partially unroll to get multiple kernels running sequentially without any loop management code in between we will saturate the pipeline even longer, giving better performance.

The amount that we want to unroll is primarily determined empirically, so it will vary from machine-to-machine, but in general a number which evenly divides the `S` tile size will perform better as it will not create an oddly sized boundary section. For this case study size, unrolling by 4 tends to perform best, so we need to add that split into our Accera implementation. Note: Accera will implicitly unroll the innermost loop of the nest that has not been explicitly unrolled or vectorized so we don't need to call an unroll on this newly split loop, but we do need to perform the split.


Updating our Accera implementation with this split and reorder:
```python
# Using nest, iteration logic, and splits declared earlier
# Updating the splits and reorders from before
vector_size, n_kernel_split_size, s_unroll_factor = acc.create_parameters(3)

kkk = schedule.split(kk, s_unroll_factor)
jjj = schedule.split(jj, n_kernel_split_size)
jjjj = schedule.split(jjj, vector_size)


schedule.reorder(j, k, i, jj, kk, kkk, ii, jjj, jjjj)

plan = schedule.create_action_plan()

# Unroll the non-vectorized kernel loops
plan.unroll(ii)
plan.unroll(jjj)

# Vectorize the innermost loop
plan.vectorize(jjjj)
```

At this point, our implementation is equivalent to the following Python code:
```python
for j in range(0, 1024, n_tile_size):
    for k in range(0, 1024, s_tile_size):
        for i in range(0, 1020, m_tile_size):
            for jj in range(0, n_tile_size, n_kernel_split_size)):
                for kk in range(0, s_tile_size, s_unroll_factor):
                    for kkk in range(0, s_tile_size, 1): # Implicitly unrolled
                        for ii in range(0, m_tile_size, 1): # Explicitly unrolled
                            for jjj in range(0, n_kernel_split_size), vector_size): # Explicitly unrolled
                                C[(i+ii), (j+jj+jjj):(j+jj+jjj+vector_size)] += A[(i+ii), (k+kk+kkk)] * B[(k+kk+kkk), (j+jj+jjj):(j+jj+jjj+vector_size)] # Vectorized index jjjj
```

## Data Layout and Temporary Buffers

So far we have determined our tiling sizes and designed our kernel, however our implementation is not as performant as it could be yet. In this case study, we've assumed that `B` has `FIRST_MAJOR` layout, however having this layout is not optimal because of how we are accessing elements in the `B` matrix.

If we examine our loop schedule and specifically what elements of `B` it reads, we see that it reads 16 elements from a row of `B` in the `jjj` and `jjjj` loops, then as the `kkk` loop iterates it jumps down to the next row and reads 16 elements from that row, and so on. Finally when the `jj` loop iterates, it will read the second set of 16 elements from the first row, and so on. This is sometimes called a Z-order traversal, and if we can arrange our `B` matrix data in memory such that the first 16 elements of each row are followed by the first 16 elements of the next row (sometimes called "Z-order packing") and so on, then the CPU's hardware prefetcher will ensure that we're consistently getting hardware cache hits as we read `B` data. For this case study, we will pack an entire `128`&times;`256` tile of `B` matrix data at a time.

As discussed in [Section 6](../Manual/06%20Action%20plans%20-%20Caching.md) of the manual, Accera has a built-in "cache" utility that will examine the access pattern of a given array and create a temporary buffer holding a local copy of array data that can be packed such that the buffer is read sequentially from front-to-back by the loopnest. The act of packing this temporary buffer serves to fill the L2 hardware cache with the data we're interested in, however it also takes some amount of time so for very small matrices it is not worthwhile. For the `1020`&times;`1024`&times;`1024` scenario in this case study, however, the matrices are plenty large enough to make using a Accera cache on the `B` matrix tiles worthwhile.

Since individual elements are read from the `A` matrix and our kernel examines the same 6 rows of `A` for a longer period of time, we don't see benefits empirically from creating a Accera cache for `A` in the same way, however this may differ on different hardware or for different input sizes.

For the `C` matrix, we don't necessarily need to re-arrange the data since we're looking at a small `6`&times;`16` region at a time that is already in `FIRST_MAJOR` layout, however having the temporary buffer is useful for Accera to be able to promote an accumulation buffer to purely accumulation registers. Therefore we also make a Accera cache of the `C` matrix tile to act as an output accumulation buffer that will be added back into the base `C` matrix at the conclusion of the `kk` loop.

Adding the Accera caching calls into our Accera implementation with the action plan `cache` API:
```python
# Using nest, iteration logic, splits, unrolls, and vectorization declared earlier
plan.cache(B, index=jj) # Cache the region of the B matrix jj active block, i.e. an entire B tile
plan.cache(C, index=ii) # Cache the region of the C matrix ii active block, i.e. the 6x16 kernel accumulation tile
```

At this point, our implementation is equivalent to the following Python code (note that Accera identifies that the `B` active block for index `jj` is identical to the `B` active block for index `i` and so can be hoisted up to cover a larger key-slice, and likewise the `C` active block for index `ii` is identical to the `C` active block for index `kk` and can be hoisted too):
```python
for j in range(0, 1024, n_tile_size):
    for k in range(0, 1024, s_tile_size):
        cached_B = CacheBuffer(B, (k, j), (s_tile_size, n_tile_size)) # pseudo-code
        for i in range(0, 1020, m_tile_size):
            for jj in range(0, n_tile_size, n_kernel_split_size)):
                cached_C = CacheBuffer(C, (i, j), (m_tile_size, n_kernel_split_size))) # pseudo-code
                for kk in range(0, s_tile_size, s_unroll_factor):
                    for kkk in range(0, s_unroll_factor, 1): # Implicitly Unrolled
                        for ii in range(0, m_tile_size, 1): # Explicitly Unrolled
                            for jjj in range(0, n_kernel_split_size), vector_size): # Explicitly Unrolled
                                cached_C[(i+ii), (j+jj+jjj):(j+jj+jjj+vector_size)] += A[(i+ii), (k+kk+kkk)] * cached_B[(k+kk+kkk), (j+jj+jjj):(j+jj+jjj+vector_size)] # Vectorized index jjjj
```

With the caching added, our implementation for this case study is complete.

## Final Implementation

Here is the full Accera implementation, with the parameter sizes we've derived along the way set for completeness:

```python
import accera as acc

# Define matrix sizes
M = 1020
N = 1024
S = 1024

A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, S))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(S, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

# Define a simple affine loop nest and name its loops i, j, k
nest = acc.Nest(shape=(M, N, S))
i, j, k = nest.get_indices()

# Define the logic of each iteration in the nest
@nest.iteration_logic
def _():
    C[i,j] += A[i,k] * B[k,j]

schedule = nest.create_schedule()

m_tile_size, n_tile_size, s_tile_size, vector_size, n_kernel_split_size, s_unroll_factor = acc.create_parameters(6)

# Tile splits
ii, jj, kk = schedule.tile((i, j, k), (m_tile_size, n_tile_size, s_tile_size))

# Kernel splits
kkk = schedule.split(kk, s_unroll_factor)
jjj = schedule.split(jj, n_kernel_split_size)
jjjj = schedule.split(jjj, vector_size)

schedule.reorder(j, k, i, jj, kk, kkk, ii, jjj, jjjj)

plan = schedule.create_action_plan()

# Unroll the non-vectorized kernel loops
plan.unroll(ii)
plan.unroll(jjj)

# Vectorize the innermost kernel loop
plan.vectorize(jjjj)

plan.cache(B, index=jj) # Cache the region of the B matrix jj active block, i.e. an entire B tile
plan.cache(C, index=ii) # Cache the region of the C matrix ii active block, i.e. the 6x16 kernel accumulation tile

# create a package
package = acc.Package()

parameter_values = {
    m_tile_size: 6,
    n_tile_size: 256,
    s_tile_size: 128,
    vector_size: 8, # 8 floats per 256-bit register
    n_kernel_split_size: 16,
    s_unroll_factor: 4
}
package.add_function(plan, args=(A, B, C), parameters=parameter_values, base_name="matmul_1020_1024_1024")

name = "avx2_matmul"
package.build(name, format=acc.Package.Format.HAT, output_dir=name)
```

This implementation is equivalent to the following Python code, with parameter values set:
```python
for j in range(0, 1024, 256):
    for k in range(0, 1024, 128):
        cached_B = CacheBuffer(B, (k, j), (128, 256)) # pseudo-code
        for i in range(0, 1020, 6):
            for jj in range(0, 256, 16)):
                cached_C = CacheBuffer(C, (i, j), (6, 16))) # pseudo-code
                for kk in range(0, 128, 4):
                    for kkk in range(0, 4, 1): # Implicitly Unrolled
                        for ii in range(0, 6, 1): # Explicitly Unrolled
                            for jjj in range(0, 16, 8): # Explicitly Unrolled
                                cached_C[(i+ii), (j+jj+jjj):(j+jj+jjj+8)] += A[(i+ii), (k+kk+kkk)] * cached_B[(k+kk+kkk), (j+jj+jjj):(j+jj+jjj+8)] # Vectorized index jjjj
```

## Tuning notes for other sizes
For matrix sizes other than those that we examined here, the same general structural choices and the kernel design decisions that we made in this case study will typically be valid, but different tile sizes may be optimal. A simple change to make would be to reshape the `128`&times;`256` `B` tile size to a different shape with a comparable volume to better fit the matrices you are operating on. For example, if `N = 128, S = 1024`, then having the `N` split be `256` will be valid and produce the correct result, but it will not wind up using as much of the hardware cache as possible. In this scenario it would be better to reshape the tile sizes so that `N` is split by `128` and `S` is split by `256`, keeping the same `B` tile volume and the same hardware cache utilization as we saw in this case study. Whether or not to use a Accera cache for the `B` matrix will also vary by input size and hardware. Empirically on AVX-2 systems, we see that if `B` is larger than roughly `128`&times;`128` then caching is worthwile, and if it is smaller than `128`&times;`128` it adds more overhead than benefit.

One could also tune the `S` unroll value that we chose to be `4` for this case study, however ensuring it evenly divides the `S` tile size is important. The `6` and `16` values were derived from the vector register hardware characteristics, so any system with this type of registers will benefit from these split sizes, but systems with other types of registers, such as AVX-512 systems or ARM systems, will have different optimal values.
