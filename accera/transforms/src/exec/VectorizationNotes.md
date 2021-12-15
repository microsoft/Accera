# Accera Vectorization

- Currently loops need to be marked for vectorization
- In the Accera DSL, `Plan::Vectorize(...)` will mark a loop with the appropriate attributes for vectorization
- Currently, it is assumed that only the inner loops will get marked for vectorization (other cases haven't been tested)

## Vectorizing a loop
- Vectorizing a loop adds an attribute with the count of SIMD registers and the number of elements that can fit in a register
- Vectorized loops will be unrolled up to a factor equal to the number of elements that can fit in a SIMD register. See `VectorizeAffineForOpConversion` in `accera\transforms\src\exec\ExecutionPlanToAffineLoweringPass.cpp`
- While unrolling, any caches that are indexed into with the unrolled loop induction variable will be marked to be vectorized.
- A vectorized cache will lower to have vector type elements that are sized to match the SIMD registers, as opposed to individual elements. E.g. a `<6 x 16 x f32>` cache will become a `<6 x 2 x vector<8 x f32>>` vectorized cache if the vector size is 8.
- Any cache loads or cache stores that target a vectorized cache will be replaced with `VectorizedCacheLoadOp` and `VectorizedCacheStoreOp`, which take the unroll index as a separate attribute from the indices used to index into the cache.
- `VectorizedCacheLoadOp(indices, unrollIdx)` will lower to an `mlir::AffineLoadOp` on the `<6 x 2 x vector<8 x f32>>` cache using the `indices` to retrieve a `vector<8 x f32>` and then run `mlir::vector::ExtractElementOp` on that vector using `unrollIdx`. `VectorizedCacheStoreOp` behaves similarly but with `AffineStoreOp` and `InsertElementOp`.
- These replacements don't directly vectorize ops, but constructs the graph in a way that MLIR / LLVM more easily recognizes vectorization patterns.
- We've also tried using vector dialect InsertOp / ExtractOp and index-casting AffineForOp induction variables to `I64`'s but those don't get recognized as maximally vectorized loads like (Extract|Insert)ElementOp usage does

## Cache Copy
- The optimal `CacheCopyOp` will fill all SIMD registers with src data, then unload all of them to dst memory and repeat
    ```
    vmovups -0x100(%rax),%ymm0
    vmovups -0xe0(%rax),%ymm1
    ...
    vmovups 0xc0(%rax),%ymm14
    vmovups 0xe0(%rax),%ymm15

    vmovaps %ymm0,-0x38020(%rdx)
    vmovaps %ymm1,-0x38000(%rdx)
    ...
    vmovaps %ymm14,-0x20(%rdx)
    vmovaps %ymm15,(%rdx)

    inc    %rcx
    add    $0x40,%rdx
    add    $0x1000,%rax
    cmp    $0x1ff,%rcx
    jb     (top)
    ```
- Note that SIMD ops have multi-cycle latency but multiple ALU ports can be used simultaneously, so running multiple sequential vector ops that don't use the same register / memory can hide some of the latency, which makes the above style of copying more performant than something like
    ```
    vmovups (%rax),%ymm0
    vmovaps %ymm0,(%rdx)
    (increment)
    jb (top)
    ```
- To generate vectorized code like the above, nested LoopNests and a temporary buffer are required:
    - Suppose a `512x128` region needs to be cached, and at most `128` elements can be fit into all SIMD registers simultaneously, then the outer LoopNest traverses the full `512x128` region while the inner LoopNests operate over the fully-vectorizable `128` elements.
    - In general, the multi-dimensional fully vectorizable shape is calculated based on the vectorization information and the LoopNest is split accordingly
    - The inner vectorizable section is 2 sequential LoopNests, one which fills a temporary buffer with input data, and another that unloads the temporary buffer to cache memory
    - The goal of this structure is to have MLIR / LLVM recognize that it can fold out the temporary buffer and just use SIMD registers, which requires a degree of unrolling and vector type usage in the inner loopnests
    ```
    // Suppose 2-D sample
    vectorizable_shape = compute_vectorizable_subtensor({input_cache_rows, input_cache_columns}, vectorization_info)
    tmpBuffer = allocate(vectorizable_shape)

    // Outer LoopNest
    for outer_row = [0, input_cache_rows, vectorizable_shape[0]]
    {
        for outer_col = [0, input_cache_columns, vectorizable_shape[1]]
        {
            // Inner LoopNest 1, move input data -> temporary buffer
            for inner_row = [0, vectorizable_shape[0], 1]
            {
                for inner_col = [0, vectorizable_shape[1], 1] // Vectorized
                {
                    tmpBuffer[inner_row, inner_col] = input[ inputMapping( outer_row, outer_col, inner_row, inner_col ) ]
                }
            }
            // Inner LoopNest 2, move temporary buffer -> cache memory
            for inner_row = [0, vectorizable_shape[0], 1]
            {
                for inner_col = [0, vectorizable_shape[1], 1] // Vectorized
                {
                    cache[ cacheMapping( outer_row, outer_col, inner_row, inner_col ) ] = tmpBuffer[inner_row, inner_col]
                }
            }
        }
    }
    ```

## Cache Reduce
- Note: cache reduce is not optimized yet in Accera V1
- Cache reduce is similar to cache copy, but instead of just doing a move from cache to output, there is also an accumulate step
- With an output cache sized such that it can be completely contained in SIMD registers (as is optimal in GEMM), the optimal cache reduce would look like this:
    ```
    (assume ymm0-ymm11 contains kernel results)

    vaddps (%rbx,%rcx,4),%ymm0,%ymm0
    vaddps 0x20(%rbx,%rcx,4),%ymm1,%ymm1
    ...
    vaddps 0x5000(%rbx,%rcx,4),%ymm10,%ymm10
    vaddps 0x5020(%rbx,%rcx,4),%ymm11,%ymm11

    vmovups %ymm0,(%rbx,%rcx,4)
    vmovups %ymm1,0x20(%rbx,%rcx,4)
    ...
    vmovups %ymm10,0x5000(%rbx,%rcx,4)
    vmovups %ymm11,0x5020(%rbx,%rcx,4)
    ```
- The above optimizes for SIMD op latency for the same reasons discussed in the Cache Copy section
- To attempt to achieve this, nested loopnests and a temporary buffer are used again as in Cache Copy
- The version that worked in Accera V0 has 3 inner loopnests:
    1. Load cache data to temporary buffer
    2. Add current output data to temporary buffer
    3. Write temporary buffer to output data
- The current "best" version in Accera V1, which is still suboptimal uses 3 inner loopnests slightly differently:
    1. Load current output data to temporary buffer
    2. Add temporary buffer and cache data as full vectors rather than individual elements
    3. Write temporary buffer to output data

### Cache Reduce Issues
- We haven't been able to get Accera V0-like optimal reduce assembly code emitting on Intel AVX-2 CPUs yet (only hardware tested on).
- MLIR / LLVM don't appear to recognize that the temporary buffer can be folded out and will also interleave ops acting on the same SIMD register:
    ```
    lea    (%r15,%rax,1),%rbx
    vmovups (%r14,%rbx,4),%ymm12   <-- Loads output memory into SIMD register (bad)
    vmovaps %ymm12,0xa0(%rsp)      <-- Stores output memory into temporary buffer. This indicatse that the temporary buffer isn't being folded out. (bad)

    ...

    add    %r11,%rdi
    vmovups (%r14,%rdi,4),%ymm12
    vmovaps %ymm12,0x200(%rsp)

    vaddps 0xa0(%rsp),%ymm11,%ymm11     <-- Keeps ymm11 value from kernel result, so output cache is being folded out and only exists in SIMD registers (good)

    vmovaps %ymm11,0xa0(%rsp)           <-- Immediately storing ymm11 accumulated value out to temporary buffer memory (bad), which has to block until the above instruction completes (bad)

    ...

    vaddps 0x200(%rsp),%ymm0,%ymm0

    vmovaps %ymm0,0x200(%rsp)

    mov    $0xffffffffffffffff,%rax
    mov    %r13,%rdi
    lea    0xc0(%rsp),%rbx
    nop

    vmovaps -0x20(%rbx),%ymm0       <-- Loads temporary buffer data into SIMD register
    vmovups %ymm0,-0x3c(%rdi)       <-- Stores data to output buffer, which is using ymm0 immediately after filling it, so is blocking on the vmovaps latency (bad)
    vmovaps (%rbx),%ymm0
    vmovups %ymm0,-0x1c(%rdi)
    inc    %rax
    add    $0x40,%rbx
    add    $0x1000,%rdi
    cmp    $0x5,%rax
    jb     (vmovaps -0x20(%rbx),%ymm0 line)
    ```

- Note: In `accera\mlirHelpers\testbed\src\VectorizedReduceTests.cpp` there are several samples that produce properly vectorized code, and generate identical IR to the above full end-to-end case, however MLIR/LLVM doesn't seem to be recognizing the same pattern in the full end-to-end GEMM case.

- Several variants have been attempted, all being equivalent or worse than the above result:
    - Change unrolling on all loopnests
    - Triple inner loopnest: (1) cache -> tmp buf, (2) tmp buf + output -> tmp buf, (2) tmp buf -> output (i.e. the Accera V0 style)
    - Double inner loopnest: (1) output + cache -> cache, (2) cache -> output
    - Double inner loopnest: (1) output + cache -> tmp buf, (2) tmp buf -> output
    - Move reduce code to separate FuncOp and call that
        - Without inlining, the output cache data in the SIMD registers is written out to memory (and therefore the output cache memory buffer is materialized), which is inefficient
        - If inlining happens too early in the lowering process, then it is exactly the same as if FuncOp was not used and the IR was emitted directly in place
        - Can't turn just the inner vectorized section into a FuncOp due to the `DimSizeOp` being used to determine the sizes and currently those results don't work as function arguments that are then used as LoopNest ranges as `mlir::Value` objects.
    - Tried just copying output cache data out without any accumulate, which doesn't propery vectorize either. Tried with both direct cache -> output and with an intermediate tmp buf like a reversed cache copy.
    - In VectorizedReduceTests, tried using `AffineVectorLoadOp` for loading vectors from the non-vector-type output buffer, but it produces very complex vectorization code that is clearly not optimized for the simple case

- Recurring issues:
    - Actually materializing the tmp buffer on the stack rather than folding actions on the tmp buffer into operations on the output cache SIMD registers
    - Actually materializing the output cache memory on the stack. The current version doesn't have this problem
    - Interleaving accumulate and copy-out instructions, even though the IR doesn't interleave them. This is suboptimal due to SIMD op latency.

- Some things worth investigating further:
    - Create a structure to represent SIMD registers and run our own custom passes to fold those usages together rather than relying on temp buffers and LLVM
        - Maybe it could just be indexed attributes on ops that are matched up and folded in a later custom pass?
    - Replace unrolled vectorized loops with non-unrolled loops that efficiently index into the vector types. This is how Accera V0 solved register spill issues, but vector dialect InsertOp / ExtractOp with index-cast induction variables haven't worked efficiently for this so far in MLIR.
    - Cast all memrefs to full vector types, including input buffers (e.g. cast `memref<1024 x 1024 x f32> to memref<vector<1024 x 1024 x f32>>`) and use vector dialect ops more heavily
    - Custom passes to replace ops with explicit vector dialect ops
    - Std.viewop or other reshaping ops to reshape the input buffers from `memref<row x col x f32> to  memref<row x (col / vec) x vector<vec x f32>>`
