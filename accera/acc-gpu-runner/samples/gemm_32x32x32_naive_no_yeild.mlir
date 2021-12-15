// matrix size = 32x32
// block size = 4
// currently does not work because of https://github.com/llvm/llvm-project/blob/55565752306e352e655bf8a4ba919c14d6b195c2/mlir/lib/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.cpp#L222

module @gpu_module3 attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 32 : i32, max_compute_workgroup_size = dense<[32, 32, 64]> : vector<3xi32>}>} {
  gpu.module @kernels {
    gpu.func @dot(%a: memref<32x32xf32>,  %b: memref<32x32xf32>, %c: memref<32x32xf32>) 
      kernel attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
      %accum_ref = alloc() : memref<1xf32,5> // SPIRV does not like alloca so we allocate private / local / register memory (unsure why alloc : memref<1xf32, 5> does not work )
      
      %blkidx = "gpu.block_id"() {dimension = "x"} : () -> index
      %blkidy = "gpu.block_id"() {dimension = "y"} : () -> index
      %tidx = "gpu.thread_id"() {dimension = "x"} : () -> index
      %tidy = "gpu.thread_id"() {dimension = "y"} : () -> index
      %bdimx = constant 4 : index
      %bdimy = constant 4 : index

      %0 = muli %blkidx, %bdimx: index
      %i = addi %0, %tidx: index // gidx = threadIdx.x + 4*blockIdx.x
      %1 = muli %blkidy, %bdimy: index
      %j = addi %1, %tidy: index // gidy = threadIdx.y + 4*blockIdx.y

      %zerof = constant 0.0 : f32
      %zero = constant 0 : index
      %n = constant 32 : index
      %one = constant 1 : index

      store %zerof, %accum_ref[%zero] : memref<1xf32,5>

      // accum = sum(A[i,:]*B[:,j])
      scf.for %k = %zero to %n step %one {
        %ai = load %a[%i, %k] : memref<32x32xf32>
        %bi = load %b[%k, %j] : memref<32x32xf32>

        %t = mulf %ai, %bi: f32
        %accum_prev = load %accum_ref[%zero] : memref<1xf32,5>
        %accum_next = addf %accum_prev, %t : f32
        store %accum_next, %accum_ref[%zero] : memref<1xf32,5>
      }

      %accum = load %accum_ref[%zero] : memref<1xf32,5>

      store %accum, %c[%i, %j] : memref<32x32xf32>
      dealloc %accum_ref : memref<1xf32,5>
      gpu.return
    }
  }
  func @main() {
    %onef = constant 1.0 : f32


    %gdimx = constant 8 : index // 32/4
    %gdimy = constant 8 : index // 32/4
    %bdimx = constant 4 : index
    %bdimy = constant 4 : index
    %one = constant 1 : index

    // Allocate the matricies and fill then with all ones
    %a = alloc() : memref<32x32xf32>
    %b = alloc() : memref<32x32xf32>
    %c = alloc() : memref<32x32xf32>
    %a0 = memref_cast %a : memref<32x32xf32> to memref<?x?xf32>
    call @fillResource2DFloat(%a0, %onef) : (memref<?x?xf32>, f32) -> ()
    %b0 = memref_cast %b : memref<32x32xf32> to memref<?x?xf32>
    call @fillResource2DFloat(%b0, %onef) : (memref<?x?xf32>, f32) -> ()
    %c0 = memref_cast %c : memref<32x32xf32> to memref<?x?xf32>
    call @fillResource2DFloat(%c0, %onef) : (memref<?x?xf32>, f32) -> ()

    // launch gpu kernel
    "gpu.launch_func"(%gdimx, %gdimy, %one, %bdimx, %bdimy, %one, %a, %b, %c) {kernel = @kernels::@dot} : (index, index, index, index, index, index, memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()

    %c2 = memref_cast %c : memref<32x32xf32> to memref<*xf32>
    call @print_memref_f32(%c2) : (memref<*xf32>) -> ()
    return
  }
  func @fillResource2DFloat(memref<?x?xf32>, f32)
  func @print_memref_f32(memref<*xf32>)
}
