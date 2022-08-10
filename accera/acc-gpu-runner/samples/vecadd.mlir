#map0 = affine_map<()[s0] -> (s0)>




module @gpu_module2 attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  gpu.module @kernels {
    gpu.func @gpu_f1_14996649427853758377(%arg0: memref<16384xf32>, %arg1: memref<16384xf32>, %arg2: memref<16384xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[1024, 1, 1]> : vector<3xi32>}} {
      %0 = gpu.block_id x loc(unknown)
      %1 = arith.index_cast %0 : index to i64 loc(unknown)
      %2 = gpu.thread_id x loc(unknown)
      %3 = arith.index_cast %2 : index to i64 loc(unknown)
      %c1024_i64 = arith.constant 1024 : i64 loc(unknown)
      %4 = "accv.bin_op"(%1, %c1024_i64) {predicate = 2 : i64} : (i64, i64) -> i64 loc(unknown)
      %5 = "accv.bin_op"(%4, %3) {predicate = 0 : i64} : (i64, i64) -> i64 loc(unknown)
      %6 = arith.index_cast %5 : i64 to index loc(unknown)
      %7 = "accv.slice"(%arg0, %6) {sliceDimensions = [0]} : (memref<16384xf32>, index) -> memref<f32, #map0> loc(unknown)
      %8 = arith.index_cast %5 : i64 to index loc(unknown)
      %9 = "accv.slice"(%arg1, %8) {sliceDimensions = [0]} : (memref<16384xf32>, index) -> memref<f32, #map0> loc(unknown)
      %10 = "accv.get_element"(%7) : (memref<f32, #map0>) -> f32 loc(unknown)
      %11 = "accv.get_element"(%9) : (memref<f32, #map0>) -> f32 loc(unknown)
      %12 = "accv.bin_op"(%10, %11) {predicate = 0 : i64} : (f32, f32) -> f32 loc(unknown)
      %13 = arith.index_cast %5 : i64 to index loc(unknown)
      %14 = "accv.slice"(%arg2, %13) {sliceDimensions = [0]} : (memref<16384xf32>, index) -> memref<f32, #map0> loc(unknown)
      "accv.copy"(%12, %14) : (f32, memref<f32, #map0>) -> () loc(unknown)
      gpu.return loc(unknown)
    } loc(unknown)
  } loc(unknown)
  func @f1_14996649427853758377(%arg0: memref<16384xf32, 3>, %arg1: memref<16384xf32, 3>, %arg2: memref<16384xf32,3 >) { // the 3 denotes that it is using global address space
    %c16 = arith.constant 16 : index loc(unknown)
    %c1 = arith.constant 1 : index loc(unknown)
    %c1_0 = arith.constant 1 : index loc(unknown)
    %c1024 = arith.constant 1024 : index loc(unknown)
    %c1_1 = arith.constant 1 : index loc(unknown)
    %c1_2 = arith.constant 1 : index loc(unknown)
    "gpu.launch_func"(%c16, %c1, %c1_0, %c1024, %c1_1, %c1_2, %arg0, %arg1, %arg2) {kernel = @kernels::@gpu_f1_14996649427853758377} : (index, index, index, index, index, index, memref<16384xf32>, memref<16384xf32>, memref<16384xf32>) -> () loc(unknown)
    return loc(unknown)
  } loc(unknown)
} loc(unknown)
