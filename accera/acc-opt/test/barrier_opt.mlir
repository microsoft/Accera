// RUN: acc-opt --verify-each=false --optimize-barriers %s | FileCheck %s

// CHECK-LABEL: module @barrier_test_1
// CHECK: %2 = "accv.alloc"()
// CHECK-NEXT: %3 = "accv.alloc"() : () -> memref<16xf32, 3>
// CHECK-NEXT: %4 = affine.load %arg0[symbol(%0) + symbol(%1) * 16] : memref<1xf32>
// CHECK-NEXT: affine.store %4, %2[symbol(%0) + symbol(%1) * 16] : memref<16xf32, 3>
// CHECK-NEXT: %5 = affine.load %arg1[symbol(%0) + symbol(%1) * 16] : memref<1xf32>
// CHECK-NEXT: affine.store %5, %2[symbol(%0) + symbol(%1) * 16] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %6 = affine.load %2[symbol(%0) + symbol(%1) * 16] : memref<16xf32, 3>
// CHECK-NEXT: %7 = affine.load %3[symbol(%0) + symbol(%1) * 16] : memref<16xf32, 3>
// CHECK-NEXT: %8 = "accv.bin_op"(%6, %7) {predicate = 0 : i64} : (f32, f32) -> f32
// CHECK-NEXT: affine.store %8, %arg2[symbol(%0) + symbol(%1) * 16] : memref<1xf32>
// CHECK: accv.return
module @barrier_test_1 attributes {llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  accv.module "barrier_test_1"  {
    accv.func nested @barrier_test_1_d9502818_impl_8438933964186859281(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: memref<1xf32>) attributes {exec_target = 0 : i64} {
      "accv.lambda"() ( {
        %0 = "gpu.thread_id"() {dimension = "x"} : () -> index
        %1 = "gpu.block_id"() {dimension = "x"} : () -> index
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 1 {
              affine.for %arg6 = 0 to 1 {
                %2 = "accv.alloc"() : () -> memref<16xf32, 3>
                %3 = "accv.alloc"() : () -> memref<16xf32, 3>
                "accv.barrier"() {scope = "Block"} : () -> ()
                %4 = affine.load %arg0[symbol(%0) + symbol(%1) * 16] : memref<1xf32>
                affine.store %4, %2[symbol(%0) + symbol(%1) * 16] : memref<16xf32, 3>
                "accv.barrier"() {scope = "Block"} : () -> ()
                %5 = affine.load %arg1[symbol(%0) + symbol(%1) * 16] : memref<1xf32>
                affine.store %5, %2[symbol(%0) + symbol(%1) * 16] : memref<16xf32, 3>
                "accv.barrier"() {scope = "Block"} : () -> ()
                %6 = affine.load %2[symbol(%0) + symbol(%1) * 16] : memref<16xf32, 3>
                %7 = affine.load %3[symbol(%0) + symbol(%1) * 16] : memref<16xf32, 3>
                %8 = "accv.bin_op"(%6, %7) {predicate = 0 : i64} : (f32, f32) -> f32
                affine.store %8, %arg2[symbol(%0) + symbol(%1) * 16] : memref<1xf32>
                "accv.barrier"() {scope = "Block"} : () -> ()
              } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{j_i,5}">, kernels = ["_"], accv_gpu_map = "ThreadY", subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [1, 1]}
            } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{i_i,3}">, accv_gpu_map = "ThreadX", subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [1, 16]}
          } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{j_o,4}">, accv_gpu_map = "BlockY", subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [16, 16]}
        } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{i_o,2}">, accv_gpu_map = "BlockX", subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [16, 256]}
        accv.return
      }) {exec_target = 1 : i64, gpu_launch = [16 : index, 16 : index, 1 : index, 16 : index, 16 : index, 1 : index], sym_name = "NestFunction_0", type = () -> ()} : () -> ()
      accv.return
    }
    accv.func @barrier_test_1_d9502818(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: memref<1xf32>) attributes {accv.base_name = "barrier_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
      accv.launch_func @barrier_test_1_d9502818_impl_8438933964186859281(%arg0, %arg1, %arg2) {exec_target = 0 : i64, gpu_launch = "gpu_launch"} : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
      accv.return
    }
  }
}

