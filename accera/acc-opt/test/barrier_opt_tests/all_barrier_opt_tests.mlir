// RUN: acc-opt --verify-each=false --optimize-barriers %s | FileCheck %s

// CHECK-LABEL: module @barrier_if_test_1
// CHECK-NEXT: func nested @barrier_if_test_1_d8c83f18cb7155ef_impl_8950868249543779986(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_if_test_1_d8c83f18cb7155ef(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {accv.base_name = "barrier_if_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_if_test_1_d8c83f18cb7155ef_impl_8950868249543779986(%arg0, %arg1) : (memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %cst = arith.constant 2.000000e+00 : f32
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %2 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %2, %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: %3 = arith.cmpi slt, %0, %c4096 : index
// CHECK-NEXT: scf.if %3 {
// CHECK-NEXT: %5 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: %6 = arith.addf %5, %cst {RelaxedPrecision} : f32
// CHECK-NEXT: memref.store %6, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: %7 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %7, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: }
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %4 = memref.load %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: memref.store %4, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_if_test_1 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_if_test_1_d8c83f18cb7155ef_impl_8950868249543779986(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>)
    return
  }
  func @barrier_if_test_1_d8c83f18cb7155ef(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {accv.base_name = "barrier_if_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_if_test_1_d8c83f18cb7155ef_impl_8950868249543779986(%arg0, %arg1) : (memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
      %cst = arith.constant 2.000000e+00 : f32
      %c4096 = arith.constant 4096 : index
      %0 = gpu.thread_id x
      %1 = memref.alloc() : memref<16xf32, 3>
      %2 = memref.load %arg0[%0] : memref<4096xf32>
      memref.store %2, %1[%0] : memref<16xf32, 3>
      %3 = arith.cmpi slt, %0, %c4096 : index
      scf.if %3 {
        %5 = memref.load %arg0[%0] : memref<4096xf32>
        %6 = arith.addf %5, %cst {RelaxedPrecision} : f32
        memref.store %6, %arg0[%0] : memref<4096xf32>
        %7 = memref.load %arg0[%0] : memref<4096xf32>
        memref.store %7, %arg0[%0] : memref<4096xf32>
      }
      "accv.barrier"() {scope = "Block"} : () -> ()
      %4 = memref.load %1[%0] : memref<16xf32, 3>
      memref.store %4, %arg0[%0] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_if_test_2
// CHECK-NEXT: func nested @barrier_if_test_2_db48cbeba64fd50d_impl_8950868249543779986(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_if_test_2_db48cbeba64fd50d(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {accv.base_name = "barrier_if_test_2", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_if_test_2_db48cbeba64fd50d_impl_8950868249543779986(%arg0, %arg1) : (memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %2 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %3 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %3, %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: %4 = arith.cmpi slt, %0, %c4096 : index
// CHECK-NEXT: scf.if %4 {
// CHECK-NEXT: %6 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %6, %2[%0] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %7 = memref.load %2[%0] : memref<16xf32, 3>
// CHECK-NEXT: memref.store %7, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: }
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %5 = memref.load %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: memref.store %5, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_if_test_2 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_if_test_2_db48cbeba64fd50d_impl_8950868249543779986(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>)
    return
  }
  func @barrier_if_test_2_db48cbeba64fd50d(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {accv.base_name = "barrier_if_test_2", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_if_test_2_db48cbeba64fd50d_impl_8950868249543779986(%arg0, %arg1) : (memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
      %c4096 = arith.constant 4096 : index
      %0 = gpu.thread_id x
      %1 = memref.alloc() : memref<16xf32, 3>
      %2 = memref.alloc() : memref<16xf32, 3>
      %3 = memref.load %arg0[%0] : memref<4096xf32>
      memref.store %3, %1[%0] : memref<16xf32, 3>
      %4 = arith.cmpi slt, %0, %c4096 : index
      scf.if %4 {
        %6 = memref.load %arg0[%0] : memref<4096xf32>
        memref.store %6, %2[%0] : memref<16xf32, 3>
        "accv.barrier"() {scope = "Block"} : () -> ()
        %7 = memref.load %2[%0] : memref<16xf32, 3>
        memref.store %7, %arg0[%0] : memref<4096xf32>
      }
      "accv.barrier"() {scope = "Block"} : () -> ()
      %5 = memref.load %1[%0] : memref<16xf32, 3>
      memref.store %5, %arg0[%0] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_if_test_3
// CHECK-NEXT: func nested @barrier_if_test_3_a8d03b514a5ecca2_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg2 : memref<4096xf32>, %arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_if_test_3_a8d03b514a5ecca2(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_if_test_3", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_if_test_3_a8d03b514a5ecca2_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %2 = memref.load %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: memref.store %2, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %3 = arith.cmpi slt, %0, %c4096 : index
// CHECK-NEXT: scf.if %3 {
// CHECK-NEXT: %5 = memref.load %arg1[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %5, %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: } else {
// CHECK-NEXT: %5 = memref.load %arg2[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %5, %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: }
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %4 = memref.load %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: memref.store %4, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_if_test_3 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_if_test_3_a8d03b514a5ecca2_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg2 : memref<4096xf32>, %arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
    return
  }
  func @barrier_if_test_3_a8d03b514a5ecca2(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_if_test_3", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_if_test_3_a8d03b514a5ecca2_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
      %c4096 = arith.constant 4096 : index
      %0 = gpu.thread_id x
      %1 = memref.alloc() : memref<16xf32, 3>
      %2 = memref.load %1[%0] : memref<16xf32, 3>
      memref.store %2, %arg0[%0] : memref<4096xf32>
      "accv.barrier"() {scope = "Block"} : () -> ()
      %3 = arith.cmpi slt, %0, %c4096 : index
      scf.if %3 {
        %5 = memref.load %arg1[%0] : memref<4096xf32>
        memref.store %5, %1[%0] : memref<16xf32, 3>
      } else {
        %5 = memref.load %arg2[%0] : memref<4096xf32>
        memref.store %5, %1[%0] : memref<16xf32, 3>
      }
      "accv.barrier"() {scope = "Block"} : () -> ()
      %4 = memref.load %1[%0] : memref<16xf32, 3>
      memref.store %4, %arg0[%0] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_if_test_4
// CHECK-NEXT: func nested @barrier_if_test_4_b93731f0de8ae55f_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c256 = arith.constant 256 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_if_test_4_b93731f0de8ae55f(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_if_test_4", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_if_test_4_b93731f0de8ae55f_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = gpu.block_dim x
// CHECK-NEXT: %2 = arith.muli %1, %c16 : index
// CHECK-NEXT: %3 = arith.addi %0, %2 : index
// CHECK-NEXT: %4 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %5 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %6 = arith.muli %1, %c16 : index
// CHECK-NEXT: %7 = arith.addi %0, %6 : index
// CHECK-NEXT: %8 = memref.load %arg0[%7] : memref<4096xf32>
// CHECK-NEXT: %9 = arith.muli %1, %c16 : index
// CHECK-NEXT: %10 = arith.addi %0, %9 : index
// CHECK-NEXT: memref.store %8, %4[%10] : memref<16xf32, 3>
// CHECK-NEXT: %11 = arith.muli %1, %c16 : index
// CHECK-NEXT: %12 = arith.addi %0, %11 : index
// CHECK-NEXT: %13 = memref.load %arg1[%12] : memref<4096xf32>
// CHECK-NEXT: %14 = arith.muli %1, %c16 : index
// CHECK-NEXT: %15 = arith.addi %0, %14 : index
// CHECK-NEXT: memref.store %13, %5[%15] : memref<16xf32, 3>
// CHECK-NEXT: %16 = arith.cmpi slt, %3, %c4096 : index
// CHECK-NEXT: scf.if %16 {
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %27 = arith.muli %1, %c16 : index
// CHECK-NEXT: %28 = arith.addi %0, %27 : index
// CHECK-NEXT: %29 = memref.load %4[%28] : memref<16xf32, 3>
// CHECK-NEXT: %30 = arith.muli %1, %c16 : index
// CHECK-NEXT: %31 = arith.addi %0, %30 : index
// CHECK-NEXT: memref.store %29, %arg1[%31] : memref<4096xf32>
// CHECK-NEXT: } else {
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %27 = arith.muli %1, %c16 : index
// CHECK-NEXT: %28 = arith.addi %0, %27 : index
// CHECK-NEXT: %29 = memref.load %5[%28] : memref<16xf32, 3>
// CHECK-NEXT: %30 = arith.muli %1, %c16 : index
// CHECK-NEXT: %31 = arith.addi %0, %30 : index
// CHECK-NEXT: memref.store %29, %arg0[%31] : memref<4096xf32>
// CHECK-NEXT: }
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %17 = arith.muli %1, %c16 : index
// CHECK-NEXT: %18 = arith.addi %0, %17 : index
// CHECK-NEXT: %19 = memref.load %arg0[%18] : memref<4096xf32>
// CHECK-NEXT: %20 = arith.muli %1, %c16 : index
// CHECK-NEXT: %21 = arith.addi %0, %20 : index
// CHECK-NEXT: memref.store %19, %4[%21] : memref<16xf32, 3>
// CHECK-NEXT: %22 = arith.muli %1, %c16 : index
// CHECK-NEXT: %23 = arith.addi %0, %22 : index
// CHECK-NEXT: %24 = memref.load %arg0[%23] : memref<4096xf32>
// CHECK-NEXT: %25 = arith.muli %1, %c16 : index
// CHECK-NEXT: %26 = arith.addi %0, %25 : index
// CHECK-NEXT: memref.store %24, %5[%26] : memref<16xf32, 3>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_if_test_4 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_if_test_4_b93731f0de8ae55f_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
    return
  }
  func @barrier_if_test_4_b93731f0de8ae55f(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_if_test_4", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_if_test_4_b93731f0de8ae55f_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
      %c4096 = arith.constant 4096 : index
      %c16 = arith.constant 16 : index
      %0 = gpu.thread_id x
      %1 = gpu.block_dim x
      %2 = arith.muli %1, %c16 : index
      %3 = arith.addi %0, %2 : index
      %4 = memref.alloc() : memref<16xf32, 3>
      %5 = memref.alloc() : memref<16xf32, 3>
      %6 = arith.muli %1, %c16 : index
      %7 = arith.addi %0, %6 : index
      %8 = memref.load %arg0[%7] : memref<4096xf32>
      %9 = arith.muli %1, %c16 : index
      %10 = arith.addi %0, %9 : index
      memref.store %8, %4[%10] : memref<16xf32, 3>
      %11 = arith.muli %1, %c16 : index
      %12 = arith.addi %0, %11 : index
      %13 = memref.load %arg1[%12] : memref<4096xf32>
      %14 = arith.muli %1, %c16 : index
      %15 = arith.addi %0, %14 : index
      memref.store %13, %5[%15] : memref<16xf32, 3>
      %16 = arith.cmpi slt, %3, %c4096 : index
      scf.if %16 {
        "accv.barrier"() {scope = "Block"} : () -> ()
        %27 = arith.muli %1, %c16 : index
        %28 = arith.addi %0, %27 : index
        %29 = memref.load %4[%28] : memref<16xf32, 3>
        %30 = arith.muli %1, %c16 : index
        %31 = arith.addi %0, %30 : index
        memref.store %29, %arg1[%31] : memref<4096xf32>
      } else {
        "accv.barrier"() {scope = "Block"} : () -> ()
        %27 = arith.muli %1, %c16 : index
        %28 = arith.addi %0, %27 : index
        %29 = memref.load %5[%28] : memref<16xf32, 3>
        %30 = arith.muli %1, %c16 : index
        %31 = arith.addi %0, %30 : index
        memref.store %29, %arg0[%31] : memref<4096xf32>
      }
      "accv.barrier"() {scope = "Block"} : () -> ()
      %17 = arith.muli %1, %c16 : index
      %18 = arith.addi %0, %17 : index
      %19 = memref.load %arg0[%18] : memref<4096xf32>
      %20 = arith.muli %1, %c16 : index
      %21 = arith.addi %0, %20 : index
      memref.store %19, %4[%21] : memref<16xf32, 3>
      %22 = arith.muli %1, %c16 : index
      %23 = arith.addi %0, %22 : index
      %24 = memref.load %arg0[%23] : memref<4096xf32>
      %25 = arith.muli %1, %c16 : index
      %26 = arith.addi %0, %25 : index
      memref.store %24, %5[%26] : memref<16xf32, 3>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_loop_test_1
// CHECK-NEXT: func nested @barrier_loop_test_1_e7e77c1476feb6e6_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c256 = arith.constant 256 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_loop_test_1_e7e77c1476feb6e6(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_loop_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_loop_test_1_e7e77c1476feb6e6_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %c32 = arith.constant 32 : index
// CHECK-NEXT: %c0 = arith.constant 0 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = gpu.block_dim x
// CHECK-NEXT: %2 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %3 = arith.muli %1, %c16 : index
// CHECK-NEXT: %4 = arith.addi %0, %3 : index
// CHECK-NEXT: %5 = memref.load %arg0[%4] : memref<4096xf32>
// CHECK-NEXT: %6 = arith.muli %1, %c16 : index
// CHECK-NEXT: %7 = arith.addi %0, %6 : index
// CHECK-NEXT: memref.store %5, %2[%7] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT: %13 = arith.muli %1, %c16 : index
// CHECK-NEXT: %14 = arith.addi %0, %13 : index
// CHECK-NEXT: %15 = memref.load %2[%14] : memref<16xf32, 3>
// CHECK-NEXT: %16 = arith.muli %1, %c16 : index
// CHECK-NEXT: %17 = arith.addi %0, %16 : index
// CHECK-NEXT: memref.store %15, %arg0[%17] : memref<4096xf32>
// CHECK-NEXT: } {sym_name = "value_loop"}
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %8 = arith.muli %1, %c16 : index
// CHECK-NEXT: %9 = arith.addi %0, %8 : index
// CHECK-NEXT: %10 = memref.load %arg1[%9] : memref<4096xf32>
// CHECK-NEXT: %11 = arith.muli %1, %c16 : index
// CHECK-NEXT: %12 = arith.addi %0, %11 : index
// CHECK-NEXT: memref.store %10, %2[%12] : memref<16xf32, 3>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_loop_test_1 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_loop_test_1_e7e77c1476feb6e6_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
    return
  }
  func @barrier_loop_test_1_e7e77c1476feb6e6(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_loop_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_loop_test_1_e7e77c1476feb6e6_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %0 = gpu.thread_id x
      %1 = gpu.block_dim x
      %2 = memref.alloc() : memref<16xf32, 3>
      %3 = arith.muli %1, %c16 : index
      %4 = arith.addi %0, %3 : index
      %5 = memref.load %arg0[%4] : memref<4096xf32>
      %6 = arith.muli %1, %c16 : index
      %7 = arith.addi %0, %6 : index
      memref.store %5, %2[%7] : memref<16xf32, 3>
      "accv.barrier"() {scope = "Block"} : () -> ()
      scf.for %arg2 = %c0 to %c32 step %c1 {
        %13 = arith.muli %1, %c16 : index
        %14 = arith.addi %0, %13 : index
        %15 = memref.load %2[%14] : memref<16xf32, 3>
        %16 = arith.muli %1, %c16 : index
        %17 = arith.addi %0, %16 : index
        memref.store %15, %arg0[%17] : memref<4096xf32>
      } {sym_name = "value_loop"}
      "accv.barrier"() {scope = "Block"} : () -> ()
      %8 = arith.muli %1, %c16 : index
      %9 = arith.addi %0, %8 : index
      %10 = memref.load %arg1[%9] : memref<4096xf32>
      %11 = arith.muli %1, %c16 : index
      %12 = arith.addi %0, %11 : index
      memref.store %10, %2[%12] : memref<16xf32, 3>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_loop_test_2
// CHECK-NEXT: func nested @barrier_loop_test_2_a5577eaf17057c99_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c256 = arith.constant 256 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_loop_test_2_a5577eaf17057c99(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_loop_test_2", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_loop_test_2_a5577eaf17057c99_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %c32 = arith.constant 32 : index
// CHECK-NEXT: %c0 = arith.constant 0 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = gpu.block_dim x
// CHECK-NEXT: %2 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %3 = arith.muli %1, %c16 : index
// CHECK-NEXT: %4 = arith.addi %0, %3 : index
// CHECK-NEXT: %5 = memref.load %2[%4] : memref<16xf32, 3>
// CHECK-NEXT: %6 = arith.muli %1, %c16 : index
// CHECK-NEXT: %7 = arith.addi %0, %6 : index
// CHECK-NEXT: memref.store %5, %arg0[%7] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT: %13 = arith.muli %1, %c16 : index
// CHECK-NEXT: %14 = arith.addi %0, %13 : index
// CHECK-NEXT: %15 = memref.load %arg1[%14] : memref<4096xf32>
// CHECK-NEXT: %16 = arith.muli %1, %c16 : index
// CHECK-NEXT: %17 = arith.addi %0, %16 : index
// CHECK-NEXT: memref.store %15, %2[%17] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %18 = arith.muli %1, %c16 : index
// CHECK-NEXT: %19 = arith.addi %0, %18 : index
// CHECK-NEXT: %20 = memref.load %2[%19] : memref<16xf32, 3>
// CHECK-NEXT: %21 = arith.muli %1, %c16 : index
// CHECK-NEXT: %22 = arith.addi %0, %21 : index
// CHECK-NEXT: memref.store %20, %arg0[%22] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: } {sym_name = "value_loop"}
// CHECK-NEXT: %8 = arith.muli %1, %c16 : index
// CHECK-NEXT: %9 = arith.addi %0, %8 : index
// CHECK-NEXT: %10 = memref.load %arg0[%9] : memref<4096xf32>
// CHECK-NEXT: %11 = arith.muli %1, %c16 : index
// CHECK-NEXT: %12 = arith.addi %0, %11 : index
// CHECK-NEXT: memref.store %10, %2[%12] : memref<16xf32, 3>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_loop_test_2 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_loop_test_2_a5577eaf17057c99_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
    return
  }
  func @barrier_loop_test_2_a5577eaf17057c99(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_loop_test_2", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_loop_test_2_a5577eaf17057c99_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %0 = gpu.thread_id x
      %1 = gpu.block_dim x
      %2 = memref.alloc() : memref<16xf32, 3>
      %3 = arith.muli %1, %c16 : index
      %4 = arith.addi %0, %3 : index
      %5 = memref.load %2[%4] : memref<16xf32, 3>
      %6 = arith.muli %1, %c16 : index
      %7 = arith.addi %0, %6 : index
      memref.store %5, %arg0[%7] : memref<4096xf32>
      "accv.barrier"() {scope = "Block"} : () -> ()
      scf.for %arg2 = %c0 to %c32 step %c1 {
        %13 = arith.muli %1, %c16 : index
        %14 = arith.addi %0, %13 : index
        %15 = memref.load %arg1[%14] : memref<4096xf32>
        %16 = arith.muli %1, %c16 : index
        %17 = arith.addi %0, %16 : index
        memref.store %15, %2[%17] : memref<16xf32, 3>
        "accv.barrier"() {scope = "Block"} : () -> ()
        %18 = arith.muli %1, %c16 : index
        %19 = arith.addi %0, %18 : index
        %20 = memref.load %2[%19] : memref<16xf32, 3>
        %21 = arith.muli %1, %c16 : index
        %22 = arith.addi %0, %21 : index
        memref.store %20, %arg0[%22] : memref<4096xf32>
        "accv.barrier"() {scope = "Block"} : () -> ()
      } {sym_name = "value_loop"}
      %8 = arith.muli %1, %c16 : index
      %9 = arith.addi %0, %8 : index
      %10 = memref.load %arg0[%9] : memref<4096xf32>
      %11 = arith.muli %1, %c16 : index
      %12 = arith.addi %0, %11 : index
      memref.store %10, %2[%12] : memref<16xf32, 3>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_loop_test_3
// CHECK-NEXT: func nested @barrier_loop_test_3_e2faeeab7fffeeb2_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c256 = arith.constant 256 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>, %arg2 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_loop_test_3_e2faeeab7fffeeb2(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_loop_test_3", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_loop_test_3_e2faeeab7fffeeb2_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %c32 = arith.constant 32 : index
// CHECK-NEXT: %c0 = arith.constant 0 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = gpu.block_dim x
// CHECK-NEXT: %2 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %3 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %4 = memref.load %arg0[%c0] : memref<4096xf32>
// CHECK-NEXT: %5 = arith.muli %1, %c16 : index
// CHECK-NEXT: %6 = arith.addi %0, %5 : index
// CHECK-NEXT: memref.store %4, %2[%6] : memref<16xf32, 3>
// CHECK-NEXT: scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT: %13 = arith.muli %1, %c16 : index
// CHECK-NEXT: %14 = arith.addi %0, %13 : index
// CHECK-NEXT: %15 = memref.load %arg0[%14] : memref<4096xf32>
// CHECK-NEXT: %16 = arith.muli %1, %c16 : index
// CHECK-NEXT: %17 = arith.addi %0, %16 : index
// CHECK-NEXT: memref.store %15, %2[%17] : memref<16xf32, 3>
// CHECK-NEXT: %18 = arith.muli %1, %c16 : index
// CHECK-NEXT: %19 = arith.addi %0, %18 : index
// CHECK-NEXT: %20 = memref.load %arg1[%19] : memref<4096xf32>
// CHECK-NEXT: %21 = arith.muli %1, %c16 : index
// CHECK-NEXT: %22 = arith.addi %0, %21 : index
// CHECK-NEXT: memref.store %20, %2[%22] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %23 = arith.muli %1, %c16 : index
// CHECK-NEXT: %24 = arith.addi %0, %23 : index
// CHECK-NEXT: %25 = memref.load %2[%24] : memref<16xf32, 3>
// CHECK-NEXT: %26 = arith.muli %1, %c16 : index
// CHECK-NEXT: %27 = arith.addi %0, %26 : index
// CHECK-NEXT: %28 = memref.load %3[%27] : memref<16xf32, 3>
// CHECK-NEXT: %29 = arith.addf %25, %28 {RelaxedPrecision} : f32
// CHECK-NEXT: %30 = arith.muli %1, %c16 : index
// CHECK-NEXT: %31 = arith.addi %0, %30 : index
// CHECK-NEXT: memref.store %29, %arg2[%31] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: } {sym_name = "value_loop"}
// CHECK-NEXT: %7 = arith.muli %1, %c16 : index
// CHECK-NEXT: %8 = arith.addi %0, %7 : index
// CHECK-NEXT: %9 = memref.load %arg1[%8] : memref<4096xf32>
// CHECK-NEXT: memref.store %9, %arg2[%c0] : memref<4096xf32>
// CHECK-NEXT: scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT: %13 = arith.muli %1, %c16 : index
// CHECK-NEXT: %14 = arith.addi %0, %13 : index
// CHECK-NEXT: %15 = memref.load %arg0[%14] : memref<4096xf32>
// CHECK-NEXT: %16 = arith.muli %1, %c16 : index
// CHECK-NEXT: %17 = arith.addi %0, %16 : index
// CHECK-NEXT: memref.store %15, %2[%17] : memref<16xf32, 3>
// CHECK-NEXT: %18 = arith.muli %1, %c16 : index
// CHECK-NEXT: %19 = arith.addi %0, %18 : index
// CHECK-NEXT: %20 = memref.load %arg1[%19] : memref<4096xf32>
// CHECK-NEXT: %21 = arith.muli %1, %c16 : index
// CHECK-NEXT: %22 = arith.addi %0, %21 : index
// CHECK-NEXT: memref.store %20, %2[%22] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %23 = arith.muli %1, %c16 : index
// CHECK-NEXT: %24 = arith.addi %0, %23 : index
// CHECK-NEXT: %25 = memref.load %2[%24] : memref<16xf32, 3>
// CHECK-NEXT: %26 = arith.muli %1, %c16 : index
// CHECK-NEXT: %27 = arith.addi %0, %26 : index
// CHECK-NEXT: %28 = memref.load %3[%27] : memref<16xf32, 3>
// CHECK-NEXT: %29 = arith.addf %25, %28 {RelaxedPrecision} : f32
// CHECK-NEXT: %30 = arith.muli %1, %c16 : index
// CHECK-NEXT: %31 = arith.addi %0, %30 : index
// CHECK-NEXT: memref.store %29, %arg2[%31] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: } {sym_name = "value_loop"}
// CHECK-NEXT: %10 = arith.muli %1, %c16 : index
// CHECK-NEXT: %11 = arith.addi %0, %10 : index
// CHECK-NEXT: %12 = memref.load %arg0[%11] : memref<4096xf32>
// CHECK-NEXT: memref.store %12, %arg2[%c1] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_loop_test_3 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_loop_test_3_e2faeeab7fffeeb2_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>, %arg2 : memref<4096xf32>)
    return
  }
  func @barrier_loop_test_3_e2faeeab7fffeeb2(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_loop_test_3", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_loop_test_3_e2faeeab7fffeeb2_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %0 = gpu.thread_id x
      %1 = gpu.block_dim x
      %2 = memref.alloc() : memref<16xf32, 3>
      %3 = memref.alloc() : memref<16xf32, 3>
      %4 = memref.load %arg0[%c0] : memref<4096xf32>
      %5 = arith.muli %1, %c16 : index
      %6 = arith.addi %0, %5 : index
      memref.store %4, %2[%6] : memref<16xf32, 3>
      scf.for %arg3 = %c0 to %c32 step %c1 {
        %13 = arith.muli %1, %c16 : index
        %14 = arith.addi %0, %13 : index
        %15 = memref.load %arg0[%14] : memref<4096xf32>
        %16 = arith.muli %1, %c16 : index
        %17 = arith.addi %0, %16 : index
        memref.store %15, %2[%17] : memref<16xf32, 3>
        %18 = arith.muli %1, %c16 : index
        %19 = arith.addi %0, %18 : index
        %20 = memref.load %arg1[%19] : memref<4096xf32>
        %21 = arith.muli %1, %c16 : index
        %22 = arith.addi %0, %21 : index
        memref.store %20, %2[%22] : memref<16xf32, 3>
        "accv.barrier"() {scope = "Block"} : () -> ()
        %23 = arith.muli %1, %c16 : index
        %24 = arith.addi %0, %23 : index
        %25 = memref.load %2[%24] : memref<16xf32, 3>
        %26 = arith.muli %1, %c16 : index
        %27 = arith.addi %0, %26 : index
        %28 = memref.load %3[%27] : memref<16xf32, 3>
        %29 = arith.addf %25, %28 {RelaxedPrecision} : f32
        %30 = arith.muli %1, %c16 : index
        %31 = arith.addi %0, %30 : index
        memref.store %29, %arg2[%31] : memref<4096xf32>
        "accv.barrier"() {scope = "Block"} : () -> ()
      } {sym_name = "value_loop"}
      %7 = arith.muli %1, %c16 : index
      %8 = arith.addi %0, %7 : index
      %9 = memref.load %arg1[%8] : memref<4096xf32>
      memref.store %9, %arg2[%c0] : memref<4096xf32>
      scf.for %arg3 = %c0 to %c32 step %c1 {
        %13 = arith.muli %1, %c16 : index
        %14 = arith.addi %0, %13 : index
        %15 = memref.load %arg0[%14] : memref<4096xf32>
        %16 = arith.muli %1, %c16 : index
        %17 = arith.addi %0, %16 : index
        memref.store %15, %2[%17] : memref<16xf32, 3>
        %18 = arith.muli %1, %c16 : index
        %19 = arith.addi %0, %18 : index
        %20 = memref.load %arg1[%19] : memref<4096xf32>
        %21 = arith.muli %1, %c16 : index
        %22 = arith.addi %0, %21 : index
        memref.store %20, %2[%22] : memref<16xf32, 3>
        "accv.barrier"() {scope = "Block"} : () -> ()
        %23 = arith.muli %1, %c16 : index
        %24 = arith.addi %0, %23 : index
        %25 = memref.load %2[%24] : memref<16xf32, 3>
        %26 = arith.muli %1, %c16 : index
        %27 = arith.addi %0, %26 : index
        %28 = memref.load %3[%27] : memref<16xf32, 3>
        %29 = arith.addf %25, %28 {RelaxedPrecision} : f32
        %30 = arith.muli %1, %c16 : index
        %31 = arith.addi %0, %30 : index
        memref.store %29, %arg2[%31] : memref<4096xf32>
        "accv.barrier"() {scope = "Block"} : () -> ()
      } {sym_name = "value_loop"}
      %10 = arith.muli %1, %c16 : index
      %11 = arith.addi %0, %10 : index
      %12 = memref.load %arg0[%11] : memref<4096xf32>
      memref.store %12, %arg2[%c1] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_loop_test_4
// CHECK-NEXT: func nested @barrier_loop_test_4_8a33c1a81937487e_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c256 = arith.constant 256 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>, %arg2 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_loop_test_4_8a33c1a81937487e(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_loop_test_4", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_loop_test_4_8a33c1a81937487e_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %c32 = arith.constant 32 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c0 = arith.constant 0 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = gpu.block_dim x
// CHECK-NEXT: %2 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %3 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %4 = memref.load %arg0[%c0] : memref<4096xf32>
// CHECK-NEXT: %5 = arith.muli %1, %c16 : index
// CHECK-NEXT: %6 = arith.addi %0, %5 : index
// CHECK-NEXT: memref.store %4, %2[%6] : memref<16xf32, 3>
// CHECK-NEXT: scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT: %10 = arith.muli %1, %c16 : index
// CHECK-NEXT: %11 = arith.addi %0, %10 : index
// CHECK-NEXT: %12 = memref.load %arg0[%11] : memref<4096xf32>
// CHECK-NEXT: %13 = arith.muli %1, %c16 : index
// CHECK-NEXT: %14 = arith.addi %0, %13 : index
// CHECK-NEXT: memref.store %12, %2[%14] : memref<16xf32, 3>
// CHECK-NEXT: scf.for %arg4 = %c0 to %c32 step %c1 {
// CHECK-NEXT: %29 = arith.muli %1, %c16 : index
// CHECK-NEXT: %30 = arith.addi %0, %29 : index
// CHECK-NEXT: %31 = memref.load %arg0[%30] : memref<4096xf32>
// CHECK-NEXT: %32 = arith.muli %1, %c16 : index
// CHECK-NEXT: %33 = arith.addi %0, %32 : index
// CHECK-NEXT: memref.store %31, %2[%33] : memref<16xf32, 3>
// CHECK-NEXT: %34 = arith.muli %1, %c16 : index
// CHECK-NEXT: %35 = arith.addi %0, %34 : index
// CHECK-NEXT: %36 = memref.load %arg1[%35] : memref<4096xf32>
// CHECK-NEXT: %37 = arith.muli %1, %c16 : index
// CHECK-NEXT: %38 = arith.addi %0, %37 : index
// CHECK-NEXT: memref.store %36, %2[%38] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %39 = arith.muli %1, %c16 : index
// CHECK-NEXT: %40 = arith.addi %0, %39 : index
// CHECK-NEXT: %41 = memref.load %2[%40] : memref<16xf32, 3>
// CHECK-NEXT: %42 = arith.muli %1, %c16 : index
// CHECK-NEXT: %43 = arith.addi %0, %42 : index
// CHECK-NEXT: %44 = memref.load %3[%43] : memref<16xf32, 3>
// CHECK-NEXT: %45 = arith.addf %41, %44 {RelaxedPrecision} : f32
// CHECK-NEXT: %46 = arith.muli %1, %c16 : index
// CHECK-NEXT: %47 = arith.addi %0, %46 : index
// CHECK-NEXT: memref.store %45, %arg2[%47] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: } {sym_name = "value_loop"}
// CHECK-NEXT: %15 = arith.muli %1, %c16 : index
// CHECK-NEXT: %16 = arith.addi %0, %15 : index
// CHECK-NEXT: %17 = memref.load %arg1[%16] : memref<4096xf32>
// CHECK-NEXT: %18 = arith.muli %1, %c16 : index
// CHECK-NEXT: %19 = arith.addi %0, %18 : index
// CHECK-NEXT: memref.store %17, %2[%19] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %20 = arith.muli %1, %c16 : index
// CHECK-NEXT: %21 = arith.addi %0, %20 : index
// CHECK-NEXT: %22 = memref.load %2[%21] : memref<16xf32, 3>
// CHECK-NEXT: %23 = arith.muli %1, %c16 : index
// CHECK-NEXT: %24 = arith.addi %0, %23 : index
// CHECK-NEXT: %25 = memref.load %3[%24] : memref<16xf32, 3>
// CHECK-NEXT: %26 = arith.addf %22, %25 {RelaxedPrecision} : f32
// CHECK-NEXT: %27 = arith.muli %1, %c16 : index
// CHECK-NEXT: %28 = arith.addi %0, %27 : index
// CHECK-NEXT: memref.store %26, %arg2[%28] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: } {sym_name = "value_loop"}
// CHECK-NEXT: %7 = arith.muli %1, %c16 : index
// CHECK-NEXT: %8 = arith.addi %0, %7 : index
// CHECK-NEXT: %9 = memref.load %arg1[%8] : memref<4096xf32>
// CHECK-NEXT: memref.store %9, %arg2[%c0] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_loop_test_4 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_loop_test_4_8a33c1a81937487e_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>, %arg2 : memref<4096xf32>)
    return
  }
  func @barrier_loop_test_4_8a33c1a81937487e(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_loop_test_4", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_loop_test_4_8a33c1a81937487e_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %0 = gpu.thread_id x
      %1 = gpu.block_dim x
      %2 = memref.alloc() : memref<16xf32, 3>
      %3 = memref.alloc() : memref<16xf32, 3>
      %4 = memref.load %arg0[%c0] : memref<4096xf32>
      %5 = arith.muli %1, %c16 : index
      %6 = arith.addi %0, %5 : index
      memref.store %4, %2[%6] : memref<16xf32, 3>
      scf.for %arg3 = %c0 to %c32 step %c1 {
        %10 = arith.muli %1, %c16 : index
        %11 = arith.addi %0, %10 : index
        %12 = memref.load %arg0[%11] : memref<4096xf32>
        %13 = arith.muli %1, %c16 : index
        %14 = arith.addi %0, %13 : index
        memref.store %12, %2[%14] : memref<16xf32, 3>
        scf.for %arg4 = %c0 to %c32 step %c1 {
          %29 = arith.muli %1, %c16 : index
          %30 = arith.addi %0, %29 : index
          %31 = memref.load %arg0[%30] : memref<4096xf32>
          %32 = arith.muli %1, %c16 : index
          %33 = arith.addi %0, %32 : index
          memref.store %31, %2[%33] : memref<16xf32, 3>
          %34 = arith.muli %1, %c16 : index
          %35 = arith.addi %0, %34 : index
          %36 = memref.load %arg1[%35] : memref<4096xf32>
          %37 = arith.muli %1, %c16 : index
          %38 = arith.addi %0, %37 : index
          memref.store %36, %2[%38] : memref<16xf32, 3>
          "accv.barrier"() {scope = "Block"} : () -> ()
          %39 = arith.muli %1, %c16 : index
          %40 = arith.addi %0, %39 : index
          %41 = memref.load %2[%40] : memref<16xf32, 3>
          %42 = arith.muli %1, %c16 : index
          %43 = arith.addi %0, %42 : index
          %44 = memref.load %3[%43] : memref<16xf32, 3>
          %45 = arith.addf %41, %44 {RelaxedPrecision} : f32
          %46 = arith.muli %1, %c16 : index
          %47 = arith.addi %0, %46 : index
          memref.store %45, %arg2[%47] : memref<4096xf32>
          "accv.barrier"() {scope = "Block"} : () -> ()
        } {sym_name = "value_loop"}
        %15 = arith.muli %1, %c16 : index
        %16 = arith.addi %0, %15 : index
        %17 = memref.load %arg1[%16] : memref<4096xf32>
        %18 = arith.muli %1, %c16 : index
        %19 = arith.addi %0, %18 : index
        memref.store %17, %2[%19] : memref<16xf32, 3>
        "accv.barrier"() {scope = "Block"} : () -> ()
        %20 = arith.muli %1, %c16 : index
        %21 = arith.addi %0, %20 : index
        %22 = memref.load %2[%21] : memref<16xf32, 3>
        %23 = arith.muli %1, %c16 : index
        %24 = arith.addi %0, %23 : index
        %25 = memref.load %3[%24] : memref<16xf32, 3>
        %26 = arith.addf %22, %25 {RelaxedPrecision} : f32
        %27 = arith.muli %1, %c16 : index
        %28 = arith.addi %0, %27 : index
        memref.store %26, %arg2[%28] : memref<4096xf32>
        "accv.barrier"() {scope = "Block"} : () -> ()
      } {sym_name = "value_loop"}
      %7 = arith.muli %1, %c16 : index
      %8 = arith.addi %0, %7 : index
      %9 = memref.load %arg1[%8] : memref<4096xf32>
      memref.store %9, %arg2[%c0] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_loop_test_5
// CHECK-NEXT: func nested @barrier_loop_test_5_1639044a1799739d_impl_8950868249543779986(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c256 = arith.constant 256 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_loop_test_5_1639044a1799739d(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {accv.base_name = "barrier_loop_test_5", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_loop_test_5_1639044a1799739d_impl_8950868249543779986(%arg0, %arg1) : (memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %c32 = arith.constant 32 : index
// CHECK-NEXT: %c0 = arith.constant 0 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = gpu.block_dim x
// CHECK-NEXT: %2 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %3 = arith.muli %1, %c16 : index
// CHECK-NEXT: %4 = arith.addi %0, %3 : index
// CHECK-NEXT: %5 = memref.load %2[%4] : memref<16xf32, 3>
// CHECK-NEXT: %6 = arith.muli %1, %c16 : index
// CHECK-NEXT: %7 = arith.addi %0, %6 : index
// CHECK-NEXT: memref.store %5, %arg0[%7] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT: %13 = arith.muli %1, %c16 : index
// CHECK-NEXT: %14 = arith.addi %0, %13 : index
// CHECK-NEXT: %15 = memref.load %arg0[%14] : memref<4096xf32>
// CHECK-NEXT: %16 = arith.muli %1, %c16 : index
// CHECK-NEXT: %17 = arith.addi %0, %16 : index
// CHECK-NEXT: memref.store %15, %2[%17] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %18 = arith.muli %1, %c16 : index
// CHECK-NEXT: %19 = arith.addi %0, %18 : index
// CHECK-NEXT: %20 = memref.load %2[%19] : memref<16xf32, 3>
// CHECK-NEXT: %21 = arith.muli %1, %c16 : index
// CHECK-NEXT: %22 = arith.addi %0, %21 : index
// CHECK-NEXT: memref.store %20, %arg1[%22] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: } {sym_name = "value_loop"}
// CHECK-NEXT: %8 = arith.muli %1, %c16 : index
// CHECK-NEXT: %9 = arith.addi %0, %8 : index
// CHECK-NEXT: %10 = memref.load %arg0[%9] : memref<4096xf32>
// CHECK-NEXT: %11 = arith.muli %1, %c16 : index
// CHECK-NEXT: %12 = arith.addi %0, %11 : index
// CHECK-NEXT: memref.store %10, %arg1[%12] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_loop_test_5 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_loop_test_5_1639044a1799739d_impl_8950868249543779986(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
    return
  }
  func @barrier_loop_test_5_1639044a1799739d(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {accv.base_name = "barrier_loop_test_5", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_loop_test_5_1639044a1799739d_impl_8950868249543779986(%arg0, %arg1) : (memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %0 = gpu.thread_id x
      %1 = gpu.block_dim x
      %2 = memref.alloc() : memref<16xf32, 3>
      %3 = arith.muli %1, %c16 : index
      %4 = arith.addi %0, %3 : index
      %5 = memref.load %2[%4] : memref<16xf32, 3>
      %6 = arith.muli %1, %c16 : index
      %7 = arith.addi %0, %6 : index
      memref.store %5, %arg0[%7] : memref<4096xf32>
      "accv.barrier"() {scope = "Block"} : () -> ()
      scf.for %arg2 = %c0 to %c32 step %c1 {
        %13 = arith.muli %1, %c16 : index
        %14 = arith.addi %0, %13 : index
        %15 = memref.load %arg0[%14] : memref<4096xf32>
        %16 = arith.muli %1, %c16 : index
        %17 = arith.addi %0, %16 : index
        memref.store %15, %2[%17] : memref<16xf32, 3>
        "accv.barrier"() {scope = "Block"} : () -> ()
        %18 = arith.muli %1, %c16 : index
        %19 = arith.addi %0, %18 : index
        %20 = memref.load %2[%19] : memref<16xf32, 3>
        %21 = arith.muli %1, %c16 : index
        %22 = arith.addi %0, %21 : index
        memref.store %20, %arg1[%22] : memref<4096xf32>
        "accv.barrier"() {scope = "Block"} : () -> ()
      } {sym_name = "value_loop"}
      %8 = arith.muli %1, %c16 : index
      %9 = arith.addi %0, %8 : index
      %10 = memref.load %arg0[%9] : memref<4096xf32>
      %11 = arith.muli %1, %c16 : index
      %12 = arith.addi %0, %11 : index
      memref.store %10, %arg1[%12] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_multi_warp_test_1
// CHECK-NEXT: func nested @barrier_multi_warp_test_1_5dfccc585c43dfb2_impl_3084064645727573566(%arg0: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c32 = arith.constant 32 : index
// CHECK-NEXT: %c8 = arith.constant 8 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c16, %c1, %c1) threads in (%c32, %c8, %c1) args(%arg0 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_multi_warp_test_1_5dfccc585c43dfb2(%arg0: memref<4096xf32>) attributes {accv.base_name = "barrier_multi_warp_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_multi_warp_test_1_5dfccc585c43dfb2_impl_3084064645727573566(%arg0) : (memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>) kernel attributes {blockSize = [32 : i32, 8 : i32, 1 : i32], gridSize = [16 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %cst = arith.constant 2.000000e+00 : f32
// CHECK-NEXT: %c256 = arith.constant 256 : index
// CHECK-NEXT: %c32 = arith.constant 32 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = gpu.thread_id y
// CHECK-NEXT: %2 = gpu.block_dim x
// CHECK-NEXT: %3 = memref.alloc() : memref<4096xf32, 3>
// CHECK-NEXT: %4 = arith.muli %2, %c256 : index
// CHECK-NEXT: %5 = arith.addi %0, %4 : index
// CHECK-NEXT: %6 = arith.muli %1, %c32 : index
// CHECK-NEXT: %7 = arith.addi %5, %6 : index
// CHECK-NEXT: %8 = memref.load %arg0[%7] : memref<4096xf32>
// CHECK-NEXT: %9 = arith.muli %2, %c256 : index
// CHECK-NEXT: %10 = arith.addi %0, %9 : index
// CHECK-NEXT: %11 = arith.muli %1, %c32 : index
// CHECK-NEXT: %12 = arith.addi %10, %11 : index
// CHECK-NEXT: memref.store %8, %3[%12] : memref<4096xf32, 3>
// CHECK-NEXT: %13 = arith.muli %2, %c256 : index
// CHECK-NEXT: %14 = arith.addi %0, %13 : index
// CHECK-NEXT: %15 = arith.muli %1, %c32 : index
// CHECK-NEXT: %16 = arith.addi %14, %15 : index
// CHECK-NEXT: %17 = memref.load %arg0[%16] : memref<4096xf32>
// CHECK-NEXT: %18 = arith.mulf %17, %cst {RelaxedPrecision} : f32
// CHECK-NEXT: %19 = arith.muli %2, %c256 : index
// CHECK-NEXT: %20 = arith.addi %0, %19 : index
// CHECK-NEXT: %21 = arith.muli %1, %c32 : index
// CHECK-NEXT: %22 = arith.addi %20, %21 : index
// CHECK-NEXT: memref.store %18, %arg0[%22] : memref<4096xf32>
// CHECK-NEXT: %23 = arith.muli %2, %c256 : index
// CHECK-NEXT: %24 = arith.addi %0, %23 : index
// CHECK-NEXT: %25 = arith.muli %1, %c32 : index
// CHECK-NEXT: %26 = arith.addi %24, %25 : index
// CHECK-NEXT: %27 = memref.load %arg0[%26] : memref<4096xf32>
// CHECK-NEXT: %28 = arith.muli %2, %c256 : index
// CHECK-NEXT: %29 = arith.addi %0, %28 : index
// CHECK-NEXT: %30 = arith.muli %1, %c32 : index
// CHECK-NEXT: %31 = arith.addi %29, %30 : index
// CHECK-NEXT: memref.store %27, %arg0[%31] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %32 = arith.muli %2, %c256 : index
// CHECK-NEXT: %33 = arith.addi %0, %32 : index
// CHECK-NEXT: %34 = arith.muli %1, %c32 : index
// CHECK-NEXT: %35 = arith.addi %33, %34 : index
// CHECK-NEXT: %36 = memref.load %3[%35] : memref<4096xf32, 3>
// CHECK-NEXT: %37 = arith.muli %2, %c256 : index
// CHECK-NEXT: %38 = arith.addi %0, %37 : index
// CHECK-NEXT: %39 = arith.muli %1, %c32 : index
// CHECK-NEXT: %40 = arith.addi %38, %39 : index
// CHECK-NEXT: memref.store %36, %arg0[%40] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_multi_warp_test_1 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_multi_warp_test_1_5dfccc585c43dfb2_impl_3084064645727573566(%arg0: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c16, %c1, %c1) threads in (%c32, %c8, %c1) args(%arg0 : memref<4096xf32>)
    return
  }
  func @barrier_multi_warp_test_1_5dfccc585c43dfb2(%arg0: memref<4096xf32>) attributes {accv.base_name = "barrier_multi_warp_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_multi_warp_test_1_5dfccc585c43dfb2_impl_3084064645727573566(%arg0) : (memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>) kernel attributes {blockSize = [32 : i32, 8 : i32, 1 : i32], gridSize = [16 : i32, 1 : i32, 1 : i32]} {
      %cst = arith.constant 2.000000e+00 : f32
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %0 = gpu.thread_id x
      %1 = gpu.thread_id y
      %2 = gpu.block_dim x
      %3 = memref.alloc() : memref<4096xf32, 3>
      %4 = arith.muli %2, %c256 : index
      %5 = arith.addi %0, %4 : index
      %6 = arith.muli %1, %c32 : index
      %7 = arith.addi %5, %6 : index
      %8 = memref.load %arg0[%7] : memref<4096xf32>
      %9 = arith.muli %2, %c256 : index
      %10 = arith.addi %0, %9 : index
      %11 = arith.muli %1, %c32 : index
      %12 = arith.addi %10, %11 : index
      memref.store %8, %3[%12] : memref<4096xf32, 3>
      %13 = arith.muli %2, %c256 : index
      %14 = arith.addi %0, %13 : index
      %15 = arith.muli %1, %c32 : index
      %16 = arith.addi %14, %15 : index
      %17 = memref.load %arg0[%16] : memref<4096xf32>
      %18 = arith.mulf %17, %cst {RelaxedPrecision} : f32
      %19 = arith.muli %2, %c256 : index
      %20 = arith.addi %0, %19 : index
      %21 = arith.muli %1, %c32 : index
      %22 = arith.addi %20, %21 : index
      memref.store %18, %arg0[%22] : memref<4096xf32>
      %23 = arith.muli %2, %c256 : index
      %24 = arith.addi %0, %23 : index
      %25 = arith.muli %1, %c32 : index
      %26 = arith.addi %24, %25 : index
      %27 = memref.load %arg0[%26] : memref<4096xf32>
      %28 = arith.muli %2, %c256 : index
      %29 = arith.addi %0, %28 : index
      %30 = arith.muli %1, %c32 : index
      %31 = arith.addi %29, %30 : index
      memref.store %27, %arg0[%31] : memref<4096xf32>
      "accv.barrier"() {scope = "Block"} : () -> ()
      %32 = arith.muli %2, %c256 : index
      %33 = arith.addi %0, %32 : index
      %34 = arith.muli %1, %c32 : index
      %35 = arith.addi %33, %34 : index
      %36 = memref.load %3[%35] : memref<4096xf32, 3>
      %37 = arith.muli %2, %c256 : index
      %38 = arith.addi %0, %37 : index
      %39 = arith.muli %1, %c32 : index
      %40 = arith.addi %38, %39 : index
      memref.store %36, %arg0[%40] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_seq_test_1
// CHECK-NEXT: func nested @barrier_seq_test_1_64e9e91045afcd3b_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_seq_test_1_64e9e91045afcd3b(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_seq_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_seq_test_1_64e9e91045afcd3b_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %cst = arith.constant 2.000000e+00 : f32
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %2 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %2, %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: %3 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: %4 = arith.mulf %3, %cst {RelaxedPrecision} : f32
// CHECK-NEXT: memref.store %4, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: %5 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %5, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %6 = memref.load %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: memref.store %6, %arg1[%0] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_seq_test_1 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_seq_test_1_64e9e91045afcd3b_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
    return
  }
  func @barrier_seq_test_1_64e9e91045afcd3b(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_seq_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_seq_test_1_64e9e91045afcd3b_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
      %cst = arith.constant 2.000000e+00 : f32
      %0 = gpu.thread_id x
      %1 = memref.alloc() : memref<16xf32, 3>
      %2 = memref.load %arg0[%0] : memref<4096xf32>
      memref.store %2, %1[%0] : memref<16xf32, 3>
      %3 = memref.load %arg0[%0] : memref<4096xf32>
      %4 = arith.mulf %3, %cst {RelaxedPrecision} : f32
      memref.store %4, %arg0[%0] : memref<4096xf32>
      %5 = memref.load %arg0[%0] : memref<4096xf32>
      memref.store %5, %arg0[%0] : memref<4096xf32>
      "accv.barrier"() {scope = "Block"} : () -> ()
      %6 = memref.load %1[%0] : memref<16xf32, 3>
      memref.store %6, %arg1[%0] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_seq_test_2
// CHECK-NEXT: func nested @barrier_seq_test_2_589be55934a4a0aa_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>, %arg2 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_seq_test_2_589be55934a4a0aa(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_seq_test_2", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_seq_test_2_589be55934a4a0aa_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %2 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %3 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %3, %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: %4 = memref.load %arg1[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %4, %2[%0] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %5 = memref.load %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: memref.store %5, %arg2[%0] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_seq_test_2 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_seq_test_2_589be55934a4a0aa_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>, %arg2 : memref<4096xf32>)
    return
  }
  func @barrier_seq_test_2_589be55934a4a0aa(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_seq_test_2", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_seq_test_2_589be55934a4a0aa_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
      %0 = gpu.thread_id x
      %1 = memref.alloc() : memref<16xf32, 3>
      %2 = memref.alloc() : memref<16xf32, 3>
      %3 = memref.load %arg0[%0] : memref<4096xf32>
      memref.store %3, %1[%0] : memref<16xf32, 3>
      %4 = memref.load %arg1[%0] : memref<4096xf32>
      memref.store %4, %2[%0] : memref<16xf32, 3>
      "accv.barrier"() {scope = "Block"} : () -> ()
      %5 = memref.load %1[%0] : memref<16xf32, 3>
      memref.store %5, %arg2[%0] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_seq_test_3
// CHECK-NEXT: func nested @barrier_seq_test_3_71a42daf88846ff1_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>, %arg2 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_seq_test_3_71a42daf88846ff1(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_seq_test_3", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_seq_test_3_71a42daf88846ff1_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %2 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %3 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %3, %2[%0] : memref<16xf32, 3>
// CHECK-NEXT: %4 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %4, %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: %5 = memref.load %arg1[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %5, %2[%0] : memref<16xf32, 3>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %6 = memref.load %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: memref.store %6, %arg2[%0] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_seq_test_3 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_seq_test_3_71a42daf88846ff1_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>, %arg2 : memref<4096xf32>)
    return
  }
  func @barrier_seq_test_3_71a42daf88846ff1(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_seq_test_3", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_seq_test_3_71a42daf88846ff1_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
      %0 = gpu.thread_id x
      %1 = memref.alloc() : memref<16xf32, 3>
      %2 = memref.alloc() : memref<16xf32, 3>
      %3 = memref.load %arg0[%0] : memref<4096xf32>
      memref.store %3, %2[%0] : memref<16xf32, 3>
      %4 = memref.load %arg0[%0] : memref<4096xf32>
      memref.store %4, %1[%0] : memref<16xf32, 3>
      %5 = memref.load %arg1[%0] : memref<4096xf32>
      memref.store %5, %2[%0] : memref<16xf32, 3>
      "accv.barrier"() {scope = "Block"} : () -> ()
      %6 = memref.load %1[%0] : memref<16xf32, 3>
      memref.store %6, %arg2[%0] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_single_warp_test_1
// CHECK-NEXT: func nested @barrier_single_warp_test_1_3af0b314f44fb508_impl_3084064576730956389(%arg0: memref<16xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<16xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_single_warp_test_1_3af0b314f44fb508(%arg0: memref<16xf32>) attributes {accv.base_name = "barrier_single_warp_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_single_warp_test_1_3af0b314f44fb508_impl_3084064576730956389(%arg0) : (memref<16xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<16xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %cst = arith.constant 2.000000e+00 : f32
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %2 = memref.load %arg0[%0] : memref<16xf32>
// CHECK-NEXT: memref.store %2, %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: %3 = memref.load %arg0[%0] : memref<16xf32>
// CHECK-NEXT: %4 = arith.mulf %3, %cst {RelaxedPrecision} : f32
// CHECK-NEXT: memref.store %4, %arg0[%0] : memref<16xf32>
// CHECK-NEXT: %5 = memref.load %arg0[%0] : memref<16xf32>
// CHECK-NEXT: memref.store %5, %arg0[%0] : memref<16xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %6 = memref.load %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: memref.store %6, %arg0[%0] : memref<16xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_single_warp_test_1 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_single_warp_test_1_3af0b314f44fb508_impl_3084064576730956389(%arg0: memref<16xf32>) attributes {exec_target = 0 : i64} {
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<16xf32>)
    return
  }
  func @barrier_single_warp_test_1_3af0b314f44fb508(%arg0: memref<16xf32>) attributes {accv.base_name = "barrier_single_warp_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_single_warp_test_1_3af0b314f44fb508_impl_3084064576730956389(%arg0) : (memref<16xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<16xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
      %cst = arith.constant 2.000000e+00 : f32
      %0 = gpu.thread_id x
      %1 = memref.alloc() : memref<16xf32, 3>
      %2 = memref.load %arg0[%0] : memref<16xf32>
      memref.store %2, %1[%0] : memref<16xf32, 3>
      %3 = memref.load %arg0[%0] : memref<16xf32>
      %4 = arith.mulf %3, %cst {RelaxedPrecision} : f32
      memref.store %4, %arg0[%0] : memref<16xf32>
      %5 = memref.load %arg0[%0] : memref<16xf32>
      memref.store %5, %arg0[%0] : memref<16xf32>
      "accv.barrier"() {scope = "Block"} : () -> ()
      %6 = memref.load %1[%0] : memref<16xf32, 3>
      memref.store %6, %arg0[%0] : memref<16xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_single_warp_test_2
// CHECK-NEXT: func nested @barrier_single_warp_test_2_b285692b8cd17bf7_impl_3084064645727573566(%arg0: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c256 = arith.constant 256 : index
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_single_warp_test_2_b285692b8cd17bf7(%arg0: memref<4096xf32>) attributes {accv.base_name = "barrier_single_warp_test_2", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_single_warp_test_2_b285692b8cd17bf7_impl_3084064645727573566(%arg0) : (memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %cst = arith.constant 2.000000e+00 : f32
// CHECK-NEXT: %c16 = arith.constant 16 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = gpu.block_dim x
// CHECK-NEXT: %2 = memref.alloc() : memref<4096xf32, 3>
// CHECK-NEXT: %3 = arith.muli %1, %c16 : index
// CHECK-NEXT: %4 = arith.addi %0, %3 : index
// CHECK-NEXT: %5 = memref.load %arg0[%4] : memref<4096xf32>
// CHECK-NEXT: %6 = arith.muli %1, %c16 : index
// CHECK-NEXT: %7 = arith.addi %0, %6 : index
// CHECK-NEXT: memref.store %5, %2[%7] : memref<4096xf32, 3>
// CHECK-NEXT: %8 = arith.muli %1, %c16 : index
// CHECK-NEXT: %9 = arith.addi %0, %8 : index
// CHECK-NEXT: %10 = memref.load %arg0[%9] : memref<4096xf32>
// CHECK-NEXT: %11 = arith.mulf %10, %cst {RelaxedPrecision} : f32
// CHECK-NEXT: %12 = arith.muli %1, %c16 : index
// CHECK-NEXT: %13 = arith.addi %0, %12 : index
// CHECK-NEXT: memref.store %11, %arg0[%13] : memref<4096xf32>
// CHECK-NEXT: %14 = arith.muli %1, %c16 : index
// CHECK-NEXT: %15 = arith.addi %0, %14 : index
// CHECK-NEXT: %16 = memref.load %arg0[%15] : memref<4096xf32>
// CHECK-NEXT: %17 = arith.muli %1, %c16 : index
// CHECK-NEXT: %18 = arith.addi %0, %17 : index
// CHECK-NEXT: memref.store %16, %arg0[%18] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %19 = arith.muli %1, %c16 : index
// CHECK-NEXT: %20 = arith.addi %0, %19 : index
// CHECK-NEXT: %21 = memref.load %2[%20] : memref<4096xf32, 3>
// CHECK-NEXT: %22 = arith.muli %1, %c16 : index
// CHECK-NEXT: %23 = arith.addi %0, %22 : index
// CHECK-NEXT: memref.store %21, %arg0[%23] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_single_warp_test_2 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_single_warp_test_2_b285692b8cd17bf7_impl_3084064645727573566(%arg0: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c256, %c1, %c1) threads in (%c16, %c1, %c1) args(%arg0 : memref<4096xf32>)
    return
  }
  func @barrier_single_warp_test_2_b285692b8cd17bf7(%arg0: memref<4096xf32>) attributes {accv.base_name = "barrier_single_warp_test_2", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_single_warp_test_2_b285692b8cd17bf7_impl_3084064645727573566(%arg0) : (memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>) kernel attributes {blockSize = [16 : i32, 1 : i32, 1 : i32], gridSize = [256 : i32, 1 : i32, 1 : i32]} {
      %cst = arith.constant 2.000000e+00 : f32
      %c16 = arith.constant 16 : index
      %0 = gpu.thread_id x
      %1 = gpu.block_dim x
      %2 = memref.alloc() : memref<4096xf32, 3>
      %3 = arith.muli %1, %c16 : index
      %4 = arith.addi %0, %3 : index
      %5 = memref.load %arg0[%4] : memref<4096xf32>
      %6 = arith.muli %1, %c16 : index
      %7 = arith.addi %0, %6 : index
      memref.store %5, %2[%7] : memref<4096xf32, 3>
      %8 = arith.muli %1, %c16 : index
      %9 = arith.addi %0, %8 : index
      %10 = memref.load %arg0[%9] : memref<4096xf32>
      %11 = arith.mulf %10, %cst {RelaxedPrecision} : f32
      %12 = arith.muli %1, %c16 : index
      %13 = arith.addi %0, %12 : index
      memref.store %11, %arg0[%13] : memref<4096xf32>
      %14 = arith.muli %1, %c16 : index
      %15 = arith.addi %0, %14 : index
      %16 = memref.load %arg0[%15] : memref<4096xf32>
      %17 = arith.muli %1, %c16 : index
      %18 = arith.addi %0, %17 : index
      memref.store %16, %arg0[%18] : memref<4096xf32>
      "accv.barrier"() {scope = "Block"} : () -> ()
      %19 = arith.muli %1, %c16 : index
      %20 = arith.addi %0, %19 : index
      %21 = memref.load %2[%20] : memref<4096xf32, 3>
      %22 = arith.muli %1, %c16 : index
      %23 = arith.addi %0, %22 : index
      memref.store %21, %arg0[%23] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_single_warp_test_3
// CHECK-NEXT: func nested @barrier_single_warp_test_3_20c93e69d024ea41_impl_3084064645727573566(%arg0: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c128 = arith.constant 128 : index
// CHECK-NEXT: %c32 = arith.constant 32 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c128, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_single_warp_test_3_20c93e69d024ea41(%arg0: memref<4096xf32>) attributes {accv.base_name = "barrier_single_warp_test_3", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_single_warp_test_3_20c93e69d024ea41_impl_3084064645727573566(%arg0) : (memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>) kernel attributes {blockSize = [32 : i32, 1 : i32, 1 : i32], gridSize = [128 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %cst = arith.constant 2.000000e+00 : f32
// CHECK-NEXT: %c32 = arith.constant 32 : index
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = gpu.block_dim x
// CHECK-NEXT: %2 = memref.alloc() : memref<4096xf32, 3>
// CHECK-NEXT: %3 = arith.muli %1, %c32 : index
// CHECK-NEXT: %4 = arith.addi %0, %3 : index
// CHECK-NEXT: %5 = memref.load %arg0[%4] : memref<4096xf32>
// CHECK-NEXT: %6 = arith.muli %1, %c32 : index
// CHECK-NEXT: %7 = arith.addi %0, %6 : index
// CHECK-NEXT: memref.store %5, %2[%7] : memref<4096xf32, 3>
// CHECK-NEXT: %8 = arith.muli %1, %c32 : index
// CHECK-NEXT: %9 = arith.addi %0, %8 : index
// CHECK-NEXT: %10 = memref.load %arg0[%9] : memref<4096xf32>
// CHECK-NEXT: %11 = arith.mulf %10, %cst {RelaxedPrecision} : f32
// CHECK-NEXT: %12 = arith.muli %1, %c32 : index
// CHECK-NEXT: %13 = arith.addi %0, %12 : index
// CHECK-NEXT: memref.store %11, %arg0[%13] : memref<4096xf32>
// CHECK-NEXT: %14 = arith.muli %1, %c32 : index
// CHECK-NEXT: %15 = arith.addi %0, %14 : index
// CHECK-NEXT: %16 = memref.load %arg0[%15] : memref<4096xf32>
// CHECK-NEXT: %17 = arith.muli %1, %c32 : index
// CHECK-NEXT: %18 = arith.addi %0, %17 : index
// CHECK-NEXT: memref.store %16, %arg0[%18] : memref<4096xf32>
// CHECK-NEXT: "accv.barrier"() {scope = "Block"} : () -> ()
// CHECK-NEXT: %19 = arith.muli %1, %c32 : index
// CHECK-NEXT: %20 = arith.addi %0, %19 : index
// CHECK-NEXT: %21 = memref.load %2[%20] : memref<4096xf32, 3>
// CHECK-NEXT: %22 = arith.muli %1, %c32 : index
// CHECK-NEXT: %23 = arith.addi %0, %22 : index
// CHECK-NEXT: memref.store %21, %arg0[%23] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_single_warp_test_3 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_single_warp_test_3_20c93e69d024ea41_impl_3084064645727573566(%arg0: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c128, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<4096xf32>)
    return
  }
  func @barrier_single_warp_test_3_20c93e69d024ea41(%arg0: memref<4096xf32>) attributes {accv.base_name = "barrier_single_warp_test_3", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_single_warp_test_3_20c93e69d024ea41_impl_3084064645727573566(%arg0) : (memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>) kernel attributes {blockSize = [32 : i32, 1 : i32, 1 : i32], gridSize = [128 : i32, 1 : i32, 1 : i32]} {
      %cst = arith.constant 2.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %0 = gpu.thread_id x
      %1 = gpu.block_dim x
      %2 = memref.alloc() : memref<4096xf32, 3>
      %3 = arith.muli %1, %c32 : index
      %4 = arith.addi %0, %3 : index
      %5 = memref.load %arg0[%4] : memref<4096xf32>
      %6 = arith.muli %1, %c32 : index
      %7 = arith.addi %0, %6 : index
      memref.store %5, %2[%7] : memref<4096xf32, 3>
      %8 = arith.muli %1, %c32 : index
      %9 = arith.addi %0, %8 : index
      %10 = memref.load %arg0[%9] : memref<4096xf32>
      %11 = arith.mulf %10, %cst {RelaxedPrecision} : f32
      %12 = arith.muli %1, %c32 : index
      %13 = arith.addi %0, %12 : index
      memref.store %11, %arg0[%13] : memref<4096xf32>
      %14 = arith.muli %1, %c32 : index
      %15 = arith.addi %0, %14 : index
      %16 = memref.load %arg0[%15] : memref<4096xf32>
      %17 = arith.muli %1, %c32 : index
      %18 = arith.addi %0, %17 : index
      memref.store %16, %arg0[%18] : memref<4096xf32>
      "accv.barrier"() {scope = "Block"} : () -> ()
      %19 = arith.muli %1, %c32 : index
      %20 = arith.addi %0, %19 : index
      %21 = memref.load %2[%20] : memref<4096xf32, 3>
      %22 = arith.muli %1, %c32 : index
      %23 = arith.addi %0, %22 : index
      memref.store %21, %arg0[%23] : memref<4096xf32>
      gpu.return
    }
  }
}



// CHECK-LABEL: module @barrier_trivial_test_1
// CHECK-NEXT: func nested @barrier_trivial_test_1_47fb9e7c730299e0_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
// CHECK-NEXT: %c4096 = arith.constant 4096 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func @barrier_trivial_test_1_47fb9e7c730299e0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_trivial_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
// CHECK-NEXT: call @barrier_trivial_test_1_47fb9e7c730299e0_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
// CHECK-NEXT: gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
// CHECK-NEXT: %cst = arith.constant 2.000000e+00 : f32
// CHECK-NEXT: %0 = gpu.thread_id x
// CHECK-NEXT: %1 = memref.alloc() : memref<16xf32, 3>
// CHECK-NEXT: %2 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %2, %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: %3 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: %4 = arith.mulf %3, %cst {RelaxedPrecision} : f32
// CHECK-NEXT: memref.store %4, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: %5 = memref.load %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: memref.store %5, %arg0[%0] : memref<4096xf32>
// CHECK-NEXT: %6 = memref.load %1[%0] : memref<16xf32, 3>
// CHECK-NEXT: memref.store %6, %arg1[%0] : memref<4096xf32>
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
module @barrier_trivial_test_1 attributes {accv.exec_runtime = "ROCM", gpu.container_module, llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  func nested @barrier_trivial_test_1_47fb9e7c730299e0_impl_11429216098513013678(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {exec_target = 0 : i64} {
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @NestFunction_0_module::@NestFunction_0 blocks in (%c1, %c1, %c1) threads in (%c4096, %c1, %c1) args(%arg0 : memref<4096xf32>, %arg1 : memref<4096xf32>)
    return
  }
  func @barrier_trivial_test_1_47fb9e7c730299e0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) attributes {accv.base_name = "barrier_trivial_test_1", accv.emit_header_decl, accv.emit_raw_pointer_api, exec_target = 0 : i64} {
    call @barrier_trivial_test_1_47fb9e7c730299e0_impl_11429216098513013678(%arg0, %arg1, %arg2) : (memref<4096xf32>, memref<4096xf32>, memref<4096xf32>) -> ()
    return
  }
  gpu.module @NestFunction_0_module attributes {accv.exec_runtime = "ROCM", gpu.binary = "HSACO"} {
    gpu.func @NestFunction_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) kernel attributes {blockSize = [4096 : i32, 1 : i32, 1 : i32], gridSize = [1 : i32, 1 : i32, 1 : i32]} {
      %cst = arith.constant 2.000000e+00 : f32
      %0 = gpu.thread_id x
      %1 = memref.alloc() : memref<16xf32, 3>
      %2 = memref.load %arg0[%0] : memref<4096xf32>
      memref.store %2, %1[%0] : memref<16xf32, 3>
      %3 = memref.load %arg0[%0] : memref<4096xf32>
      %4 = arith.mulf %3, %cst {RelaxedPrecision} : f32
      memref.store %4, %arg0[%0] : memref<4096xf32>
      %5 = memref.load %arg0[%0] : memref<4096xf32>
      memref.store %5, %arg0[%0] : memref<4096xf32>
      %6 = memref.load %1[%0] : memref<16xf32, 3>
      memref.store %6, %arg1[%0] : memref<4096xf32>
      gpu.return
    }
  }
}



