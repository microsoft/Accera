// RUN: acc-opt --convert-value-to-std %s | FileCheck %s

// CHECK-LABEL: module @test_bin_op_cast_op_folding_module
module @test_bin_op_cast_op_folding_module {
  accv.module "test_bin_op_cast_op_folding_module" {

    // CHECK-LABEL: func @bin_op_cast_output(%arg0: memref<1xi64>, %arg1: memref<1xi64>, %arg2: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = affine.load %arg1[0] : memref<1xi64>
    // CHECK-NEXT:    %2 = arith.addi %0, %1 : i64
    // CHECK-NEXT:    %3 = arith.trunci %2 : i64 to i32
    // CHECK-NEXT:    affine.store %3, %arg2[0] : memref<1xi32>
    builtin.func @bin_op_cast_output(%arg0: memref<1xi64>, %arg1: memref<1xi64>, %arg2: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = affine.load %arg1[0] : memref<1xi64>
      %2 = "accv.bin_op"(%0, %1) {predicate = 0 : i64} : (i64, i64) -> i32
      affine.store %2, %arg2[0] : memref<1xi32>
      return
    }

    // CHECK-LABEL: func @bin_op_cast_input(%arg0: memref<1xi64>, %arg1: memref<1xi32>, %arg2: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = affine.load %arg1[0] : memref<1xi32>
    // CHECK-NEXT:    %2 = arith.extsi %1 : i32 to i64
    // CHECK-NEXT:    %3 = arith.addi %0, %2 : i64
    // CHECK-NEXT:    %4 = arith.trunci %3 : i64 to i32
    // CHECK-NEXT:    affine.store %4, %arg2[0] : memref<1xi32>
    builtin.func @bin_op_cast_input(%arg0: memref<1xi64>, %arg1: memref<1xi32>, %arg2: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = affine.load %arg1[0] : memref<1xi32>
      %2 = "accv.bin_op"(%0, %1) {predicate = 0 : i64} : (i64, i32) -> i32
      affine.store %2, %arg2[0] : memref<1xi32>
      return
    }

    // CHECK-LABEL: func @bin_op_cast_input_to_f32(%arg0: memref<1xf32>, %arg1: memref<1xi32>, %arg2: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf32>
    // CHECK-NEXT:    %1 = affine.load %arg1[0] : memref<1xi32>
    // CHECK-NEXT:    %2 = arith.sitofp %1 : i32 to f32
    // CHECK-NEXT:    %3 = arith.mulf %0, %2 : f32
    // CHECK-NEXT:    %4 = arith.fptosi %3 : f32 to i32
    // CHECK-NEXT:    affine.store %4, %arg2[0] : memref<1xi32>
    builtin.func @bin_op_cast_input_to_f32(%arg0: memref<1xf32>, %arg1: memref<1xi32>, %arg2: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xf32>
      %1 = affine.load %arg1[0] : memref<1xi32>
      %2 = "accv.bin_op"(%0, %1) {predicate = 2 : i64} : (f32, i32) -> i32
      affine.store %2, %arg2[0] : memref<1xi32>
      return
    }

    // CHECK-LABEL: func @fold_bin_op_internal_cast_op(%arg0: memref<1xi64>, %arg1: memref<1xi32>, %arg2: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = affine.load %arg1[0] : memref<1xi32>
    // CHECK-NEXT:    %2 = arith.extsi %1 : i32 to i64
    // CHECK-NEXT:    %3 = arith.addi %0, %2 : i64
    // CHECK-NEXT:    affine.store %3, %arg2[0] : memref<1xi64>
    builtin.func @fold_bin_op_internal_cast_op(%arg0: memref<1xi64>, %arg1: memref<1xi32>, %arg2: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = affine.load %arg1[0] : memref<1xi32>
      %2 = "accv.cast"(%1) : (i32) -> i64
      %3 = "accv.bin_op"(%0, %2) {predicate = 0 : i64} : (i64, i64) -> i32
      %4 = "accv.cast"(%3) {internal} : (i32) -> i64
      affine.store %4, %arg2[0] : memref<1xi64>
      return
    }

    // CHECK-LABEL: func @fold_bin_op_internal_cast_explicit_cast_op(%arg0: memref<1xi64>, %arg1: memref<1xi32>, %arg2: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = affine.load %arg1[0] : memref<1xi32>
    // CHECK-NEXT:    %2 = arith.extsi %1 : i32 to i64
    // CHECK-NEXT:    %3 = arith.addi %0, %2 : i64
    // CHECK-NEXT:    affine.store %3, %arg2[0] : memref<1xi64>
    builtin.func @fold_bin_op_internal_cast_explicit_cast_op(%arg0: memref<1xi64>, %arg1: memref<1xi32>, %arg2: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = affine.load %arg1[0] : memref<1xi32>
      %2 = "accv.cast"(%1) : (i32) -> i64
      %3 = "accv.bin_op"(%0, %2) {predicate = 0 : i64} : (i64, i64) -> i32
      %4 = "accv.cast"(%3) : (i32) -> i64
      affine.store %4, %arg2[0] : memref<1xi64>
      return
    }

    // CHECK-LABEL: func @widen_bin_op_inputs_to_output(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi32>
    // CHECK-NEXT:    %1 = affine.load %arg1[0] : memref<1xi32>
    // CHECK-NEXT:    %2 = arith.extsi %0 : i32 to i64
    // CHECK-NEXT:    %3 = arith.extsi %1 : i32 to i64
    // CHECK-NEXT:    %4 = arith.addi %2, %3 : i64
    // CHECK-NEXT:    affine.store %4, %arg2[0] : memref<1xi64>
    builtin.func @widen_bin_op_inputs_to_output(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi32>
      %1 = affine.load %arg1[0] : memref<1xi32>
      %2 = "accv.bin_op"(%0, %1) {predicate = 0 : i64} : (i32, i32) -> i64
      affine.store %2, %arg2[0] : memref<1xi64>
      return
    }

    // CHECK-LABEL: func @widen_and_cast_bin_op_output(%arg0: memref<1xi64>, %arg1: memref<1xi64>, %arg2: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = affine.load %arg1[0] : memref<1xi64>
    // CHECK-NEXT:    %2 = arith.addi %0, %1 : i64
    // CHECK-NEXT:    %3 = arith.trunci %2 : i64 to i32
    // CHECK-NEXT:    affine.store %3, %arg2[0] : memref<1xi32>
    builtin.func @widen_and_cast_bin_op_output(%arg0: memref<1xi64>, %arg1: memref<1xi64>, %arg2: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = affine.load %arg1[0] : memref<1xi64>
      %2 = "accv.bin_op"(%0, %1) {predicate = 0 : i64} : (i64, i64) -> i32
      affine.store %2, %arg2[0] : memref<1xi32>
      return
    }

    // CHECK-LABEL: func @fold_intermediate_bin_op_types(%arg0: memref<1xi64>, %arg1: memref<1xi32>, %arg2: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = affine.load %arg1[0] : memref<1xi32>
    // CHECK-NEXT:    %2 = arith.extsi %1 : i32 to i64
    // CHECK-NEXT:    %3 = arith.addi %0, %2 : i64
    // CHECK-NEXT:    %4 = arith.muli %0, %3 : i64
    // CHECK-NEXT:    affine.store %4, %arg2[0] : memref<1xi64>
    builtin.func @fold_intermediate_bin_op_types(%arg0: memref<1xi64>, %arg1: memref<1xi32>, %arg2: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = affine.load %arg1[0] : memref<1xi32>
      %2 = "accv.bin_op"(%0, %1) {predicate = 0 : i64} : (i64, i32) -> i32
      %3 = "accv.bin_op"(%0, %2) {predicate = 2 : i64} : (i64, i32) -> i64
      affine.store %3, %arg2[0] : memref<1xi64>
      return
    }
  }
}
