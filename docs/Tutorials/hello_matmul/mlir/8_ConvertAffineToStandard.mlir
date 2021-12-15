module @hello_matmul attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "hello_matmul"  {
    func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
      %c0 = constant 0 : index
      %c128 = constant 128 : index
      %c1 = constant 1 : index
      scf.for %arg3 = %c0 to %c128 step %c1 {
        %c0_0 = constant 0 : index
        %c256 = constant 256 : index
        %c1_1 = constant 1 : index
        scf.for %arg4 = %c0_0 to %c256 step %c1_1 {
          %c0_2 = constant 0 : index
          %c256_3 = constant 256 : index
          %c4 = constant 4 : index
          scf.for %arg5 = %c0_2 to %c256_3 step %c4 {
            %0 = load %arg0[%arg3, %arg5] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %1 = load %arg1[%arg5, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %2 = "accv.bin_op"(%0, %1) {predicate = 2 : i64} : (f32, f32) -> f32
            %3 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %4 = "accv.bin_op"(%3, %2) {predicate = 0 : i64} : (f32, f32) -> f32
            store %4, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %5 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %5, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %c1_4 = constant 1 : index
            %6 = addi %arg5, %c1_4 : index
            %7 = load %arg0[%arg3, %6] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %8 = load %arg1[%6, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %9 = "accv.bin_op"(%7, %8) {predicate = 2 : i64} : (f32, f32) -> f32
            %10 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %11 = "accv.bin_op"(%10, %9) {predicate = 0 : i64} : (f32, f32) -> f32
            store %11, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %12 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %12, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %c2 = constant 2 : index
            %13 = addi %arg5, %c2 : index
            %14 = load %arg0[%arg3, %13] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %15 = load %arg1[%13, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %16 = "accv.bin_op"(%14, %15) {predicate = 2 : i64} : (f32, f32) -> f32
            %17 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %18 = "accv.bin_op"(%17, %16) {predicate = 0 : i64} : (f32, f32) -> f32
            store %18, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %19 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %19, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %c3 = constant 3 : index
            %20 = addi %arg5, %c3 : index
            %21 = load %arg0[%arg3, %20] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %22 = load %arg1[%20, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %23 = "accv.bin_op"(%21, %22) {predicate = 2 : i64} : (f32, f32) -> f32
            %24 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %25 = "accv.bin_op"(%24, %23) {predicate = 0 : i64} : (f32, f32) -> f32
            store %25, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %26 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %26, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          }
        }
      }
      return
    }
    func @hello_matmul_py_0f07b3ac(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, accv.base_name = "hello_matmul_py", accv.emit_header_decl, accv.emit_raw_pointer_api} {
      accv.launch_func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0, %arg1, %arg2) {exec_target = 0 : i64} : (memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) -> ()
      return
    }
  }
}
