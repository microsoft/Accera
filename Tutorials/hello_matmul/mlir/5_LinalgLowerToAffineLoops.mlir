module @hello_matmul attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "hello_matmul"  {
    func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 256 {
          affine.for %arg5 = 0 to 256 step 4 {
            %0 = load %arg0[%arg3, %arg5] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %1 = load %arg1[%arg5, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %2 = "accv.bin_op"(%0, %1) {predicate = 2 : i64} : (f32, f32) -> f32
            %3 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %4 = "accv.bin_op"(%3, %2) {predicate = 0 : i64} : (f32, f32) -> f32
            store %4, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %5 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %5, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %6 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg5)
            %7 = load %arg0[%arg3, %6] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %8 = load %arg1[%6, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %9 = "accv.bin_op"(%7, %8) {predicate = 2 : i64} : (f32, f32) -> f32
            %10 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %11 = "accv.bin_op"(%10, %9) {predicate = 0 : i64} : (f32, f32) -> f32
            store %11, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %12 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %12, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %13 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg5)
            %14 = load %arg0[%arg3, %13] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %15 = load %arg1[%13, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %16 = "accv.bin_op"(%14, %15) {predicate = 2 : i64} : (f32, f32) -> f32
            %17 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %18 = "accv.bin_op"(%17, %16) {predicate = 0 : i64} : (f32, f32) -> f32
            store %18, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %19 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %19, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %20 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg5)
            %21 = load %arg0[%arg3, %20] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %22 = load %arg1[%20, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %23 = "accv.bin_op"(%21, %22) {predicate = 2 : i64} : (f32, f32) -> f32
            %24 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %25 = "accv.bin_op"(%24, %23) {predicate = 0 : i64} : (f32, f32) -> f32
            store %25, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %26 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %26, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{k_o,3}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 4]}
        } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{j,1}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 256]}
      } {begin = 0 : i64, end = 128 : i64, index = #accln<"index{i,0}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 256, 256]}
      return
    }
    func @hello_matmul_py_0f07b3ac(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, accv.base_name = "hello_matmul_py", accv.emit_header_decl, accv.emit_raw_pointer_api} {
      accv.launch_func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0, %arg1, %arg2) {exec_target = 0 : i64} : (memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) -> ()
      return
    }
  }
}
