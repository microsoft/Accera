module @hello_matmul attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "hello_matmul"  {
    func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
      %c0 = constant 0 : index
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 256 {
          affine.for %arg5 = 0 to 256 step 4 {
            %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %c0)
            %1 = load %arg0[%arg3, %0] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %2 = load %arg1[%0, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %3 = "accv.bin_op"(%1, %2) {predicate = 2 : i64} : (f32, f32) -> f32
            %4 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %5 = "accv.bin_op"(%4, %3) {predicate = 0 : i64} : (f32, f32) -> f32
            store %5, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %6 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %6, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %7 = affine.apply affine_map<(d0) -> (d0 + 1)>(%c0)
            %8 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %7)
            %9 = load %arg0[%arg3, %8] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %10 = load %arg1[%8, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %11 = "accv.bin_op"(%9, %10) {predicate = 2 : i64} : (f32, f32) -> f32
            %12 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %13 = "accv.bin_op"(%12, %11) {predicate = 0 : i64} : (f32, f32) -> f32
            store %13, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %14 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %14, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %15 = affine.apply affine_map<(d0) -> (d0 + 2)>(%c0)
            %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %15)
            %17 = load %arg0[%arg3, %16] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %18 = load %arg1[%16, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %19 = "accv.bin_op"(%17, %18) {predicate = 2 : i64} : (f32, f32) -> f32
            %20 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %21 = "accv.bin_op"(%20, %19) {predicate = 0 : i64} : (f32, f32) -> f32
            store %21, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %22 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %22, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %23 = affine.apply affine_map<(d0) -> (d0 + 3)>(%c0)
            %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %23)
            %25 = load %arg0[%arg3, %24] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %26 = load %arg1[%24, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %27 = "accv.bin_op"(%25, %26) {predicate = 2 : i64} : (f32, f32) -> f32
            %28 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %29 = "accv.bin_op"(%28, %27) {predicate = 0 : i64} : (f32, f32) -> f32
            store %29, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %30 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %30, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{k_o,3}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 4]}
        } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{j,1}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 256]}
      } {begin = 0 : i64, end = 128 : i64, index = #accln<"index{i,0}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 256, 256]}
      return
    }
    func @hello_matmul_py_0f07b3ac(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, accv.base_name = "hello_matmul_py", accv.emit_header_decl, accv.emit_raw_pointer_api} {
      accv.launch_func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0, %arg1, %arg2) {exec_target = 0 : i64} : (memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) -> ()
      return
    }
    func @NestFunction_0(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "private"} {
      %c0 = constant 0 : index
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 256 {
          affine.for %arg5 = 0 to 256 step 4 {
            %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %c0)
            %1 = load %arg0[%arg3, %0] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %2 = load %arg1[%0, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %3 = "accv.bin_op"(%1, %2) {predicate = 2 : i64} : (f32, f32) -> f32
            %4 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %5 = "accv.bin_op"(%4, %3) {predicate = 0 : i64} : (f32, f32) -> f32
            store %5, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %6 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %6, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %7 = affine.apply affine_map<(d0) -> (d0 + 1)>(%c0)
            %8 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %7)
            %9 = load %arg0[%arg3, %8] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %10 = load %arg1[%8, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %11 = "accv.bin_op"(%9, %10) {predicate = 2 : i64} : (f32, f32) -> f32
            %12 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %13 = "accv.bin_op"(%12, %11) {predicate = 0 : i64} : (f32, f32) -> f32
            store %13, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %14 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %14, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %15 = affine.apply affine_map<(d0) -> (d0 + 2)>(%c0)
            %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %15)
            %17 = load %arg0[%arg3, %16] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %18 = load %arg1[%16, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %19 = "accv.bin_op"(%17, %18) {predicate = 2 : i64} : (f32, f32) -> f32
            %20 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %21 = "accv.bin_op"(%20, %19) {predicate = 0 : i64} : (f32, f32) -> f32
            store %21, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %22 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %22, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %23 = affine.apply affine_map<(d0) -> (d0 + 3)>(%c0)
            %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %23)
            %25 = load %arg0[%arg3, %24] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %26 = load %arg1[%24, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %27 = "accv.bin_op"(%25, %26) {predicate = 2 : i64} : (f32, f32) -> f32
            %28 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %29 = "accv.bin_op"(%28, %27) {predicate = 0 : i64} : (f32, f32) -> f32
            store %29, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            %30 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
            store %30, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{k_o,3}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 4]}
        } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{j,1}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 256]}
      } {begin = 0 : i64, end = 128 : i64, index = #accln<"index{i,0}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 256, 256]}
      return
    }
  }
}
