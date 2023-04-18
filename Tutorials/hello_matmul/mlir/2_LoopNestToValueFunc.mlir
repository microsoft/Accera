module @hello_matmul attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "hello_matmul"  {
    accv.func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
      "accv.lambda"() ( {
        affine.for %arg3 = 0 to 128 {
          affine.for %arg4 = 0 to 256 {
            affine.for %arg5 = 0 to 256 step 4 {
              affine.for %arg6 = 0 to 4 {
                %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %arg6)
                %1 = load %arg0[%arg3, %0] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
                %2 = load %arg1[%0, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
                %3 = "accv.bin_op"(%1, %2) {predicate = 2 : i64} : (f32, f32) -> f32
                %4 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
                %5 = "accv.bin_op"(%4, %3) {predicate = 0 : i64} : (f32, f32) -> f32
                store %5, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
                %6 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
                store %6, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
              } {begin = 0 : i64, end = 4 : i64, index = #accln<"index{k_i,4}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 1]}
            } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{k_o,3}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 4]}
          } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{j,1}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 256]}
        } {begin = 0 : i64, end = 128 : i64, index = #accln<"index{i,0}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 256, 256]}
        accv.return
      }) {exec_target = 0 : i64, sym_name = "NestFunction_0", type = () -> ()} : () -> ()
      accv.return
    }
    accv.func @hello_matmul_py_0f07b3ac(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, accv.base_name = "hello_matmul_py", accv.emit_header_decl, accv.emit_raw_pointer_api} {
      accv.launch_func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0, %arg1, %arg2) {exec_target = 0 : i64} : (memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) -> ()
      accv.return
    }
  }
}
