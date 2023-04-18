module @hello_matmul attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "hello_matmul"  {
    accv.func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
      %0 = accln.sym_index {name = "i"} #accln<"index{i,0}">
      %1 = accln.sym_index {name = "j"} #accln<"index{j,1}">
      %2 = accln.sym_index {name = "k"} #accln<"index{k,2}">
      "accln.nest"() ( {
        "accln.kernel"() ( {
          %4 = load %arg0[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %5 = load %arg1[%2, %1] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %6 = "accv.bin_op"(%4, %5) {predicate = 2 : i64} : (f32, f32) -> f32
          %7 = load %arg2[%0, %1] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %8 = "accv.bin_op"(%7, %6) {predicate = 0 : i64} : (f32, f32) -> f32
          store %8, %arg2[%0, %1] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %9 = load %arg2[%0, %1] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          store %9, %arg2[%0, %1] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          accln.terminator
        }) {sym_name = "_"} : () -> ()
        %3 = "accln.null_pred"() : () -> i1
        "accln.scheduled_kernel"(%3) {kernel = @_, sym_name = "scheduled__"} : (i1) -> ()
        "accln.schedule"() ( {
          "accln.exec_plan"() {exec_target = 0 : i64} : () -> ()
          accln.terminator
        }) {domain = #accln<"xfdomain{dims: {{i,0}, {j,1}, {k,2}}, indices: {{{i,0} : {0:128:1}}, {{j,1} : {0:256:1}}, {{k,2} : {0:256:1} = {(d0, d1) -> (d0 + d1), {{k_o,3}, {k_i,4}}}}, {{k_o,3} : {0:256:4}}, {{k_i,4} : {0:4:1}}}}">, kernels = [@scheduled__], loopattrs = [], order = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k_o,3}">, #accln<"index{k_i,4}">], parallel = [], unroll_and_jammed = {}, unrolled = [4 : index]} : () -> ()
        accln.terminator
      }) {domain = #accln<"idomain{{i,0}={0:128:1}, {j,1}={0:256:1}, {k,2}={0:256:1}}">, exec_target = 0 : i64, kernels = []} : () -> ()
      accv.return
    }
    accv.func @hello_matmul_py_0f07b3ac(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, accv.base_name = "hello_matmul_py", accv.emit_header_decl, accv.emit_raw_pointer_api} {
      accv.launch_func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0, %arg1, %arg2) {exec_target = 0 : i64} : (memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) -> ()
      accv.return
    }
  }
}
