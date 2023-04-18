#map0 = affine_map<(d0, d1) -> (d0 * 256 + d1)>
#map1 = affine_map<()[s0] -> (s0)>


#domain0 = #accln<"idomain{{i,0}={0:128:1}, {j,1}={0:256:1}, {k,2}={0:256:1}}">

#xdomain0 = #accln<"xfdomain{dims: {{i,0}, {j,1}, {k,2}}, indices: {{{i,0} : {0:128:1}}, {{j,1} : {0:256:1}}, {{k,2} : {0:256:1} = {(d0, d1) -> (d0 + d1), {{k_o,3}, {k_i,4}}}}, {{k_o,3} : {0:256:4}}, {{k_i,4} : {0:4:1}}}}">

module @hello_matmul attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "hello_matmul"  {
    accv.func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0: memref<128x256xf32, #map0>, %arg1: memref<256x256xf32, #map0>, %arg2: memref<128x256xf32, #map0>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
      "accln.nest"() ( {
        %0 = accln.sym_index {name = "k_i", reference = "k"} #accln<"index{k_i,4}"> loc(unknown)
        %1 = accln.sym_index {name = "k_o", reference = "k"} #accln<"index{k_o,3}"> loc(unknown)
        %2 = accln.sym_index {name = "k_i"} #accln<"index{k_i,4}"> loc(unknown)
        %3 = accln.sym_index {name = "k_o"} #accln<"index{k_o,3}"> loc(unknown)
        %4 = accln.sym_index {name = "i"} #accln<"index{i,0}"> loc(unknown)
        %5 = accln.sym_index {name = "j"} #accln<"index{j,1}"> loc(unknown)
        %6 = accln.sym_index {name = "k"} #accln<"index{k,2}"> loc(unknown)
        "accln.kernel"() ( {
          %8 = "accv.slice"(%arg2, %4, %5) {sliceDimensions = [0, 1]} : (memref<128x256xf32, #map0>, index, index) -> memref<f32, #map1> loc(unknown)
          %9 = "accv.slice"(%arg0, %4, %6) {sliceDimensions = [0, 1]} : (memref<128x256xf32, #map0>, index, index) -> memref<f32, #map1> loc(unknown)
          %10 = "accv.slice"(%arg1, %6, %5) {sliceDimensions = [0, 1]} : (memref<256x256xf32, #map0>, index, index) -> memref<f32, #map1> loc(unknown)
          %11 = "accv.get_element"(%9) : (memref<f32, #map1>) -> f32 loc(unknown)
          %12 = "accv.get_element"(%10) : (memref<f32, #map1>) -> f32 loc(unknown)
          %13 = "accv.bin_op"(%11, %12) {predicate = 2 : i64} : (f32, f32) -> f32 loc(unknown)
          %14 = "accv.get_element"(%8) : (memref<f32, #map1>) -> f32 loc(unknown)
          %15 = "accv.bin_op"(%14, %13) {predicate = 0 : i64} : (f32, f32) -> f32 loc(unknown)
          "accv.copy"(%15, %8) : (f32, memref<f32, #map1>) -> () loc(unknown)
          %16 = "accv.slice"(%arg2, %4, %5) {sliceDimensions = [0, 1]} : (memref<128x256xf32, #map0>, index, index) -> memref<f32, #map1> loc(unknown)
          %17 = "accv.get_element"(%8) : (memref<f32, #map1>) -> f32 loc(unknown)
          "accv.copy"(%17, %16) : (f32, memref<f32, #map1>) -> () loc(unknown)
          accln.terminator loc(unknown)
        }) {sym_name = "_"} : () -> () loc(unknown)
        %7 = "accln.null_pred"() : () -> i1 loc(unknown)
        "accln.scheduled_kernel"(%7) {kernel = @_, sym_name = "scheduled__"} : (i1) -> () loc(unknown)
        "accln.schedule"() ( {
          "accln.exec_plan"() {exec_target = 0 : i64} : () -> () loc(unknown)
          accln.terminator loc(unknown)
        }) {domain = #xdomain0, kernels = [@scheduled__], loopattrs = [], order = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k_o,3}">, #accln<"index{k_i,4}">], parallel = [], unroll_and_jammed = {}, unrolled = [4 : index]} : () -> () loc(unknown)
        accln.terminator loc(unknown)
      }) {domain = #domain0, exec_target = 0 : i64, kernels = []} : () -> () loc(unknown)
      accv.return loc(unknown)
    } loc(unknown)
    accv.func @hello_matmul_py_0f07b3ac(%arg0: memref<128x256xf32, #map0>, %arg1: memref<256x256xf32, #map0>, %arg2: memref<128x256xf32, #map0>) attributes {exec_target = 0 : i64, accv.base_name = "hello_matmul_py", accv.emit_header_decl, accv.emit_raw_pointer_api} {
      accv.launch_func @hello_matmul_py_0f07b3ac_impl_16252232176815793891(%arg0, %arg1, %arg2) {exec_target = 0 : i64} : (memref<128x256xf32, #map0>, memref<256x256xf32, #map0>, memref<128x256xf32, #map0>) -> () loc(unknown)
      accv.return loc(unknown)
    } loc(unknown)
  } loc(unknown)
} loc(unknown)
