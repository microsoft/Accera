module @optimized_matmul attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "optimized_matmul"  {
    accv.func @optimized_matmul_py_4a6286d9_impl_17630232307017152746(%arg0: memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, %arg1: memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, %arg2: memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
      %0 = accln.sym_index {name = "i_o"} #accln<"index{i_o,7}">
      %1 = accln.sym_index {name = "k_o"} #accln<"index{k_o,5}">
      %2 = accln.sym_index {name = "j_o"} #accln<"index{j_o,3}">
      %3 = accln.sym_index {name = "i"} #accln<"index{i,0}">
      %4 = accln.sym_index {name = "j"} #accln<"index{j,1}">
      %5 = accln.sym_index {name = "k"} #accln<"index{k,2}">
      "accln.nest"() ( {
        "accln.kernel"() ( {
          %11 = load %arg0[%3, %5] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
          %12 = load %arg1[%5, %4] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
          %13 = "accv.bin_op"(%11, %12) {predicate = 2 : i64} : (f32, f32) -> f32
          %14 = load %arg2[%3, %4] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
          %15 = "accv.bin_op"(%14, %13) {predicate = 0 : i64} : (f32, f32) -> f32
          store %15, %arg2[%3, %4] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
          %16 = load %arg2[%3, %4] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
          store %16, %arg2[%3, %4] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
          accln.terminator
        }) {sym_name = "_"} : () -> ()
        %6 = "accln.null_pred"() : () -> i1
        "accln.scheduled_kernel"(%6) {kernel = @_, sym_name = "scheduled__"} : (i1) -> ()
        %7 = "accxp.make_cache"() {memorySpace = 0 : i64} : () -> memref<16x128x16xf32, 3>
        %8 = "accxp.cache_region"(%arg1, %7, %2, %1) ( {
          accxp.cache_region_terminator
        }) {cacheAccessIndexing = 0 : i64, cacheAccessMaps = {globalInputToLogicalCache = affine_map<(d0, d1, d2, d3) -> (d2 - d1, d3 - d0)>, globalInputToPhysicalCache = affine_map<(d0, d1, d2, d3) -> (((d3 - d0) floordiv 16) mod 16, (d2 - d1) mod 128, (d3 - d0) mod 16)>, logicalCacheToGlobalInput = affine_map<(d0, d1, d2, d3) -> (d2 + d1, d3 + d0)>, logicalCacheToPhysicalCache = affine_map<(d0, d1) -> ((d1 floordiv 16) mod 16, d0 mod 128, d1 mod 16)>}, cacheDimGlobalIndices = [#accln<"index{k,2}">, #accln<"index{j,1}">], cacheGlobalDimensionSizes = [128, 512], id = 0 : i64, injectionIndex = #accln<"index{k_o,5}">, inputAccessIndexing = 0 : i64, inputAccessMaps = {globalInputToPhysicalCache = affine_map<(d0, d1) -> (d0, d1)>}} : (memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, memref<16x128x16xf32, 3>, index, index) -> index
        %9 = "accxp.make_cache"() {memorySpace = 0 : i64} : () -> memref<16x6x16xf32, 3>
        %10 = "accxp.cache_region"(%arg2, %9, %2, %1, %0) ( {
          accxp.cache_region_terminator
        }) {cacheAccessIndexing = 0 : i64, cacheAccessMaps = {globalInputToLogicalCache = affine_map<(d0, d1, d2, d3, d4) -> (d3 - d2, d4 - d0)>, globalInputToPhysicalCache = affine_map<(d0, d1, d2, d3, d4) -> (((d4 - d0) floordiv 16) mod 16, (d3 - d2) mod 6, (d4 - d0) mod 16)>, logicalCacheToGlobalInput = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d2, d4 + d0)>, logicalCacheToPhysicalCache = affine_map<(d0, d1) -> ((d1 floordiv 16) mod 16, d0 mod 6, d1 mod 16)>}, cacheDimGlobalIndices = [#accln<"index{i,0}">, #accln<"index{j,1}">], cacheGlobalDimensionSizes = [784, 512], id = 1 : i64, injectionIndex = #accln<"index{i_o,7}">, inputAccessIndexing = 0 : i64, inputAccessMaps = {globalInputToPhysicalCache = affine_map<(d0, d1) -> (d0, d1)>}} : (memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, memref<16x6x16xf32, 3>, index, index, index) -> index
        "accln.schedule"(%8, %10) ( {
          "accln.exec_plan"() {exec_target = 0 : i64} : () -> ()
          accln.terminator
        }) {domain = #accln<"xfdomain{dims: {{i,0}, {j,1}, {k,2}}, indices: {{{i,0} : {0:784:1} = {(d0, d1) -> (d0 + d1), {{i_o,7}, {i_i,8}}}}, {{j,1} : {0:512:1} = {(d0, d1) -> (d0 + d1), {{j_o,3}, {j_i,4}}}}, {{k,2} : {0:128:1} = {(d0, d1) -> (d0 + d1), {{k_o,5}, {k_i,6}}}}, {{j_o,3} : {0:512:256}}, {{j_i,4} : {0:256:1} = {(d0, d1) -> (d0 + d1), {{j_i_o,13}, {j_i_i,14}}}}, {{k_o,5} : {0:128:128}}, {{k_i,6} : {0:128:1} = {(d0, d1) -> (d0 + d1), {{k_i_o,9}, {k_i_i,10}}}}, {{i_o,7} : {0:784:1}}, {{i_i,8} : {0:1:1} = {(d0, d1) -> (d0 + d1), {{i_i_o,11}, {i_i_i,12}}}}, {{k_i_o,9} : {0:128:4}}, {{k_i_i,10} : {0:4:1}}, {{i_i_o,11} : {0:1:6}}, {{i_i_i,12} : {0:6:1}}, {{j_i_o,13} : {0:256:16}}, {{j_i_i,14} : {0:16:1} = {(d0, d1) -> (d0 + d1), {{j_i_i_o,15}, {j_i_i_i,16}}}}, {{j_i_i_o,15} : {0:16:8}}, {{j_i_i_i,16} : {0:8:1}}}}">, kernels = [@scheduled__], loopattrs = [{accxp_vectorizationInfo = #accxp<"vectorizationinfo{8,16,1}">, scheduledIndex = #accln<"index{j_i_i_i,16}">}], order = [#accln<"index{j_o,3}">, #accln<"index{k_o,5}">, #accln<"index{i_o,7}">, #accln<"index{j_i_o,13}">, #accln<"index{k_i_o,9}">, #accln<"index{i_i_o,11}">, #accln<"index{k_i_i,10}">, #accln<"index{i_i_i,12}">, #accln<"index{j_i_i_o,15}">, #accln<"index{j_i_i_i,16}">], parallel = [], unroll_and_jammed = {}, unrolled = [15 : index, 11 : index]} : (index, index) -> ()
        accln.terminator
      }) {domain = #accln<"idomain{{i,0}={0:784:1}, {j,1}={0:512:1}, {k,2}={0:128:1}}">, exec_target = 0 : i64, kernels = []} : () -> ()
      accv.return
    }
    accv.func @optimized_matmul_py_4a6286d9(%arg0: memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, %arg1: memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, %arg2: memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) attributes {exec_target = 0 : i64, accv.base_name = "optimized_matmul_py", accv.emit_header_decl, accv.emit_raw_pointer_api} {
      accv.launch_func @optimized_matmul_py_4a6286d9_impl_17630232307017152746(%arg0, %arg1, %arg2) {exec_target = 0 : i64} : (memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) -> ()
      accv.return
    }
  }
}
