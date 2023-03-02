// RUN: acc-opt --verify-each=false --pass-pipeline="accv.module(accv.func(loopnest-to-value-func))" %s | FileCheck %s

// This function has two caches initially, both marked thrifty, and one of them should
// get elided based on thrifty checks but the other should not

// This is the graph at the LoopNestToValueFuncPass_Subpasses_0_10_Canonicalize.mlir stage,
// which is the last canonicalize stage before the thrifty checks and the subpasses 
// before the thrifty phase create ops that the thrifty check depends on not being
// canonicalized before it runs
module @test_thrifty_caching_simple_input_cache attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"}  {
  accv.module "test_thrifty_caching_simple_input_cache"  {
    accv.func nested @test_thrifty_caching_simple_input_cache_1127a105_impl_6891397719071098712(%arg0: memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>, %arg1: memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>, %arg2: memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>) attributes {exec_target = 0 : i64} {
      %0 = accln.sym_index {name = "i_i"} #accln<"index{i_i,4}">
      %1 = accln.sym_index {name = "i_o"} #accln<"index{i_o,3}">
      %2 = accln.sym_index {name = "k_o"} #accln<"index{k_o,7}">
      %3 = accln.sym_index {name = "j_i"} #accln<"index{j_i,6}">
      %4 = accln.sym_index {name = "k_i"} #accln<"index{k_i,8}">
      %5 = accln.sym_index {name = "j_o"} #accln<"index{j_o,5}">
      "accv.lambda"() ({
        %6 = "accxp.make_cache"() {activeBlockToCacheMap = affine_map<(d0, d1) -> (d0, d1)>, memorySpace = 0 : i64, multiCacheAccessIndices = [], offsetAccessIndices = [], offsetArrayToCacheAccessMap = affine_map<(d0) -> (d0)>} : () -> memref<?xf32, 3>
        %7 = "accxp.begin_create_cache"(%arg0, %6, %arg0, %1, %2, %0, %4, %1, %2) {activeBlockCache, cacheAccessMaps = {manualCacheDimOrder = [0, 1]}, cacheHierarchyLevel = 0 : i64, cacheIndex = #accln<"index{i_i,4}">, cacheRegionBaseIndices = [[#accln<"index{i,0}">], [#accln<"index{k,2}">]], cacheRegionRelevantIndexRanges = [#accln<"indexrange{i_i,4}={0:4:1}">, #accln<"indexrange{k_i,8}={0:32:1}">], dimReorderCache, id = 0 : i64, operand_segment_sizes = dense<[1, 1, 1, 4, 2]> : vector<5xi32>, strategy = 1 : i32, thrifty, triggerIndex = #accln<"index{i_i,4}">} : (memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>, memref<?xf32, 3>, memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>, index, index, index, index, index, index) -> index
        "accxp.end_cache_region"(%7) : (index) -> ()
        %8 = "accxp.make_cache"() {activeBlockToCacheMap = affine_map<(d0, d1) -> (d0, d1)>, memorySpace = 0 : i64, multiCacheAccessIndices = [], offsetAccessIndices = [], offsetArrayToCacheAccessMap = affine_map<(d0) -> (d0)>} : () -> memref<?xf32, 3>
        %9 = "accxp.begin_create_cache"(%arg1, %8, %arg1, %5, %2, %3, %4, %5) {activeBlockCache, cacheAccessMaps = {manualCacheDimOrder = [0, 1]}, cacheHierarchyLevel = 0 : i64, cacheIndex = #accln<"index{k_o,7}">, cacheRegionBaseIndices = [[#accln<"index{k,2}">], [#accln<"index{j,1}">], [#accln<"index{k,2}">]], cacheRegionRelevantIndexRanges = [#accln<"indexrange{k_o,7}={0:32:32}">, #accln<"indexrange{j_i,6}={0:16:1}">, #accln<"indexrange{k_i,8}={0:32:1}">], dimReorderCache, id = 1 : i64, operand_segment_sizes = dense<[1, 1, 1, 4, 1]> : vector<5xi32>, strategy = 1 : i32, thrifty, triggerIndex = #accln<"index{k_o,7}">} : (memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>, memref<?xf32, 3>, memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>, index, index, index, index, index) -> index
        "accxp.end_cache_region"(%9) : (index) -> ()
        affine.for %arg3 = 0 to 32 step 4 {
          affine.for %arg4 = 0 to 32 step 16 {
            %10 = "accxp.begin_create_cache"(%arg1, %8, %arg1, %arg4, %2, %3, %4, %arg4) {activeBlockCache, cacheAccessMaps = {manualCacheDimOrder = [0, 1]}, cacheHierarchyLevel = 0 : i64, cacheIndex = #accln<"index{k_o,7}">, cacheRegionBaseIndices = [[#accln<"index{k,2}">], [#accln<"index{j,1}">], [#accln<"index{k,2}">]], cacheRegionRelevantIndexRanges = [#accln<"indexrange{k_o,7}={0:32:32}">, #accln<"indexrange{j_i,6}={0:16:1}">, #accln<"indexrange{k_i,8}={0:32:1}">], dimReorderCache, id = 1 : i64, operand_segment_sizes = dense<[1, 1, 1, 4, 1]> : vector<5xi32>, strategy = 1 : i32, thrifty, triggerIndex = #accln<"index{k_o,7}">} : (memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>, memref<?xf32, 3>, memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>, index, index, index, index, index) -> index
            affine.for %arg5 = 0 to 32 step 32 {
              %11 = "accxp.begin_create_cache"(%arg0, %6, %arg0, %arg3, %arg5, %0, %4, %arg3, %arg5) {activeBlockCache, cacheAccessMaps = {manualCacheDimOrder = [0, 1]}, cacheHierarchyLevel = 0 : i64, cacheIndex = #accln<"index{i_i,4}">, cacheRegionBaseIndices = [[#accln<"index{i,0}">], [#accln<"index{k,2}">]], cacheRegionRelevantIndexRanges = [#accln<"indexrange{i_i,4}={0:4:1}">, #accln<"indexrange{k_i,8}={0:32:1}">], dimReorderCache, id = 0 : i64, operand_segment_sizes = dense<[1, 1, 1, 4, 2]> : vector<5xi32>, strategy = 1 : i32, thrifty, triggerIndex = #accln<"index{i_i,4}">} : (memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>, memref<?xf32, 3>, memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>, index, index, index, index, index, index) -> index
              affine.for %arg6 = 0 to 4 {
                affine.for %arg7 = 0 to 16 {
                  affine.for %arg8 = 0 to 32 {
                    %12 = affine.load %arg0[%arg3 + %arg6, %arg5 + %arg8] : memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>
                    %13 = affine.load %arg1[%arg5 + %arg8, %arg4 + %arg7] : memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>
                    %14 = "accv.bin_op"(%12, %13) {predicate = 2 : i64} : (f32, f32) -> f32
                    %15 = affine.load %arg2[%arg3 + %arg6, %arg4 + %arg7] : memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>
                    %16 = "accv.bin_op"(%15, %14) {predicate = 0 : i64} : (f32, f32) -> f32
                    affine.store %16, %arg2[%arg3 + %arg6, %arg4 + %arg7] : memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>
                    %17 = affine.load %arg2[%arg3 + %arg6, %arg4 + %arg7] : memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>
                    affine.store %17, %arg2[%arg3 + %arg6, %arg4 + %arg7] : memref<32x32xf32, affine_map<(d0, d1) -> (d0 * 32 + d1)>>
                  } {begin = 0 : i64, end = 32 : i64, index = #accln<"index{k_i,8}">, kernels = ["_"], subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 1]}
                } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{j_i,6}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 32]}
              } {begin = 0 : i64, end = 4 : i64, index = #accln<"index{i_i,4}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 16, 32]}
              "accxp.end_cache_region"(%11) : (index) -> ()
            } {begin = 0 : i64, end = 32 : i64, index = #accln<"index{k_o,7}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [4, 16, 32]}
            "accxp.end_cache_region"(%10) : (index) -> ()
          } {begin = 0 : i64, end = 32 : i64, index = #accln<"index{j_o,5}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [4, 16, 32]}
        } {begin = 0 : i64, end = 32 : i64, index = #accln<"index{i_o,3}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [4, 32, 32]}
        accv.return
      }) {exec_target = 0 : i64, sym_name = "NestFunction_0", type = () -> ()} : () -> ()
      accv.return
    }
  }
}

// CHECK: #map0 = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK: #map1 = affine_map<() -> (0)>
// CHECK: #map2 = affine_map<() -> (16)>
// CHECK: #map3 = affine_map<() -> (32)>
// CHECK: module @test_thrifty_caching_simple_input_cache
// CHECK:   accv.module "test_thrifty_caching_simple_input_cache" {
// CHECK:     "accv.global"() {sym_name = "cache_6", type = memref<32x16xf32, 3>} : () -> ()
// CHECK:     accv.func nested @test_thrifty_caching_simple_input_cache_1127a105_impl_6891397719071098712(%arg0: memref<32x32xf32, #map0>, %arg1: memref<32x32xf32, #map0>, %arg2: memref<32x32xf32, #map0>) attributes {exec_target = 0 : i64} {
// CHECK:       %0 = "accv.ref_global"() {global_name = @cache_6} : () -> memref<32x16xf32, 3>
// CHECK:       affine.for %arg3 = 0 to 32 step 4 {
// CHECK:         affine.for %arg4 = 0 to 32 step 16 {
// CHECK:           affine.for %arg5 = 0 to 32 {
// CHECK:             affine.for %arg6 = 0 to 16 {
// CHECK:               %1 = affine.load %arg1[%arg5, %arg4 + %arg6] : memref<32x32xf32, #map0>
// CHECK:               affine.store %1, %0[%arg5, %arg6] : memref<32x16xf32, 3>
// CHECK:             } {accaffine.access_bounds_check, beginMap = #map1, endMap = #map2, index = #accln<"index{j,7}">, kernels = ["cache_internal_loopnest_kernel_active_block_copy"], operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, scheduledIndex = #accln<"index{j,7}">, subdomainIndexOrder = [#accln<"index{i,6}">, #accln<"index{j,7}">], subdomainSize = [32, 16]}
// CHECK:           } {accaffine.access_bounds_check, beginMap = #map1, endMap = #map3, index = #accln<"index{i,6}">, operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, scheduledIndex = #accln<"index{i,6}">, subdomainIndexOrder = [#accln<"index{i,6}">, #accln<"index{j,7}">], subdomainSize = [32, 16]}
// CHECK:           affine.for %arg5 = 0 to 4 {
// CHECK:             affine.for %arg6 = 0 to 16 {
// CHECK:               affine.for %arg7 = 0 to 32 {
// CHECK:                 %1 = affine.load %arg0[%arg3 + %arg5, %arg7] : memref<32x32xf32, #map0>
// CHECK:                 %2 = affine.load %0[%arg7, %arg6] : memref<32x16xf32, 3>
// CHECK:                 %3 = "accv.bin_op"(%1, %2) {predicate = 2 : i64} : (f32, f32) -> f32
// CHECK:                 %4 = affine.load %arg2[%arg3 + %arg5, %arg4 + %arg6] : memref<32x32xf32, #map0>
// CHECK:                 %5 = "accv.bin_op"(%4, %3) {predicate = 0 : i64} : (f32, f32) -> f32
// CHECK:                 affine.store %5, %arg2[%arg3 + %arg5, %arg4 + %arg6] : memref<32x32xf32, #map0>
// CHECK:                 %6 = affine.load %arg2[%arg3 + %arg5, %arg4 + %arg6] : memref<32x32xf32, #map0>
// CHECK:                 affine.store %6, %arg2[%arg3 + %arg5, %arg4 + %arg6] : memref<32x32xf32, #map0>
// CHECK:               } {begin = 0 : i64, end = 32 : i64, index = #accln<"index{k_i,8}">, kernels = ["_"], subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 1]}
// CHECK:             } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{j_i,6}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 32]}
// CHECK:           } {begin = 0 : i64, end = 4 : i64, index = #accln<"index{i_i,4}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 16, 32]}
// CHECK:         } {begin = 0 : i64, end = 32 : i64, index = #accln<"index{j_o,5}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [4, 16, 32]}
// CHECK:       } {begin = 0 : i64, end = 32 : i64, index = #accln<"index{i_o,3}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [4, 32, 32]}
// CHECK:       accv.return
// CHECK:     }
// CHECK:   }
// CHECK: }
