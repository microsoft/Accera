module @optimized_matmul attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "optimized_matmul"  {
    "accv.global"() {sym_name = "cache_17", type = memref<16x128x2xvector<8xf32>>} : () -> ()
    "accv.global"() {sym_name = "cache_16", type = memref<16x6x2xvector<8xf32>>} : () -> ()
    accv.func @optimized_matmul_py_4a6286d9_impl_17630232307017152746(%arg0: memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, %arg1: memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, %arg2: memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
      %cst = constant dense<0.000000e+00> : vector<8xf32>
      %cst_0 = constant 0.000000e+00 : f32
      %c0_i64 = constant 0 : i64
      %c1_i64 = constant 1 : i64
      %c2_i64 = constant 2 : i64
      %c3_i64 = constant 3 : i64
      %c4_i64 = constant 4 : i64
      %c5_i64 = constant 5 : i64
      %c6_i64 = constant 6 : i64
      %c7_i64 = constant 7 : i64
      %0 = alloca() {alignment = 32 : i64} : memref<1x16xvector<8xf32>>
      %1 = alloca() {alignment = 32 : i64} : memref<1x16xvector<8xf32>>
      "accv.lambda"() ( {
        %2 = "accv.ref_global"() {global_name = @cache_16} : () -> memref<16x6x2xvector<8xf32>>
        %3 = "accv.ref_global"() {global_name = @cache_17} : () -> memref<16x128x2xvector<8xf32>>
        affine.for %arg3 = 0 to 512 step 256 {
          affine.for %arg4 = 0 to 128 step 128 {
            "accv.lambda"() ( {
              affine.for %arg5 = 0 to 128 {
                affine.for %arg6 = 0 to 256 step 128 {
                  affine.if affine_set<() : (0 == 0)>() {
                    "accv.lambda"() ( {
                      affine.for %arg7 = 0 to 1 {
                        affine.for %arg8 = 0 to 16 {
                          %4 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg5, %arg7)
                          %5 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2 * 8)>(%arg3, %arg6, %arg8)
                          %6 = vector.transfer_read %arg1[%4, %5], %cst_0 {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
                          store %6, %0[%arg7, %arg8] : memref<1x16xvector<8xf32>>
                        } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{i_1,24}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{i_0,23}">, #accln<"index{i_1,24}">], subdomainSize = [1, 1]}
                      } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{i_0,23}">, subdomainIndexOrder = [#accln<"index{i_0,23}">, #accln<"index{i_1,24}">], subdomainSize = [1, 16]}
                      accv.return
                    }) {exec_target = 0 : i64, sym_name = "NestFunction_15", type = () -> ()} : () -> ()
                    "accv.lambda"() ( {
                      affine.for %arg7 = 0 to 1 {
                        affine.for %arg8 = 0 to 16 {
                          %4 = load %0[%arg7, %arg8] : memref<1x16xvector<8xf32>>
                          affine.store %4, %3[((%arg6 + %arg8 * 8) floordiv 16) mod 16, (%arg5 + %arg7) mod 128, (((%arg6 + %arg8 * 8) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                        } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{i_1,26}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{i_0,25}">, #accln<"index{i_1,26}">], subdomainSize = [1, 1]}
                      } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{i_0,25}">, subdomainIndexOrder = [#accln<"index{i_0,25}">, #accln<"index{i_1,26}">], subdomainSize = [1, 16]}
                      accv.return
                    }) {exec_target = 0 : i64, sym_name = "NestFunction_14", type = () -> ()} : () -> ()
                  } else {
                    "accv.lambda"() ( {
                      affine.for %arg7 = 0 to 1 {
                        affine.for %arg8 = 0 to 16 {
                          %4 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg5, %arg7)
                          %5 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2 * 8)>(%arg3, %arg6, %arg8)
                          %6 = vector.transfer_read %arg1[%4, %5], %cst_0 : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
                          store %6, %0[%arg7, %arg8] : memref<1x16xvector<8xf32>>
                        } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{i_1,28}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{i_0,27}">, #accln<"index{i_1,28}">], subdomainSize = [1, 1]}
                      } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{i_0,27}">, subdomainIndexOrder = [#accln<"index{i_0,27}">, #accln<"index{i_1,28}">], subdomainSize = [1, 16]}
                      accv.return
                    }) {exec_target = 0 : i64, sym_name = "NestFunction_13", type = () -> ()} : () -> ()
                    "accv.lambda"() ( {
                      affine.for %arg7 = 0 to 1 {
                        affine.for %arg8 = 0 to 16 {
                          %4 = load %0[%arg7, %arg8] : memref<1x16xvector<8xf32>>
                          affine.store %4, %3[((%arg6 + %arg8 * 8) floordiv 16) mod 16, (%arg5 + %arg7) mod 128, (((%arg6 + %arg8 * 8) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                        } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{i_1,30}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{i_0,29}">, #accln<"index{i_1,30}">], subdomainSize = [1, 1]}
                      } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{i_0,29}">, subdomainIndexOrder = [#accln<"index{i_0,29}">, #accln<"index{i_1,30}">], subdomainSize = [1, 16]}
                      accv.return
                    }) {exec_target = 0 : i64, sym_name = "NestFunction_12", type = () -> ()} : () -> ()
                  }
                } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{j_o,21}">, subdomainIndexOrder = [#accln<"index{i,17}">, #accln<"index{j,18}">], subdomainSize = [1, 128]}
              } {begin = 0 : i64, end = 128 : i64, index = #accln<"index{i_o,19}">, subdomainIndexOrder = [#accln<"index{i,17}">, #accln<"index{j,18}">], subdomainSize = [1, 256]}
              accv.return
            }) {exec_target = 0 : i64, sym_name = "NestFunction_7", type = () -> ()} : () -> ()
            affine.for %arg5 = 0 to 784 {
              "accv.lambda"() ( {
                affine.for %arg6 = 0 to 16 {
                  affine.for %arg7 = 0 to 6 {
                    affine.for %arg8 = 0 to 2 {
                      store %cst, %2[%arg6, %arg7, %arg8] : memref<16x6x2xvector<8xf32>>
                    } {begin = 0 : i64, end = 2 : i64, index = #accln<"index{j_i_i_i,16}">, subdomainIndexOrder = [#accln<"index{j_i_i,14}">, #accln<"index{j_i_i_o,15}">, #accln<"index{j_i_i_i,16}">], subdomainSize = [1, 1, 1]}
                  } {begin = 0 : i64, end = 6 : i64, index = #accln<"index{j_i_i_o,15}">, subdomainIndexOrder = [#accln<"index{j_i_i,14}">, #accln<"index{j_i_i_o,15}">, #accln<"index{j_i_i_i,16}">], subdomainSize = [1, 1, 2]}
                } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{j_i_i,14}">, subdomainIndexOrder = [#accln<"index{j_i_i,14}">, #accln<"index{j_i_i_o,15}">, #accln<"index{j_i_i_i,16}">], subdomainSize = [1, 6, 2]}
                accv.return
              }) {exec_target = 0 : i64, sym_name = "NestFunction_6", type = () -> ()} : () -> ()
              affine.for %arg6 = 0 to 256 step 16 {
                affine.for %arg7 = 0 to 128 step 4 {
                  affine.for %arg8 = 0 to 0 step 6 {
                    affine.for %arg9 = 0 to 4 {
                      affine.for %arg10 = 0 to 0 {
                        affine.for %arg11 = 0 to 16 step 8 {
                          affine.for %arg12 = 0 to 8 step 8 {
                            %4 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %5 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %6 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %8 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %12 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %13 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %19 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %21 = load %arg0[%4, %13] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %22 = load %arg0[%5, %14] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %23 = load %arg0[%6, %15] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %24 = load %arg0[%7, %16] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %25 = load %arg0[%8, %17] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %26 = load %arg0[%9, %18] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %27 = load %arg0[%10, %19] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %28 = load %arg0[%11, %20] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %29 = affine.load %3[((%12 - %arg3) floordiv 16) mod 16, (%13 - %arg4) mod 128, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %30 = vector.extractelement %29[%c0_i64 : i64] : vector<8xf32>
                            %31 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %32 = affine.load %3[((%31 - %arg3) floordiv 16) mod 16, (%14 - %arg4) mod 128, (((%31 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %33 = vector.extractelement %32[%c1_i64 : i64] : vector<8xf32>
                            %34 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %35 = affine.load %3[((%34 - %arg3) floordiv 16) mod 16, (%15 - %arg4) mod 128, (((%34 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %36 = vector.extractelement %35[%c2_i64 : i64] : vector<8xf32>
                            %37 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %38 = affine.load %3[((%37 - %arg3) floordiv 16) mod 16, (%16 - %arg4) mod 128, (((%37 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %39 = vector.extractelement %38[%c3_i64 : i64] : vector<8xf32>
                            %40 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %41 = affine.load %3[((%40 - %arg3) floordiv 16) mod 16, (%17 - %arg4) mod 128, (((%40 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %42 = vector.extractelement %41[%c4_i64 : i64] : vector<8xf32>
                            %43 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %44 = affine.load %3[((%43 - %arg3) floordiv 16) mod 16, (%18 - %arg4) mod 128, (((%43 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %45 = vector.extractelement %44[%c5_i64 : i64] : vector<8xf32>
                            %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %47 = affine.load %3[((%46 - %arg3) floordiv 16) mod 16, (%19 - %arg4) mod 128, (((%46 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %48 = vector.extractelement %47[%c6_i64 : i64] : vector<8xf32>
                            %49 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %50 = affine.load %3[((%49 - %arg3) floordiv 16) mod 16, (%20 - %arg4) mod 128, (((%49 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %51 = vector.extractelement %50[%c7_i64 : i64] : vector<8xf32>
                            %52 = "accv.bin_op"(%21, %30) {predicate = 2 : i64} : (f32, f32) -> f32
                            %53 = "accv.bin_op"(%22, %33) {predicate = 2 : i64} : (f32, f32) -> f32
                            %54 = "accv.bin_op"(%23, %36) {predicate = 2 : i64} : (f32, f32) -> f32
                            %55 = "accv.bin_op"(%24, %39) {predicate = 2 : i64} : (f32, f32) -> f32
                            %56 = "accv.bin_op"(%25, %42) {predicate = 2 : i64} : (f32, f32) -> f32
                            %57 = "accv.bin_op"(%26, %45) {predicate = 2 : i64} : (f32, f32) -> f32
                            %58 = "accv.bin_op"(%27, %48) {predicate = 2 : i64} : (f32, f32) -> f32
                            %59 = "accv.bin_op"(%28, %51) {predicate = 2 : i64} : (f32, f32) -> f32
                            %60 = affine.load %2[((%12 - %arg3) floordiv 16) mod 16, (%4 - %arg5) mod 6, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %61 = vector.extractelement %60[%c0_i64 : i64] : vector<8xf32>
                            %62 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %63 = affine.load %2[((%62 - %arg3) floordiv 16) mod 16, (%5 - %arg5) mod 6, (((%62 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %64 = vector.extractelement %63[%c1_i64 : i64] : vector<8xf32>
                            %65 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %66 = affine.load %2[((%65 - %arg3) floordiv 16) mod 16, (%6 - %arg5) mod 6, (((%65 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %67 = vector.extractelement %66[%c2_i64 : i64] : vector<8xf32>
                            %68 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %69 = affine.load %2[((%68 - %arg3) floordiv 16) mod 16, (%7 - %arg5) mod 6, (((%68 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %70 = vector.extractelement %69[%c3_i64 : i64] : vector<8xf32>
                            %71 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %72 = affine.load %2[((%71 - %arg3) floordiv 16) mod 16, (%8 - %arg5) mod 6, (((%71 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %73 = vector.extractelement %72[%c4_i64 : i64] : vector<8xf32>
                            %74 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %75 = affine.load %2[((%74 - %arg3) floordiv 16) mod 16, (%9 - %arg5) mod 6, (((%74 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %76 = vector.extractelement %75[%c5_i64 : i64] : vector<8xf32>
                            %77 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %78 = affine.load %2[((%77 - %arg3) floordiv 16) mod 16, (%10 - %arg5) mod 6, (((%77 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %79 = vector.extractelement %78[%c6_i64 : i64] : vector<8xf32>
                            %80 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %81 = affine.load %2[((%80 - %arg3) floordiv 16) mod 16, (%11 - %arg5) mod 6, (((%80 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %82 = vector.extractelement %81[%c7_i64 : i64] : vector<8xf32>
                            %83 = "accv.bin_op"(%61, %52) {predicate = 0 : i64} : (f32, f32) -> f32
                            %84 = "accv.bin_op"(%64, %53) {predicate = 0 : i64} : (f32, f32) -> f32
                            %85 = "accv.bin_op"(%67, %54) {predicate = 0 : i64} : (f32, f32) -> f32
                            %86 = "accv.bin_op"(%70, %55) {predicate = 0 : i64} : (f32, f32) -> f32
                            %87 = "accv.bin_op"(%73, %56) {predicate = 0 : i64} : (f32, f32) -> f32
                            %88 = "accv.bin_op"(%76, %57) {predicate = 0 : i64} : (f32, f32) -> f32
                            %89 = "accv.bin_op"(%79, %58) {predicate = 0 : i64} : (f32, f32) -> f32
                            %90 = "accv.bin_op"(%82, %59) {predicate = 0 : i64} : (f32, f32) -> f32
                            %91 = affine.load %2[((%12 - %arg3) floordiv 16) mod 16, (%4 - %arg5) mod 6, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %92 = vector.insertelement %83, %91[%c0_i64 : i64] : vector<8xf32>
                            affine.store %92, %2[((%12 - %arg3) floordiv 16) mod 16, (%4 - %arg5) mod 6, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %93 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %94 = affine.load %2[((%93 - %arg3) floordiv 16) mod 16, (%5 - %arg5) mod 6, (((%93 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %95 = vector.insertelement %84, %94[%c1_i64 : i64] : vector<8xf32>
                            affine.store %95, %2[((%93 - %arg3) floordiv 16) mod 16, (%5 - %arg5) mod 6, (((%93 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %96 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %97 = affine.load %2[((%96 - %arg3) floordiv 16) mod 16, (%6 - %arg5) mod 6, (((%96 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %98 = vector.insertelement %85, %97[%c2_i64 : i64] : vector<8xf32>
                            affine.store %98, %2[((%96 - %arg3) floordiv 16) mod 16, (%6 - %arg5) mod 6, (((%96 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %100 = affine.load %2[((%99 - %arg3) floordiv 16) mod 16, (%7 - %arg5) mod 6, (((%99 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %101 = vector.insertelement %86, %100[%c3_i64 : i64] : vector<8xf32>
                            affine.store %101, %2[((%99 - %arg3) floordiv 16) mod 16, (%7 - %arg5) mod 6, (((%99 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %102 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %103 = affine.load %2[((%102 - %arg3) floordiv 16) mod 16, (%8 - %arg5) mod 6, (((%102 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %104 = vector.insertelement %87, %103[%c4_i64 : i64] : vector<8xf32>
                            affine.store %104, %2[((%102 - %arg3) floordiv 16) mod 16, (%8 - %arg5) mod 6, (((%102 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %105 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %106 = affine.load %2[((%105 - %arg3) floordiv 16) mod 16, (%9 - %arg5) mod 6, (((%105 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %107 = vector.insertelement %88, %106[%c5_i64 : i64] : vector<8xf32>
                            affine.store %107, %2[((%105 - %arg3) floordiv 16) mod 16, (%9 - %arg5) mod 6, (((%105 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %108 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %109 = affine.load %2[((%108 - %arg3) floordiv 16) mod 16, (%10 - %arg5) mod 6, (((%108 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %110 = vector.insertelement %89, %109[%c6_i64 : i64] : vector<8xf32>
                            affine.store %110, %2[((%108 - %arg3) floordiv 16) mod 16, (%10 - %arg5) mod 6, (((%108 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %111 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %112 = affine.load %2[((%111 - %arg3) floordiv 16) mod 16, (%11 - %arg5) mod 6, (((%111 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %113 = vector.insertelement %90, %112[%c7_i64 : i64] : vector<8xf32>
                            affine.store %113, %2[((%111 - %arg3) floordiv 16) mod 16, (%11 - %arg5) mod 6, (((%111 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %114 = affine.load %2[((%12 - %arg3) floordiv 16) mod 16, (%4 - %arg5) mod 6, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %115 = vector.insertelement %83, %114[%c0_i64 : i64] : vector<8xf32>
                            affine.store %115, %2[((%12 - %arg3) floordiv 16) mod 16, (%4 - %arg5) mod 6, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %116 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %117 = affine.load %2[((%116 - %arg3) floordiv 16) mod 16, (%5 - %arg5) mod 6, (((%116 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %118 = vector.insertelement %84, %117[%c1_i64 : i64] : vector<8xf32>
                            affine.store %118, %2[((%116 - %arg3) floordiv 16) mod 16, (%5 - %arg5) mod 6, (((%116 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %119 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %120 = affine.load %2[((%119 - %arg3) floordiv 16) mod 16, (%6 - %arg5) mod 6, (((%119 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %121 = vector.insertelement %85, %120[%c2_i64 : i64] : vector<8xf32>
                            affine.store %121, %2[((%119 - %arg3) floordiv 16) mod 16, (%6 - %arg5) mod 6, (((%119 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %122 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %123 = affine.load %2[((%122 - %arg3) floordiv 16) mod 16, (%7 - %arg5) mod 6, (((%122 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %124 = vector.insertelement %86, %123[%c3_i64 : i64] : vector<8xf32>
                            affine.store %124, %2[((%122 - %arg3) floordiv 16) mod 16, (%7 - %arg5) mod 6, (((%122 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %125 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %126 = affine.load %2[((%125 - %arg3) floordiv 16) mod 16, (%8 - %arg5) mod 6, (((%125 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %127 = vector.insertelement %87, %126[%c4_i64 : i64] : vector<8xf32>
                            affine.store %127, %2[((%125 - %arg3) floordiv 16) mod 16, (%8 - %arg5) mod 6, (((%125 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %128 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %129 = affine.load %2[((%128 - %arg3) floordiv 16) mod 16, (%9 - %arg5) mod 6, (((%128 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %130 = vector.insertelement %88, %129[%c5_i64 : i64] : vector<8xf32>
                            affine.store %130, %2[((%128 - %arg3) floordiv 16) mod 16, (%9 - %arg5) mod 6, (((%128 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %131 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %132 = affine.load %2[((%131 - %arg3) floordiv 16) mod 16, (%10 - %arg5) mod 6, (((%131 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %133 = vector.insertelement %89, %132[%c6_i64 : i64] : vector<8xf32>
                            affine.store %133, %2[((%131 - %arg3) floordiv 16) mod 16, (%10 - %arg5) mod 6, (((%131 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %134 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %135 = affine.load %2[((%134 - %arg3) floordiv 16) mod 16, (%11 - %arg5) mod 6, (((%134 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %136 = vector.insertelement %90, %135[%c7_i64 : i64] : vector<8xf32>
                            affine.store %136, %2[((%134 - %arg3) floordiv 16) mod 16, (%11 - %arg5) mod 6, (((%134 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                          } {begin = 0 : i64, end = 8 : i64, index = #accln<"index{j_i_i_i,16}">, scheduledIndex = #accln<"index{j_i_i_i,16}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [0, 1, 1]}
                        } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{j_i_i_o,15}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [0, 8, 1]}
                      } {begin = 0 : i64, end = 0 : i64, index = #accln<"index{i_i_i,12}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [0, 16, 1]}
                    } {begin = 0 : i64, end = 4 : i64, index = #accln<"index{k_i_i,10}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [0, 16, 1]}
                  } {begin = 0 : i64, end = 0 : i64, index = #accln<"index{i_i_o,11}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [0, 16, 4]}
                  affine.for %arg8 = 0 to 1 step 6 {
                    affine.for %arg9 = 0 to 4 {
                      affine.for %arg10 = 0 to 1 {
                        affine.for %arg11 = 0 to 16 step 8 {
                          affine.for %arg12 = 0 to 8 step 8 {
                            %4 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %5 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %6 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %8 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg8, %arg10)
                            %12 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %13 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %19 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg4, %arg7, %arg9)
                            %21 = load %arg0[%4, %13] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %22 = load %arg0[%5, %14] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %23 = load %arg0[%6, %15] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %24 = load %arg0[%7, %16] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %25 = load %arg0[%8, %17] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %26 = load %arg0[%9, %18] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %27 = load %arg0[%10, %19] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %28 = load %arg0[%11, %20] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                            %29 = affine.load %3[((%12 - %arg3) floordiv 16) mod 16, (%13 - %arg4) mod 128, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %30 = vector.extractelement %29[%c0_i64 : i64] : vector<8xf32>
                            %31 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %32 = affine.load %3[((%31 - %arg3) floordiv 16) mod 16, (%14 - %arg4) mod 128, (((%31 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %33 = vector.extractelement %32[%c1_i64 : i64] : vector<8xf32>
                            %34 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %35 = affine.load %3[((%34 - %arg3) floordiv 16) mod 16, (%15 - %arg4) mod 128, (((%34 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %36 = vector.extractelement %35[%c2_i64 : i64] : vector<8xf32>
                            %37 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %38 = affine.load %3[((%37 - %arg3) floordiv 16) mod 16, (%16 - %arg4) mod 128, (((%37 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %39 = vector.extractelement %38[%c3_i64 : i64] : vector<8xf32>
                            %40 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %41 = affine.load %3[((%40 - %arg3) floordiv 16) mod 16, (%17 - %arg4) mod 128, (((%40 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %42 = vector.extractelement %41[%c4_i64 : i64] : vector<8xf32>
                            %43 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %44 = affine.load %3[((%43 - %arg3) floordiv 16) mod 16, (%18 - %arg4) mod 128, (((%43 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %45 = vector.extractelement %44[%c5_i64 : i64] : vector<8xf32>
                            %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %47 = affine.load %3[((%46 - %arg3) floordiv 16) mod 16, (%19 - %arg4) mod 128, (((%46 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %48 = vector.extractelement %47[%c6_i64 : i64] : vector<8xf32>
                            %49 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %50 = affine.load %3[((%49 - %arg3) floordiv 16) mod 16, (%20 - %arg4) mod 128, (((%49 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x128x2xvector<8xf32>>
                            %51 = vector.extractelement %50[%c7_i64 : i64] : vector<8xf32>
                            %52 = "accv.bin_op"(%21, %30) {predicate = 2 : i64} : (f32, f32) -> f32
                            %53 = "accv.bin_op"(%22, %33) {predicate = 2 : i64} : (f32, f32) -> f32
                            %54 = "accv.bin_op"(%23, %36) {predicate = 2 : i64} : (f32, f32) -> f32
                            %55 = "accv.bin_op"(%24, %39) {predicate = 2 : i64} : (f32, f32) -> f32
                            %56 = "accv.bin_op"(%25, %42) {predicate = 2 : i64} : (f32, f32) -> f32
                            %57 = "accv.bin_op"(%26, %45) {predicate = 2 : i64} : (f32, f32) -> f32
                            %58 = "accv.bin_op"(%27, %48) {predicate = 2 : i64} : (f32, f32) -> f32
                            %59 = "accv.bin_op"(%28, %51) {predicate = 2 : i64} : (f32, f32) -> f32
                            %60 = affine.load %2[((%12 - %arg3) floordiv 16) mod 16, (%4 - %arg5) mod 6, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %61 = vector.extractelement %60[%c0_i64 : i64] : vector<8xf32>
                            %62 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %63 = affine.load %2[((%62 - %arg3) floordiv 16) mod 16, (%5 - %arg5) mod 6, (((%62 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %64 = vector.extractelement %63[%c1_i64 : i64] : vector<8xf32>
                            %65 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %66 = affine.load %2[((%65 - %arg3) floordiv 16) mod 16, (%6 - %arg5) mod 6, (((%65 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %67 = vector.extractelement %66[%c2_i64 : i64] : vector<8xf32>
                            %68 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %69 = affine.load %2[((%68 - %arg3) floordiv 16) mod 16, (%7 - %arg5) mod 6, (((%68 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %70 = vector.extractelement %69[%c3_i64 : i64] : vector<8xf32>
                            %71 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %72 = affine.load %2[((%71 - %arg3) floordiv 16) mod 16, (%8 - %arg5) mod 6, (((%71 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %73 = vector.extractelement %72[%c4_i64 : i64] : vector<8xf32>
                            %74 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %75 = affine.load %2[((%74 - %arg3) floordiv 16) mod 16, (%9 - %arg5) mod 6, (((%74 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %76 = vector.extractelement %75[%c5_i64 : i64] : vector<8xf32>
                            %77 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %78 = affine.load %2[((%77 - %arg3) floordiv 16) mod 16, (%10 - %arg5) mod 6, (((%77 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %79 = vector.extractelement %78[%c6_i64 : i64] : vector<8xf32>
                            %80 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %81 = affine.load %2[((%80 - %arg3) floordiv 16) mod 16, (%11 - %arg5) mod 6, (((%80 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %82 = vector.extractelement %81[%c7_i64 : i64] : vector<8xf32>
                            %83 = "accv.bin_op"(%61, %52) {predicate = 0 : i64} : (f32, f32) -> f32
                            %84 = "accv.bin_op"(%64, %53) {predicate = 0 : i64} : (f32, f32) -> f32
                            %85 = "accv.bin_op"(%67, %54) {predicate = 0 : i64} : (f32, f32) -> f32
                            %86 = "accv.bin_op"(%70, %55) {predicate = 0 : i64} : (f32, f32) -> f32
                            %87 = "accv.bin_op"(%73, %56) {predicate = 0 : i64} : (f32, f32) -> f32
                            %88 = "accv.bin_op"(%76, %57) {predicate = 0 : i64} : (f32, f32) -> f32
                            %89 = "accv.bin_op"(%79, %58) {predicate = 0 : i64} : (f32, f32) -> f32
                            %90 = "accv.bin_op"(%82, %59) {predicate = 0 : i64} : (f32, f32) -> f32
                            %91 = affine.load %2[((%12 - %arg3) floordiv 16) mod 16, (%4 - %arg5) mod 6, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %92 = vector.insertelement %83, %91[%c0_i64 : i64] : vector<8xf32>
                            affine.store %92, %2[((%12 - %arg3) floordiv 16) mod 16, (%4 - %arg5) mod 6, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %93 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %94 = affine.load %2[((%93 - %arg3) floordiv 16) mod 16, (%5 - %arg5) mod 6, (((%93 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %95 = vector.insertelement %84, %94[%c1_i64 : i64] : vector<8xf32>
                            affine.store %95, %2[((%93 - %arg3) floordiv 16) mod 16, (%5 - %arg5) mod 6, (((%93 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %96 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %97 = affine.load %2[((%96 - %arg3) floordiv 16) mod 16, (%6 - %arg5) mod 6, (((%96 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %98 = vector.insertelement %85, %97[%c2_i64 : i64] : vector<8xf32>
                            affine.store %98, %2[((%96 - %arg3) floordiv 16) mod 16, (%6 - %arg5) mod 6, (((%96 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %100 = affine.load %2[((%99 - %arg3) floordiv 16) mod 16, (%7 - %arg5) mod 6, (((%99 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %101 = vector.insertelement %86, %100[%c3_i64 : i64] : vector<8xf32>
                            affine.store %101, %2[((%99 - %arg3) floordiv 16) mod 16, (%7 - %arg5) mod 6, (((%99 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %102 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %103 = affine.load %2[((%102 - %arg3) floordiv 16) mod 16, (%8 - %arg5) mod 6, (((%102 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %104 = vector.insertelement %87, %103[%c4_i64 : i64] : vector<8xf32>
                            affine.store %104, %2[((%102 - %arg3) floordiv 16) mod 16, (%8 - %arg5) mod 6, (((%102 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %105 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %106 = affine.load %2[((%105 - %arg3) floordiv 16) mod 16, (%9 - %arg5) mod 6, (((%105 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %107 = vector.insertelement %88, %106[%c5_i64 : i64] : vector<8xf32>
                            affine.store %107, %2[((%105 - %arg3) floordiv 16) mod 16, (%9 - %arg5) mod 6, (((%105 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %108 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %109 = affine.load %2[((%108 - %arg3) floordiv 16) mod 16, (%10 - %arg5) mod 6, (((%108 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %110 = vector.insertelement %89, %109[%c6_i64 : i64] : vector<8xf32>
                            affine.store %110, %2[((%108 - %arg3) floordiv 16) mod 16, (%10 - %arg5) mod 6, (((%108 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %111 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %112 = affine.load %2[((%111 - %arg3) floordiv 16) mod 16, (%11 - %arg5) mod 6, (((%111 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %113 = vector.insertelement %90, %112[%c7_i64 : i64] : vector<8xf32>
                            affine.store %113, %2[((%111 - %arg3) floordiv 16) mod 16, (%11 - %arg5) mod 6, (((%111 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %114 = affine.load %2[((%12 - %arg3) floordiv 16) mod 16, (%4 - %arg5) mod 6, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %115 = vector.insertelement %83, %114[%c0_i64 : i64] : vector<8xf32>
                            affine.store %115, %2[((%12 - %arg3) floordiv 16) mod 16, (%4 - %arg5) mod 6, (((%12 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %116 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %117 = affine.load %2[((%116 - %arg3) floordiv 16) mod 16, (%5 - %arg5) mod 6, (((%116 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %118 = vector.insertelement %84, %117[%c1_i64 : i64] : vector<8xf32>
                            affine.store %118, %2[((%116 - %arg3) floordiv 16) mod 16, (%5 - %arg5) mod 6, (((%116 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %119 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %120 = affine.load %2[((%119 - %arg3) floordiv 16) mod 16, (%6 - %arg5) mod 6, (((%119 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %121 = vector.insertelement %85, %120[%c2_i64 : i64] : vector<8xf32>
                            affine.store %121, %2[((%119 - %arg3) floordiv 16) mod 16, (%6 - %arg5) mod 6, (((%119 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %122 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %123 = affine.load %2[((%122 - %arg3) floordiv 16) mod 16, (%7 - %arg5) mod 6, (((%122 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %124 = vector.insertelement %86, %123[%c3_i64 : i64] : vector<8xf32>
                            affine.store %124, %2[((%122 - %arg3) floordiv 16) mod 16, (%7 - %arg5) mod 6, (((%122 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %125 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %126 = affine.load %2[((%125 - %arg3) floordiv 16) mod 16, (%8 - %arg5) mod 6, (((%125 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %127 = vector.insertelement %87, %126[%c4_i64 : i64] : vector<8xf32>
                            affine.store %127, %2[((%125 - %arg3) floordiv 16) mod 16, (%8 - %arg5) mod 6, (((%125 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %128 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %129 = affine.load %2[((%128 - %arg3) floordiv 16) mod 16, (%9 - %arg5) mod 6, (((%128 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %130 = vector.insertelement %88, %129[%c5_i64 : i64] : vector<8xf32>
                            affine.store %130, %2[((%128 - %arg3) floordiv 16) mod 16, (%9 - %arg5) mod 6, (((%128 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %131 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %132 = affine.load %2[((%131 - %arg3) floordiv 16) mod 16, (%10 - %arg5) mod 6, (((%131 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %133 = vector.insertelement %89, %132[%c6_i64 : i64] : vector<8xf32>
                            affine.store %133, %2[((%131 - %arg3) floordiv 16) mod 16, (%10 - %arg5) mod 6, (((%131 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %134 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>(%arg3, %arg6, %arg11, %arg12)
                            %135 = affine.load %2[((%134 - %arg3) floordiv 16) mod 16, (%11 - %arg5) mod 6, (((%134 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %136 = vector.insertelement %90, %135[%c7_i64 : i64] : vector<8xf32>
                            affine.store %136, %2[((%134 - %arg3) floordiv 16) mod 16, (%11 - %arg5) mod 6, (((%134 - %arg3) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                          } {begin = 0 : i64, end = 8 : i64, index = #accln<"index{j_i_i_i,16}">, scheduledIndex = #accln<"index{j_i_i_i,16}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 1, 1]}
                        } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{j_i_i_o,15}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 8, 1]}
                      } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{i_i_i,12}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 16, 1]}
                    } {begin = 0 : i64, end = 4 : i64, index = #accln<"index{k_i_i,10}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 16, 1]}
                  } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{i_i_o,11}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 16, 4]}
                } {begin = 0 : i64, end = 128 : i64, index = #accln<"index{k_i_o,9}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 16, 4]}
              } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{j_i_o,13}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 16, 128]}
              "accv.lambda"() ( {
                affine.for %arg6 = 0 to 1 {
                  affine.for %arg7 = 0 to 256 step 128 {
                    affine.if affine_set<() : (0 == 0)>() {
                      "accv.lambda"() ( {
                        affine.for %arg8 = 0 to 1 {
                          affine.for %arg9 = 0 to 16 {
                            %4 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg6, %arg8)
                            %5 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2 * 8)>(%arg3, %arg7, %arg9)
                            %6 = vector.transfer_read %arg2[%4, %5], %cst_0 {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
                            %7 = affine.load %2[((%arg7 + %arg9 * 8) floordiv 16) mod 16, (%arg6 + %arg8) mod 6, (((%arg7 + %arg9 * 8) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %8 = addf %6, %7 : vector<8xf32>
                            store %8, %1[%arg8, %arg9] : memref<1x16xvector<8xf32>>
                          } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{i_o,7}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{k_i,6}">, #accln<"index{i_o,7}">], subdomainSize = [1, 1]}
                        } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{k_i,6}">, subdomainIndexOrder = [#accln<"index{k_i,6}">, #accln<"index{i_o,7}">], subdomainSize = [1, 16]}
                        accv.return
                      }) {exec_target = 0 : i64, sym_name = "NestFunction_11", type = () -> ()} : () -> ()
                      "accv.lambda"() ( {
                        affine.for %arg8 = 0 to 1 {
                          affine.for %arg9 = 0 to 16 {
                            %4 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg6, %arg8)
                            %5 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2 * 8)>(%arg3, %arg7, %arg9)
                            %6 = load %1[%arg8, %arg9] : memref<1x16xvector<8xf32>>
                            vector.transfer_write %6, %arg2[%4, %5] {masked = [false]} : vector<8xf32>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
                          } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{k_i_o,9}">, subdomainIndexOrder = [#accln<"index{i_i,8}">, #accln<"index{k_i_o,9}">], subdomainSize = [1, 1]}
                        } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{i_i,8}">, subdomainIndexOrder = [#accln<"index{i_i,8}">, #accln<"index{k_i_o,9}">], subdomainSize = [1, 16]}
                        accv.return
                      }) {exec_target = 0 : i64, sym_name = "NestFunction_10", type = () -> ()} : () -> ()
                    } else {
                      "accv.lambda"() ( {
                        affine.for %arg8 = 0 to 1 {
                          affine.for %arg9 = 0 to 16 {
                            %4 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg6, %arg8)
                            %5 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2 * 8)>(%arg3, %arg7, %arg9)
                            %6 = vector.transfer_read %arg2[%4, %5], %cst_0 : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
                            %7 = affine.load %2[((%arg7 + %arg9 * 8) floordiv 16) mod 16, (%arg6 + %arg8) mod 6, (((%arg7 + %arg9 * 8) mod 16) floordiv 8) mod 2] : memref<16x6x2xvector<8xf32>>
                            %8 = addf %6, %7 : vector<8xf32>
                            store %8, %1[%arg8, %arg9] : memref<1x16xvector<8xf32>>
                          } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{i_i_o,11}">, accv_unrolled, subdomainIndexOrder = [#accln<"index{k_i_i,10}">, #accln<"index{i_i_o,11}">], subdomainSize = [1, 1]}
                        } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{k_i_i,10}">, subdomainIndexOrder = [#accln<"index{k_i_i,10}">, #accln<"index{i_i_o,11}">], subdomainSize = [1, 16]}
                        accv.return
                      }) {exec_target = 0 : i64, sym_name = "NestFunction_9", type = () -> ()} : () -> ()
                      "accv.lambda"() ( {
                        affine.for %arg8 = 0 to 1 {
                          affine.for %arg9 = 0 to 16 {
                            %4 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg5, %arg6, %arg8)
                            %5 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2 * 8)>(%arg3, %arg7, %arg9)
                            %6 = load %1[%arg8, %arg9] : memref<1x16xvector<8xf32>>
                            vector.transfer_write %6, %arg2[%4, %5] : vector<8xf32>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
                          } {begin = 0 : i64, end = 16 : i64, index = #accln<"index{j_i_o,13}">, subdomainIndexOrder = [#accln<"index{i_i_i,12}">, #accln<"index{j_i_o,13}">], subdomainSize = [1, 1]}
                        } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{i_i_i,12}">, subdomainIndexOrder = [#accln<"index{i_i_i,12}">, #accln<"index{j_i_o,13}">], subdomainSize = [1, 16]}
                        accv.return
                      }) {exec_target = 0 : i64, sym_name = "NestFunction_8", type = () -> ()} : () -> ()
                    }
                  } {begin = 0 : i64, end = 256 : i64, index = #accln<"index{j_i,4}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [1, 128]}
                } {begin = 0 : i64, end = 1 : i64, index = #accln<"index{k,2}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [1, 256]}
                accv.return
              }) {exec_target = 0 : i64, sym_name = "NestFunction_5", type = () -> ()} : () -> ()
            } {begin = 0 : i64, end = 784 : i64, index = #accln<"index{i_o,7}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [1, 256, 128]}
          } {begin = 0 : i64, end = 128 : i64, index = #accln<"index{k_o,5}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [784, 256, 128]}
        } {begin = 0 : i64, end = 512 : i64, index = #accln<"index{j_o,3}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">, #accln<"index{k,2}">], subdomainSize = [784, 256, 128]}
        accv.return
      }) {exec_target = 0 : i64, sym_name = "NestFunction_0", type = () -> ()} : () -> ()
      accv.return
    }
    accv.func @optimized_matmul_py_4a6286d9(%arg0: memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, %arg1: memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, %arg2: memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) attributes {exec_target = 0 : i64, accv.base_name = "optimized_matmul_py", accv.emit_header_decl, accv.emit_raw_pointer_api} {
      accv.launch_func @optimized_matmul_py_4a6286d9_impl_17630232307017152746(%arg0, %arg1, %arg2) {exec_target = 0 : i64} : (memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) -> ()
      accv.return
    }
  }
}
