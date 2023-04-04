// RUN: acc-opt --verify-each=false --acc-vectorize -split-input-file %s | FileCheck %s

module @test_accera_vectorization attributes {accv.target_device_features = "-avx512pf,-tsxldtrk,+cx16,+sahf,-tbm,-avx512ifma,-sha,+crc32,-fma4,-vpclmulqdq,-prfchw,+bmi2,-cldemote,+fsgsbase,-ptwrite,-amx-tile,-uintr,-gfni,+popcnt,-widekl,+aes,-avx512bitalg,-movdiri,-xsaves,-avx512er,-avxvnni,-avx512fp16,-avx512vnni,-amx-bf16,-avx512vpopcntdq,-pconfig,-clwb,-avx512f,-xsavec,-clzero,-pku,+mmx,-lwp,-rdpid,-xop,-rdseed,-waitpkg,-kl,-movdir64b,-sse4a,-avx512bw,-clflushopt,+xsave,-avx512vbmi2,+64bit,-avx512vl,-serialize,-hreset,+invpcid,-avx512cd,+avx,-vaes,-avx512bf16,+cx8,+fma,-rtm,+bmi,-enqcmd,+rdrnd,-mwaitx,+sse4.1,+sse4.2,+avx2,+fxsr,-wbnoinvd,+sse,+lzcnt,+pclmul,-prefetchwt1,+f16c,+ssse3,-sgx,-shstk,+cmov,-avx512vbmi,-amx-int8,+movbe,-avx512vp2intersect,+xsaveopt,-avx512dq,+sse2,-adx,+sse"} {
    accv.module "test_accera_vectorization"  {

        // Single-op cases:
        // TODO : implement test cases for these
        // mlir::memref::AllocaOp
        // mlir::arith::ConstantOp
        // mlir::memref::LoadOp sequential
        // mlir::memref::LoadOp non-sequential
        // mlir::memref::StoreOp sequential
        // mlir::memref::StoreOp non-sequential
        // mlir::affine::AffineLoadOp sequential
        // mlir::affine::AffineLoadOp non-sequential
        // mlir::affine::AffineStoreOp sequential
        // mlir::affine::AffineStoreOp non-sequential
        // mlir::SelectOp
        // mlir::arith::ShLIOp
        // mlir::arith::FPToSIOp
        // mlir::arith::ExtSIOp
        // mlir::math::AbsOp
        // mlir::math::ExpOp
        // value::CastOp
        // value::RoundOp
        // value::BitCastOp
        // value::BinOp
        // value::CmpOp
        // value::ReferenceGlobalOp

        // Special cases:
        // TODO : implement test cases for these
        // horizontal reduction
        // multi-loop sequential cast
        // two-row interleaved pack
        // vpmaddwd avx 2
        // vpmaddwd avx 512
        // masked load
        // two-row interleaved masked load and pack


        // CHECK-LABEL builtin.func nested @test_view_split_dim_interleaved_pack
        builtin.func nested @test_view_split_dim_interleaved_pack(%arg0: memref<1885x256xui8>, %arg1: memref<483840xui8>) attributes {accv.dyn_arg_size_refs = [[-1, -1], [-1]], accv.usages = [1 : i8, 0 : i8], args_name = ["", ""], args_size = ["1885*256", "483840"], args_symbol = ["args_symbol_name_0", "args_symbol_name_1"], exec_target = 0 : i64} {
            %c1024 = arith.constant 1024 : index
            %c1 = arith.constant 1 : index
            %c482816 = arith.constant 482816 : index
            %c98304 = arith.constant 98304 : index
            %c2 = arith.constant 2 : index
            %c16 = arith.constant 16 : index
            %c192 = arith.constant 192 : index
            affine.for %arg2 = 0 to 1536 step 384 {
                %0 = "accv.view"(%arg1, %c482816, %c1024, %c1) {operand_segment_sizes = dense<1> : vector<4xi32>} : (memref<483840xui8>, index, index, index) -> memref<482816xui8, affine_map<(d0) -> (d0 + 1024)>>
                %1 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg2)
                %2 = "accv.view"(%0, %c98304, %1, %c1) {operand_segment_sizes = dense<1> : vector<4xi32>} : (memref<482816xui8, affine_map<(d0) -> (d0 + 1024)>>, index, index, index) -> memref<98304xui8, affine_map<(d0)[s0] -> (d0 + s0 + 1024)>>
                %3 = "accv.split_dim"(%2, %c2) {dim = 0 : i64} : (memref<98304xui8, affine_map<(d0)[s0] -> (d0 + s0 + 1024)>>, index) -> memref<49152x2xui8, affine_map<(d0, d1)[s0] -> (d0 * 2 + d1 + s0 + 1024)>>
                %4 = "accv.split_dim"(%3, %c16) {dim = 0 : i64} : (memref<49152x2xui8, affine_map<(d0, d1)[s0] -> (d0 * 2 + d1 + s0 + 1024)>>, index) -> memref<3072x16x2xui8, affine_map<(d0, d1, d2)[s0] -> ((d0 * 16 + d1) * 2 + d2 + s0 + 1024)>>
                %5 = "accv.split_dim"(%4, %c192) {dim = 0 : i64} : (memref<3072x16x2xui8, affine_map<(d0, d1, d2)[s0] -> ((d0 * 16 + d1) * 2 + d2 + s0 + 1024)>>, index) -> memref<16x192x16x2xui8, affine_map<(d0, d1, d2, d3)[s0] -> (((d0 * 192 + d1) * 16 + d2) * 2 + d3 + s0 + 1024)>>
                // CHECK: affine.for %arg3 = 0 to 256 step 16 {
                affine.for %arg3 = 0 to 256 step 16 {
                    // CHECK-NEXT: affine.for %arg4 = 0 to 384 step 2 {
                    affine.for %arg4 = 0 to 384 step 2 {
                        affine.for %arg5 = 0 to 16 {
                            affine.for %arg6 = 0 to 2 {
                                %8 = affine.load %arg0[%arg6 + %arg4 + symbol(%arg2), %arg5 + %arg3] : memref<1885x256xui8>
                                affine.store %8, %5[symbol(%arg3) floordiv 16, symbol(%arg4) floordiv 2, %arg5, %arg6] : memref<16x192x16x2xui8, affine_map<(d0, d1, d2, d3)[s0] -> (((d0 * 192 + d1) * 16 + d2) * 2 + d3 + s0 + 1024)>>
                            } {beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (2)>, index = #accln<"index{j_i,245}">, kernels = ["_cache_fill"], operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, subdomainIndexOrder = [#accln<"index{i,240}">, #accln<"index{j,241}">], subdomainSize = [16, 2]}
                        } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">, beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (16)>, index = #accln<"index{i_i,243}">, operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, scheduledIndex = #accln<"index{i_i,243}">, subdomainIndexOrder = [#accln<"index{i,240}">, #accln<"index{j,241}">], subdomainSize = [16, 2]}
                        // CHECK-NEXT: %6 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [482560], strides: [1] : memref<1885x256xui8> to memref<482560xui8>
                        // CHECK-NEXT: %7 = affine.apply #map6(%arg4, %c0, %arg3)[%arg2]
                        // CHECK-NEXT: %8 = vector.load %6[%7] : memref<482560xui8>, vector<16xui8>
                        // CHECK-NEXT: %9 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [482560], strides: [1] : memref<1885x256xui8> to memref<482560xui8>
                        // CHECK-NEXT: %10 = affine.apply #map7(%arg4, %c0, %arg3)[%arg2]
                        // CHECK-NEXT: %11 = vector.load %9[%10] : memref<482560xui8>, vector<16xui8>
                        // CHECK-NEXT: %12 = vector.shuffle %8, %11 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31] : vector<16xui8>, vector<16xui8>
                        // CHECK-NEXT: %13 = memref.reinterpret_cast %5 to offset: [0], sizes: [98304], strides: [1] : memref<16x192x16x2xui8, #map5> to memref<98304xui8>
                        // CHECK-NEXT: %14 = affine.apply #map8(%c0, %c0, %arg2)[%arg3, %arg4]
                        // CHECK-NEXT: vector.store %12, %13[%14] : memref<98304xui8>, vector<32xui8>
                    } {beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (384)>, index = #accln<"index{i_i_o,257}">, operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, subdomainIndexOrder = [#accln<"index{i,254}">], subdomainSize = [-1]}
                } {beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (256)>, index = #accln<"index{i_i_o,262}">, operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, subdomainIndexOrder = [#accln<"index{i,259}">], subdomainSize = [-1]}
            } {beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (1536)>, index = #accln<"index{i_o,268}">, operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, subdomainIndexOrder = [#accln<"index{i,266}">, #accln<"index{j,267}">], subdomainSize = [1885, 256]}
            return
        }

        // CHECK-LABEL builtin.func nested @test_view_split_dim_interleaved_pack
        builtin.func nested @test_int16_to_int32_horizontal_vector_add(%arg0: memref<256x16xi16>, %arg1: memref<256xi32>) attributes {accv.dyn_arg_size_refs = [[-1, -1], [-1]], accv.usages = [1 : i8, 1 : i8], args_name = ["", ""], args_size = ["256*16", "256"], args_symbol = ["args_symbol_name_0", "args_symbol_name_1"], exec_target = 0 : i64} {
            // CHECK: affine.for %arg2 = 0 to 256 step 4 {
            affine.for %arg2 = 0 to 256 step 4 {
                // %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096], strides: [1] : memref<256x16xi16> to memref<4096xi16>
                // %1 = affine.apply affine_map<(d0, d1, d2) -> ((d1 + d2) * 16 + d0)>(%c0, %arg2, %c0)
                // %2 = vector.load %0[%1] : memref<4096xi16>, vector<16xi16>
                // %3 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096], strides: [1] : memref<256x16xi16> to memref<4096xi16>
                // %4 = affine.apply affine_map<(d0, d1, d2) -> ((d1 + d2) * 16 + d0)>(%c0, %arg2, %c1)
                // %5 = vector.load %3[%4] : memref<4096xi16>, vector<16xi16>
                // %6 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096], strides: [1] : memref<256x16xi16> to memref<4096xi16>
                // %7 = affine.apply affine_map<(d0, d1, d2) -> ((d1 + d2) * 16 + d0)>(%c0, %arg2, %c2)
                // %8 = vector.load %6[%7] : memref<4096xi16>, vector<16xi16>
                // %9 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096], strides: [1] : memref<256x16xi16> to memref<4096xi16>
                // %10 = affine.apply affine_map<(d0, d1, d2) -> ((d1 + d2) * 16 + d0)>(%c0, %arg2, %c3)
                // %11 = vector.load %9[%10] : memref<4096xi16>, vector<16xi16>
                // %12 = "accv.vpmaddwd"(%2, %cst) : (vector<16xi16>, vector<16xi16>) -> vector<8xi32>
                // %13 = "accv.vpmaddwd"(%5, %cst) : (vector<16xi16>, vector<16xi16>) -> vector<8xi32>
                // %14 = "accv.vpmaddwd"(%8, %cst) : (vector<16xi16>, vector<16xi16>) -> vector<8xi32>
                // %15 = "accv.vpmaddwd"(%11, %cst) : (vector<16xi16>, vector<16xi16>) -> vector<8xi32>
                // %16 = "accv.vhadd"(%12, %13) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
                // %17 = "accv.vhadd"(%14, %15) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
                // %18 = "accv.vhadd"(%16, %17) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
                // %19 = vector.shuffle %18, %18 [0, 1, 2, 3] : vector<8xi32>, vector<8xi32>
                // %20 = vector.shuffle %18, %18 [4, 5, 6, 7] : vector<8xi32>, vector<8xi32>
                // %21 = "accv.bin_op"(%19, %20) {predicate = 0 : i64} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
                // %22 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<256xi32> to memref<256xi32>
                // %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg2, %c0)
                // %24 = vector.load %22[%23] : memref<256xi32>, vector<4xi32>
                // %25 = "accv.bin_op"(%24, %21) {predicate = 0 : i64} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
                // %26 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<256xi32> to memref<256xi32>
                // %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg2, %c0)
                // vector.store %25, %26[%27] : memref<256xi32>, vector<4xi32>
                affine.for %arg3 = 0 to 4 {
                    affine.for %arg4 = 0 to 16 {
                        %0 = affine.load %arg0[%arg2 + %arg3, %arg4] : memref<256x16xi16>
                        %1 = "accv.cast"(%0) : (i16) -> i32
                        %2 = affine.load %arg1[%arg2 + %arg3] : memref<256xi32>
                        %3 = "accv.bin_op"(%2, %1) {predicate = 0 : i64} : (i32, i32) -> i32
                        affine.store %3, %arg1[%arg2 + %arg3] : memref<256xi32>
                    } {beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (16)>, index = #accln<"index{j,1}">, kernels = ["_"], operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [256, 16]}
                } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">, beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (4)>, index = #accln<"index{i_i,3}">, operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, scheduledIndex = #accln<"index{i_i,3}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [256, 16]}
            } {beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (256)>, index = #accln<"index{i_o,2}">, operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [256, 16]}
            return
        }

        // CHECK-LABEL builtin.func nested @test_int32_horizontal_vector_add_simple
        builtin.func nested @test_int32_horizontal_vector_add_simple(%arg0: memref<256x8xi32>, %arg1: memref<256xi32>) attributes {accv.dyn_arg_size_refs = [[-1, -1], [-1]], accv.usages = [1 : i8, 1 : i8], args_name = ["", ""], args_size = ["256*8", "256"], args_symbol = ["args_symbol_name_0", "args_symbol_name_1"], exec_target = 0 : i64} {
            // CHECK: affine.for %arg2 = 0 to 256 step 4 {
            affine.for %arg2 = 0 to 256 step 4 {
                // %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2048], strides: [1] : memref<256x8xi32> to memref<2048xi32>
                // %1 = affine.apply affine_map<(d0, d1, d2) -> ((d1 + d2) * 8 + d0)>(%c0, %arg2, %c0)
                // %2 = vector.load %0[%1] : memref<2048xi32>, vector<8xi32>
                // %3 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2048], strides: [1] : memref<256x8xi32> to memref<2048xi32>
                // %4 = affine.apply affine_map<(d0, d1, d2) -> ((d1 + d2) * 8 + d0)>(%c0, %arg2, %c1)
                // %5 = vector.load %3[%4] : memref<2048xi32>, vector<8xi32>
                // %6 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2048], strides: [1] : memref<256x8xi32> to memref<2048xi32>
                // %7 = affine.apply affine_map<(d0, d1, d2) -> ((d1 + d2) * 8 + d0)>(%c0, %arg2, %c2)
                // %8 = vector.load %6[%7] : memref<2048xi32>, vector<8xi32>
                // %9 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2048], strides: [1] : memref<256x8xi32> to memref<2048xi32>
                // %10 = affine.apply affine_map<(d0, d1, d2) -> ((d1 + d2) * 8 + d0)>(%c0, %arg2, %c3)
                // %11 = vector.load %9[%10] : memref<2048xi32>, vector<8xi32>
                // %12 = "accv.vhadd"(%2, %5) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
                // %13 = "accv.vhadd"(%8, %11) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
                // %14 = "accv.vhadd"(%12, %13) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
                // %15 = vector.shuffle %14, %14 [0, 1, 2, 3] : vector<8xi32>, vector<8xi32>
                // %16 = vector.shuffle %14, %14 [4, 5, 6, 7] : vector<8xi32>, vector<8xi32>
                // %17 = "accv.bin_op"(%15, %16) {predicate = 0 : i64} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
                // %18 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<256xi32> to memref<256xi32>
                // %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg2, %c0)
                // %20 = vector.load %18[%19] : memref<256xi32>, vector<4xi32>
                // %21 = "accv.bin_op"(%20, %17) {predicate = 0 : i64} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
                // %22 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<256xi32> to memref<256xi32>
                // %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg2, %c0)
                // vector.store %21, %22[%23] : memref<256xi32>, vector<4xi32>
                affine.for %arg3 = 0 to 4 {
                    affine.for %arg4 = 0 to 8 {
                        %0 = affine.load %arg1[%arg2 + %arg3] : memref<256xi32>
                        %1 = affine.load %arg0[%arg2 + %arg3, %arg4] : memref<256x8xi32>
                        %2 = "accv.bin_op"(%0, %1) {predicate = 0 : i64} : (i32, i32) -> i32
                        affine.store %2, %arg1[%arg2 + %arg3] : memref<256xi32>
                    } {beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (8)>, index = #accln<"index{j,1}">, kernels = ["_"], operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [256, 8]}
                } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">, beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (4)>, index = #accln<"index{i_i,3}">, operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, scheduledIndex = #accln<"index{i_i,3}">, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [256, 8]}
            } {beginMap = affine_map<() -> (0)>, endMap = affine_map<() -> (256)>, index = #accln<"index{i_o,2}">, operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>, subdomainIndexOrder = [#accln<"index{i,0}">, #accln<"index{j,1}">], subdomainSize = [256, 8]}
            return
        }
    }
}

// -----

module @test_transpose_8x4 attributes {accv.target_device_features = "+avx2", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "test_transpose_8x4" {

    // CHECK-LABEL builtin.func nested @test_transpose_8x4_423849238228f332_impl_17576985214141312005(%arg0: memref<8x4xf32>, %arg1: memref<4x8xf32>) {
    // CHECK-NEXT  %c0 = arith.constant 0 : index
    // CHECK-NEXT  %c4 = arith.constant 4 : index
    // CHECK-NEXT  %c8 = arith.constant 8 : index
    // CHECK-NEXT  %c12 = arith.constant 12 : index
    // CHECK-NEXT  %c16 = arith.constant 16 : index
    // CHECK-NEXT  %c20 = arith.constant 20 : index
    // CHECK-NEXT  %c24 = arith.constant 24 : index
    // CHECK-NEXT  %c28 = arith.constant 28 : index
    // CHECK-NEXT  %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<8x4xf32> to memref<32xf32>
    // CHECK-NEXT  %1 = vector.load %0[%c0] : memref<32xf32>, vector<4xf32>
    // CHECK-NEXT  %2 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<8x4xf32> to memref<32xf32>
    // CHECK-NEXT  %3 = vector.load %2[%c4] : memref<32xf32>, vector<4xf32>
    // CHECK-NEXT  %4 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<8x4xf32> to memref<32xf32>
    // CHECK-NEXT  %5 = vector.load %4[%c8] : memref<32xf32>, vector<4xf32>
    // CHECK-NEXT  %6 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<8x4xf32> to memref<32xf32>
    // CHECK-NEXT  %7 = vector.load %6[%c12] : memref<32xf32>, vector<4xf32>
    // CHECK-NEXT  %8 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<8x4xf32> to memref<32xf32>
    // CHECK-NEXT  %9 = vector.load %8[%c16] : memref<32xf32>, vector<4xf32>
    // CHECK-NEXT  %10 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<8x4xf32> to memref<32xf32>
    // CHECK-NEXT  %11 = vector.load %10[%c20] : memref<32xf32>, vector<4xf32>
    // CHECK-NEXT  %12 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<8x4xf32> to memref<32xf32>
    // CHECK-NEXT  %13 = vector.load %12[%c24] : memref<32xf32>, vector<4xf32>
    // CHECK-NEXT  %14 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<8x4xf32> to memref<32xf32>
    // CHECK-NEXT  %15 = vector.load %14[%c28] : memref<32xf32>, vector<4xf32>
    // CHECK-NEXT  %16 = vector.shuffle %1, %9 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
    // CHECK-NEXT  %17 = vector.shuffle %3, %11 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
    // CHECK-NEXT  %18 = vector.shuffle %5, %13 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
    // CHECK-NEXT  %19 = vector.shuffle %7, %15 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
    // CHECK-NEXT  %20 = vector.shuffle %16, %17 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT  %21 = vector.shuffle %16, %17 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT  %22 = vector.shuffle %18, %19 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT  %23 = vector.shuffle %18, %19 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT  %24 = vector.shuffle %20, %22 [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT  %25 = vector.shuffle %20, %22 [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT  %26 = vector.shuffle %21, %23 [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT  %27 = vector.shuffle %21, %23 [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT  %28 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32], strides: [1] : memref<4x8xf32> to memref<32xf32>
    // CHECK-NEXT  vector.store %24, %28[%c0] : memref<32xf32>, vector<8xf32>
    // CHECK-NEXT  %29 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32], strides: [1] : memref<4x8xf32> to memref<32xf32>
    // CHECK-NEXT  vector.store %25, %29[%c8] : memref<32xf32>, vector<8xf32>
    // CHECK-NEXT  %30 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32], strides: [1] : memref<4x8xf32> to memref<32xf32>
    // CHECK-NEXT  vector.store %26, %30[%c16] : memref<32xf32>, vector<8xf32>
    // CHECK-NEXT  %31 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32], strides: [1] : memref<4x8xf32> to memref<32xf32>
    // CHECK-NEXT  vector.store %27, %31[%c24] : memref<32xf32>, vector<8xf32>
    // CHECK-NEXT  return

    builtin.func nested @test_transpose_8x4_423849238228f332_impl_17576985214141312005(%arg0: memref<8x4xf32>, %arg1: memref<4x8xf32>) {
      affine.for %arg2 = 0 to 8 {
        affine.for %arg3 = 0 to 4 {
          %0 = affine.load %arg0[%arg2, %arg3] : memref<8x4xf32>
          affine.store %0, %arg1[%arg3, %arg2] : memref<4x8xf32>
        } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      return
    }
  }
}

// -----

module @test_transpose_16x4 attributes {accv.target_device_features = "+avx2", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "test_transpose_16x4" {

    // CHECK-LABEL builtin.func nested @test_transpose_16x4_d3ea7863380b434f_impl_10155054494031908713(%arg0: memref<16x4xf32>, %arg1: memref<4x16xf32>) {
    // CHECK-NEXT  %c0 = arith.constant 0 : index
    // CHECK-NEXT  %c1 = arith.constant 1 : index
    // CHECK-NEXT  %c2 = arith.constant 2 : index
    // CHECK-NEXT  %c3 = arith.constant 3 : index
    // CHECK-NEXT  %c4 = arith.constant 4 : index
    // CHECK-NEXT  %c5 = arith.constant 5 : index
    // CHECK-NEXT  %c6 = arith.constant 6 : index
    // CHECK-NEXT  %c7 = arith.constant 7 : index
    // CHECK-NEXT  affine.for %arg2 = 0 to 16 step 8 {
    // CHECK-NEXT    %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<16x4xf32> to memref<64xf32>
    // CHECK-NEXT    %1 = affine.apply #map0(%arg2, %c0, %c0)
    // CHECK-NEXT    %2 = vector.load %0[%1] : memref<64xf32>, vector<4xf32>
    // CHECK-NEXT    %3 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<16x4xf32> to memref<64xf32>
    // CHECK-NEXT    %4 = affine.apply #map0(%arg2, %c1, %c0)
    // CHECK-NEXT    %5 = vector.load %3[%4] : memref<64xf32>, vector<4xf32>
    // CHECK-NEXT    %6 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<16x4xf32> to memref<64xf32>
    // CHECK-NEXT    %7 = affine.apply #map0(%arg2, %c2, %c0)
    // CHECK-NEXT    %8 = vector.load %6[%7] : memref<64xf32>, vector<4xf32>
    // CHECK-NEXT    %9 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<16x4xf32> to memref<64xf32>
    // CHECK-NEXT    %10 = affine.apply #map0(%arg2, %c3, %c0)
    // CHECK-NEXT    %11 = vector.load %9[%10] : memref<64xf32>, vector<4xf32>
    // CHECK-NEXT    %12 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<16x4xf32> to memref<64xf32>
    // CHECK-NEXT    %13 = affine.apply #map0(%arg2, %c4, %c0)
    // CHECK-NEXT    %14 = vector.load %12[%13] : memref<64xf32>, vector<4xf32>
    // CHECK-NEXT    %15 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<16x4xf32> to memref<64xf32>
    // CHECK-NEXT    %16 = affine.apply #map0(%arg2, %c5, %c0)
    // CHECK-NEXT    %17 = vector.load %15[%16] : memref<64xf32>, vector<4xf32>
    // CHECK-NEXT    %18 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<16x4xf32> to memref<64xf32>
    // CHECK-NEXT    %19 = affine.apply #map0(%arg2, %c6, %c0)
    // CHECK-NEXT    %20 = vector.load %18[%19] : memref<64xf32>, vector<4xf32>
    // CHECK-NEXT    %21 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<16x4xf32> to memref<64xf32>
    // CHECK-NEXT    %22 = affine.apply #map0(%arg2, %c7, %c0)
    // CHECK-NEXT    %23 = vector.load %21[%22] : memref<64xf32>, vector<4xf32>
    // CHECK-NEXT    %24 = vector.shuffle %2, %14 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
    // CHECK-NEXT    %25 = vector.shuffle %5, %17 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
    // CHECK-NEXT    %26 = vector.shuffle %8, %20 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
    // CHECK-NEXT    %27 = vector.shuffle %11, %23 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
    // CHECK-NEXT    %28 = vector.shuffle %24, %25 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT    %29 = vector.shuffle %24, %25 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT    %30 = vector.shuffle %26, %27 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT    %31 = vector.shuffle %26, %27 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT    %32 = vector.shuffle %28, %30 [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT    %33 = vector.shuffle %28, %30 [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT    %34 = vector.shuffle %29, %31 [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT    %35 = vector.shuffle %29, %31 [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
    // CHECK-NEXT    %36 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64], strides: [1] : memref<4x16xf32> to memref<64xf32>
    // CHECK-NEXT    %37 = affine.apply #map1(%c0, %arg2, %c0)
    // CHECK-NEXT    vector.store %32, %36[%37] : memref<64xf32>, vector<8xf32>
    // CHECK-NEXT    %38 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64], strides: [1] : memref<4x16xf32> to memref<64xf32>
    // CHECK-NEXT    %39 = affine.apply #map1(%c1, %arg2, %c0)
    // CHECK-NEXT    vector.store %33, %38[%39] : memref<64xf32>, vector<8xf32>
    // CHECK-NEXT    %40 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64], strides: [1] : memref<4x16xf32> to memref<64xf32>
    // CHECK-NEXT    %41 = affine.apply #map1(%c2, %arg2, %c0)
    // CHECK-NEXT    vector.store %34, %40[%41] : memref<64xf32>, vector<8xf32>
    // CHECK-NEXT    %42 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64], strides: [1] : memref<4x16xf32> to memref<64xf32>
    // CHECK-NEXT    %43 = affine.apply #map1(%c3, %arg2, %c0)
    // CHECK-NEXT    vector.store %35, %42[%43] : memref<64xf32>, vector<8xf32>
    // CHECK-NEXT  }
    // CHECK-NEXT  return

    builtin.func nested @test_transpose_16x4_d3ea7863380b434f_impl_10155054494031908713(%arg0: memref<16x4xf32>, %arg1: memref<4x16xf32>) {
      affine.for %arg2 = 0 to 16 step 8 {
        affine.for %arg3 = 0 to 8 {
          affine.for %arg4 = 0 to 4 {
            %0 = affine.load %arg0[%arg2 + %arg3, %arg4] : memref<16x4xf32>
            affine.store %0, %arg1[%arg4, %arg2 + %arg3] : memref<4x16xf32>
          } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
        } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      }
      return
    }
  }
}

// -----

module @test_vectorized_masked_buffer_fill attributes {accv.target_device_features = "+avx2", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "test_vectorized_masked_buffer_fill" {

// CHECK-LABEL    builtin.func nested @test_vectorized_masked_buffer_fill_11c7743a08b6567e_impl_8352600288936674199(%arg0: memref<5xi32>, %arg1: memref<8xi32>) {
// CHECK-NEXT      %cst = arith.constant dense<false> : vector<8xi1>
// CHECK-NEXT      %c0 = arith.constant 0 : index
// CHECK-NEXT      %c1 = arith.constant 1 : index
// CHECK-NEXT      %c2 = arith.constant 2 : index
// CHECK-NEXT      %c3 = arith.constant 3 : index
// CHECK-NEXT      %c4 = arith.constant 4 : index
// CHECK-NEXT      %c6 = arith.constant 6 : index
// CHECK-NEXT      %c7 = arith.constant 7 : index
// CHECK-NEXT      %c5 = arith.constant 5 : index
// CHECK-NEXT      %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT      %0 = "accv.cmp"(%c0, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %1 = vector.insertelement %0, %cst[%c0 : index] : vector<8xi1>
// CHECK-NEXT      %2 = "accv.cmp"(%c1, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %3 = vector.insertelement %2, %1[%c1 : index] : vector<8xi1>
// CHECK-NEXT      %4 = "accv.cmp"(%c2, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %5 = vector.insertelement %4, %3[%c2 : index] : vector<8xi1>
// CHECK-NEXT      %6 = "accv.cmp"(%c3, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %7 = vector.insertelement %6, %5[%c3 : index] : vector<8xi1>
// CHECK-NEXT      %8 = "accv.cmp"(%c4, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %9 = vector.insertelement %8, %7[%c4 : index] : vector<8xi1>
// CHECK-NEXT      %10 = "accv.cmp"(%c5, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %11 = vector.insertelement %10, %9[%c5 : index] : vector<8xi1>
// CHECK-NEXT      %12 = "accv.cmp"(%c6, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %13 = vector.insertelement %12, %11[%c6 : index] : vector<8xi1>
// CHECK-NEXT      %14 = "accv.cmp"(%c7, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %15 = vector.insertelement %14, %13[%c7 : index] : vector<8xi1>
// CHECK-NEXT      %16 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [5], strides: [1] : memref<5xi32> to memref<5xi32>
// CHECK-NEXT      %17 = vector.transfer_read %16[%c0], %c0_i32, %15 {in_bounds = [true]} : memref<5xi32>, vector<8xi32>
// CHECK-NEXT      %18 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8], strides: [1] : memref<8xi32> to memref<8xi32>
// CHECK-NEXT      vector.store %17, %18[%c0] : memref<8xi32>, vector<8xi32>
// CHECK-NEXT      return
// CHECK-NEXT    }

    builtin.func nested @test_vectorized_masked_buffer_fill_11c7743a08b6567e_impl_8352600288936674199(%arg0: memref<5xi32>, %arg1: memref<8xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c5 = arith.constant 5 : index
      affine.for %arg2 = 0 to 8 {
        %0 = "accv.cmp"(%arg2, %c5) {predicate = 2 : i64} : (index, index) -> i1
        scf.if %0 {
          %1 = affine.load %arg0[%arg2] : memref<5xi32>
          affine.store %1, %arg1[%arg2] : memref<8xi32>
        } else {
          affine.store %c0_i32, %arg1[%arg2] : memref<8xi32>
        }
      } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      return
    }
  }
}


// -----

module @test_vectorized_masked_store attributes {accv.target_device_features = "+avx2", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "test_vectorized_masked_store" {

// CHECK-LABEL    builtin.func nested @test_vectorized_masked_store_b5f3e7123c911744_impl_8352646542180030969(%arg0: memref<8xi32>, %arg1: memref<5xi32>) {
// CHECK-NEXT      %cst = arith.constant dense<false> : vector<8xi1>
// CHECK-NEXT      %c0 = arith.constant 0 : index
// CHECK-NEXT      %c1 = arith.constant 1 : index
// CHECK-NEXT      %c2 = arith.constant 2 : index
// CHECK-NEXT      %c3 = arith.constant 3 : index
// CHECK-NEXT      %c4 = arith.constant 4 : index
// CHECK-NEXT      %c6 = arith.constant 6 : index
// CHECK-NEXT      %c7 = arith.constant 7 : index
// CHECK-NEXT      %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT      %c5 = arith.constant 5 : index
// CHECK-NEXT      %0 = "accv.cmp"(%c0, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %1 = vector.insertelement %0, %cst[%c0 : index] : vector<8xi1>
// CHECK-NEXT      %2 = "accv.cmp"(%c1, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %3 = vector.insertelement %2, %1[%c1 : index] : vector<8xi1>
// CHECK-NEXT      %4 = "accv.cmp"(%c2, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %5 = vector.insertelement %4, %3[%c2 : index] : vector<8xi1>
// CHECK-NEXT      %6 = "accv.cmp"(%c3, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %7 = vector.insertelement %6, %5[%c3 : index] : vector<8xi1>
// CHECK-NEXT      %8 = "accv.cmp"(%c4, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %9 = vector.insertelement %8, %7[%c4 : index] : vector<8xi1>
// CHECK-NEXT      %10 = "accv.cmp"(%c5, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %11 = vector.insertelement %10, %9[%c5 : index] : vector<8xi1>
// CHECK-NEXT      %12 = "accv.cmp"(%c6, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %13 = vector.insertelement %12, %11[%c6 : index] : vector<8xi1>
// CHECK-NEXT      %14 = "accv.cmp"(%c7, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %15 = vector.insertelement %14, %13[%c7 : index] : vector<8xi1>
// CHECK-NEXT      %16 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<8xi32> to memref<8xi32>
// CHECK-NEXT      %17 = vector.transfer_read %16[%c0], %c0_i32, %15 {in_bounds = [true]} : memref<8xi32>, vector<8xi32>
// CHECK-NEXT      %18 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [5], strides: [1] : memref<5xi32> to memref<5xi32>
// CHECK-NEXT      vector.transfer_write %17, %18[%c0], %15 {in_bounds = [true]} : vector<8xi32>, memref<5xi32>
// CHECK-NEXT      return
// CHECK-NEXT    }

    builtin.func nested @test_vectorized_masked_store_b5f3e7123c911744_impl_8352646542180030969(%arg0: memref<8xi32>, %arg1: memref<5xi32>){
      %c5 = arith.constant 5 : index
      affine.for %arg2 = 0 to 8 {
        %0 = "accv.cmp"(%arg2, %c5) {predicate = 2 : i64} : (index, index) -> i1
        scf.if %0 {
          %1 = affine.load %arg0[%arg2] : memref<8xi32>
          affine.store %1, %arg1[%arg2] : memref<5xi32>
        }
      } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      return
    }
  }
}

// -----

module @test_vectorized_masked_accumulate attributes {accv.target_device_features = "+avx2", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "test_vectorized_masked_accumulate" {

// CHECK-LABEL    builtin.func nested @test_vectorized_masked_accumulate_9145c49af2395c43_impl_8352646542180030969(%arg0: memref<8xi32>, %arg1: memref<5xi32>) {
// CHECK-NEXT      %cst = arith.constant dense<false> : vector<8xi1>
// CHECK-NEXT      %c0 = arith.constant 0 : index
// CHECK-NEXT      %c1 = arith.constant 1 : index
// CHECK-NEXT      %c2 = arith.constant 2 : index
// CHECK-NEXT      %c3 = arith.constant 3 : index
// CHECK-NEXT      %c4 = arith.constant 4 : index
// CHECK-NEXT      %c6 = arith.constant 6 : index
// CHECK-NEXT      %c7 = arith.constant 7 : index
// CHECK-NEXT      %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT      %c5 = arith.constant 5 : index
// CHECK-NEXT      %0 = "accv.cmp"(%c0, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %1 = vector.insertelement %0, %cst[%c0 : index] : vector<8xi1>
// CHECK-NEXT      %2 = "accv.cmp"(%c1, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %3 = vector.insertelement %2, %1[%c1 : index] : vector<8xi1>
// CHECK-NEXT      %4 = "accv.cmp"(%c2, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %5 = vector.insertelement %4, %3[%c2 : index] : vector<8xi1>
// CHECK-NEXT      %6 = "accv.cmp"(%c3, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %7 = vector.insertelement %6, %5[%c3 : index] : vector<8xi1>
// CHECK-NEXT      %8 = "accv.cmp"(%c4, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %9 = vector.insertelement %8, %7[%c4 : index] : vector<8xi1>
// CHECK-NEXT      %10 = "accv.cmp"(%c5, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %11 = vector.insertelement %10, %9[%c5 : index] : vector<8xi1>
// CHECK-NEXT      %12 = "accv.cmp"(%c6, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %13 = vector.insertelement %12, %11[%c6 : index] : vector<8xi1>
// CHECK-NEXT      %14 = "accv.cmp"(%c7, %c5) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %15 = vector.insertelement %14, %13[%c7 : index] : vector<8xi1>
// CHECK-NEXT      %16 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [5], strides: [1] : memref<5xi32> to memref<5xi32>
// CHECK-NEXT      %17 = vector.transfer_read %16[%c0], %c0_i32, %15 {in_bounds = [true]} : memref<5xi32>, vector<8xi32>
// CHECK-NEXT      %18 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<8xi32> to memref<8xi32>
// CHECK-NEXT      %19 = vector.transfer_read %18[%c0], %c0_i32, %15 {in_bounds = [true]} : memref<8xi32>, vector<8xi32>
// CHECK-NEXT      %20 = "accv.bin_op"(%17, %19) {predicate = 0 : i64} : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
// CHECK-NEXT      %21 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [5], strides: [1] : memref<5xi32> to memref<5xi32>
// CHECK-NEXT      vector.transfer_write %20, %21[%c0], %15 {in_bounds = [true]} : vector<8xi32>, memref<5xi32>
// CHECK-NEXT      return
// CHECK-NEXT    }    
    builtin.func nested @test_vectorized_masked_accumulate_9145c49af2395c43_impl_8352646542180030969(%arg0: memref<8xi32>, %arg1: memref<5xi32>){
      %c5 = arith.constant 5 : index
      affine.for %arg2 = 0 to 8 {
        %0 = "accv.cmp"(%arg2, %c5) {predicate = 2 : i64} : (index, index) -> i1
        scf.if %0 {
          %1 = affine.load %arg1[%arg2] : memref<5xi32>
          %2 = affine.load %arg0[%arg2] : memref<8xi32>
          %3 = "accv.bin_op"(%1, %2) {predicate = 0 : i64} : (i32, i32) -> i32
          affine.store %3, %arg1[%arg2] : memref<5xi32>
        }
      } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      return
    }
  }
}


// -----

module @test_dynamic_vectorized_masked_buffer_fill attributes {accv.target_device_features = "+avx2", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "test_dynamic_vectorized_masked_buffer_fill" {
// CHECK-LABEL    builtin.func nested @test_dynamic_vectorized_masked_buffer_fill_a2082ab34abc4983_impl_7667734917396200012(%arg0: memref<?xi32>, %arg1: index, %arg2: memref<8xi32>) attributes {accv.dyn_arg_size_refs = [[1], [-1], [-1]]} {
// CHECK-NEXT      %cst = arith.constant dense<false> : vector<8xi1>
// CHECK-NEXT      %c2 = arith.constant 2 : index
// CHECK-NEXT      %c3 = arith.constant 3 : index
// CHECK-NEXT      %c4 = arith.constant 4 : index
// CHECK-NEXT      %c5 = arith.constant 5 : index
// CHECK-NEXT      %c6 = arith.constant 6 : index
// CHECK-NEXT      %c7 = arith.constant 7 : index
// CHECK-NEXT      %c1 = arith.constant 1 : index
// CHECK-NEXT      %c0 = arith.constant 0 : index
// CHECK-NEXT      %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT      %0 = "accv.cmp"(%c0, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %1 = vector.insertelement %0, %cst[%c0 : index] : vector<8xi1>
// CHECK-NEXT      %2 = "accv.cmp"(%c1, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %3 = vector.insertelement %2, %1[%c1 : index] : vector<8xi1>
// CHECK-NEXT      %4 = "accv.cmp"(%c2, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %5 = vector.insertelement %4, %3[%c2 : index] : vector<8xi1>
// CHECK-NEXT      %6 = "accv.cmp"(%c3, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %7 = vector.insertelement %6, %5[%c3 : index] : vector<8xi1>
// CHECK-NEXT      %8 = "accv.cmp"(%c4, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %9 = vector.insertelement %8, %7[%c4 : index] : vector<8xi1>
// CHECK-NEXT      %10 = "accv.cmp"(%c5, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %11 = vector.insertelement %10, %9[%c5 : index] : vector<8xi1>
// CHECK-NEXT      %12 = "accv.cmp"(%c6, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %13 = vector.insertelement %12, %11[%c6 : index] : vector<8xi1>
// CHECK-NEXT      %14 = "accv.cmp"(%c7, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %15 = vector.insertelement %14, %13[%c7 : index] : vector<8xi1>
// CHECK-NEXT      %16 = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [%arg1], strides: [%c1] : memref<?xi32> to memref<?xi32>
// CHECK-NEXT      %17 = vector.transfer_read %16[%c0], %c0_i32, %15 {in_bounds = [true]} : memref<?xi32>, vector<8xi32>
// CHECK-NEXT      %18 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8], strides: [1] : memref<8xi32> to memref<8xi32>
// CHECK-NEXT      vector.store %17, %18[%c0] : memref<8xi32>, vector<8xi32>
// CHECK-NEXT      return
// CHECK-NEXT    }
    builtin.func nested @test_dynamic_vectorized_masked_buffer_fill_a2082ab34abc4983_impl_7667734917396200012(%arg0: memref<?xi32>, %arg1: index, %arg2: memref<8xi32>) attributes {accv.dyn_arg_size_refs = [[1], [-1], [-1]]} {
      %c0_i32 = arith.constant 0 : i32
      affine.for %arg3 = 0 to 8 {
        %0 = "accv.cmp"(%arg3, %arg1) {predicate = 2 : i64} : (index, index) -> i1
        scf.if %0 {
          %1 = affine.load %arg0[%arg3] : memref<?xi32>
          affine.store %1, %arg2[%arg3] : memref<8xi32>
        } else {
          affine.store %c0_i32, %arg2[%arg3] : memref<8xi32>
        }
      } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      return
    }
  }
}

// -----

module @test_dynamic_vectorized_masked_store attributes {accv.target_device_features = "+avx2", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "test_dynamic_vectorized_masked_store" {
// CHECK-LABEL   builtin.func nested @test_dynamic_vectorized_masked_store_56cdc5710413acf9_impl_17986325373560006911(%arg0: memref<8xi32>, %arg1: memref<?xi32>, %arg2: index) attributes {accv.dyn_arg_size_refs = [[-1], [2], [-1]]} {
// CHECK-NEXT      %cst = arith.constant dense<false> : vector<8xi1>
// CHECK-NEXT      %c2 = arith.constant 2 : index
// CHECK-NEXT      %c3 = arith.constant 3 : index
// CHECK-NEXT      %c4 = arith.constant 4 : index
// CHECK-NEXT      %c5 = arith.constant 5 : index
// CHECK-NEXT      %c6 = arith.constant 6 : index
// CHECK-NEXT      %c7 = arith.constant 7 : index
// CHECK-NEXT      %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT      %c1 = arith.constant 1 : index
// CHECK-NEXT      %c0 = arith.constant 0 : index
// CHECK-NEXT      %0 = "accv.cmp"(%c0, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %1 = vector.insertelement %0, %cst[%c0 : index] : vector<8xi1>
// CHECK-NEXT      %2 = "accv.cmp"(%c1, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %3 = vector.insertelement %2, %1[%c1 : index] : vector<8xi1>
// CHECK-NEXT      %4 = "accv.cmp"(%c2, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %5 = vector.insertelement %4, %3[%c2 : index] : vector<8xi1>
// CHECK-NEXT      %6 = "accv.cmp"(%c3, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %7 = vector.insertelement %6, %5[%c3 : index] : vector<8xi1>
// CHECK-NEXT      %8 = "accv.cmp"(%c4, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %9 = vector.insertelement %8, %7[%c4 : index] : vector<8xi1>
// CHECK-NEXT      %10 = "accv.cmp"(%c5, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %11 = vector.insertelement %10, %9[%c5 : index] : vector<8xi1>
// CHECK-NEXT      %12 = "accv.cmp"(%c6, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %13 = vector.insertelement %12, %11[%c6 : index] : vector<8xi1>
// CHECK-NEXT      %14 = "accv.cmp"(%c7, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %15 = vector.insertelement %14, %13[%c7 : index] : vector<8xi1>
// CHECK-NEXT      %16 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<8xi32> to memref<8xi32>
// CHECK-NEXT      %17 = vector.transfer_read %16[%c0], %c0_i32, %15 {in_bounds = [true]} : memref<8xi32>, vector<8xi32>
// CHECK-NEXT      %18 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [%arg2], strides: [%c1] : memref<?xi32> to memref<?xi32>
// CHECK-NEXT      vector.transfer_write %17, %18[%c0], %15 {in_bounds = [true]} : vector<8xi32>, memref<?xi32>
// CHECK-NEXT      return
// CHECK-NEXT    }
    builtin.func nested @test_dynamic_vectorized_masked_store_56cdc5710413acf9_impl_17986325373560006911(%arg0: memref<8xi32>, %arg1: memref<?xi32>, %arg2: index) attributes {accv.dyn_arg_size_refs = [[-1], [2], [-1]]} {
      affine.for %arg3 = 0 to 8 {
        %0 = "accv.cmp"(%arg3, %arg2) {predicate = 2 : i64} : (index, index) -> i1
        scf.if %0 {
          %1 = affine.load %arg0[%arg3] : memref<8xi32>
          affine.store %1, %arg1[%arg3] : memref<?xi32>
        }
      } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      return
    }
  }
}

// -----

module @test_dynamic_vectorized_masked_accumulate attributes {accv.target_device_features = "+avx2", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "test_dynamic_vectorized_masked_accumulate" {
// CHECK-LABEL    builtin.func nested @test_dynamic_vectorized_masked_accumulate_54da4d9483f52971_impl_18007838800914849862(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: index) attributes {accv.dyn_arg_size_refs = [[-1], [-1], [-1]]} {
// CHECK-NEXT      %cst = arith.constant dense<false> : vector<8xi1>
// CHECK-NEXT      %c0 = arith.constant 0 : index
// CHECK-NEXT      %c1 = arith.constant 1 : index
// CHECK-NEXT      %c2 = arith.constant 2 : index
// CHECK-NEXT      %c3 = arith.constant 3 : index
// CHECK-NEXT      %c4 = arith.constant 4 : index
// CHECK-NEXT      %c5 = arith.constant 5 : index
// CHECK-NEXT      %c6 = arith.constant 6 : index
// CHECK-NEXT      %c7 = arith.constant 7 : index
// CHECK-NEXT      %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT      %0 = "accv.cmp"(%c0, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %1 = vector.insertelement %0, %cst[%c0 : index] : vector<8xi1>
// CHECK-NEXT      %2 = "accv.cmp"(%c1, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %3 = vector.insertelement %2, %1[%c1 : index] : vector<8xi1>
// CHECK-NEXT      %4 = "accv.cmp"(%c2, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %5 = vector.insertelement %4, %3[%c2 : index] : vector<8xi1>
// CHECK-NEXT      %6 = "accv.cmp"(%c3, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %7 = vector.insertelement %6, %5[%c3 : index] : vector<8xi1>
// CHECK-NEXT      %8 = "accv.cmp"(%c4, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %9 = vector.insertelement %8, %7[%c4 : index] : vector<8xi1>
// CHECK-NEXT      %10 = "accv.cmp"(%c5, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %11 = vector.insertelement %10, %9[%c5 : index] : vector<8xi1>
// CHECK-NEXT      %12 = "accv.cmp"(%c6, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %13 = vector.insertelement %12, %11[%c6 : index] : vector<8xi1>
// CHECK-NEXT      %14 = "accv.cmp"(%c7, %arg2) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %15 = vector.insertelement %14, %13[%c7 : index] : vector<8xi1>
// CHECK-NEXT      %16 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8], strides: [1] : memref<8xi32> to memref<8xi32>
// CHECK-NEXT      %17 = vector.transfer_read %16[%c0], %c0_i32, %15 {in_bounds = [true]} : memref<8xi32>, vector<8xi32>
// CHECK-NEXT      %18 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<8xi32> to memref<8xi32>
// CHECK-NEXT      %19 = vector.transfer_read %18[%c0], %c0_i32, %15 {in_bounds = [true]} : memref<8xi32>, vector<8xi32>
// CHECK-NEXT      %20 = "accv.bin_op"(%17, %19) {predicate = 0 : i64} : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
// CHECK-NEXT      %21 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8], strides: [1] : memref<8xi32> to memref<8xi32>
// CHECK-NEXT      vector.transfer_write %20, %21[%c0], %15 {in_bounds = [true]} : vector<8xi32>, memref<8xi32>
// CHECK-NEXT      return
// CHECK-NEXT    }    
    builtin.func nested @test_dynamic_vectorized_masked_accumulate_54da4d9483f52971_impl_18007838800914849862(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: index) attributes {accv.dyn_arg_size_refs = [[-1], [-1], [-1]]} {
      affine.for %arg3 = 0 to 8 {
        %0 = "accv.cmp"(%arg3, %arg2) {predicate = 2 : i64} : (index, index) -> i1
        scf.if %0 {
          %1 = affine.load %arg1[%arg3] : memref<8xi32>
          %2 = affine.load %arg0[%arg3] : memref<8xi32>
          %3 = "accv.bin_op"(%1, %2) {predicate = 0 : i64} : (i32, i32) -> i32
          affine.store %3, %arg1[%arg3] : memref<8xi32>
        }
      } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      return
    }
  }
}


// -----

module @test_dynamic_vectorized_mask_lower_zero_with_fusion attributes {accv.target_device_features = "+avx2", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "test_dynamic_vectorized_mask_lower_zero_with_fusion" {
// CHECK-LABEL    builtin.func nested @test_dynamic_vectorized_mask_lower_zero_with_fusion_bd008a592f2068ba_impl_10116581887106739959(%arg0: memref<8xi32>, %arg1: index) attributes {accv.dyn_arg_size_refs = [[-1], [-1]]} {
// CHECK-NEXT      %cst = arith.constant dense<false> : vector<8xi1>
// CHECK-NEXT      %c0 = arith.constant 0 : index
// CHECK-NEXT      %c1 = arith.constant 1 : index
// CHECK-NEXT      %c2 = arith.constant 2 : index
// CHECK-NEXT      %c3 = arith.constant 3 : index
// CHECK-NEXT      %c4 = arith.constant 4 : index
// CHECK-NEXT      %c5 = arith.constant 5 : index
// CHECK-NEXT      %c6 = arith.constant 6 : index
// CHECK-NEXT      %c7 = arith.constant 7 : index
// CHECK-NEXT      %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT      %0 = "accv.alloc"() {allocType = 0 : i64} : () -> memref<8xi32, 3>
// CHECK-NEXT      affine.store %c0_i32, %0[0] : memref<8xi32, 3>
// CHECK-NEXT      affine.store %c0_i32, %0[1] : memref<8xi32, 3>
// CHECK-NEXT      affine.store %c0_i32, %0[2] : memref<8xi32, 3>
// CHECK-NEXT      affine.store %c0_i32, %0[3] : memref<8xi32, 3>
// CHECK-NEXT      affine.store %c0_i32, %0[4] : memref<8xi32, 3>
// CHECK-NEXT      affine.store %c0_i32, %0[5] : memref<8xi32, 3>
// CHECK-NEXT      affine.store %c0_i32, %0[6] : memref<8xi32, 3>
// CHECK-NEXT      affine.store %c0_i32, %0[7] : memref<8xi32, 3>
// CHECK-NEXT      %1 = "accv.cmp"(%c0, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %2 = vector.insertelement %1, %cst[%c0 : index] : vector<8xi1>
// CHECK-NEXT      %3 = "accv.cmp"(%c1, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %4 = vector.insertelement %3, %2[%c1 : index] : vector<8xi1>
// CHECK-NEXT      %5 = "accv.cmp"(%c2, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %6 = vector.insertelement %5, %4[%c2 : index] : vector<8xi1>
// CHECK-NEXT      %7 = "accv.cmp"(%c3, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %8 = vector.insertelement %7, %6[%c3 : index] : vector<8xi1>
// CHECK-NEXT      %9 = "accv.cmp"(%c4, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %10 = vector.insertelement %9, %8[%c4 : index] : vector<8xi1>
// CHECK-NEXT      %11 = "accv.cmp"(%c5, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %12 = vector.insertelement %11, %10[%c5 : index] : vector<8xi1>
// CHECK-NEXT      %13 = "accv.cmp"(%c6, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %14 = vector.insertelement %13, %12[%c6 : index] : vector<8xi1>
// CHECK-NEXT      %15 = "accv.cmp"(%c7, %arg1) {predicate = 2 : i64} : (index, index) -> i1
// CHECK-NEXT      %16 = vector.insertelement %15, %14[%c7 : index] : vector<8xi1>
// CHECK-NEXT      %17 = memref.reinterpret_cast %0 to offset: [0], sizes: [8], strides: [1] : memref<8xi32, 3> to memref<8xi32, 3>
// CHECK-NEXT      %18 = vector.transfer_read %17[%c0], %c0_i32, %16 {in_bounds = [true]} : memref<8xi32, 3>, vector<8xi32>
// CHECK-NEXT      %19 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<8xi32> to memref<8xi32>
// CHECK-NEXT      vector.transfer_write %18, %19[%c0], %16 {in_bounds = [true]} : vector<8xi32>, memref<8xi32>
// CHECK-NEXT      return
// CHECK-NEXT    }    
    builtin.func nested @test_dynamic_vectorized_mask_lower_zero_with_fusion_bd008a592f2068ba_impl_10116581887106739959(%arg0: memref<8xi32>, %arg1: index) attributes {accv.dyn_arg_size_refs = [[-1], [-1]]} {
      %c0_i32 = arith.constant 0 : i32
      %0 = "accv.alloc"() {allocType = 0 : i64} : () -> memref<8xi32, 3>
      affine.for %arg2 = 0 to 8 {
        affine.store %c0_i32, %0[%arg2] : memref<8xi32, 3>
      } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      affine.for %arg2 = 0 to 8 {
        %1 = "accv.cmp"(%arg2, %arg1) {predicate = 2 : i64} : (index, index) -> i1
        scf.if %1 {
          %2 = affine.load %0[%arg2] : memref<8xi32, 3>
          affine.store %2, %arg0[%arg2] : memref<8xi32>
        }
      } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      return
    }
  }
}


// -----

module @test_dynamic_vectorized_mask_lower_zero attributes {accv.target_device_features = "+avx2", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  accv.module "test_dynamic_vectorized_mask_lower_zero" {
// CHECK-LABEL TODO
    builtin.func nested @test_dynamic_vectorized_mask_lower_zero_3d7d26dd8c9c1178_impl_10116581887106739959(%arg0: memref<8xi32>, %arg1: index) {
      %c0_i32 = arith.constant 0 : i32
      affine.for %arg2 = 0 to 8 {
        %0 = "accv.cmp"(%arg2, %arg1) {predicate = 2 : i64} : (index, index) -> i1
        scf.if %0 {
          affine.store %c0_i32, %arg0[%arg2] : memref<8xi32>
        }
      } {accxp_vectorizationInfo = #accxp<"vectorizationinfo{32,16,0}">}
      return
    }
  }
}
