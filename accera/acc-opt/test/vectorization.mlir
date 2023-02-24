// RUN: acc-opt --verify-each=false --acc-vectorize %s | FileCheck %s

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
        builtin.func nested @test_view_split_dim_interleaved_pack(%arg0: memref<1885x256xui8> loc(unknown), %arg1: memref<483840xui8> loc(unknown)) attributes {accv.dyn_arg_size_refs = [[-1, -1], [-1]], accv.usages = [1 : i8, 0 : i8], args_name = ["", ""], args_size = ["1885*256", "483840"], args_symbol = ["args_symbol_name_0", "args_symbol_name_1"], exec_target = 0 : i64} {
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
    }
}
