// RUN: acc-opt --verify-each=false --acc-affine-simplify %s | FileCheck %s

module @test_accera_affine_simplification {
    accv.module "test_accera_affine_simplification"  {

        // FloorDiv simplification tests

        // CHECK-LABEL accv.func nested @test_simplify_floordiv_no_terms_strides
        accv.func nested @test_simplify_floordiv_no_terms_strides(%arg0: memref<32xf32>) attributes {exec_target = 0 : i64} {
            %0 = memref.alloc() : memref<32xf32>
            affine.for %arg1 = 0 to 16 {
                affine.for %arg2 = 0 to 16 {
                    affine.for %arg3 = 0 to 4 {
                        // CHECK: %1 = affine.load %arg0[(%arg1 * 64 + %arg2 * 33 + %arg3 * 31) floordiv 32] : memref<32xf32>
                        %1 = affine.load %arg0[(%arg1 * 64 + %arg2 * 33 + %arg3 * 31) floordiv 32] : memref<32xf32>
                        // CHECK: affine.store %1, %0[(%arg1 * 64 + %arg2 * 33 + %arg3 * 31) floordiv 32] : memref<32xf32>
                        affine.store %1, %0[(%arg1 * 64 + %arg2 * 33 + %arg3 * 31) floordiv 32] : memref<32xf32>
                    } {begin = 0 : i64, end = 4 : i64}
                } {begin = 0 : i64, end = 16 : i64}
            } {begin = 0 : i64, end = 16 : i64}
            accv.return
        }

        // CHECK-LABEL accv.func nested @test_simplify_floordiv_no_terms_range
        accv.func nested @test_simplify_floordiv_no_terms_range(%arg0: memref<32xf32>) attributes {exec_target = 0 : i64} {
            %0 = memref.alloc() : memref<32xf32>
            affine.for %arg1 = 0 to 16 {
                affine.for %arg2 = 0 to 16 {
                    affine.for %arg3 = 0 to 5 { // This range being 5 will prevent the simplification from removing this term
                        // CHECK: %1 = affine.load %arg0[(%arg1 * 64 + %arg2 * 4 + %arg3) floordiv 32] : memref<32xf32>
                        %1 = affine.load %arg0[(%arg1 * 64 + %arg2 * 4 + %arg3) floordiv 32] : memref<32xf32>
                        // CHECK: affine.store %1, %0[(%arg1 * 64 + %arg2 * 4 + %arg3) floordiv 32] : memref<32xf32>
                        affine.store %1, %0[(%arg1 * 64 + %arg2 * 4 + %arg3) floordiv 32] : memref<32xf32>
                    } {begin = 0 : i64, end = 4 : i64}
                } {begin = 0 : i64, end = 16 : i64}
            } {begin = 0 : i64, end = 16 : i64}
            accv.return
        }

        // CHECK-LABEL accv.func nested @test_simplify_floordiv_one_term
        accv.func nested @test_simplify_floordiv_one_term(%arg0: memref<32xf32>) attributes {exec_target = 0 : i64} {
            %0 = memref.alloc() : memref<32xf32>
            affine.for %arg1 = 0 to 16 {
                affine.for %arg2 = 0 to 16 {
                    affine.for %arg3 = 0 to 4 {
                        // CHECK: %1 = affine.load %arg0[%arg1 * 2 + (%arg2 * 48) floordiv 32] : memref<32xf32>
                        %1 = affine.load %arg0[(%arg1 * 64 + %arg2 * 48 + %arg3) floordiv 32] : memref<32xf32>
                        // CHECK: affine.store %1, %0[%arg1 * 2 + (%arg2 * 48) floordiv 32] : memref<32xf32>
                        affine.store %1, %0[(%arg1 * 64 + %arg2 * 48 + %arg3) floordiv 32] : memref<32xf32>
                    } {begin = 0 : i64, end = 4 : i64}
                } {begin = 0 : i64, end = 16 : i64}
            } {begin = 0 : i64, end = 16 : i64}
            accv.return
        }

        // CHECK-LABEL accv.func nested @test_simplify_floordiv_two_terms
        accv.func nested @test_simplify_floordiv_two_terms(%arg0: memref<32xf32>) attributes {exec_target = 0 : i64} {
            %0 = memref.alloc() : memref<32xf32>
            affine.for %arg1 = 0 to 16 {
                affine.for %arg2 = 0 to 16 {
                    affine.for %arg3 = 0 to 4 {
                        // CHECK: %1 = affine.load %arg0[%arg1 * 2] : memref<32xf32>
                        %1 = affine.load %arg0[(%arg1 * 128 + %arg2 * 4 + %arg3) floordiv 64] : memref<32xf32>
                        // CHECK: affine.store %1, %0[%arg1 * 2] : memref<32xf32>
                        affine.store %1, %0[(%arg1 * 128 + %arg2 * 4 + %arg3) floordiv 64] : memref<32xf32>
                    } {begin = 0 : i64, end = 4 : i64}
                } {begin = 0 : i64, end = 16 : i64}
            } {begin = 0 : i64, end = 16 : i64}
            accv.return
        }

        // Mod simplification tests

        // CHECK-LABEL accv.func nested @test_simplify_mod_no_terms_strides
        accv.func nested @test_simplify_mod_no_terms_strides(%arg0: memref<32xf32>) attributes {exec_target = 0 : i64} {
            %0 = memref.alloc() : memref<32xf32>
            affine.for %arg1 = 0 to 16 {
                affine.for %arg2 = 0 to 16 {
                    affine.for %arg3 = 0 to 4 {
                        // CHECK: %1 = affine.load %arg0[(%arg1 * 68 + %arg2 * 33 + %arg3 * 31) mod 32] : memref<32xf32>
                        %1 = affine.load %arg0[(%arg1 * 68 + %arg2 * 33 + %arg3 * 31) mod 32] : memref<32xf32>
                        // CHECK: affine.store %1, %0[(%arg1 * 68 + %arg2 * 33 + %arg3 * 31) mod 32] : memref<32xf32>
                        affine.store %1, %0[(%arg1 * 68 + %arg2 * 33 + %arg3 * 31) mod 32] : memref<32xf32>
                    } {begin = 0 : i64, end = 4 : i64}
                } {begin = 0 : i64, end = 16 : i64}
            } {begin = 0 : i64, end = 16 : i64}
            accv.return
        }

        // CHECK-LABEL accv.func nested @test_simplify_mod_no_terms_range
        accv.func nested @test_simplify_mod_no_terms_range(%arg0: memref<32xf32>) attributes {exec_target = 0 : i64} {
            %0 = memref.alloc() : memref<32xf32>
            affine.for %arg1 = 0 to 16 {
                affine.for %arg2 = 0 to 16 {
                    affine.for %arg3 = 0 to 5 { // This range being 5 will prevent the simplification from removing this term
                        // CHECK: %1 = affine.load %arg0[(%arg1 * 64 + %arg2 * 4 + %arg3) mod 32] : memref<32xf32>
                        %1 = affine.load %arg0[(%arg1 * 64 + %arg2 * 4 + %arg3) mod 32] : memref<32xf32>
                        // CHECK: affine.store %1, %0[(%arg1 * 64 + %arg2 * 4 + %arg3) mod 32] : memref<32xf32>
                        affine.store %1, %0[(%arg1 * 64 + %arg2 * 4 + %arg3) mod 32] : memref<32xf32>
                    } {begin = 0 : i64, end = 4 : i64}
                } {begin = 0 : i64, end = 16 : i64}
            } {begin = 0 : i64, end = 16 : i64}
            accv.return
        }

        // CHECK-LABEL accv.func nested @test_simplify_mod_one_term
        accv.func nested @test_simplify_mod_one_term(%arg0: memref<32xf32>) attributes {exec_target = 0 : i64} {
            %0 = memref.alloc() : memref<32xf32>
            affine.for %arg1 = 0 to 16 {
                affine.for %arg2 = 0 to 16 {
                    affine.for %arg3 = 0 to 4 {
                        // CHECK: %1 = affine.load %arg0[%arg3 + (%arg1 * 68 + %arg2 * 48) mod 32] : memref<32xf32>
                        %1 = affine.load %arg0[(%arg1 * 68 + %arg2 * 48 + %arg3) mod 32] : memref<32xf32>
                        // CHECK: affine.store %1, %0[%arg3 + (%arg1 * 68 + %arg2 * 48) mod 32] : memref<32xf32>
                        affine.store %1, %0[(%arg1 * 68 + %arg2 * 48 + %arg3) mod 32] : memref<32xf32>
                    } {begin = 0 : i64, end = 4 : i64}
                } {begin = 0 : i64, end = 16 : i64}
            } {begin = 0 : i64, end = 16 : i64}
            accv.return
        }

        // CHECK-LABEL accv.func nested @test_simplify_mod_all_terms
        accv.func nested @test_simplify_mod_all_terms(%arg0: memref<64xf32>) attributes {exec_target = 0 : i64} {
            %0 = memref.alloc() : memref<64xf32>
            affine.for %arg1 = 0 to 16 {
                affine.for %arg2 = 0 to 16 {
                    affine.for %arg3 = 0 to 4 {
                        // CHECK: %1 = affine.load %arg0[%arg3 + %arg2 * 4] : memref<64xf32>
                        %1 = affine.load %arg0[(%arg1 * 128 + %arg2 * 4 + %arg3) mod 64] : memref<64xf32>
                        // CHECK: affine.store %1, %0[%arg3 + %arg2 * 4] : memref<64xf32>
                        affine.store %1, %0[(%arg1 * 128 + %arg2 * 4 + %arg3) mod 64] : memref<64xf32>
                    } {begin = 0 : i64, end = 4 : i64}
                } {begin = 0 : i64, end = 16 : i64}
            } {begin = 0 : i64, end = 16 : i64}
            accv.return
        }
    }
}
