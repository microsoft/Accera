////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

// RUN:  value_mlir_test | FileCheck %s

// RUN:  value_mlir_test -r compact | python process_tests.py -p "jit_" | FileCheck --allow-empty --check-prefix=JIT %s

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

#include <value/include/Array.h>
#include <value/include/EmitterContext.h>
#include <value/include/FastMath.h>
#include <value/include/IterationDomain.h>
#include <value/include/Kernel.h>
#include <value/include/KernelPredicate.h>
#include <value/include/MLIREmitterContext.h>
#include <value/include/MLOperations.h>
#include <value/include/Matrix.h>
#include <value/include/Nest.h>
#include <value/include/Plan.h>
#include <value/include/Profiling.h>
#include <value/include/Schedule.h>
#include <value/include/Tensor.h>

#include <utilities/include/MathUtil.h>

#include <ir/include/DialectRegistry.h>

#include <llvm/Support/InitLLVM.h>

#include <cstdio>

using namespace std::string_literals;
using namespace accera::value;
using namespace accera::utilities;

struct TestCaseWrapper : Catch::EventListenerBase
{
    bool first = true;
    std::optional<ContextGuard<MLIRContext>> _guard;
    using EventListenerBase::EventListenerBase;

    void testCaseStarting(const Catch::TestCaseInfo& tci) override
    {
        _guard.emplace(tci.name);

        if (first)
        {
            first = false;
        }
        else
        {
            std::puts("\n\n");
            std::puts("// -----");
            std::puts("\n\n");
        }

        auto header = "// @" + tci.name;
        std::puts(header.c_str());
    }

    void testCaseEnded(const Catch::TestCaseStats& tcs) override
    {
        auto tci = tcs.testInfo;
        auto& guard = *_guard;
        guard.GetContext().print();
        guard.GetContext().save(tci->name + ".mlir");

        std::puts("\n\n");
    }
};
CATCH_REGISTER_LISTENER(TestCaseWrapper)

TEST_CASE("scalar_test1")
{
    DeclareFunction("f1").Define(
        [] {
            Scalar i = 10;
            Scalar j = 20;
            auto i2 = i;
            i += 2;
            auto k = i + j;
            auto k2 = i2 + j;
        });
    SUCCEED();
}

// CHECK-LABEL: module @function_decl1
// CHECK-NEXT: accv.module "function_decl1"
TEST_CASE("function_decl1")
{
    // CHECK-NEXT: accv.func nested @f1_
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    auto f1 =
        DeclareFunction("f1")
            .Define([] {});
    CHECK(f1);
    // CHECK: accv.func nested @f2_{{[0-9]+}}(%arg0: i32)
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    auto f2 =
        DeclareFunction("f2")
            .Parameters(Value{ ValueType::Int32, ScalarLayout })
            .Define([](Scalar) {});
    CHECK(f2);
    // CHECK: accv.func nested @f3_{{[0-9]+}}(%arg0: memref<10xf32>)
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    auto f3 =
        DeclareFunction("f3")
            .Parameters(Value{ ValueType::Float, MemoryLayout{ { 10 } } })
            .Define([](Value) {});
    CHECK(f3);
    // CHECK: accv.func nested @f4_{{[0-9]+}}(%arg0: memref<3x4xf64>)
    // COM: CHECK: accv.func @f4_{{[0-9]+}}(%arg0: memref<3x4xf64>)
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    auto f4 =
        DeclareFunction("f4")
            .Parameters(Value{ ValueType::Double, MemoryLayout{ { 3, 4 } } })
            .Define([](Value) {});
    CHECK(f4);
    // CHECK: accv.func nested @f5_{{[0-9]+}}(%arg0: i32, %arg1: i64)
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    auto f5 =
        DeclareFunction("f5")
            .Parameters(
                Value{ ValueType::Int32, ScalarLayout },
                Value{ ValueType::Int64, ScalarLayout })
            .Define([](Value, Value) {});
    CHECK(f5);
    // CHECK: accv.func nested @f6_{{[0-9]+}}(%arg0: index, %arg1: memref<?x16xf32>)
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    auto f6 =
        DeclareFunction("f6")
            .Parameters(
                Value{ ValueType::Index, ScalarLayout },
                Value{ ValueType::Float, MemoryLayout{ { mlir::ShapedType::kDynamicSize, 16 } } })
            .Define([](Value, Value) {});
    // CHECK: accv.func nested @f7_{{[0-9]+}}(%arg0: index, %arg1: index, %arg2: memref<?x?xf32>)
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    auto f7 =
        DeclareFunction("f7")
            .Parameters(
                Value{ ValueType::Index, ScalarLayout },
                Value{ ValueType::Index, ScalarLayout },
                Value{ ValueType::Float, MemoryLayout{ { mlir::ShapedType::kDynamicSize, mlir::ShapedType::kDynamicSize } } })
            .Define([](Value, Value, Value) {});
    // CHECK: accv.func nested @f8_{{[0-9]+}}(%arg0: index, %arg1: memref<1xindex>, %arg2: memref<1xmemref<?x?xf32>>)
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    auto f8 =
        DeclareFunction("f8")
            .Parameters(
                Value{ ValueType::Index, ScalarLayout },
                Value{ ValueType::Index, ScalarLayout, /*pointerLevel=*/1 },
                Value{ ValueType::Float, MemoryLayout{ { mlir::ShapedType::kDynamicSize, mlir::ShapedType::kDynamicSize } }, /*pointerLevel=*/2 })
            .Define([](Value, Value, Value) {});
    // CHECK-NEXT: }
}

TEST_CASE("gpu_module1")
{
    auto gpu_f1 =
        DeclareFunction("gpu_f1").Target(targets::GPU()).Define([] {});

    auto f1 = DeclareFunction("f1").Define([&] {
        gpu_f1();
    });
    SUCCEED();
}

TEST_CASE("gpu_module2")
{
    auto gpu_f1 =
        DeclareFunction("gpu_f1")
            .Target(targets::GPU({ 32, 32, 32 }, { 1, 1, 1 }))
            .Parameters(Value{ ValueType::Float, MemoryLayout{ { 16384 } } },
                        Value{ ValueType::Float, MemoryLayout{ { 16384 } } },
                        Value{ ValueType::Float, MemoryLayout{ { 16384 } } })
            .Define([](Vector A, Vector B, Vector C) {
                Scalar blockIdX = GPU::BlockId().X();
                Scalar threadIdX = GPU::ThreadId().X();
                Scalar N = int64_t{ 128 };
                auto offset = blockIdX * N + threadIdX;
                auto loadA = A[offset];
                auto loadB = B[offset];
                auto summed = loadA + loadB;
                C[offset] = summed;
            });

    auto f1 = DeclareFunction("f1")
                  .Parameters(Value{ ValueType::Float, MemoryLayout{ { 16384 } } },
                              Value{ ValueType::Float, MemoryLayout{ { 16384 } } },
                              Value{ ValueType::Float, MemoryLayout{ { 16384 } } })

                  .Define([&](Vector a, Vector b, Vector c) {
                      gpu_f1(a, b, c);
                  });
    SUCCEED();
}

TEST_CASE("MainFunc")
{
    DeclareFunction("main")
        .Decorated(false)
        .Define([] {
            auto arg0 = MakeVector<float>(16384);
            auto arg1 = MakeVector<float>(16384);
            auto arg2 = MakeVector<float>(16384);
            Scalar c0 = 0;
            Scalar c1 = 1;
            Scalar c2 = 2;
            Scalar value0 = 0.0f;
            Scalar value1 = 1.1f;
            Scalar value2 = 2.2f;
            FillResource(arg0, value1);
            FillResource(arg1, value2);
            FillResource(arg2, value0);
        });

    SUCCEED();
}

TEST_CASE("gpu_module3")
{
    auto gpu_f1 =
        DeclareFunction("gpu_f1")
            .Target(targets::GPU({ 128, 1, 1 }, { 128, 1, 1 }))
            .Parameters(Value{ ValueType::Float, MemoryLayout{ { 16384 } } },
                        Value{ ValueType::Float, MemoryLayout{ { 16384 } } },
                        Value{ ValueType::Float, MemoryLayout{ { 16384 } } })
            .Define([](Vector A, Vector B, Vector C) {
                Scalar blockIdX = GPU::BlockId().X();
                Scalar threadIdX = GPU::ThreadId().X();
                Scalar N = GPU::BlockDim().X();
                auto offset = blockIdX * N + threadIdX;
                auto loadA = A[offset];
                auto loadB = B[offset];
                auto summed = loadA + loadB;
                C[offset] = summed;
            });

    DeclareFunction("main")
        .Decorated(false)
        .Define([&gpu_f1] {
            auto arg0 = MakeVector<float>(16384);
            auto arg1 = MakeVector<float>(16384);
            auto arg2 = MakeVector<float>(16384);
            Scalar c0 = 0;
            Scalar c1 = 1;
            Scalar c2 = 2;
            Scalar value0 = 0.0f;
            Scalar value1 = 1.1f;
            Scalar value2 = 2.2f;
            FillResource(arg0, value1);
            FillResource(arg1, value2);
            FillResource(arg2, value0);

            gpu_f1(arg0, arg1, arg2);

            PrintMemref(arg2);
        });

    SUCCEED();
}

// CHECK-LABEL: module @mlir_test1
// CHECK-NEXT: accv.module "mlir_test1" {
// CHECK-NEXT: }
TEST_CASE("mlir_test1")
{
    SUCCEED();
}

// CHECK-LABEL: module @mlir_test2
// CHECK-NEXT: accv.module "mlir_test2" {
// CHECK-NEXT:  "accv.global"() {
// CHECK-SAME: sym_name = "mlir_test2_global1"
// CHECK-SAME: type = memref<10x20xf32
// CHECK-NEXT:  "accv.global"() {
// CHECK-SAME: constant
// CHECK-SAME: sym_name = "mlir_test2_global2"
// CHECK-SAME: type = memref<2x2xi32
// CHECK-SAME: }
// CHECK-NEXT: }
TEST_CASE("mlir_test2")
{
    CHECK_NOTHROW(GlobalAllocate<float>("global1", MemoryLayout(MemoryShape{ 10, 20 })));
    CHECK_NOTHROW(
        GlobalAllocate<int>(
            "global2", std::vector<int>{ 1, 2, 3, 4 }, MemoryLayout(MemoryShape{ 2, 2 })));
}

// CHECK-LABEL: module @mlir_test3
// CHECK-NEXT: accv.module "mlir_test3" {
TEST_CASE("mlir_test3")
{
    // CHECK-NEXT: accv.func nested @foo_{{[0-9]+}}(%arg0: i32)
    auto fooFn =
        DeclareFunction("foo")
            .Parameters(Value({ ValueType::Int32, ScalarLayout }))
            .Define([](Scalar i) {
                // COM: Doesn't result in emitted code
                CHECK_NOTHROW(MakeScalar<int>());

                // CHECK-NEXT: [[v0:%[a-z0-9_]+]] = "accv.alloc"() {allocType = 0 : i64} : () -> memref<100xf32, 3>
                CHECK_NOTHROW(MakeVector<float>(100));
                // CHECK-NEXT: [[v1:%[a-z0-9_]+]] = "accv.alloc"() {allocType = 0 : i64} : () -> memref<2x3xi16
                CHECK_NOTHROW(MakeMatrix<int16_t>(2, 3));
                // CHECK-NEXT: return
                // CHECK-NEXT: }
            });

    // CHECK-NEXT: }
    REQUIRE(fooFn);
}

// CHECK-LABEL: module @mlir_test4
// CHECK-NEXT: accv.module "mlir_test4" {
// CHECK-NEXT: accv.func nested @foo_{{[0-9]+}}(%arg0: memref<10x10xi32>)
// COM: CHECK-NEXT: accv.func @foo_{{[0-9]+}}(%arg0: memref<10x10xi32>) attributes {args_symbol = ["{{[a-z0-9_]+}}"], exec_target = 0 : i64, sym_visibility = "nested"} {
// CHECK-NEXT: [[c0:%c[0-9]+]] = arith.constant 0 : index
// CHECK-NEXT: [[c10_1:%c[0-9_]+]] = arith.constant 10 : index
// CHECK-NEXT: [[c10_2:%c[0-9_]+]] = arith.constant 10 : index
// CHECK-NEXT: affine.for [[iv0:%arg[0-9]+]] = 0 to 10 {
// CHECK: affine.for [[iv1:%arg[0-9]+]] = 0 to 10 {
// CHECK: }
// CHECK: }
// CHECK: return
// CHECK-NEXT: }
// CHECK-NEXT: }
TEST_CASE("mlir_test4")
{
    auto fooFn =
        DeclareFunction("foo")
            .Parameters(Value({ ValueType::Int32, MemoryLayout(MemoryShape{ 10, 10 }) }))
            .Define([](Matrix m) {
                CHECK(m.Columns() == 10);
                CHECK(m.Rows() == 10);
                For(m, [](Scalar x, Scalar y) {
                    CHECK_NOTHROW(MakeScalar<int>());
                });
            });

    REQUIRE(fooFn);
}

// TODO : revert this IR after aligned_alloc fix and Value allocate uses std_alloc again instead of accv.global

// CHECK-LABEL: module @mlir_test5
// CHECK-NEXT: accv.module "mlir_test5" {
TEST_CASE("mlir_test5")
{
    // CHECK-NEXT: accv.global
    // CHECK-SAME: constant
    // CHECK-SAME: sym_name = "mlir_test5_bar_
    // CHECK-SAME: type = memref<4xi32>
    // CHECK-SAME: value = dense<[1, 2, 3, 4]>

    // CHECK-NEXT: accv.func nested @[[BAR_FN:bar_[0-9]+]](%arg0: i32)
    auto barFn =
        DeclareFunction("bar")
            .Parameters(Value({ ValueType::Int32, ScalarLayout }))
            .Define([](Scalar i) {
                CHECK_NOTHROW(StaticAllocate<int>("foo", std::vector{ 1, 2, 3, 4 }));

                // CHECK-NEXT: "accv.alloc"() {allocType = 0 : i64} : () -> memref<100xf32, 3>
                CHECK_NOTHROW(MakeVector<float>(100));

                // CHECK-NEXT: return
                // CHECK-NEXT: }
            });
    REQUIRE(barFn);

    // CHECK-NEXT: accv.func nested @foo_{{[0-9]+}}(%arg0: memref<10x10xi32
    auto fooFn =
        DeclareFunction("foo")
            .Parameters(Value({ ValueType::Int32, MemoryLayout(MemoryShape{ 10, 10 }) }))
            .Define([&barFn](Matrix m) {
                For("matrix_loop", m, [&barFn, &m](Scalar x, Scalar y) {
                    auto i = MakeScalar<int>();
                    REQUIRE(i.GetValue().IsEmpty());
                    i = m(x, y);

                    // CHECK: accv.launch_func @[[BAR_FN]]
                    CHECK_NOTHROW(barFn(i));
                });
            });
    REQUIRE(fooFn);
}

TEST_CASE("mlir_test6")
{
    auto barFn =
        DeclareFunction("bar")
            .Parameters(Value({ ValueType::Int32, MemoryLayout(MemoryShape{ 5, 7 }) }))
            .Define([](Matrix m) {
                auto copyM = MakeMatrix<int>(5, 7);
                CHECK_NOTHROW(copyM = m);
            });
    REQUIRE(barFn);

    auto fooFn =
        DeclareFunction("foo")
            .Parameters(Value({ ValueType::Int32, MemoryLayout(MemoryShape{ 5, 7 }) }))
            .Define([&barFn](Matrix m) {
                CHECK_NOTHROW(barFn(m));
            });
    REQUIRE(fooFn);
}

TEST_CASE("mlir_test7")
{
    DeclareFunction("constant_scalar_add")
        .Define([] {
            Scalar s1 = 10;
            Scalar s2 = 20;
            Scalar s3 = s1.Copy();
            Scalar s4 = s1 + s2;
        });

    SUCCEED();
}

TEST_CASE("mlir_test8")
{
    DeclareFunction("MatMatElemwiseSum")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ 256, 256 }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ 256, 256 }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ 256, 256 }) }))
        .Define([](Matrix m1, Matrix m2, Matrix m3) {
            REQUIRE_NOTHROW(m3 = m1 + m2);
        });
}

TEST_CASE("mlir_test9")
{
    DeclareFunction("constant_scalar_assign")
        .Define([] {
            Scalar a = MakeScalar<int>("a");
            Scalar c = 4;
            REQUIRE_NOTHROW(a = c);
        });
}

TEST_CASE("mlir_test10")
{
    DeclareFunction("constant_scalar_add")
        .Define([] {
            auto m = MakeMatrix<int>(3, 3);
            auto s = m(1, 1);
            Scalar c = 10;
            REQUIRE_NOTHROW(s = c);
        });
}

// CHECK-LABEL: module @mlir_test11
// CHECK-NEXT: accv.module "mlir_test11" {
TEST_CASE("mlir_test11")
{
    // CHECK-NEXT:    accv.func nested @constant_scalar_test_{{[0-9]+}}()
    DeclareFunction("constant_scalar_test")
        .Define([] {
            // CHECK-NEXT:      [[c0_0:%c[0-9a-z_]+]] = arith.constant 0 : i32
            // CHECK-NEXT:      [[c4_0:%c[0-9a-z_]+]] = arith.constant 4
            // CHECK-NEXT:      [[c4_1:%c[0-9a-z_]+]] = arith.constant 4
            // CHECK-NEXT:      [[v0:%[a-z0-9_]+]] = "accv.alloc"()  {allocType = 0 : i64, sym_name = "a"} : () -> memref<1xi32, 3>
            Scalar a = MakeVector<int>(1, "a")[0];
            Scalar c = 4;
            // CHECK-NEXT:      %[[v1:[a-z0-9_]+]] = arith.index_cast [[c0_0]] : i32 to index
            // CHECK-NEXT:      [[v2:%[a-z0-9_]+]] = "accv.slice"([[v0]], %[[v1]]) {sliceDimensions = [0]} : (memref<1xi32, 3>, index) -> memref<i32>
            // CHECK-NEXT:      [[v3:%[0-9]+]] = "accv.cmp"([[c4_0]], [[c4_1]]) {predicate = 0 : i64} : (i32, i32) -> i1
            // CHECK-NEXT:      scf.if [[v3]] {
            // CHECK-NEXT:        [[c2_0:%c[0-9a-z_]+]] = arith.constant 2 : i32
            // CHECK-NEXT:        "accv.copy"([[c2_0]], [[v2]]) : (i32, memref<i32>) -> ()
            // CHECK-NEXT:      }
            If(c == 4, [&] { a = 2; });
            // CHECK-NEXT:      return
            // CHECK-NEXT:    }
        });
    // CHECK-NEXT:  }

    SUCCEED();
}

// CHECK-LABEL: module @mlir_test12
// CHECK-NEXT: accv.module "mlir_test12" {
// CHECK-NEXT:    accv.func nested @constant_scalar_test_{{[0-9]+}}()
// CHECK:           scf.if
// CHECK:           } else {
// CHECK-NEXT:      scf.if
// CHECK:           } else {
// CHECK:         }
// CHECK-NEXT:  }
// CHECK: scf.if
// CHECK: } else {
// CHECK-NEXT: scf.if
// CHECK: } else {
// CHECK: }
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
TEST_CASE("mlir_test12")
{
    DeclareFunction("constant_scalar_test")
        .Define([] {
            Scalar a = MakeScalar<int>("a");
            Scalar c = 4;
            If(c == 4, [&] { a = 2; }).ElseIf(c == 3, [&] { a = 7; }).Else([&] { a = 11; });
            If(a == 2, [&] { a = 13; }).ElseIf(a == 7, [&] { a = 17; }).Else([&] { a = 19; });
        });
    SUCCEED();
}

TEST_CASE("mlir_test13")
{
    DeclareFunction("MatMatMul")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ 256, 256 }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ 256, 256 }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ 256, 256 }) }))
        .Define([](Matrix m1, Matrix m2, Matrix m3) {
            auto M = (int64_t)m3.Rows();
            auto N = (int64_t)m3.Columns();
            auto K = (int64_t)m1.Columns();
            CHECK(M == 256);
            CHECK(N == 256);
            CHECK(K == 256);
            auto sum = MakeScalar<float>("inner_sum");
            ForRange(M, [&](Scalar m) {
                ForRange(N, [&](Scalar n) {
                    CHECK_NOTHROW(sum = 0.f);
                    ForRange(K, [&](Scalar k) {
                        CHECK_NOTHROW(sum += m1(m, k) * m2(k, n));
                    });
                    CHECK_NOTHROW(m3(m, n) = sum);
                });
            });
        });
}

// CHECK-LABEL: module @test_emit_c_interface
// CHECK-NEXT: accv.module "test_emit_c_interface" {
TEST_CASE("test_emit_c_interface")
{
    // CHECK-NEXT:    accv.func nested @external_func_decl_{{[0-9]+}}() attributes {accv.dyn_arg_size_refs = [], accv.usages = [], exec_target = 0 : i64, llvm.emit_c_interface} {
    auto externDecl = DeclareFunction("external_func_decl")
                          .External(true)
                          .CWrapper(true)
                          // CHECK: return
                          // CHECK-NEXT:    }
                          .Define([] {});

    // CHECK-NEXT:    accv.func nested @internal_func_decl_{{[0-9]+}}() attributes {accv.dyn_arg_size_refs = [], accv.usages = [], exec_target = 0 : i64, llvm.emit_c_interface} {
    DeclareFunction("internal_func_decl")
        .External(false)
        .CWrapper(true)
        // CHECK: return
        // CHECK-NEXT:    }
        .Define([] {});

    // CHECK-NEXT:    }
    SUCCEED();
}

// CHECK-LABEL: module @test_raw_pointer_api
// CHECK-NEXT: accv.module "test_raw_pointer_api" {
TEST_CASE("test_raw_pointer_api")
{
    // CHECK-NEXT:    accv.func nested @external_func_decl_{{[0-9]+}}() attributes {accv.dyn_arg_size_refs = [], accv.emit_raw_pointer_api, accv.usages = [], exec_target = 0 : i64} {
    auto externDecl = DeclareFunction("external_func_decl")
                          .External(true)
                          .RawPointerAPI(true)
                          // CHECK: return
                          // CHECK-NEXT:    }
                          .Define([] {});

    // CHECK-NEXT:    accv.func nested @internal_func_decl_{{[0-9]+}}() attributes {accv.dyn_arg_size_refs = [], accv.emit_raw_pointer_api, accv.usages = [], exec_target = 0 : i64} {
    DeclareFunction("internal_func_decl")
        .External(false)
        .RawPointerAPI(true)
        // CHECK: return
        // CHECK-NEXT:    }
        .Define([] {});

    // CHECK-NEXT:    }
    SUCCEED();
}

// CHECK-LABEL: module @test_emit_header_decl
// CHECK-NEXT: accv.module "test_emit_header_decl" {
TEST_CASE("test_emit_header_decl")
{
    // CHECK-NEXT:    accv.func nested @external_func_decl_{{[0-9]+}}() attributes {accv.dyn_arg_size_refs = [], accv.emit_header_decl, accv.usages = [], exec_target = 0 : i64} {
    auto externDecl = DeclareFunction("external_func_decl")
                          .External(true)
                          .HeaderDecl(true)
                          // CHECK: return
                          // CHECK-NEXT:    }
                          .Define([] {});

    // CHECK-NEXT:    accv.func nested @internal_func_decl_{{[0-9]+}}() attributes {accv.dyn_arg_size_refs = [], accv.emit_header_decl, accv.usages = [], exec_target = 0 : i64} {
    DeclareFunction("internal_func_decl")
        .External(false)
        .HeaderDecl(true)
        // CHECK: return
        // CHECK-NEXT:    }
        .Define([] {});

    // CHECK-NEXT:    }
    SUCCEED();
}

// CHECK-LABEL: module @test_function_tags
// CHECK-NEXT: accv.module "test_function_tags" {
TEST_CASE("test_function_tags")
{
    // CHECK-NEXT:    accv.func nested @no_func_tags_{{[0-9]+}}() attributes {accv.dyn_arg_size_refs = [], accv.usages = [], exec_target = 0 : i64} {
    auto externDecl = DeclareFunction("no_func_tags")
                          // CHECK: return
                          // CHECK-NEXT:    }
                          .Define([] {});

    // CHECK-NEXT:    accv.func nested @has_func_tags_{{[0-9]+}}() attributes {accv.dyn_arg_size_refs = [], accv.function_tags = {tag_a, tag_b}, accv.usages = [], exec_target = 0 : i64} {
    DeclareFunction("has_func_tags")
        .AddTag("tag_a")
        .AddTag("tag_b")
        // CHECK: return
        // CHECK-NEXT:    }
        .Define([] {});

    // CHECK-NEXT:    }
    SUCCEED();
}

// CHECK-LABEL: module @scalar_binary_ops_test
// CHECK-NEXT: accv.module "scalar_binary_ops_test" {
TEST_CASE("scalar_binary_ops_test")
{
    // CHECK: accv.func nested @scalar_binary_ops_test_{{[0-9]*}}
    // CHECK-SAME: [[AI:%arg0]]: i32
    // CHECK-SAME: [[BI:%arg1]]: i32
    // CHECK-SAME: [[AF:%arg2]]: f32
    // CHECK-SAME: [[BF:%arg3]]: f32
    DeclareFunction("scalar_binary_ops_test")
        .Parameters(Value({ ValueType::Int32, ScalarLayout }),
                    Value({ ValueType::Int32, ScalarLayout }),
                    Value({ ValueType::Float, ScalarLayout }),
                    Value({ ValueType::Float, ScalarLayout }))
        .Define([](Scalar a_i, Scalar b_i, Scalar a_f, Scalar b_f) {
            // CHECK: "accv.bin_op"([[AI]], [[BI]]) {predicate = 0 : i64} : (i32, i32) -> i32
            Scalar sum_i = a_i + b_i;

            // CHECK: "accv.bin_op"([[AI]], [[BI]]) {predicate = 1 : i64} : (i32, i32) -> i32
            Scalar diff_i = a_i - b_i;

            // CHECK: "accv.bin_op"([[AI]], [[BI]]) {predicate = 2 : i64} : (i32, i32) -> i32
            Scalar prod_i = a_i * b_i;

            // CHECK: "accv.bin_op"([[AI]], [[BI]]) {predicate = 3 : i64} : (i32, i32) -> i32
            Scalar quot_i = a_i / b_i;

            // CHECK: "accv.bin_op"([[AF]], [[BF]]) {predicate = 0 : i64} : (f32, f32) -> f32
            Scalar sum_f = a_f + b_f;

            // CHECK: "accv.bin_op"([[AF]], [[BF]]) {predicate = 1 : i64} : (f32, f32) -> f32
            Scalar diff_f = a_f - b_f;

            // CHECK: "accv.bin_op"([[AF]], [[BF]]) {predicate = 2 : i64} : (f32, f32) -> f32
            Scalar prod_f = a_f * b_f;

            // CHECK: "accv.bin_op"([[AF]], [[BF]]) {predicate = 3 : i64} : (f32, f32) -> f32
            Scalar quot_f = a_f / b_f;
        });

    SUCCEED();
}

TEST_CASE("mlir_nest_test")
{
    const int M = 8;
    const int N = 10;
    const int K = 11;

    DeclareFunction("NestMatMul")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
        .Define([=](Matrix A, Matrix B, Matrix C) {
            Nest matmul(MemoryShape{ M, N, K });
            auto indices = matmul.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            matmul.Set([&]() { C(i, j) += A(i, k) * B(k, j); });
        });

    SUCCEED();
}

TEST_CASE("mlir_schedule_test")
{
    const int M = 8;
    const int N = 10;
    const int K = 11;

    DeclareFunction("NestMatMul")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
        .Define([=](Matrix A, Matrix B, Matrix C) {
            Nest matmul(MemoryShape{ M, N, K });
            auto indices = matmul.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            matmul.Set([&]() { C(i, j) += A(i, k) * B(k, j); });

            auto schedule = matmul.CreateSchedule();
            auto [iOuter, iInner] = schedule.Split(i, 4);
            // schedule.SetOrder({ iOuter, indices[1], indices[2], iInner });
            schedule.Unroll(iInner);
        });

    SUCCEED();
}

TEST_CASE("mlir_schedule_test_2")
{
    const int M = 8;
    const int N = 10;

    DeclareFunction("TwoScheduleTest")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
        .Define([=](Matrix C) {
            Index i, j;
            IterationDomain domain{ { i, { 0, M } },
                                    { j, { 0, N } } };

            Nest nest1(domain);
            Nest nest2(domain);

            ScalarIndex ii, jj;
            std::tie(ii, jj) = nest1.GetIndices<2>();

            auto initCKernel = Kernel("initC", [&]() {
                C(ii, jj) = 0.0f;
            });

            auto schedule1 = nest1.CreateSchedule();
            auto schedule2 = nest2.CreateSchedule();

            schedule1.AddKernel(initCKernel);
            schedule2.AddKernel(initCKernel);
        });

    SUCCEED();
}

TEST_CASE("mlir_schedule_test_3")
{
    const int M = 8;
    const int N = 10;

    DeclareFunction("TwoSchedulePerNestTest")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
        .Define([=](Matrix C) {
            Nest nest(MemoryShape{ M, N });
            ScalarIndex i, j;
            std::tie(i, j) = nest.GetIndices<2>();
            nest.Set([&]() { C(i, j) = 0.0f; });

            auto schedule1 = nest.CreateSchedule();
            auto schedule2 = nest.CreateSchedule();
        });

    SUCCEED();
}

// CHECK-LABEL: module @mlir_schedule_test_4
// CHECK-NEXT: accv.module "mlir_schedule_test_4" {

TEST_CASE("mlir_schedule_test_4")
{
    const int M = 18;
    const int N = 384;
    const int K = 96;

    auto T = ValueType::Int32;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto A = MakeArray({ M, K }, T, AllocateFlags::Stack);
            auto B = MakeArray({ K, N }, T, AllocateFlags::Stack);
            auto C = MakeArray({ M, N }, T, AllocateFlags::Stack);
            auto accum = MakeArray({ 1 }, T, AllocateFlags::Stack);

            Nest matMulNest(MemoryShape{ M, N, K });
            auto [i, j, k] = matMulNest.GetIndices<3>();
            auto schedule = matMulNest.CreateSchedule();

            auto clearAccumKernel = Kernel("clearAccum", [&]() {
                accum(0) = Cast(0, T);
            });

            auto computeKernel = Kernel("compute", [&, i = i, j = j, k = k]() {
                accum(0) += A(i, k) * B(k, j);
            });

            auto accumKernel = Kernel("accum", [&, i = i, j = j]() {
                C(i, j) += accum(0);
            });

            schedule.AddKernel(clearAccumKernel, First(k));
            schedule.AddKernel(computeKernel);
            schedule.AddKernel(accumKernel, Last(k));

            schedule.SetOrder({ i, j, k });
        });
    SUCCEED();
}

// CHECK-LABEL: module @mlir_matrix_view_test
// CHECK-NEXT: accv.module "mlir_matrix_view_test" {
// CHECK-NEXT: accv.func nested @MatrixView_{{[0-9]+}}(%arg0: memref<10x10xf32
// COM: CHECK: memref.subview %arg0[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}] [1, 1] [10, 1] : memref<10x10xf32, #map0> to memref<f32, #map1>
// COM: CHECK: memref.subview %arg0[%{{[a-z0-9_]+}}, 0] [1, 10] [10, 1] : memref<10x10xf32, #map0> to memref<10xf32, #map2>
// COM: CHECK: memref.subview %arg0[0, %{{[a-z0-9_]+}}] [10, 1] [10, 1] : memref<10x10xf32, #map0> to memref<10xf32, #map3>
// COM: CHECK: memref.subview %arg0[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}] [3, 4] [1, 1] : memref<10x10xf32, #map0> to memref<3x4xf32, #map4>
// COM: CHECK-NEXT: accv.func @MatrixView_{{[0-9]+}}(%arg0: memref<10x10xf32
// CHECK: "accv.slice"(%arg0, %{{[0-9]+}}, %{{[0-9]+}}) {sliceDimensions = [0, 1]} : (memref<10x10xf32>, index, index) -> memref<f32>
// CHECK: "accv.slice"(%arg0, %{{[a-z0-9_]+}}) {sliceDimensions = [0]} : (memref<10x10xf32>, index) -> memref<10xf32, #map0>
// CHECK: "accv.slice"(%arg0, %{{[a-z0-9_]+}}) {sliceDimensions = [1]} : (memref<10x10xf32>, index) -> memref<10xf32, #map1>
// CHECK: "accv.view"(%arg0, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) {operand_segment_sizes = dense<[1, 2, 2, 2, 0]> : vector<5xi32>} : (memref<10x10xf32>, index, index, index, index, index, index) -> memref<3x4xf32, #map2>
TEST_CASE("mlir_matrix_view_test")
{
    DeclareFunction("MatrixView")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ 10, 10 }) }))
        .Define([](Matrix A) {
            CHECK_NOTHROW(A(3, 4) = 1.0f);

            auto row = A.Row(Scalar(2));
            auto column = A.Column(Scalar(3));
            auto subview = A.SubMatrix(Scalar(2), Scalar(3), 3, 4);
        });
}

// CHECK-LABEL: module @mlir_tensor_view_test
// CHECK-NEXT: accv.module "mlir_tensor_view_test" {
// CHECK-NEXT: accv.func nested @TensorView_{{[0-9]+}}(%arg0: memref<5x10x15xf32
// COM: CHECK: memref.subview %arg0[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}] [1, 1, 1] [150, 15, 1] : memref<5x10x15xf32, #map0> to memref<f32, #map1>
// COM: CHECK: memref.subview %arg0[%{{[a-z0-9_]+}}, 0, 0] [1, 10, 15] [150, 15, 1] : memref<5x10x15xf32, #map0> to memref<10x15xf32, #map2>
// COM: CHECK: memref.subview %arg0[0, %{{[a-z0-9_]+}}, 0] [5, 1, 15] [150, 15, 1] : memref<5x10x15xf32, #map0> to memref<5x15xf32, #map3>
// COM: CHECK: memref.subview %arg0[0, 0, %{{[a-z0-9_]+}}] [5, 10, 1] [150, 15, 1] : memref<5x10x15xf32, #map0> to memref<5x10xf32, #map4>
// COM: CHECK: memref.subview %arg0[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, 0] [1, 1, 15] [150, 15, 1] : memref<5x10x15xf32, #map0> to memref<15xf32, #map5>
// COM: CHECK: memref.subview %arg0[%{{[a-z0-9_]+}}, 0, %{{[a-z0-9_]+}}] [1, 10, 1] [150, 15, 1] : memref<5x10x15xf32, #map0> to memref<10xf32, #map6>
// COM: CHECK: memref.subview %arg0[0, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}] [5, 1, 1] [150, 15, 1] : memref<5x10x15xf32, #map0> to memref<5xf32, #map7>
// COM: CHECK: memref.subview %arg0[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}] [3, 2, 1] [1, 1, 1] : memref<5x10x15xf32, #map0> to memref<3x2x1xf32, #map8>
// COM: CHECK-NEXT: accv.func @TensorView_{{[0-9]+}}(%arg0: memref<5x10x15xf32
// CHECK: "accv.slice"(%arg0, %{{[a-z0-9_]+}}) {sliceDimensions = [0]} : (memref<5x10x15xf32>, index) -> memref<10x15xf32, #map0>
// CHECK: "accv.slice"(%arg0, %{{[a-z0-9_]+}}) {sliceDimensions = [1]} : (memref<5x10x15xf32>, index) -> memref<5x15xf32, #map1>
// CHECK: "accv.slice"(%arg0, %{{[a-z0-9_]+}}) {sliceDimensions = [2]} : (memref<5x10x15xf32>, index) -> memref<5x10xf32, #map2>
// CHECK: "accv.slice"(%arg0, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) {sliceDimensions = [0, 1]} : (memref<5x10x15xf32>, index, index) -> memref<15xf32, #map3>
// CHECK: "accv.slice"(%arg0, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) {sliceDimensions = [0, 2]} : (memref<5x10x15xf32>, index, index) -> memref<10xf32, #map4>
// CHECK: "accv.slice"(%arg0, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) {sliceDimensions = [1, 2]} : (memref<5x10x15xf32>, index, index) -> memref<5xf32, #map5>
// CHECK: "accv.view"(%arg0, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) {operand_segment_sizes = dense<[1, 3, 3, 3, 0]> : vector<5xi32>} : (memref<5x10x15xf32>, index, index, index, index, index, index, index, index, index) -> memref<3x2x1xf32, #map6>
TEST_CASE("mlir_tensor_view_test")
{
    DeclareFunction("TensorView")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ 5, 10, 15 }) }))
        .Define([](Tensor A) {
            CHECK_NOTHROW(A(1, 2, 3) = 1.0f);

            auto rowSlice = A.Slice(1, Slice::All, Slice::All);
            auto columnSlice = A.Slice(Slice::All, 2, Slice::All);
            auto channelSlice = A.Slice(Slice::All, Slice::All, 3);

            auto rowColumnSlice = A.Slice(1, 2, Slice::All);
            auto rowChannelSlice = A.Slice(1, Slice::All, 3);
            auto columnChannelSlice = A.Slice(Slice::All, 2, 3);

            auto subTensor = A.SubTensor(Scalar(1), Scalar(2), Scalar(3), 3, 2, 1);
        });
}

// CHECK-LABEL: module @mlir_intrinsic_test
// CHECK-NEXT: accv.module "mlir_intrinsic_test" {
// CHECK-NEXT: accv.func nested @intrinsics_float
TEST_CASE("mlir_intrinsic_test")
{
    DeclareFunction("intrinsics_float")
        .Parameters(Value(ValueType::Float, ScalarLayout), Value(ValueType::Float, ScalarLayout))
        .Define([](Scalar x, Scalar y) {
            Scalar acc = MakeScalar<float>();
            acc = x + y;

            // CHECK: math.abs %{{[0-9]+}} : f32
            acc += Abs(acc);
            // CHECK: math.cos %{{[0-9]+}} : f32
            acc += Cos(acc);
            // CHECK: math.exp %{{[0-9]+}} : f32
            acc += Exp(acc);
            // CHECK: math.log %{{[0-9]+}} : f32
            acc += Log(acc);
            // CHECK: math.log10 %{{[0-9]+}} : f32
            acc += Log10(acc);
            // CHECK: math.log2 %{{[0-9]+}} : f32
            acc += Log2(acc);
            // CHECK: math.sin %{{[0-9]+}} : f32
            acc += Sin(acc);
            // CHECK: math.sqrt %{{[0-9]+}} : f32
            acc += Sqrt(acc);
            // CHECK: math.tanh %{{[0-9]+}} : f32
            acc += Tanh(acc);
            // CHECK: math.ceil %{{[0-9]+}} : f32
            acc += Ceil(acc);
            // CHECK: math.powf %{{[0-9]+}}, %arg1 : f32
            acc += Pow(acc, y);

            // CHECK:[[c:%[0-9]+]] = "accv.cmp"(%arg0, %arg1)
            // CHECK: select [[c]], %arg0, %arg1 : f32
            acc += Select(x > y, x, y);

            y = x;
        });
}

// CHECK-LABEL: module @mlir_index_arithmetic_test
// CHECK-NEXT: accv.module "mlir_index_arithmetic_test" {
// CHECK-NEXT:   accv.func nested @IndexArithmetic_{{[0-9]+}}(%arg0: memref<8x18xf32
// CHECK-SAME: %arg1: memref<8x10xf32
// CHECK: "accln.nest"()
// CHECK-DAG: %[[v_j:[0-9]+]] = accln.sym_index {name = "j"} #accln<"index{j
// CHECK-DAG: %[[v_i:[0-9]+]] = accln.sym_index {name = "i"} #accln<"index{i
// CHECK-NEXT: "accln.kernel"
// COM: CHECK-NEXT: %[[v3:[0-9]+]] = "accv.bin_op"(%[[v_i]], %[[v_j]]) {predicate = 0 : i64} : (index, index) -> index
// COM: CHECK-NEXT: %[[v4:[0-9]+]] = memref.subview %arg0[%[[v_i]], %[[v3]]] [1, 1] [18, 1] : memref<8x18xf32, #map0> to memref<f32, #map2>
// COM: CHECK-NEXT: %[[v6:[0-9]+]] = memref.subview %arg1[%[[v_i]], %[[v_j]]] [1, 1] [10, 1] : memref<8x10xf32, #map1> to memref<f32, #map2>
// COM: CHECK-NEXT: %[[v8:[0-9]+]] = "accv.get_element"(%[[v4]]) : (memref<f32, #map2>) -> f32
// COM: CHECK-NEXT: "accv.copy"(%[[v8]], %[[v6]]) : (f32, memref<f32, #map2>) -> ()
// CHECK-NEXT: [[v2:%[0-9]+]] = "accv.bin_op"([[v0]], %[[v1]]) {predicate = 0 : i64} : (index, index) -> index
// CHECK-NEXT: [[v3:%[0-9]+]] = "accv.slice"(%arg0, [[v0]], [[v2]]) {sliceDimensions = [0, 1]} : (memref<8x18xf32>, index, index) -> memref<f32>
// CHECK-NEXT: [[v4:%[0-9]+]] = "accv.slice"(%arg1, [[v0]], %[[v1]]) {sliceDimensions = [0, 1]} : (memref<8x10xf32>, index, index) -> memref<f32>
// CHECK-NEXT: [[v5:%[0-9]+]] = "accv.get_element"([[v3]]) : (memref<f32>) -> f32
// CHECK-NEXT: "accv.copy"([[v5]], [[v4]]) : (f32, memref<f32>) -> ()
TEST_CASE("mlir_index_arithmetic_test")
{
    const int M = 8;
    const int N = 10;

    DeclareFunction("IndexArithmetic")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, M + N }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
        .Define([=](Matrix A, Matrix B) {
            Nest copydiag(MemoryShape{ M, N });
            auto indices = copydiag.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            copydiag.Set([&]() {
                CHECK_NOTHROW(B(i, j) = A(i, i + j));
            });
        });
}

// CHECK-LABEL: module @mlir_scalar_float_test
// CHECK-NEXT: accv.module "mlir_scalar_float_test" {
TEST_CASE("mlir_scalar_float_test")
{
    // CHECK-NEXT: accv.func nested @ScalarFloatTest_{{[0-9]+}}
    DeclareFunction("ScalarFloatTest")
        // CHECK-SAME: %[[IDX:arg0]]: i32
        .Parameters(Value{ ValueType::Int32, ScalarLayout },
                    // CHECK-SAME: %[[A:arg1]]: f32
                    Value{ ValueType::Float, ScalarLayout },
                    // CHECK-SAME: %[[B:arg2]]: memref<10xf32>
                    Value{ ValueType::Float, MemoryLayout{ { 10 } } },
                    // CHECK-SAME: %[[C:arg3]]: memref<100x100xf32
                    Value{ ValueType::Float, MemoryLayout{ { 100, 100 } } },
                    // CHECK-SAME: %[[D:arg4]]: memref<1000x1000x1000xf32
                    Value{ ValueType::Float, MemoryLayout{ { 1000, 1000, 1000 } } },
                    // CHECK-SAME: %[[E:arg5]]: memref<10000x10000x10000x10000xf32
                    Value{ ValueType::Float, MemoryLayout{ { 10000, 10000, 10000, 10000 } } })
        .Define([](Scalar idx, Scalar A, Vector B, Matrix C, Tensor D, Array E) {
            auto c = 2;

            // CHECK-DAG:  %[[CST:cst[0-9_]*]] = arith.constant 2.000000e+00 : f32
            // CHECK-NEXT: %[[v0:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // CHECK-NEXT: %[[v1:[0-9]+]] = "accv.slice"(%[[B]], %[[v0]]) {sliceDimensions = [0]} : (memref<10xf32>, index) -> memref<f32>
            // CHECK-NEXT: %[[v2:[0-9]+]] = "accv.get_element"(%[[v1]]) : (memref<f32>) -> f32
            // CHECK-NEXT: %[[v3:[0-9]+]] = "accv.cmp"(%[[v2]], %[[A]]) {predicate = 1 : i64} : (f32, f32) -> i1
            // CHECK-NEXT: scf.if %[[v3]] {
            If(B[idx] != A, [&] {
                // CHECK-NEXT:  %[[v0:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
                // CHECK-NEXT:  %[[v1:[0-9]+]] = "accv.slice"(%[[B]], %[[v0]]) {sliceDimensions = [0]} : (memref<10xf32>, index) -> memref<f32>
                // CHECK-NEXT:  "accv.copy"(%[[A]], %[[v1]]) : (f32, memref<f32>) -> ()
                B[idx] = A;
                // CHECK-NEXT: }
            });

            // COM: CHECK-NEXT: %[[v0:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // COM: CHECK-NEXT: %[[v1:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // COM: CHECK-NEXT: %[[v2:[0-9]+]] = memref.subview %[[C]][%[[v0]], %[[v1]]] [1, 1] [100, 1] : memref<100x100xf32, #map0> to memref<f32, #map3>
            // COM: CHECK-NEXT: %[[v3:[0-9]+]] = "accv.get_element"(%[[v2]]) : (memref<f32, #map3>) -> f32
            // COM: CHECK-NEXT: %[[v4:[0-9]+]] = "accv.cmp"(%[[v3]], %[[A]]) {predicate = 1 : i64} : (f32, f32) -> i1
            // COM: CHECK-NEXT: scf.if %[[v4]] {
            // CHECK-NEXT: [[v0:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // CHECK-NEXT: [[v1:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // CHECK-NEXT: [[v2:%[0-9]+]] = "accv.slice"(%[[C]], [[v0]], [[v1]]) {sliceDimensions = [0, 1]} : (memref<100x100xf32>, index, index) -> memref<f32>
            // CHECK-NEXT: [[v3:%[0-9]+]] = "accv.get_element"([[v2]]) : (memref<f32>) -> f32
            // CHECK-NEXT: [[v4:%[0-9]+]] = "accv.cmp"([[v3]], %[[A]]) {predicate = 1 : i64} : (f32, f32) -> i1
            // CHECK-NEXT: scf.if [[v4]] {
            If(C(idx, idx) != A, [&] {
                // COM: CHECK-DAG:  %[[CST0:cst[0-9_]*]] = arith.constant 2.000000e+00 : f32
                // COM: CHECK-DAG:  %[[v0:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
                // COM: CHECK-DAG:  %[[Bslice:[0-9]+]] = memref.subview %[[B]][%[[v0]]] [1] [1] : memref<10xf32> to memref<f32, #map3>
                // COM: CHECK-DAG:  %[[v2:[0-9]+]] = "accv.get_element"(%[[Bslice]]) : (memref<f32, #map3>) -> f32
                // COM: CHECK:      %[[v3:[0-9]+]] = "accv.bin_op"(%[[v2]], %[[CST0]]) {predicate = 0 : i64} : (f32, f32) -> f32
                // COM: CHECK-NEXT: %[[v0:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
                // COM: CHECK-NEXT: %[[v1:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
                // COM: CHECK-NEXT: %[[Cslice:[0-9]+]] = memref.subview %[[C]][%[[v0]], %[[v1]]] [1, 1] [100, 1] : memref<100x100xf32, #map0> to memref<f32, #map3>
                // COM: CHECK-NEXT: "accv.copy"(%[[v3]], %[[Cslice]]) : (f32, memref<f32, #map3>) -> ()
                // CHECK-DAG:  [[CST0:%cst[0-9_]*]] = arith.constant 2.000000e+00 : f32
                // CHECK-DAG:  [[v0:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
                // CHECK-DAG:  [[Bslice:%[0-9]+]] = "accv.slice"(%[[B]], [[v0]]) {sliceDimensions = [0]} : (memref<10xf32>, index) -> memref<f32>
                // CHECK-DAG:  [[v2:%[0-9]+]] = "accv.get_element"([[Bslice]]) : (memref<f32>) -> f32
                // CHECK:      [[v3:%[0-9]+]] = "accv.bin_op"([[v2]], [[CST0]]) {predicate = 0 : i64} : (f32, f32) -> f32
                // CHECK-NEXT: [[v0:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
                // CHECK-NEXT: [[v1:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
                // CHECK-NEXT: [[Cslice:%[0-9]+]] = "accv.slice"(%[[C]], [[v0]], [[v1]]) {sliceDimensions = [0, 1]} : (memref<100x100xf32>, index, index) -> memref<f32>
                // CHECK-NEXT: "accv.copy"([[v3]], [[Cslice]]) : (f32, memref<f32>) -> ()
                C(idx, idx) = B[idx] + Cast(c, A.GetType());

                // CHECK-NEXT: }
            });

            // COM: CHECK-NEXT: %[[v0:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // COM: CHECK-NEXT: %[[v1:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // COM: CHECK-NEXT: %[[v2:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // COM: CHECK-NEXT: %[[Dslice:[0-9]+]] = memref.subview %[[D]][%[[v0]], %[[v1]], %[[v2]]] [1, 1, 1] [1000000, 1000, 1] : memref<1000x1000x1000xf32, #map1> to memref<f32, #map3>
            // CHECK-NEXT: [[v0:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // CHECK-NEXT: [[v1:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // CHECK-NEXT: [[v2:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // CHECK-NEXT: [[Dslice:%[0-9]+]] = "accv.slice"(%[[D]], [[v0]], [[v1]], [[v2]]) {sliceDimensions = [0, 1, 2]} : (memref<1000x1000x1000xf32>, index, index, index) -> memref<f32>
            auto dVal = D(idx, idx, idx);

            // CHECK-NEXT: %[[v3:[0-9]+]] = "accv.get_element"([[Dslice]]) : (memref<f32>) -> f32
            // CHECK-NEXT: %[[v4:[0-9]+]] = "accv.cmp"(%[[v3]], %[[A]]) {predicate = 1 : i64} : (f32, f32) -> i1
            // CHECK-NEXT: scf.if %[[v4]] {
            If(dVal != A, [&] {
                // COM: CHECK-NEXT:  %[[c2_0:c2[0-9a-z_]+]] = arith.constant 2 : i32
                // COM: CHECK-NEXT:  %[[v0:[0-9]+]] = "accv.bin_op"(%[[IDX]], %[[c2_0]]) {predicate = 0 : i64} : (i32, i32) -> i32
                // COM: CHECK-DAG:   %[[v1:[0-9]+]] = arith.index_cast %[[v0]] : i32 to index
                // COM: CHECK-DAG:   %[[v2:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
                // COM: CHECK:       %[[v3:[0-9]+]] = memref.subview %[[C]][%[[v1]], %[[v2]]] [1, 1] [100, 1] : memref<100x100xf32, #map0> to memref<f32, #map3>
                // COM: CHECK-NEXT:  %[[v4:[0-9]+]] = "accv.get_element"(%[[v3]]) : (memref<f32, #map3>) -> f32
                // COM: CHECK-NEXT:  "accv.copy"(%[[v4]], %[[Dslice]]) : (f32, memref<f32, #map3>) -> ()
                // CHECK-NEXT:  [[c2_0:%c2[0-9a-z_]+]] = arith.constant 2 : i32
                // CHECK-NEXT:  [[v0:%[0-9]+]] = "accv.bin_op"(%[[IDX]], [[c2_0]]) {predicate = 0 : i64} : (i32, i32) -> i32
                // CHECK-DAG:   [[v1:%[0-9]+]] = arith.index_cast [[v0]] : i32 to index
                // CHECK-DAG:   [[v2:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
                // CHECK:       [[v3:%[0-9]+]] = "accv.slice"(%[[C]], [[v1]], [[v2]]) {sliceDimensions = [0, 1]} : (memref<100x100xf32>, index, index) -> memref<f32>
                // CHECK-NEXT:  [[v4:%[0-9]+]] = "accv.get_element"([[v3]]) : (memref<f32>) -> f32
                // CHECK-NEXT:  "accv.copy"([[v4]], [[Dslice]]) : (f32, memref<f32>) -> ()
                dVal = C(idx + c, idx);

                // CHECK-NEXT: }
            });

            // CHECK-DAG:  %[[v0:[0-9]+]] = "accv.get_element"([[Dslice]]) : (memref<f32>) -> f32
            auto dValCopy = dVal.Copy();

            // CHECK:      %[[v1:[0-9]+]] = "accv.bin_op"(%[[v0]], %[[CST]]) {predicate = 2 : i64} : (f32, f32) -> f32
            dValCopy *= Cast(c, dVal.GetType());

            // COM: CHECK-NEXT: %[[v2:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // COM: CHECK-NEXT: %[[v3:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // COM: CHECK-NEXT: %[[v4:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // COM: CHECK-NEXT: %[[v5:[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // COM: CHECK-NEXT: %[[Eslice:[0-9]+]] = memref.subview %[[E]][%[[v2]], %[[v3]], %[[v4]], %[[v5]]] [1, 1, 1, 1] [1000000000000, 100000000, 10000, 1] : memref<10000x10000x10000x10000xf32, #map2> to memref<f32, #map3>
            // CHECK-NEXT: [[v2:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // CHECK-NEXT: [[v3:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // CHECK-NEXT: [[v4:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // CHECK-NEXT: [[v5:%[0-9]+]] = arith.index_cast %[[IDX]] : i32 to index
            // CHECK-NEXT: [[Eslice:%[0-9]+]] = "accv.slice"(%[[E]], [[v2]], [[v3]], [[v4]], [[v5]]) {sliceDimensions = [0, 1, 2, 3]} : (memref<10000x10000x10000x10000xf32>, index, index, index, index) -> memref<f32>
            auto eVal = E(idx, idx, idx, idx);

            // CHECK-NEXT: %[[v7:[0-9]+]] = "accv.get_element"([[Eslice]]) : (memref<f32>) -> f32
            // CHECK-NEXT: %[[v8:[0-9]+]] = "accv.cmp"(%[[v7]], %[[A]]) {predicate = 1 : i64} : (f32, f32) -> i1
            // CHECK-NEXT: scf.if %[[v8]] {
            If(eVal != A, [&] {
                // CHECK-NEXT:  "accv.copy"(%[[v1]], [[Eslice]]) : (f32, memref<f32>) -> ()
                eVal = dValCopy;

                // CHECK-NEXT: }
            });
        });

    SUCCEED();
}

// CHECK-LABEL: module @jit_print_scalar_test
// CHECK-NEXT: accv.module "jit_print_scalar_test" {
// CHECK: "accv.print"
// JIT-LABEL: @jit_print_scalar_test
TEST_CASE("jit_print_scalar_test")
{
    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Scalar iVal = MakeScalar<int>("i");
            Scalar fVal = MakeScalar<float>("f");
            Scalar dVal = MakeScalar<double>("d");
            iVal = 1;
            fVal = 3.14f;
            dVal = 1.1;

            // JIT-LABEL: i:
            Print("i: "s);
            // JIT-SAME: 1
            Print(iVal);
            Print("\n"s);

            // JIT-LABEL: f:
            Print("f: "s);
            // JIT-SAME: 3.14
            Print(fVal);
            Print("\n"s);

            // JIT-LABEL: d:
            Print("d: "s);
            // JIT-SAME: 1.1
            Print(dVal);
            Print("\n"s);
        });
}

// CHECK-LABEL: module @jit_matrix_test
// CHECK-NEXT: accv.module "jit_matrix_test" {
// CHECK: "accv.print"
// JIT-LABEL: @jit_matrix_test
TEST_CASE("jit_matrix_test")
{
    const int M = 3;
    const int N = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix m = MakeMatrix<float>(M, N);
            Nest fillNest(MemoryShape{ M, N });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            fillNest.Set([&]() {
                CHECK_NOTHROW(m(i, j) = 3.14f);
            });
            fillNest.CreateSchedule();

            Print(m);
            // JIT: 3.140000 3.140000 3.140000 3.140000
            // JIT-NEXT: 3.140000 3.140000 3.140000 3.140000
            // JIT-NEXT: 3.140000 3.140000 3.140000 3.140000
        });
}

// CHECK-LABEL: module @jit_matrix_slice_test
// CHECK-NEXT: accv.module "jit_matrix_slice_test" {
// CHECK: "accv.print"
// JIT-LABEL: @jit_matrix_slice_test
TEST_CASE("jit_matrix_slice_test")
{
    const int M = 3;
    const int N = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix m = MakeMatrix<float>(M, N);

            Nest fillNest(MemoryShape{ M, N });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            fillNest.Set([&]() {
                m(i, j) = 0.0f;
            });
            fillNest.CreateSchedule();

            For(Scalar(0), Scalar(M), Scalar(1), [&](Scalar i) {
                m(i, 0) = Scalar(1.0f);

                auto r = m.Row(i);
                r(1) = Scalar(2.0f);
            });

            Print(m);
            // JIT: 1.000000 2.000000 0.000000 0.000000
            // JIT-NEXT: 1.000000 2.000000 0.000000 0.000000
            // JIT-NEXT: 1.000000 2.000000 0.000000 0.000000
        });
}

// CHECK-LABEL: module @jit_matrix_view_test
// CHECK-NEXT: accv.module "jit_matrix_view_test" {
// CHECK: "accv.print"
// JIT-LABEL: @jit_matrix_view_test

TEST_CASE("jit_matrix_view_test")
{
    const int M = 3;
    const int N = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix m = MakeMatrix<float>(M, N);

            Nest fillNest(MemoryShape{ M, N });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            fillNest.Set([&]() {
                m(i, j) = 0.0f;
            });
            fillNest.CreateSchedule();

            auto subview = m.SubMatrix(Scalar(1), Scalar(1), 2, 2);
            subview(1, 1) = 2.0f;

            Print(m);
            // JIT: 0.000000 0.000000 0.000000 0.000000
            // JIT-NEXT: 0.000000 0.000000 0.000000 0.000000
            // JIT-NEXT: 0.000000 0.000000 2.000000 0.000000
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_row_sum_test
// CHECK-NEXT: accv.module "jit_row_sum_test" {
// JIT-LABEL: @jit_row_sum_test

TEST_CASE("jit_row_sum_test")
{
    const int M = 4;
    const int N = 5;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Array A = MakeArray<float>({ M, N }, "A");
            Array Asum = MakeArray<float>({ M }, "Asum");

            {
                // Easy but inefficient method to set all values (but should be amenable to optimization)
                Nest fillNest(MemoryShape{ M, N });
                auto [i, j] = fillNest.GetIndices<2>();
                fillNest.Set([&, i = i, j = j]() {
                    auto iVal = Scalar(Cast(i, ValueType::Int32));
                    auto jVal = Scalar(Cast(j, ValueType::Int32));

                    A(i, j) = Scalar(Cast((iVal * 10) + jVal, ValueType::Float));
                    Asum(i) = Scalar(0.0f);
                });
                fillNest.CreateSchedule();
            }

            {
                Nest sumNest(MemoryShape{ M, N });
                auto [i, j] = sumNest.GetIndices<2>();
                sumNest.Set([&, i = i, j = j]() {
                    Asum(i) += A(i, j);
                });
                auto schedule = sumNest.CreateSchedule();
                auto [iOuter, iInner] = schedule.Split(i, 2);
                auto plan = schedule.CreatePlan();

                schedule.SetOrder({ iOuter, j, iInner });
                plan.AddCache(Asum, j);

                // JIT-LABEL: Asum:
                Print("Asum:\n"s);
                // JIT: 10.000000 60.000000 110.000000 160.000000
                Print(Asum);
            }
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_row_product_test
// CHECK-NEXT: accv.module "jit_row_product_test" {
// JIT-LABEL: @jit_row_product_test

TEST_CASE("jit_row_product_test")
{
    const int M = 4;
    const int N = 5;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Array A = MakeArray<float>({ M, N }, "A");
            Array Aprod = MakeArray<float>({ M }, "Aprod");

            {
                // Easy but inefficient method to set all values (but should be amenable to optimization)
                Nest fillNest(MemoryShape{ M, N });
                auto [i, j] = fillNest.GetIndices<2>();
                fillNest.Set([&, i = i, j = j]() {
                    auto iVal = Scalar(Cast(i, ValueType::Int32));
                    auto jVal = Scalar(Cast(j, ValueType::Int32));

                    A(i, j) = Scalar(Cast(iVal + jVal, ValueType::Float));
                    Aprod(i) = Scalar(1.0f);
                });
                fillNest.CreateSchedule();
            }

            {
                Nest prodNest(MemoryShape{ M, N });
                auto [i, j] = prodNest.GetIndices<2>();
                prodNest.Set([&, i = i, j = j]() {
                    Aprod(i) *= A(i, j);
                });
                auto schedule = prodNest.CreateSchedule();
                auto [iOuter, iInner] = schedule.Split(i, 2);
                auto plan = schedule.CreatePlan();

                schedule.SetOrder({ iOuter, j, iInner });
                // plan.AddCache(Aprod, j); // caching currently doesn't work for this example

                // JIT-LABEL: Aprod:
                Print("Aprod:\n"s);
                // JIT: 0.000000 120.000000 720.000000 2520.000000
                Print(Aprod);
            }
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_array_view_test1
// CHECK-NEXT: accv.module "jit_array_view_test1" {
// CHECK: "accv.print"
// JIT-LABEL: @jit_array_view_test1
TEST_CASE("jit_array_view_test1")
{
    const int M = 3;
    const int N = 4;
    const int K = 5;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Array arr = MakeArray({ M, N, K }, ValueType::Int32);

            Nest fillNest(MemoryShape{ M, N, K });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                arr(i, j, k) = (iVal * 100) + (jVal * 10) + kVal;
            });
            fillNest.CreateSchedule();

            // No offset
            auto subview = arr.SubArray({ Scalar(0), Scalar(0), Scalar(0) }, { 2, 2, 2 });
            subview(0, 0, 0) = 1000;
            subview(0, 0, 1) = 2000;
            subview(0, 1, 0) = 3000;
            subview(0, 1, 1) = 4000;

            // JIT: 1000 2000 2 3 4
            // JIT-NEXT: 3000 4000 12 13 14
            // JIT-NEXT: 20 21 22 23 24
            // JIT-NEXT: 30 31 32 33 34
            // JIT: 100 101 102 103 104
            // JIT-NEXT: 110 111 112 113 114
            // JIT-NEXT: 120 121 122 123 124
            // JIT-NEXT: 130 131 132 133 134
            // JIT: 200 201 202 203 204
            // JIT-NEXT: 210 211 212 213 214
            // JIT-NEXT: 220 221 222 223 224
            // JIT-NEXT: 230 231 232 233 234
            // JIT: 1000 2000
            // JIT-NEXT: 3000 4000
            // JIT: 100 101
            // JIT-NEXT: 110 111
            Print(arr);
            Print(subview);
        });
}

// CHECK-LABEL: module @jit_array_view_test2
// CHECK-NEXT: accv.module "jit_array_view_test2" {
// CHECK: "accv.print"
// JIT-LABEL: @jit_array_view_test2
TEST_CASE("jit_array_view_test2")
{
    const int M = 3;
    const int N = 4;
    const int K = 5;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Array arr = MakeArray({ M, N, K }, ValueType::Int32);

            Nest fillNest(MemoryShape{ M, N, K });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                arr(i, j, k) = (iVal * 100) + (jVal * 10) + kVal;
            });
            fillNest.CreateSchedule();

            // Offset by (1,1,1)
            auto subview = arr.SubArray({ Scalar(1), Scalar(1), Scalar(1) }, { 2, 2, 2 });
            subview(0, 0, 0) = 1000;
            subview(0, 0, 1) = 2000;
            subview(0, 1, 0) = 3000;
            subview(0, 1, 1) = 4000;

            Print(arr);
            Print(subview);
        });
}

// CHECK-LABEL: module @jit_array_view_test3
// CHECK-NEXT: accv.module "jit_array_view_test3" {
// JIT-LABEL: @jit_array_view_test3
TEST_CASE("jit_array_view_test3")
{
    const int M = 2;
    const int N = 3;
    const int K = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Array arr = MakeArray({ M, N, K }, ValueType::Int32, "arr");

            Nest fillNest(MemoryShape{ M, N, K });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                arr(i, j, k) = (iVal * 100) + (jVal * 10) + kVal;
            });
            fillNest.CreateSchedule();

            auto slice0 = arr.Slice({ 0 }, { Scalar(0) });
            auto slice1 = slice0.Slice({ 0 }, { Scalar(0) });

            // Print(arr);
            Print(slice0);
        });
}

// COM: CHECK-LABEL: module @jit_matrix_multiply_test
// COM: CHECK-NEXT: accv.module "jit_matrix_multiply_test" {
// COM: JIT-LABEL: @jit_matrix_multiply_test
TEST_CASE("jit_matrix_multiply_test")
{
    const int M = 3;
    const int N = 4;
    const int K = 5;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix A = MakeMatrix<float>(M, K);
            Matrix At = MakeMatrix<float>(K, M);
            auto Att = At.TransposedView();

            Matrix B = MakeMatrix<float>(K, N);
            Matrix Bt = MakeMatrix<float>(N, K);
            auto Btt = Bt.TransposedView();

            Matrix AB = MakeMatrix<float>(M, N);
            Matrix AttB = MakeMatrix<float>(M, N);
            Matrix ABtt = MakeMatrix<float>(M, N);
            Matrix AttBtt = MakeMatrix<float>(M, N);

            Matrix AB_t = MakeMatrix<float>(N, M);
            Matrix AttB_t = MakeMatrix<float>(N, M);
            Matrix ABtt_t = MakeMatrix<float>(N, M);
            Matrix AttBtt_t = MakeMatrix<float>(N, M);

            auto AB_tt = AB_t.TransposedView();
            auto AttB_tt = AttB_t.TransposedView();
            auto ABtt_tt = ABtt_t.TransposedView();
            auto AttBtt_tt = AttBtt_t.TransposedView();

            // Easy but inefficient method to set all values (but should be amenable to optimization)
            Nest fillNest(MemoryShape{ M, N, K });
            Scalar i, j, k;
            std::tie(i, j, k) = fillNest.GetIndices<3>();
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));

                A(i, k) = Scalar(Cast((iVal * 10) + kVal, ValueType::Float));
                At(k, i) = Scalar(Cast((iVal * 10) + kVal, ValueType::Float));
                B(k, j) = Scalar(Cast((kVal * 10) + jVal, ValueType::Float));
                Bt(j, k) = Scalar(Cast((kVal * 10) + jVal, ValueType::Float));
                AB(i, j) = 0.0f;
                AttB(i, j) = 0.0f;
                ABtt(i, j) = 0.0f;
                AttBtt(i, j) = 0.0f;
                AB_t(j, i) = 0.0f;
                AttB_t(j, i) = 0.0f;
                ABtt_t(j, i) = 0.0f;
                AttBtt_t(j, i) = 0.0f;
            });
            fillNest.CreateSchedule();

            // COM: JIT-LABEL: A*B:
            Print("A*B:\n"s);
            // COM: JIT: 300.000000 310.000000 320.000000 330.000000
            // COM: JIT-NEXT: 1300.000000 1360.000000 1420.000000 1480.000000
            // COM: JIT-NEXT: 2300.000000 2410.000000 2520.000000 2630.000000
            MatrixMatrixMultiply(A, B, AB);
            Print(AB);

            // COM: JIT-LABEL: Att*B:
            Print("Att*B:\n"s);
            // COM: JIT: 300.000000 310.000000 320.000000 330.000000
            // COM: JIT-NEXT: 1300.000000 1360.000000 1420.000000 1480.000000
            // COM: JIT-NEXT: 2300.000000 2410.000000 2520.000000 2630.000000
            MatrixMatrixMultiply(Att, B, AttB);
            Print(AttB);

            // COM: JIT-LABEL: A*Btt:
            Print("A*Btt:\n"s);
            // COM: JIT: 300.000000 310.000000 320.000000 330.000000
            // COM: JIT-NEXT: 1300.000000 1360.000000 1420.000000 1480.000000
            // COM: JIT-NEXT: 2300.000000 2410.000000 2520.000000 2630.000000
            MatrixMatrixMultiply(A, Btt, ABtt);
            Print(ABtt);

            // COM: JIT-LABEL: Att*Btt:
            Print("Att*Btt:\n"s);
            // COM: JIT: 300.000000 310.000000 320.000000 330.000000
            // COM: JIT-NEXT: 1300.000000 1360.000000 1420.000000 1480.000000
            // COM: JIT-NEXT: 2300.000000 2410.000000 2520.000000 2630.000000
            MatrixMatrixMultiply(Att, Btt, AttBtt);
            Print(AttBtt);

            //
            // Cases with a transposed result matrix:
            //

            // COM: JIT-LABEL: A*B_t:
            Print("A*B_t:\n"s);
            // COM: JIT: 300.000000 1300.000000 2300.000000
            // COM: JIT-NEXT: 310.000000 1360.000000 2410.000000
            // COM: JIT-NEXT: 320.000000 1420.000000 2520.000000
            // COM: JIT-NEXT: 330.000000 1480.000000 2630.000000
            MatrixMatrixMultiply(A, B, AB_tt);
            Print(AB_t);

            // COM: JIT-LABEL: Att*B_t:
            Print("Att*B_t:\n"s);
            // COM: JIT: 300.000000 1300.000000 2300.000000
            // COM: JIT-NEXT: 310.000000 1360.000000 2410.000000
            // COM: JIT-NEXT: 320.000000 1420.000000 2520.000000
            // COM: JIT-NEXT: 330.000000 1480.000000 2630.000000
            MatrixMatrixMultiply(Att, B, AttB_tt);
            Print(AttB_t);

            // COM: JIT-LABEL: A*Btt_t:
            Print("A*Btt_t:\n"s);
            // COM: JIT: 300.000000 1300.000000 2300.000000
            // COM: JIT-NEXT: 310.000000 1360.000000 2410.000000
            // COM: JIT-NEXT: 320.000000 1420.000000 2520.000000
            // COM: JIT-NEXT: 330.000000 1480.000000 2630.000000
            MatrixMatrixMultiply(A, Btt, ABtt_tt);
            Print(ABtt_t);

            // COM: JIT-LABEL: Att*Btt_t:
            Print("Att*Btt_t:\n"s);
            // COM: JIT: 300.000000 1300.000000 2300.000000
            // COM: JIT-NEXT: 310.000000 1360.000000 2410.000000
            // COM: JIT-NEXT: 320.000000 1420.000000 2520.000000
            // COM: JIT-NEXT: 330.000000 1480.000000 2630.000000
            MatrixMatrixMultiply(Att, Btt, AttBtt_tt);
            Print(AttBtt_t);
        });
    SUCCEED();
}
// CHECK-LABEL: module @jit_int8_matrix_multiply_test
// CHECK-NEXT: accv.module "jit_int8_matrix_multiply_test" {
// JIT-LABEL: @jit_int8_matrix_multiply_test

TEST_CASE("jit_int8_matrix_multiply_test")
{
    const int M = 3;
    const int N = 4;
    const int K = 5;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix A = MakeMatrix<int8_t>(M, K);
            Matrix B = MakeMatrix<int8_t>(K, N);
            Matrix AB = MakeMatrix<int8_t>(M, N);

            // Easy but inefficient method to set all values (but should be amenable to optimization)
            Nest fillNest(MemoryShape{ M, N, K });
            auto [i, j, k] = fillNest.GetIndices<3>();
            fillNest.Set([&, i = i, j = j, k = k]() {
                auto iVal = Scalar(Cast(i, ValueType::Int8));
                auto jVal = Scalar(Cast(j, ValueType::Int8));
                auto kVal = Scalar(Cast(k, ValueType::Int8));

                A(i, k) = Scalar(Cast(iVal + kVal, ValueType::Int8));
                B(k, j) = Scalar(Cast((kVal * Cast(2, ValueType::Int8)) + jVal, ValueType::Int8));
                AB(i, j) = Scalar(Cast(0, ValueType::Int8));
            });
            fillNest.CreateSchedule();

            // JIT-LABEL: A*B:
            Print("A*B:\n"s);
            // JIT:60 70 80 90
            // JIT-NEXT:80 95 110 125
            // JIT-NEXT:100 120 140 160
            MatrixMatrixMultiply(A, B, AB);
            Print(AB);
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_int8_simple_matrix_multiply_test1
// CHECK-NEXT: accv.module "jit_int8_simple_matrix_multiply_test1" {
// JIT-LABEL: @jit_int8_simple_matrix_multiply_test1

TEST_CASE("jit_int8_simple_matrix_multiply_test1")
{
    const int M = 4;
    const int N = 16;
    const int K = 4;

    [[maybe_unused]] const int kernelM = 4;
    [[maybe_unused]] const int kernelN = 8;
    const int kernelK = 4;
    const int innerKernelK = 2;

    auto aType = ValueType::Int8;
    auto bType = ValueType::Int8;
    auto cType = ValueType::Int32;
    [[maybe_unused]] auto i16 = ValueType::Int16; // signed
    auto i32 = ValueType::Int32; // signed

    auto ind = [](auto i) { return Scalar(Cast(i, ValueType::Index)); };

    auto fillFn = DeclareFunction("fill")
                      .Public(true)
                      .Decorated(false)
                      .Parameters(Value{ aType, MemoryLayout{ { M, K } } }, // A
                                  Value{ bType, MemoryLayout{ { K, N } } }, // B
                                  Value{ cType, MemoryLayout{ { M, N } } }) // C
                      .Inlined(FunctionInlining::never)
                      .Define([&](Array A, Array B, Array C) {
                          Nest fillNest(MemoryShape{ M, N, K });
                          auto [ii, jj, kk] = fillNest.GetIndices<3>();
                          fillNest.Set([&, i = ii, j = jj, k = kk]() {
                              auto iVal = Scalar(Cast(i, ValueType::Int8));
                              auto jVal = Scalar(Cast(j, ValueType::Int8));
                              auto kVal = Scalar(Cast(k, ValueType::Int8));

                              A(i, k) = Scalar(Cast(iVal + kVal, aType));
                              B(k, j) = Scalar(Cast((kVal * Cast(2, bType)) + jVal, bType));
                              C(i, j) = Scalar(Cast(0, cType));
                          });
                          fillNest.CreateSchedule();
                      });

    auto computeFn = DeclareFunction("compute")
                         .Public(true)
                         .Decorated(false)
                         .Parameters(Value{ aType, MemoryLayout{ { M, K } } }, // A
                                     Value{ bType, MemoryLayout{ { K, N } } }, // B
                                     Value{ cType, MemoryLayout{ { M, N } } }) // C
                         .Inlined(FunctionInlining::never)
                         .Define([&](Array A, Array B, Array C) {
                             auto sums = MakeArray({ innerKernelK }, i32, AllocateFlags::Stack);
                             auto minI16Val = Cast(Scalar(-32768), cType);
                             auto maxI16Val = Cast(Scalar(32767), cType);

                             {
                                 Nest computeNest(MemoryShape{ M, N, K });
                                 auto [i, j, k] = computeNest.GetIndices<3>();
                                 auto schedule = computeNest.CreateSchedule();
                                 auto [kOuter, kInner] = schedule.Split(k, kernelK);
                                 auto [kOuter2, kInner2] = schedule.Split(kInner, innerKernelK);

                                 auto clearAccumKernel = Kernel("clearAccum", [&, i = i]() {
                                     for (int k = 0; k < innerKernelK; ++k)
                                     {
                                         sums(k) = Cast(0, i32);
                                     }
                                 });

                                 auto computeKernel = Kernel("compute", [&, i = i, j = j, k = k]() {
                                     auto a = Cast(A(i, k), cType);
                                     auto b = Cast(B(k, j), cType);
                                     auto prod = a * b;

                                     auto kInnerIndex = k % ind(kernelK);
                                     auto divisor = kernelK / innerKernelK;
                                     //  auto kSumIndex = kInnerIndex / ind(divisor);
                                     auto kSumIndex = kInnerIndex / ind(divisor);
                                     sums(kSumIndex) += prod;
                                 });

                                 auto accumKernel = Kernel("accum", [&, i = i, j = j]() {
                                     for (int k = 0; k < innerKernelK; ++k)
                                     {
                                         Scalar sum = sums(k);
                                         auto satVal = Clamp(sum, minI16Val, maxI16Val);
                                         C(i, j) += satVal;
                                     }
                                 });

                                 schedule.AddKernel(clearAccumKernel, First(kInner));
                                 schedule.AddKernel(computeKernel);
                                 schedule.AddKernel(accumKernel, Last(kInner));
                                 schedule.SetOrder({ kOuter, i, j, kOuter2, kInner2 });

                                 auto plan = schedule.CreatePlan();
                                 plan.AddCache(B, j);
                             }
                         });

    auto accessFn = DeclareFunction("access")
                        .Public(true)
                        .External(true)
                        .Decorated(false)
                        .Inlined(FunctionInlining::never)
                        .Parameters(Value{ cType, MemoryLayout{ { M, N } } })
                        .Define([&](Array result) {
                            Print(result);
                        });

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto A = MakeArray({ M, K }, aType, AllocateFlags::Stack);
            auto B = MakeArray({ K, N }, bType, AllocateFlags::Stack);
            auto C = MakeArray({ M, N }, cType, AllocateFlags::Stack);

            fillFn(A.GetValue(), B.GetValue(), C.GetValue());
            computeFn(A.GetValue(), B.GetValue(), C.GetValue());
            accessFn(C.GetValue());

            Print("\nA:\n"s);
            Print(A);
            Print("\nB:\n"s);
            Print(B);
            Print("\nResult:\n"s);
            Print(C);

            // JIT: 28 34 40 46 52 58 64 70 76 82 88 94 100 106 112 118
            // JIT-NEXT: 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190
            // JIT-NEXT: 52 66 80 94 108 122 136 150 164 178 192 206 220 234 248 262
            // JIT-NEXT: 64 82 100 118 136 154 172 190 208 226 244 262 280 298 316 334
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_int8_simple_matrix_multiply_test3
// CHECK-NEXT: accv.module "jit_int8_simple_matrix_multiply_test3" {
// JIT-LABEL: @jit_int8_simple_matrix_multiply_test3

TEST_CASE("jit_int8_simple_matrix_multiply_test3")
{
    const int M = 4;
    const int N = 16;
    const int K = 4;

    [[maybe_unused]] const int kernelM = 4;
    [[maybe_unused]] const int kernelN = 8;
    const int kernelK = 4;

    auto aType = ValueType::Int8;
    auto bType = ValueType::Int8;
    auto cType = ValueType::Int32;
    [[maybe_unused]] auto i16 = ValueType::Int16; // signed
    [[maybe_unused]] auto i32 = ValueType::Int32; // signed

    [[maybe_unused]] auto ind = [](auto i) { return Scalar(Cast(i, ValueType::Index)); };

    auto fillFn = DeclareFunction("fill")
                      .Public(true)
                      .Decorated(false)
                      .Parameters(Value{ aType, MemoryLayout{ { M, K } } }, // A
                                  Value{ bType, MemoryLayout{ { K, N } } }, // B
                                  Value{ cType, MemoryLayout{ { M, N } } }) // C
                      .Inlined(FunctionInlining::never)
                      .Define([&](Array A, Array B, Array C) {
                          Nest fillNest(MemoryShape{ M, N, K });
                          auto [ii, jj, kk] = fillNest.GetIndices<3>();
                          fillNest.Set([&, i = ii, j = jj, k = kk]() {
                              auto iVal = Scalar(Cast(i, ValueType::Int8));
                              auto jVal = Scalar(Cast(j, ValueType::Int8));
                              auto kVal = Scalar(Cast(k, ValueType::Int8));

                              A(i, k) = Scalar(Cast(iVal + kVal, aType));
                              B(k, j) = Scalar(Cast((kVal * Cast(2, bType)) + jVal, bType));
                              C(i, j) = Scalar(Cast(0, cType));
                          });
                          fillNest.CreateSchedule();
                      });

    auto computeFn = DeclareFunction("compute")
                         .Public(true)
                         .Decorated(false)
                         .Parameters(Value{ aType, MemoryLayout{ { M, K } } }, // A
                                     Value{ bType, MemoryLayout{ { K, N } } }, // B
                                     Value{ cType, MemoryLayout{ { M, N } } }) // C
                         .Inlined(FunctionInlining::never)
                         .Define([&](Array A, Array B, Array C) {
                             Nest computeNest(MemoryShape{ M, N, K });
                             auto [i, j, k] = computeNest.GetIndices<3>();
                             auto schedule = computeNest.CreateSchedule();
                             auto [kOuter, kInner] = schedule.Split(k, kernelK);

                             auto computeKernel = Kernel("compute", [&, i = i, j = j, k = k]() {
                                 auto a = Cast(A(i, k), cType);
                                 auto b = Cast(B(k, j), cType);
                                 auto prod = a * b;
                                 C(i, j) += prod;
                             });

                             schedule.AddKernel(computeKernel);
                             schedule.SetOrder({ kOuter, i, j, kInner });
                             schedule.SetSaturatedFlag(kInner);

                             auto plan = schedule.CreatePlan();
                             plan.AddCache(B, j);
                         });

    auto accessFn = DeclareFunction("access")
                        .Public(true)
                        .External(true)
                        .Decorated(false)
                        .Inlined(FunctionInlining::never)
                        .Parameters(Value{ cType, MemoryLayout{ { M, N } } })
                        .Define([&](Array result) {
                            Print(result);
                        });

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto A = MakeArray({ M, K }, aType, AllocateFlags::Stack);
            auto B = MakeArray({ K, N }, bType, AllocateFlags::Stack);
            auto C = MakeArray({ M, N }, cType, AllocateFlags::Stack);

            fillFn(A.GetValue(), B.GetValue(), C.GetValue());
            computeFn(A.GetValue(), B.GetValue(), C.GetValue());
            accessFn(C.GetValue());

            Print("\nA:\n"s);
            Print(A);
            Print("\nB:\n"s);
            Print(B);
            Print("\nResult:\n"s);
            Print(C);

            // JIT: 28 34 40 46 52 58 64 70 76 82 88 94 100 106 112 118
            // JIT-NEXT: 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190
            // JIT-NEXT: 52 66 80 94 108 122 136 150 164 178 192 206 220 234 248 262
            // JIT-NEXT: 64 82 100 118 136 154 172 190 208 226 244 262 280 298 316 334
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_int8_cached_matrix_multiply_test
// CHECK-NEXT: accv.module "jit_int8_cached_matrix_multiply_test" {
// JIT-LABEL: @jit_int8_cached_matrix_multiply_test

TEST_CASE("jit_int8_cached_matrix_multiply_test")
{
    const int M = 3;
    const int N = 4;
    const int K = 5;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix A = MakeMatrix<int8_t>(M, K);
            Matrix B = MakeMatrix<int8_t>(K, N);
            Matrix C = MakeMatrix<int8_t>(M, N);

            // Easy but inefficient method to set all values (but should be amenable to optimization)
            Nest fillNest(MemoryShape{ M, N, K });
            auto [ii, jj, kk] = fillNest.GetIndices<3>();
            fillNest.Set([&, ii = ii, jj = jj, kk = kk]() {
                auto iVal = Scalar(Cast(ii, ValueType::Int8));
                auto jVal = Scalar(Cast(jj, ValueType::Int8));
                auto kVal = Scalar(Cast(kk, ValueType::Int8));

                A(ii, kk) = Scalar(Cast(iVal + kVal, ValueType::Int8));
                B(kk, jj) = Scalar(Cast((kVal * Cast(2, ValueType::Int8)) + jVal, ValueType::Int8));
                C(ii, jj) = Scalar(Cast(0, ValueType::Int8));
            });
            fillNest.CreateSchedule();

            auto simpleMatMul = [](Array A, Array B, Array C) -> Nest {
                auto M = A.Shape()[0];
                auto N = B.Shape()[1];
                auto K = A.Shape()[1];
                Nest nest(MemoryShape{ M, N, K });
                Scalar i, j, k;
                std::tie(i, j, k) = nest.GetIndices<3>();
                nest.Set([&]() {
                    C(i, j) += A(i, k) * B(k, j);
                });
                return nest;
            };

            const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            const int vectorBytes = vectorSize * 4; // 4 bytes per float
            const int vectorUnits = 16; // AVX-2 has 16 256-bit registers

            auto matMulNest = simpleMatMul(A, B, C);
            auto [i, j, k] = matMulNest.GetIndices<3>();
            auto schedule = matMulNest.CreateSchedule();
            auto [iOuter, iInner] = schedule.Split(i, 6);
            auto [jOuter, jInner] = schedule.Split(j, vectorSize);
            schedule.SetOrder({ jOuter, iOuter, iInner, k, jInner });
            auto plan = schedule.CreatePlan();
            plan.AddCache(C, iInner);

            schedule.Unroll(iInner);
            schedule.Unroll(jInner);
            plan.Vectorize(jInner, { vectorBytes, vectorUnits });

            // JIT-LABEL: A*B:
            Print("A*B:\n"s);
            // JIT:60 70 80 90
            // JIT-NEXT:80 95 110 125
            // JIT-NEXT:100 120 140 160
            Print(C);
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_int8_expvec_matrix_multiply_test
// CHECK-NEXT: accv.module "jit_int8_expvec_matrix_multiply_test" {
// JIT-LABEL: @jit_int8_expvec_matrix_multiply_test
TEST_CASE("jit_int8_expvec_matrix_multiply_test")
{
    const int M = 32;
    const int N = 32;
    const int K = 32;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix A = MakeMatrix<int8_t>(M, K);
            Matrix B = MakeMatrix<int8_t>(K, N);
            Matrix C = MakeMatrix<int8_t>(M, N);

            // Easy but inefficient method to set all values (but should be amenable to optimization)
            Nest fillNest(MemoryShape{ M, N, K });
            auto [ii, jj, kk] = fillNest.GetIndices<3>();
            fillNest.Set([&, i = ii, j = jj, k = kk]() {
                auto iVal = Scalar(Cast(i, ValueType::Int8));
                auto jVal = Scalar(Cast(j, ValueType::Int8));
                auto kVal = Scalar(Cast(k, ValueType::Int8));

                A(i, k) = Scalar(Cast(iVal + kVal, ValueType::Int8));
                B(k, j) = Scalar(Cast((kVal * Cast(2, ValueType::Int8)) + jVal, ValueType::Int8));
                C(i, j) = Scalar(Cast(0, ValueType::Int8));
            });
            fillNest.CreateSchedule();

            auto simpleMatMul = [](Array A, Array B, Array C) -> Nest {
                auto M = A.Shape()[0];
                auto N = B.Shape()[1];
                auto K = A.Shape()[1];
                Nest nest(MemoryShape{ M, N, K });
                Scalar i, j, k;
                std::tie(i, j, k) = nest.GetIndices<3>();
                nest.Set([&]() {
                    C(i, j) += A(i, k) * B(k, j);
                });
                return nest;
            };

            const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            const int vectorBytes = vectorSize * 4; // 4 bytes per float
            const int vectorUnits = 16; // AVX-2 has 16 256-bit registers

            auto matMulNest = simpleMatMul(A, B, C);
            auto [i, j, k] = matMulNest.GetIndices<3>();
            auto schedule = matMulNest.CreateSchedule();
            auto [iOuter, iInner] = schedule.Split(i, 6);
            auto [jOuter, jInner] = schedule.Split(j, vectorSize);
            schedule.SetOrder({ jOuter, iOuter, iInner, k, jInner });
            auto plan = schedule.CreatePlan();
            // plan.AddCache(C, iInner);

            schedule.Unroll(iInner);
            schedule.Unroll(jInner);
            plan.Vectorize(jInner, { vectorBytes, vectorUnits, true });

            // JIT-LABEL: A*B:
            Print("A*B:\n"s);
            // JIT:96 80 64 48 32 16
            // JIT-NEXT:64 80 96 112 128
            // JIT-NEXT:32 80 128 176 224
            Print(C);
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_int16_matrix_multiply_test
// CHECK-NEXT: accv.module "jit_int16_matrix_multiply_test" {
// JIT-LABEL: @jit_int16_matrix_multiply_test

TEST_CASE("jit_int16_matrix_multiply_test")
{
    const int M = 3;
    const int N = 4;
    const int K = 5;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix A = MakeMatrix<int16_t>(M, K);
            Matrix B = MakeMatrix<int16_t>(K, N);
            Matrix AB = MakeMatrix<int16_t>(M, N);

            // Easy but inefficient method to set all values (but should be amenable to optimization)
            Nest fillNest(MemoryShape{ M, N, K });
            auto [i, j, k] = fillNest.GetIndices<3>();
            fillNest.Set([&, i = i, j = j, k = k]() {
                auto iVal = Scalar(Cast(i, ValueType::Int16));
                auto jVal = Scalar(Cast(j, ValueType::Int16));
                auto kVal = Scalar(Cast(k, ValueType::Int16));

                A(i, k) = Scalar(Cast((iVal * Cast(10, ValueType::Int16)) + kVal, ValueType::Int16));
                B(k, j) = Scalar(Cast((kVal * Cast(10, ValueType::Int16)) + jVal, ValueType::Int16));
                AB(i, j) = Scalar(Cast(0, ValueType::Int16));
            });
            fillNest.CreateSchedule();

            // JIT-LABEL: A*B:
            Print("A*B:\n"s);
            // JIT: 300 310 320 330
            // JIT-NEXT: 1300 1360 1420 1480
            // JIT-NEXT: 2300 2410 2520 2630
            MatrixMatrixMultiply(A, B, AB);
            Print(AB);
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_int32_matrix_multiply_test
// CHECK-NEXT: accv.module "jit_int32_matrix_multiply_test" {
// JIT-LABEL: @jit_int32_matrix_multiply_test

TEST_CASE("jit_int32_matrix_multiply_test")
{
    const int M = 3;
    const int N = 4;
    const int K = 5;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix A = MakeMatrix<int32_t>(M, K);
            Matrix B = MakeMatrix<int32_t>(K, N);
            Matrix AB = MakeMatrix<int32_t>(M, N);

            // Easy but inefficient method to set all values (but should be amenable to optimization)
            Nest fillNest(MemoryShape{ M, N, K });
            auto [i, j, k] = fillNest.GetIndices<3>();
            fillNest.Set([&, i = i, j = j, k = k]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));

                A(i, k) = Scalar(Cast((iVal * 10) + kVal, ValueType::Int32));
                B(k, j) = Scalar(Cast((kVal * 10) + jVal, ValueType::Int32));
                AB(i, j) = Scalar(Cast(0, ValueType::Int32));
            });
            fillNest.CreateSchedule();

            // JIT-LABEL: A*B:
            Print("A*B:\n"s);
            // JIT: 300 310 320 330
            // JIT-NEXT: 1300 1360 1420 1480
            // JIT-NEXT: 2300 2410 2520 2630
            MatrixMatrixMultiply(A, B, AB);
            Print(AB);
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_int32_cached_matrix_multiply_test
// CHECK-NEXT: accv.module "jit_int32_cached_matrix_multiply_test" {
// JIT-LABEL: @jit_int32_cached_matrix_multiply_test

TEST_CASE("jit_int32_cached_matrix_multiply_test")
{
    const int M = 3;
    const int N = 4;
    const int K = 5;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix A = MakeMatrix<int32_t>(M, K);
            Matrix B = MakeMatrix<int32_t>(K, N);
            Matrix C = MakeMatrix<int32_t>(M, N);

            // Easy but inefficient method to set all values (but should be amenable to optimization)
            Nest fillNest(MemoryShape{ M, N, K });
            auto [ii, jj, kk] = fillNest.GetIndices<3>();
            fillNest.Set([&, i = ii, j = jj, k = kk]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));

                A(i, k) = Scalar(Cast(iVal + kVal, ValueType::Int32));
                B(k, j) = Scalar(Cast((kVal * Cast(2, ValueType::Int32)) + jVal, ValueType::Int32));
                C(i, j) = Scalar(Cast(0, ValueType::Int32));
            });
            fillNest.CreateSchedule();

            auto simpleMatMul = [](Array A, Array B, Array C) -> Nest {
                auto M = A.Shape()[0];
                auto N = B.Shape()[1];
                auto K = A.Shape()[1];
                Nest nest(MemoryShape{ M, N, K });
                Scalar i, j, k;
                std::tie(i, j, k) = nest.GetIndices<3>();
                nest.Set([&]() {
                    C(i, j) += A(i, k) * B(k, j);
                });
                return nest;
            };

            const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            [[maybe_unused]] const int vectorUnits = 16; // AVX-2 has 16 256-bit registers

            auto matMulNest = simpleMatMul(A, B, C);
            auto [i, j, k] = matMulNest.GetIndices<3>();
            auto schedule = matMulNest.CreateSchedule();
            auto [iOuter, iInner] = schedule.Split(i, 6);
            auto [jOuter, jInner] = schedule.Split(j, vectorSize);
            schedule.SetOrder({ jOuter, iOuter, iInner, k, jInner });
            auto plan = schedule.CreatePlan();
            plan.AddCache(C, iInner);

            // JIT-LABEL: A:
            Print("A:\n"s);
            Print(A);
            // JIT-LABEL: B:
            Print("B:\n"s);
            Print(B);
            // JIT-LABEL: A*B:
            Print("A*B:\n"s);
            // JIT:60 70 80 90
            // JIT-NEXT:80 95 110 125
            // JIT-NEXT:100 120 140 160
            Print(C);
        });
    SUCCEED();
}

// CHECK-LABEL: module @jit_float_cached_matrix_multiply_test
// CHECK-NEXT: accv.module "jit_float_cached_matrix_multiply_test" {
// JIT-LABEL: @jit_float_cached_matrix_multiply_test

TEST_CASE("jit_float_cached_matrix_multiply_test")
{
    const int M = 32;
    const int N = 32;
    const int K = 32;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix A = MakeMatrix<float>(M, K);
            Matrix B = MakeMatrix<float>(K, N);
            Matrix C = MakeMatrix<float>(M, N);

            // Easy but inefficient method to set all values (but should be amenable to optimization)
            Nest fillNest(MemoryShape{ M, N, K });
            auto [ii, jj, kk] = fillNest.GetIndices<3>();
            fillNest.Set([&, i = ii, j = jj, k = kk]() {
                auto iVal = Scalar(Cast(i, ValueType::Float));
                auto jVal = Scalar(Cast(j, ValueType::Float));
                auto kVal = Scalar(Cast(k, ValueType::Float));

                A(i, k) = Scalar(Cast(iVal + kVal, ValueType::Float));
                B(k, j) = Scalar(Cast((kVal * Cast(2, ValueType::Float)) + jVal, ValueType::Float));
                C(i, j) = Scalar(Cast(0, ValueType::Float));
            });
            fillNest.CreateSchedule();

            auto simpleMatMul = [](Array A, Array B, Array C) -> Nest {
                auto M = A.Shape()[0];
                auto N = B.Shape()[1];
                auto K = A.Shape()[1];
                Nest nest(MemoryShape{ M, N, K });
                Scalar i, j, k;
                std::tie(i, j, k) = nest.GetIndices<3>();
                nest.Set([&]() {
                    C(i, j) += A(i, k) * B(k, j);
                });
                return nest;
            };

            const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            const int vectorBytes = vectorSize * 4; // 4 bytes per float
            const int vectorUnits = 16; // AVX-2 has 16 256-bit registers

            auto matMulNest = simpleMatMul(A, B, C);
            auto [i, j, k] = matMulNest.GetIndices<3>();
            auto schedule = matMulNest.CreateSchedule();
            auto [iOuter, iInner] = schedule.Split(i, 6);
            auto [jOuter, jInner] = schedule.Split(j, vectorSize);
            schedule.SetOrder({ jOuter, iOuter, k, iInner, jInner });
            auto plan = schedule.CreatePlan();
            plan.AddCache(C, iInner);

            schedule.Unroll(iInner);
            schedule.Unroll(jInner);
            plan.Vectorize(jInner, { vectorBytes, vectorUnits });

            // JIT-LABEL: A*B:
            Print("A*B:\n"s);
            // JIT: 20832.000000 21328.000000 21824.000000 22320.000000 22816.000000 23312.000000 23808.000000 24304.000000
            // JIT: 21824.000000 22352.000000 22880.000000 23408.000000 23936.000000 24464.000000 24992.000000 25520.000000
            // JIT: 22816.000000 23376.000000 23936.000000 24496.000000 25056.000000 25616.000000 26176.000000 26736.000000
            // JIT: 23808.000000 24400.000000 24992.000000 25584.000000 26176.000000 26768.000000 27360.000000 27952.000000
            // JIT: 24800.000000 25424.000000 26048.000000 26672.000000 27296.000000 27920.000000 28544.000000 29168.000000
            Print(C);
        });
    SUCCEED();
}

// TODO: Enable when functionality is needed and semantics are fully cleared
#if 0

// COM: CHECK-LABEL: module @jit_vectorize_outer_loop_test
// COM: CHECK-NEXT: accv.module "jit_vectorize_outer_loop_test" {
// COM: JIT-LABEL: @jit_vectorize_outer_loop_test
TEST_CASE("jit_vectorize_outer_loop_test")
{
    const int M = 32;
    const int N = 32;
    const int K = 32;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Matrix A = MakeMatrix<float>(M, K);
            Matrix B = MakeMatrix<float>(K, N);
            Matrix C = MakeMatrix<float>(M, N);

            // Easy but inefficient method to set all values (but should be amenable to optimization)
            Nest fillNest(MemoryShape{ M, N, K });
            auto [ii, jj, kk] = fillNest.GetIndices<3>();
            fillNest.Set([&, i = ii, j = jj, k = kk]() {
                auto iVal = Scalar(Cast(i, ValueType::Float));
                auto jVal = Scalar(Cast(j, ValueType::Float));
                auto kVal = Scalar(Cast(k, ValueType::Float));

                A(i, k) = Scalar(Cast(iVal + kVal, ValueType::Float));
                B(k, j) = Scalar(Cast((kVal * Cast(2, ValueType::Float)) + jVal, ValueType::Float));
                C(i, j) = Scalar(Cast(0, ValueType::Float));
            });
            fillNest.CreateSchedule();

            auto simpleMatMul = [](Array A, Array B, Array C) -> Nest {
                auto M = A.Shape()[0];
                auto N = B.Shape()[1];
                auto K = A.Shape()[1];
                Nest nest(MemoryShape{ M, N, K });
                Scalar i, j, k;
                std::tie(i, j, k) = nest.GetIndices<3>();
                nest.Set([&]() {
                    C(i, j) += A(i, k) * B(k, j);
                });
                return nest;
            };

            int vectorSize = 8;
            int vectorBytes = vectorSize * 4; // 4 bytes per float
            int vectorUnits = 16;

            int64_t jBlock = 256;
            int64_t kBlock = 128;
            int64_t iBlock = 24;
            int64_t numRowsInKernel = 4;
            int64_t numColumnsInKernel = 2 * vectorSize;
            int64_t kUnroll = 2;

            auto matMulNest = simpleMatMul(A, B, C);
            auto [i, j, k] = matMulNest.GetIndices<3>();
            auto schedule = matMulNest.CreateSchedule();

            auto [kCache, kInner1] = schedule.Split(k, kBlock);
            auto [jCache, jInner1] = schedule.Split(j, jBlock);
            auto [iOuter, iKernelBlock] = schedule.Split(i, iBlock);

            auto [kKernelOuter, kInner2] = schedule.Split(kInner1, kUnroll);
            auto [iKernelOuter, iInner] = schedule.Split(iKernelBlock, numRowsInKernel);
            auto [jKernelOuter2, jInner2] = schedule.Split(jInner1, numColumnsInKernel);
            auto [jKernelOuter, jInner3] = schedule.Split(jInner2, vectorSize);

            // Set the order
            schedule.SetOrder({ iOuter, jCache, kCache, kKernelOuter, jKernelOuter2, iKernelOuter, iInner, jKernelOuter, jInner3, kInner2 });

            auto plan = schedule.CreatePlan();
            // plan.AddCache(C, iInner);

            schedule.Unroll(kInner2);
            plan.Vectorize(jInner3, { vectorBytes, vectorUnits });

            // COM: JIT-LABEL: A*B:
            Print("A*B:\n"s);
            // COM: JIT-NEXT: 20832.000000 21328.000000 21824.000000 22320.000000 22816.000000 23312.000000 23808.000000 24304.000000
            // COM: JIT-NEXT: 21824.000000 22352.000000 22880.000000 23408.000000 23936.000000 24464.000000 24992.000000 25520.000000
            // COM: JIT-NEXT: 22816.000000 23376.000000 23936.000000 24496.000000 25056.000000 25616.000000 26176.000000 26736.000000
            // COM: JIT-NEXT: 23808.000000 24400.000000 24992.000000 25584.000000 26176.000000 26768.000000 27360.000000 27952.000000
            // COM: JIT-NEXT: 24800.000000 25424.000000 26048.000000 26672.000000 27296.000000 27920.000000 28544.000000 29168.000000
            Print(C);
        });
    SUCCEED();
}

// COM: CHECK: [[colMap:#map[0-9]+]] = affine_map<(d0, d1) -> (d1 * 4 + d0)>
// COM: CHECK: [[rowMap:#map[0-9]+]] = affine_map<(d0, d1) -> (d1 + d0 * 4)>
// COM: CHECK-LABEL: module @jit_matrix_transpose_test
// COM: CHECK-NEXT: accv.module "jit_matrix_transpose_test" {
// COM: JIT-LABEL: @jit_matrix_transpose_test

TEST_CASE("jit_matrix_transpose_test")
{
    const int M = 3;
    const int N = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            // COM: CHECK: [[m:%[0-9]+]] = "accv.alloc"() {allocType = 0 : i64} : () -> memref<3x4xf32, #map0, 3>
            Matrix m = MakeMatrix<float>(M, N);
            CHECK(m.GetMatrixLayout() == Matrix::MatrixLayout::rowMajor);

            Nest fillNest(MemoryShape{ M, N });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                m(i, j) = Scalar(Cast((iVal * 10) + jVal, ValueType::Float));
            });
            fillNest.CreateSchedule();

            // COM: JIT-LABEL: m:
            Print("m:\n"s);
            // COM: JIT: 0.000000 1.000000 2.000000 3.000000
            // COM: JIT-NEXT: 10.000000 11.000000 12.000000 13.000000
            // COM: JIT-NEXT: 20.000000 21.000000 22.000000 23.000000
            Print(m);

            // COM: CHECK: [[m_t:%[0-9]+]] = "accv.reorder"([[m]]) {order = [1, 0]} : (memref<3x4xf32, #map0, 3>) -> memref<4x3xf32, [[colMap]], 3>
            auto m_t = m.TransposedView();
            CHECK(m_t.GetMatrixLayout() == Matrix::MatrixLayout::columnMajor);

            // COM: JIT-LABEL: m_t:
            Print("m_t:\n"s);
            // COM: JIT: 0.000000 10.000000 20.000000
            // COM: JIT-NEXT: 1.000000 11.000000 21.000000
            // COM: JIT-NEXT: 2.000000 12.000000 22.000000
            // COM: JIT-NEXT: 3.000000 13.000000 23.000000
            Print(m_t);

            // COM: CHECK: [[m_t_t:%[0-9]+]] = "accv.reorder"([[m_t]]) {order = [1, 0]} : (memref<4x3xf32, [[colMap]], 3>) -> memref<3x4xf32, [[rowMap]], 3>
            auto m_t_t = m_t.TransposedView();
            CHECK(m_t_t.GetMatrixLayout() == Matrix::MatrixLayout::rowMajor);

            // COM: JIT-LABEL: m_t_t:
            Print("m_t_t:\n"s);
            // COM: JIT: 0.000000 1.000000 2.000000 3.000000
            // COM: JIT-NEXT: 10.000000 11.000000 12.000000 13.000000
            // COM: JIT-NEXT: 20.000000 21.000000 22.000000 23.000000
            Print(m_t_t);
        });
    SUCCEED();
}


// COM: CHECK-LABEL: module @jit_array_order_test
// COM: CHECK-NEXT: accv.module "jit_array_order_test" {
// COM: CHECK: "accv.print"
// COM: JIT-LABEL: @jit_array_order_test
TEST_CASE("jit_array_order_test")
{
    using accera::utilities::DimensionOrder;
    using accera::utilities::MemoryLayout;
    using accera::utilities::MemoryShape;
    const int M = 2;
    const int N = 3;
    const int K = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto layout1 = MemoryLayout{ MemoryShape{ M, N, K } };
            Array arr1 = MakeArray(layout1, ValueType::Int32, "arr1");
            REQUIRE(arr1.Shape()[0] == M);
            REQUIRE(arr1.Shape()[1] == N);
            REQUIRE(arr1.Shape()[2] == K);

            auto layout2 = layout1.ReorderedCopy({ 1, 0, 2 });
            Array arr2 = MakeArray(layout2, ValueType::Int32, "arr2");
            REQUIRE(arr2.Shape()[0] == M);
            REQUIRE(arr2.Shape()[1] == N);
            REQUIRE(arr2.Shape()[2] == K);

            Nest fillNest(MemoryShape{ M, N, K });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                arr1(i, j, k) = (iVal * 100) + (jVal * 10) + kVal;
                arr2(i, j, k) = (iVal * 100) + (jVal * 10) + kVal + 1000;
            });
            fillNest.CreateSchedule();

            // COM: JIT-LABEL: arr1:
            Print("arr1:\n"s);
            // COM: JIT: 0 1 2 3
            // COM: JIT: 10 11 12 13
            // COM: JIT: 20 21 22 23
            // COM: JIT: 100 101 102 103
            // COM: JIT: 110 111 112 113
            // COM: JIT: 120 121 122 123
            Print(arr1);

            // COM: JIT-LABEL: arr2:
            Print("arr2:\n"s);
            // COM: JIT: 1000 1001 1002 1003
            // COM: JIT: 1010 1011 1012 1013
            // COM: JIT: 1020 1021 1022 1023
            // COM: JIT: 1100 1101 1102 1103
            // COM: JIT: 1110 1111 1112 1113
            // COM: JIT: 1120 1121 1122 1123
            Print(arr2);

            // COM: JIT-LABEL: mem1:
            Print("mem1:\n"s);
            // COM: JIT: 0 1 2 3 10 11 12 13 20 21 22 23 100 101 102 103 110 111 112 113 120 121 122 123
            PrintRawMemory(arr1);

            // COM: JIT-LABEL: mem2:
            Print("\nmem2:\n"s);
            // COM: JIT: 1000 1001 1002 1003 1100 1101 1102 1103 1010 1011 1012 1013 1110 1111 1112 1113 1020 1021 1022 1023 1120 1121 1122 1123
            PrintRawMemory(arr2);
            Print("\n"s);
        });
}


// COM: CHECK-LABEL: module @jit_array_slice_test1
// COM: CHECK-NEXT: accv.module "jit_array_slice_test1" {
// COM: CHECK: "accv.print"
// COM: JIT-LABEL: @jit_array_slice_test1
TEST_CASE("jit_array_slice_test1")
{
    const int M = 2;
    const int N = 3;
    const int K = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Array arr = MakeArray({ M, N, K }, ValueType::Int32);

            Nest fillNest(MemoryShape{ M, N, K });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                arr(i, j, k) = (iVal * 100) + (jVal * 10) + kVal;
            });
            fillNest.CreateSchedule();
            // At this point, arr should be:
            //
            // 0 1 2 3
            // 10 11 12 13
            // 20 21 22 23
            //
            // 100 101 102 103
            // 110 111 112 113
            // 120 121 122 123

            // Get the 2nd slice of the first (outer) dimension:
            auto slice0a = arr.Slice({ 0 }, { Scalar(0) });
            auto slice0b = arr.Slice({ 0 }, { Scalar(1) });
            auto slice1a = arr.Slice({ 1 }, { Scalar(0) });
            auto slice1b = arr.Slice({ 1 }, { Scalar(1) });
            auto slice1c = arr.Slice({ 1 }, { Scalar(2) });
            auto slice2d = arr.Slice({ 2 }, { Scalar(3) });

            slice0a(0, 0) = Scalar(1000);
            slice0b(0, 0) = Scalar(2000);

            // COM: JIT-LABEL: arr:
            Print("arr:\n"s);
            // COM: JIT: 1000 1 2 3
            // COM: JIT-NEXT: 10 11 12 13
            // COM: JIT-NEXT: 20 21 22 23
            // COM: JIT: 2000 101 102 103
            // COM: JIT-NEXT: 110 111 112 113
            // COM: JIT-NEXT: 120 121 122 123
            Print(arr);

            // COM: JIT-LABEL: slice0a:
            Print("\nslice0a:\n"s);
            // COM: JIT: 1000 1 2 3
            // COM: JIT-NEXT: 10 11 12 13
            // COM: JIT-NEXT: 20 21 22 23
            Print(slice0a);

            // COM: JIT-LABEL: slice0b:
            Print("\nslice0b:\n"s);
            // COM: JIT: 2000 101 102 103
            // COM: JIT-NEXT: 110 111 112 113
            // COM: JIT-NEXT: 120 121 122 123
            Print(slice0b);

            // COM: JIT-LABEL: slice1a:
            Print("\nslice1a:\n"s);
            // COM: JIT: 1000 1 2 3
            // COM: JIT-NEXT: 2000 101 102 103
            Print(slice1a);

            // COM: JIT-LABEL: slice1b:
            Print("\nslice1b:\n"s);
            // COM: JIT: 10 11 12 13
            // COM: JIT-NEXT: 110 111 112 113
            Print(slice1b);

            // COM: JIT-LABEL: slice1c:
            Print("\nslice1c:\n"s);
            // COM: JIT: 20 21 22 23
            // COM: JIT-NEXT: 120 121 122 123
            Print(slice1c);

            // COM: JIT-LABEL: slice2d:
            Print("\nslice2d:\n"s);
            // COM: JIT: 3 13 23
            // COM: JIT-NEXT: 103 113 123
            Print(slice2d);

            Print("\n"s);
        });
}

// COM: CHECK-LABEL: module @jit_array_slice_test2
// COM: CHECK-NEXT: accv.module "jit_array_slice_test2" {
// COM: CHECK: "accv.print"
// COM: JIT-LABEL: @jit_array_slice_test2
TEST_CASE("jit_array_slice_test2")
{
    const int M = 2;
    const int N = 3;
    const int K = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            Array arr = MakeArray({ M, N, K }, ValueType::Int32);

            Nest fillNest(MemoryShape{ M, N, K });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                arr(i, j, k) = (iVal * 100) + (jVal * 10) + kVal;
            });
            fillNest.CreateSchedule();

            auto slice = arr.Slice({ 0, 1 }, { Scalar(1), Scalar(1) });
            slice(0) = Scalar(1000);

            // COM: JIT-LABEL: arr:
            Print("arr:\n"s);
            // 0 1 2 3
            // 10 11 12 13
            // 20 21 22 23
            // 100 101 102 103
            // 1000 111 112 113
            // 120 121 122 123
            Print(arr);

            // COM: JIT-LABEL: slice:
            Print("\nslice:\n"s);
            // 1000 111 112 113
            Print(slice);
            Print("\n"s);
        });
}

// COM: CHECK-LABEL: module @jit_merge_dim_test
// COM: CHECK-NEXT: accv.module "jit_merge_dim_test" {
// COM: CHECK: "accv.print"
// COM: JIT-LABEL: @jit_merge_dim_test
TEST_CASE("jit_merge_dim_test")
{
    const int M = 2;
    const int N = 3;
    const int K = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto layout = MemoryLayout(M, N, K);

            Array arr1 = MakeArray(layout, ValueType::Int32, "arr1");
            REQUIRE(arr1.Shape()[0] == M);
            REQUIRE(arr1.Shape()[1] == N);
            REQUIRE(arr1.Shape()[2] == K);

            Array arr2 = arr1.MergeDimensions(0, 1);
            REQUIRE(arr2.Shape()[0] == M * N);
            REQUIRE(arr2.Shape()[1] == K);

            Array arr3 = arr1.MergeDimensions(1, 2);
            REQUIRE(arr3.Shape()[0] == M);
            REQUIRE(arr3.Shape()[1] == N * K);

            Nest fillNest(MemoryShape{ M, N, K });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                arr1(i, j, k) = (iVal * 100) + (jVal * 10) + kVal;
            });
            fillNest.CreateSchedule();

            // COM: JIT-LABEL: arr1:
            Print("arr1:\n"s);
            // COM: JIT: 0 1 2 3
            // COM: JIT: 10 11 12 13
            // COM: JIT: 20 21 22 23
            // COM: JIT: 100 101 102 103
            // COM: JIT: 110 111 112 113
            // COM: JIT: 120 121 122 123
            Print(arr1);

            // COM: JIT-LABEL: arr2:
            Print("\narr2:\n"s);
            // COM: JIT: 0 1 2 3
            // COM: JIT: 10 11 12 13
            // COM: JIT: 20 21 22 23
            // COM: JIT: 100 101 102 103
            // COM: JIT: 110 111 112 113
            // COM: JIT: 120 121 122 123
            Print(arr2);

            // COM: JIT-LABEL: arr3:
            Print("\narr3:\n"s);
            // COM: JIT: 0 1 2 3 10 11 12 13 20 21 22 23
            // COM: JIT: 100 101 102 103 110 111 112 113 120 121 122 123
            Print(arr3);

            // COM: JIT-LABEL: mem1:
            Print("\nmem1:\n"s);
            // COM: JIT: 0 1 2 3 10 11 12 13 20 21 22 23 100 101 102 103 110 111 112 113 120 121 122 123
            PrintRawMemory(arr1);
            // COM: JIT-LABEL: mem2:
            Print("\nmem2:\n"s);
            // COM: JIT: 0 1 2 3 10 11 12 13 20 21 22 23 100 101 102 103 110 111 112 113 120 121 122 123
            PrintRawMemory(arr2);
            // COM: JIT-LABEL: mem3:
            Print("\nmem3:\n"s);
            // COM: JIT: 0 1 2 3 10 11 12 13 20 21 22 23 100 101 102 103 110 111 112 113 120 121 122 123
            PrintRawMemory(arr3);
            Print("\n"s);
        });
}

// COM: CHECK-LABEL: module @jit_split_dim_test
// COM: CHECK-NEXT: accv.module "jit_split_dim_test" {
// COM: CHECK: "accv.print"
// COM: JIT-LABEL: @jit_split_dim_test
TEST_CASE("jit_split_dim_test")
{
    const int M = 2;
    const int N = 3;
    const int K = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto layout = MemoryLayout(M * N, K);

            Array arr1 = MakeArray(layout, ValueType::Int32, "arr1");
            REQUIRE(arr1.Shape()[0] == M * N);
            REQUIRE(arr1.Shape()[1] == K);
            Array arr2 = arr1.SplitDimension(0, N);
            REQUIRE(arr2.Shape()[0] == M);
            REQUIRE(arr2.Shape()[1] == N);
            REQUIRE(arr2.Shape()[2] == K);

            Nest fillNest(MemoryShape{ M * N, K });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                arr1(i, j) = (iVal * 100) + jVal;
            });
            fillNest.CreateSchedule();

            // COM: JIT-LABEL: arr1:
            Print("arr1:\n"s);
            // COM: JIT: 0 1 2 3
            // COM: JIT: 100 101 102 103
            // COM: JIT: 200 201 202 203
            // COM: JIT: 300 301 302 303
            // COM: JIT: 400 401 402 403
            // COM: JIT: 500 501 502 503
            Print(arr1);

            // COM: JIT-LABEL: arr2:
            Print("arr2:\n"s);
            // COM: JIT: 0 1 2 3
            // COM: JIT: 100 101 102 103
            // COM: JIT: 200 201 202 203
            // COM: JIT: 300 301 302 303
            // COM: JIT: 400 401 402 403
            // COM: JIT: 500 501 502 503
            Print(arr2);

            // COM: JIT-LABEL: mem1:
            Print("mem1:\n"s);
            // COM: JIT: 0 1 2 3 100 101 102 103 200 201 202 203 300 301 302 303 400 401 402 403 500 501 502 503
            PrintRawMemory(arr1);

            // COM: JIT-LABEL: mem2:
            Print("\nmem2:\n"s);
            // COM: JIT: 0 1 2 3 100 101 102 103 200 201 202 203 300 301 302 303 400 401 402 403 500 501 502 503
            PrintRawMemory(arr2);
            Print("\n"s);
        });
}

// COM: CHECK-LABEL: module @jit_array_reshape_test1
// COM: CHECK-NEXT: accv.module "jit_array_reshape_test1" {
// COM: CHECK: "accv.print"
// COM: JIT-LABEL: @jit_array_reshape_test1
TEST_CASE("jit_array_reshape_test1")
{
    const int M = 2;
    const int N = 3;
    const int K = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto layout1 = MemoryLayout(M, N, K);
            auto layout2 = MemoryLayout(M * N, K);
            auto layout3 = MemoryLayout(M, N * K);

            Array arr1 = MakeArray(layout1, ValueType::Int32, "arr1");
            REQUIRE(arr1.Shape()[0] == M);
            REQUIRE(arr1.Shape()[1] == N);
            REQUIRE(arr1.Shape()[2] == K);
            Array arr2 = arr1.Reshape(layout2);
            REQUIRE(arr2.Shape()[0] == M * N);
            REQUIRE(arr2.Shape()[1] == K);
            Array arr3 = arr1.Reshape(layout3);
            REQUIRE(arr3.Shape()[0] == M);
            REQUIRE(arr3.Shape()[1] == N * K);

            Nest fillNest(MemoryShape{ M, N, K });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                arr1(i, j, k) = (iVal * 100) + (jVal * 10) + kVal;
            });
            fillNest.CreateSchedule();

            // COM: JIT-LABEL: arr1:
            Print("arr1:\n"s);
            // COM: JIT: 0 1 2 3
            // COM: JIT: 10 11 12 13
            // COM: JIT: 20 21 22 23
            // COM: JIT: 100 101 102 103
            // COM: JIT: 110 111 112 113
            // COM: JIT: 120 121 122 123
            Print(arr1);

            // COM: JIT-LABEL: arr2:
            Print("arr2:\n"s);
            // COM: JIT: 0 1 2 3
            // COM: JIT: 10 11 12 13
            // COM: JIT: 20 21 22 23
            // COM: JIT: 100 101 102 103
            // COM: JIT: 110 111 112 113
            // COM: JIT: 120 121 122 123
            Print(arr2);

            // COM: JIT-LABEL: arr3:
            Print("arr3:\n"s);
            // COM: JIT: 0 1 2 3 10 11 12 13 20 21 22 23
            // COM: JIT: 100 101 102 103 110 111 112 113 120 121 122 123
            Print(arr3);

            // COM: JIT-LABEL: mem1:
            Print("mem1:\n"s);
            // COM: JIT: 0 1 2 3 10 11 12 13 20 21 22 23 100 101 102 103 110 111 112 113 120 121 122 123
            PrintRawMemory(arr1);

            // COM: JIT-LABEL: mem2:
            Print("\nmem2:\n"s);
            // COM: JIT: 0 1 2 3 10 11 12 13 20 21 22 23 100 101 102 103 110 111 112 113 120 121 122 123
            PrintRawMemory(arr2);

            // COM: JIT-LABEL: mem3:
            Print("\nmem3:\n"s);
            // COM: JIT: 0 1 2 3 10 11 12 13 20 21 22 23
            PrintRawMemory(arr3);
            Print("\n"s);
        });
}

// COM: CHECK-LABEL: module @jit_array_reorder_test1
// COM: CHECK-NEXT: accv.module "jit_array_reorder_test1" {
// COM: CHECK: "accv.print"
// COM: JIT-LABEL: @jit_array_reorder_test1
TEST_CASE("jit_array_reorder_test1")
{
    const int M = 2;
    const int N = 3;
    const int K = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto layout1 = MemoryLayout(M, N, K);

            Array arr1 = MakeArray(layout1, ValueType::Int32, "arr1");
            REQUIRE(arr1.Shape()[0] == M);
            REQUIRE(arr1.Shape()[1] == N);
            REQUIRE(arr1.Shape()[2] == K);

            Array arr2 = arr1.Reorder(DimensionOrder{ 1, 2, 0 });
            REQUIRE(arr2.Shape()[0] == N);
            REQUIRE(arr2.Shape()[1] == K);
            REQUIRE(arr2.Shape()[2] == M);

            Nest fillNest(MemoryShape{ M, N, K });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                arr1(i, j, k) = (iVal * 100) + (jVal * 10) + kVal;
            });
            fillNest.CreateSchedule();

            // COM: JIT-LABEL: arr1:
            Print("arr1:\n"s);
            // COM: JIT: 0 1 2 3
            // COM: JIT: 10 11 12 13
            // COM: JIT: 20 21 22 23
            // COM: JIT: 100 101 102 103
            // COM: JIT: 110 111 112 113
            // COM: JIT: 120 121 122 123
            Print(arr1);

            // COM: JIT-LABEL: arr2:
            Print("arr2:\n"s);
            // COM: JIT: 0 100
            // COM: JIT: 1 101
            // COM: JIT: 2 102
            // COM: JIT: 3 103
            // COM: JIT: 10 110
            // COM: JIT: 11 111
            // COM: JIT: 12 112
            // COM: JIT: 13 113
            // COM: JIT: 20 120
            // COM: JIT: 21 121
            // COM: JIT: 22 122
            // COM: JIT: 23 123
            Print(arr2);
        });
}

// COM: CHECK: [[map0:#map[0-9]+]] = affine_map<(d0, d1, d2) ->
// COM: CHECK: [[map1:#map[0-9]+]] = affine_map<(d0, d1, d2) ->
// COM: CHECK-LABEL: module @jit_array_reorder_test2
// COM: CHECK-NEXT: accv.module "jit_array_reorder_test2" {
// COM: CHECK: %0 = "accv.alloc"() {allocType = 0 : i64}
// COM: CHECK-SAME: () -> memref<2x3x4xi32, [[map0]], 3>
// COM: CHECK: %1 = memref.transpose %0 (d0, d1, d2) -> (d1, d2, d0)
// COM: JIT-LABEL: @jit_array_reorder_test2
TEST_CASE("jit_array_reorder_test2")
{
    const int M = 2;
    const int N = 3;
    const int K = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto layout1 = MemoryLayout(MemoryShape{ M, N, K }, DimensionOrder{ 1, 2, 0 }); // N,K,M order

            Array arr1 = MakeArray(layout1, ValueType::Int32, "arr1");
            REQUIRE(arr1.Shape()[0] == M);
            REQUIRE(arr1.Shape()[1] == N);
            REQUIRE(arr1.Shape()[2] == K);
            REQUIRE(arr1.GetLayout().GetDimensionOrder() == DimensionOrder{ 1, 2, 0 });
            Print(arr1);

            Array arr2 = arr1.Reorder(arr1.GetLayout().GetDimensionOrder());
            REQUIRE(arr2.Shape()[0] == N);
            REQUIRE(arr2.Shape()[1] == K);
            REQUIRE(arr2.Shape()[2] == M);
            REQUIRE(arr2.GetLayout().IsCanonicalOrder());
            Print(arr2);
        });
}

// COM: CHECK-LABEL: module @jit_array_merge_and_slice_test
// COM: CHECK-NEXT: accv.module "jit_array_merge_and_slice_test" {
// COM: CHECK: "accv.print"
// COM: JIT-LABEL: @jit_array_merge_and_slice_test
TEST_CASE("jit_array_merge_and_slice_test")
{
    const int M = 2;
    const int N = 3;
    const int K = 4;
    const int L = 5;

    // Create an M x N x K x L array
    // Merge the M and N dimensions to get a M*N x K x L array
    // Take a slice from the K dimension to get a M*N x L array
    // Merge the K and L dimensions to get a M x N x K*L array
    // Take a slice from the N dimension to get a M x K*L array

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto layout = MemoryLayout{ MemoryShape{ M, N, K, L } };
            Array arr = MakeArray(layout, ValueType::Int32, "arr");

            Array merged1 = arr.MergeDimensions(0, 1);
            auto refLayout1 = MemoryLayout{ MemoryShape{ M * N, K, L } };
            CHECK(merged1.GetLayout() == refLayout1);
            Array slice1a = merged1.Slice({ 1 }, { Scalar(0) });
            Array slice1b = merged1.Slice({ 1 }, { Scalar(1) });

            Array merged2 = arr.MergeDimensions(2, 3);
            auto refLayout2 = MemoryLayout{ MemoryShape{ M, N, K * L } };
            CHECK(merged2.GetLayout() == refLayout2);
            Array slice2a = merged2.Slice({ 1 }, { Scalar(0) });
            Array slice2b = merged2.Slice({ 1 }, { Scalar(1) });

            Nest fillNest(MemoryShape{ M, N, K, L });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            Scalar l = indices[3];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                auto lVal = Scalar(Cast(l, ValueType::Int32));
                arr(i, j, k, l) = (iVal * 1000) + (jVal * 100) + (kVal * 10) + lVal;
            });
            fillNest.CreateSchedule();

            // COM: JIT-LABEL: arr:
            Print("arr:\n"s);
            // COM: JIT: 0 1 2 3 4
            // COM: JIT: 10 11 12 13 14
            // COM: JIT: 20 21 22 23 24
            // COM: JIT: 30 31 32 33 34
            // COM: JIT: 100 101 102 103 104
            // COM: JIT: 110 111 112 113 114
            // COM: JIT: 120 121 122 123 124
            // COM: JIT: 130 131 132 133 134
            // COM: JIT: 200 201 202 203 204
            // COM: JIT: 210 211 212 213 214
            // COM: JIT: 220 221 222 223 224
            // COM: JIT: 230 231 232 233 234
            // COM: JIT: 1000 1001 1002 1003 1004
            // COM: JIT: 1010 1011 1012 1013 1014
            // COM: JIT: 1020 1021 1022 1023 1024
            // COM: JIT: 1030 1031 1032 1033 1034
            // COM: JIT: 1100 1101 1102 1103 1104
            // COM: JIT: 1110 1111 1112 1113 1114
            // COM: JIT: 1120 1121 1122 1123 1124
            // COM: JIT: 1130 1131 1132 1133 1134
            // COM: JIT: 1200 1201 1202 1203 1204
            // COM: JIT: 1210 1211 1212 1213 1214
            // COM: JIT: 1220 1221 1222 1223 1224
            // COM: JIT: 1230 1231 1232 1233 1234
            Print(arr);
            Print("\n"s);

            // COM: JIT-LABEL: merged1:
            Print("merged1:\n"s);
            // COM: JIT: 0 1 2 3 4
            // COM: JIT: 10 11 12 13 14
            // COM: JIT: 20 21 22 23 24
            // COM: JIT: 30 31 32 33 34
            // COM: JIT: 100 101 102 103 104
            // COM: JIT: 110 111 112 113 114
            // COM: JIT: 120 121 122 123 124
            // COM: JIT: 130 131 132 133 134
            // COM: JIT: 200 201 202 203 204
            // COM: JIT: 210 211 212 213 214
            // COM: JIT: 220 221 222 223 224
            // COM: JIT: 230 231 232 233 234
            // COM: JIT: 1000 1001 1002 1003 1004
            // COM: JIT: 1010 1011 1012 1013 1014
            // COM: JIT: 1020 1021 1022 1023 1024
            // COM: JIT: 1030 1031 1032 1033 1034
            // COM: JIT: 1100 1101 1102 1103 1104
            // COM: JIT: 1110 1111 1112 1113 1114
            // COM: JIT: 1120 1121 1122 1123 1124
            // COM: JIT: 1130 1131 1132 1133 1134
            // COM: JIT: 1200 1201 1202 1203 1204
            // COM: JIT: 1210 1211 1212 1213 1214
            // COM: JIT: 1220 1221 1222 1223 1224
            // COM: JIT: 1230 1231 1232 1233 1234
            Print(merged1);
            Print("\n"s);

            // COM: JIT-LABEL: slice1a:
            Print("slice1a:\n"s);
            // COM: JIT: 0 1 2 3 4
            // COM: JIT: 100 101 102 103 104
            // COM: JIT: 200 201 202 203 204
            // COM: JIT: 1000 1001 1002 1003 1004
            // COM: JIT: 1100 1101 1102 1103 1104
            // COM: JIT: 1200 1201 1202 1203 1204
            Print(slice1a);
            Print("\n"s);

            // COM: JIT-LABEL: slice1b:
            Print("slice1b:\n"s);
            // COM: JIT: 10 11 12 13 14
            // COM: JIT: 110 111 112 113 114
            // COM: JIT: 210 211 212 213 214
            // COM: JIT: 1010 1011 1012 1013 1014
            // COM: JIT: 1110 1111 1112 1113 1114
            // COM: JIT: 1210 1211 1212 1213 1214
            Print(slice1b);
            Print("\n"s);

            // COM: JIT-LABEL: merged2:
            Print("merged2:\n"s);
            // COM: JIT: 0 1 2 3 4 10 11 12 13 14 20 21 22 23 24 30 31 32 33 34
            // COM: JIT: 100 101 102 103 104 110 111 112 113 114 120 121 122 123 124 130 131 132 133 134
            // COM: JIT: 200 201 202 203 204 210 211 212 213 214 220 221 222 223 224 230 231 232 233 234
            // COM: JIT: 1000 1001 1002 1003 1004 1010 1011 1012 1013 1014 1020 1021 1022 1023 1024 1030 1031 1032 1033 1034
            // COM: JIT: 1100 1101 1102 1103 1104 1110 1111 1112 1113 1114 1120 1121 1122 1123 1124 1130 1131 1132 1133 1134
            // COM: JIT: 1200 1201 1202 1203 1204 1210 1211 1212 1213 1214 1220 1221 1222 1223 1224 1230 1231 1232 1233 1234
            Print(merged2);
            Print("\n"s);

            // COM: JIT-LABEL: slice2a:
            Print("slice2a:\n"s);
            // COM: JIT: 0 1 2 3 4 10 11 12 13 14 20 21 22 23 24 30 31 32 33 34
            // COM: JIT: 1000 1001 1002 1003 1004 1010 1011 1012 1013 1014 1020 1021 1022 1023 1024 1030 1031 1032 1033 1034

            Print(slice2a);
            Print("\n"s);

            // COM: JIT-LABEL: slice2b:
            Print("slice2b:\n"s);
            // COM: JIT: 100 101 102 103 104 110 111 112 113 114 120 121 122 123 124 130 131 132 133 134
            // COM: JIT: 1100 1101 1102 1103 1104 1110 1111 1112 1113 1114 1120 1121 1122 1123 1124 1130 1131 1132 1133 1134
            Print(slice2b);
            Print("\n"s);
        });
}

// COM: CHECK-LABEL: module @jit_reordered_array_merge_and_slice_test
// COM: JIT-LABEL: @jit_reordered_array_merge_and_slice_test
TEST_CASE("jit_reordered_array_merge_and_slice_test")
{
    const int M = 2;
    const int N = 3;
    const int K = 4;
    const int L = 5;

    // TODO:
    // Create an M x N x K x L array, with memory order (N, K, M, L)
    // Merge the N and K dimensions to get a M x N*K x L array with memory order (N*K, M, L)
    // Take a slice from the M dimension to get a N*K x L array with memory order (N*K, L)
    // Merge the M and L dimensions to get a M*L x N x K array with memory order (N, K, M*L)
    // Take a slice from the N dimension to get a M*L x K array with memory order (K, M*L)

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto order = DimensionOrder{ 1, 2, 0, 3 };
            auto layout = MemoryLayout{ MemoryShape{ M, N, K, L }, order };
            Array arr = MakeArray(layout, ValueType::Int32, "arr");

            Nest fillNest(MemoryShape{ M, N, K, L });
            auto indices = fillNest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            Scalar l = indices[3];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));
                auto kVal = Scalar(Cast(k, ValueType::Int32));
                auto lVal = Scalar(Cast(l, ValueType::Int32));
                arr(i, j, k, l) = (iVal * 1000) + (jVal * 100) + (kVal * 10) + lVal;
            });
            fillNest.CreateSchedule();

            // COM: JIT-LABEL: arr:
            Print("arr:\n"s);
            // COM: JIT: 0 1 2 3 4
            // COM: JIT: 10 11 12 13 14
            // COM: JIT: 20 21 22 23 24
            // COM: JIT: 30 31 32 33 34
            // COM: JIT: 100 101 102 103 104
            // COM: JIT: 110 111 112 113 114
            // COM: JIT: 120 121 122 123 124
            // COM: JIT: 130 131 132 133 134
            // COM: JIT: 200 201 202 203 204
            // COM: JIT: 210 211 212 213 214
            // COM: JIT: 220 221 222 223 224
            // COM: JIT: 230 231 232 233 234
            // COM: JIT: 1000 1001 1002 1003 1004
            // COM: JIT: 1010 1011 1012 1013 1014
            // COM: JIT: 1020 1021 1022 1023 1024
            // COM: JIT: 1030 1031 1032 1033 1034
            // COM: JIT: 1100 1101 1102 1103 1104
            // COM: JIT: 1110 1111 1112 1113 1114
            // COM: JIT: 1120 1121 1122 1123 1124
            // COM: JIT: 1130 1131 1132 1133 1134
            // COM: JIT: 1200 1201 1202 1203 1204
            // COM: JIT: 1210 1211 1212 1213 1214
            // COM: JIT: 1220 1221 1222 1223 1224
            // COM: JIT: 1230 1231 1232 1233 1234
            Print(arr);
            Print("\n"s);

            // // We can't merge things that aren't in canonical order. So, reorder them
            Array canonicalArr = arr.Reorder(order);
            CHECK(canonicalArr.GetLayout().IsCanonicalOrder());
            // COM: JIT-LABEL: canonicalArr:
            Print("canonicalArr:\n"s);
            // COM: JIT: 0 1 2 3 4
            // COM: JIT: 1000 1001 1002 1003 1004
            // COM: JIT: 10 11 12 13 14
            // COM: JIT: 1010 1011 1012 1013 1014
            // COM: JIT: 20 21 22 23 24
            // COM: JIT: 1020 1021 1022 1023 1024
            // COM: JIT: 30 31 32 33 34
            // COM: JIT: 1030 1031 1032 1033 1034
            // COM: JIT: 100 101 102 103 104
            // COM: JIT: 1100 1101 1102 1103 1104
            // COM: JIT: 110 111 112 113 114
            // COM: JIT: 1110 1111 1112 1113 1114
            // COM: JIT: 120 121 122 123 124
            // COM: JIT: 1120 1121 1122 1123 1124
            // COM: JIT: 130 131 132 133 134
            // COM: JIT: 1130 1131 1132 1133 1134
            // COM: JIT: 200 201 202 203 204
            // COM: JIT: 1200 1201 1202 1203 1204
            // COM: JIT: 210 211 212 213 214
            // COM: JIT: 1210 1211 1212 1213 1214
            // COM: JIT: 220 221 222 223 224
            // COM: JIT: 1220 1221 1222 1223 1224
            // COM: JIT: 230 231 232 233 234
            // COM: JIT: 1230 1231 1232 1233 1234
            Print(canonicalArr);
            Print("\n"s);

            // canonicalArr is in N, K, M, L logical order
            // Merge N and K dimensions
            Array merged1 = canonicalArr.MergeDimensions(0, 1);
            auto refLayout1 = MemoryLayout{ MemoryShape{ N * K, M, L } };
            CHECK(merged1.GetLayout() == refLayout1);

            // COM: JIT-LABEL: merged1:
            Print("merged1:\n"s);
            // COM: JIT: 0 1 2 3 4
            // COM: JIT: 1000 1001 1002 1003 1004
            // COM: JIT: 10 11 12 13 14
            // COM: JIT: 1010 1011 1012 1013 1014
            // COM: JIT: 20 21 22 23 24
            // COM: JIT: 1020 1021 1022 1023 1024
            // COM: JIT: 30 31 32 33 34
            // COM: JIT: 1030 1031 1032 1033 1034
            // COM: JIT: 100 101 102 103 104
            // COM: JIT: 1100 1101 1102 1103 1104
            // COM: JIT: 110 111 112 113 114
            // COM: JIT: 1110 1111 1112 1113 1114
            // COM: JIT: 120 121 122 123 124
            // COM: JIT: 1120 1121 1122 1123 1124
            // COM: JIT: 130 131 132 133 134
            // COM: JIT: 1130 1131 1132 1133 1134
            // COM: JIT: 200 201 202 203 204
            // COM: JIT: 1200 1201 1202 1203 1204
            // COM: JIT: 210 211 212 213 214
            // COM: JIT: 1210 1211 1212 1213 1214
            // COM: JIT: 220 221 222 223 224
            // COM: JIT: 1220 1221 1222 1223 1224
            // COM: JIT: 230 231 232 233 234
            // COM: JIT: 1230 1231 1232 1233 1234
            Print(merged1);
            Print("\n"s);

            // Slice out dimension M
            Array slice1a = merged1.Slice({ 1 }, { Scalar(0) });
            Array slice1b = merged1.Slice({ 1 }, { Scalar(1) });

            // COM: JIT-LABEL: slice1a:
            Print("slice1a:\n"s);
            // COM: JIT: 0 1 2 3 4
            // COM: JIT: 10 11 12 13 14
            // COM: JIT: 20 21 22 23 24
            // COM: JIT: 30 31 32 33 34
            // COM: JIT: 100 101 102 103 104
            // COM: JIT: 110 111 112 113 114
            // COM: JIT: 120 121 122 123 124
            // COM: JIT: 130 131 132 133 134
            // COM: JIT: 200 201 202 203 204
            // COM: JIT: 210 211 212 213 214
            // COM: JIT: 220 221 222 223 224
            // COM: JIT: 230 231 232 233 234
            Print(slice1a);
            Print("\n"s);

            // COM: JIT-LABEL: slice1b:
            Print("slice1b:\n"s);
            // COM: JIT: 1000 1001 1002 1003 1004
            // COM: JIT: 1010 1011 1012 1013 1014
            // COM: JIT: 1020 1021 1022 1023 1024
            // COM: JIT: 1030 1031 1032 1033 1034
            // COM: JIT: 1100 1101 1102 1103 1104
            // COM: JIT: 1110 1111 1112 1113 1114
            // COM: JIT: 1120 1121 1122 1123 1124
            // COM: JIT: 1130 1131 1132 1133 1134
            // COM: JIT: 1200 1201 1202 1203 1204
            // COM: JIT: 1210 1211 1212 1213 1214
            // COM: JIT: 1220 1221 1222 1223 1224
            // COM: JIT: 1230 1231 1232 1233 1234
            Print(slice1b);
            Print("\n"s);

            // Merge M and L dimensions
            Array merged2 = canonicalArr.MergeDimensions(2, 3);
            auto refLayout2 = MemoryLayout{ MemoryShape{ N, K, M * L } };
            CHECK(merged2.GetLayout() == refLayout2);
            Array slice2a = merged2.Slice({ 1 }, { Scalar(0) });
            Array slice2b = merged2.Slice({ 1 }, { Scalar(1) });

            // COM: JIT-LABEL: merged2:
            Print("merged2:\n"s);
            // COM: JIT: 0 1 2 3 4 1000 1001 1002 1003 1004
            // COM: JIT: 10 11 12 13 14 1010 1011 1012 1013 1014
            // COM: JIT: 20 21 22 23 24 1020 1021 1022 1023 1024
            // COM: JIT: 30 31 32 33 34 1030 1031 1032 1033 1034
            // COM: JIT: 100 101 102 103 104 1100 1101 1102 1103 1104
            // COM: JIT: 110 111 112 113 114 1110 1111 1112 1113 1114
            // COM: JIT: 120 121 122 123 124 1120 1121 1122 1123 1124
            // COM: JIT: 130 131 132 133 134 1130 1131 1132 1133 1134
            // COM: JIT: 200 201 202 203 204 1200 1201 1202 1203 1204
            // COM: JIT: 210 211 212 213 214 1210 1211 1212 1213 1214
            // COM: JIT: 220 221 222 223 224 1220 1221 1222 1223 1224
            // COM: JIT: 230 231 232 233 234 1230 1231 1232 1233 1234

            Print(merged2);
            Print("\n"s);

            // COM: JIT-LABEL: slice2a:
            Print("slice2a:\n"s);
            // COM: JIT: 0 1 2 3 4 1000 1001 1002 1003 1004
            // COM: JIT: 100 101 102 103 104 1100 1101 1102 1103 1104
            // COM: JIT: 200 201 202 203 204 1200 1201 1202 1203 1204
            Print(slice2a);
            Print("\n"s);

            // COM: JIT-LABEL: slice2b:
            Print("slice2b:\n"s);
            // COM: JIT: 10 11 12 13 14 1010 1011 1012 1013 1014
            // COM: JIT: 110 111 112 113 114 1110 1111 1112 1113 1114
            // COM: JIT: 210 211 212 213 214 1210 1211 1212 1213 1214
            Print(slice2b);
            Print("\n"s);
        });
}

// COM: CHECK-LABEL: module @jit_reordered_array_slice_test
// COM: CHECK-NEXT: accv.module "jit_reordered_array_slice_test" {
// COM: JIT-LABEL: @jit_reordered_array_slice_test
TEST_CASE("jit_reordered_array_slice_test")
{
    const int M = 3;
    const int N = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto At = MakeArray<float>({ N, M });
            auto A = At.Reorder({ 1, 0 }); // M x N

            For(0, M, 1, [&](Scalar i) {
                auto row = A.Slice({ 0 }, { i });
                For(0, N, 1, [&](Scalar j) {
                    auto iVal = Scalar(Cast(i, ValueType::Int32));
                    auto jVal = Scalar(Cast(j, ValueType::Int32));
                    auto val = Scalar(Cast((iVal * 10) + jVal, ValueType::Float));
                    row(j) = val;
                });
            });

            // slices of reordered arrays not working yet
            // COM: JIT: 0.000000 1.000000 2.000000 3.000000
            // COM: JIT-NEXT: 10.000000 11.000000 12.000000 13.000000
            // COM: JIT-NEXT: 20.000000 21.000000 22.000000 23.000000
            Print(A);
        });
}

// COM: CHECK-LABEL: module @jit_reordered_array_view_test
// COM: CHECK-NEXT: accv.module "jit_reordered_array_view_test" {
// COM: JIT-LABEL: @jit_reordered_array_view_test
TEST_CASE("jit_reordered_array_view_test")
{
    const int M = 3;
    const int N = 4;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto At = MakeArray<float>({ N, M });
            auto A = At.Reorder({ 1, 0 }); // M x N

            For(0, M, 1, [&](Scalar i) {
                auto row = A.SubArray({ i, Scalar(0) }, { 1, N });
                For(0, N, 1, [&](Scalar j) {
                    auto iVal = Scalar(Cast(i, ValueType::Int32));
                    auto jVal = Scalar(Cast(j, ValueType::Int32));
                    auto val = Scalar(Cast((iVal * 10) + jVal, ValueType::Float));
                    row(Scalar(0), j) = val;
                });
            });

            // COM: JIT: 0.000000 1.000000 2.000000 3.000000
            // COM: JIT-NEXT: 10.000000 11.000000 12.000000 13.000000
            // COM: JIT-NEXT: 20.000000 21.000000 22.000000 23.000000
            Print(A);
        });
}

// COM: CHECK-LABEL: module @jit_map_reduce_test
// COM: JIT-LABEL: @jit_map_reduce_test
TEST_CASE("jit_map_reduce_test")
{
    const int N = 20;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto A = MakeArray({ N }, ValueType::Int32);
            Nest fillNest(A.Shape());
            Scalar i = fillNest.GetIndices()[0];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                A(i) = iVal;
            });
            fillNest.CreateSchedule();

            // Compute sum-of-squares
            auto x = MapReduce(
                A,
                Scalar(0),
                [&](Scalar a) { return a * a; },
                [&](Scalar a, Scalar p) { return a + p; });
            Print(x);
            // COM: JIT: 2470
        });

    SUCCEED();
}

// COM: CHECK-LABEL: module @jit_reduce_test
// COM: JIT-LABEL: @jit_reduce_test
TEST_CASE("jit_reduce_test")
{
    const int N = 20;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto A = MakeArray({ N }, ValueType::Int32);
            Nest fillNest(A.Shape());
            Scalar i = fillNest.GetIndices()[0];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                A(i) = iVal;
            });
            fillNest.CreateSchedule();

            auto x = Reduce(A, Scalar(0), [&](Scalar a, Scalar p) { return a + p; });
            Print(x);
            // COM: JIT: 190
        });

    SUCCEED();
}


// COM: CHECK-LABEL: module @jit_profile_region_test
// COM: JIT-LABEL: @jit_profile_region_test
TEST_CASE("jit_profile_region_test")
{
    const int M = 200;
    const int N = 200;
    const int K = 200;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            EnterProfileRegion("region");
            // nothing
            ExitProfileRegion("region");

            EnterProfileRegion("region");
            Matrix A = MakeMatrix<float>(M, K);
            Matrix B = MakeMatrix<float>(K, N);
            Matrix C = MakeMatrix<float>(M, N);

            MatrixMatrixMultiply(A, B, C);
            ExitProfileRegion("region");

            PrintProfileResults();
        });

    SUCCEED();
}
#endif // 0

// CHECK-LABEL: module @vectorized_add_test
TEST_CASE("vectorized_add_test")
{
    const int M = 2048;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto A = MakeArray<float>({ M });
            auto B = MakeArray<float>({ M });
            auto C = MakeArray<float>({ M });

            Nest fillNest(A.Shape());
            Scalar i = fillNest.GetIndices()[0];
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));

                // A(i) = Scalar(Cast((iVal * 10), ValueType::Float));
                // B(i) = Scalar(Cast((iVal * 15), ValueType::Float));
                A(i) = 1.0f;
                B(i) = 2.0f;
                C(i) = 0.0f;
            });
            fillNest.CreateSchedule();

            Print("X\n"s);

            Nest computeNest(A.Shape());
            Scalar ii = computeNest.GetIndices()[0];
            computeNest.Set([&]() {
                C(ii) = A(ii) + B(ii);
            });
            auto schedule = computeNest.CreateSchedule();
            auto plan = schedule.CreatePlan();
            plan.Vectorize(ii, { 32, 16 });

            Print("X\n"s);
            Print(C);
        });

    SUCCEED();
}

// CHECK-LABEL: module @vectorized_sum_test
TEST_CASE("vectorized_sum_test")
{
    const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
    const int vectorBytes = vectorSize * 4; // 4 bytes per float
    const int vectorUnits = 16; // AVX-2 has 16 256-bit registers
    const int M = 32;
    const int N = 500;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto A = MakeArray<float>({ M, N });
            auto B = MakeArray<float>({ N });

            Nest fillNest(A.Shape());
            Scalar i, j;
            std::tie(i, j) = fillNest.GetIndices<2>();
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));

                A(i, j) = Scalar(Cast((iVal * 10) + jVal, ValueType::Float));
                B(j) = 0.0f;
            });
            fillNest.CreateSchedule();

            Print("X\n"s);

            Nest computeNest(A.Shape());
            Scalar ii, jj;
            std::tie(ii, jj) = computeNest.GetIndices<2>();
            computeNest.Set([&]() {
                B(jj) += A(ii, jj);
            });
            auto schedule = computeNest.CreateSchedule();
            auto plan = schedule.CreatePlan();
            plan.Vectorize(j, { vectorBytes, vectorUnits });
        });

    SUCCEED();
}

// CHECK-LABEL: module @vectorized_max_test
TEST_CASE("vectorized_max_test")
{
    const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
    const int vectorBytes = vectorSize * 4; // 4 bytes per float
    const int vectorUnits = 16; // AVX-2 has 16 256-bit registers
    const int M = 32;
    const int N = 500;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto At = MakeArray<float>({ N, M });
            auto A = At.Reorder({ 1, 0 }); // M x N
            auto B = MakeArray<float>({ M });

            Nest fillNest(A.Shape());
            Scalar i, j;
            std::tie(i, j) = fillNest.GetIndices<2>();
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));

                A(i, j) = Scalar(Cast((iVal * 10) + jVal, ValueType::Float));
                B(j) = 0.0f;
            });
            fillNest.CreateSchedule();

            Print("X\n"s);

            Nest computeNest(A.Shape());
            Scalar ii, jj;
            std::tie(ii, jj) = computeNest.GetIndices<2>();
            computeNest.Set([&]() {
                auto val1 = A(ii, jj);
                auto val2 = B(ii);
                B(ii) = Max(val1, val2);
            });
            auto schedule = computeNest.CreateSchedule();
            schedule.SetOrder({ jj, ii });
            auto plan = schedule.CreatePlan();
            plan.Vectorize(ii, { vectorBytes, vectorUnits });

            Print(B);
        });

    SUCCEED();
}

// CHECK-LABEL: module @vectorized_exp_test
TEST_CASE("vectorized_exp_test")
{
    const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
    const int vectorBytes = vectorSize * 4; // 4 bytes per float
    const int vectorUnits = 16; // AVX-2 has 16 256-bit registers
    const int M = 32;
    const int N = 500;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto A = MakeArray<float>({ M, N });
            auto B = MakeArray<float>({ M, N });

            Nest fillNest(A.Shape());
            Scalar i, j;
            std::tie(i, j) = fillNest.GetIndices<2>();
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));

                A(i, j) = Scalar(Cast((iVal * 10) + jVal, ValueType::Float));
                B(i, j) = 0.0f;
            });
            fillNest.CreateSchedule();

            Print("X\n"s);

            Nest computeNest(A.Shape());
            Scalar ii, jj;
            std::tie(ii, jj) = computeNest.GetIndices<2>();
            computeNest.Set([&]() {
                B(ii, jj) = FastExp(A(ii, jj));
            });
            auto schedule = computeNest.CreateSchedule();

            schedule.SetOrder({ ii, jj });
            auto plan = schedule.CreatePlan();
            plan.Vectorize(jj, { vectorBytes, vectorUnits });

            Print(B);
        });

    SUCCEED();
}

// CHECK-LABEL: module @softmax_test
TEST_CASE("softmax_test")
{
    const int M = 32;
    const int N = 500;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {

#if 1
            auto A = MakeArray<float>({ M, N });
#else
            auto At = MakeArray<float>({ N, M });
            auto A = At.Reorder({ 1, 0 });
#endif
            Nest fillNest(A.Shape());
            Scalar i, j;
            std::tie(i, j) = fillNest.GetIndices<2>();
            fillNest.Set([&]() {
                auto iVal = Scalar(Cast(i, ValueType::Int32));
                auto jVal = Scalar(Cast(j, ValueType::Int32));

                A(i, j) = Scalar(Cast((iVal * 10) + jVal, ValueType::Float));
            });
            fillNest.CreateSchedule();

            Print("X\n"s);

            SoftmaxifyRowsVectorized(A);

            Print("X\n"s);
            Print(A);
        });

    SUCCEED();
}

// CHECK-LABEL: module @jit_reduce_n_test
// JIT-LABEL: @jit_reduce_n_test
TEST_CASE("jit_reduce_n_test")
{
    const int N = 20;

    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto x = ReduceN(N, Scalar(0), [&](Scalar i, Scalar p) { return Scalar(Cast(i, ValueType::Int32)) + p; });
            Print(x);
            // JIT: 190
        });

    SUCCEED();
}

// CHECK-LABEL: module @jit_get_time_test
// JIT-LABEL: @jit_get_time_test
TEST_CASE("jit_get_time_test")
{
    DeclareFunction("main")
        .Public(true)
        .Decorated(false)
        .Define([=]() {
            auto t = GetTime();
            Print(t);
        });

    SUCCEED();
}

int main(int argc, char* argv[])
{
    llvm::InitLLVM initLLVM(argc, argv);

    int result = Catch::Session().run(argc, argv);

    return result;
}
