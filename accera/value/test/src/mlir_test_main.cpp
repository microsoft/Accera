////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <value/include/EmitterContext.h>
#include <value/include/MLIREmitterContext.h>
#include <value/include/Matrix.h>
#include <value/include/Nest.h>
#include <value/include/Schedule.h>

#include <testing/include/testing.h>

#include <mlirHelpers/include/InitializeMLIR.h>

#include <llvm/Support/InitLLVM.h>

#include <cstdio>

using namespace accera::value;
using namespace accera::utilities;

#define PRINT_AND_VERIFY                                                        \
    do                                                                          \
    {                                                                           \
        guard.GetContext().verify();                                            \
        std::printf("Verification of MLIRModule for %s succeeded\n", __func__); \
        guard.GetContext().print();                                             \
        guard.GetContext().save(std::string{ __func__ } + ".mlir");             \
        std::puts("\n\n");                                                      \
    } while (false)

/*

Can be tested with

for x in *.mlir; do (
    ../../llvm-project/build/install/bin/mlir-opt
        --mlir-print-stacktrace-on-diagnostic
        --convert-std-to-llvm-use-alloca
        --canonicalize
        --convert-linalg-to-affine-loops
        --convert-linalg-to-llvm
        --convert-std-to-llvm
        $x
    | ../../llvm-project/build/bin/mlir-translate
    --mlir-to-llvmir
    | ../../llvm-project/build/install/bin/opt
    -O3
    -fp-contract=fast
    | ../../llvm-project/build/install/bin/llc
    -O3
    -relocation-model=pic
    -fp-contract=fast
    -o ${x:r}.S
) ; done




*/

void mlir_test1()
{
    ContextGuard<MLIRContext> guard(__func__);

    PRINT_AND_VERIFY;
}

void mlir_test2()
{
    ContextGuard<MLIRContext> guard(__func__);

    auto global1 = GlobalAllocate<float>("global1", MemoryLayout({ 10, 20 }));
    auto global2 = GlobalAllocate<int>("global2", std::vector<int>{ 1, 2, 3, 4 }, MemoryLayout({ 2, 2 }));

    PRINT_AND_VERIFY;
}

void mlir_test3()
{
    ContextGuard<MLIRContext> guard(__func__);

    DeclareFunction("foo")
        .Parameters(Value({ ValueType::Int32, ScalarLayout }))
        .Define([](Scalar i) {
            auto x = MakeScalar<int>();
        });

    PRINT_AND_VERIFY;
}

void mlir_test4()
{
    ContextGuard<MLIRContext> guard(__func__);

    DeclareFunction("foo")
        .Parameters(Value({ ValueType::Int32, MemoryLayout({ 10, 10 }) }))
        .Define([](Matrix m) {
            For(m, [](Scalar x, Scalar y) {
                auto i = MakeScalar<int>();
            });
        });

    PRINT_AND_VERIFY;
}

void mlir_test5()
{
    ContextGuard<MLIRContext> guard(__func__);

    auto bar = DeclareFunction("bar")
                   .Parameters(Value({ ValueType::Int32, ScalarLayout }))
                   .Define([](Scalar i) {
                       auto l = StaticAllocate<int>("foo", std::vector{ 1, 2, 3, 4 });
                       auto x = MakeVector<float>(100);
                   });

    DeclareFunction("foo")
        .Parameters(Value({ ValueType::Int32, MemoryLayout({ 10, 10 }) }))
        .Define([&bar](Matrix m) {
            For("matrix_loop", m, [&bar](Scalar x, Scalar y) {
                auto i = MakeScalar<int>();
                bar(i);
            });
        });

    PRINT_AND_VERIFY;
}

void mlir_test6()
{
    ContextGuard<MLIRContext> guard(__func__);

    auto bar = DeclareFunction("bar")
                   .Parameters(Value({ ValueType::Int32, MemoryLayout({ 5, 7 }) }))
                   .Define([](Matrix m) {
                       auto copyM = MakeMatrix<int>(5, 7);
                       copyM = m;
                   });

    DeclareFunction("foo")
        .Parameters(Value({ ValueType::Int32, MemoryLayout({ 5, 7 }) }))
        .Define([&bar](Matrix m) {
            bar(m);
        });

    PRINT_AND_VERIFY;
}

void mlir_test7()
{
    ContextGuard<MLIRContext> guard(__func__);

    DeclareFunction("constant_scalar_add")
        .Define([] {
            Scalar s1 = 10;
            Scalar s2 = 20;
            Scalar s3 = s1.Copy();
            Scalar s4 = s1 + s2;
        });

    PRINT_AND_VERIFY;
}

// test with
// mlir-opt --mlir-print-stacktrace-on-diagnostic --convert-std-to-llvm-use-alloca
//      --canonicalize --convert-linalg-to-affine-loops
//      --convert-linalg-to-llvm --convert-std-to-llvm mlir_test8.mlir |
// mlir-translate --mlir-to-llvmir |
// llc -O3
void mlir_test8()
{
    ContextGuard<MLIRContext> guard(__func__);

    DeclareFunction("MatMatElemwiseSum")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout({ 256, 256 }) }),
            Value({ ValueType::Float, MemoryLayout({ 256, 256 }) }),
            Value({ ValueType::Float, MemoryLayout({ 256, 256 }) }))
        .Define([](Matrix m1, Matrix m2, Matrix m3) {
            m3 = m1 + m2;
        });

    PRINT_AND_VERIFY;
}

void mlir_test9()
{
    ContextGuard<MLIRContext> guard(__func__);

    DeclareFunction("constant_scalar_assign")
        .Define([] {
            Scalar a = MakeScalar<int>("a");
            Scalar c = 4;
            a = c;
        });

    PRINT_AND_VERIFY;
}

void mlir_test10()
{
    ContextGuard<MLIRContext> guard(__func__);

    DeclareFunction("constant_scalar_add")
        .Define([] {
            auto m = MakeMatrix<int>(3, 3);
            auto s = m(1, 1);
            Scalar c = 10;
            s = c;
        });

    PRINT_AND_VERIFY;
}

void mlir_test11()
{
    ContextGuard<MLIRContext> guard(__func__);

    DeclareFunction("constant_scalar_test")
        .Define([] {
            Scalar a = MakeScalar<int>("a");
            Scalar c = 4;
            If(c == 4, [&] { a = 2; });
        });

    PRINT_AND_VERIFY;
}

void mlir_test12()
{
    ContextGuard<MLIRContext> guard(__func__);

    DeclareFunction("constant_scalar_test")
        .Define([] {
            Scalar a = MakeScalar<int>("a");
            Scalar c = 4;
            If(c == 4, [&] { a = 2; }).ElseIf(c == 3, [&] { a = 7; }).Else([&] { a = 11; });
            If(a == 2, [&] { a = 13; }).ElseIf(a == 7, [&] { a = 17; }).Else([&] { a = 19; });
        });

    PRINT_AND_VERIFY;
}

void mlir_test13()
{
    ContextGuard<MLIRContext> guard(__func__);

    DeclareFunction("MatMatMul")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout({ 256, 256 }) }),
            Value({ ValueType::Float, MemoryLayout({ 256, 256 }) }),
            Value({ ValueType::Float, MemoryLayout({ 256, 256 }) }))
        .Define([](Matrix m1, Matrix m2, Matrix m3) {
            auto M = (int)m3.Rows();
            auto N = (int)m3.Columns();
            auto K = (int)m1.Columns();
            auto sum = MakeScalar<float>("inner_sum");
            ForRange(M, [&](Scalar m) {
                ForRange(N, [&](Scalar n) {
                    sum = 0.f;
                    ForRange(K, [&](Scalar k) {
                        sum += m1(m, k) * m2(k, n);
                    });
                    m3(m, n) = sum;
                });
            });
        });

    PRINT_AND_VERIFY;
}

void mlir_nest_test()
{
    ContextGuard<MLIRContext> guard(__func__);

    const int M = 8;
    const int N = 10;
    const int K = 11;

    DeclareFunction("NestMatMul")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout({ M, K }) }),
            Value({ ValueType::Float, MemoryLayout({ K, N }) }),
            Value({ ValueType::Float, MemoryLayout({ M, N }) }))
        .Define([](Matrix A, Matrix B, Matrix C) {
            Nest matmul({ M, N, K });
            auto indices = matmul.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];
            matmul.Set([&]() { C(i, j) += A(i, k) * B(k, j); });
        });

    PRINT_AND_VERIFY;
}

void mlir_schedule_test()
{
    ContextGuard<MLIRContext> guard(__func__);

    const int M = 8;
    const int N = 10;
    const int K = 11;

    DeclareFunction("NestMatMul")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout({ M, K }) }),
            Value({ ValueType::Float, MemoryLayout({ K, N }) }),
            Value({ ValueType::Float, MemoryLayout({ M, N }) }))
        .Define([](Matrix A, Matrix B, Matrix C) {
            Nest matmul({ M, N, K });
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

    PRINT_AND_VERIFY;
}

void mlir_matrix_test()
{
    ContextGuard<MLIRContext> guard(__func__);

    DeclareFunction("Test")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout({ 10, 10 }) }))
        .Define([](Matrix A) {
            A(3, 4) = 1.0f;
        });

    PRINT_AND_VERIFY;
}

int main(int argc, char** argv)
{
    llvm::InitLLVM initLLVM(argc, argv);
    accera::mlirHelpers::InitializeMLIR initMLIR;

    mlir_test1();
    mlir_test2();
    mlir_test3();
    mlir_test4();
    mlir_test5();
    mlir_test6();
    mlir_test7();
    mlir_test8();
    mlir_test9();
    mlir_test10();
    mlir_test11();
    mlir_test12();
    mlir_test13();
    mlir_nest_test();
    mlir_schedule_test();
    mlir_matrix_test();

    return 0;
}
