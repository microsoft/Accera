////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <optional>
#define SNAPSHOT_ENABLED 1

#include <catch2/catch.hpp>

#include <ir/include/DialectRegistry.h>
#include <ir/include/value/ValueDialect.h>

#include <utilities/include/MathUtil.h>
#include <value/include/EmitterContext.h>
#include <value/include/FunctionDeclaration.h>
#include <value/include/Kernel.h>
#include <value/include/MLIREmitterContext.h>
#include <value/include/Nest.h>
#include <value/include/Plan.h>
#include <value/include/Schedule.h>

#include <transforms/include/AcceraPasses.h>

#include <mlir/Analysis/LoopAnalysis.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/LoopUtils.h>

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/SourceMgr.h>

using namespace std::literals;
using namespace llvm;
using namespace mlir;

namespace lntr = accera::transforms::loopnest;
namespace xptr = accera::transforms::executionPlan;

#if SNAPSHOT_ENABLED
#define SNAPSHOT(pm, ...) pm.addPass(createLocationSnapshotPass(OpPrintingFlags{}.enableDebugInfo(), __VA_ARGS__))
#else
#define SNAPSHOT(pm, ...)
#endif // SNAPSHOT_ENABLED

namespace v = accera::ir::value;

namespace
{
enum ConversionTarget
{
    accera,
    mlir,
    llvm
};

std::string stringify(ConversionTarget target)
{
    switch (target)
    {
    case ConversionTarget::accera:
        return "accera";
    case ConversionTarget::mlir:
        return "mlir";
    case ConversionTarget::llvm:
        return "llvm";
    }
    llvm_unreachable("unexpected");
}

// Simple helper function that returns a string as printed from a op.
template <typename T>
static std::string debugString(T& op)
{
    std::string instr_str;
    llvm::raw_string_ostream os(instr_str);
    mlir::OpPrintingFlags flags;
    op.print(os, flags);
    return os.str();
}

struct Fixture
{
protected:
    MLIRContext mlirContext;
    llvm::SourceMgr sourceMgr;

private:
    mlir::OwningModuleRef _ownedModule;

protected:
    mlir::ModuleOp module;
    OpBuilder b;
    OpBuilder::InsertionGuard guard;
    Location loc;
    accera::value::ContextGuard<accera::value::MLIRContext> _valueContext;

public:
    Fixture() :
        _ownedModule(mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlirContext), llvm::StringRef("test_module"))),
        module(*_ownedModule),
        b(&mlirContext),
        guard(b),
        loc(b.getUnknownLoc()),
        _valueContext(module)
    {
        b.setInsertionPoint(module.getBody(), module.getBody()->begin());
        mlirContext.appendDialectRegistry(accera::ir::GetDialectRegistry());
        mlirContext.loadAllAvailableDialects();

        using namespace accera::value;
        auto moduleName = module.getName()->str();
        (void)DeclareFunction(moduleName + "_initialize")
            .Public(false)
            .Decorated(false)
            .Define([]() {
                // No common initialization currently, but later passes may add initialization steps
            });
        (void)DeclareFunction(moduleName + "_deinitialize")
            .Public(false)
            .Decorated(false)
            .Define([]() {
                // No common de-initialization currently, but later passes may add de-initialization steps
            });
    }

    v::ValueModuleOp GetValueModuleOp()
    {
        return *(module.getOps<v::ValueModuleOp>().begin());
    }

    void RunConversionPasses(ConversionTarget target, const std::string& filename, const accera::transforms::AcceraPassPipelineOptions& options = {})
    {
        mlir::PassManager pm(&mlirContext);

        // TODO: Enable verification after turning scf.for into affine.for
        pm.enableVerifier(false);

        size_t idx = 0;
        pm.addPass(createLocationSnapshotPass(OpPrintingFlags{}.enableDebugInfo(), llvm::formatv("{0}_{1}_initial.mlir", filename, idx++).str()));

        accera::transforms::AcceraPassPipelineOptions opts{};
        opts.copyOptionValuesFrom(options);
        opts.basename = filename;
        accera::transforms::addAcceraToLLVMPassPipeline(pm, opts);

        CHECK(succeeded(mlir::verify(module)));
        if (target > ConversionTarget::accera)
        {
            std::string pipelineDesc;
            llvm::raw_string_ostream ss(pipelineDesc);
            pm.printAsTextualPipeline(ss);
            CHECK(succeeded(pm.run(module)));
            CHECK(succeeded(mlir::verify(module)));
        }
    }
};
} // namespace

TEST_CASE_METHOD(Fixture, "Test1", "[cpu][gpu][ir]")
{
    auto target = GENERATE(v::ExecutionTarget::CPU, v::ExecutionTarget::GPU);
    auto targetStr = stringifyEnum(target).str();

    auto conversionTarget = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    SECTION(targetStr)
    {
        v::ValueModuleOp vModuleOp = GetValueModuleOp();
        OpBuilder::InsertionGuard guard1(b);
        b.setInsertionPoint(vModuleOp.getBody(), vModuleOp.getBody()->begin());

        v::ValueFuncOp f = b.create<v::ValueFuncOp>(loc, "foo", b.getFunctionType({}, {}), target);
        CHECK(f.getRegion().getBlocks().size() == 1);
        CHECK(llvm::hasSingleElement(f->getRegion(0)));
        REQUIRE(succeeded(mlir::detail::verifySymbolTable(f)));

        OpBuilder::InsertionGuard guard2(b);
        b.setInsertionPoint(&f.body().back(), f.body().back().begin());
        (void)b.create<v::ReturnOp>(loc);
        CHECK(succeeded(verify(module)));

        RunConversionPasses(conversionTarget, "Test1_" + targetStr + "_" + stringify(conversionTarget));
    }

    SUCCEED("targeting " << targetStr << " " << stringify(conversionTarget) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "Test2", "[cpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    [[maybe_unused]] auto f =
        DeclareFunction("func_test")
            .Parameters(Value({ ValueType::Float, 0 }, ScalarLayout))
            .Define([](Scalar) {});

    RunConversionPasses(target, "Test2_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "Test3", "[cpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    [[maybe_unused]] auto f =
        DeclareFunction("func_test")
            .Parameters(Value({ ValueType::Float, 0 }, { 1 }))
            .Define([](Vector x) {
                x[0] = 3.2f;
            });

    RunConversionPasses(target, "Test3_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "Test4", "[cpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    [[maybe_unused]] auto f =
        DeclareFunction("func_test")
            .Parameters(Value({ ValueType::Float, 0 }, { 1 }))
            .Define([](Vector x) {
                If(x[0] == 3.2f, [&] {
                    x[0] *= 2.1f;
                }).ElseIf(x[0] == 1.2f, [&] {
                      x[0] -= 0.4f;
                  }).Else([&] {
                    x[0] += 300.2f;
                });
            });

    RunConversionPasses(target, "Test4_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "strided_subvector", "[cpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value;

    [[maybe_unused]] auto f =
        DeclareFunction("func_test")
            .Parameters(Value({ ValueType::Float, 0 }, { 5 }))
            .Define([](Vector x) {
                Vector y = x.SubVector(1, 2, 2);
                Return(y);
            });

    RunConversionPasses(target, "strided_subvector_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "strided_subvector2", "[cpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value;

    const auto N = 16;

    [[maybe_unused]] auto f =
        DeclareFunction("func_test")
            .Parameters(Value({ ValueType::Float, 0 }, { N }))
            .Define([=](Vector x) {
                Vector y = x.SubVector(1, N / 2, 2);
                y(0) = 4.0f;
            });

    RunConversionPasses(target, "strided_subvector2_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "strided_submatrix", "[cpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    const auto N = 16;

    [[maybe_unused]] auto f =
        DeclareFunction("func_test")
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ N, N }) }))
            .Define([](Matrix a) {
                auto slice = a.SubMatrix(1, 1, 2, 2, 1, 1);
                slice(0, 0) = 3.0f;
            });

    RunConversionPasses(target, "strided_submatrix_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "fp32_vector_add", "[cpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value;

    [[maybe_unused]] auto f =
        DeclareFunction("func_test")
            .Public(true)
            .Parameters(
                Value({ ValueType::Float, 0 }, { 2 }),
                Value({ ValueType::Float, 0 }, { 2 }),
                Value({ ValueType::Float, 0 }, { 2 }))
            .Define([](Vector a, Vector b, Vector c) {
                c[0] = a[0] + b[0];
            });

    RunConversionPasses(target, "fp32_vector_add_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "fp16_vector_add", "[cpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value;

    [[maybe_unused]] auto f =
        DeclareFunction("func_test")
            .Public(true)
            .Parameters(
                Value({ ValueType::Float16, 0 }, { 1 }),
                Value({ ValueType::Float16, 0 }, { 1 }),
                Value({ ValueType::Float16, 0 }, { 1 }))
            .Define([](Vector a, Vector b, Vector c) {
                c[0] = a[0] + b[0];
            });

    RunConversionPasses(target, "fp16_vector_add_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "vector_add", "[gpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

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

    RunConversionPasses(target, "vector_sum_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "vector_add_rocm", "[gpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    auto gpu_f1 =
        DeclareFunction("gpu_f1")
            .Target(targets::GPU({ 128, 1, 1 }, { 128, 1, 1 }))
            .Runtime(ExecutionRuntime::ROCM)
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
    accera::transforms::AcceraPassPipelineOptions opts{};
    opts.runtime = accera::value::ExecutionRuntime::ROCM;
    RunConversionPasses(target, "vector_sum_rocm_" + stringify(target), opts);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "reduction_for_1", "[cpu][lang]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    [[maybe_unused]] auto f =
        DeclareFunction("func_test")
            .Parameters(Value(ValueType::Float, MemoryLayout{ { 1 } }))
            .Define([](Vector x) {
                x[0] = ReduceN(10, x[0], [](Scalar, Scalar x) {
                    return x * 0.1f;
                });
            });

    RunConversionPasses(target, "reduction_for_1" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "mlir_nest_test", "[cpu][gpu][nest]")
{
    const int M = 8;
    const int N = 10;
    const int K = 11;

    auto conversionTarget = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);
    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    auto executionTarget = GENERATE(ExecutionTarget{ targets::CPU{} }, ExecutionTarget{ targets::GPU() });
    std::string testName = "mlir_nest_test_" + std::string{ std::visit(VariantVisitor{ [](targets::GPU) { return "gpu"; }, [](targets::CPU) { return "cpu"; } }, executionTarget) };
    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(executionTarget)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=, &testName](Matrix A, Matrix B, Matrix C) {
                Nest matmul({ M, N });
                auto indices = matmul.GetIndices();
                Scalar i = indices[0];
                Scalar j = indices[1];
                matmul.Set([&]() {
                    C(i, j) = ReduceN(K, C(i, j), [&](Scalar k, Scalar accum) {
                        return accum + A(i, k) * B(k, j);
                    });
                });
                auto sched = matmul.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, 4);
                auto [jOuter, jInner] = sched.Split(j, 4);

                SECTION("unrolled")
                {
                    testName += "_unrolled";
                    sched.Unroll(iInner);
                }
                SECTION("unroll_and_jammed")
                {
                    testName += "_unroll_and_jammed";
                    sched.InterleavedUnroll(iInner, 2);
                }
                std::visit(VariantVisitor{
                               [&](targets::GPU target) { (void)sched.CreateGPUPlan(target); },
                               [&](targets::CPU) { (void)sched.CreatePlan(); } },
                           executionTarget);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &matmul] {
            auto A = MakeMatrix<float>(M, K);
            auto B = MakeMatrix<float>(K, N);
            auto C = MakeMatrix<float>(M, N);
            Scalar c0 = 0;
            Scalar c1 = 1;
            Scalar c2 = 2;
            Scalar value0 = 0.0f;
            Scalar value1 = 1.1f;
            Scalar value2 = 2.2f;
            FillResource(A, value1);
            FillResource(B, value2);
            FillResource(C, value0);

            matmul(A, B, C);

            PrintMemref(A);
            PrintMemref(B);
            PrintMemref(C);
        });

    RunConversionPasses(conversionTarget, testName + "_" + stringify(conversionTarget));
    SUCCEED("targeting " << stringify(conversionTarget) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "barrier_test", "[gpu][lang]")
{
    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    auto f = DeclareFunction("barrier_test")
                 .Target(targets::GPU())
                 .Define([=] {
                     SECTION("One barrier")
                     {
                         GPU::Barrier();
                         SECTION("another barrier")
                         {
                             GPU::Barrier();
                         }
                     }
                 });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &f] {
            f();
        });

    RunConversionPasses(target, "barrier_test_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "constant_fill_early_return", "[gpu]")
{
    const int64_t N = 32;
    const int blockDim = 8;

    auto conversionTarget = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto block = targets::Dim3{ blockDim, blockDim, 1 };
    auto grid = targets::Dim3{ N / blockDim, N / blockDim, 1 };

    auto executionTarget = GENERATE_COPY(ExecutionTarget{ targets::CPU{} }, ExecutionTarget{ targets::GPU{ grid, block } });
    std::string testName = "constant_fill_early_return_";
    testName += std::visit(VariantVisitor{ [](targets::GPU) { return "gpu"; }, [](targets::CPU) { return "cpu"; } }, executionTarget);
    auto constant_fill_index =
        DeclareFunction("NestConstantFillIndex")
            .Target(executionTarget)
            .Parameters(
                Value({ ValueType::Int32, MemoryLayout(MemoryShape{ N, N }) }))
            .Define([=](Matrix Output) {
                Nest n{ { N, N } };

                auto i = n.GetIndices()[0];
                auto j = n.GetIndices()[1];
                n.Set([&] {
                    // since this is a simple test, we use a += operation in place of assignment
                    // this would catch if the kernel is launched multiple times or with more threads
                    // than it should. Using equality would hide a bug such as that one.

                    If(Cast(i, ValueType::Int32) % 2 == 0 || Cast(j, ValueType::Int32) % 2 == 0, [] { Return(); });

                    Output(i, j) += Scalar(99);
                });

                auto sched = n.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, blockDim);
                auto [jOuter, jInner] = sched.Split(j, blockDim);

                std::visit(VariantVisitor{
                               [&, iOuter = iOuter, iInner = iInner, jOuter = jOuter, jInner = jInner](targets::GPU target) {
                                   auto plan = sched.CreateGPUPlan(target);
                                   plan.MapIndexToProcessor(iOuter, Processor::BlockX);
                                   plan.MapIndexToProcessor(jOuter, Processor::BlockY);
                                   plan.MapIndexToProcessor(iInner, Processor::ThreadX);
                                   plan.MapIndexToProcessor(jInner, Processor::ThreadY);
                               },
                               [&](targets::CPU) { (void)sched.CreatePlan(); } },
                           executionTarget);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &constant_fill_index] {
            auto Output = MakeMatrix<int>(N, N);
            Scalar value = 0;
            FillResource(Output, value);

            constant_fill_index(Output);

            PrintMemref(Output);
        });

    RunConversionPasses(conversionTarget, testName + "_" + stringify(conversionTarget));
    SUCCEED("targeting " << stringify(conversionTarget) << ":\n\n"
                         << debugString(module));
}

// this just fills the matrix with the constant 99.
// you should not expect any zeros in the output
TEST_CASE_METHOD(Fixture, "constant_fill_index_gpu", "[gpu]")
{
    const int64_t N = 32;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU();
    const int blockDim = 8;
    gpuConfig.block = targets::Dim3{ blockDim, blockDim, 1 };
    gpuConfig.grid = targets::Dim3{ N / blockDim, N / blockDim, 1 };
    auto constant_fill_index =
        DeclareFunction("NestConstantFillIndex")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Int32, MemoryLayout(MemoryShape{ N, N }) }))
            .Define([=](Matrix Output) {
                Nest n{ { N, N } };

                auto i = n.GetIndices()[0];
                auto j = n.GetIndices()[1];
                n.Set([&] {
                    // since this is a simple test, we use a += operation in place of assignment
                    // this would catch if the kernel is launched multiple times or with more threads
                    // than it should. Using equality would hide a bug such as that one.
                    Output(i, j) += Scalar(99);
                });

                auto sched = n.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, blockDim);
                auto [jOuter, jInner] = sched.Split(j, blockDim);

                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(iOuter, Processor::BlockX);
                plan.MapIndexToProcessor(jOuter, Processor::BlockY);
                plan.MapIndexToProcessor(iInner, Processor::ThreadX);
                plan.MapIndexToProcessor(jInner, Processor::ThreadY);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &constant_fill_index] {
            auto Output = MakeMatrix<int>(N, N);
            Scalar value = 0;
            FillResource(Output, value);

            constant_fill_index(Output);

            PrintMemref(Output);
        });

    RunConversionPasses(target, "constant_fill_index_gpu_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

// this just fills the matrix with the constant 99 if the row is even and 33 if it is odd.
// you should not expect any zeros in the output
// since this is a simple test, we use a += operation in place of assignment
// this would catch if the kernel is launched multiple times or with more threads
// than it should. Using equality would hide a bug such as that one.
TEST_CASE_METHOD(Fixture, "constant_fill_index_pattern_gpu", "[gpu]")
{
    const int64_t N = 32;
    const int64_t blockDim = 8;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU();
    gpuConfig.block = targets::Dim3{ blockDim, blockDim, 1 };
    gpuConfig.grid = targets::Dim3{ N / blockDim, N / blockDim, 1 };
    auto constant_fill_index =
        DeclareFunction("NestConstantFillIndex")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Int32, MemoryLayout(MemoryShape{ N, N }) }))
            .Define([=](Matrix Output) {
                Nest n{ { N, N } };

                auto i = n.GetIndices()[0];
                auto j = n.GetIndices()[1];
                n.Set([&] {
                    Scalar zero = Cast(0, ValueType::Int64);
                    Scalar two = Cast(2, ValueType::Int64);
                    auto modValue = Modulo(Cast(i, ValueType::Int64), two);
#if 0 // does not currently work because of issues in serializing the mlir into spirv (this is a bug in mlir itself)
                    If(modValue == zero, [&]() {
                        Output(i, j) += Scalar(99);
                    }).Else([&]() {
                        Output(i, j) += Scalar(33);
                    });
#else
                    If(modValue == zero, [&]() {
                        Output(i, j) += Scalar(99);
                    }).ElseIf(modValue != zero, [&]() {
                        Output(i, j) += Scalar(33);
                    });
#endif
                });

                auto sched = n.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, blockDim);
                auto [jOuter, jInner] = sched.Split(j, blockDim);

                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(iOuter, Processor::BlockX);
                plan.MapIndexToProcessor(jOuter, Processor::BlockY);
                plan.MapIndexToProcessor(iInner, Processor::ThreadX);
                plan.MapIndexToProcessor(jInner, Processor::ThreadY);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &constant_fill_index] {
            auto Output = MakeMatrix<int>(N, N);
            Scalar value = 0;
            FillResource(Output, value);

            constant_fill_index(Output);

            PrintMemref(Output);
        });

    RunConversionPasses(target, "constant_fill_index_pattern_gpu_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

// this just fills the matrix so that each row has the row-index (threadIdx.y + blockIdx.y * blockDim.y or global_id(1))
// you should expect a matrix of the form
// [1 1 ... 1]
// [2 2 ... 2]
// ...
// [31 31 ... 31]
TEST_CASE_METHOD(Fixture, "horizontal_fill_index_gpu", "[gpu]")
{
    const int64_t N = 32;
    const int64_t blockDim = 8;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU();
    gpuConfig.block = targets::Dim3{ blockDim, blockDim, 1 };
    gpuConfig.grid = targets::Dim3{ N / blockDim, N / blockDim, 1 };
    auto horizontal_fill_index =
        DeclareFunction("NestHorizontalFillIndex")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Int32, MemoryLayout(MemoryShape{ N, N }) }))
            .Define([=](Matrix Output) {
                Nest n{ { N, N } };

                auto i = n.GetIndices()[0];
                auto j = n.GetIndices()[1];
                n.Set([&] {
                    Output(i, j) = Cast(i, ValueType::Int32);
                });

                auto sched = n.CreateSchedule();
                auto plan = sched.CreateGPUPlan(gpuConfig);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &horizontal_fill_index] {
            auto Output = MakeMatrix<int>(N, N);
            Scalar value = 0;
            FillResource(Output, value);

            horizontal_fill_index(Output);

            PrintMemref(Output);
        });

    RunConversionPasses(target, "horizontal_fill_index_gpu_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

// this just fills the matrix so that each row has the row-index (threadIdx.y + blockIdx.y * blockDim.y or global_id(1))
// you should expect a matrix of the form
// [1 2 ... 31]
// [1 2 ... 31]
// ...
// [1 2 ... 31]
TEST_CASE_METHOD(Fixture, "vertical_fill_index_gpu", "[gpu]")
{
    const int64_t N = 32;
    const int64_t blockDim = 8;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU();
    gpuConfig.block = targets::Dim3{ blockDim, blockDim, 1 };
    gpuConfig.grid = targets::Dim3{ N / blockDim, N / blockDim, 1 };
    auto vertical_fill_index =
        DeclareFunction("NestVerticalFillIndex")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Int32, MemoryLayout(MemoryShape{ N, N }) }))
            .Define([=](Matrix Output) {
                Nest n{ { N, N } };

                auto i = n.GetIndices()[0];
                auto j = n.GetIndices()[1];
                n.Set([&] {
                    Output(i, j) = Cast(j, ValueType::Int32);
                });

                auto sched = n.CreateSchedule();
                auto plan = sched.CreateGPUPlan(gpuConfig);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &vertical_fill_index] {
            auto Output = MakeMatrix<int>(N, N);
            Scalar value = 0;
            FillResource(Output, value);

            vertical_fill_index(Output);

            PrintMemref(Output);
        });

    RunConversionPasses(target, "vertical_fill_index_gpu_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "matmul_value_gpu", "[gpu]")
{
    const int64_t M = 32;
    const int64_t N = 32;
    const int64_t K = 32;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU();
    gpuConfig.grid = targets::Dim3{ 1, 1, 1 };
    gpuConfig.block = targets::Dim3{ 1, 1, 1 };
    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                auto blockIdX = Cast(GPU::BlockId().X(), ValueType::Int64);
                auto blockIdY = Cast(GPU::BlockId().Y(), ValueType::Int64);
                auto threadIdX = Cast(GPU::ThreadId().X(), ValueType::Int64);
                auto threadIdY = Cast(GPU::ThreadId().Y(), ValueType::Int64);

                Nest n{ { M, N } };

                auto i = n.GetIndices()[0];
                auto j = n.GetIndices()[1];
                n.Set([&] {
                    C(i, j) = ReduceN(K, C(i, j), [&](Scalar k, Scalar temp) {
                        auto mul = A(i, k) * B(k, j);
                        return temp + mul;
                    });
                });

                auto sched = n.CreateSchedule();
                auto plan = sched.CreateGPUPlan(gpuConfig);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &matmul] {
            auto A = MakeMatrix<float>(M, K);
            auto B = MakeMatrix<float>(K, N);
            auto C = MakeMatrix<float>(M, N);
            Scalar c0 = 0;
            Scalar c1 = 1;
            Scalar c2 = 2;
            Scalar value0 = 0.0f;
            Scalar value1 = 1.1f;
            Scalar value2 = 2.2f;
            FillResource(A, value1);
            FillResource(B, value2);
            FillResource(C, value0);

            matmul(A, B, C);

            PrintMemref(A);
            PrintMemref(B);
            PrintMemref(C);
        });

    RunConversionPasses(target, "matmul_value_gpu_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "mlir_nest_test_gpu", "[gpu][nest]")
{
    const int M = 256;
    const int N = 256;
    const int K = 256;
    const int blockDimX = 8;
    const int blockDimY = 8;
    REQUIRE(M % blockDimX == 0);
    REQUIRE(N % blockDimY == 0);

    const int gridDimX = M / blockDimX;
    const int gridDimY = N / blockDimY;
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU();
    gpuConfig.grid = targets::Dim3{ gridDimX, gridDimY, 1 };
    gpuConfig.block = targets::Dim3{ blockDimX, blockDimY, 1 };

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                Nest matmul({ M, N });
                auto indices = matmul.GetIndices();
                Scalar i = indices[0];
                Scalar j = indices[1];

                matmul.Set([&]() {
                    C(i, j) = ReduceN(K, C(i, j), [&](Scalar k, Scalar accum) {
                        return accum + (A(i, k) * B(k, j));
                    });
                });

                auto sched = matmul.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, blockDimY);
                auto [jOuter, jInner] = sched.Split(j, blockDimX);

                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(iOuter, Processor::BlockY);
                plan.MapIndexToProcessor(jOuter, Processor::BlockX);
                plan.MapIndexToProcessor(iInner, Processor::ThreadY);
                plan.MapIndexToProcessor(jInner, Processor::ThreadX);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &matmul] {
            auto A = MakeMatrix<float>(M, K);
            auto B = MakeMatrix<float>(K, N);
            auto C = MakeMatrix<float>(M, N);
            Scalar c0 = 0;
            Scalar c1 = 1;
            Scalar c2 = 2;
            Scalar value0 = 0.0f;
            Scalar value1 = 1.1f;
            Scalar value2 = 2.2f;
            FillResource(A, value1);
            FillResource(B, value2);
            FillResource(C, value0);

            matmul(A, B, C);

            PrintMemref(A);
            PrintMemref(B);
            PrintMemref(C);
        });

    RunConversionPasses(target, "mlir_nest_test_gpu_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "mlir_nest_test_gemm_tiled", "[gpu][nest][main]")
{
    const int64_t M = 32;
    const int64_t N = 32;
    const int64_t K = 32;

    const int64_t blockDim = 8;
    const int64_t tileSize = blockDim;

    REQUIRE(M % tileSize == 0);
    REQUIRE(N % tileSize == 0);

    const int64_t gridDimX = N / blockDim;
    const int64_t gridDimY = M / blockDim;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU();
    gpuConfig.grid = targets::Dim3{ gridDimX, gridDimY, 1 };
    gpuConfig.block = targets::Dim3{ blockDim, blockDim, 1 };

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                Nest matmul({ M, N });
                auto indices = matmul.GetIndices();
                Scalar i = indices[0];
                Scalar j = indices[1];

                matmul.Set([&]() {
                    Scalar tidX = GPU::ThreadId().X();
                    Scalar tidY = GPU::ThreadId().Y();
                    Matrix aSh = MakeMatrix(tileSize, tileSize, A.GetType());
                    Matrix bSh = MakeMatrix(tileSize, tileSize, B.GetType());
                    ForRange(Scalar{ int64_t{ 0 } }, K / tileSize, int64_t{ 1 }, [&](Scalar m) {
                        aSh(tidX, tidY) = A(i, Cast(m, ValueType::Int64) * tileSize + tidX);
                        bSh(tidX, tidY) = B(Cast(m, ValueType::Int64) * tileSize + tidY, j);
                        GPU::Barrier();
                        C(i, j) = ReduceN(tileSize, C(i, j), [&](Scalar k, Scalar accum) {
                            return accum + aSh(tidY, k) * bSh(k, tidX);
                        });
                        GPU::Barrier();
                    });
                });

                auto sched = matmul.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, blockDim);
                auto [jOuter, jInner] = sched.Split(j, blockDim);
                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(iOuter, Processor::BlockY);
                plan.MapIndexToProcessor(jOuter, Processor::BlockX);
                plan.MapIndexToProcessor(iInner, Processor::ThreadY);
                plan.MapIndexToProcessor(jInner, Processor::ThreadX);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &matmul] {
            auto A = MakeMatrix<float>(M, K);
            auto B = MakeMatrix<float>(K, N);
            auto C = MakeMatrix<float>(M, N);
            Scalar c0 = 0;
            Scalar c1 = 1;
            Scalar c2 = 2;
            Scalar value0 = 0.0f;
            Scalar value1 = 1.1f;
            Scalar value2 = 2.2f;
            FillResource(A, value1);
            FillResource(B, value2);
            FillResource(C, value0);

            matmul(A, B, C);

            PrintMemref(C);
        });

    RunConversionPasses(target, "mlir_nest_test_gemm_tiled_");
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

// same as mlir_nest_test_gemm_tiled but uses 2-level reduction to avoid always reducing
// into global memory.
TEST_CASE_METHOD(Fixture, "mlir_nest_test_gemm_tiled_2", "[gpu][nest][main]")
{
    const int64_t M = 32;
    const int64_t N = 32;
    const int64_t K = 32;

    const int64_t blockDim = 8;
    const int64_t tileSize = blockDim;

    REQUIRE(M % tileSize == 0);
    REQUIRE(N % tileSize == 0);

    const int64_t gridDimX = N / blockDim;
    const int64_t gridDimY = M / blockDim;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU();
    gpuConfig.grid = targets::Dim3{ gridDimX, gridDimY, 1 };
    gpuConfig.block = targets::Dim3{ blockDim, blockDim, 1 };

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                Nest matmul({ M, N });
                auto indices = matmul.GetIndices();
                Scalar i = indices[0];
                Scalar j = indices[1];

                matmul.Set([&]() {
                    Scalar tidX = GPU::ThreadId().X();
                    Scalar tidY = GPU::ThreadId().Y();
                    Matrix aSh = MakeMatrix(tileSize, tileSize, A.GetType());
                    Matrix bSh = MakeMatrix(tileSize, tileSize, B.GetType());
                    C(i, j) = ReduceN(K / tileSize, C(i, j), [&](Scalar m0, Scalar accumOuter) {
                        auto m = Cast(m0, ValueType::Int64);
                        aSh(tidY, tidX) = A(i, m * tileSize + tidX);
                        bSh(tidY, tidX) = B(m * tileSize + tidY, j);
                        GPU::Barrier();
                        auto accumInner = ReduceN(tileSize, accumOuter, [&](Scalar k, Scalar accum) {
                            return accum + aSh(tidY, k) * bSh(k, tidX);
                        });
                        GPU::Barrier();
                        return accumInner;
                    });
                });

                auto sched = matmul.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, blockDim);
                auto [jOuter, jInner] = sched.Split(j, blockDim);
                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(iOuter, Processor::BlockY);
                plan.MapIndexToProcessor(jOuter, Processor::BlockX);
                plan.MapIndexToProcessor(iInner, Processor::ThreadY);
                plan.MapIndexToProcessor(jInner, Processor::ThreadX);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &matmul] {
            auto A = MakeMatrix<float>(M, K);
            auto B = MakeMatrix<float>(K, N);
            auto C = MakeMatrix<float>(M, N);
            Scalar c0 = 0;
            Scalar c1 = 1;
            Scalar c2 = 2;
            Scalar value0 = 0.0f;
            Scalar value1 = 1.1f;
            Scalar value2 = 2.2f;
            FillResource(A, value1);
            FillResource(B, value2);
            FillResource(C, value0);

            matmul(A, B, C);

            PrintMemref(C);
        });

    RunConversionPasses(target, "mlir_nest_test_gemm_tiled_2_");
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

// same as mlir_nest_test_gemm_tiled_2 but do not assume that the input size is a multiple of the block dim.
TEST_CASE_METHOD(Fixture, "mlir_nest_test_gemm_tiled_non_multiple", "[gpu][nest][main]")
{
    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    const int64_t M = 37;
    const int64_t N = 31;
    const int64_t K = 61;

    const int64_t blockDim = 8;
    const int64_t tileSize = blockDim;

    const int64_t gridDimX = CeilDiv(N, blockDim);
    const int64_t gridDimY = CeilDiv(M, blockDim);

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    auto gpuConfig = targets::GPU();
    gpuConfig.grid = targets::Dim3{ gridDimX, gridDimY, 1 };
    gpuConfig.block = targets::Dim3{ blockDim, blockDim, 1 };

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                // since we manually perform the padding, we need to make sure
                // that the iteration domain is a multiple of the blockDim.
                // TODO: regardless, the GPU loop-unswitching is incorrect currently
                // which makes up rounding a requirement.
                Nest matmul({ RoundUpToMultiple(M, blockDim), RoundUpToMultiple(N, blockDim) });
                auto indices = matmul.GetIndices();
                Scalar i = indices[0];
                Scalar j = indices[1];

                matmul.Set([&]() {
                    Scalar tidX = GPU::ThreadId().X();
                    Scalar tidY = GPU::ThreadId().Y();
                    Matrix aSh = MakeMatrix(tileSize, tileSize, A.GetType());
                    Matrix bSh = MakeMatrix(tileSize, tileSize, B.GetType());
                    auto iVal = Cast(i, ValueType::Int64);
                    auto jVal = Cast(j, ValueType::Int64);
                    auto zero = Cast(0, ValueType::Float);
                    auto accum = ReduceN(CeilDiv(K, tileSize), zero, [&](Scalar m0, Scalar accumOuter) {
                        auto m = Cast(m0, ValueType::Int64);
                        If(iVal < M && m * tileSize + tidX < K, [&]() {
                            aSh(tidY, tidX) = A(i, m * tileSize + tidX);
                        }).Else([&]() {
                            aSh(tidY, tidX) = zero;
                        });

                        If(jVal < N && m * tileSize + tidY < K, [&]() {
                            bSh(tidY, tidX) = B(m * tileSize + tidY, j);
                        }).Else([&]() {
                            bSh(tidY, tidX) = zero;
                        });
                        GPU::Barrier();
                        auto accumInner = ReduceN(tileSize, accumOuter, [&](Scalar k, Scalar accum) {
                            return accum + aSh(tidY, k) * bSh(k, tidX);
                        });
                        GPU::Barrier();
                        return accumInner;
                    });
                    If(iVal < M && jVal < N, [&]() {
                        C(i, j) += accum;
                    });
                });

                auto sched = matmul.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, blockDim);
                auto [jOuter, jInner] = sched.Split(j, blockDim);
                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(iOuter, Processor::BlockY);
                plan.MapIndexToProcessor(jOuter, Processor::BlockX);
                plan.MapIndexToProcessor(iInner, Processor::ThreadY);
                plan.MapIndexToProcessor(jInner, Processor::ThreadX);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &matmul] {
            auto A = MakeMatrix<float>(M, K);
            auto B = MakeMatrix<float>(K, N);
            auto C = MakeMatrix<float>(M, N);
            Scalar c0 = 0;
            Scalar c1 = 1;
            Scalar c2 = 2;
            Scalar value0 = 0.0f;
            Scalar value1 = 1.1f;
            Scalar value2 = 2.2f;
            FillResource(A, value1);
            FillResource(B, value2);
            FillResource(C, value0);

            matmul(A, B, C);

            PrintMemref(C);
        });

    RunConversionPasses(target, "mlir_nest_test_gemm_tiled_non_multiple");
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

// same as mlir_nest_test_gemm_tiled_2, but allocates twice as much shared memory and double buffers the
// the loads
TEST_CASE_METHOD(Fixture, "mlir_nest_test_gemm_tiled_double_buffer", "[gpu][nest][main]")
{
    const int64_t M = 32;
    const int64_t N = 32;
    const int64_t K = 32;

    const int64_t blockDim = 8;
    const int64_t tileSize = blockDim;

    REQUIRE(M % tileSize == 0);
    REQUIRE(N % tileSize == 0);

    const int64_t gridDimX = N / blockDim;
    const int64_t gridDimY = M / blockDim;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU();
    gpuConfig.grid = targets::Dim3{ gridDimX, gridDimY, 1 };
    gpuConfig.block = targets::Dim3{ blockDim, blockDim, 1 };

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                Nest matmul({ M, N });
                auto indices = matmul.GetIndices();
                Scalar i = indices[0];
                Scalar j = indices[1];

                matmul.Set([&]() {
                    Scalar tidX = GPU::ThreadId().X();
                    Scalar tidY = GPU::ThreadId().Y();
                    // allocate twice as much shared memory to perform double buffering
                    // the shared memory are stored as slice
                    Tensor aSh = MakeTensor(2, tileSize, tileSize, A.GetType());
                    Tensor bSh = MakeTensor(2, tileSize, tileSize, B.GetType());
                    aSh(0, tidY, tidX) = A(i, tidX);
                    bSh(0, tidY, tidX) = B(tidX, j);
                    C(i, j) = ReduceN(K / tileSize, C(i, j), [&](Scalar m0, Scalar accumOuter) {
                        GPU::Barrier();
                        Scalar one = Cast(1, ValueType::Int64); // TODO: maybe there is a better way to do this
                        Scalar two = Cast(2, ValueType::Int64); // or we can add correct coercion to the operators
                        Scalar m = Cast(m0, ValueType::Int64);
                        // chooses if you are using the first or the second buffer slice in the shared matrix
                        auto currBuffer = Modulo(m, two);
                        auto nextBuffer = Modulo(m + one, two);

                        aSh(nextBuffer, tidY, tidX) = A(i, (m + one) * tileSize + tidX);
                        bSh(nextBuffer, tidY, tidX) = B((m + one) * tileSize + tidY, j);
                        auto accumInner = ReduceN(tileSize, accumOuter, [&](Scalar k, Scalar accum) {
                            return accum + aSh(currBuffer, tidY, k) * bSh(currBuffer, k, tidX);
                        });
                        return accumInner;
                    });
                });

                auto sched = matmul.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, blockDim);
                auto [jOuter, jInner] = sched.Split(j, blockDim);
                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(iOuter, Processor::BlockY);
                plan.MapIndexToProcessor(jOuter, Processor::BlockX);
                plan.MapIndexToProcessor(iInner, Processor::ThreadY);
                plan.MapIndexToProcessor(jInner, Processor::ThreadX);
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &matmul] {
            auto A = MakeMatrix<float>(M, K);
            auto B = MakeMatrix<float>(K, N);
            auto C = MakeMatrix<float>(M, N);
            Scalar c0 = 0;
            Scalar c1 = 1;
            Scalar c2 = 2;
            Scalar value0 = 0.0f;
            Scalar value1 = 1.1f;
            Scalar value2 = 2.2f;
            FillResource(A, value1);
            FillResource(B, value2);
            FillResource(C, value0);

            matmul(A, B, C);

            PrintMemref(C);
        });

    RunConversionPasses(target, "mlir_nest_test_gemm_tiled_double_buffer");
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "basic_nest", "[cpu][nest]")
{
    const int M = 32;
    const int N = 32;
    const int K = 32;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    DeclareFunction("NestMatMul")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
        .Define([=](Matrix A, Matrix B, Matrix C) {
            Nest nest{ MemoryShape{ M } };
        });

    RunConversionPasses(target, "basic_nest_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

static std::tuple<int, int, int> GetOuterStrides(int M, int N, int K)
{
    const int maxStride = 64;

    int StrideM = std::min<int>(M, maxStride);
    int StrideN = std::min<int>(N, maxStride);
    int StrideK = std::min<int>(K, maxStride);

    return { StrideM, StrideN, StrideK };
}

TEST_CASE_METHOD(Fixture, "basic_gemm_loopnest", "[cpu][nest]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    auto [M_, N_, K_] = GENERATE(std::tuple{ 64, 64, 64 }, std::tuple{ 256, 256, 256 });

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    DeclareFunction("NestMatMul")
        .Public(true)
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M_, K_ }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ K_, N_ }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M_, N_ }) }))
        .Define([=](Matrix A, Matrix B, Matrix C) {
            // Declare and/or calculate constants
            const int M = (int)(A.Rows());
            const int N = (int)(B.Columns());
            const int K = (int)(A.Columns());

            // Define Nest
            Nest nest({ M, N, K });

            // Get indexes
            auto indices = nest.GetIndices();

            auto i = indices[0];
            auto j = indices[1];
            auto k = indices[2];

            auto [OuterStrideM, OuterStrideN, OuterStrideK] = GetOuterStrides(M, N, K);

            auto schedule = nest.CreateSchedule();

            auto computeKernel = Kernel("compute", [&]() {
                C(i, j) += A(i, k) * B(k, j);
            });

            schedule.AddKernel(computeKernel);

            // Tile the nest
            auto [iOuter, iInner] = schedule.Split(i, OuterStrideM);
            auto [jOuter, jInner] = schedule.Split(j, OuterStrideN);
            auto [kOuter, kInner] = schedule.Split(k, OuterStrideK);
            schedule.SetOrder({ iOuter, jOuter, kOuter, jInner, kInner, iInner });
        });

    std::string moduleName = "basic_gemm_loopnest_";
    RunConversionPasses(target, moduleName + std::to_string(M_) + "_" + std::to_string(N_) + "_" + std::to_string(K_) + "_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "matmul_value_gpu_private_mem", "[gpu][lang]")
{
    const int64_t M = 32;
    const int64_t N = 32;
    const int64_t K = 32;
    const int blockDimX = 4;
    const int blockDimY = 4;
    const int gridDimX = N / blockDimX;
    const int gridDimY = M / blockDimY;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU();
    gpuConfig.grid = targets::Dim3{ gridDimX, gridDimY, 1 };
    gpuConfig.block = targets::Dim3{ blockDimX, blockDimY, 1 };

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                auto blockIdX = Cast(GPU::BlockId().X(), ValueType::Int32);
                auto blockIdY = Cast(GPU::BlockId().Y(), ValueType::Int32);
                auto threadIdX = Cast(GPU::ThreadId().X(), ValueType::Int32);
                auto threadIdY = Cast(GPU::ThreadId().Y(), ValueType::Int32);

                auto i = blockIdX * blockDimX + threadIdX;
                auto j = blockIdY * blockDimY + threadIdY;

                Vector accum_ref = Allocate(C.GetType(), MemoryLayout{ { 1 } }.SetMemorySpace(MemorySpace::Private));
                accum_ref[0] = Cast(0, C.GetType());

                ForRange(K, [&](Scalar k) {
                    auto a_ik = A(i, k);
                    auto b_kj = B(k, j);
                    auto t = a_ik * b_kj;

                    auto accum_prev = accum_ref[0];
                    auto accum_next = accum_prev + t;

                    accum_ref[0] = accum_next;
                });

                auto accum = accum_ref[0];
                C(i, j) = accum;
            });

    DeclareFunction("main")
        .Decorated(false)
        .Public(true)
        .Define([=, &matmul] {
            auto A = MakeMatrix<float>(M, K);
            auto B = MakeMatrix<float>(K, N);
            auto C = MakeMatrix<float>(M, N);
            Scalar c0 = 0;
            Scalar c1 = 1;
            Scalar c2 = 2;
            Scalar value0 = 0.0f;
            Scalar value1 = 1.1f;
            Scalar value2 = 2.2f;
            FillResource(A, value1);
            FillResource(B, value2);
            FillResource(C, value0);

            matmul(A, B, C);

            PrintMemref(A);
            PrintMemref(B);
            PrintMemref(C);
        });

    RunConversionPasses(target, "matmul_value_gpu_private_mem_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "gemm_mlas_value", "[cpu][nest]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    auto [M_, N_, K_] = GENERATE(std::tuple{ 32, 32, 32 }, std::tuple{ 64, 64, 64 }, std::tuple{ 256, 256, 256 }, std::tuple{ 49, 128, 256 });

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    DeclareFunction("NestMatMul")
        .Public(true)
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M_, K_ }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ K_, N_ }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M_, N_ }) }))
        .Define([=](Matrix A, Matrix B, Matrix C) {
            // MLAS Value MatrixMatrixMultiply

            // Declare and/or calculate constants
            const int OutputRows = (int)(A.Rows()); // M
            const int OutputColumns = (int)(B.Columns()); // N
            const int InnerDimension = (int)(A.Columns()); // K

            // Schedule constants
            // TODO : read these values from the target system
            int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            int vectorBytes = vectorSize * 4; // 4 bytes per float
            int vectorUnits = 16; // AVX-2 has 16 256-bit registers
            int kUnroll = 4;

            int NumRowsInKernel = 6;
            int NumColumnsInKernel = 2 * vectorSize;

            int columnBlock = std::min(128, OutputColumns);
            int innerDimensionBlock = std::min(256, InnerDimension);

            if (OutputRows < NumRowsInKernel)
            {
                while (NumRowsInKernel > OutputRows)
                {
                    NumRowsInKernel /= 2;
                    NumColumnsInKernel *= 2;
                }
            }
            NumColumnsInKernel = std::min(NumColumnsInKernel, OutputColumns);

            // Define Nest
            Nest nest({ OutputRows, OutputColumns, InnerDimension });

            // Get indexes
            auto indices = nest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];

            nest.Set([&]() { C(i, j) += A(i, k) * B(k, j); });

            auto schedule = nest.CreateSchedule();

            // Declare splits
            auto [jCache, jInner1] = schedule.Split(j, columnBlock);
            auto [kCache, kInner1] = schedule.Split(k, innerDimensionBlock);
            auto [kBlock, kInner2] = schedule.Split(kInner1, kUnroll);
            auto [jKernelOuter2, jInner2] = schedule.Split(jInner1, NumColumnsInKernel);
            auto [jKernelOuter, jInner3] = schedule.Split(jInner2, vectorSize);
            auto [iKernelOuter, iInner] = schedule.Split(i, NumRowsInKernel);

            // Set the order
            schedule.SetOrder({ jCache, kCache, iKernelOuter, jKernelOuter2, kBlock, kInner2, iInner, jKernelOuter, jInner3 });

            auto plan = schedule.CreatePlan();
            std::vector<ScalarIndex> bIndices{ indices[2], indices[1] };
            std::vector<ScalarIndex> cIndices{ indices[0], indices[1] };
            if (OutputColumns > 128 && (OutputColumns * InnerDimension) > (128 * 128))
            {
                plan.AddCache(B, jKernelOuter2);
            }
            plan.AddCache(C, iInner);

            // Set unrolling
            schedule.Unroll(jKernelOuter);
            schedule.Unroll(iInner);
            if (NumColumnsInKernel >= vectorSize)
            {
                plan.Vectorize(jInner3, { vectorBytes, vectorUnits });
            }
        });

    RunConversionPasses(target, "gemm_mlas_value_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

namespace StrassenL1
{
using namespace accera::value;
using namespace accera::utilities;
namespace value = accera::value;

enum class Operation : int
{
    Same,
    Add,
    Subtract
};

// Implements C = A - B
//          or C = A + B
void AddSubtract(accera::value::Matrix A, accera::value::Matrix B, accera::value::Matrix C, Operation op)
{
    using namespace value;

    [[maybe_unused]] int NumRowsInKernel = 6;

    [[maybe_unused]] int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
    [[maybe_unused]] int vectorUnits = 16; // AVX-2 has 16 256-bit registers

    // Declare and/or calculate constants
    const int OutputRows = (int)(A.Rows());
    const int OutputColumns = (int)(A.Columns());

    const std::string kernelId = "StrassenAddSubKernel";

    // Define Nest
    Nest nest({ OutputRows, OutputColumns });

    // Get indexes
    auto indices = nest.GetIndices();
    Scalar i = indices[0];
    Scalar j = indices[1];

    nest.Set([&, op]() {
        // Do either addition or subtraction
        if (op == Operation::Add)
        {
            C(i, j) = A(i, j) + B(i, j);
        }
        else
        {
            C(i, j) = A(i, j) - B(i, j);
        }
    });

    auto schedule = nest.CreateSchedule();
    auto plan = schedule.CreatePlan();
}

// Implements A += C, or A -= C, and B += C or B -= C
void AddSubtract2(accera::value::Matrix A, accera::value::Matrix B, accera::value::Matrix C, Operation op, Operation op2 = Operation::Same)
{
    using namespace value;

    if (op2 == Operation::Same)
    {
        op2 = op;
    }

    [[maybe_unused]] int NumRowsInKernel = 8;

    [[maybe_unused]] int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
    [[maybe_unused]] int vectorUnits = 16; // AVX-2 has 16 256-bit registers

    // Declare and/or calculate constants
    const int OutputRows = (int)(A.Rows());
    const int OutputColumns = (int)(A.Columns());

    const std::string kernelId = "StrassenAddSubKernel2";

    // Define Nest
    Nest nest({ OutputRows, OutputColumns });

    // Get indexes
    auto indices = nest.GetIndices();
    Scalar i = indices[0];
    Scalar j = indices[1];

    nest.Set([&, op, op2]() {
        if (op == Operation::Add)
        {
            if (op2 == Operation::Add)
            {
                A(i, j) += C(i, j);
                B(i, j) += C(i, j);
            }
            else
            {
                A(i, j) += C(i, j);
                B(i, j) -= C(i, j);
            }
        }
        else
        {
            if (op2 == Operation::Add)
            {
                A(i, j) -= C(i, j);
                B(i, j) += C(i, j);
            }
            else
            {
                A(i, j) -= C(i, j);
                B(i, j) -= C(i, j);
            }
        }
    });
    auto schedule = nest.CreateSchedule();
}

void Gemm(accera::value::Matrix A, accera::value::Matrix B, accera::value::Matrix C)
{
    using namespace value;
    // MLAS Value MatrixMatrixMultiply

    // Declare and/or calculate constants
    const int OutputRows = (int)(A.Rows()); // M
    const int OutputColumns = (int)(B.Columns()); // N
    const int InnerDimension = (int)(A.Columns()); // K

    // Schedule constants
    // TODO : read these values from the target system
    int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
    int vectorBytes = vectorSize * 4; // 4 bytes per float
    int vectorUnits = 16; // AVX-2 has 16 256-bit registers
    int kUnroll = 4;

    int NumRowsInKernel = 6;
    int NumColumnsInKernel = 2 * vectorSize;

    int columnBlock = std::min(256, OutputColumns);
    int innerDimensionBlock = std::min(256, InnerDimension);

    if (OutputRows < NumRowsInKernel)
    {
        while (NumRowsInKernel > OutputRows)
        {
            NumRowsInKernel /= 2;
            NumColumnsInKernel *= 2;
        }
    }
    NumColumnsInKernel = std::min(NumColumnsInKernel, OutputColumns);

    // Define Nest
    Nest nest({ OutputRows, OutputColumns, InnerDimension });

    // Get indexes
    auto indices = nest.GetIndices();
    Scalar i = indices[0];
    Scalar j = indices[1];
    Scalar k = indices[2];

    nest.Set([&]() { C(i, j) += A(i, k) * B(k, j); });

    auto schedule = nest.CreateSchedule();

    // Declare splits
    auto [jCache, jInner1] = schedule.Split(j, columnBlock);
    auto [kCache, kInner1] = schedule.Split(k, innerDimensionBlock);
    auto [kBlock, kInner2] = schedule.Split(kInner1, kUnroll);
    auto [jKernelOuter2, jInner2] = schedule.Split(jInner1, NumColumnsInKernel);
    auto [jKernelOuter, jInner3] = schedule.Split(jInner2, vectorSize);
    auto [iKernelOuter, iInner] = schedule.Split(i, NumRowsInKernel);

    // Set the order
    schedule.SetOrder({ jCache, kCache, iKernelOuter, jKernelOuter2, kBlock, kInner2, iInner, jKernelOuter, jInner3 });

    auto plan = schedule.CreatePlan();
    std::vector<ScalarIndex> bIndices{ indices[2], indices[1] };
    std::vector<ScalarIndex> cIndices{ indices[0], indices[1] };
    if (OutputColumns > 128 && (OutputColumns * InnerDimension) > (128 * 128))
    {
        plan.AddCache(B, jKernelOuter2);
    }
    plan.AddCache(C, iInner);

    // Set unrolling
    schedule.Unroll(jKernelOuter);
    schedule.Unroll(iInner);
    if (NumColumnsInKernel >= vectorSize)
    {
        plan.Vectorize(jInner3, { vectorBytes, vectorUnits });
    }
}

#define ZeroMatrix(matrix) For(matrix, [&](Scalar i, Scalar j) { matrix(i, j) = Cast(0, matrix.GetType()); });

// Sample matrix matrix multiply function using Accera
// This function signature matches the one in
// AcceraGemmSampleEmitter::GemmFnType
void Accera_Sample_Function(accera::value::Matrix A, accera::value::Matrix B, accera::value::Matrix C)
{
    using namespace value;

    // Only generate code for matrices that are even
    if (!((A.Rows() % 2 == 0) && (B.Rows() % 2 == 0) && (C.Rows() % 2 == 0) && (A.Columns() % 2 == 0) && (B.Columns() % 2 == 0) && (C.Columns() % 2 == 0)))
    {
        // Don't generate a function body for this
        return;
    }

    // Split A, B and C into sub matrices
    int aRows = static_cast<int>(A.Rows());
    int bRows = static_cast<int>(B.Rows());
    int cRows = static_cast<int>(C.Rows());
    int aCols = static_cast<int>(A.Columns());
    int bCols = static_cast<int>(B.Columns());
    int cCols = static_cast<int>(C.Columns());
    auto A00 = A.SubMatrix(0, 0, aRows / 2, aCols / 2);
    auto A01 = A.SubMatrix(0, aCols / 2, aRows / 2, aCols / 2);
    auto A10 = A.SubMatrix(aRows / 2, 0, aRows / 2, aCols / 2);
    auto A11 = A.SubMatrix(aRows / 2, aCols / 2, aRows / 2, aCols / 2);

    auto B00 = B.SubMatrix(0, 0, bRows / 2, bCols / 2);
    auto B01 = B.SubMatrix(0, bCols / 2, bRows / 2, bCols / 2);
    auto B10 = B.SubMatrix(bRows / 2, 0, bRows / 2, bCols / 2);
    auto B11 = B.SubMatrix(bRows / 2, bCols / 2, bRows / 2, bCols / 2);

    auto C00 = C.SubMatrix(0, 0, cRows / 2, cCols / 2);
    auto C01 = C.SubMatrix(0, cCols / 2, cRows / 2, cCols / 2);
    auto C10 = C.SubMatrix(cRows / 2, 0, cRows / 2, cCols / 2);
    auto C11 = C.SubMatrix(cRows / 2, cCols / 2, cRows / 2, cCols / 2);

    auto aLayout = MemoryLayout(aRows / 2, aCols / 2);
    auto bLayout = MemoryLayout(bRows / 2, bCols / 2);
    auto abLayout = MemoryLayout(aRows / 2, bCols / 2);
    auto ATemp1 = MakeMatrix(aRows / 2, aCols / 2, C.GetType(), "ATemp1");
    auto BTemp1 = MakeMatrix(bRows / 2, bCols / 2, C.GetType(), "BTemp1");
    auto ABTemp1 = MakeMatrix(aRows / 2, bCols / 2, C.GetType(), "ABTemp1");

    // C01 +=
    //          A00 x (B01 - B11)
    // C11 +=
    ZeroMatrix(ABTemp1);
    AddSubtract(B01, B11, BTemp1, Operation::Subtract);
    Gemm(A00, BTemp1, ABTemp1);
    AddSubtract2(C01, C11, ABTemp1, Operation::Add);

    // C00 +=
    //          A11 x (B10 - B00)
    // C10 +=
    ZeroMatrix(ABTemp1);
    AddSubtract(B10, B00, BTemp1, Operation::Subtract);
    Gemm(A11, BTemp1, ABTemp1);
    AddSubtract2(C00, C10, ABTemp1, Operation::Add);

    // C01 +=
    //          (A00 + A01) x B11
    // C00 -=
    ZeroMatrix(ABTemp1);
    AddSubtract(A00, A01, ATemp1, Operation::Add);
    Gemm(ATemp1, B11, ABTemp1);
    AddSubtract2(C01, C00, ABTemp1, Operation::Add, Operation::Subtract);

    // C10 +=
    //          (A10 + A11) x B00
    // C11 -=
    ZeroMatrix(ABTemp1);
    AddSubtract(A10, A11, ATemp1, Operation::Add);
    Gemm(ATemp1, B00, ABTemp1);
    AddSubtract2(C10, C11, ABTemp1, Operation::Add, Operation::Subtract);

    // C00 +=
    //          (A00 + A11) x (B00 + B11)
    // C11 -=
    ZeroMatrix(ABTemp1);
    AddSubtract(A00, A11, ATemp1, Operation::Add);
    AddSubtract(B00, B11, BTemp1, Operation::Add);
    Gemm(ATemp1, BTemp1, ABTemp1);
    AddSubtract2(C00, C11, ABTemp1, Operation::Add, Operation::Add);

    // C11 += (A10 - A00) x (B00 + B01)
    ZeroMatrix(ABTemp1);
    AddSubtract(A10, A00, ATemp1, Operation::Subtract);
    AddSubtract(B00, B01, BTemp1, Operation::Add);
    Gemm(ATemp1, BTemp1, ABTemp1);
    AddSubtract(C11, ABTemp1, C11, Operation::Add);

    // C00 += (A01 - A11) x (B10 + B11)
    ZeroMatrix(ABTemp1);
    AddSubtract(A01, A11, ATemp1, Operation::Subtract);
    AddSubtract(B10, B11, BTemp1, Operation::Add);
    Gemm(ATemp1, BTemp1, ABTemp1);
    AddSubtract(C00, ABTemp1, C00, Operation::Add);
}
} // namespace StrassenL1

TEST_CASE_METHOD(Fixture, "strassen l1", "[cpu][nest][demo]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    auto [M_, N_, K_] = GENERATE(std::tuple{ 64, 64, 64 }, std::tuple{ 256, 256, 256 });

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    DeclareFunction("NestMatMul")
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M_, K_ }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ K_, N_ }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M_, N_ }) }))
        .Define([=](Matrix A, Matrix B, Matrix C) {
            StrassenL1::Accera_Sample_Function(A, B, C);
        });

    RunConversionPasses(target, "strassen_l1_"s + std::to_string(M_) + std::to_string(N_) + std::to_string(K_) + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "logical_operators", "[cpu][nest]")
{
    const int M = 32;
    const int N = 32;
    const int K = 32;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    DeclareFunction("NestLogicalOps")
        .Parameters(
            Value({ ValueType::Int32, MemoryLayout(MemoryShape{ M, K }) }),
            Value({ ValueType::Int32, MemoryLayout(MemoryShape{ K, N }) }),
            Value({ ValueType::Int32, MemoryLayout(MemoryShape{ M, N }) }))
        .Define([=](Matrix A, Matrix B, Matrix C) {
            Nest nest{ MemoryShape{ M, N, K } };

            auto indices = nest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];

            nest.Set([&]() {
                C(i, j) += LogicalNot(A(i, k));
                C(i, j) += A(i, k) || B(k, j);
                C(i, j) += A(i, k) && B(k, j);
            });
        });

    RunConversionPasses(target, "logical_operators_" + stringify(target));
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "basic_conv_skew", "[cpu][nest]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    const int N = 10;
    const int K = 3;
    const int M = N - K + 1;

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    std::string testName = "basic_conv_skew_";

    DeclareFunction("NestBasicConv")
        // .Public(true)
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ N }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ K }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M }) }))
        .Define([=, &testName](Array A, Array B, Array C) {
            const int K = (int)(B.Shape()[0]);
            const int M = (int)(C.Shape()[0]);

            Nest nest({ M, K });

            auto indices = nest.GetIndices();
            auto i = indices[0];
            auto j = indices[1];

            nest.Set([&]() {
                C(i) += A(i + j) * B(j);
            });

            auto schedule = nest.CreateSchedule();

            SECTION("row_col")
            {
                testName += "row_col_";
                schedule.Skew(i, j);
            }
            SECTION("row_col_unroll")
            {
                testName += "row_col_unroll_";
                auto iSkewed = schedule.Skew(i, j);
                schedule.Unroll(iSkewed, 3);
                schedule.Unroll(j, 3);
            }
            SECTION("col_row")
            {
                testName += "col_row_";
                schedule.Skew(j, i);
            }
            SECTION("col_row_unroll")
            {
                testName += "col_row_unroll_";
                auto jSkewed = schedule.Skew(j, i);
                schedule.Unroll(i, 3);
                schedule.Unroll(jSkewed, 3);
            }
        });

    accera::transforms::AcceraPassPipelineOptions options;
    // options.dumpPasses = true;   // set .Public(true) in the function above to see the full IR
    // options.dumpIntraPassIR = true;
    // options.printLoops = true;

    RunConversionPasses(target, testName + stringify(target), options);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "basic_matmul_pad", "[cpu][nest][pad]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    const int M = 63;
    const int N = 34;
    const int K = 18;

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    std::string testName = "basic_matmul_pad";

    DeclareFunction("NestMatmulPad")
        // .Public(true)
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
        .Define([=, &testName](Array A, Array B, Array C) {
            const int M = (int)(A.Shape()[0]);
            const int K = (int)(B.Shape()[0]);
            const int N = (int)(C.Shape()[1]);

            Nest nest({ M, N, K });

            auto indices = nest.GetIndices();
            auto i = indices[0];
            auto j = indices[1];
            auto k = indices[2];

            nest.Set([&]() {
                C(i, j) += A(i, k) * B(k, j);
            });

            auto sched = nest.CreateSchedule();
            auto iPadded = sched.Pad(i, 1);
            auto [iOuter, iInner] = sched.Split(iPadded, 8); // (1+63) // 8 (manual front padding only)
            // Expected iOuter, iInner ranges:
            //  [1, 8:8), [0, 7:1)
            //  [8, 64:8), [0, 8:1)

            auto kPadded = sched.Pad(k, 7);
            auto [kOuter, kInner] = sched.Split(kPadded, 4); // (7+18) // 4 (manual front, automatic back padding)
            // Expected kOuter, kInner ranges:
            //  [7, 8:4), [0, 1:1)
            //  [8, 24:4), [0, 4:1)
            //  [24, 25:4), [0, 1:1)

            auto [jOuter, jInner] = sched.Split(j, 4); // automatic back padding only
            // Expected jOuter, jInner ranges:
            //  [0, 32:4), [0, 4:1)
            //  [32, 34:4), [0, 2:1)
        });

    accera::transforms::AcceraPassPipelineOptions options;
    // options.dumpPasses = true;   // set .Public(true) in the function above to see the full IR
    // options.dumpIntraPassIR = true;
    // options.printLoops = true;

    RunConversionPasses(target, testName + stringify(target), options);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "fused_unequal_shapes", "[cpu][nest][pad]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    const int M = 16;
    const int N0 = 32;
    const int N1 = 20;
    const int N2 = 30;

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value;

    std::string testName = "fused_unequal_shapes_";

    DeclareFunction("FusedUnequalShapes")
        .Public(true)
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N0 }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N0 }) }))
        .Define([=, &testName](Array A, Array B) {
            Nest nest0({ M, N0 });
            auto indices0 = nest0.GetIndices();
            auto i0 = indices0[0];
            auto j0 = indices0[1];
            nest0.Set([&]() {
                A(i0, j0) += B(i0, j0);
            });
            auto sched0 = nest0.CreateSchedule();

            Nest nest1({ M, N1 });
            auto indices1 = nest1.GetIndices();
            auto i1 = indices1[0];
            auto j1 = indices1[1];
            nest1.Set([&]() {
                A(i1, j1) *= B(i1, j1);
            });
            auto sched1 = nest1.CreateSchedule();

            Nest nest2({ M, N2 });
            auto indices2 = nest2.GetIndices();
            auto i2 = indices2[0];
            auto j2 = indices2[1];
            nest2.Set([&]() {
                A(i2, j2) -= B(i2, j2);
            });
            auto sched2 = nest2.CreateSchedule();

            // Fuse when the iteration spaces are not the same shape
            auto jp1 = sched1.Pad(j1, N0 - N1, /*padFront=*/false);
            auto jp2 = sched2.Pad(j2, N0 - N2, /*padFront=*/false);

            std::vector<Schedule> otherScheds{ sched1, sched2 };
            sched0.Fuse(otherScheds, { { i0, i1, i2 }, { j0, jp1, jp2 } });

            auto indices3 = sched0.GetIndices();
            auto f = indices3[0];
            auto i3 = indices3[1];
            auto j3 = indices3[2];
            sched0.SetOrder({ i3, j3, f });

            auto [j3Outer, j3Inner] = sched0.Split(j3, 3);
        });

    accera::transforms::AcceraPassPipelineOptions options;
    // options.dumpPasses = true;   // set .Public(true) in the function above to see the full IR
    // options.dumpIntraPassIR = true;
    // options.printLoops = true;

    RunConversionPasses(target, testName + stringify(target), options);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "fused_unequal_shapes_smaller_first", "[cpu][nest][pad]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    const int M = 16;
    const int N0 = 20; // smaller shape first
    const int N1 = 32;
    const int N2 = 30;

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value;

    std::string testName = "fused_unequal_shapes_smaller_first_";

    // BUGBUG: .Public(true) results in this lowering error (does not reproduce with E2E testing from the Python DSL):
    //
    // fused_unequal_shapes_smaller_first_llvm_9_ConvertSCFToOpenMP.mlir:4:6: error: failed to legalize operation 'accln.sym_index'
    //       %0 = accln.sym_index {name = "j_p"} #accln<"index{j_p,28}"> loc("fused_unequal_shapes_smaller_first_llvm_8_ConvertAffineToStandard.mlir":4:6)
    //      ^
    // fused_unequal_shapes_smaller_first_llvm_9_ConvertSCFToOpenMP.mlir:4:6: note: see current operation: %0 = "accln.sym_index"() {index = #accln<"index{j_p,28}">, name = "j_p"} : () -> index
    //
    DeclareFunction("FusedUnequalShapes")
        // .Public(true)
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N1 }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N1 }) }))
        .Define([=, &testName](Array A, Array B) {
            Nest nest0({ M, N0 });
            auto indices0 = nest0.GetIndices();
            auto i0 = indices0[0];
            auto j0 = indices0[1];
            nest0.Set([&]() {
                A(i0, j0) += B(i0, j0);
            });
            auto sched0 = nest0.CreateSchedule();

            Nest nest1({ M, N1 });
            auto indices1 = nest1.GetIndices();
            auto i1 = indices1[0];
            auto j1 = indices1[1];
            nest1.Set([&]() {
                A(i1, j1) *= B(i1, j1);
            });
            auto sched1 = nest1.CreateSchedule();

            Nest nest2({ M, N2 });
            auto indices2 = nest2.GetIndices();
            auto i2 = indices2[0];
            auto j2 = indices2[1];
            nest2.Set([&]() {
                A(i2, j2) -= B(i2, j2);
            });
            auto sched2 = nest2.CreateSchedule();

            // Fuse when the iteration spaces are not the same shape
            auto jp0 = sched0.Pad(j0, N1 - N0, /*padFront=*/false);
            auto jp2 = sched2.Pad(j2, N1 - N2, /*padFront=*/false);

            std::vector<Schedule> otherScheds{ sched1, sched2 };
            sched0.Fuse(otherScheds, { { i0, i1, i2 }, { jp0, j1, jp2 } });

            auto indices3 = sched0.GetIndices();
            auto f = indices3[0];
            auto i3 = indices3[1];
            auto j3 = indices3[2];
            sched0.SetOrder({ i3, j3, f });

            auto [j3Outer, j3Inner] = sched0.Split(j3, 3);
        });

    accera::transforms::AcceraPassPipelineOptions options;
    // options.dumpPasses = true;   // set .Public(true) in the function above to see the full IR
    // options.dumpIntraPassIR = true;
    // options.printLoops = true;

    RunConversionPasses(target, testName + stringify(target), options);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "fused_unequal_shapes_tiled", "[cpu][nest][pad]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    const int M = 16;
    const int N0 = 16;
    const int N1 = 10;

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value;

    std::string testName = "fused_unequal_shapes_tiled_";

    DeclareFunction("FusedUnequalShapes")
        .Public(true)
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N0 }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N0 }) }))
        .Define([=, &testName](Array A, Array B) {
            Nest nest0({ M, N0 });
            auto indices0 = nest0.GetIndices();
            auto i0 = indices0[0];
            auto j0 = indices0[1];
            nest0.Set([&]() {
                A(i0, j0) += B(i0, j0);
            });
            auto sched0 = nest0.CreateSchedule();

            Nest nest1({ M, N1 });
            auto indices1 = nest1.GetIndices();
            auto i1 = indices1[0];
            auto j1 = indices1[1];
            nest1.Set([&]() {
                A(i1, j1) *= B(i1, j1);
            });
            auto sched1 = nest1.CreateSchedule();
            auto jp1 = sched1.Pad(j1, N0 - N1, /*padFront=*/false);

            std::vector<Schedule> otherScheds{ sched1 };
            sched0.Fuse(otherScheds, { { i0, i1 }, { j0, jp1 } });

            auto indices2 = sched0.GetIndices();
            auto f = indices2[0];
            auto i2 = indices2[1];
            auto j2 = indices2[2];

            auto [i2Outer, i2Inner] = sched0.Split(i2, 4);
            auto [j2Outer, j2Inner] = sched0.Split(j2, 4);

            sched0.SetOrder({ i2Outer, j2Outer, f, i2Inner, j2Inner });
        });

    accera::transforms::AcceraPassPipelineOptions options;
    // options.dumpPasses = true;   // set .Public(true) in the function above to see the full IR
    // options.dumpIntraPassIR = true;
    // options.printLoops = true;

    RunConversionPasses(target, testName + stringify(target), options);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "parallelize_gemm", "[cpu][nest][parallel]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    auto [M_, N_, K_] = GENERATE(std::tuple{ 256, 256, 256 }, std::tuple{ 512, 512, 512 }, std::tuple{ 1024, 1024, 1024 }, std::tuple{ 2048, 2048, 2048 });
    int64_t numThreads = GENERATE(1, 4, 8, 16);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value;

    DeclareFunction("NestMatMul")
        .Public(true)
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M_, K_ }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ K_, N_ }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M_, N_ }) }))
        .Define([=](Array A, Array B, Array C) {
            const int OutputRows = (int)(A.Shape()[0]); // M
            const int OutputColumns = (int)(B.Shape()[1]); // N
            const int InnerDimension = (int)(A.Shape()[1]); // K

            Nest nest({ OutputRows, OutputColumns, InnerDimension });

            auto indices = nest.GetIndices();
            auto i = indices[0];
            auto j = indices[1];
            auto k = indices[2];

            nest.Set([&]() { C(i, j) += A(i, k) * B(k, j); });

            auto schedule = nest.CreateSchedule();

            if (numThreads > 1)
            {
                auto [iOuter, iInner] = schedule.Split(i, OutputRows / numThreads);
                schedule.SetOrder({ iOuter, j, k, iInner });
                auto plan = schedule.CreatePlan();
                plan.Parallelize({ iOuter, j, k }, numThreads, ParallelizationPolicy::Dynamic);
            }
        });

    accera::transforms::AcceraPassPipelineOptions options;
    // options.dumpPasses = true;   // set .Public(true) in the function above to see the full IR
    // options.dumpIntraPassIR = true;

    RunConversionPasses(target, "gemm_parallelized_" + std::to_string(M_) + "_" + std::to_string(N_) + "_" + std::to_string(K_) + "_" + "p" + std::to_string(numThreads) + "_" + stringify(target), options);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "parallelize_gemm_mlas_value", "[cpu][nest]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    auto [M_, N_, K_] = GENERATE(std::tuple{ 256, 256, 256 }, std::tuple{ 512, 512, 512 }, std::tuple{ 1024, 1024, 1024 }, std::tuple{ 2048, 2048, 2048 });
    int64_t numThreads = GENERATE(1, 4, 8, 16);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value, accera::value::Matrix;

    DeclareFunction("NestMatMul")
        .Public(true)
        .Parameters(
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M_, K_ }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ K_, N_ }) }),
            Value({ ValueType::Float, MemoryLayout(MemoryShape{ M_, N_ }) }))
        .Define([=](Matrix A, Matrix B, Matrix C) {
            // MLAS Value MatrixMatrixMultiply

            // Declare and/or calculate constants
            const int OutputRows = (int)(A.Rows()); // M
            const int OutputColumns = (int)(B.Columns()); // N
            const int InnerDimension = (int)(A.Columns()); // K

            // Schedule constants
            // TODO : read these values from the target system
            int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            int vectorBytes = vectorSize * 4; // 4 bytes per float
            int vectorUnits = 16; // AVX-2 has 16 256-bit registers
            int kUnroll = 4;

            int NumRowsInKernel = 6;
            int NumColumnsInKernel = 2 * vectorSize;

            int columnBlock = std::min(128, (int)(OutputColumns / numThreads));
            int innerDimensionBlock = std::min(256, InnerDimension);

            if (OutputRows < NumRowsInKernel)
            {
                while (NumRowsInKernel > OutputRows)
                {
                    NumRowsInKernel /= 2;
                    NumColumnsInKernel *= 2;
                }
            }
            NumColumnsInKernel = std::min(NumColumnsInKernel, OutputColumns);

            // Define Nest
            Nest nest({ OutputRows, OutputColumns, InnerDimension });

            // Get indexes
            auto indices = nest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];

            nest.Set([&]() { C(i, j) += A(i, k) * B(k, j); });

            auto schedule = nest.CreateSchedule();

            // Declare splits
            auto [jCache, jInner1] = schedule.Split(j, columnBlock);
            auto [kCache, kInner1] = schedule.Split(k, innerDimensionBlock);
            auto [kBlock, kInner2] = schedule.Split(kInner1, kUnroll);
            auto [jKernelOuter2, jInner2] = schedule.Split(jInner1, NumColumnsInKernel);
            auto [jKernelOuter, jInner3] = schedule.Split(jInner2, vectorSize);
            auto [iKernelOuter, iInner] = schedule.Split(i, NumRowsInKernel);

            // Set the order
            schedule.SetOrder({ jCache, kCache, iKernelOuter, jKernelOuter2, kBlock, kInner2, iInner, jKernelOuter, jInner3 });

            auto plan = schedule.CreatePlan();
            std::vector<ScalarIndex> bIndices{ indices[2], indices[1] };
            std::vector<ScalarIndex> cIndices{ indices[0], indices[1] };
            if (OutputColumns > 128 && (OutputColumns * InnerDimension) > (128 * 128))
            {
                plan.AddCache(B, jKernelOuter2);
            }
            // plan.AddCache(C, iInner); // BUGBUG: fails correctness

            // Set unrolling
            schedule.Unroll(jKernelOuter);
            schedule.Unroll(iInner);
            if (NumColumnsInKernel >= vectorSize)
            {
                plan.Vectorize(jInner3, { vectorBytes, vectorUnits });
            }

            if (numThreads > 1)
            {
                // parallelize the outermost index
                plan.Parallelize({ jCache }, numThreads, ParallelizationPolicy::Static);
            }
        });

    accera::transforms::AcceraPassPipelineOptions options;
    // options.dumpPasses = true;
    // options.dumpIntraPassIR = true;

    RunConversionPasses(target, "gemm_mlas_value_parallelized_" + std::to_string(M_) + "_" + std::to_string(N_) + "_" + std::to_string(K_) + "_" + "p" + std::to_string(numThreads) + "_" + stringify(target), options);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "unsigned_int_ops", "[cpu][nest]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    const int M = 32;
    const int N = 32;
    const int K = 32;

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value;

    auto type = GENERATE(ValueType::Byte, ValueType::Uint16, ValueType::Uint32, ValueType::Uint64);

    DeclareFunction("NestUnsignedIntOps")
        .Public(true)
        .Parameters(
            Value({ type, MemoryLayout(MemoryShape{ M, K }) }),
            Value({ type, MemoryLayout(MemoryShape{ K, N }) }),
            Value({ type, MemoryLayout(MemoryShape{ M, N }) }))
        .Define([=](Array A, Array B, Array C) {
            Nest nest{ MemoryShape{ M, N, K } };

            auto indices = nest.GetIndices();
            Scalar i = indices[0];
            Scalar j = indices[1];
            Scalar k = indices[2];

            nest.Set([&]() {
                C(i, j) += A(i, k) + B(k, j);
                C(i, j) += A(i, k) - B(k, j);
                C(i, j) += A(i, k) * B(k, j);
                C(i, j) += A(i, k) / B(k, j);
                C(i, j) += A(i, k) % B(k, j);
                C(i, j) += BitwiseNot(C(i, j));
            });
        });

    accera::transforms::AcceraPassPipelineOptions options;
    // options.dumpPasses = true;
    // options.dumpIntraPassIR = true;

    RunConversionPasses(target, "unsigned_int_ops_" + stringify(target), options);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));        
}

// Testbed for unit testing / tweaking the debug_check_all_close function
TEST_CASE_METHOD(Fixture, "debug_check_all_close", "[cpu][nest]")
{
    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    const int M = 32;

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::value::Value;

    auto type = GENERATE(ValueType::Float16, ValueType::Float, ValueType::Double, ValueType::Byte, ValueType::Uint16, ValueType::Uint32, ValueType::Uint64);

    DeclareFunction("CheckAllClose")
        .Public(true)
        .Parameters(
            Value({ type, MemoryLayout(MemoryShape{ M, M }) }),
            Value({ type, MemoryLayout(MemoryShape{ M, M }) }))
        .Define([=](Array actual, Array desired) {

            using namespace std::string_literals;
            auto diff = MakeArray(actual.Shape(), ValueType::Float, "diff");
            auto atol = Scalar(0.0001f);

            auto maxAbsoluteDiff = MakeArray(MemoryShape{ 1 }, diff.GetType(), "maxAbsoluteDiff");
            auto count = MakeArray(MemoryShape{ 1 }, ValueType::Int32, "count");

            auto zero = Scalar(0.0f);
            auto max = Scalar(std::numeric_limits<float>::max());
            auto zeroCount = Cast(Scalar(0), count.GetType());
            auto oneCount = Cast(Scalar(1), count.GetType());
            auto total = Cast(Scalar(actual.Size()), count.GetType());

            Nest nest(actual.Shape());
            auto indices = nest.GetIndices();
            nest.Set([&]() {
                diff(indices) = Cast(actual(indices) - desired(indices), diff.GetType());
                diff(indices) = Clamp(Abs(diff(indices)), zero, max); // over/underflow
                maxAbsoluteDiff(0) = Select(maxAbsoluteDiff(0) >= diff(indices), maxAbsoluteDiff(0), diff(indices));
                count(0) += Select(diff(indices) <= atol, zeroCount, oneCount);
            });

            If(count(0) > zeroCount, [&]() {
                bool toStderr = true;
                Print("\nERROR: Not equal to tolerance: "s, toStderr);
                Print(atol, toStderr);
                Print("\n\nMismatched elements: "s, toStderr);
                Print(count, toStderr);
                Print("/ "s, toStderr);

                Print(total, toStderr);
                Print(" ("s, toStderr);
                auto percent = Scalar(100.0f) * Cast(count(0), ValueType::Float) / Cast(total, ValueType::Float);
                Print(percent, toStderr);
                Print(" %)\nMax absolute difference: "s, toStderr);
                Print(maxAbsoluteDiff, toStderr);

                // TODO: which is more useful, printing a summary or printing the full diff to reveal possible patterns?
                Print("\nDifferences:\n"s, toStderr);
                Print(diff, toStderr);
                Print("\n\n"s, toStderr);
            })
            .Else([&] {
                Print("\nOK (no mismatches detected)\n"s);
            });
    
        });

    accera::transforms::AcceraPassPipelineOptions options;
    // options.dumpPasses = true;
    // options.dumpIntraPassIR = true;

    RunConversionPasses(target, "debug_check_all_close_" + stringify(target), options);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));        
}

TEST_CASE_METHOD(Fixture, "mlir_nest_test_gemm_tiled_mfma_rocm", "[gpu][nest][cache][main]")
{
    const int64_t M = 32;
    const int64_t N = 32;
    const int64_t K = 32;

    const int64_t blockDim = 16;
    const int64_t tileSize = blockDim;

    REQUIRE(M % tileSize == 0);
    REQUIRE(N % tileSize == 0);

    const int64_t gridDimX = N / blockDim;
    const int64_t gridDimY = M / blockDim;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU{};
    gpuConfig.grid = targets::Dim3(gridDimX, gridDimY);
    gpuConfig.block = targets::Dim3(blockDim, blockDim);

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Runtime(ExecutionRuntime::ROCM)
            .Decorated(false)
            .Public(true)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                Nest matmul({ M, N });
                auto indices = matmul.GetIndices();
                Scalar i = indices[0];
                Scalar j = indices[1];

                matmul.Set([&]() {
                    Scalar tidX = GPU::ThreadId().X();
                    Scalar tidY = GPU::ThreadId().Y();

                    auto mfmaAMatrix = MFMALoad(A.GetValue(), { 16, 16 }, "AOp");
                    auto mfmaBMatrix = MFMALoad(B.GetValue(), { 16, 16 }, "BOp");
                    auto mfmaCMatrix = MFMALoad(C.GetValue(), { 16, 16 }, "COp");
                    auto mfmaDMatrix = MFMACompute(mfmaAMatrix, mfmaBMatrix, mfmaCMatrix);
                    MFMAStore(mfmaDMatrix, C.GetValue());
                });

                auto sched = matmul.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, blockDim);
                auto [jOuter, jInner] = sched.Split(j, blockDim);
                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(iOuter, Processor::BlockY);
                plan.MapIndexToProcessor(jOuter, Processor::BlockX);
                plan.MapIndexToProcessor(iInner, Processor::ThreadY);
                plan.MapIndexToProcessor(jInner, Processor::ThreadX);
            });

    accera::transforms::AcceraPassPipelineOptions opts{};
    opts.dumpPasses = true;
    opts.dumpIntraPassIR = false;
    opts.gpuOnly = true;
    opts.runtime = accera::value::ExecutionRuntime::ROCM;

    RunConversionPasses(target, "mlir_nest_test_gemm_tiled_mfma_rocm_", opts);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "mlir_nest_test_tensorize_rocm_single_block_single_warp", "[gpu][nest][mfma][main]")
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;

    const int64_t blockDim = 16;
    const int64_t tileSize = blockDim;

    REQUIRE(M % tileSize == 0);
    REQUIRE(N % tileSize == 0);

    const int64_t outerTileX = 64;
    const int64_t outerTileY = outerTileX;
    const int64_t outerTileK = 64;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU{};
    gpuConfig.grid = targets::Dim3(16, 16);
    gpuConfig.block = targets::Dim3(32, 32);

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Runtime(ExecutionRuntime::ROCM)
            .Decorated(false)
            .Public(true)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                Nest nest({ M, N, K });
                auto indices = nest.GetIndices();
                Scalar i_ = indices[0];
                Scalar j_ = indices[1];
                Scalar k_ = indices[2];

                nest.Set([&]() { C(i_, j_) += A(i_, k_) * B(k_, j_); });

                auto sched = nest.CreateSchedule();
                auto [i, ii_] = sched.Split(i_, outerTileY);
                auto [j, jj_] = sched.Split(j_, outerTileX);
                auto [k, kk_] = sched.Split(k_, outerTileK);
                auto [ii, iii] = sched.Split(ii_, 2);
                auto [jj, jjj] = sched.Split(jj_, 2);
                auto [kk, kkk] = sched.Split(kk_, 16);
                sched.SetOrder({ i, j, k, ii, jj, kk, iii, jjj, kkk });

                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(i, Processor::BlockY);
                plan.MapIndexToProcessor(j, Processor::BlockX);
                plan.MapIndexToProcessor(ii, Processor::ThreadY);
                plan.MapIndexToProcessor(jj, Processor::ThreadX);
            });

    accera::transforms::AcceraPassPipelineOptions opts{};
    opts.dumpPasses = true;
    opts.dumpIntraPassIR = false;
    opts.gpuOnly = true;
    opts.runtime = accera::value::ExecutionRuntime::ROCM;

    RunConversionPasses(target, "mlir_nest_test_tensorize_rocm_single_block_single_warp_", opts);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "test_rocm_cache", "[gpu][nest][cache][main]")
{
    const int64_t M = 32;
    const int64_t N = 32;
    const int64_t K = 32;

    const int64_t blockDim = 16;
    const int64_t tileSize = blockDim;

    REQUIRE(M % tileSize == 0);
    REQUIRE(N % tileSize == 0);

    const int64_t gridDimX = N / blockDim;
    const int64_t gridDimY = M / blockDim;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU{};
    gpuConfig.grid = targets::Dim3(gridDimX, gridDimY);
    gpuConfig.block = targets::Dim3(blockDim, blockDim);

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Runtime(ExecutionRuntime::ROCM)
            .Decorated(false)
            .Public(true)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                Nest matmul({ M, N });
                auto indices = matmul.GetIndices();
                Scalar i = indices[0];
                Scalar j = indices[1];

                matmul.Set([&]() {
                    Scalar tidX = GPU::ThreadId().X();
                    Scalar tidY = GPU::ThreadId().Y();

                    auto mfmaAMatrix = MFMALoad(A.GetValue(), { 16, 16 }, "AOp");
                    auto mfmaBMatrix = MFMALoad(B.GetValue(), { 16, 16 }, "BOp");
                    auto mfmaCMatrix = MFMALoad(C.GetValue(), { 16, 16 }, "COp");
                    auto mfmaDMatrix = MFMACompute(mfmaAMatrix, mfmaBMatrix, mfmaCMatrix);
                    MFMAStore(mfmaDMatrix, C.GetValue());
                });

                auto sched = matmul.CreateSchedule();
                auto [iOuter, iInner] = sched.Split(i, blockDim);
                auto [jOuter, jInner] = sched.Split(j, blockDim);
                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(iOuter, Processor::BlockY);
                plan.MapIndexToProcessor(jOuter, Processor::BlockX);
                plan.MapIndexToProcessor(iInner, Processor::ThreadY);
                plan.MapIndexToProcessor(jInner, Processor::ThreadX);
            });

    accera::transforms::AcceraPassPipelineOptions opts{};
    opts.dumpPasses = true;
    opts.dumpIntraPassIR = false;
    opts.gpuOnly = true;
    opts.runtime = accera::value::ExecutionRuntime::ROCM;

    RunConversionPasses(target, "mlir_nest_test_gemm_tiled_mfma_rocm_", opts);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "test_rocm_cache_double_buffer", "[gpu][nest][cache][main]")
{
    const int64_t M = 1024;
    const int64_t N = 1024;
    const int64_t K = 1024;

    const int64_t outerTileX = 64;
    const int64_t outerTileY = outerTileX;
    const int64_t outerTileK = 64;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU{};
    gpuConfig.grid = targets::Dim3(16, 16);
    gpuConfig.block = targets::Dim3(32, 32);

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Runtime(ExecutionRuntime::ROCM)
            .Decorated(false)
            .Public(true)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                Nest nest({ M, N, K });
                auto indices = nest.GetIndices();
                Scalar i_ = indices[0];
                Scalar j_ = indices[1];
                Scalar k_ = indices[2];

                nest.Set([&]() { C(i_, j_) += A(i_, k_) * B(k_, j_); });

                auto sched = nest.CreateSchedule();
                auto [i, ii_] = sched.Split(i_, outerTileY);
                auto [j, jj_] = sched.Split(j_, outerTileX);
                auto [k, kk_] = sched.Split(k_, outerTileK);
                auto [ii, iii] = sched.Split(ii_, 2);
                auto [jj, jjj] = sched.Split(jj_, 2);
                auto [kk, kkk] = sched.Split(kk_, 16);
                sched.SetOrder({ i, j, k, ii, jj, kk, iii, jjj, kkk });

                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(i, Processor::BlockY);
                plan.MapIndexToProcessor(j, Processor::BlockX);
                plan.MapIndexToProcessor(ii, Processor::ThreadY);
                plan.MapIndexToProcessor(jj, Processor::ThreadX);

                plan.AddCache(A, ii, ii, accera::utilities::DimensionOrder(2), false, true, std::nullopt, CacheIndexing::GlobalToPhysical, CacheAllocation::Automatic, MemorySpace::Shared, MemorySpace::Private);
                plan.AddCache(B, ii, ii, accera::utilities::DimensionOrder(2), false, true, std::nullopt, CacheIndexing::GlobalToPhysical, CacheAllocation::Automatic, MemorySpace::Shared, MemorySpace::Private);
            });

    accera::transforms::AcceraPassPipelineOptions opts{};
    opts.dumpPasses = true;
    opts.dumpIntraPassIR = false;
    opts.gpuOnly = true;
    opts.runtime = accera::value::ExecutionRuntime::ROCM;

    RunConversionPasses(target, "test_rocm_cache_double_buffer_", opts);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}

TEST_CASE_METHOD(Fixture, "test_rocm_cache_tensorize", "[gpu][nest][cache][tensorize][main]")
{
    const int64_t M = 1024;
    const int64_t N = 1024;
    const int64_t K = 1024;

    const int64_t outerTileX = 64;
    const int64_t outerTileY = outerTileX;
    const int64_t outerTileK = 64;

    auto target = GENERATE(ConversionTarget::accera, ConversionTarget::mlir, ConversionTarget::llvm);

    using namespace accera::value;
    using namespace accera::utilities;
    using accera::utilities::MemorySpace;
    using accera::value::Value, accera::value::Matrix;

    auto gpuConfig = targets::GPU{};
    gpuConfig.grid = targets::Dim3(16, 16);
    gpuConfig.block = targets::Dim3(32, 32);

    auto matmul =
        DeclareFunction("NestMatMul")
            .Target(gpuConfig)
            .Runtime(ExecutionRuntime::ROCM)
            .Decorated(false)
            .Public(true)
            .Parameters(
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, K }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ K, N }) }),
                Value({ ValueType::Float, MemoryLayout(MemoryShape{ M, N }) }))
            .Define([=](Matrix A, Matrix B, Matrix C) {
                Nest nest({ M, N, K });
                auto indices = nest.GetIndices();
                Scalar i_ = indices[0];
                Scalar j_ = indices[1];
                Scalar k_ = indices[2];

                nest.Set([&]() { C(i_, j_) += A(i_, k_) * B(k_, j_); });

                auto sched = nest.CreateSchedule();
                auto [i, ii_] = sched.Split(i_, outerTileY);
                auto [j, jj_] = sched.Split(j_, outerTileX);
                auto [k, kk_] = sched.Split(k_, outerTileK);
                auto [ii, iii] = sched.Split(ii_, 2);
                auto [jj, jjj] = sched.Split(jj_, 2);
                auto [kk, kkk] = sched.Split(kk_, 16);
                sched.SetOrder({ i, j, k, ii, jj, kk, iii, jjj, kkk });

                auto plan = sched.CreateGPUPlan(gpuConfig);
                plan.MapIndexToProcessor(i, Processor::BlockY);
                plan.MapIndexToProcessor(j, Processor::BlockX);
                plan.MapIndexToProcessor(ii, Processor::ThreadY);
                plan.MapIndexToProcessor(jj, Processor::ThreadX);

                plan.Tensorize({ iii, jjj, kkk }, { 2, 2, 16 });
                plan.AddCache(A, ii, ii, accera::utilities::DimensionOrder(2), false, false, std::nullopt, CacheIndexing::GlobalToPhysical, CacheAllocation::Automatic, MemorySpace::Shared, MemorySpace::None);
                plan.AddCache(B, ii, ii, accera::utilities::DimensionOrder(2), false, false, std::nullopt, CacheIndexing::GlobalToPhysical, CacheAllocation::Automatic, MemorySpace::Shared, MemorySpace::None);
            });

    accera::transforms::AcceraPassPipelineOptions opts{};
    opts.dumpPasses = true;
    opts.dumpIntraPassIR = true;
    opts.gpuOnly = true;
    opts.runtime = accera::value::ExecutionRuntime::ROCM;

    RunConversionPasses(target, "test_rocm_cache_tensorize_", opts);
    SUCCEED("targeting " << stringify(target) << ":\n\n"
                         << debugString(module));
}