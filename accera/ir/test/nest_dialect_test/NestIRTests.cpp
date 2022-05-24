////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include "IRTestVerification.h"
#include "TestContext.h"

#include <ir/include/DialectRegistry.h>
#include <ir/include/IRUtil.h>
#include <ir/include/InitializeAccera.h>
#include <ir/include/nest/Index.h>
#include <ir/include/nest/IndexRange.h>
#include <ir/include/nest/IterationDomain.h>
#include <ir/include/nest/LoopNestAttributes.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/nest/LoopNestTypes.h>
#include <ir/include/nest/Range.h>
#include <ir/include/nest/TransformedDomain.h>
#include <ir/include/value/ValueDialect.h>

#include <transforms/include/nest/LoopNestPasses.h>

#include <utilities/include/Logger.h>
#include <utilities/include/MemoryLayout.h>
#include <utilities/include/TypeTraits.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace accera::ir;
using namespace loopnest;
using namespace value;
using namespace mlir;

//
// Utility stuff
//

namespace
{

mlir::Value Alloca(mlir::OpBuilder& builder, mlir::MemRefType bufferType)
{
    // TODO : return to using std_alloc when aligned_alloc/_aligned_malloc issue in MLIR is fixed on Windows
    return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), bufferType);
}

mlir::Value Alloca(mlir::OpBuilder& builder, mlir::MemRefType bufferType, mlir::Value value)
{
    // TODO : return to using std_alloc when aligned_alloc/_aligned_malloc issue in MLIR is fixed on Windows
    return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), bufferType, mlir::ValueRange{ value });
}

std::vector<int64_t> GetMatMulDimensions(ShapedType aType, ShapedType bType, ShapedType cType)
{
    assert(aType.getRank() == 2);
    assert(bType.getRank() == 2);
    assert(cType.getRank() == 2);
    assert(aType.hasStaticShape());
    assert(bType.hasStaticShape());
    assert(cType.hasStaticShape());
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();
    auto cShape = cType.getShape();
    assert(aShape[0] == cShape[0]);
    assert(aShape[1] == bShape[0]);
    assert(bShape[1] == cShape[1]);

    return { aShape[0], bShape[1], aShape[1] }; // M, N, K
}

NestOp MakeNest(ArrayRef<int64_t> sizes, std::function<void(mlir::OpBuilder&, mlir::Location, NestOp&)> body)
{
    auto& builder = GetTestBuilder();
    auto nestOp = loopnest::MakeNest(builder, sizes);
    MakeKernel(builder, [&](mlir::OpBuilder& builder, mlir::Location loc) { body(builder, loc, nestOp); });

    return nestOp;
}

NestOp CreateMatMulNestOp(mlir::Value A, mlir::Value B, mlir::Value C)
{
    auto aType = A.getType().dyn_cast<ShapedType>();
    auto bType = B.getType().dyn_cast<ShapedType>();
    auto cType = C.getType().dyn_cast<ShapedType>();
    auto dimensions = GetMatMulDimensions(aType, bType, cType);

    [[maybe_unused]] auto& builder = GetTestBuilder();
    auto nest = MakeNest(dimensions, [&](mlir::OpBuilder& builder, mlir::Location loc, NestOp& nest) {
        auto [i, j, k] = nest.getIndices<3>(builder);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i, j });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ i, k });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ k, j });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ i, j });
    });

    return nest;
}

struct MatMul3Parameters
{
    int M;
    int N;
    int K;
    int L;
    mlir::Value A;
    mlir::Value B;
    mlir::Value C;
    mlir::Value D;
    mlir::Value E;
    // value::Matrix expectedC;
    // value::Matrix expectedE;
};

MatMul3Parameters GetMatMul3Parameters(int M, int N, int K, int L)
{
    auto& builder = GetTestBuilder();
    auto floatType = builder.getF64Type();

    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };
    auto D = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ N, L }, floatType)) };
    auto E = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, L }, floatType)) };
    return { M, N, K, L, A, B, C, D, E };
}

[[maybe_unused]] MatMul3Parameters GetMatMul3ParametersWithCachedTemp(int M, int N, int K, int L, int cacheM, int cacheN)
{
    auto& builder = GetTestBuilder();
    auto floatType = builder.getF64Type();

    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ cacheM, cacheN }, floatType)) };
    auto D = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ N, L }, floatType)) };
    auto E = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, L }, floatType)) };
    return { M, N, K, L, A, B, C, D, E };
}
} // namespace

//
// Misc tests
//
TEST_CASE("UnrankedMemRefTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        auto loc = builder.getUnknownLoc();

        auto int32Type = builder.getIntegerType(32);
        auto memrefType1 = mlir::MemRefType::get({}, int32Type); // scalar <i32>
        auto memrefType2 = mlir::MemRefType::get({ 5 }, int32Type); // <5xi32>
        auto memrefType3 = mlir::MemRefType::get(10, int32Type); // <10xi32>
        auto memrefType4 = mlir::MemRefType::get({ 2, 5 }, int32Type); // <2x5xi32>
        auto memrefType5 = mlir::MemRefType::get({ -1 }, int32Type); // <?xi32>
        auto memrefType6 = mlir::MemRefType::get({ 2, -1 }, int32Type); // <5x?xi32>

        auto memrefType7 = mlir::UnrankedMemRefType::get(int32Type, 0); // <*xi32>

        [[maybe_unused]] auto val1 = Alloca(builder, memrefType1);
        [[maybe_unused]] auto val2 = Alloca(builder, memrefType2);
        mlir::Value val3 = Alloca(builder, memrefType3);
        mlir::Value val4 = Alloca(builder, memrefType4);
        mlir::Value val5 = Alloca(builder, memrefType5, Value{ builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 10) });
        mlir::Value val6 = Alloca(builder, memrefType6, Value{ builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 10) });

        // auto cast_3_4 = builder.create<mlir::memref::CastOp>(loc, val3, val4.getType()); // cast <10xi32> -> <2x5xi32> -- illegal
        [[maybe_unused]] auto cast_4_5 = builder.create<mlir::memref::CastOp>(loc, val3, val5.getType()); // cast <10xi32> -> <?xi32>
        [[maybe_unused]] auto cast_4_6 = builder.create<mlir::memref::CastOp>(loc, val4, val6.getType()); // cast <2x5xi32> -> <2x?xi32>
        [[maybe_unused]] auto cast_4_7 = builder.create<mlir::memref::CastOp>(loc, val4, memrefType7); // cast <2x5xi32> -> <*xi32>
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

//
// Attribute creation tests
//

TEST_CASE("CreateIndexAttributeTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        auto context = builder.getContext();

        Index index("idx");
        auto attr = IndexAttr::get(index, context);

        auto attrValue = attr.getValue();
        assert((attrValue == index));

        auto int32Type = builder.getIntegerType(32);
        Value val = Alloca(builder, mlir::MemRefType::get({ 1 }, int32Type));
        val.getDefiningOp()->setAttr("attr", attr);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateIndexRangeAttributeTest")
{
    TestContext context([] {
    auto& builder = GetTestBuilder();
    auto context = builder.getContext();

    int begin = 0;
    int end = 10;
    int increment = 2;

    loopnest::Range range(begin, end, increment);
    IndexRange indexRange("i", range);

    auto attr = IndexRangeAttr::get(indexRange, context);

    auto attrValue = attr.getValue();
    assert((attrValue == IndexRangeAttr::ValueType{ indexRange }));

    auto int32Type = builder.getIntegerType(32);
    Value val = Alloca(builder, mlir::MemRefType::get({ 1 }, int32Type));
    val.getDefiningOp()->setAttr("attr", attr); });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateIterationDomainAttributeTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        auto context = builder.getContext();

        IndexRange i("i", { 0, 8, 1 });
        IndexRange j("j", { 0, 10, 1 });
        IndexRange k("k", { 0, 12, 1 });
        IterationDomain domain{ { i, j, k } };
        auto attr = IterationDomainAttr::get(domain, context);

        auto attrValue = attr.getValue();
        // assert((attrValue == domain));

        auto int32Type = builder.getIntegerType(32);
        Value val = Alloca(builder, mlir::MemRefType::get({ 1 }, int32Type));
        val.getDefiningOp()->setAttr("attr", attr);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateRangeAttributeTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        auto context = builder.getContext();

        int begin = 0;
        int end = 10;
        int increment = 2;
        auto attr = RangeAttr::get(begin, end, increment, context);

        auto attrValue = attr.getValue();
        assert((attrValue == RangeAttr::ValueType{ begin, end, increment }));

        auto int32Type = builder.getIntegerType(32);
        Value val = Alloca(builder, mlir::MemRefType::get({ 1 }, int32Type));
        val.getDefiningOp()->setAttr("attr", attr);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateSplitIndexAttributeTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        auto context = builder.getContext();

        Index outer("i");
        Index inner("j");
        SplitIndex splitIndex{ outer, inner };
        auto attr = SplitIndexAttr::get(splitIndex, context);

        auto attrValue = attr.getValue();
        assert((attrValue == splitIndex));

        auto int32Type = builder.getIntegerType(32);
        Value val = Alloca(builder, mlir::MemRefType::get({ 1 }, int32Type));
        val.getDefiningOp()->setAttr("attr", attr);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateTransformedDomainAttributeTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        auto context = builder.getContext();

        IndexRange i("i", { 0, 8, 1 });
        IndexRange j("j", { 0, 10, 1 });
        IndexRange k("k", { 0, 12, 1 });
        TransformedDomain domain{ { i, j, k } };
        auto attr = TransformedDomainAttr::get(domain, context);

        auto attrValue = attr.getValue();
        // assert((attrValue == domain));

        auto int32Type = builder.getIntegerType(32);
        Value val = Alloca(builder, mlir::MemRefType::get({ 1 }, int32Type));
        val.getDefiningOp()->setAttr("attr", attr);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateFragmentTypePredicateAttributeTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();

        auto int32Type = builder.getIntegerType(32);
        Value val = Alloca(builder, mlir::MemRefType::get({ 1 }, int32Type));

        auto fragmentType = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::last));
        val.getDefiningOp()->setAttr("fragment", fragmentType);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreatePlacementPredicateAttributeTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();

        auto int32Type = builder.getIntegerType(32);
        Value val = Alloca(builder, mlir::MemRefType::get({ 1 }, int32Type));

        auto attr = builder.getI64IntegerAttr(static_cast<int64_t>(PlacementType::before));
        val.getDefiningOp()->setAttr("placement", attr);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

//
// Type creation tests
//
TEST_CASE("CreateSymbolicIndexTypeTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        [[maybe_unused]] auto type = builder.getType<loopnest::SymbolicIndexType>();
        // type.dump();
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateKernelTypeTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        [[maybe_unused]] auto type = builder.getType<loopnest::KernelType>();
        // type.dump();
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

//
// Op creation tests
//
TEST_CASE("CreateSymbolicIndexTest")
{
    TestContext context([] {
        [[maybe_unused]] auto& builder = GetTestBuilder();
        // [[maybe_unused]] auto index = builder.create<loopnest::SymbolicIndexOp>(ScopedContext::getLocation(), "i", 1);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateScheduledLoopTest")
{
    TestContext context([] {
        [[maybe_unused]] auto& builder = GetTestBuilder();
        Index i("i");
        // auto index = builder.create<loopnest::SymbolicIndexOp>(ScopedContext::getLocation(), i);
        [[maybe_unused]] int64_t begin = 0;
        [[maybe_unused]] int64_t end = 32;
        [[maybe_unused]] int64_t step = 1;
        std::vector<int64_t> subdomainSize = { 32 };
        std::vector<Index> subdomainIndexOrder = {};
        // [[maybe_unused]] auto loop = builder.create<loopnest::ScheduledLoopOp>(ScopedContext::getLocation(), begin, end, step, index, subdomainSize, subdomainIndexOrder);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateKernelTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        int64_t M = 8;
        int64_t N = 10;
        int64_t K = 12;

        auto floatType = builder.getF64Type();

        [[maybe_unused]] auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
        [[maybe_unused]] auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
        [[maybe_unused]] auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

        // TODO: get these from a domain object?
        // auto i = builder.create<loopnest::SymbolicIndexOp>(ScopedContext::getLocation(), "i", 1);
        // auto j = builder.create<loopnest::SymbolicIndexOp>(ScopedContext::getLocation(), "j", 2);
        // auto k = builder.create<loopnest::SymbolicIndexOp>(ScopedContext::getLocation(), "k", 3);

        [[maybe_unused]] auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
            // C(i, j) += A(i, k) * B(k, j);
        });
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateNullPredicateTest")
{
    TestContext context([] {
        [[maybe_unused]] auto& builder = GetTestBuilder();
        // [[maybe_unused]] auto pred = builder.create<loopnest::NullPredicateOp>(ScopedContext::getLocation());
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateConstantPredicateTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        [[maybe_unused]] auto value = builder.getBoolAttr(true);
        // [[maybe_unused]] auto pred = builder.create<loopnest::ConstantPredicateOp>(ScopedContext::getLocation(), value);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateFragmentTypePredicateTest")
{
    auto fragmentType = GENERATE(FragmentType::first, FragmentType::last, FragmentType::endBoundary, FragmentType::select, FragmentType::all, FragmentType::range);
    auto fragmentTypeStr = stringifyEnum(fragmentType).str();

    TestContext context([=] {
        auto& builder = GetTestBuilder();
        Index i("i");

        SECTION(fragmentTypeStr)
        {
            [[maybe_unused]] auto fragmentAttr = builder.getI64IntegerAttr(static_cast<int64_t>(fragmentType));
            loopnest::FragmentTypePredicateOp pred;
            switch (fragmentType)
            {
            case FragmentType::select:
                // pred = builder.create<loopnest::FragmentTypePredicateOp>(ScopedContext::getLocation(), fragmentAttr, i, std::vector<int64_t>{ 42 });
                break;
            case FragmentType::range:
                // pred = builder.create<loopnest::FragmentTypePredicateOp>(ScopedContext::getLocation(), fragmentAttr, i, std::vector<int64_t>{ 0, 21, 1 });
                break;
            default:
                // pred = builder.create<loopnest::FragmentTypePredicateOp>(ScopedContext::getLocation(), fragmentAttr, i);
                break;
            }
        }
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreatePlacementPredicateTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        Index i("i");
        auto placement = PlacementType::after;
        [[maybe_unused]] auto placementAttr = builder.getI64IntegerAttr(static_cast<int64_t>(placement));
        // [[maybe_unused]] auto pred = builder.create<loopnest::PlacementPredicateOp>(ScopedContext::getLocation(), placementAttr, i);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateIndexDefinedPredicateTest")
{
    TestContext context([] {
        [[maybe_unused]] auto& builder = GetTestBuilder();
        Index i("i");
        // [[maybe_unused]] auto pred = builder.create<loopnest::IndexDefinedPredicateOp>(ScopedContext::getLocation(), i);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateConjunctionPredicateTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        Index i("i");
        Index j("j");

        // auto pred1 = builder.create<loopnest::IndexDefinedPredicateOp>(ScopedContext::getLocation(), i);

        auto fragment = FragmentType::endBoundary;
        [[maybe_unused]] auto fragmentAttr = builder.getI64IntegerAttr(static_cast<int64_t>(fragment));
        // auto pred2 = builder.create<loopnest::FragmentTypePredicateOp>(ScopedContext::getLocation(), fragmentAttr, j);

        // [[maybe_unused]] auto pred = builder.create<loopnest::ConjunctionPredicateOp>(ScopedContext::getLocation(), ValueRange{ pred1, pred2 });
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateDisjunctionPredicateTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        Index i("i");
        Index j("j");

        // auto pred1 = builder.create<loopnest::IndexDefinedPredicateOp>(ScopedContext::getLocation(), i);

        auto fragment = FragmentType::endBoundary;
        [[maybe_unused]] auto fragmentAttr = builder.getI64IntegerAttr(static_cast<int64_t>(fragment));
        // auto pred2 = builder.create<loopnest::FragmentTypePredicateOp>(ScopedContext::getLocation(), fragmentAttr, j);

        // [[maybe_unused]] auto pred = builder.create<loopnest::DisjunctionPredicateOp>(ScopedContext::getLocation(), ValueRange{ pred1, pred2 });
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

TEST_CASE("CreateScheduledKernelTest")
{
    TestContext context([] {
        auto& builder = GetTestBuilder();
        int64_t M = 8;
        int64_t N = 10;
        int64_t K = 12;

        auto floatType = builder.getF64Type();

        [[maybe_unused]] auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
        [[maybe_unused]] auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
        [[maybe_unused]] auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

        // TODO: get these from a domain object?
        // auto i = builder.create<loopnest::SymbolicIndexOp>(ScopedContext::getLocation(), "i", 1);
        // auto j = builder.create<loopnest::SymbolicIndexOp>(ScopedContext::getLocation(), "j", 2);
        // auto k = builder.create<loopnest::SymbolicIndexOp>(ScopedContext::getLocation(), "k", 3);

        auto kernel = MakeKernel(builder, [&](OpBuilder&, Location) {
            // C(i, j) += A(i, k) * B(k, j);
        });

        auto value = builder.getBoolAttr(true);
        auto loc = builder.getUnknownLoc();
        auto pred = builder.create<loopnest::ConstantPredicateOp>(loc, value);
        [[maybe_unused]] auto scheduledKernelOp = MakeKernel(builder, kernel, pred);
    });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context));
    }

    SECTION("Lowering")
    {
        REQUIRE(VerifyLowerToStd(context));
    }
}

// BUGBUG: The CreateNestAnd* tests are failing because the symbolic indices are being used before they are declared
// possibly due to the interaction between MakeNest() and getOrCreateSchedule()
void CreateNestAndKernelTest()
{
    auto& builder = GetTestBuilder();
    int64_t M = 8;
    int64_t N = 10;
    int64_t K = 12;

    auto floatType = builder.getF64Type();

    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };

    auto nest = MakeNest(builder, domain);
    auto indices = nest.getIndices(builder);
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // now add a kernel
    auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        // C(i, j) += A(i, k) * B(k, j);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i, j });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ i, k });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ k, j });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ i, j });
    });

    nest.getOrCreateSchedule().addKernel(kernel);
}

void CreateNestAndScheduleTest()
{
    auto& builder = GetTestBuilder();
    int64_t iSize = 8;
    int64_t jSize = 10;
    int64_t kSize = 12;

    auto floatType = builder.getF64Type();
    auto outerValue = Alloca(builder, mlir::MemRefType::get({ 1 }, floatType));
    std::vector<int64_t> sizes{ iSize, jSize, kSize };

    auto nest = MakeNest(sizes, [&](OpBuilder& builder, Location loc, NestOp& nest) {
        auto pi = builder.create<ConstantFloatOp>(loc, llvm::APFloat(3.14), floatType);
        auto innerValue = Alloca(builder, mlir::MemRefType::get({ 1 }, floatType));
        auto [i, j, k] = nest.getIndices<3>(builder);
        (void)builder.create<memref::StoreOp>(loc, pi, innerValue, ValueRange{ i });
        (void)builder.create<memref::StoreOp>(loc, pi, outerValue, ValueRange{ j });
    });

    // now create a schedule
    auto schedule = nest.getOrCreateSchedule();
    auto dims = nest.getDomain().getValue().GetDimensions();
    schedule.setOrder({ dims[2], dims[0], dims[1] });

    assert(schedule.numDimensions() == nest.numDimensions());
}

void IndexMultiplicityTest()
{
    auto& builder = GetTestBuilder();
    int64_t iSize = 8;
    int64_t jSize = 10;
    int64_t kSize = 12;

    auto floatType = builder.getF64Type();
    auto outerValue = Alloca(builder, mlir::MemRefType::get({ 1 }, floatType));
    std::vector<int64_t> indices{ iSize, jSize, kSize };

    auto nest = MakeNest(indices, [&](OpBuilder& builder, Location loc, NestOp& nest) {
        auto pi = builder.create<ConstantFloatOp>(loc, llvm::APFloat(3.14), floatType);
        auto innerValue = Alloca(builder, mlir::MemRefType::get({ 1 }, floatType));
        auto [i, j, k] = nest.getIndices<3>(builder);
        (void)builder.create<memref::StoreOp>(loc, pi, innerValue, ValueRange{ i });
        (void)builder.create<memref::StoreOp>(loc, pi, outerValue, ValueRange{ j });
    });

    // Get SymbolicIndexOps for the indices
    nest.getIndices(builder);

    // now create a schedule
    auto schedule = nest.getOrCreateSchedule();
    auto dims = nest.getDomain().getValue().GetDimensions();
    schedule.setOrder({ dims[2], dims[0], dims[1] });

    // Get SymbolicIndexOps for the indices again
    nest.getIndices(builder);

    assert(schedule.numDimensions() == nest.numDimensions());
}

void CreateMatMulNestOpTest()
{
    int M = 8;
    int N = 10;
    int K = 12;

    auto& builder = GetTestBuilder();
    auto floatType = builder.getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    [[maybe_unused]] auto nest = CreateMatMulNestOp(A, B, C);
}

void LowerNestTest()
{
    int M = 8;
    int N = 10;
    int K = 12;

    auto& builder = GetTestBuilder();
    auto floatType = builder.getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    [[maybe_unused]] auto nest = CreateMatMulNestOp(A, B, C);

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
}

void LowerScheduleTest()
{
    auto& builder = GetTestBuilder();

    int64_t M = 8;
    int64_t N = 10;
    int64_t K = 12;

    auto floatType = builder.getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };
    auto nest = MakeNest(builder, domain);
    auto indices = nest.getIndices(builder);
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // now add a kernel
    auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        // C(i, j) += A(i, k) * B(k, j);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i, j });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ i, k });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ k, j });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ i, j });
    });

    nest.getOrCreateSchedule().addKernel(kernel);

    // now create a schedule
    [[maybe_unused]] auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
}

void LowerMultipleKernelsTest()
{
    auto& builder = GetTestBuilder();

    int64_t M = 8;
    int64_t N = 10;
    int64_t K = 12;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };
    auto nest = MakeNest(builder, domain);

    // Create 3 kernels
    auto kernelA = MakeKernel(builder, "A", [&](OpBuilder& builder, Location loc) {
        (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), A);
    });

    auto kernelB = MakeKernel(builder, "B", [&](OpBuilder& builder, Location loc) {
        (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), B);
    });

    auto kernelC = MakeKernel(builder, "C", [&](OpBuilder& builder, Location loc) {
        (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
    });

    nest.getOrCreateSchedule().addKernel(kernelA);
    nest.getOrCreateSchedule().addKernel(kernelB);
    nest.getOrCreateSchedule().addKernel(kernelC);

    // now create a schedule
    [[maybe_unused]] auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());
}

void LowerMultipleScheduledKernelsTest()
{
    auto& builder = GetTestBuilder();
    auto loc = builder.getUnknownLoc();

    int64_t M = 8;
    int64_t N = 10;
    int64_t K = 12;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };
    auto nest = MakeNest(builder, domain);
    auto indices = nest.getIndices(builder);

    // Create 3 kernels
    auto kernelA = MakeKernel(builder, "A", [&](OpBuilder& builder, Location loc) {
        (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), A);
    });

    auto kernelB = MakeKernel(builder, "B", [&](OpBuilder& builder, Location loc) {
        (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), B);
    });

    auto kernelC = MakeKernel(builder, "C", [&](OpBuilder& builder, Location loc) {
        (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
    });

    auto pred = builder.create<loopnest::ConstantPredicateOp>(loc, builder.getBoolAttr(true));
    auto scheduledKernelA = MakeKernel(builder, kernelA, pred);
    // auto scheduledKernelB = MakeKernel(builder, kernelB, pred);
    auto scheduledKernelC = MakeKernel(builder, kernelC, pred);

    nest.getOrCreateSchedule().addKernel(scheduledKernelA);
    nest.getOrCreateSchedule().addKernel(kernelB);
    // nest.getOrCreateSchedule().addKernel(scheduledKernelB);
    nest.getOrCreateSchedule().addKernel(scheduledKernelC);

    // now create a schedule
    [[maybe_unused]] auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());
}

template <typename PredFn>
void LowerScheduledKernelTest(PredFn&& getPred)
{
    auto& builder = GetTestBuilder();

    int64_t M = 8;
    int64_t N = 10;
    int64_t K = 12;

    auto floatType = builder.getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };
    auto nest = MakeNest(builder, domain);
    auto indices = nest.getIndices(builder);
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // create scheduled kernel
    auto kernel = MakeKernel(builder, "print", [&](OpBuilder& builder, Location loc) {
        (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), A);
    });

    // create body kernel
    auto bodyKernel = MakeKernel(builder, "body", [&](OpBuilder& builder, Location loc) {
        (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), B);
    });

    auto pred = getPred(builder, i, j, k);
    auto scheduledKernel = MakeKernel(builder, kernel, pred);

    nest.getOrCreateSchedule().addKernel(scheduledKernel);
    nest.getOrCreateSchedule().addKernel(bodyKernel);

    [[maybe_unused]] auto schedule = nest.getOrCreateSchedule();
}

void LowerFirstKernelTest()
{
    LowerScheduledKernelTest([](auto& builder, auto i, auto j, auto k) { return First(builder, i); });
    // This should produce something like:
    //
    // for i = 0..1
    //   for j
    //     for k
    //       print(%0)
    //       print(%1)
    // for i = 1..M
    //   for j
    //     for k
    //       print(%1)
}

void LowerLastKernelTest()
{
    LowerScheduledKernelTest([](auto& builder, auto i, auto j, auto k) { return Last(builder, i); });
    // This should produce something like:
    //
    // for i = 0..M-1
    //   for j
    //     for k
    //       print(%1)
    // for i = M-1..M
    //   for j
    //     for k
    //       print(%0)
    //       print(%1)
}

void LowerBeforeKernelTest()
{
    LowerScheduledKernelTest([](auto& builder, auto i, auto j, auto k) { return Before(builder, i); });
}

void LowerAfterKernelTest()
{
    LowerScheduledKernelTest([](auto& builder, auto i, auto j, auto k) { return After(builder, i); });
}

void LowerIndexDefinedKernelTest()
{
    // TODO: this
    // LowerScheduledKernelTest([](auto i, auto j, auto k) { return Last(i); });
}

void LowerConjunctionKernelTest()
{
    LowerScheduledKernelTest([](auto& builder, auto i, auto j, auto k) { return Conjunction(builder, First(builder, i), First(builder, j)); });
    // This should produce something like:
    //
    // for i = 0..1
    //   for j = 0..1
    //     for k
    //       print(%0)
    //       print(%1)
    //   for j = 1..N
    //     for k
    //       print(%1)
    // for i = 1..M
    //   for j
    //     for k
    //       print(%1)

    // ERROR: currently splits 'j' loop in 2nd split of 'i'
}

void LowerDisjunctionKernelTest()
{
    LowerScheduledKernelTest([](auto& builder, auto i, auto j, auto k) { return Disjunction(builder, First(builder, i), First(builder, j)); });
    // This should produce something like:
    //
    // for i = 0..1
    //   for j
    //     for k
    //       print(%0)
    //       print(%1)
    // for i = 1..M
    //   for j = 0..1
    //     for k
    //       print(%0)
    //       print(%1)
    //   for j = 1..N
    //     for k
    //       print(%1)

    // ERROR: currently splits 'j' loop in 1st split of 'i'
}

void LowerScheduledKernelsTest()
{
    auto& builder = GetTestBuilder();

    int64_t M = 8;
    int64_t N = 10;
    int64_t K = 12;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };
    auto nest = MakeNest(builder, domain);
    auto indices = nest.getIndices(builder);
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // create initial "clear-and-print" kernel
    auto clearKernel = MakeKernel(builder, "clear", [&](OpBuilder& builder, Location loc) {
        (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), A);

        // A(i, k) = builder.create<ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(0.0), floatType);
        (void)builder.create<memref::StoreOp>(loc,
                                              builder.create<ConstantFloatOp>(
                                                  loc,
                                                  llvm::APFloat(0.0),
                                                  floatType),
                                              A,
                                              ValueRange{ i, j });
    });

    auto scheduledClearKernel = MakeKernel(builder, clearKernel, First(builder, k));

    // create "print result" kernel
    auto printKernel = MakeKernel(builder, "print", [&](OpBuilder& builder, Location loc) {
        (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
    });

    auto scheduledPrintKernel = MakeKernel(builder, printKernel, Last(builder, k));

    // create main kernel
    auto kernel = MakeKernel(builder, "compute", [&](OpBuilder& builder, Location loc) {
        // C(i, j) += A(i, k) * B(k, j);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i, j });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ i, k });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ k, j });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ i, j });
    });

    nest.getOrCreateSchedule().addKernel(scheduledClearKernel);
    nest.getOrCreateSchedule().addKernel(kernel);
    nest.getOrCreateSchedule().addKernel(scheduledPrintKernel);

    [[maybe_unused]] auto schedule = nest.getOrCreateSchedule();
}

void LowerSplitScheduleTest()
{
    int M = 8;
    int N = 10;
    int K = 12;

    auto& builder = GetTestBuilder();
    auto floatType = builder.getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    auto nest = CreateMatMulNestOp(A, B, C);

    // now create a schedule
    auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());

    auto dims = nest.getDomain().getValue().GetDimensions();
    schedule.split(dims[2], 4);

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
}

void LowerSplitScheduleWithKernelTest()
{
    auto& builder = GetTestBuilder();

    int64_t M = 8;
    int64_t N = 10;
    int64_t K = 12;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };
    auto nest = MakeNest(builder, domain);
    auto indices = nest.getIndices(builder);
    // TODO: make the following look something like:
    // auto [i, j, k] nest.getIndices<3>();
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // now add a kernel
    auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        // C(i, j) += A(i, k) * B(k, j);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i, j });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ i, k });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ k, j });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ i, j });
    });

    nest.getOrCreateSchedule().addKernel(kernel);

    // now create a schedule
    auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());

    auto dims = nest.getDomain().getValue().GetDimensions();
    schedule.setOrder({ dims[2], dims[0], dims[1] });
    schedule.split(dims[0], 4);

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
}

void LowerSplitUnrolledScheduleWithKernelTest()
{
    auto& builder = GetTestBuilder();

    int64_t M = 8;
    int64_t N = 10;
    int64_t K = 12;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };

    auto nest = MakeNest(builder, domain);
    auto indices = nest.getIndices(builder);
    // TODO: make the following look something like:
    // auto [i, j, k] nest.getIndices<3>();
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // now add a kernel
    auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        // C(i, j) += A(i, k) * B(k, j);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i, j });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ i, k });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ k, j });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ i, j });
    });

    nest.getOrCreateSchedule().addKernel(kernel);

    // now create a schedule
    auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());

    auto dims = nest.getDomain().getValue().GetDimensions();
    schedule.setOrder({ dims[2], dims[0], dims[1] });
    auto [iOuter, iInner] = schedule.split(dims[0], 4);
    schedule.unroll(iInner);

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
}

void LowerSplitOrderScheduleWithKernelTest()
{
    auto& builder = GetTestBuilder();

    int64_t M = 8;
    int64_t N = 10;
    int64_t K = 12;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };

    auto nest = MakeNest(builder, domain);
    auto indices = nest.getIndices(builder);
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // now add a kernel
    auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        // C(i, j) += A(i, k) * B(k, j);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i, j });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ i, k });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ k, j });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ i, j });
    });

    nest.getOrCreateSchedule().addKernel(kernel);

    // now create a schedule
    auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());

    auto dims = nest.getIndices(builder);
    auto splitIIndex = schedule.split(dims[0], 4);
    auto splitJIndex = schedule.split(dims[1], 4);

    schedule.setOrder({ dims[2], splitIIndex.outer, splitJIndex.outer, splitIIndex.inner, splitJIndex.inner });

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
}

void LowerSplitScheduleWithBoundaryTest()
{
    int M = 9;
    int N = 11;
    int K = 13;

    auto& builder = GetTestBuilder();
    auto floatType = builder.getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    auto nest = CreateMatMulNestOp(A, B, C);

    // now create a schedule
    auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());

    auto dims = nest.getDomain().getValue().GetDimensions();
    schedule.setOrder({ dims[2], dims[0], dims[1] });
    schedule.split(dims[2], 4);

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
}

void LowerSplitScheduleWithBoundaryKernelTest()
{
    auto& builder = GetTestBuilder();

    int64_t M = 9;
    int64_t N = 11;
    int64_t K = 13;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };

    auto nest = MakeNest(builder, domain);
    auto indices = nest.getIndices(builder);
    // TODO: make the following look something like:
    // auto [i, j, k] nest.getIndices<3>();
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // now add a kernel
    [[maybe_unused]] auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        // C(i, j) += A(i, k) * B(k, j);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i, j });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ i, k });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ k, j });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ i, j });
    });

    nest.getOrCreateSchedule().addKernel(kernel);

    // now create a schedule
    auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());

    auto dims = nest.getDomain().getValue().GetDimensions();
    schedule.setOrder({ dims[2], dims[0], dims[1] });
    schedule.split(dims[2], 4);

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
}

void LowerSplitScheduleWithScheduledBoundaryKernelTest()
{
    auto& builder = GetTestBuilder();

    int64_t M = 9;
    int64_t N = 11;
    int64_t K = 13;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    auto nest = MakeNest(builder, { M, N, K });
    auto indices = nest.getIndices(builder);
    // TODO: make the following look something like:
    // auto [i, j, k] nest.getIndices<3>();
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // now add a kernel
    [[maybe_unused]] auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        // C(i, j) += A(i, k) * B(k, j);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i, j });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ i, k });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ k, j });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ i, j });
    });

    auto loc = builder.getUnknownLoc();
    [[maybe_unused]] auto pred = builder.create<loopnest::ConstantPredicateOp>(loc, builder.getBoolAttr(true));
    // auto scheduledKernel = MakeKernel(builder, kernel, pred);
    // nest.getOrCreateSchedule().addKernel(scheduledKernel);
    nest.getOrCreateSchedule().addKernel(kernel);

    // now create a schedule
    auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());

    auto dims = nest.getDomain().getValue().GetDimensions();
    schedule.setOrder({ dims[2], dims[0], dims[1] });
    schedule.split(dims[2], 4);

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
}

void LowerSplitScheduleWithScheduledBoundaryKernelTest2()
{
    auto& builder = GetTestBuilder();

    int64_t M = 9;
    int64_t N = 11;
    int64_t K = 13;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    [[maybe_unused]] auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    std::vector<int64_t> domain{ M, N, K };

    auto fillANest = MakeNest(builder, { M, K });
    auto [ai, aj] = fillANest.getIndices<2>(builder);
    auto fillAKernel = MakeKernel(builder, [&, ai = mlir::Value(ai), aj = mlir::Value(aj)](OpBuilder& builder, Location loc) {
        (void)builder.create<memref::StoreOp>(
            loc,
            builder.create<ConstantFloatOp>(
                loc,
                llvm::APFloat(3.14),
                floatType),
            A,
            ValueRange{ ai, aj });
    });
    fillANest.getOrCreateSchedule().addKernel(fillAKernel);
    fillANest.getOrCreateSchedule();

    auto nest = MakeNest(builder, domain);
    auto indices = nest.getIndices(builder);
    // TODO: make the following look something like:
    // auto [i, j, k] nest.getIndices<3>();
    auto i = indices[0];
    auto j = indices[1];
    [[maybe_unused]] auto k = indices[2];

    // now add a kernel
    [[maybe_unused]] auto pi = builder.create<ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(3.14), floatType);
    [[maybe_unused]] auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        (void)builder.create<memref::StoreOp>(
            loc,
            builder.create<ConstantFloatOp>(
                loc,
                llvm::APFloat(3.14),
                floatType),
            C,
            ValueRange{ i, j });
    });

    auto loc = builder.getUnknownLoc();
    auto pred = builder.create<loopnest::FragmentTypePredicateOp>(loc, FragmentType::first, i);
    auto scheduledKernel = MakeKernel(builder, kernel, pred);
    nest.getOrCreateSchedule().addKernel(scheduledKernel);

    // now create a schedule
    auto schedule = nest.getOrCreateSchedule();
    assert(schedule.numDimensions() == nest.numDimensions());

    auto dims = nest.getDomain().getValue().GetDimensions();
    schedule.setOrder({ dims[2], dims[0], dims[1] });
    schedule.split(dims[2], 4);

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), A);
    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), C);
}

void PrintArrayTest()
{
    auto& builder = GetTestBuilder();

    int64_t M = 8;
    int64_t K = 8;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };

    auto fillANest = MakeNest(builder, { M, K });
    auto fillIndices = fillANest.getIndices(builder);
    auto ai = fillIndices[0];
    auto aj = fillIndices[1];
    auto fillAKernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        (void)builder.create<memref::StoreOp>(
            loc,
            builder.create<ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(3.14), floatType),
            A,
            ValueRange{ ai, aj });
    });
    fillANest.getOrCreateSchedule().addKernel(fillAKernel);
    fillANest.getOrCreateSchedule();

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), A);
}

void LoopNest_test1()
{
    auto& builder = GetTestBuilder();

    auto intType = builder.getIntegerType(32);
    auto m = Alloca(builder, mlir::MemRefType::get({ 4, 5 }, intType));

    std::vector<int64_t> sizes{ 4, 5 };
    auto nest = MakeNest(builder, sizes);
    auto [i, j] = nest.getIndices<2>(builder);

    // Get edsc values to make writing kernel easier
    auto mm = mlir::Value{ m };
    auto ii = i;
    auto jj = j;

    // using plus = ValueBuilder<AddIOp>;
    auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        auto y = builder.create<AddIOp>(
            loc,
            builder.create<memref::LoadOp>(
                loc,
                mm,
                ValueRange{ ii, jj }),
            builder.create<ConstantIntOp>(
                loc,
                2,
                32));
        (void)builder.create<memref::StoreOp>(loc, y, mm, ValueRange{ ii, jj });
        // mm(ii, jj) = y;
    });

    nest.getOrCreateSchedule().addKernel(kernel);

    [[maybe_unused]] auto schedule = nest.getOrCreateSchedule();

    // return matrix(2, 3) - 19; // will return 0 if calculation is correct
}

void MlasValueTest()
{
    auto& builder = GetTestBuilder();

    //
    // Inputs
    //
    int64_t M = 8;
    int64_t N = 10;
    int64_t K = 12;

    auto floatType = GetTestBuilder().getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    //
    // Define nest and main kernel
    //
    auto nest = MakeNest(builder, { M, N, K });

    auto [i, j, k] = nest.getIndices<3>(builder);
    auto ii = i;
    auto jj = j;
    auto kk = k;

    // now add a kernel
    auto kernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        // C(ii, jj) += A(ii, kk) * B(kk, jj);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ ii, jj });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ ii, kk });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ kk, jj });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ ii, jj });
    });

    nest.getOrCreateSchedule().addKernel(kernel);

    //
    // Scheduling operations
    //
    int vectorSize = 8;
    int NumRowsInKernel = 6;

    // if (machineVectorSize > 8)
    // {
    //     // The vector width is 16 floats instead of 8 (e.g. in AVX-512), so double the sizes as needed
    //     vectorSize = 16;
    //     NumRowsInKernel = 12;
    // }

    int NumColumnsInKernel = 2 * vectorSize;

    // Declare and/or calculate constants
    // const auto OutputRows = (int)M;
    const auto OutputColumns = (int)N;
    const auto InnerDimension = (int)K;
    const int kUnroll = 4;

    int columnBlock = std::min(128, OutputColumns);
    int innerDimensionBlock = std::min(512, InnerDimension);

    auto schedule = nest.getOrCreateSchedule();

    // Declare splits
    // i -> [iKernelOuter, iInner]
    // j -> [jCache, jInner1] -> [jCache, [jKernelOuter2, jInner2]] -> [jCache, [jKernelOuter2, [jKernelOuter, jInner3]]]
    // k -> [kCache, kInner1] -> [kCache, [kBlock, kInner2]]

    auto [jCache, jInner1] = schedule.split(j, columnBlock);
    auto [kCache, kInner1] = schedule.split(k, innerDimensionBlock);
    auto [kBlock, kInner2] = schedule.split(kInner1, kUnroll);
    auto [jKernelOuter2, jInner2] = schedule.split(jInner1, NumColumnsInKernel);
    auto [jKernelOuter, jInner3] = schedule.split(jInner2, vectorSize);
    auto [iKernelOuter, iInner] = schedule.split(i, NumRowsInKernel);

    // Set the order
    schedule.setOrder({ jCache, kCache, iKernelOuter, jKernelOuter2, kBlock, kInner2, iInner, jKernelOuter, jInner3 });

#if 0
    // Set up caching
    if (OutputColumns > 128)
    {
        ArgumentType ArgTypeB = ArgumentType::Input;
        std::string cacheNameB = "cacheBInput";
        size_t maxCacheEltsB = innerDimensionBlock * columnBlock;
        size_t fillThresholdB = maxCacheEltsB;
        std::function<void(Scalar, Scalar)> reduceFunctionB = CopyReduce;
        auto extraCacheBParams = std::make_tuple(ArgTypeB,
                                                 cacheNameB,
                                                 maxCacheEltsB,
                                                 fillThresholdB,
                                                 reduceFunctionB,
                                                 false);
        schedule.Cache<GeneralCachingStrategy>(B,
                                               { topLevelK, topLevelJ },
                                               {},
                                               {},
                                               std::nullopt,
                                               extraCacheBParams);
    }

    ArgumentType argTypeC = ArgumentType::Output;
    std::string cacheNameC = "cacheCOutput";
    size_t maxCacheEltsC = NumRowsInKernel * NumColumnsInKernel;
    size_t fillThresholdC = maxCacheEltsC;
    std::function<void(Scalar, Scalar)> reduceFunctionC = SumReduce;
    auto extraCacheCParams = std::make_tuple(argTypeC,
                                             cacheNameC,
                                             maxCacheEltsC,
                                             fillThresholdC,
                                             reduceFunctionC,
                                             true);
    schedule.Cache<GeneralCachingStrategy>(C,
                                           { topLevelI, topLevelJ },
                                           {},
                                           {},
                                           std::nullopt,
                                           extraCacheCParams);
#endif
    // Unroll loops
    // schedule.unroll(jKernelOuter);
    // schedule.unroll(iInner);
    // schedule.unroll(kInner2);
}

void FusionTest1()
{
    auto& builder = GetTestBuilder();

    const int64_t M = 8;
    const int64_t N = 8;
    const int64_t K = 8;
    const int64_t L = 8;

    auto floatType = builder.getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };
    auto D = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ N, L }, floatType)) };
    auto E = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, L }, floatType)) };

    IndexRange iRange{ Index{ "i" }, { 0, (int)M } };
    IndexRange jRange{ Index{ "j" }, { 0, (int)N } };
    IndexRange kRange{ Index{ "k" }, { 0, (int)K } };
    IndexRange lRange{ Index{ "l" }, { 0, (int)L } };

    IterationDomain domain1 = { { iRange, jRange, kRange } };
    IterationDomain domain2 = { { iRange, lRange, jRange } };

    auto nest1 = MakeNest(builder, domain1);
    auto indices1 = nest1.getIndices(builder);
    auto i1 = indices1[0];
    auto j1 = indices1[1];
    auto k1 = indices1[2];

    auto nest2 = MakeNest(builder, domain2);
    auto indices2 = nest2.getIndices(builder);
    auto i2 = indices2[0];
    auto l2 = indices2[1];
    auto j2 = indices2[2];

    auto kernel1 = MakeKernel(builder, "init", [&](OpBuilder& builder, Location loc) {
        // C(i1, j1) += A(i1, k1) * B(k1, j1);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i1, j1 });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ i1, k1 });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ k1, j1 });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ i1, j1 });
    });

    auto kernel2 = MakeKernel(builder, "compute", [&](OpBuilder& builder, Location loc) {
        // E(i2, l2) += C(i2, j2) * D(j2, l2);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, E, ValueRange{ i2, l2 });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ i2, j2 });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, D, ValueRange{ j2, l2 });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, E, ValueRange{ i2, l2 });
    });

    nest1.getOrCreateSchedule().addKernel(kernel1);
    nest2.getOrCreateSchedule().addKernel(kernel2);

    // create schedules
    auto schedule1 = nest1.getOrCreateSchedule();
    auto schedule2 = nest2.getOrCreateSchedule();

    // fuse them
    [[maybe_unused]] auto fusedSchedule = Fuse(builder, schedule1, schedule2);

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), E);
}

// This test is more-or-less a transcription of `LoopNestFuse_test1` in LoopNest_test.cpp
void FusionTest2()
{
    auto& builder = GetTestBuilder();
    auto floatType = GetTestBuilder().getF64Type();
    auto loc = builder.getUnknownLoc();

    const int64_t M = 8;
    const int64_t N = 9;
    const int64_t K = 10;
    const int64_t L = 11;
    auto p = GetMatMul3Parameters(M, N, K, L);

    Index i("i"), j("j"), k("k"), l("l");
    IterationDomain domainC({ { i, { 0, M } },
                              { j, { 0, N } },
                              { k, { 0, K } } });
    IterationDomain domainE({ { i, { 0, M } },
                              { j, { 0, N } },
                              { l, { 0, L } } });

    auto ii = builder.create<SymbolicIndexOp>(loc, i);
    auto jj = builder.create<SymbolicIndexOp>(loc, j);
    auto kk = builder.create<SymbolicIndexOp>(loc, k);
    auto ll = builder.create<SymbolicIndexOp>(loc, l);

    auto initCKernel = MakeKernel(builder, "initC", [&](OpBuilder& builder, Location loc) {
        // p.C(ii, jj) = std_constant_float(llvm::APFloat(0.0), floatType);
        (void)builder.create<memref::StoreOp>(
            loc,
            builder.create<ConstantFloatOp>(
                builder.getUnknownLoc(), llvm::APFloat(0.0), floatType),
            p.C,
            ValueRange{ ii, jj });
    });

    auto computeCKernel = MakeKernel(builder, "matmulC", [&](OpBuilder& builder, Location loc) {
        // p.C(ii, jj) += p.A(ii, kk) * p.B(kk, jj);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, p.C, ValueRange{ ii, jj });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, p.A, ValueRange{ ii, kk });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, p.B, ValueRange{ kk, jj });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, p.C, ValueRange{ ii, jj });
    });

    auto initEKernel = MakeKernel(builder, "initE", [&](OpBuilder& builder, Location loc) {
        // p.E(ii, ll) = builder.create<ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(0.0), floatType);
        (void)builder.create<memref::StoreOp>(
            loc,
            builder.create<ConstantFloatOp>(
                builder.getUnknownLoc(), llvm::APFloat(0.0), floatType),
            p.E,
            ValueRange{ ii, ll });
    });

    auto computeEKernel = MakeKernel(builder, "matmulE", [&](OpBuilder& builder, Location loc) {
        // p.E(ii, ll) += p.C(ii, jj) * p.D(jj, ll);
        auto Cij = builder.create<mlir::memref::LoadOp>(loc, p.E, ValueRange{ ii, ll });
        auto Aik = builder.create<mlir::memref::LoadOp>(loc, p.C, ValueRange{ ii, jj });
        auto Bkj = builder.create<mlir::memref::LoadOp>(loc, p.D, ValueRange{ jj, ll });
        auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
        auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
        (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, p.E, ValueRange{ ii, ll });
    });

    auto firstAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::first));
    auto firstK = builder.create<loopnest::FragmentTypePredicateOp>(loc, firstAttr, k);
    auto firstJ = builder.create<loopnest::FragmentTypePredicateOp>(loc, firstAttr, j);

    auto scheduledInitCKernel = builder.create<ScheduledKernelOp>(loc, "scheduled_initC", initCKernel, firstK.getResult());
    auto scheduledInitEKernel = builder.create<ScheduledKernelOp>(loc, "scheduled_initE", initEKernel, firstJ.getResult());

    auto nestC = MakeNest(builder, domainC);
    nestC.getOrCreateSchedule().addKernel(scheduledInitCKernel);
    nestC.getOrCreateSchedule().addKernel(computeCKernel);

    auto nestE = MakeNest(builder, domainE);
    nestE.getOrCreateSchedule().addKernel(scheduledInitEKernel);
    nestE.getOrCreateSchedule().addKernel(computeEKernel);

    [[maybe_unused]] auto scheduleC = nestC.getOrCreateSchedule();
    [[maybe_unused]] auto scheduleE = nestE.getOrCreateSchedule();

    // auto fusedSchedule = Fuse(builder, scheduleC, scheduleE, { l }, { k });
    // fusedSchedule.setOrder({ i, j, k, l });

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), p.E);
}

// This test is more-or-less the first example in the document
void FusionTest3()
{
    auto& builder = GetTestBuilder();

    const int64_t M = 8;
    const int64_t N = 10;

    const int64_t K = M;
    const int64_t L = N;

    auto floatType = builder.getF64Type();
    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    IndexRange iRange{ Index{ "i" }, { 0, (int)M } };
    IndexRange jRange{ Index{ "j" }, { 0, (int)N } };
    IndexRange kRange{ Index{ "k" }, { 0, (int)K } };
    IndexRange lRange{ Index{ "l" }, { 0, (int)L } };

    IterationDomain domain1 = { { iRange, jRange } };
    IterationDomain domain2 = { { iRange, jRange } };
    // IterationDomain domain2 = { { kRange, lRange } };

    auto nest1 = MakeNest(builder, domain1);
    auto indices1 = nest1.getIndices(builder);
    auto i1 = indices1[0];
    auto j1 = indices1[1];

    auto nest2 = MakeNest(builder, domain2);
    auto indices2 = nest2.getIndices(builder);
    auto k2 = indices2[0];
    auto l2 = indices2[1];

    auto kernel1 = MakeKernel(builder, "kernel_Z", [&](OpBuilder& builder, Location loc) {
        // A(i1, j1) += builder.create<ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(3.0), floatType);
        builder.create<memref::StoreOp>(
            loc,
            builder.create<AddFOp>(
                loc,
                builder.create<memref::LoadOp>(
                    loc,
                    A,
                    ValueRange{ i1, j1 }),
                builder.create<ConstantFloatOp>(
                    builder.getUnknownLoc(),
                    llvm::APFloat(3.0),
                    floatType)),
            A,
            ValueRange{ i1, j1 });
    });

    auto kernel2 = MakeKernel(builder, "kernel_A", [&](OpBuilder& builder, Location loc) {
        // A(k2, l2) *= builder.create<ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(5.0), floatType);
        builder.create<memref::StoreOp>(
            loc,
            builder.create<MulFOp>(
                loc,
                builder.create<memref::LoadOp>(
                    loc,
                    A,
                    ValueRange{ k2, l2 }),
                builder.create<ConstantFloatOp>(
                    builder.getUnknownLoc(),
                    llvm::APFloat(5.0),
                    floatType)),
            A,
            ValueRange{ k2, l2 });
    });

    nest1.getOrCreateSchedule().addKernel(kernel1);
    nest2.getOrCreateSchedule().addKernel(kernel2);

    // create schedules
    auto schedule1 = nest1.getOrCreateSchedule();
    auto schedule2 = nest2.getOrCreateSchedule();

    // fuse them
    [[maybe_unused]] auto fusedSchedule = Fuse(builder, schedule1, schedule2);

    (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), A);
}

void NestedNestTest()
{
    auto& builder = GetTestBuilder();
    auto loc = builder.getUnknownLoc();
    auto floatType = builder.getF64Type();

    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;

    const int64_t innerM = 4;
    const int64_t innerN = 4;
    const int64_t innerK = 4;

    auto A = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, K }, floatType)) };
    auto B = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ K, N }, floatType)) };
    auto C = mlir::Value{ Alloca(builder, mlir::MemRefType::get({ M, N }, floatType)) };

    // Outer nest
    std::vector<int64_t> outerDomain{ M, N, K };

    auto outerNest = MakeNest(builder, outerDomain);
    auto outerIndices = outerNest.getIndices(builder);
    auto i = outerIndices[0];
    auto j = outerIndices[1];
    auto k = outerIndices[2];

    auto outerSchedule = outerNest.getOrCreateSchedule();
    auto [iOuter, iInner] = outerSchedule.split(i, innerM);
    auto [jOuter, jInner] = outerSchedule.split(j, innerN);
    auto [kOuter, kInner] = outerSchedule.split(k, innerK);
    outerSchedule.setOrder({ iOuter, jOuter, kOuter, iInner, jInner, kInner });

    // Inner nest kernel
    auto outerKernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
        // auto& newBuilder = ScopedContext::getBuilderRef();

        std::vector<int64_t> innerDomain{ innerM, innerN, innerK };

        auto innerNest = MakeNest(builder, innerDomain);
        auto innerIndices = innerNest.getIndices(builder);
        auto inneri = innerIndices[0];
        auto innerj = innerIndices[1];
        auto innerk = innerIndices[2];

        auto innerKernel = MakeKernel(builder, [&](OpBuilder& builder, Location loc) {
            // C(inneri, innerj) += A(inneri, innerk) * B(innerk, innerj);
            auto Cij = builder.create<mlir::memref::LoadOp>(loc, C, ValueRange{ inneri, innerj });
            auto Aik = builder.create<mlir::memref::LoadOp>(loc, A, ValueRange{ inneri, innerk });
            auto Bkj = builder.create<mlir::memref::LoadOp>(loc, B, ValueRange{ innerk, innerj });
            auto AikTimesBkj = builder.create<mlir::MulFOp>(loc, Aik, Bkj);
            auto AikTimesBkjPlusCij = builder.create<mlir::AddFOp>(loc, AikTimesBkj, Cij);
            (void)builder.create<mlir::memref::StoreOp>(loc, AikTimesBkjPlusCij, C, ValueRange{ inneri, innerj });
        });

        [[maybe_unused]] auto innerSchedule = innerNest.getOrCreateSchedule();

        auto pred = builder.create<loopnest::NullPredicateOp>(loc);
        [[maybe_unused]] auto scheduledKernelOp = MakeKernel(builder, innerKernel, pred);

        // innerSchedule.addKernel(innerKernel);
        innerSchedule.addKernel(scheduledKernelOp);
    });

    auto firstAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::first));
    auto firstK = builder.create<loopnest::FragmentTypePredicateOp>(loc, firstAttr, k);
    auto scheduledKernelOp = MakeKernel(builder, outerKernel, firstK);
    outerSchedule.addKernel(scheduledKernelOp);
}

// TEST_CASE("PrintArraySliceTest")
// {
//     TestContext context([] {
//     auto& builder = GetTestBuilder();

//     int64_t M = 8;
//     int64_t N = 8;

// auto floatType = ScopedContext::getBuilderRef().getF64Type();
//     auto buffer = Alloca(builder, mlir::MemRefType::get({ M, N }, floatType));
//     auto A = mlir::Value{ buffer };

//     auto fillANest = MakeNest({ M, N });
//     auto fillIndices = fillANest.getIndices(builder);
//     auto ai = fillIndices[0];
//     auto aj = fillIndices[1];
//     auto fillAKernel = MakeKernel(builder, [&]() {
//         A(ai, aj) = builder.create<ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(3.14), floatType);
//     });
//     fillANest.getOrCreateSchedule().addKernel(fillAKernel);
//     fillANest.getOrCreateSchedule();
//     // (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), A);

//     // Get a slice of A
//     llvm::SmallVector<mlir::Value, 4> coordHandles;
//     coordHandles.push_back(builder.create<ConstantIndexOp>{builder.getUnknownLoc(),  2 });
//     std::vector<int64_t> slicedDimensions = { 0 };
//     auto rowMemRefType = mlir::MemRefType::get({ N }, floatType);
//     mlir::Value row = value::Slice{ mlir::Value{ buffer }, slicedDimensions, coordHandles, rowMemRefType };

//     (void)builder.create<loopnest::PrintOp>(builder.getUnknownLoc(), row);
// });
//     SECTION("Parsing")
//     {
//         REQUIRE(VerifyParse(context));
//     }

//     SECTION("Lowering")
//     {
//         REQUIRE(VerifyLowerToStd(context));
//     }
// }
