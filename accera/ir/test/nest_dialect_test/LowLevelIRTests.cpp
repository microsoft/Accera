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

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/LoopUtils.h>
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
mlir::Value Alloca(mlir::MemRefType bufferType)
{
    // TODO : return to using std_alloc when aligned_alloc/_aligned_malloc issue in MLIR is fixed on Windows
    mlir::OpBuilder builder(bufferType.getContext());
    return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), bufferType);
}

class IndexedValue
{
public:
    IndexedValue(mlir::Value array) :
        _array(array)
    {}

    template <typename... Indices>
    mlir::Value Get(mlir::OpBuilder& builder, Indices... indices)
    {
        return builder.create<mlir::memref::LoadOp>(builder.getUnknownLoc(), _array, mlir::ValueRange{ indices... });
    }

    template <typename... Indices>
    void Set(mlir::OpBuilder& builder, mlir::Value value, Indices... indices)
    {
        builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), _array, value, mlir::ValueRange{ indices... });
    }

private:
    mlir::Value _array;
};

std::pair<mlir::ArrayAttr, mlir::ArrayAttr> GetEvenOddShuffleMasks(mlir::OpBuilder& builder, int64_t fullSize)
{
    std::vector<int64_t> evenVals;
    std::vector<int64_t> oddVals;
    for (int64_t i = 0; i < fullSize; i += 2)
    {
        evenVals.push_back(i);
        oddVals.push_back(i + 1);
    }
    auto evenMask = builder.getI64ArrayAttr(evenVals);
    auto oddMask = builder.getI64ArrayAttr(oddVals);
    return { evenMask, oddMask };
}

} // namespace

//
// Small int tests
//

// Minimal example that generates vpmaddubsw instructions
// (Note that when the input was only a single (a, b) pair, LLVM didn't generate a vpmaddubsw for some reason)
//
// sat((evens(a) * evens(a)) + (odds(a) * odds(b)))
//
// unsigned A / signed B
// size: A: 16x2, B: 16x2, output: 16 (not broadcasted)
// vector datatypes
// shuffled
// result = dot2
//
// Int8ub_16x2_vec_shuf_dot2
TEST_CASE("Int8Test1")
{
    int64_t vecSize = 16;
    TestContext context(
        "Int8Test1",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i16Type = builder.getIntegerType(16);

                            auto aType = mlir::VectorType::get({ 2*vecSize }, i8Type);
                            auto bType = mlir::VectorType::get({ 2*vecSize }, i8Type);
                            auto cVecType = mlir::VectorType::get({ vecSize }, i16Type);
                            auto cType = mlir::MemRefType::get({}, cVecType);

                           return { aType, bType, aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto a1 = args[0];
            auto b1 = args[1];
            auto a2 = args[2];
            auto b2 = args[3];
            auto C = args[4];
            auto cType = args[4].getType().cast<mlir::MemRefType>();
            auto cElemType = cType.getElementType();

            auto i8Type = builder.getIntegerType(8);
            auto i32Type = builder.getIntegerType(32);
            auto halfVecType = mlir::VectorType::get({ vecSize }, i8Type);
            auto bigVecType = mlir::VectorType::get({ vecSize }, i32Type);

            // extract evens/odds from a and b
            auto evenMask = builder.getI64ArrayAttr({ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 });
            auto oddMask = builder.getI64ArrayAttr({ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 });

            auto mulSum = [&](auto a, auto b) {
                auto aEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, evenMask);
                auto aOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, oddMask);
                auto bEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, evenMask);
                auto bOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, oddMask);

                // extend to 32 bits
                auto aEvenExt = builder.create<mlir::SignExtendIOp>(loc, aEven, bigVecType);
                auto bEvenExt = builder.create<mlir::ZeroExtendIOp>(loc, bEven, bigVecType);
                auto aOddExt = builder.create<mlir::SignExtendIOp>(loc, aOdd, bigVecType);
                auto bOddExt = builder.create<mlir::ZeroExtendIOp>(loc, bOdd, bigVecType);

                auto mulEven = builder.create<mlir::MulIOp>(loc, aEvenExt, bEvenExt);
                auto mulOdd = builder.create<mlir::MulIOp>(loc, aOddExt, bOddExt);

                auto sum = builder.create<mlir::AddIOp>(loc, mulEven, mulOdd);

                // Make sum be saturated
                auto minI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, -32768, 32), bigVecType);
                auto maxI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 32767, 32), bigVecType);

                auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, cElemType);
                return truncVal;
            };

            auto r1 = mulSum(a1, b1);
            auto r2 = mulSum(a2, b2);
            auto finalSum = builder.create<mlir::AddIOp>(loc, r1, r2);
            builder.create<mlir::memref::StoreOp>(loc, finalSum, C);
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test1.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test1_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "Int8Test1_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test1.ll"));
    }
}

// Version of the above that broadcasts a pair of values from 'a'
// sat((evens(a) * evens(a)) + (odds(a) * odds(b)))
//
// unsigned A / signed B
// size: A: 1x2, B: 16x2, output: 16 (broadcast A)
// vector datatypes
// shuffled
// result = dot2
//
// Int8ub_16x2bcast_vec_shuf_dot2
// Generates vpmaddubsw
TEST_CASE("Int8Test1b")
{
    int64_t vecSize = 16;
    TestContext context(
        "Int8Test1b",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i16Type = builder.getIntegerType(16);

                            auto aType = mlir::VectorType::get({ 2 }, i8Type);
                            auto bType = mlir::VectorType::get({ 2*vecSize }, i8Type);
                            auto cVecType = mlir::VectorType::get({ vecSize }, i16Type);
                            auto cType = mlir::MemRefType::get({}, cVecType);

                           return { aType, bType, aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto a1 = args[0];
            auto b1 = args[1];
            auto a2 = args[2];
            auto b2 = args[3];
            auto C = args[4];
            auto bType = args[1].getType().cast<mlir::VectorType>();
            auto cType = args[4].getType().cast<mlir::MemRefType>();
            auto cElemType = cType.getElementType();

            auto i8Type = builder.getIntegerType(8);
            auto i32Type = builder.getIntegerType(32);
            auto halfVecType = mlir::VectorType::get({ vecSize }, i8Type);
            auto bigVecType = mlir::VectorType::get({ vecSize }, i32Type);

            auto upconvertAMask = builder.getI64ArrayAttr({ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
            a1 = builder.create<mlir::vector::ShuffleOp>(loc, bType, a1, a1, upconvertAMask);
            a2 = builder.create<mlir::vector::ShuffleOp>(loc, bType, a2, a2, upconvertAMask);

            // extract evens/odds from a and b
            auto evenMask = builder.getI64ArrayAttr({ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 });
            auto oddMask = builder.getI64ArrayAttr({ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 });

            auto mulSum = [&](auto a, auto b) {
                auto aEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, evenMask);
                auto aOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, oddMask);
                auto bEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, evenMask);
                auto bOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, oddMask);

                // extend to 32 bits
                auto aEvenExt = builder.create<mlir::SignExtendIOp>(loc, aEven, bigVecType);
                auto bEvenExt = builder.create<mlir::ZeroExtendIOp>(loc, bEven, bigVecType);
                auto aOddExt = builder.create<mlir::SignExtendIOp>(loc, aOdd, bigVecType);
                auto bOddExt = builder.create<mlir::ZeroExtendIOp>(loc, bOdd, bigVecType);

                auto mulEven = builder.create<mlir::MulIOp>(loc, aEvenExt, bEvenExt);
                auto mulOdd = builder.create<mlir::MulIOp>(loc, aOddExt, bOddExt);

                auto sum = builder.create<mlir::AddIOp>(loc, mulEven, mulOdd);

                // Make sum be saturated
                auto minI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, -32768, 32), bigVecType);
                auto maxI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 32767, 32), bigVecType);

                auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, cElemType);
                return truncVal;
            };

            auto r1 = mulSum(a1, b1);
            auto r2 = mulSum(a2, b2);
            auto finalSum = builder.create<mlir::AddIOp>(loc, r1, r2);
            builder.create<mlir::memref::StoreOp>(loc, finalSum, C);
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test1b.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test1b_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyLowerToLLVM(context, true, "Int8Test1b_llvm.mlir"));
    }

    SECTION("LLVMIR")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "Int8Test1b_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test1b.ll"));
    }
}

// Example that generates vpmaddubsw instructions without explicit shuffle operation
//
// sat((evens(a) * evens(a)) + (odds(a) * odds(b)))
//
// unsigned A / signed B
// size: A: 16x2, B: 16x2, output: 16 (not broadcasted)
// 2-d vector datatypes
// sliced
// result = dot2
//
// Int8ub_16x2_vec_slice_dot2
TEST_CASE("Int8Test1C")
{
    int64_t vecSize = 16;
    TestContext context(
        "Int8Test1C",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i16Type = builder.getIntegerType(16);

                            auto aType = mlir::VectorType::get({ 2, vecSize }, i8Type);
                            auto bType = mlir::VectorType::get({ 2, vecSize }, i8Type);
                            auto cVecType = mlir::VectorType::get({ vecSize }, i16Type);
                            auto cType = mlir::MemRefType::get({}, cVecType);

                           return { aType, bType, aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto a1 = args[0];
            auto b1 = args[1];
            auto a2 = args[2];
            auto b2 = args[3];
            auto C = args[4];
            auto cType = args[4].getType().cast<mlir::MemRefType>();
            auto cElemType = cType.getElementType();

            auto i8Type = builder.getIntegerType(8);
            auto i32Type = builder.getIntegerType(32);
            [[maybe_unused]] auto halfVecType = mlir::VectorType::get({ vecSize }, i8Type);
            auto bigVecType = mlir::VectorType::get({ vecSize }, i32Type);

            auto mulSum = [&](auto a, auto b) {
                // extract evens/odds from a and b
                auto aEven = builder.create<mlir::vector::ExtractOp>(loc, a, std::vector<int64_t>{ 0 });
                auto aOdd = builder.create<mlir::vector::ExtractOp>(loc, a, std::vector<int64_t>{ 1 });
                auto bEven = builder.create<mlir::vector::ExtractOp>(loc, b, std::vector<int64_t>{ 0 });
                auto bOdd = builder.create<mlir::vector::ExtractOp>(loc, b, std::vector<int64_t>{ 1 });

                // extend to 32 bits
                auto aEvenExt = builder.create<mlir::SignExtendIOp>(loc, aEven, bigVecType);
                auto bEvenExt = builder.create<mlir::ZeroExtendIOp>(loc, bEven, bigVecType);
                auto aOddExt = builder.create<mlir::SignExtendIOp>(loc, aOdd, bigVecType);
                auto bOddExt = builder.create<mlir::ZeroExtendIOp>(loc, bOdd, bigVecType);

                auto mulEven = builder.create<mlir::MulIOp>(loc, aEvenExt, bEvenExt);
                auto mulOdd = builder.create<mlir::MulIOp>(loc, aOddExt, bOddExt);

                auto sum = builder.create<mlir::AddIOp>(loc, mulEven, mulOdd);

                // Make sum be saturated
                auto minI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, -32768, 32), bigVecType);
                auto maxI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 32767, 32), bigVecType);

                auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, cElemType);
                return truncVal;
            };

            auto r1 = mulSum(a1, b1);
            auto r2 = mulSum(a2, b2);
            auto finalSum = builder.create<mlir::AddIOp>(loc, r1, r2);
            builder.create<mlir::memref::StoreOp>(loc, finalSum, C);
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test1C.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test1C_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "Int8Test1C_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test1C.ll"));
    }
}

// Test that tries to do a 4-elem dot product x 8 columns
//
// y = sat((evens(a) * evens(a)) + (odds(a) * odds(b)))
// z = evens(y) + odds(y)
//
// unsigned A / signed B
// size: A: 8x4, B: 8x4, output: 8 (not broadcasted)
// vector datatypes
// shuffled
// result = dot4
//
// Int8ub_8x4_vec_shuf_dot4
//
// Produces:
// vpmaddubsw	%ymm0, %ymm1, %ymm0
// vpmaddubsw	%ymm2, %ymm3, %ymm1
// vbroadcasti128	LCPI0_0(%rip), %ymm2    ## ymm2 = [1,3,5,7,9,11,13,15,1,3,5,7,9,11,13,15]
//                                          ## ymm2 = mem[0,1,0,1]
// vpermw	%ymm1, %ymm2, %ymm2
// vpmovdw	%ymm1, %xmm1
// vpmovsxwd	%xmm1, %ymm1
// vpmovsxwd	%xmm2, %ymm2
// vpcmpeqd	%ymm3, %ymm3, %ymm3
// vpmaddwd	%ymm3, %ymm0, %ymm0
// vpaddd	%ymm1, %ymm0, %ymm0
// vpaddd	%ymm0, %ymm2, %ymm0
// vmovdqa	%ymm0, (%rsi)
// vzeroupper
// retq
//
TEST_CASE("Int8Test2")
{
    int64_t vecN = 8;
    int64_t vecK = 4;
    TestContext context(
        "Int8Test2",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i32Type = builder.getIntegerType(32);

                            auto aType = mlir::VectorType::get({ vecK * vecN }, i8Type);
                            auto bType = mlir::VectorType::get({ vecK * vecN }, i8Type);
                            auto cVecType = mlir::VectorType::get({ vecN }, i32Type);
                            auto cType = mlir::MemRefType::get({}, cVecType);

                           return { aType, bType, aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto a1 = args[0];
            auto b1 = args[1];
            auto a2 = args[2];
            auto b2 = args[3];
            auto C = args[4];

            auto i8Type = builder.getIntegerType(8);
            auto i16Type = builder.getIntegerType(16);
            auto i32Type = builder.getIntegerType(32);
            auto halfVecType = mlir::VectorType::get({ 2 * vecN }, i8Type);
            auto bigVecType = mlir::VectorType::get({ 2 * vecN }, i32Type);
            auto truncVecType = mlir::VectorType::get({ 2 * vecN }, i16Type);
            auto halfResultVecType = mlir::VectorType::get({ vecN }, i16Type);
            auto resultVecType = mlir::VectorType::get({ vecN }, i32Type);

            // a1, b1
            auto mulSum = [&](auto a, auto b) {
                // extract evens/odds from a and b
                auto evenMask = builder.getI64ArrayAttr({ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 });
                auto oddMask = builder.getI64ArrayAttr({ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 });
                auto aEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, evenMask);
                auto aOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, oddMask);
                auto bEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, evenMask);
                auto bOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, oddMask);

                // extend to 32 bits
                auto aEvenExt = builder.create<mlir::SignExtendIOp>(loc, aEven, bigVecType);
                auto bEvenExt = builder.create<mlir::ZeroExtendIOp>(loc, bEven, bigVecType);
                auto aOddExt = builder.create<mlir::SignExtendIOp>(loc, aOdd, bigVecType);
                auto bOddExt = builder.create<mlir::ZeroExtendIOp>(loc, bOdd, bigVecType);

                auto mulEven = builder.create<mlir::MulIOp>(loc, aEvenExt, bEvenExt);
                auto mulOdd = builder.create<mlir::MulIOp>(loc, aOddExt, bOddExt);

                auto sum = builder.create<mlir::AddIOp>(loc, mulEven, mulOdd);

                // Make sum be saturated
                auto minI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, -32768, 32), bigVecType);
                auto maxI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 32767, 32), bigVecType);

                auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, truncVecType);
                return truncVal;
            };

            auto r1 = mulSum(a1, b1); // truncVecType
            auto r2 = mulSum(a2, b2);

            auto evenMask = builder.getI64ArrayAttr({ 0, 2, 4, 6, 8, 10, 12, 14 });
            auto oddMask = builder.getI64ArrayAttr({ 1, 3, 5, 7, 9, 11, 13, 15 });
            auto r1Even = builder.create<mlir::vector::ShuffleOp>(loc, halfResultVecType, r1, r1, evenMask);
            auto r1Odd = builder.create<mlir::vector::ShuffleOp>(loc, halfResultVecType, r1, r1, oddMask);
            auto r2Even = builder.create<mlir::vector::ShuffleOp>(loc, halfResultVecType, r2, r2, evenMask);
            auto r2Odd = builder.create<mlir::vector::ShuffleOp>(loc, halfResultVecType, r2, r2, oddMask);

            auto r1Sum = builder.create<mlir::AddIOp>(loc, builder.create<mlir::SignExtendIOp>(loc, r1Even, resultVecType), builder.create<mlir::SignExtendIOp>(loc, r1Odd, resultVecType));
            auto r2Sum = builder.create<mlir::AddIOp>(loc, builder.create<mlir::SignExtendIOp>(loc, r2Even, resultVecType), builder.create<mlir::SignExtendIOp>(loc, r2Odd, resultVecType));
            // auto r1Sum = builder.create<mlir::AddIOp>(loc, builder.create<mlir::SignExtendIOp>(loc, r1Even, resultVecType), builder.create<mlir::SignExtendIOp>(loc, r1Odd, resultVecType));
            // auto r2Sum = builder.create<mlir::AddIOp>(loc, builder.create<mlir::SignExtendIOp>(loc, r2Even, resultVecType), builder.create<mlir::SignExtendIOp>(loc, r2Odd, resultVecType));
            auto finalSum = builder.create<mlir::AddIOp>(loc, r1Sum, r2Sum);
            builder.create<mlir::memref::StoreOp>(loc, finalSum, C);
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test2.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test2_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "Int8Test2_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test2.ll"));
    }
}

// Test that tries to do a 4-elem dot product x 8 columns
//
// y = sat((evens(a) * evens(a)) + (odds(a) * odds(b)))
// z = evens(y) + odds(y)
//
// unsigned A / signed B
// size: A: 1x4, B: 8x4, output: 8 (broadcasted)
// vector datatypes
// shuffled
// result = dot4
//
// Int8ub_8x4_vec_shuf_dot4
//
// Doesn't produce vpmaddubsw / vpmaddwd sequence
TEST_CASE("Int8Test2b")
{
    int64_t vecN = 8;
    int64_t vecK = 4;
    TestContext context(
        "Int8Test2b",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i32Type = builder.getIntegerType(32);

                            auto aType = mlir::VectorType::get({ vecK * vecN }, i8Type);
                            auto bType = mlir::VectorType::get({ vecK * vecN }, i8Type);
                            auto cVecType = mlir::VectorType::get({ vecN }, i32Type);
                            auto cType = mlir::MemRefType::get({}, cVecType);

                           return { aType, bType, aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto a1 = args[0];
            auto b1 = args[1];
            auto a2 = args[2];
            auto b2 = args[3];
            auto C = args[4];

            auto bType = args[1].getType().cast<mlir::VectorType>();
            auto i8Type = builder.getIntegerType(8);
            auto i16Type = builder.getIntegerType(16);
            auto i32Type = builder.getIntegerType(32);
            auto halfVecType = mlir::VectorType::get({ 2 * vecN }, i8Type);
            auto bigVecType = mlir::VectorType::get({ 2 * vecN }, i32Type);
            auto truncVecType = mlir::VectorType::get({ 2 * vecN }, i16Type);
            auto halfResultVecType = mlir::VectorType::get({ vecN }, i16Type);
            auto resultVecType = mlir::VectorType::get({ vecN }, i32Type);

            auto upconvertAMask = builder.getI64ArrayAttr({ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
            a1 = builder.create<mlir::vector::ShuffleOp>(loc, bType, a1, a1, upconvertAMask);
            a2 = builder.create<mlir::vector::ShuffleOp>(loc, bType, a2, a2, upconvertAMask);

            // a1, b1
            auto mulSum = [&](auto a, auto b) {
                // extract evens/odds from a and b
                auto evenMask = builder.getI64ArrayAttr({ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 });
                auto oddMask = builder.getI64ArrayAttr({ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 });
                auto aEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, evenMask);
                auto aOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, oddMask);
                auto bEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, evenMask);
                auto bOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, oddMask);

                // extend to 32 bits
                auto aEvenExt = builder.create<mlir::SignExtendIOp>(loc, aEven, bigVecType);
                auto bEvenExt = builder.create<mlir::ZeroExtendIOp>(loc, bEven, bigVecType);
                auto aOddExt = builder.create<mlir::SignExtendIOp>(loc, aOdd, bigVecType);
                auto bOddExt = builder.create<mlir::ZeroExtendIOp>(loc, bOdd, bigVecType);

                auto mulEven = builder.create<mlir::MulIOp>(loc, aEvenExt, bEvenExt);
                auto mulOdd = builder.create<mlir::MulIOp>(loc, aOddExt, bOddExt);

                auto sum = builder.create<mlir::AddIOp>(loc, mulEven, mulOdd);

                // Make sum be saturated
                auto minI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, -32768, 32), bigVecType);
                auto maxI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 32767, 32), bigVecType);

                auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, truncVecType);
                return truncVal;
            };

            auto r1 = mulSum(a1, b1); // truncVecType
            auto r2 = mulSum(a2, b2);

            auto evenMask = builder.getI64ArrayAttr({ 0, 2, 4, 6, 8, 10, 12, 14 });
            auto oddMask = builder.getI64ArrayAttr({ 1, 3, 5, 7, 9, 11, 13, 15 });
            auto r1Even = builder.create<mlir::vector::ShuffleOp>(loc, halfResultVecType, r1, r1, evenMask);
            auto r1Odd = builder.create<mlir::vector::ShuffleOp>(loc, halfResultVecType, r1, r1, oddMask);
            auto r2Even = builder.create<mlir::vector::ShuffleOp>(loc, halfResultVecType, r2, r2, evenMask);
            auto r2Odd = builder.create<mlir::vector::ShuffleOp>(loc, halfResultVecType, r2, r2, oddMask);

            auto r1Sum = builder.create<mlir::AddIOp>(loc, builder.create<mlir::SignExtendIOp>(loc, r1Even, resultVecType), builder.create<mlir::SignExtendIOp>(loc, r1Odd, resultVecType));
            auto r2Sum = builder.create<mlir::AddIOp>(loc, builder.create<mlir::SignExtendIOp>(loc, r2Even, resultVecType), builder.create<mlir::SignExtendIOp>(loc, r2Odd, resultVecType));
            // auto r1Sum = builder.create<mlir::AddIOp>(loc, builder.create<mlir::SignExtendIOp>(loc, r1Even, resultVecType), builder.create<mlir::SignExtendIOp>(loc, r1Odd, resultVecType));
            // auto r2Sum = builder.create<mlir::AddIOp>(loc, builder.create<mlir::SignExtendIOp>(loc, r2Even, resultVecType), builder.create<mlir::SignExtendIOp>(loc, r2Odd, resultVecType));
            auto finalSum = builder.create<mlir::AddIOp>(loc, r1Sum, r2Sum);
            builder.create<mlir::memref::StoreOp>(loc, finalSum, C);
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test2b.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test2b_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "Int8Test2b_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test2b.ll"));
    }
}

// Computes many dot products in a loop, accumulating into output buffer
//
// sat((evens(a) * evens(a)) + (odds(a) * odds(b)))
//
// unsigned A / signed B
// size: A: N * 16x2, B: N * 16x2, output: N*16 (not broadcasted)
// vector datatypes
// shuffled
// result = dot2
//
// Int8ub_Nx2_vec_shuf_dot2_bufaccum
//
// Generates vpmaddubsw
TEST_CASE("Int8Test3")
{
    int64_t N = 212;
    int64_t vecSize = 16;
    TestContext context(
        "Int8Test3",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i16Type = builder.getIntegerType(16);

                            auto vecType1 = mlir::VectorType::get({ 2*vecSize }, i8Type);
                            auto vecType2 = mlir::VectorType::get({ vecSize }, i16Type);

                            auto arrType1 = mlir::MemRefType::get({ N }, vecType1);
                            auto arrType2 = mlir::MemRefType::get({ N }, vecType2);
                            return { arrType1, arrType1, arrType2 }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto A = IndexedValue{ args[0] };
            [[maybe_unused]] auto B = IndexedValue{ args[1] };
            auto C = IndexedValue{ args[2] };

            auto aType = args[0].getType().cast<mlir::MemRefType>();
            auto bType = args[1].getType().cast<mlir::MemRefType>();
            auto cType = args[2].getType().cast<mlir::MemRefType>();
            auto cElemType = cType.getElementType();

            auto aaType = aType;
            auto bbType = bType;
            auto ccType = cType;

            auto AA = IndexedValue{ Alloca(aaType) };
            auto BB = IndexedValue{ Alloca(bbType) };
            auto CC = IndexedValue{ Alloca(ccType) };

            auto i8Type = builder.getIntegerType(8);
            auto i32Type = builder.getIntegerType(32);
            auto halfVecType = mlir::VectorType::get({ vecSize }, i8Type);
            auto bigVecType = mlir::VectorType::get({ vecSize }, i32Type);

            // Init A, B, and C
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto i = iLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(iLoop);

                    AA.Set(builder, A.Get(builder, i));
                    BB.Set(builder, A.Get(builder, i));
                    CC.Set(builder, A.Get(builder, i));
                }
            }

            // Compute
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto i = iLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(iLoop);

                    auto a = AA.Get(builder, i);
                    auto b = BB.Get(builder, i);

                    // extract evens/odds from a and b
                    auto [evenMask, oddMask] = GetEvenOddShuffleMasks(builder, 32);

                    auto aEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, evenMask);
                    auto aOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, oddMask);
                    auto bEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, evenMask);
                    auto bOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, oddMask);

                    // extend to 32 bits
                    auto aEvenExt = builder.create<mlir::SignExtendIOp>(loc, aEven, bigVecType);
                    auto bEvenExt = builder.create<mlir::ZeroExtendIOp>(loc, bEven, bigVecType);
                    auto aOddExt = builder.create<mlir::SignExtendIOp>(loc, aOdd, bigVecType);
                    auto bOddExt = builder.create<mlir::ZeroExtendIOp>(loc, bOdd, bigVecType);

                    auto mulEven = builder.create<mlir::MulIOp>(loc, aEvenExt, bEvenExt);
                    auto mulOdd = builder.create<mlir::MulIOp>(loc, aOddExt, bOddExt);

                    auto sum = builder.create<mlir::AddIOp>(loc, mulEven, mulOdd);

                    // Make sum be saturated
                    auto minI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, -32768, 32), bigVecType);
                    auto maxI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 32767, 32), bigVecType);

                    auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                    auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                    auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                    auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                    auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, cElemType);

                    auto c = CC.Get(builder, i);
                    auto cPlus = builder.create<mlir::AddIOp>(loc, c, truncVal);
                    CC.Set(builder, cPlus, i);
                }
            }

            // Copy to output
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto i = iLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(iLoop);
                    C.Set(builder, CC.Get(builder, i), i);
                }
            }
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test3.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test3_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "Int8Test3_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test3.ll"));
    }
}

// Doesn't generate vpmaddubsw -- never calls detectPMADDUBSW
//
// unsigned A / signed B
// size: A: N * 8x2, B: N * 8x2, output: N * 8 (not broadcasted)
// vector datatypes, vecsize = 8
// shuffled
// result = dot2
// intermediate datatype = i16
// Int8ub_16x2_vec_shuf_dot2_accum16
TEST_CASE("Int8Test3b")
{
    int64_t N = 212;
    int64_t vecSize = 8;
    TestContext context(
        "Int8Test3b",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i16Type = builder.getIntegerType(16);

                            auto vecType1 = mlir::VectorType::get({ 2*vecSize }, i8Type);
                            auto vecType2 = mlir::VectorType::get({ vecSize }, i16Type);

                            auto arrType1 = mlir::MemRefType::get({ N }, vecType1);
                            auto arrType2 = mlir::MemRefType::get({ N }, vecType2);
                            return { arrType1, arrType1, arrType2 }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto A = IndexedValue{ args[0] };
            auto B = IndexedValue{ args[1] };
            auto C = IndexedValue{ args[2] };

            auto aType = args[0].getType().cast<mlir::MemRefType>();
            auto bType = args[1].getType().cast<mlir::MemRefType>();
            auto cType = args[2].getType().cast<mlir::MemRefType>();
            auto cElemType = cType.getElementType();

            auto aaType = aType;
            auto bbType = bType;
            auto ccType = cType;

            auto AA = IndexedValue{ Alloca(aaType) };
            auto BB = IndexedValue{ Alloca(bbType) };
            auto CC = IndexedValue{ Alloca(ccType) };

            auto i8Type = builder.getIntegerType(8);
            auto i16Type = builder.getIntegerType(16);
            auto i32Type = builder.getIntegerType(32);
            auto halfVecType = mlir::VectorType::get({ vecSize }, i8Type);
            auto medVecType = mlir::VectorType::get({ vecSize }, i16Type);
            auto bigVecType = mlir::VectorType::get({ vecSize }, i32Type);

            // Init A, B, and C
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto i = iLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(iLoop);

                    AA.Set(builder, A.Get(builder, i), i);
                    BB.Set(builder, B.Get(builder, i), i);
                    CC.Set(builder, C.Get(builder, i), i);
                }
            }

            // Compute
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto i = iLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(iLoop);

                    auto a = AA.Get(builder, i);
                    auto b = BB.Get(builder, i);

                    // extract evens/odds from a and b
                    auto evenMask = builder.getI64ArrayAttr({ 0, 2, 4, 6, 8, 10, 12, 14 });
                    auto oddMask = builder.getI64ArrayAttr({ 1, 3, 5, 7, 9, 11, 13, 15 });

                    auto aEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, evenMask);
                    auto aOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, oddMask);
                    auto bEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, evenMask);
                    auto bOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, oddMask);

                    // extend to 16 bits
                    auto aEvenExt = builder.create<mlir::SignExtendIOp>(loc, aEven, medVecType);
                    auto bEvenExt = builder.create<mlir::ZeroExtendIOp>(loc, bEven, medVecType);
                    auto aOddExt = builder.create<mlir::SignExtendIOp>(loc, aOdd, medVecType);
                    auto bOddExt = builder.create<mlir::ZeroExtendIOp>(loc, bOdd, medVecType);

                    auto mulEven = builder.create<mlir::MulIOp>(loc, aEvenExt, bEvenExt);
                    auto mulOdd = builder.create<mlir::MulIOp>(loc, aOddExt, bOddExt);

                    // extend to 32 bits
                    auto mulEvenExt = builder.create<mlir::SignExtendIOp>(loc, mulEven, bigVecType);
                    auto mulOddExt = builder.create<mlir::SignExtendIOp>(loc, mulOdd, bigVecType);

                    auto sum = builder.create<mlir::AddIOp>(loc, mulEvenExt, mulOddExt);

                    // Make sum be saturated
                    auto minI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, -32768, 32), bigVecType);
                    auto maxI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 32767, 32), bigVecType);

                    auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                    auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                    auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                    auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                    auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, cElemType);

                    auto c = CC.Get(builder, i);
                    auto cPlus = builder.create<mlir::AddIOp>(loc, c, truncVal);
                    CC.Set(builder, cPlus, i);
                }
            }

            // Copy to output
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto i = iLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(iLoop);
                    C.Set(builder, CC.Get(builder, i), i);
                }
            }
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test3b.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test3b_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test3b.ll"));
    }
}

// Doesn't generate vpmaddubsw instruction -- "Bad result type"
TEST_CASE("Int8Test3c")
{
    // sizes in terms of vector widths
    int64_t M = 123;
    int64_t N = 456;
    int64_t K = 1;

    // int64_t vecM = 1;
    int64_t vecN = 16;
    int64_t vecK = 2;

    TestContext context(
        "Int8Test3c",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i32Type = builder.getIntegerType(32);

                            auto aElemType = mlir::VectorType::get({ vecK }, i8Type);
                            auto bElemType = mlir::VectorType::get({ vecN * vecK }, i8Type);
                            auto cElemType = mlir::VectorType::get({ vecN }, i32Type);

                            auto aType = mlir::MemRefType::get({ M, K }, aElemType);
                            auto bType = mlir::MemRefType::get({ K, N }, bElemType);
                            auto cType = mlir::MemRefType::get({ M, N }, cElemType);
                            return { aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto A = IndexedValue{ args[0] };
            auto B = IndexedValue{ args[1] };
            auto C = IndexedValue{ args[2] };

            auto aType = args[0].getType().cast<mlir::MemRefType>();
            auto bType = args[1].getType().cast<mlir::MemRefType>();
            auto cType = args[2].getType().cast<mlir::MemRefType>();
            auto bElemType = bType.getElementType();

            auto aaType = aType;
            auto bbType = bType;
            auto ccType = cType;

            auto AA = IndexedValue{ Alloca(aaType) };
            auto BB = IndexedValue{ Alloca(bbType) };
            auto CC = IndexedValue{ Alloca(ccType) };

            auto i8Type = builder.getIntegerType(8);
            auto i16Type = builder.getIntegerType(16);
            auto i32Type = builder.getIntegerType(32);
            auto halfVecType = mlir::VectorType::get({ vecN }, i8Type);
            auto midVecType = mlir::VectorType::get({ vecN }, i16Type);
            auto bigVecType = mlir::VectorType::get({ vecN }, i32Type);

            // Init A and C
            {
                auto loop = builder.create<AffineForOp>(loc, 0, M, 1);
                auto i = loop.getInductionVar();
                auto builder = util::MakeBodyBuilder(loop);
                {
                    // Init A
                    {
                        auto loop = builder.create<AffineForOp>(loc, 0, K, 1);
                        auto k = loop.getInductionVar();
                        auto builder = util::MakeBodyBuilder(loop);
                        {
                            AA.Set(builder, A.Get(builder, i, k), i, k);
                        }
                    }

                    // Init C
                    {
                        auto loop = builder.create<AffineForOp>(loc, 0, N, 1);
                        auto j = loop.getInductionVar();
                        auto builder = util::MakeBodyBuilder(loop);
                        {
                            CC.Set(builder, C.Get(builder, i, j), i, j);
                        }
                    }
                }
            }

            // Init B
            {
                auto loop = builder.create<AffineForOp>(loc, 0, K, 1);
                auto k = loop.getInductionVar();
                auto builder = util::MakeBodyBuilder(loop);
                {
                    {
                        auto loop = builder.create<AffineForOp>(loc, 0, N, 1);
                        auto j = loop.getInductionVar();
                        auto builder = util::MakeBodyBuilder(loop);
                        {
                            BB.Set(builder, B.Get(builder, k, j), k, j);
                        }
                    }
                }
            }

            // Compute
            {
                auto loop = builder.create<AffineForOp>(loc, 0, M, 1);
                auto i = loop.getInductionVar();
                auto builder = util::MakeBodyBuilder(loop);
                {
                    auto loop = builder.create<AffineForOp>(loc, 0, N, 1);
                    auto j = loop.getInductionVar();
                    auto builder = util::MakeBodyBuilder(loop);
                    {
                        auto loop = builder.create<AffineForOp>(loc, 0, K, 1);
                        auto k = loop.getInductionVar();
                        auto builder = util::MakeBodyBuilder(loop);
                        {
                            auto a = AA.Get(builder, i, k);
                            auto b = BB.Get(builder, k, j);

                            // extract evens/odds from a and b
                            // auto evenBMask = builder.getI64ArrayAttr({ 0, 2, 4, 6, 8, 10, 12, 14 });
                            // auto oddBMask = builder.getI64ArrayAttr({ 1, 3, 5, 7, 9, 11, 13, 15 });
                            // auto upconvertAMask = builder.getI64ArrayAttr({ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
                            auto upconvertAMask = builder.getI64ArrayAttr({ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
                            auto evenBMask = builder.getI64ArrayAttr({ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 });
                            auto oddBMask = builder.getI64ArrayAttr({ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 });

                            auto splatA = builder.create<mlir::vector::ShuffleOp>(loc, bElemType, a, a, upconvertAMask);

                            // auto aEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, evenAMask);
                            // auto aOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, a, a, oddAMask);
                            auto aEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, splatA, splatA, evenBMask);
                            auto aOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, splatA, splatA, oddBMask);
                            auto bEven = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, evenBMask);
                            auto bOdd = builder.create<mlir::vector::ShuffleOp>(loc, halfVecType, b, b, oddBMask);

                            // extend to 32 bits
                            auto aEvenExt = builder.create<mlir::SignExtendIOp>(loc, aEven, bigVecType);
                            auto bEvenExt = builder.create<mlir::ZeroExtendIOp>(loc, bEven, bigVecType);
                            auto aOddExt = builder.create<mlir::SignExtendIOp>(loc, aOdd, bigVecType);
                            auto bOddExt = builder.create<mlir::ZeroExtendIOp>(loc, bOdd, bigVecType);

                            auto mulEven = builder.create<mlir::MulIOp>(loc, aEvenExt, bEvenExt);
                            auto mulOdd = builder.create<mlir::MulIOp>(loc, aOddExt, bOddExt);

                            auto sum = builder.create<mlir::AddIOp>(loc, mulEven, mulOdd);

                            // Make sum be saturated
                            auto minI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, -32768, 32), bigVecType);
                            auto maxI16Val = builder.create<mlir::SplatOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 32767, 32), bigVecType);

                            auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                            auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                            auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                            auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                            auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, midVecType);

                            auto embiggenVal = builder.create<mlir::SignExtendIOp>(loc, truncVal, bigVecType);

                            auto c = CC.Get(builder, i, j);
                            auto cPlus = builder.create<mlir::AddIOp>(loc, c, embiggenVal);
                            CC.Set(builder, cPlus, i, j);
                        }
                    }
                }
            }

            // Copy to output
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, M, 1);
                auto builder = util::MakeBodyBuilder(iLoop);
                auto i = iLoop.getInductionVar();
                {
                    auto loop = builder.create<AffineForOp>(loc, 0, N, 1);
                    auto j = loop.getInductionVar();
                    auto builder = util::MakeBodyBuilder(loop);
                    {
                        C.Set(builder, CC.Get(builder, i, j), i, j);
                    }
                }
            }
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test3c.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test3c_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyLowerToLLVM(context, true, "Int8Test3c_llvm.mlir"));
    }

    SECTION("LLVMIR")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "Int8Test3c_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test3c.ll"));
    }
}

// Test case that generates vpmadd without using explicit vector types
//
// unsigned A / signed B
// size: A: N * 2, B: N * 2, output: N (not broadcasted)
// scalar datatypes
// interleaved (shuffled)
// result = dot2
// Int8ub_16x2_shuf_dot2
TEST_CASE("Int8Test4")
{
    int64_t N = 212;
    TestContext context(
        "Int8Test4",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i16Type = builder.getIntegerType(16);

                            auto aType = mlir::MemRefType::get({ 2*N }, i8Type);
                            auto bType = mlir::MemRefType::get({ 2*N }, i8Type);
                            auto cType = mlir::MemRefType::get({ N }, i16Type);
                            return { aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto A = IndexedValue{ args[0] };
            auto B = IndexedValue{ args[1] };
            auto C = IndexedValue{ args[2] };

            auto aType = args[0].getType().cast<mlir::MemRefType>();
            auto bType = args[1].getType().cast<mlir::MemRefType>();
            auto cType = args[2].getType().cast<mlir::MemRefType>();
            auto cElemType = cType.getElementType();

            auto aaType = aType;
            auto bbType = bType;
            auto ccType = cType;

            auto i32Type = builder.getIntegerType(32);

            auto AA = IndexedValue{ Alloca(aaType) };
            auto BB = IndexedValue{ Alloca(bbType) };
            auto CC = IndexedValue{ Alloca(ccType) };

            auto i1 = builder.create<mlir::ConstantIndexOp>(loc, 1);
            auto i2 = builder.create<mlir::ConstantIndexOp>(loc, 2);

            // Init A and B
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, 2 * N, 1);
                auto i = iLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(iLoop);
                    AA.Set(builder, A.Get(builder, i), i);
                    BB.Set(builder, B.Get(builder, i), i);
                }
            }

            // Init C
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto i = iLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(iLoop);
                    CC.Set(builder, C.Get(builder, i), i);
                }
            }

            // Compute
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto i = iLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(iLoop);

                    auto iEven = builder.create<mlir::MulIOp>(loc, i, i2);
                    auto iOdd = builder.create<mlir::AddIOp>(loc, iEven, i1);

                    auto aEven = AA.Get(builder, iEven);
                    auto aOdd = AA.Get(builder, iOdd);
                    auto bEven = BB.Get(builder, iEven);
                    auto bOdd = BB.Get(builder, iOdd);

                    // extend to 32 bits
                    auto aEvenExt = builder.create<mlir::SignExtendIOp>(loc, aEven, i32Type);
                    auto bEvenExt = builder.create<mlir::ZeroExtendIOp>(loc, bEven, i32Type);
                    auto aOddExt = builder.create<mlir::SignExtendIOp>(loc, aOdd, i32Type);
                    auto bOddExt = builder.create<mlir::ZeroExtendIOp>(loc, bOdd, i32Type);

                    auto mulEven = builder.create<mlir::MulIOp>(loc, aEvenExt, bEvenExt);
                    auto mulOdd = builder.create<mlir::MulIOp>(loc, aOddExt, bOddExt);
                    auto sum = builder.create<mlir::AddIOp>(loc, mulEven, mulOdd);

                    // Make sum be saturated
                    auto minI16Val = builder.create<mlir::ConstantIntOp>(loc, -32768, 32);
                    auto maxI16Val = builder.create<mlir::ConstantIntOp>(loc, 32767, 32);

                    auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                    auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                    auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                    auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                    auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, cElemType);

                    auto c = CC.Get(builder, i);
                    auto cPlus = builder.create<mlir::AddIOp>(loc, c, truncVal);
                    CC.Set(builder, cPlus, i);
                }
            }

            // Copy to output
            {
                auto iLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto i = iLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(iLoop);
                    C.Set(builder, CC.Get(builder, i), i);
                }
            }
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test4.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test4_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyLowerToLLVM(context, true, "Int8Test4_llvm.mlir"));
    }

    SECTION("LLVMIR")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "Int8Test4_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test4.ll"));
    }
}

// Test case that generates vpmadd without using explicit vector types (2 loops)
//
// unsigned A / signed B
// size: A: N * 2, B: N * 2, output: N (not broadcasted)
// scalar datatypes
// not interleaved
// scalar accumulator
// result = dot2
// Int8ub_16x2_dot2
//
// Generates vpmadd
TEST_CASE("Int8Test5")
{
    int64_t vecN = 16;
    int64_t vecM = 2;
    int64_t N = vecN;
    int64_t M = vecM;
    TestContext context(
        "Int8Test5",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i16Type = builder.getIntegerType(16);

                            auto aType = mlir::MemRefType::get({ M, N }, i8Type);
                            auto bType = mlir::MemRefType::get({ M, N }, i8Type);
                            auto cType = mlir::MemRefType::get({ N }, i16Type);
                            return { aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto A = IndexedValue{ args[0] };
            auto B = IndexedValue{ args[1] };
            auto C = IndexedValue{ args[2] };

            auto aType = args[0].getType().cast<mlir::MemRefType>();
            auto bType = args[1].getType().cast<mlir::MemRefType>();
            auto cType = args[2].getType().cast<mlir::MemRefType>();
            auto aElemType = aType.getElementType();
            auto bElemType = bType.getElementType();
            auto cElemType = cType.getElementType();

            // AA and BB should be transposed...
            auto shape = aType.getShape();
            auto M = shape[0];
            auto N = shape[1];
            auto aaType = mlir::MemRefType::get({ N, M }, aElemType);
            auto bbType = mlir::MemRefType::get({ N, M }, bElemType);
            auto ccType = cType;

            auto i32Type = builder.getIntegerType(32);

            auto AA = IndexedValue{ Alloca(aaType) };
            auto BB = IndexedValue{ Alloca(bbType) };
            auto CC = IndexedValue{ Alloca(ccType) };
            auto accumType = mlir::MemRefType::get({}, i32Type);

            // Init A, B, and C
            {
                auto jLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto j = jLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(jLoop);
                    CC.Set(builder, C.Get(builder, j), j);

                    auto iLoop = builder.create<AffineForOp>(loc, 0, M, 1);
                    auto i = iLoop.getInductionVar();
                    {
                        auto builder = util::MakeBodyBuilder(iLoop);

                        AA.Set(builder, A.Get(builder, i, j), j, i);
                        BB.Set(builder, B.Get(builder, i, j), j, i);
                    }
                }
            }

            // Compute
            {
                auto jLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto j = jLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(jLoop);

                    auto accum = Alloca(accumType);
                    builder.create<mlir::memref::StoreOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 0, 32), accum);

                    auto iLoop = builder.create<AffineForOp>(loc, 0, M, 1);
                    auto i = iLoop.getInductionVar();
                    {
                        auto builder = util::MakeBodyBuilder(iLoop);

                        auto a = AA.Get(builder, j, i);
                        auto b = BB.Get(builder, j, i);

                        // extend to 32 bits
                        auto aExt = builder.create<mlir::ZeroExtendIOp>(loc, a, i32Type);
                        auto bExt = builder.create<mlir::SignExtendIOp>(loc, b, i32Type);

                        auto prod = builder.create<mlir::MulIOp>(loc, aExt, bExt);
                        auto accumVal = builder.create<mlir::memref::LoadOp>(loc, accum);
                        auto sum = builder.create<mlir::AddIOp>(loc, accumVal, prod);
                        builder.create<mlir::memref::StoreOp>(loc, sum, accum);
                    }
                    // mlir::loopUnrollFull(iLoop);

                    auto sum = builder.create<mlir::memref::LoadOp>(loc, accum);

                    // Make sum be saturated
                    auto minI16Val = builder.create<mlir::ConstantIntOp>(loc, -32768, 32);
                    auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                    auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);

                    auto maxI16Val = builder.create<mlir::ConstantIntOp>(loc, 32767, 32);
                    auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                    auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);

                    auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, cElemType);
                    auto c = CC.Get(builder, j);
                    auto cPlus = builder.create<mlir::AddIOp>(loc, c, truncVal);
                    CC.Set(builder, cPlus, j);
                }
            }

            // Copy to output
            {
                auto jLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto j = jLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(jLoop);
                    C.Set(builder, CC.Get(builder, j), j);
                }
            }
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test5.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test5_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyLowerToLLVM(context, true, "Int8Test5_llvm.mlir"));
    }

    SECTION("LLVMIR")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "Int8Test5_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test5.ll"));
    }
}

// Test case that generates vpmadd without using explicit vector types (2 loops)
//
// unsigned A / signed B
// size: A: 2, B: N * 2, output: N (broadcasted)
// scalar datatypes
// not interleaved
// scalar accumulator
// result = dot2
// Int8ub_16x2_dot2
//
// Produces vpmaddubsw
TEST_CASE("Int8Test5b")
{
    int64_t vecN = 16;
    int64_t vecM = 2;
    int64_t M = vecM;
    int64_t N = vecN;
    TestContext context(
        "Int8Test5b",
        [&]() -> std::vector<mlir::Type> {
        auto& builder = GetTestBuilder();

        auto i8Type = builder.getIntegerType(8);
        auto i16Type = builder.getIntegerType(16);

        auto aType = mlir::MemRefType::get({ M }, i8Type);
        auto bType = mlir::MemRefType::get({ M, N }, i8Type);
        auto cType = mlir::MemRefType::get({ N }, i16Type);
        return { aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto A = IndexedValue{ args[0] };
            auto B = IndexedValue{ args[1] };
            auto C = IndexedValue{ args[2] };

            auto aType = args[0].getType().cast<mlir::MemRefType>();
            auto bType = args[1].getType().cast<mlir::MemRefType>();
            auto cType = args[2].getType().cast<mlir::MemRefType>();
            auto aElemType = aType.getElementType();
            auto bElemType = bType.getElementType();
            auto cElemType = cType.getElementType();

            auto shape = bType.getShape();
            auto M = shape[0];
            auto N = shape[1];
            auto aaType = mlir::MemRefType::get({ M }, aElemType);
            auto bbType = mlir::MemRefType::get({ N, M }, bElemType);
            auto ccType = cType;

            auto i32Type = builder.getIntegerType(32);

            auto AA = IndexedValue{ Alloca(aaType) };
            auto BB = IndexedValue{ Alloca(bbType) };
            auto CC = IndexedValue{ Alloca(ccType) };
            auto accumType = mlir::MemRefType::get({}, i32Type);

            // Init A, B, and C
            {
                {
                    // AA
                    auto iLoop = builder.create<AffineForOp>(loc, 0, M, 1);
                    auto i = iLoop.getInductionVar();
                    {
                        auto builder = util::MakeBodyBuilder(iLoop);
                        AA.Set(builder, A.Get(builder, i), i);
                    }
                }

                auto jLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto j = jLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(jLoop);

                    // CC
                    CC.Set(builder, C.Get(builder, j), j);

                    // BB
                    auto iLoop = builder.create<AffineForOp>(loc, 0, M, 1);
                    auto i = iLoop.getInductionVar();
                    {
                        auto builder = util::MakeBodyBuilder(iLoop);

                        BB.Set(builder, B.Get(builder, i, j), j, i);
                    }
                }
            }

            // Compute
            {
                auto jLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto j = jLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(jLoop);

                    auto accum = Alloca(accumType);
                    builder.create<mlir::memref::StoreOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 0, 32), accum);

                    auto iLoop = builder.create<AffineForOp>(loc, 0, M, 1);
                    auto i = iLoop.getInductionVar();
                    {
                        auto builder = util::MakeBodyBuilder(iLoop);

                        auto a = AA.Get(builder, i);
                        auto b = BB.Get(builder, j, i);

                        // extend to 32 bits
                        auto aExt = builder.create<mlir::SignExtendIOp>(loc, a, i32Type);
                        auto bExt = builder.create<mlir::ZeroExtendIOp>(loc, b, i32Type);

                        auto mul = builder.create<mlir::MulIOp>(loc, aExt, bExt);
                        auto accumVal = builder.create<mlir::memref::LoadOp>(loc, accum);
                        auto sum = builder.create<mlir::AddIOp>(loc, accumVal, mul);
                        builder.create<mlir::memref::StoreOp>(loc, sum, accum);
                    }

                    auto sum = builder.create<mlir::memref::LoadOp>(loc, accum);

                    // Make sum be saturated
                    auto minI16Val = builder.create<mlir::ConstantIntOp>(loc, -32768, 32);
                    auto maxI16Val = builder.create<mlir::ConstantIntOp>(loc, 32767, 32);

                    auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                    auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                    auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                    auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                    auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, cElemType);

                    auto c = CC.Get(builder, j);
                    auto cPlus = builder.create<mlir::AddIOp>(loc, c, truncVal);
                    CC.Set(builder, cPlus, j);
                }
            }

            // Copy to output
            {
                auto jLoop = builder.create<AffineForOp>(loc, 0, N, 1);
                auto j = jLoop.getInductionVar();
                {
                    auto builder = util::MakeBodyBuilder(jLoop);
                    C.Set(builder, CC.Get(builder, j), j);
                }
            }
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test5b.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test5b_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test5b.ll"));
    }
}

// Test case that generates vpmadd without using explicit vector types (2 loops)
//
// unsigned A / signed B
// size: A: N * 2, B: N * 2, output: N (not broadcasted)
// scalar datatypes
// not interleaved
// scalar accumulator
// result = dot2
// Int8ub_16x2_dot2
//
// Generates vpmaddubsw
TEST_CASE("Int8Test6")
{
    int64_t vecN = 16;
    int64_t vecM = 2;
    int64_t N = 1534 * vecN;
    int64_t M = 103 * vecM;
    TestContext context(
        "Int8Test6",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i16Type = builder.getIntegerType(16);

                            auto aType = mlir::MemRefType::get({ M, N }, i8Type);
                            auto bType = mlir::MemRefType::get({ M, N }, i8Type);
                            auto cType = mlir::MemRefType::get({ N }, i16Type);
                            return { aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto A = IndexedValue{ args[0] };
            auto B = IndexedValue{ args[1] };
            auto C = IndexedValue{ args[2] };

            auto aType = args[0].getType().cast<mlir::MemRefType>();
            auto bType = args[1].getType().cast<mlir::MemRefType>();
            auto cType = args[2].getType().cast<mlir::MemRefType>();
            auto aElemType = aType.getElementType();
            auto bElemType = bType.getElementType();
            auto cElemType = cType.getElementType();

            // AA and BB should be transposed...
            auto shape = aType.getShape();
            auto M = shape[0];
            auto N = shape[1];
            auto aaType = mlir::MemRefType::get({ vecN, vecM }, aElemType);
            auto bbType = mlir::MemRefType::get({ vecN, vecM }, bElemType);
            auto ccType = mlir::MemRefType::get({ vecN }, cElemType);
            // auto ccType = cType;

            auto i32Type = builder.getIntegerType(32);

            auto accumType = mlir::MemRefType::get({}, i32Type);

            // Compute
            {
                auto jOuterLoop = builder.create<AffineForOp>(loc, 0, N, vecN);
                {
                    auto builder = util::MakeBodyBuilder(jOuterLoop);
                    auto jOuter = jOuterLoop.getInductionVar();

                    // Init CC
                    auto CC = IndexedValue{ Alloca(ccType) };
                    auto jInitLoop = builder.create<AffineForOp>(loc, 0, vecN, 1);
                    {
                        auto builder = util::MakeBodyBuilder(jInitLoop);
                        auto jInner = jInitLoop.getInductionVar();
                        {
                            auto j = builder.create<mlir::AddIOp>(loc, jOuter, jInner);
                            CC.Set(builder, C.Get(builder, j), jInner);
                        }
                    }

                    auto iOuterLoop = builder.create<AffineForOp>(loc, 0, M, vecM);
                    {
                        auto builder = util::MakeBodyBuilder(iOuterLoop);
                        auto iOuter = iOuterLoop.getInductionVar();

                        auto AA = IndexedValue{ Alloca(aaType) };
                        auto BB = IndexedValue{ Alloca(bbType) };

                        // Init AA and BB
                        auto jInitLoop = builder.create<AffineForOp>(loc, 0, vecN, 1);
                        {
                            auto builder = util::MakeBodyBuilder(jInitLoop);
                            auto jInner = jInitLoop.getInductionVar();
                            {
                                auto j = builder.create<mlir::AddIOp>(loc, jOuter, jInner);
                                auto iInitLoop = builder.create<AffineForOp>(loc, 0, vecM, 1);
                                {
                                    auto builder = util::MakeBodyBuilder(iInitLoop);
                                    auto iInner = iInitLoop.getInductionVar();

                                    auto i = builder.create<mlir::AddIOp>(loc, iOuter, iInner);
                                    AA.Set(builder, A.Get(builder, i, j), jInner, iInner);
                                    BB.Set(builder, B.Get(builder, i, j), jInner, iInner);
                                }
                            }
                        }

                        auto jInnerLoop = builder.create<AffineForOp>(loc, 0, vecN, 1);
                        {
                            auto builder = util::MakeBodyBuilder(jInnerLoop);
                            auto jInner = jInnerLoop.getInductionVar();

                            auto accum = Alloca(accumType);
                            builder.create<mlir::memref::StoreOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 0, 32), accum);

                            auto iInnerLoop = builder.create<AffineForOp>(loc, 0, vecM, 1);
                            {
                                auto builder = util::MakeBodyBuilder(iInnerLoop);
                                auto iInner = iInnerLoop.getInductionVar();

                                auto a = AA.Get(builder, jInner, iInner);
                                auto b = BB.Get(builder, jInner, iInner);

                                // extend to 32 bits
                                auto aExt = builder.create<mlir::SignExtendIOp>(loc, a, i32Type);
                                auto bExt = builder.create<mlir::ZeroExtendIOp>(loc, b, i32Type);

                                auto mul = builder.create<mlir::MulIOp>(loc, aExt, bExt);
                                auto accumVal = builder.create<mlir::memref::LoadOp>(loc, accum);
                                auto sum = builder.create<mlir::AddIOp>(loc, accumVal, mul);
                                builder.create<mlir::memref::StoreOp>(loc, sum, accum);
                            } // end of iInner loop

                            auto sum = builder.create<mlir::memref::LoadOp>(loc, accum);

                            // Make sum be saturated
                            auto minI16Val = builder.create<mlir::ConstantIntOp>(loc, -32768, 32);
                            auto maxI16Val = builder.create<mlir::ConstantIntOp>(loc, 32767, 32);

                            auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                            auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                            auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                            auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);
                            auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, cElemType);

                            auto c = CC.Get(builder, jInner);
                            auto cPlus = builder.create<mlir::AddIOp>(loc, c, truncVal);
                            CC.Set(builder, cPlus, jInner);
                        } // end of jInner loop
                    } // end of iOuter loop

                    // Copy to output
                    auto jOutputLoop = builder.create<AffineForOp>(loc, 0, vecN, 1);
                    {
                        auto builder = util::MakeBodyBuilder(jOutputLoop);
                        auto jInner = jOutputLoop.getInductionVar();
                        {
                            auto j = builder.create<mlir::AddIOp>(loc, jOuter, jInner);
                            C.Set(builder, CC.Get(builder, jInner), j);
                        }
                    }
                } // end of jOuter loop
            }
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test6.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test6_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test6.ll"));
    }
}

// Test case that generates vpmadd without using explicit vector types (3 loops)
//
// unsigned A / signed B
// size: A: N * 2, B: N * 2, output: N (not broadcasted)
// scalar datatypes
// not interleaved
// scalar accumulator
// result = dot2
// Int8ub_16x2_dot2
//
// Doesn't generate vpmadd
TEST_CASE("Int8Test8")
{
    int64_t vecM = 1;
    int64_t vecN = 8;
    int64_t vecK = 4;
    int64_t M = 1 * vecM;
    int64_t N = 234 * vecN;
    int64_t K = 1534 * vecK;
    TestContext context(
        "Int8Test8",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto i8Type = builder.getIntegerType(8);
                            auto i32Type = builder.getIntegerType(32);

                            auto aType = mlir::MemRefType::get({ M, K }, i8Type);
                            auto bType = mlir::MemRefType::get({ K, N }, i8Type);
                            auto cType = mlir::MemRefType::get({ M, N }, i32Type);
                            return { aType, bType, cType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto A = IndexedValue{ args[0] };
            auto B = IndexedValue{ args[1] };
            auto C = IndexedValue{ args[2] };

            auto aType = args[0].getType().cast<mlir::MemRefType>();
            auto bType = args[1].getType().cast<mlir::MemRefType>();
            auto cType = args[2].getType().cast<mlir::MemRefType>();
            auto aElemType = aType.getElementType();
            auto bElemType = bType.getElementType();
            auto cElemType = cType.getElementType();

            auto i16Type = builder.getIntegerType(16);
            auto i32Type = builder.getIntegerType(32);

            auto aaType = mlir::MemRefType::get({ vecM, vecK }, aElemType);
            auto aaaType = mlir::MemRefType::get({ vecK }, aElemType);
            auto bbType = mlir::MemRefType::get({ vecN, vecK }, bElemType);

            auto ccElemType = i32Type;

#if 1
            int cccBits = 16;
#else
            int cccBits = 32;
#endif
            auto cccElemType = builder.getIntegerType(cccBits);
            auto ccType = mlir::MemRefType::get({ vecM, vecN }, ccElemType);
            auto cccType = mlir::MemRefType::get({ vecN, 2 }, cccElemType);

            auto accumType = mlir::MemRefType::get({}, i32Type);

            // Alloc caches and scratchpad
            auto AA = IndexedValue{ Alloca(aaType) };
            auto BB = IndexedValue{ Alloca(bbType) };
            auto CC = IndexedValue{ Alloca(ccType) };
            auto AAA = IndexedValue{ Alloca(aaaType) };
            auto CCC = IndexedValue{ Alloca(cccType) };
            auto accum = Alloca(accumType);
            {
                auto jOuterLoop = builder.create<AffineForOp>(loc, 0, N, vecN);
                {
                    auto builder = util::MakeBodyBuilder(jOuterLoop);
                    auto jOuter = jOuterLoop.getInductionVar();

                    auto iOuterLoop = builder.create<AffineForOp>(loc, 0, M, vecM);
                    {
                        auto builder = util::MakeBodyBuilder(iOuterLoop);
                        auto iOuter = iOuterLoop.getInductionVar();

                        // Init CC
                        auto iInitLoop = builder.create<AffineForOp>(loc, 0, vecM, 1);
                        {
                            auto builder = util::MakeBodyBuilder(iInitLoop);
                            auto iInner = iInitLoop.getInductionVar();

                            [[maybe_unused]] auto i = builder.create<mlir::AddIOp>(loc, iOuter, iInner);

                            auto jInitLoop = builder.create<AffineForOp>(loc, 0, vecN, 1);
                            {
                                auto builder = util::MakeBodyBuilder(jInitLoop);
                                auto jInner = jInitLoop.getInductionVar();

                                [[maybe_unused]] auto j = builder.create<mlir::AddIOp>(loc, jOuter, jInner);
                                if (ccElemType == cElemType)
                                {
                                    CC.Set(builder, builder.create<mlir::ConstantIntOp>(loc, 0, 32), iInner, jInner);
                                }
                                else
                                {
                                    CC.Set(builder, builder.create<mlir::ConstantIntOp>(loc, 0, 16), iInner, jInner);
                                }
                            }
                        }

                        auto kOuterLoop = builder.create<AffineForOp>(loc, 0, K, vecK);
                        {
                            auto builder = util::MakeBodyBuilder(kOuterLoop);
                            auto kOuter = kOuterLoop.getInductionVar();

                            // Init AA
                            auto iInitLoop = builder.create<AffineForOp>(loc, 0, vecM, 1);
                            {
                                auto builder = util::MakeBodyBuilder(iInitLoop);
                                auto iInner = iInitLoop.getInductionVar();

                                auto i = builder.create<mlir::AddIOp>(loc, iOuter, iInner);

                                auto kInitLoop = builder.create<AffineForOp>(loc, 0, vecK, 1);
                                {
                                    auto builder = util::MakeBodyBuilder(kInitLoop);
                                    auto kInner = kInitLoop.getInductionVar();

                                    auto k = builder.create<mlir::AddIOp>(loc, kOuter, kInner);
                                    AA.Set(builder, A.Get(builder, i, k), iInner, kInner);
                                }
                            }

                            // Init BB
                            auto jInitLoop = builder.create<AffineForOp>(loc, 0, vecN, 1);
                            {
                                auto builder = util::MakeBodyBuilder(jInitLoop);
                                auto jInner = jInitLoop.getInductionVar();
                                {
                                    auto j = builder.create<mlir::AddIOp>(loc, jOuter, jInner);

                                    auto kInitLoop = builder.create<AffineForOp>(loc, 0, vecK, 1);
                                    {
                                        auto builder = util::MakeBodyBuilder(kInitLoop);
                                        auto kInner = kInitLoop.getInductionVar();

                                        auto k = builder.create<mlir::AddIOp>(loc, kOuter, kInner);
                                        BB.Set(builder, B.Get(builder, k, j), jInner, kInner);
                                    }
                                }
                            }

                            auto iInnerLoop = builder.create<AffineForOp>(loc, 0, vecM, 1);
                            {
                                auto builder = util::MakeBodyBuilder(iInnerLoop);
                                auto iInner = iInnerLoop.getInductionVar();

                                // Init AAA
                                auto kInitLoop = builder.create<AffineForOp>(loc, 0, vecK, 1);
                                {
                                    auto builder = util::MakeBodyBuilder(kInitLoop);
                                    auto kInner = kInitLoop.getInductionVar();
                                    auto a = AA.Get(builder, iInner, kInner);
                                    AAA.Set(builder, a, kInner);
                                }

                                // Compute
                                auto kInnerLoop1 = builder.create<AffineForOp>(loc, 0, 2, 1);
                                {
                                    auto builder = util::MakeBodyBuilder(kInnerLoop1);
                                    auto kInner1Count = kInnerLoop1.getInductionVar();

                                    auto kInner1 = builder.create<mlir::MulIOp>(loc, kInner1Count, builder.create<mlir::ConstantIndexOp>(loc, 2));

                                    // Init CCC
                                    auto kInitLoop = builder.create<AffineForOp>(loc, 0, 2, 1);
                                    {
                                        auto builder = util::MakeBodyBuilder(kInitLoop);
                                        auto kInner = kInitLoop.getInductionVar();

                                        auto jInitLoop = builder.create<AffineForOp>(loc, 0, vecN, 1);
                                        {
                                            auto builder = util::MakeBodyBuilder(jInitLoop);
                                            auto jInner = jInitLoop.getInductionVar();

                                            CCC.Set(builder, builder.create<mlir::ConstantIntOp>(loc, 0, cccBits), jInner, kInner);
                                        }
                                    }

                                    auto jInnerLoop = builder.create<AffineForOp>(loc, 0, vecN, 1);
                                    {
                                        auto builder = util::MakeBodyBuilder(jInnerLoop);
                                        auto jInner = jInnerLoop.getInductionVar();

                                        builder.create<mlir::memref::StoreOp>(loc, builder.create<mlir::ConstantIntOp>(loc, 0, 32), accum);

                                        auto kInnerLoop2 = builder.create<AffineForOp>(loc, 0, 2, 1);
                                        {
                                            auto builder = util::MakeBodyBuilder(kInnerLoop2);
                                            auto kInner2 = kInnerLoop2.getInductionVar();

                                            auto kInner = builder.create<mlir::AddIOp>(loc, kInner1, kInner2);

                                            auto a = AAA.Get(builder, kInner);
                                            auto b = BB.Get(builder, jInner, kInner);

                                            // extend to 32 bits
                                            auto aExt = builder.create<mlir::SignExtendIOp>(loc, a, i32Type);
                                            auto bExt = builder.create<mlir::ZeroExtendIOp>(loc, b, i32Type);

                                            auto mul = builder.create<mlir::MulIOp>(loc, aExt, bExt);
                                            auto accumVal = builder.create<mlir::memref::LoadOp>(loc, accum);
                                            auto sum = builder.create<mlir::AddIOp>(loc, accumVal, mul);
                                            builder.create<mlir::memref::StoreOp>(loc, sum, accum);
                                        } // end of kInner2 loop

                                        auto sum = builder.create<mlir::memref::LoadOp>(loc, accum);

                                        // Make sum be saturated
                                        auto minI16Val = builder.create<mlir::ConstantIntOp>(loc, -32768, 32);
                                        auto maxI16Val = builder.create<mlir::ConstantIntOp>(loc, 32767, 32);

                                        auto maxCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, sum, minI16Val);
                                        auto maxVal = builder.create<mlir::SelectOp>(loc, maxCmp, sum, minI16Val);
                                        auto minCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, maxVal, maxI16Val);
                                        auto minVal = builder.create<mlir::SelectOp>(loc, minCmp, maxVal, maxI16Val);

                                        if (cccElemType == cElemType)
                                        {
                                            auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, i16Type);
                                            auto expandVal = builder.create<mlir::SignExtendIOp>(loc, truncVal, cccElemType);
                                            CCC.Set(builder, expandVal, jInner, kInner1Count);
                                        }
                                        else
                                        {
                                            auto truncVal = builder.create<mlir::TruncateIOp>(loc, minVal, cccElemType);
                                            auto c = CCC.Get(builder, jInner, kInner1Count);
                                            auto cPlus = builder.create<mlir::AddIOp>(loc, c, truncVal);
                                            CCC.Set(builder, cPlus, jInner, kInner1Count);
                                        }
                                    } // end of jInner loop

                                    // Copy back to CC
                                    auto kCopyBackLoop = builder.create<AffineForOp>(loc, 0, 2, 1);
                                    {
                                        auto builder = util::MakeBodyBuilder(kCopyBackLoop);
                                        auto kInner = kCopyBackLoop.getInductionVar();

                                        auto jCopyBackLoop = builder.create<AffineForOp>(loc, 0, vecN, 1);
                                        {
                                            auto builder = util::MakeBodyBuilder(jCopyBackLoop);
                                            auto jInner = jCopyBackLoop.getInductionVar();

                                            auto c = CCC.Get(builder, jInner, kInner);
                                            if (cccElemType != ccElemType)
                                            {
                                                c = builder.create<mlir::SignExtendIOp>(loc, c, ccElemType);
                                            }

                                            auto c2 = CC.Get(builder, iInner, jInner);
                                            auto cPlus = builder.create<mlir::AddIOp>(loc, c, c2);
                                            CC.Set(builder, cPlus, iInner, jInner);
                                        }
                                    }

                                } // end of kInner1 loop
                            } // end of iInner loop
                        } // end of kOuter loop

                        // Copy to output
                        auto iOutputLoop = builder.create<AffineForOp>(loc, 0, vecM, 1);
                        {
                            auto builder = util::MakeBodyBuilder(iOutputLoop);
                            auto iInner = iOutputLoop.getInductionVar();
                            auto i = builder.create<mlir::AddIOp>(loc, iOuter, iInner);

                            auto jOutputLoop = builder.create<AffineForOp>(loc, 0, vecN, 1);
                            {
                                auto builder = util::MakeBodyBuilder(jOutputLoop);
                                auto jInner = jOutputLoop.getInductionVar();
                                auto j = builder.create<mlir::AddIOp>(loc, jOuter, jInner);

                                if (ccElemType == cElemType)
                                {
                                    C.Set(builder, CC.Get(builder, iInner, jInner), i, j);
                                }
                                else
                                {
                                    auto ccVal = builder.create<mlir::SignExtendIOp>(loc, CC.Get(builder, iInner, jInner), i32Type);
                                    C.Set(builder, ccVal, i, j);
                                }
                            }
                        }
                    } // end of iOuter loop
                } // end of jOuter loop
            }
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "Int8Test8.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "Int8Test8_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyLowerToLLVM(context, true, "Int8Test8_llvm.mlir"));
    }

    SECTION("LLVMIR")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "Int8Test8_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "Int8Test8.ll"));
    }
}
