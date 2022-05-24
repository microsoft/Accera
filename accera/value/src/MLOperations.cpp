////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MLOperations.h"
#include "Array.h"
#include "ArrayOperations.h"
#include "Cache.h"
#include "Debugging.h"
#include "FastMath.h"
#include "Index.h"
#include "Kernel.h"
#include "KernelPredicate.h"
#include "Nest.h"
#include "Plan.h"
#include "Profiling.h"
#include "ScalarOperations.h"
#include "Schedule.h"

#include <utilities/include/Exception.h>
#include <utilities/include/MemoryLayout.h>

#include <optional>

namespace accera
{
using namespace utilities;

namespace value
{
    namespace
    {
        void SoftmaxifyRowsRowMajor(Array m)
        {
            auto elementType = m.GetType();
            auto epsilon = Cast(Scalar(1e-7), elementType);

            int numRows = static_cast<int>(m.Shape()[0]);
            int numColumns = static_cast<int>(m.Shape()[1]);

            Nest nest(MemoryShape{ numRows });
            auto i = nest.GetIndices()[0];

            nest.Set([&]() {
                auto row = m.Slice({ 0 }, { i });
                auto maxVal = VectorMax(row);
                Scalar sum = Allocate(elementType, ScalarLayout);
                sum = Cast(Scalar(0.0f), elementType);

                For(0, numColumns, 1, [&](Scalar j) {
                    auto eulerVal = Exp(row(j) - maxVal);
                    row(j) = eulerVal;
                    sum += eulerVal;
                });

                sum = Select(sum < epsilon, Cast(Scalar(1.0f), elementType), sum);

                row /= sum;
            });

            nest.CreateSchedule();
        }

        void SoftmaxifyRowsColumnMajor(Array mt)
        {
            Array m = mt.Reorder({ 1, 0 });
            auto elementType = m.GetType();
            auto epsilon = Cast(Scalar(1e-7), elementType);

            int numRows = static_cast<int>(m.Shape()[1]);
            int numColumns = static_cast<int>(m.Shape()[0]);

            Nest nest(MemoryShape{ numRows });
            auto i = nest.GetIndices()[0];

            nest.Set([&]() {
                auto row = m.Slice({ 1 }, { i });
                auto maxVal = VectorMax(row);
                Scalar sum = Allocate(elementType, ScalarLayout);
                sum = Cast(Scalar(0.0f), elementType);

                For(0, numColumns, 1, [&](Scalar j) {
                    auto eulerVal = Exp(row(j) - maxVal);
                    row(j) = eulerVal;
                    sum += eulerVal;
                });

                sum = Select(sum < epsilon, Cast(Scalar(1.0f), elementType), sum);

                row /= sum;
            });

            nest.CreateSchedule();
        }

        template <typename ExpFnType>
        void SoftmaxifyRowsVectorizedRowMajor(Array m, ExpFnType ExpFn)
        {
            LocationGuard region(GET_LOCATION());
            ProfileRegion profileRegion("softmax_0_all");

            const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            const int vectorUnits = 16; // AVX-2 has 16 256-bit registers
            auto elementType = m.GetType();
            auto epsilon = Cast(Scalar(1e-7f), elementType);

            int numRows = static_cast<int>(m.Shape()[0]);
            int numColumns = static_cast<int>(m.Shape()[1]);

            Nest nest(MemoryShape{ numRows });
            auto i = nest.GetIndices()[0];

            nest.Set([&]() {
                auto row = m.Slice({ 0 }, { i });

                // loop 1: VectorMax
                Scalar maxVal;
                {
                    LocationGuard region_(GET_LOCATION());
                    ProfileRegion profileRegion_("softmax_1_vecmax");
                    maxVal = VectorMax(row);
                }

                // loop 2: Compute exp(x_i-max), and sum of exp(x_i-max)
                Scalar sum;
                {
                    LocationGuard region_(GET_LOCATION());
                    ProfileRegion profileRegion_("softmax_2_expsum");
                    sum = MapReduce(
                        row,
                        Scalar(0.0f),
                        [&](Scalar a) { return ExpFn(a - maxVal); },
                        [&](Scalar a, Scalar p) { return a + p; });
                }

                // loop 3: Scale to sum to 1
                {
                    LocationGuard region_(GET_LOCATION());
                    ProfileRegion profileRegion_("softmax_3_scale");

                    auto reciprocal = Cast(Scalar(1.0), sum.GetType()) / sum;

                    Nest nest4(numColumns);
                    auto j4 = nest4.GetIndices()[0];
                    nest4.Set([&] {
                        row(j4) *= reciprocal;
                    });
                    auto nest4Schedule = nest4.CreateSchedule();
                    auto nest4Plan = nest4Schedule.CreatePlan();
                    nest4Plan.Vectorize(j4, { vectorSize, vectorUnits, true });
                }
            });

            auto schedule = nest.CreateSchedule();
        }

        template <typename ExpFnType>
        void SoftmaxifyRowsVectorizedColumnMajor(Array m, ExpFnType ExpFn)
        {
            ProfileRegion profileRegion("softmax_0_all");
            const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            const int vectorUnits = 16; // AVX-2 has 16 256-bit registers
            const int splitSize = 0;

            auto elementType = m.GetType();
            auto epsilon = Cast(Scalar(1e-7f), elementType);

            int numRows = static_cast<int>(m.Shape()[0]);
            int numColumns = static_cast<int>(m.Shape()[1]);

            auto max = MakeArray({ numRows }, elementType, "max");
            auto sum = MakeArray({ numRows }, elementType, "sum");

            ClearArray(sum);
            auto minFloat = Cast(Scalar(std::numeric_limits<float>::lowest()), elementType);
            FillArray(max, minFloat);

            // loop 1: row max
            {
                ProfileRegion profileRegion_("softmax_1_vecmax");
                Nest nest1(MemoryShape{ numRows, numColumns });
                auto i1 = nest1.GetIndices()[0];
                auto j1 = nest1.GetIndices()[1];
                nest1.Set([&]() {
                    auto val = m(i1, j1);
                    auto maxVal = Max(max(i1), val);
                    max(i1) = maxVal;
                });
                auto schedule1 = nest1.CreateSchedule();
                auto plan1 = schedule1.CreatePlan();
                if (splitSize > 0 && numRows > splitSize)
                {
                    auto [iOuter1, iInner1] = schedule1.Split(i1, splitSize);
                    schedule1.SetOrder({ iOuter1, j1, iInner1 });
                    if (splitSize >= vectorSize)
                    {
                        auto vectorUnitsToUse = splitSize >= vectorUnits ? vectorUnits : vectorSize;
                        plan1.Vectorize(iInner1, { vectorSize, vectorUnitsToUse, true });
                    }
                }
                else
                {
                    schedule1.SetOrder({ j1, i1 });
                    if (numRows >= vectorSize)
                    {
                        auto vectorUnitsToUse = numRows >= vectorUnits ? vectorUnits : vectorSize;
                        plan1.Vectorize(i1, { vectorSize, vectorUnitsToUse, true });
                    }
                }
            }

            // loop 2: compute exp(x_i-max), sum of exp(x_i-max)
            {
                ProfileRegion profileRegion_("softmax_2_expsum");
                Nest nest2(MemoryShape{ numRows, numColumns });
                auto i2 = nest2.GetIndices()[0];
                auto j2 = nest2.GetIndices()[1];
                nest2.Set([&]() {
                    auto maxVal = max(i2);
                    auto eulerVal = ExpFn(m(i2, j2) - maxVal);
                    m(i2, j2) = eulerVal;
                    sum(i2) += eulerVal;
                });
                auto schedule2 = nest2.CreateSchedule();
                auto plan2 = schedule2.CreatePlan();
                if (splitSize > 0 && numRows > splitSize)
                {
                    auto [iOuter2, iInner2] = schedule2.Split(i2, splitSize);
                    schedule2.SetOrder({ iOuter2, j2, iInner2 });
                    if (splitSize >= vectorSize)
                    {
                        auto vectorUnitsToUse = splitSize >= vectorUnits ? vectorUnits : vectorSize;
                        plan2.Vectorize(iInner2, { vectorSize, vectorUnitsToUse, true });
                    }
                }
                else
                {
                    schedule2.SetOrder({ j2, i2 });
                    if (numRows >= vectorSize)
                    {
                        auto vectorUnitsToUse = numRows >= vectorUnits ? vectorUnits : vectorSize;
                        plan2.Vectorize(i2, { vectorSize, vectorUnitsToUse, true });
                    }
                }
            }

            // loop 3: normalize
            {
                ProfileRegion profileRegion_("softmax_3_scale");
                Nest nest3(MemoryShape{ numRows, numColumns });
                auto i3 = nest3.GetIndices()[0];
                auto j3 = nest3.GetIndices()[1];
                nest3.Set([&]() {
                    auto rawSumVal = sum(i3);
                    auto sumVal = Select(rawSumVal < epsilon, Cast(Scalar(1.0f), elementType), rawSumVal);
                    m(i3, j3) /= sumVal;
                    // auto reciprocal = Cast(Scalar(1.0), sumVal.GetType()) / sumVal;
                    // m(i3, j3) *= reciprocal;
                });
                auto schedule3 = nest3.CreateSchedule();
                auto plan3 = schedule3.CreatePlan();
                if (splitSize > 0 && numRows > splitSize)
                {
                    auto [iOuter3, iInner3] = schedule3.Split(i3, splitSize);
                    schedule3.SetOrder({ iOuter3, j3, iInner3 });
                    if (splitSize >= vectorSize)
                    {
                        auto vectorUnitsToUse = splitSize >= vectorUnits ? vectorUnits : vectorSize;
                        plan3.Vectorize(iInner3, { vectorSize, vectorUnitsToUse, true });
                    }
                }
                else
                {
                    schedule3.SetOrder({ j3, i3 });
                    if (numRows >= vectorSize)
                    {
                        auto vectorUnitsToUse = numRows >= vectorUnits ? vectorUnits : vectorSize;
                        plan3.Vectorize(i3, { vectorSize, vectorUnitsToUse, true });
                    }
                }
            }
        }

        template <typename ExpFnType>
        void SoftmaxifyRowsVectorizedMixedLayout(Array m, ExpFnType ExpFn)
        {
            const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            const int vectorUnits = 16; // AVX-2 has 16 256-bit registers
            const int splitSize = 0;

            auto elementType = m.GetType();
            auto epsilon = Cast(Scalar(1e-7), elementType);

            int numRowChunks = static_cast<int>(m.Shape()[0]);
            int numColumns = static_cast<int>(m.Shape()[1]);
            int numRowsPerChunk = static_cast<int>(m.Shape()[2]);

            auto sum = MakeArray({ numRowsPerChunk }, elementType, "sum");
            auto max = MakeArray({ numRowsPerChunk }, elementType, "max");

            // outer loop: row chunks
            Nest outerNest(MemoryShape{ numRowChunks });
            ScalarIndex iOuter = outerNest.GetIndices()[0];

            outerNest.Set([&]() {
                // loop 1: row max
                Nest nest1(MemoryShape{ numRowsPerChunk, numColumns });
                auto i1 = nest1.GetIndices()[0];
                auto j1 = nest1.GetIndices()[1];

                ClearArray(sum);
                auto minFloat = Cast(Scalar(std::numeric_limits<float>::lowest()), elementType);
                FillArray(max, minFloat);

                nest1.Set([&]() {
                    auto val = m(iOuter, j1, i1);
                    auto maxVal = Max(max(i1), val);
                    max(i1) = maxVal;
                });

                // loop 2: compute exp(x_i-max), sum of exp(x_i-max)
                Nest nest2(MemoryShape{ numRowsPerChunk, numColumns });
                auto i2 = nest2.GetIndices()[0];
                auto j2 = nest2.GetIndices()[1];
                nest2.Set([&]() {
                    // sum(i2) = Cast(Scalar(0.0f), elementType);
                    auto maxVal = max(i2);
                    auto eulerVal = ExpFn(m(iOuter, j2, i2) - maxVal);
                    m(iOuter, j2, i2) = eulerVal;
                    sum(i2) += eulerVal;
                });

                // loop 3: vector div
                Nest nest3(MemoryShape{ numRowsPerChunk, numColumns });
                auto i3 = nest3.GetIndices()[0];
                auto j3 = nest3.GetIndices()[1];
                nest3.Set([&]() {
                    auto rawSumVal = sum(i3);
                    auto sumVal = Select(rawSumVal < epsilon, Cast(Scalar(1.0f), elementType), rawSumVal);

                    m(iOuter, j3, i3) /= sumVal;
                });

                auto schedule1 = nest1.CreateSchedule();
                auto schedule2 = nest2.CreateSchedule();
                auto schedule3 = nest3.CreateSchedule();

                auto plan1 = schedule1.CreatePlan();
                auto plan2 = schedule2.CreatePlan();
                auto plan3 = schedule3.CreatePlan();

                if (splitSize > 0)
                {
                    auto [iOuter1, iInner1] = schedule1.Split(i1, splitSize);
                    auto [iOuter2, iInner2] = schedule2.Split(i2, splitSize);
                    auto [iOuter3, iInner3] = schedule3.Split(i3, splitSize);

                    schedule1.SetOrder({ iOuter1, j1, iInner1 });
                    schedule2.SetOrder({ iOuter2, j2, iInner2 });
                    schedule3.SetOrder({ iOuter3, j3, iInner3 });

                    if (splitSize >= vectorSize)
                    {
                        plan1.Vectorize(iInner1, { vectorSize, vectorUnits });
                        plan2.Vectorize(iInner2, { vectorSize, vectorUnits });
                        plan3.Vectorize(iInner3, { vectorSize, vectorUnits });
                    }
                }
                else
                {
                    schedule1.SetOrder({ j1, i1 });
                    schedule2.SetOrder({ j2, i2 });
                    schedule3.SetOrder({ j3, i3 });
                    if (numRowsPerChunk >= vectorSize)
                    {
                        plan1.Vectorize(i1, { vectorSize, vectorUnits });
                        plan2.Vectorize(i2, { vectorSize, vectorUnits });
                        plan3.Vectorize(i3, { vectorSize, vectorUnits });
                    }
                }
            });

            auto outerSchedule = outerNest.CreateSchedule();
        }

        // For FusedFeedforward
        struct MatMul3Params
        {
            int M;
            int N;
            int K;
            int L;
        };

        MatMul3Params GetOuterStrides(int M, int N, int K, int L)
        {
            // Sets total size (before truncating due to matrix size)
            const int defaultStrideM = 256;
            const int defaultStrideN = 256;

            int StrideM = defaultStrideM;
            int StrideN = defaultStrideN;

            StrideM = std::min<int>(M, StrideM);
            StrideN = std::min<int>(N, StrideN);

            int StrideK = K;
            int StrideL = L;

            return { StrideM, StrideN, StrideK, StrideL };
        }

        void AccumMatMul(Array A, Array B, Array C)
        {
            MatMulMlas(A, B, C, false);
        }
    } // namespace

    void SoftmaxifyRows(Array m)
    {
        if (m.GetLayout().GetDimensionOrder() == DimensionOrder{ 0, 1 })
        {
            SoftmaxifyRowsRowMajor(m);
        }
        else if (m.GetLayout().GetDimensionOrder() == DimensionOrder{ 1, 0 })
        {
            SoftmaxifyRowsColumnMajor(m);
        }
        else
        {
            throw LogicException(LogicExceptionErrors::illegalState, "Bad layout");
        }
    }

    void SoftmaxifyRowsVectorized(Array m)
    {
        if (m.GetLayout().GetDimensionOrder() == DimensionOrder{ 0, 1 })
        {
            SoftmaxifyRowsVectorizedRowMajor(m, FastExpMlas);
        }
        else if (m.GetLayout().GetDimensionOrder() == DimensionOrder{ 1, 0 })
        {
            SoftmaxifyRowsVectorizedColumnMajor(m, FastExpMlas);
        }
        else if (m.GetLayout().GetDimensionOrder() == DimensionOrder{ 0, 1, 2 })
        {
            SoftmaxifyRowsVectorizedMixedLayout(m, FastExpMlas);
        }
        else
        {
            throw LogicException(LogicExceptionErrors::illegalState, "Bad layout");
        }
    }

    namespace
    {
        void LayerNormalizeRowsVectorizedRowMajor(Array m, Array alpha, Array beta, std::optional<Array> residual)
        {
            // Computes LayerNormalize(m) or LayerNormalize(m + residual)
            auto elementType = m.GetType();

            const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            const int vectorUnits = 16; // AVX-2 has 16 256-bit registers

            int numRows = static_cast<int>(m.Shape()[0]);
            int numColumns = static_cast<int>(m.Shape()[1]);

            Nest nest({ Range{ 0, numRows, 1 } });
            auto i = nest.GetIndices()[0];

            const float epsilon = 1e-7f;

            Scalar sum = Allocate(elementType, ScalarLayout);
            Scalar sumSquares = Allocate(elementType, ScalarLayout);
            nest.Set([&]() {
                sum = 0.0f;
                sumSquares = 0.0f;
                auto row = m.Slice({ 0 }, { i });
                auto residualRow = residual ? std::optional<Array>{ residual->Slice({ 0 }, { i }) } : std::nullopt;
                For(0, numColumns, 1, [&](Scalar j) {
                    auto val = row(j);
                    if (residualRow)
                    {
                        val = val + (*residualRow)(j);
                    }
                    sum += val;
                    sumSquares += val * val;
                });

                sum = Max(sum, epsilon);
                sumSquares = Max(sumSquares, epsilon);

                auto mean = sum / Scalar((float)numColumns);
                auto variance = (sumSquares - ((sum * sum) / Scalar((float)numColumns))) / Scalar((float)numColumns); // == (sumSquares - mean*mean*N) / N
                variance = Select(variance < Scalar(epsilon), Cast(Scalar(1.0f), elementType), variance);
                auto stdDev = Sqrt(variance);

                Nest nest2({ Range{ 0, numColumns, 1 } });
                auto j = nest2.GetIndices()[0];
                nest2.Set([&] {
                    row(j) = alpha(j) * ((row(j) - mean) / stdDev) + beta(j);
                });
                auto schedule2 = nest2.CreateSchedule();
                auto plan2 = schedule2.CreatePlan();
                plan2.Vectorize(j, { vectorSize, vectorUnits, true });
            });

            auto schedule = nest.CreateSchedule();
        }

        void LayerNormalizeRowsVectorizedColumnMajor(Array m, Array alpha, Array beta, std::optional<Array> residual)
        {
            // Computes LayerNormalize(m) or LayerNormalize(m + residual)
            const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
            const int vectorUnits = 16; // AVX-2 has 16 256-bit registers

            auto elementType = m.GetType();

            int numRows = static_cast<int>(m.Shape()[0]);
            int numColumns = static_cast<int>(m.Shape()[1]);

            auto sum = MakeArray({ numRows }, elementType, "sum");
            auto sumSquares = MakeArray({ numRows }, elementType, "sumSquares");
            auto mean = MakeArray({ numRows }, elementType, "mean");
            auto stdDev = MakeArray({ numRows }, elementType, "stdDev");

            ClearArray(sum);
            ClearArray(sumSquares);

            Nest nest({ Range{ 0, numRows, 1 } });
            auto i = nest.GetIndices()[0];

            const float epsilon = 1e-7f;

            // loop 1: sum, sum-of-squares
            Nest nest1(MemoryShape{ numRows, numColumns });
            auto i1 = nest1.GetIndices()[0];
            auto j1 = nest1.GetIndices()[1];
            nest1.Set([&]() {
                auto val = m(i1, j1);
                if (residual)
                {
                    val = val + (*residual)(i1, j1);
                }
                sum(i1) += val;
                sumSquares(i1) += (val * val);
            });

            // loop 2: compute mean, stdDev
            Nest nest2(MemoryShape{ numRows });
            auto i2 = nest2.GetIndices()[0];
            nest2.Set([&]() {
                auto sumVal = Max(sum(i2), epsilon);
                auto sumSquaresVal = Max(sumSquares(i2), epsilon);

                auto meanVal = sumVal / Scalar((float)numColumns);
                auto varianceVal = (sumSquaresVal - ((sumVal * sumVal) / Scalar((float)numColumns))) / Scalar((float)numColumns); // == (sumSquares - mean*mean*N) / N
                varianceVal = Select(varianceVal < Scalar(epsilon), Cast(Scalar(1.0f), elementType), varianceVal);
                mean(i2) = meanVal;
                stdDev(i2) = Sqrt(varianceVal);
            });

            // loop 3: normalize
            Nest nest3(MemoryShape{ numRows, numColumns });
            auto i3 = nest3.GetIndices()[0];
            auto j3 = nest3.GetIndices()[1];
            nest3.Set([&]() {
                m(i3, j3) = alpha(j3) * ((m(i3, j3) - mean(i3)) / stdDev(i3)) + beta(j3);
            });

            auto schedule1 = nest1.CreateSchedule();
            auto schedule2 = nest2.CreateSchedule();
            auto schedule3 = nest3.CreateSchedule();

            schedule1.SetOrder({ j1, i1 });
            schedule3.SetOrder({ j3, i3 });

            auto plan1 = schedule1.CreatePlan();
            auto plan2 = schedule2.CreatePlan();
            auto plan3 = schedule3.CreatePlan();

            plan1.Vectorize(i1, { vectorSize, vectorUnits, true });
            plan3.Vectorize(i3, { vectorSize, vectorUnits, true });
        }

        void LayerNormalizeVectorized(Array m, Array alpha, Array beta, std::optional<Array> residual)
        {
            if (m.GetLayout().GetDimensionOrder() == DimensionOrder{ 0, 1 })
            {
                LayerNormalizeRowsVectorizedRowMajor(m, alpha, beta, residual);
            }
            else if (m.GetLayout().GetDimensionOrder() == DimensionOrder{ 1, 0 })
            {
                LayerNormalizeRowsVectorizedColumnMajor(m, alpha, beta, residual);
            }
            else
            {
                throw LogicException(LogicExceptionErrors::illegalState, "Bad layout");
            }
        }

        void LayerNormalize(Array m, Array alpha, Array beta, std::optional<Array> residual)
        {
            auto elementType = m.GetType();

            int numRows = static_cast<int>(m.Shape()[0]);
            int numColumns = static_cast<int>(m.Shape()[1]);

            Nest nest({ Range{ 0, numRows, 1 } });
            auto i = nest.GetIndices()[0];

            const float epsilon = 1e-7f;

            Scalar sum = Allocate(elementType, ScalarLayout);
            Scalar sumSquares = Allocate(elementType, ScalarLayout);
            nest.Set([&]() {
                auto row = m.Slice({ 0 }, { i });
                auto residualRow = residual ? std::optional<Array>{ residual->Slice({ 0 }, { i }) } : std::nullopt;

                sum = 0.0f;
                sumSquares = 0.0f;
                For(0, numColumns, 1, [&](Scalar j) {
                    auto val = row(j);
                    if (residual)
                    {
                        val = val + (*residualRow)(j);
                    }

                    sum += val;
                    sumSquares += val * val;
                    row(j) = val;
                });

                sum = Max(sum, epsilon);
                sumSquares = Max(sumSquares, epsilon);

                auto mean = sum / Scalar((float)numColumns);
                auto variance = (sumSquares - ((sum * sum) / Scalar((float)numColumns))) / Scalar((float)numColumns); // == (sumSquares - mean*mean*N) / N
                variance = Select(variance < Scalar(epsilon), Cast(Scalar(1.0f), elementType), variance);
                auto stdDev = Sqrt(variance);

                For(0, numColumns, 1, [&](Scalar j) {
                    row(j) = alpha(j) * ((row(j) - mean) / stdDev) + beta(j);
                });
            });

            nest.CreateSchedule();
        }
    } // namespace

    void LayerNormalize(Array m, Array alpha, Array beta)
    {
        LayerNormalize(m, alpha, beta, std::nullopt);
    }

    void LayerNormalizeFused(Array m, Array alpha, Array beta, Array residual)
    {
        LayerNormalize(m, alpha, beta, residual);
    }

    void LayerNormalizeVectorized(Array m, Array alpha, Array beta)
    {
        LayerNormalizeVectorized(m, alpha, beta, std::nullopt);
    }

    void LayerNormalizeVectorizedFused(Array m, Array alpha, Array beta, Array residual)
    {
        LayerNormalizeVectorized(m, alpha, beta, residual);
    }

    void ReLU(Array m)
    {
        const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
        const int vectorUnits = 16; // AVX-2 has 16 256-bit registers

        Nest nest(m.Shape());
        auto i = nest.GetIndices()[0];
        auto j = nest.GetIndices()[1];

        Scalar zero = 0.0f;
        nest.Set([&]() {
            auto v = Max(zero, m(i, j));
            m(i, j) = v;
        });

        auto schedule = nest.CreateSchedule();
        auto plan = schedule.CreatePlan();
        plan.Vectorize(j, { vectorSize, vectorUnits });
    }

    void Feedforward(Array attn, Array Wff1, Array Wff2, Array ffTemp, Array output)
    {
        ProfileRegion profileRegion("feedforward_0_all");
        {
            ProfileRegion profileRegion("feedforward_1_matmul");
            MatMulMlas(attn, Wff1, ffTemp);
        }
        {
            ProfileRegion profileRegion("feedforward_2_relu");
            ReLU(ffTemp);
        }
        {
            ProfileRegion profileRegion("feedforward_3_matmul");
            MatMulMlas(ffTemp, Wff2, output);
        }
    } // namespace value

    void FusedFeedforward(Array attn, Array Wff1, Array Wff2, Array ffTemp, Array output)
    {
        ProfileRegion profileRegion("fusedfeedforward_0_all");

        // output = ReLU(attn * Wff1) * Wff2  --> E = ReLU(A * B) * D

        // Renaming arguments
        auto& A = attn;
        auto& B = Wff1;
        auto& D = Wff2;
        auto& E = output;

        auto elementType = A.GetType();

        // Declare and/or calculate constants
        const int M = (int)(A.Shape()[0]);
        const int N = (int)(B.Shape()[1]);
        const int K = (int)(A.Shape()[1]);
        const int L = (int)(D.Shape()[1]);
        const int S = 2;

        auto OuterStrides = GetOuterStrides(M, N, K, L);

#if 0
    // Debugging
    std::cout << "Problem size -- M: " << M << ", N: " << N << ", K: " << K << ", L: " << L << std::endl;
    std::cout << "OuterStrides -- M: " << OuterStrides.M << ", N: " << OuterStrides.N << ", K: " << OuterStrides.K << ", L: " << OuterStrides.L << std::endl;
    std::cout << std::endl;
#endif

        auto C = MakeArray({ OuterStrides.M, OuterStrides.N }, elementType, "CCache");

        {
            ProfileRegion profileRegion("fusedfeedforward_1_clear");
            ClearArray(output);
        }

        // Define Nest
        Nest nest({ M, N, K, L, S });

        // Get indexes
        auto [i, j, k, l, s] = nest.GetIndices<5>();

        auto schedule = nest.CreateSchedule();
        ScalarIndex iOuter, jOuter, kOuter, lOuter, iInner, jInner, kInner, lInner;
        std::tie(iOuter, iInner) = schedule.Split(i, OuterStrides.M);
        std::tie(jOuter, jInner) = schedule.Split(j, OuterStrides.N);
        std::tie(kOuter, kInner) = schedule.Split(k, OuterStrides.K);
        std::tie(lOuter, lInner) = schedule.Split(l, OuterStrides.L);

        // Note: The E matrix cache has its dimensions ordered based on this schedule order, specifically
        //       the inner split indices. So for the E matrix, iInner occurring before lInner will produce
        //       a row-major cache when there are only two indices to account for.
        //       In order for vectorization to work correctly, the order of those indices here must be
        //       the same as what is assumed by the submatrix GEMM calls or the different loopnests won't
        //       end up agreeing on which logical dimension of the cache is actually being vectorized.
        schedule.SetOrder({ jOuter, kOuter, iOuter, s, lOuter, lInner, jInner, kInner, iInner });

        // kernels for C = A * B
        auto cKernel = value::Kernel("ComputeC", [&]() {
            ProfileRegion profileRegion("fusedfeedforward_2_ckernel");
            // LogMatmulParams("Inner MatMul C with M, N, K = ", OuterStrides.M, OuterStrides.N, OuterStrides.K);
            // LogMatmulParams("    at ", iOuter, jOuter, kOuter);

            // TODO: use current loop extents to get submatrix size
            auto viewA = A.SubArray({ iOuter, kOuter }, { OuterStrides.M, OuterStrides.K });
            auto viewB = B.SubArray({ kOuter, jOuter }, { OuterStrides.K, OuterStrides.N });

            ClearArray(C);
            MatMulMlas(viewA, viewB, C);
        });

        auto eKernel = value::Kernel("ComputeE", [&]() {
            ProfileRegion profileRegion("fusedfeedforward_3_ekernel");
            // LogMatmulParams("Inner MatMul E with M, L, J = ", OuterStrides.M, OuterStrides.L, OuterStrides.N);
            // LogMatmulParams("    at ", iOuter, lOuter, lOuter);

            // TODO: use current loop extents to get submatrix size
            auto viewD = D.SubArray({ jOuter, lOuter }, { OuterStrides.N, OuterStrides.L });
            auto viewE = E.SubArray({ iOuter, lOuter }, { OuterStrides.M, OuterStrides.L });

            ReLU(C.GetValue());
            AccumMatMul(C, viewD, viewE);
        });

        schedule.AddKernel(cKernel, First(iInner) && First(jInner) && First(kInner) && First(lInner) && First(s), IsDefined(iOuter) && IsDefined(jOuter) && IsDefined(kOuter) && IsDefined(lOuter) && IsDefined(s));
        schedule.AddKernel(eKernel, First(iInner) && First(lInner) && First(jInner) && Last(kInner) && Last(s), IsDefined(iOuter) && IsDefined(jOuter) && IsDefined(kOuter) && IsDefined(lOuter) && IsDefined(s));
    }
} // namespace value
} // namespace accera
