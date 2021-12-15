////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ArrayOperations.h"
#include "Array.h"
#include "Cache.h"
#include "Kernel.h"
#include "KernelPredicate.h"
#include "Matrix.h"
#include "Nest.h"
#include "Plan.h"
#include "Schedule.h"
#include "Vector.h"

#include <utilities/include/Exception.h>

#include <limits>

namespace accera
{
using namespace utilities;

namespace value
{
    Scalar VectorMax(Array v)
    {
        auto elementType = v.GetType();
        auto minVal = Cast(Scalar(std::numeric_limits<float>::lowest()), elementType);
        return Reduce(v, minVal, [](Scalar a, Scalar s) { return Max(s, a); });
    }

    Scalar VectorSum(Array v)
    {
        auto elementType = v.GetType();
        return Reduce(v, Cast(Scalar(0.0f), elementType), [](Scalar a, Scalar s) { return a + s; });
    }

    void ClearArray(Array A)
    {
        Nest nest(A.Shape());
        auto elementType = A.GetType();

        auto indices = nest.GetIndices();
        nest.Set([&]() {
            A(indices) = Cast(Scalar(0.0f), elementType);
        });

        auto schedule = nest.CreateSchedule();
    }

    void FillArray(Array A, Scalar val)
    {
        Nest nest(A.Shape());

        auto indices = nest.GetIndices();
        nest.Set([&]() {
            A(indices) = val;
        });

        auto schedule = nest.CreateSchedule();
    }

    void CopyArray(Array A, Array B)
    {
        ThrowIf(A.Shape() != B.Shape(), InputExceptionErrors::invalidSize, "Arrays must have the same size");

        Nest nest(A.Shape());
        auto indices = nest.GetIndices();
        nest.Set([&]() {
            B(indices) = A(indices);
        });

        auto schedule = nest.CreateSchedule();
    }

    void ClearMatrix(Array A)
    {
        ClearArray(A);
    }

    void TransposeMatrix(Array Aarr, Array Barr)
    {
        Matrix A(Aarr.GetValue());
        Matrix B(Barr.GetValue());

        const int M = (int)A.Rows();
        const int N = (int)A.Columns();

        const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
        const int vectorUnits = 16; // AVX-2 has 16 256-bit registers

        int rowBlock = std::min(128, M);
        int columnBlock = std::min(128, N);
        const int NumRowsInKernel = 6;
        const int NumColumnsInKernel = 2 * vectorSize;

        Nest nest(MemoryShape{ M, N });

        ScalarIndex i, j;
        std::tie(i, j) = nest.GetIndices<2>();

        auto schedule = nest.CreateSchedule();

        auto computeKernel = value::Kernel("transpose", [&]() {
            B(j, i) = A(i, j);
        });

        schedule.AddKernel(computeKernel);

        auto [iBlock, iInner1] = schedule.Split(i, rowBlock);
        auto [jBlock, jInner1] = schedule.Split(j, columnBlock);
        auto [iKernelOuter, iInner] = schedule.Split(iInner1, NumRowsInKernel);
        auto [jKernelOuter1, jInner2] = schedule.Split(jInner1, NumColumnsInKernel);
        auto [jKernelOuter, jInner] = schedule.Split(jInner2, vectorSize);

        schedule.SetOrder({ iBlock, jBlock, iKernelOuter, jKernelOuter1, iInner, jKernelOuter, jInner });

        auto plan = schedule.CreatePlan();

        if (M > 128)
        {
            plan.AddCache(B, iKernelOuter);
        }

        schedule.Unroll(jKernelOuter);
        schedule.Unroll(iInner);
        plan.Vectorize(jInner, { vectorSize, vectorUnits });
    }

    //
    // Matrix-multiply implementations
    //
    void MatMulBasic(Array A, Array B, Array C, bool clearC)
    {
        const int M = (int)A.Shape()[0];
        const int K = (int)A.Shape()[1];
        const int N = (int)B.Shape()[1];
        ThrowIfNot(M == (int)C.Shape()[0]);
        ThrowIfNot(K == (int)B.Shape()[0]);
        ThrowIfNot(N == (int)C.Shape()[1]);

        if (clearC) ClearMatrix(C);

        Nest nest(MemoryShape{ M, N, K });

        ScalarIndex i, j, k;
        std::tie(i, j, k) = nest.GetIndices<3>();

        nest.Set([&]() {
            C(i, j) += A(i, k) * B(k, j);
        });

        auto schedule = nest.CreateSchedule();
    }

    void MatMulSimpleTiled(Array A, Array B, Array C, bool clearC)
    {
        const int M = (int)A.Shape()[0];
        const int K = (int)A.Shape()[1];
        const int N = (int)B.Shape()[1];
        ThrowIfNot(M == (int)C.Shape()[0]);
        ThrowIfNot(K == (int)B.Shape()[0]);
        ThrowIfNot(N == (int)C.Shape()[1]);

        int cacheRows = 32;
        int cacheColumns = 32;

        Nest nest(MemoryShape{ M, N, K });

        ScalarIndex i, j, k;
        std::tie(i, j, k) = nest.GetIndices<3>();

        auto schedule = nest.CreateSchedule();

        auto [iOuter, iInnerRef] = schedule.Split(i, cacheRows);
        auto [jOuter, jInnerRef] = schedule.Split(j, cacheColumns);
        ScalarIndex iInner = iInnerRef;
        ScalarIndex jInner = jInnerRef;

        auto initCKernel = value::Kernel("initC", [&]() {
            C(i, j) = 0.0f;
        });

        auto computeCKernel = value::Kernel("matmulC", [&]() {
            C(i, j) += A(i, k) * B(k, j);
        });

        if (clearC)
            schedule.AddKernel(initCKernel, { First(k) });
        schedule.AddKernel(computeCKernel);

        schedule.SetOrder({ iOuter, jOuter, k, iInner, jInner });
    }

    // MLAS Value MatrixMatrixMultiply
    void MatMulMlas(Array A, Array B, Array C, bool clearC)
    {
        using namespace value;

        const int M = (int)A.Shape()[0];
        const int K = (int)A.Shape()[1];
        const int N = (int)B.Shape()[1];
        ThrowIfNot(M == (int)C.Shape()[0]);
        ThrowIfNot(K == (int)B.Shape()[0]);
        ThrowIfNot(N == (int)C.Shape()[1]);

        if (clearC)
        {
            ClearMatrix(C);
        }

        Nest nest(MemoryShape{ M, N, K });

        ScalarIndex i, j, k;
        std::tie(i, j, k) = nest.GetIndices<3>();

        nest.Set([&]() {
            C(i, j) += A(i, k) * B(k, j);
        });

        auto schedule = nest.CreateSchedule();

        // Schedule constants
        // TODO : read these values from the target system
        int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
        int vectorUnits = 16; // AVX-2 has 16 256-bit registers
        int kUnroll = 4;

        int NumRowsInKernel = 6;
        int NumColumnsInKernel = 2 * vectorSize;

        int columnBlock = 256;
        int innerDimensionBlock = 128;
        if (N < K)
        {
            std::swap(columnBlock, innerDimensionBlock);
        }

        // Apply a simple stretching to the kernel size to fit the output shape
        if (NumColumnsInKernel > N)
        {
            while (NumColumnsInKernel > N)
            {
                NumRowsInKernel *= 2;
                NumColumnsInKernel /= 2;
            }
        }
        else if (NumRowsInKernel > M)
        {
            while (NumRowsInKernel > M)
            {
                NumRowsInKernel /= 2;
                NumColumnsInKernel *= 2;
            }
        }
        // Now clamp
        NumRowsInKernel = std::min(NumRowsInKernel, M);
        NumColumnsInKernel = std::min(NumColumnsInKernel, N);

        // Apply a simple stretching to the block sizes to use as much of
        // the original columnBlock x innerDimensionBlock area as possible
        while (columnBlock > N)
        {
            if ((columnBlock / 2) < NumColumnsInKernel)
            {
                // Don't shrink the column block smaller than NumColumnsInKernel
                break;
            }
            columnBlock /= 2;
            innerDimensionBlock *= 2;
        }
        while (innerDimensionBlock > K)
        {
            innerDimensionBlock /= 2;
            columnBlock *= 2;
        }
        // Now clamp
        columnBlock = std::min(columnBlock, N);
        innerDimensionBlock = std::min(innerDimensionBlock, K);

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

        [[maybe_unused]] auto bColumnStride = B.GetLayout().GetIncrement(0);
        if (((N * K) > (128 * 128)) || (B.GetLayout().GetIncrement(0) < B.GetLayout().GetIncrement(1)))
        {
            plan.AddCache(B, jKernelOuter2, CacheIndexing::GlobalToPhysical, CacheAllocation::Automatic, MemorySpace::Shared);
        }
        plan.AddCache(C, iInner, CacheIndexing::GlobalToPhysical, CacheAllocation::Automatic, MemorySpace::Shared);

        // Set unrolling
        schedule.Unroll(jKernelOuter);
        schedule.Unroll(iInner);
        if (NumColumnsInKernel >= vectorSize)
        {
            plan.Vectorize(jInner3, { vectorSize, vectorUnits });
        }
    }

} // namespace value
} // namespace accera
