////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MatrixOperations.h"
#include "Cache.h"
#include "EmitterContext.h"
#include "Index.h"
#include "Matrix.h"
#include "Nest.h"
#include "Plan.h"
#include "Scalar.h"
#include "Schedule.h"

#include <utilities/include/StringUtil.h>

namespace accera
{
using namespace utilities;

namespace value
{

    Matrix ToMatrix(Value data, int numRows, int numCols)
    {
        Value matrix = data;
        auto size = data.GetLayout().GetActiveSize().NumElements();
        if (size != numRows * numCols || !data.GetLayout().IsContiguous())
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 accera::utilities::FormatString("data must be contiguous and have size %zu = %d * %d", size, numRows, numCols));
        }
        matrix.SetLayout(utilities::MemoryLayout{ { numRows, numCols } });
        return matrix;
    }

    Scalar Sum(Matrix matrix)
    {
        Scalar result = Allocate(matrix.GetType(), ScalarLayout);

        For(matrix, [&](auto row, auto column) {
            result += matrix(row, column);
        });

        return result;
    }

    void For(Matrix matrix, std::function<void(Scalar, Scalar)> fn)
    {
        For(std::string{}, matrix, fn);
    }

    void For(const std::string& name, Matrix matrix, std::function<void(Scalar, Scalar)> fn)
    {
        auto layout = matrix.GetValue().GetLayout();
        if (layout.NumDimensions() != 2)
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Layout being looped over must be two-dimensional");
        }

        GetContext().For(
            layout,
            [fn = std::move(fn)](std::vector<Scalar> coordinates) {
                fn(coordinates[0], coordinates[1]);
            },
            name);
    }

    Matrix MatrixMatrixMultiply(Matrix m1, Matrix m2)
    {
        const int OutputRows = (int)(m1.Rows()); // M
        const int OutputColumns = (int)(m2.Columns()); // N
        [[maybe_unused]] const int InnerDimension = (int)(m1.Columns()); // K
        Matrix output = MakeMatrix(OutputRows, OutputColumns, m1.GetType());
        MatrixMatrixMultiply(m1, m2, output);
        return output;
    }

    void MatrixMatrixMultiply(Matrix m1, Matrix m2, Matrix output)
    {
        // MLAS Value MatrixMatrixMultiply

        // Declare and/or calculate constants
        const int OutputRows = (int)(m1.Rows()); // M
        const int OutputColumns = (int)(m2.Columns()); // N
        const int InnerDimension = (int)(m1.Columns()); // K

        // Schedule constants
        // TODO : read these values from the target system
        const int vectorSize = 8; // AVX-2 gives 256-bit registers, which can hold 8 floats
        const int vectorUnits = 16; // AVX-2 has 16 256-bit registers
        const int kUnroll = 4;

        const int NumRowsInKernel = 6;
        const int NumColumnsInKernel = 2 * vectorSize;

        int columnBlock = std::min(128, OutputColumns);
        int innerDimensionBlock = std::min(512, InnerDimension);

        // Define Nest
        Nest nest({ OutputRows, OutputColumns, InnerDimension });

        // Get indexes
        auto indices = nest.GetIndices();
        Scalar i = indices[0];
        Scalar j = indices[1];
        Scalar k = indices[2];
        nest.Set([&]() { output(i, j) += m1(i, k) * m2(k, j); });

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
        if (OutputColumns > 128)
        {
            plan.AddCache(m2, jKernelOuter2);
        }
        plan.AddCache(output, iInner);

        // Set unrolling
        schedule.Unroll(jKernelOuter);
        schedule.Unroll(iInner);
        plan.Vectorize(jInner3, { vectorSize, vectorUnits });
    }

    Vector MatrixVectorMultiply(Matrix m, Vector v)
    {
        Vector result = Allocate(v.GetType(), m.Rows());
        Scalar first = Allocate(ValueType::Int32, ScalarLayout);
        if (m.Columns() != v.Size())
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 accera::utilities::FormatString("Vector size %d must match number of columns in the matrix %d", v.Size(), m.Columns()));
        }
        first = 1;
        For(m, [&](Scalar row, Scalar col) {
            If(first == 1, [&] {
                result[row] = m(row, col) * v(col);
                first = 0;
            }).Else([&] {
                result[row] += m(row, col) * v(col);
            });
        });
        return result;
    }
} // namespace value
} // namespace accera
