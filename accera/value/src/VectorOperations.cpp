////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "VectorOperations.h"
#include "EmitterContext.h"
#include "FunctionDeclaration.h"
#include "Scalar.h"
#include "Vector.h"

namespace accera
{
using namespace utilities;

namespace value
{
    Scalar Sum(Vector input)
    {
        return GetContext().Sum(input);
    }

    Vector ToVector(Value data)
    {
        Value flat = data;
        flat.SetLayout(data.GetLayout().Flatten());
        return flat;
    }

    Scalar Dot(Vector v1, Vector v2)
    {
        if (v1.Size() != v2.Size())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }
        if (v1.GetType() != v2.GetType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        auto defaultImpl = [](Vector v1, Vector v2) {
            Scalar result = Allocate(v1.GetType(), ScalarLayout);
            For(v1, [&](auto index) {
                result += v1[index] * v2[index];
            });

            return result;
        };

        return defaultImpl(v1, v2);
    }

    Scalar Max(Vector input)
    {
        return GetContext().Max(input);
    }

    void For(Vector v, std::function<void(Scalar)> fn)
    {
        For(std::string{}, v, fn);
    }

    void For(const std::string& name, Vector v, std::function<void(Scalar)> fn)
    {
        auto layout = v.GetValue().GetLayout();

        if (layout.NumDimensions() != 1)
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Layout being looped over must be one-dimensional");
        }

        GetContext().For(
            layout,
            [fn = std::move(fn)](std::vector<Scalar> coordinates) { fn(coordinates[0]); },
            name);
    }

} // namespace value
} // namespace accera
