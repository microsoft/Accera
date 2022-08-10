////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Debugging.h"
#include "MLIREmitterContext.h"
#include "Nest.h"
#include "Schedule.h"

#include <ir/include/IRUtil.h>

#include <utilities/include/Exception.h>

namespace accera
{
using namespace utilities;

namespace value
{
    class LocationGuardImpl
    {
    public:
        LocationGuardImpl(FileLocation location) :
            _loc(
                accera::ir::util::GetLocation(
                    ::accera::value::GetMLIRContext().GetOpBuilder(),
                    location.file,
                    location.line)) {}

        ~LocationGuardImpl() = default;
        mlir::Location GetLocation() const
        {
            return _loc;
        }

    private:
        mlir::Location _loc;
    };

    LocationGuard::LocationGuard(FileLocation location) :
        _impl(std::make_unique<LocationGuardImpl>(location)) {}
    LocationGuard::~LocationGuard() = default;
    mlir::Location LocationGuard::GetLocation() const { return _impl->GetLocation(); }

    // Compares two arrays by checking whether they are equal up to the specified tolerance
    // Outputs mismatches to stderr
    // Inspired by numpy.testing.assert_allclose
    void CheckAllClose(Array actual, Array desired, float tolerance, const std::vector<ScalarDimension>& runtimeSizes)
    {
        using namespace std::string_literals;

        ThrowIfNot(actual.Shape() == desired.Shape());
        auto atol = Scalar(tolerance);
        auto diff = MakeArray(actual.Shape(), ValueType::Float, "diff");

        // BUGBUG: Scalar binary ops pointer deferencing error, using Arrays as a workaround
        auto maxAbsoluteDiff = MakeArray(MemoryShape{ 1 }, diff.GetType(), "maxAbsoluteDiff");
        auto count = MakeArray(MemoryShape{ 1 }, ValueType::Int32, "count");

        auto zero = Scalar(0.0f);
        auto max = Scalar(std::numeric_limits<float>::max());
        auto zeroCount = Cast(Scalar(0), count.GetType());
        auto oneCount = Cast(Scalar(1), count.GetType());
        auto total = Cast(Scalar(actual.Size()), count.GetType());

        Nest nest(actual.Shape(), runtimeSizes);
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

        auto schedule = nest.CreateSchedule();
    }

} // namespace value

} // namespace accera
