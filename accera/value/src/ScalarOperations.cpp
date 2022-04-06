////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ScalarOperations.h"
#include "EmitterContext.h"
#include "MLIREmitterContext.h"
#include "Scalar.h"
#include "ValueType.h"

#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <stdexcept>

namespace accera
{
using namespace utilities;

namespace value
{
    namespace
    {
        template <typename Op>
        struct ScalarOpBuilder
        {
            template <typename... Args>
            ScalarOpBuilder(Args... args)
            {
                using namespace mlir;

                auto& builder = ::accera::value::GetMLIRContext().GetOpBuilder();
                auto loc = builder.getUnknownLoc();
                Operation* op = builder.create<Op>(loc, accera::ir::util::ToSignlessMLIRValue(builder, UnwrapScalar(args))...); // args must be signless
                assert(op->getNumResults() == 1 && "Op must have a single return value");
                value = Wrap(op->getResult(0));
            }

            operator Scalar()
            {
                return value;
            }

            Scalar value;
        };

    } // namespace

    Scalar Bitcast(Scalar value, ValueType type)
    {
        return GetContext().Bitcast(value, type);
    }

    Scalar BitwiseAnd(Scalar s1, Scalar s2)
    {
        return ScalarOpBuilder<mlir::AndOp>(s1, s2);
    }

    Scalar BitwiseOr(Scalar s1, Scalar s2)
    {
        return ScalarOpBuilder<mlir::OrOp>(s1, s2);
    }

    Scalar BitwiseXOr(Scalar s1, Scalar s2)
    {
        return ScalarOpBuilder<mlir::XOrOp>(s1, s2);
    }

    Scalar BitwiseNot(Scalar value)
    {
        // BUGBUG: no flip or invert op in LLVM
        // Calculation:
        //   ~x = -(x+1) = -1-x
        // An alternative:
        //   ~x = 2^n - 1 - x, where n = bit width of x
        // consider making an op if this will be composed with other ops
        return Subtract(Cast(-1, value.GetType()), value);
    }

    Scalar ShiftLeft(Scalar value, Scalar shift)
    {
        return ScalarOpBuilder<mlir::ShiftLeftOp>(value, shift);
    }

    Scalar SignedShiftRight(Scalar value, Scalar shift)
    {
        return ScalarOpBuilder<mlir::SignedShiftRightOp>(value, shift);
    }

    Scalar UnsignedShiftRight(Scalar value, Scalar shift)
    {
        return ScalarOpBuilder<mlir::UnsignedShiftRightOp>(value, shift);
    }

    Scalar Add(Scalar s1, Scalar s2)
    {
        Scalar copy = s1.Copy();
        return copy += s2;
    }

    Scalar Subtract(Scalar s1, Scalar s2)
    {
        Scalar copy = s1.Copy();
        return copy -= s2;
    }

    Scalar Multiply(Scalar s1, Scalar s2)
    {
        Scalar copy = s1.Copy();
        return copy *= s2;
    }

    Scalar Divide(Scalar s1, Scalar s2)
    {
        Scalar copy = s1.Copy();
        return copy /= s2;
    }

    Scalar Modulo(Scalar s1, Scalar s2)
    {
        Scalar copy = s1.Copy();
        return copy %= s2;
    }

    Scalar FusedMultiplyAdd(Scalar a, Scalar b, Scalar c)
    {
        return Fma(a, b, c);
    }

    Scalar Fma(Scalar a, Scalar b, Scalar c)
    {
        return (a * b) + c;
    }

    Scalar Abs(Scalar s)
    {
        switch (s.GetType())
        {
        case ValueType::Float16:
            [[fallthrough]];
        case ValueType::Float:
            [[fallthrough]];
        case ValueType::Double:
            return ScalarOpBuilder<mlir::AbsFOp>(s);
        case ValueType::Undefined:
            [[fallthrough]];
        case ValueType::Void:
            throw std::logic_error("Called Abs on invalid type");
        default:
            return Select(s < Cast(0, s.GetType()), -s, s);
        }
    }

    Scalar Ceil(Scalar s)
    {
        return ScalarOpBuilder<mlir::CeilFOp>(s);
    }

    Scalar Floor(Scalar s)
    {
        return ScalarOpBuilder<mlir::FloorFOp>(s);
    }

    Scalar CopySign(Scalar s1, Scalar s2)
    {
        return ScalarOpBuilder<mlir::CopySignOp>(s1, s2);
    }

    Scalar Cos(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::CosOp>(s);
    }

    Scalar Exp(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::ExpOp>(s);
    }

    Scalar Log(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::LogOp>(s);
    }

    Scalar Log10(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::Log10Op>(s);
    }

    Scalar Log2(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::Log2Op>(s);
    }

    Scalar Max(Scalar s1, Scalar s2)
    {
        return Select(s1 > s2, s1, s2);
    }

    Scalar Min(Scalar s1, Scalar s2)
    {
        return Select(s1 < s2, s1, s2);
    }

    Scalar Clamp(Scalar s, Scalar min, Scalar max)
    {
        return Min(max, Max(s, min));
    }

    Scalar Pow(Scalar base, Scalar exp)
    {
        if (base.GetValue().IsIntegral() && exp.GetValue().IsIntegral())
        {
            auto baseF = Cast(base, ValueType::Float);
            auto expF = Cast(exp, ValueType::Float);
            return Cast(Pow(baseF, expF), base.GetType());
        }
        return ScalarOpBuilder<mlir::math::PowFOp>(base, exp);
    }

    Scalar Select(Scalar cmp, Scalar a, Scalar b)
    {
        return ScalarOpBuilder<mlir::SelectOp>(cmp, a, b);
    }

    Scalar Sin(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::SinOp>(s);
    }

    Scalar Sign(Scalar s)
    {
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
    }

    Scalar Sqrt(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::SqrtOp>(s);
    }

    Scalar Tan(Scalar s)
    {
        // consider making an op if this will be composed with other ops
        return Divide(Sin(s), Cos(s));
    }

    Scalar Sinh(Scalar s)
    {
        // consider making an op if this will be composed with other ops
        auto negS = ScalarOpBuilder<mlir::NegFOp>(s);
        return Divide(Exp(s) - Exp(negS), Cast(2, s.GetType()));
    }

    Scalar Cosh(Scalar s)
    {
        // consider making an op if this will be composed with other ops
        auto negS = ScalarOpBuilder<mlir::NegFOp>(s);
        return Divide(Exp(s) + Exp(negS), Cast(2, s.GetType()));
    }

    Scalar Tanh(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::TanhOp>(s);
    }

    Scalar Square(const Scalar x)
    {
        auto y = x.Copy();
        return x * y;
    }

    // TODO: should this live in Scalar.cpp?
    Scalar LogicalNot(const Scalar v)
    {
        // Returns True if if v is 0, else False
        Scalar r;
        if (v.GetType() == ValueType::Boolean)
        {
            accera::utilities::Boolean t(true);
            r = (v != t);
        }
        else
        {
            if (!v.GetValue().IsIntegral())
            {
                throw LogicException(LogicExceptionErrors::illegalState);
            }

            r = (v == Cast(0, v.GetType()));
        }
        return r;
    }

} // namespace value
} // namespace accera
