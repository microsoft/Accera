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

#include "ir/include/value/ValueDialect.h"

#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <stdexcept>
#include <tuple>

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
        return ScalarOpBuilder<mlir::arith::AndIOp>(s1, s2);
    }

    Scalar BitwiseOr(Scalar s1, Scalar s2)
    {
        return ScalarOpBuilder<mlir::arith::OrIOp>(s1, s2);
    }

    Scalar BitwiseXOr(Scalar s1, Scalar s2)
    {
        return ScalarOpBuilder<mlir::arith::XOrIOp>(s1, s2);
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
        return ScalarOpBuilder<mlir::arith::ShLIOp>(value, shift);
    }

    Scalar SignedShiftRight(Scalar value, Scalar shift)
    {
        return ScalarOpBuilder<mlir::arith::ShRSIOp>(value, shift);
    }

    Scalar UnsignedShiftRight(Scalar value, Scalar shift)
    {
        return ScalarOpBuilder<mlir::arith::ShRUIOp>(value, shift);
    }

    Scalar Add(Scalar s1, Scalar s2)
    {
        Scalar copy = s1.Copy();
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(copy, s2);
        return lhs += rhs;
    }

    Scalar Subtract(Scalar s1, Scalar s2)
    {
        Scalar copy = s1.Copy();
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(copy, s2);
        return lhs -= rhs;
    }

    Scalar Multiply(Scalar s1, Scalar s2)
    {
        Scalar copy = s1.Copy();
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(copy, s2);
        return lhs *= rhs;
    }

    Scalar Divide(Scalar s1, Scalar s2)
    {
        Scalar copy = s1.Copy();
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(copy, s2);
        return lhs /= rhs;
    }

    Scalar Modulo(Scalar s1, Scalar s2)
    {
        Scalar copy = s1.Copy();
        auto&& [lhs, rhs] = Scalar::MakeTypeCompatible(copy, s2);
        return lhs %= rhs;
    }

    Scalar FusedMultiplyAdd(Scalar a, Scalar b, Scalar c)
    {
        return Fma(a, b, c);
    }

    Scalar Fma(Scalar a, Scalar b, Scalar c)
    {
        return ScalarOpBuilder<mlir::math::FmaOp>(a, b, c);
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
            return ScalarOpBuilder<mlir::math::AbsOp>(s);
        case ValueType::Undefined:
            [[fallthrough]];
        case ValueType::Void:
            throw std::logic_error("Called Abs on invalid type");
        default:
            return Select(s < Cast(0, s.GetType()), -s, s);
        }
    }

    Scalar Round(Scalar s)
    {
        return GetContext().Round(s);
    }

    Scalar Remainderf(Scalar numer, Scalar denom)
    {
        static auto remainderfFunction = [&]() {
            FunctionDeclaration remainderfDecl("remainderf");
            remainderfDecl.External(true)
                            .Decorated(false)
                            .Parameters(Value(ValueType::Float, ScalarLayout), Value(ValueType::Float, ScalarLayout))
                            .Returns(Value(ValueType::Float, ScalarLayout));
            return GetContext().DeclareExternalFunction(remainderfDecl);
        }();
        return Scalar(*remainderfFunction(std::vector<Value>{Wrap(UnwrapScalar(numer)), Wrap(UnwrapScalar(denom))})); // TODO : fix this Wrap(Unwrap(...)) pattern... it's currently needed to invoke GetElement on a sliced array
    }

    Scalar Ceil(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::CeilOp>(s);
    }

    Scalar Floor(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::FloorOp>(s);
    }

    Scalar CopySign(Scalar s1, Scalar s2)
    {
        return ScalarOpBuilder<mlir::math::CopySignOp>(s1, s2);
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
        return GetContext().BinaryOperation(ValueBinaryOperation::max, s1.GetValue(), s2.GetValue());
    }

    Scalar Min(Scalar s1, Scalar s2)
    {
        return GetContext().BinaryOperation(ValueBinaryOperation::min, s1.GetValue(), s2.GetValue());
    }

    Scalar Clamp(Scalar s, Scalar min, Scalar max)
    {
        std::tie(min, max) = Scalar::MakeTypeCompatible(min, max);
        std::tie(s, min) = Scalar::MakeTypeCompatible(s, min);
        std::tie(s, max) = Scalar::MakeTypeCompatible(s, max);

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
        std::tie(a, b) = Scalar::MakeTypeCompatible(a, b);
        return ScalarOpBuilder<mlir::SelectOp>(cmp, a, b);
    }

    Scalar Sin(Scalar s)
    {
        return ScalarOpBuilder<mlir::math::SinOp>(s);
    }

    Scalar Sign(Scalar s)
    {
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, "Sign operator not implemented.");
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
        auto negS = ScalarOpBuilder<mlir::arith::NegFOp>(s);
        return Divide(Exp(s) - Exp(negS), Cast(2, s.GetType()));
    }

    Scalar Cosh(Scalar s)
    {
        // consider making an op if this will be composed with other ops
        auto negS = ScalarOpBuilder<mlir::arith::NegFOp>(s);
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
                throw LogicException(LogicExceptionErrors::illegalState, "Logical NOT (!) operator can only be applied on integer types but got " + ToString(v.GetType()) + ".");
            }

            r = (v == Cast(0, v.GetType()));
        }
        return r;
    }

} // namespace value
} // namespace accera
