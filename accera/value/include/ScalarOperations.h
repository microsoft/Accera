////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ValueType.h"

namespace accera
{
namespace value
{
    class Scalar;

    /// <summary> Integer / bitwise operators </summary>
    Scalar Bitcast(Scalar value, ValueType destType);
    Scalar BitwiseAnd(Scalar s1, Scalar s2);
    Scalar BitwiseOr(Scalar s1, Scalar s2);
    Scalar BitwiseXOr(Scalar s1, Scalar s2);
    Scalar BitwiseNot(Scalar value);
    Scalar ShiftLeft(Scalar value, Scalar shift);
    Scalar SignedShiftRight(Scalar value, Scalar shift);
    Scalar UnsignedShiftRight(Scalar value, Scalar shift);

    /// <summary> Arithmetic operators </summary>
    Scalar Add(Scalar, Scalar);
    Scalar Subtract(Scalar, Scalar);
    Scalar Multiply(Scalar, Scalar);
    Scalar Divide(Scalar, Scalar);
    Scalar Modulo(Scalar, Scalar);
    Scalar Fma(Scalar a, Scalar b, Scalar c);
    Scalar FusedMultiplyAdd(Scalar a, Scalar b, Scalar c); // returns (a*b)+c

    /// <summary> Math intrinsics </summary>
    Scalar Abs(Scalar s);
    Scalar Cos(Scalar s);
    Scalar Exp(Scalar s);
    Scalar Log(Scalar s);
    Scalar Log10(Scalar s);
    Scalar Log2(Scalar s);
    Scalar Max(Scalar s1, Scalar s2);
    Scalar Min(Scalar s1, Scalar s2);
    Scalar Clamp(Scalar s, Scalar min, Scalar max);
    Scalar Pow(Scalar base, Scalar exp);
    Scalar Sin(Scalar s);
    Scalar Sqrt(Scalar s);
    Scalar Tan(Scalar s);
    Scalar Sinh(Scalar s);
    Scalar Cosh(Scalar s);
    Scalar Tanh(Scalar s);
    Scalar Square(Scalar s);

    Scalar Round(Scalar s);
    Scalar Remainderf(Scalar numer, Scalar denom);
    Scalar Floor(Scalar s);
    Scalar Ceil(Scalar s);
    Scalar CopySign(Scalar s1, Scalar s2); // Note: not implemented
    Scalar Sign(Scalar s); // Note: not implemented

    Scalar Select(Scalar cmp, Scalar a, Scalar b); // returns (cmp ? a : b)
    Scalar LogicalNot(Scalar v);
} // namespace value
} // namespace accera
