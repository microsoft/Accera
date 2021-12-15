////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FastMath.h"
#include "EmitterContext.h"
#include "MLIREmitterContext.h"
#include "Scalar.h"
#include "ScalarOperations.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

namespace accera
{
using namespace utilities;

namespace value
{
    namespace
    {
        const struct
        {
            float LowerRange;
            float UpperRange;
            float LowerRangeSumExp;
            float UpperRangeSumExp;
            float RoundingBias;
            float Log2Reciprocal;
            float Log2High;
            float Log2Low;
            float poly_0;
            float poly_1;
            float poly_2;
            float poly_3;
            float poly_4;
            float poly_56;
            int32_t MinimumExponent;
            int32_t MaximumExponent;
        } MlasExpConstants = {
            -103.9720840454f, // LowerRange
            88.7762626647950f, // UpperRange
            -88.3762626647949f, // LowerRangeSumExp
            88.3762626647949f, // UpperRangeSumExp
            12582912.0f, // RoundingBias
            1.44269504088896341f, // Log2Reciprocal
            -6.93145752e-1f, // Log2High
            -1.42860677e-6f, // Log2Low
            0x1.694000p-10, // poly_0
            0x1.125edcp-7, // poly_1
            0x1.555b5ap-5, // poly_2
            0x1.555450p-3, // poly_3
            0x1.fffff6p-2, // poly_4
            0x1.000000p+0, // poly_56
            int32_t(0xC1000000), // MinimumExponent
            int32_t(0x3F800000), // MaximumExponent
        };

        Scalar IntAsFloat(Scalar i)
        {
            // TODO: assert i is int32
            return Bitcast(i, ValueType::Float);
        }

        Scalar FloatAsInt(Scalar f)
        {
            // TODO: assert f is float32
            return Bitcast(f, ValueType::Int32);
        }
    } // namespace

    Scalar FastExp(Scalar a)
    {
        // adapted from https://forums.developer.nvidia.com/t/a-more-accurate-performance-competitive-implementation-of-expf/47528

        // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
        auto j = Fma(1.442695f, a, 12582912.f) - 12582912.f; // 0x1.715476p0, 0x1.8p23
        auto f = Fma(j, -6.93145752e-1f, a); // -0x1.62e400p-1  // log_2_hi
        f = Fma(j, -1.42860677e-6f, f); // -0x1.7f7d1cp-20 // log_2_lo
        auto i = Cast(j, ValueType::Int32); // floor?

        // approximate r = exp(f) on interval [-log(2)/2, +log(2)/2]
        Scalar r = 1.37805939e-3f; // 0x1.694000p-10
        r = Fma(r, f, 8.37312452e-3f); // 0x1.125edcp-7
        r = Fma(r, f, 4.16695364e-2f); // 0x1.555b5ap-5
        r = Fma(r, f, 1.66664720e-1f); // 0x1.555450p-3
        r = Fma(r, f, 4.99999851e-1f); // 0x1.fffff6p-2
        r = Fma(r, f, 1.00000000e+0f); // 0x1.000000p+0
        r = Fma(r, f, 1.00000000e+0f); // 0x1.000000p+0
        // exp(a) = 2**i * r;
        auto ia = Select(i > 0, Scalar((int32_t)0), Scalar((int32_t)0x83000000));
        auto s = IntAsFloat(0x7f000000 + ia);

        auto t = IntAsFloat(ShiftLeft(i, 23) - ia);
        r = r * s;
        r = r * t;

        // handle special cases: severe overflow / underflow
        auto overflow = Abs(a) >= 104.0f;
        auto ia2 = FloatAsInt(a);
        auto overflowResult = IntAsFloat((Select(ia2 > 0, Scalar((int32_t)0x7f800000), Scalar((int32_t)0))));
        r = Select(overflow, overflowResult, r);

        return r;
    }

    Scalar FastExpMlas(Scalar a)
    {
        // adapted from MLAS: ProcessSingleVector loop of routine MlasComputeSumExpF32KernelFma3
        // (in TransKernelFma34.S)

        auto x = Max(a, MlasExpConstants.LowerRangeSumExp);
        // vbroadcastss ymm11,.LExpConstants_LowerRangeSumExp[rax]
        // vmaxps  ymm0,ymm11,ymm0                 # clamp lower bound

        // Range reduction of the input by computing "(2 ^ m) * exp(reduced)".
        auto biased = Fma(x, MlasExpConstants.Log2Reciprocal, MlasExpConstants.RoundingBias);
        // vbroadcastss ymm15,.LExpConstants_RoundingBias[rax]
        // vfmadd213ps ymm2,ymm0,ymm15             # (input / ln2) plus rounding bias

        auto m = biased - MlasExpConstants.RoundingBias;
        // vsubps  ymm1,ymm2,ymm15                 # round(input / ln2)

        x = Fma(m, MlasExpConstants.Log2High, x);
        // vbroadcastss ymm13,.LExpConstants_Log2High[rax]
        // vfmadd231ps ymm0,ymm1,ymm13             # range reduce: x -= (m * ln2_high)

        x = Fma(m, MlasExpConstants.Log2Low, x);
        // vbroadcastss ymm14,.LExpConstants_Log2Low[rax]
        // vfmadd231ps ymm0,ymm1,ymm14             # range reduce: x -= (m * ln2_low)

        // The following code is manually interleaved to produce better ILP for AVX:
#if 1 // interleaved version
        auto p = Scalar(MlasExpConstants.poly_0);
        p = Fma(p, x, MlasExpConstants.poly_1);
        // vbroadcastss ymm1,.LExpConstants_poly_0[rax]
        // vbroadcastss ymm13,.LExpConstants_poly_1[rax]
        // vfmadd213ps ymm1,ymm0,ymm13             # p = p * x + poly_1

        auto scale = ShiftLeft(FloatAsInt(biased), 23);
        // vpslld  ymm2,ymm2,23                    # shift m to exponent field

        p = Fma(p, x, MlasExpConstants.poly_2);
        // vbroadcastss ymm14,.LExpConstants_poly_2[rax]
        // vfmadd213ps ymm1,ymm0,ymm14             # p = p * x + poly_2

        scale = scale + MlasExpConstants.MaximumExponent;
        // vbroadcastss ymm15,.LExpConstants_MaximumExponent[rax]
        // vpaddd  ymm2,ymm2,ymm15                 # add exponent bias to scale

        p = Fma(p, x, MlasExpConstants.poly_3);
        // vbroadcastss ymm13,.LExpConstants_poly_3[rax]
        // vfmadd213ps ymm1,ymm0,ymm13             # p = p * x + poly_3

        p = Fma(p, x, MlasExpConstants.poly_4);
        // vbroadcastss ymm14,.LExpConstants_poly_4[rax]
        // vfmadd213ps ymm1,ymm0,ymm14             # p = p * x + poly_4

        p = Fma(p, x, MlasExpConstants.poly_56);
        p = Fma(p, x, MlasExpConstants.poly_56);
        // vbroadcastss ymm15,.LExpConstants_poly_56[rax]
        // vfmadd213ps ymm1,ymm0,ymm15             # p = p * x + poly_5
        // vfmadd213ps ymm1,ymm0,ymm15             # p = p * x + poly_6

        p = p * IntAsFloat(scale);
        // vmulps  ymm1,ymm1,ymm2 # p = p * scale
#else // non-interleaved, for documentation
        auto scale = ShiftLeft(FloatAsInt(biased), 23);
        scale = scale + MlasExpConstants.MaximumExponent;
        auto p = Scalar(MlasExpConstants.poly_0);
        p = Fma(p, x, MlasExpConstants.poly_1);
        p = Fma(p, x, MlasExpConstants.poly_2);
        p = Fma(p, x, MlasExpConstants.poly_3);
        p = Fma(p, x, MlasExpConstants.poly_4);
        p = Fma(p, x, MlasExpConstants.poly_56);
        p = Fma(p, x, MlasExpConstants.poly_56);
        p = p * IntAsFloat(scale);
#endif
        return p;
    }
} // namespace value
} // namespace accera
