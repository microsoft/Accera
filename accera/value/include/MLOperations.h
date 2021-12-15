////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Array.h"

#include "Array.h"
#include "Scalar.h"

namespace accera
{
namespace value
{
    void SoftmaxifyRows(Array m);
    void SoftmaxifyRowsVectorized(Array m);

    void LayerNormalize(Array m, Array alpha, Array beta);
    void LayerNormalizeFused(Array m, Array alpha, Array beta, Array residual);
    void LayerNormalizeVectorized(Array m, Array alpha, Array beta);
    void LayerNormalizeVectorizedFused(Array m, Array alpha, Array beta, Array residual);

    void ReLU(Array m);

    void Feedforward(Array attn, Array Wff1, Array Wff2, Array ffTemp, Array output);
    void FusedFeedforward(Array attn, Array Wff1, Array Wff2, Array ffTemp, Array output);
} // namespace value
} // namespace accera
