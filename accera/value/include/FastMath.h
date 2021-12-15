////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace accera
{
namespace value
{
    class Scalar;

    Scalar FastExp(Scalar s);
    Scalar FastExpMlas(Scalar s);

} // namespace value
} // namespace accera
