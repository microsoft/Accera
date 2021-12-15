////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Emittable.h"

namespace accera
{
namespace value
{

    Emittable::Emittable() = default;

    Emittable::Emittable(void* data) :
        _data(data) {}

} // namespace value
} // namespace accera
