////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <value/include/Scalar.h>

namespace accera
{
value::Scalar Scalar_test1();
value::Scalar Scalar_test2();
value::Scalar ScalarRefTest();
value::Scalar ScalarRefRefTest();
value::Scalar ScalarRefRefRefTest();
value::Scalar RefScalarRefTest();
value::Scalar RefScalarRefCtorsTest();
value::Scalar RefScalarRefRefTest();
value::Scalar RefScalarRefRefRefTest();
value::Scalar SequenceLogicalAndTest();
value::Scalar SequenceLogicalAndTestWithCopy();
} // namespace accera
