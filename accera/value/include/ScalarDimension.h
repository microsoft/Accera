////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Scalar.h"

namespace accera
{
namespace value
{
    class ScalarDimension : public Scalar
    {
    public:
        ScalarDimension(Role role = Role::Input);
        ScalarDimension(const std::string& name, Role role = Role::Input);
        ScalarDimension(Value value, const std::string& name = "", Role role = Role::Input);

        virtual void SetValue(Value value) final;
    };
} // namespace value
} // namespace accera
