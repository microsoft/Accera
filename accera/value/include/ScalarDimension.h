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

        virtual void SetName(const std::string& name) final;
        virtual std::string GetName() const final;

        virtual void SetValue(Value value) final;

    private:
        std::string _name;
    };
} // namespace value
} // namespace accera
