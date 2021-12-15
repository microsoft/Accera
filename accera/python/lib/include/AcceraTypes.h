////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <value/include/EmitterContext.h>
#include <value/include/MLIREmitterContext.h>
#include <value/include/Value.h>

namespace accera
{
namespace python
{
namespace lang
{
    void DefineContainerTypes(pybind11::module& module, pybind11::module& subModule);
    void DefineNestTypes(pybind11::module& module);
    void DefineSchedulingTypes(pybind11::module& module);
    void DefineExecutionPlanTypes(pybind11::module& module);
    void DefinePackagingTypes(pybind11::module& module, pybind11::module& subModule);
    void DefineOperations(pybind11::module& module);

} // namespace lang
} // namespace python
} // namespace accera
