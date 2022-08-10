////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraTypes.h"

#include <value/include/Kernel.h>
#include <value/include/KernelPredicate.h>
#include <value/include/Nest.h>
#include <value/include/Schedule.h>

namespace py = pybind11;
namespace value = accera::value;
namespace util = accera::utilities;

using namespace pybind11::literals;

namespace accera::python::lang
{
namespace
{
void DefineIterationLogicTypes(py::module& module)
{
    py::class_<value::EmitterContext::IfContext>(module, "_If")
        .def(py::init([](value::Scalar test, std::function<void()> fn) {
                return std::make_unique<value::EmitterContext::IfContext>(value::If(test, fn));
            }),
            "test"_a,
            "fn"_a)
        .def(
            "ElseIf", [](value::EmitterContext::IfContext& ifC, value::Scalar test, std::function<void()> fn) -> value::EmitterContext::IfContext& {
                return ifC.ElseIf(test, fn);
            },
            "test"_a,
            "fn"_a,
            py::return_value_policy::reference_internal)
        .def(
            "Else", [](value::EmitterContext::IfContext& ifC, std::function<void()> fn) {
                ifC.Else(fn);
            },
            "fn"_a);

    py::class_<value::Range>(module, "Range", "A class representing the half-open interval `[begin, end)`, with an increment between points of _increment.")
        .def(py::init<int64_t, int64_t, int64_t>(), "begin"_a, "end"_a, "increment"_a = 1);

    py::class_<value::Kernel>(module, "_Logic", "Represents a logic function")
        .def(py::init<std::string, std::function<void()>>(), "id"_a, "logic_fn"_a)
        .def("_dump", &value::Kernel::dump)
        .def("_get_indices", &value::Kernel::GetIndices);

    #define ADD_METHOD(NAME) def( \
        #NAME, [](value::ScalarIndex idx) { return std::make_unique<value::KernelPredicate>(value::NAME(idx)); }, "Constructs a predicate on a scalar index")

    py::class_<value::KernelPredicate, std::unique_ptr<value::KernelPredicate>>(module, "LogicPredicate")
        .ADD_METHOD(First)
        .ADD_METHOD(Last)
        .ADD_METHOD(EndBoundary)
        .ADD_METHOD(Before)
        .ADD_METHOD(After)
        .ADD_METHOD(IsDefined)
        .def(
            "__and__", [](const value::KernelPredicate& p1, const value::KernelPredicate& p2) { return std::make_unique<value::KernelPredicate>(p1 && p2); }, "Constructs a predicate for (a and b)")
        .def(
            "__or__", [](const value::KernelPredicate& p1, const value::KernelPredicate& p2) { return std::make_unique<value::KernelPredicate>(p1 || p2); }, "Constructs a predicate for (a or b)");
    #undef ADD_METHOD

    // helper to cast an integer value into a scalar index for conditionals
    module.def("as_index", [](int64_t value) {
        return value::Cast(value, value::ValueType::Index);
    }, "value"_a, "Converts an integer value into a Scalar index");
}

void DefineNestClass(py::module& module)
{
    py::class_<value::Nest>(module, "_Nest")
        .def(py::init([](value::Nest& nest) {
                 return value::Nest(std::move(nest));
             }),
             py::return_value_policy::move)
        .def(py::init([](const std::vector<int64_t>& sizes, const std::vector<value::ScalarDimension>& runtimeSizes) {
                 return std::make_unique<value::Nest>(util::MemoryShape{ sizes }, runtimeSizes);
             }),
             "shape"_a,
             "runtime_sizes"_a,
             "Constructor that creates a nest from a list representing the iteration space")
        .def(py::init<const std::vector<value::Range>&, const std::vector<value::ScalarDimension>&>(),
             "shape"_a,
             "runtime_sizes"_a,
             "Constructor that creates a nest from a list of ranges representing the iteration space")
        .def(
            "get_indices", [](value::Nest& nest) { return nest.GetIndices(); }, "Returns the indices for this nest, starting from the outermost index")

        // TODO: Nest's cpp impl call's Schedule's cpp impl, so no reason to expose multiple API points
        // .def("add_iteration_logic", &value::Nest::Set, "logic"_a, "Sets the default logic function to be run in the innermost loop")

        .def("create_schedule", &value::Nest::CreateSchedule, "Creates a schedule to run this nest")
        .def("_dump", &value::Nest::dump);
}
} // namespace

void DefineNestTypes(py::module& module)
{
    DefineIterationLogicTypes(module);
    DefineNestClass(module);
}
} // namespace accera::python::lang
