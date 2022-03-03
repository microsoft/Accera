////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraTypes.h"

#include <value/include/Kernel.h>
#include <value/include/KernelPredicate.h>
#include <value/include/Plan.h>
#include <value/include/Schedule.h>

namespace py = pybind11;
namespace value = accera::value;

using namespace pybind11::literals;
namespace accera::python::lang
{
namespace
{
void DefineScheduleClass(py::module& module)
{
    py::class_<value::Schedule>(module, "_Schedule")
        .def(py::init([](value::Schedule& other) {
                 return value::Schedule(std::move(other));
             }),
             py::return_value_policy::move)
        .def(
            "split", [](value::Schedule& sched, value::ScalarIndex& i, int factor) {
                auto ret = sched.Split(i, factor);
                using std::swap;
                swap(i, ret.first);
                return ret.second;
            },
            "i"_a,
            "factor"_a)
        .def("unroll", py::overload_cast<value::ScalarIndex, std::optional<uint64_t>>(&value::Schedule::Unroll), "i"_a, "size"_a = std::nullopt)
        .def("interleaved_unroll", py::overload_cast<value::ScalarIndex, uint64_t>(&value::Schedule::InterleavedUnroll), "i"_a, "factor"_a,
             R"pbdoc(
Partially unroll the loop along a dimension

Args:
    i: The dimension to unroll
    factor: The number of times to unroll the loop
)pbdoc")
        .def("set_order", py::overload_cast<std::vector<value::ScalarIndex>>(&value::Schedule::SetOrder), "order"_a)
        .def(
            "add_kernel", [](value::Schedule& sched, const value::Kernel& krnl, value::KernelPredicate* pred, value::KernelPredicate* placement) {
                if (placement)
                {
                    return sched.AddKernel(krnl, *pred, *placement);
                }
                else if (pred)
                {
                    return sched.AddKernel(krnl, *pred);
                }
                else
                {
                    return sched.AddKernel(krnl);
                }
            },
            "logic"_a,
            "predicate"_a.none() = py::none(),
            "placement"_a.none() = py::none())
        .def("fuse", py::overload_cast<std::vector<value::Schedule>&, const std::vector<std::vector<value::ScalarIndex>>&>(&value::Schedule::Fuse), "others"_a, "index_map"_a,
             R"pbdoc(
Fuse other schedules into this one, destroying the other ones.

Args:
    others: The other schedules to be fused
    index_map: A list of index list, indicating which indices are being fused together.

Returns:
    The 'fusing' index, which will be the outermost loop index
)pbdoc")
        .def("pad", py::overload_cast<value::ScalarIndex, int, bool>(&value::Schedule::Pad), "i"_a, "size"_a, "pad_front"_a)
        .def("skew", py::overload_cast<value::ScalarIndex, value::ScalarIndex>(&value::Schedule::Skew), "i"_a, "reference_index"_a)
        .def("create_plan", &value::Schedule::CreatePlan, "Creates a plan for the host")
        .def("create_gpu_plan", &value::Schedule::CreateGPUPlan, "Creates a plan for the GPU", "gpu_options"_a, "runtime"_a = value::ExecutionRuntime::Default)
        .def(
            "get_indices", [](value::Schedule& sched) { return sched.GetIndices(); }, "Returns the indices for this schedule, starting from the outermost index");
}
} // namespace

void DefineSchedulingTypes(py::module& module)
{
    DefineScheduleClass(module);
}
} // namespace accera::python::lang
