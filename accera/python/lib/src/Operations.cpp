////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraTypes.h"
#include <value/include/Debugging.h>

namespace py = pybind11;
namespace util = accera::utilities;

using namespace pybind11::literals;

namespace accera::python::lang
{
void DefineOperations(py::module& module)
{
    module.def(
              "Allocate",
              [](value::ValueType type, util::MemoryLayout layout, size_t alignment, value::AllocateFlags flags, const std::vector<value::ScalarDimension>& runtimeSizes) {
                  return value::Allocate(type, layout, alignment, flags, runtimeSizes);
              },
              "type"_a,
              "layout"_a = util::ScalarLayout,
              "alignment"_a = 0,
              "flags"_a = value::AllocateFlags::None,
              "runtimeSizes"_a = std::vector<value::ScalarDimension>{})
        .def(
            "StaticAllocate",
            [](std::string name, value::ValueType type, util::MemoryLayout layout, value::AllocateFlags flags) {
                return value::StaticAllocate(name, type, layout, flags);
            },
            "name"_a,
            "type"_a,
            "layout"_a,
            "flags"_a = value::AllocateFlags::None)
        .def(
            "GlobalAllocate",
            [](std::string name, value::ValueType type, util::MemoryLayout layout, value::AllocateFlags flags) {
                return value::GlobalAllocate(name, type, layout, flags);
            },
            "name"_a,
            "type"_a,
            "layout"_a,
            "flags"_a = value::AllocateFlags::None)
        .def("If", py::overload_cast<value::Scalar, std::function<void()>>(&value::If))
        .def("ForRange", py::overload_cast<value::Scalar, value::Scalar, value::Scalar, std::function<void(value::Scalar)>>(&value::ForRange))
        .def("ForRanges", py::overload_cast<std::vector<value::Scalar>, std::function<void(std::vector<value::Scalar>)>>(&value::ForRanges))
        .def("Print", py::overload_cast<value::ViewAdapter, bool>(&value::Print), "value"_a, "to_stderr"_a = false)
        .def("Print", py::overload_cast<const std::string&, bool>(&value::Print), "message"_a, "to_stderr"_a = false)
        .def("PrintRawMemory", &value::PrintRawMemory)
        .def("AsFullView", &value::AsFullView<value::Vector>)
        .def("AsFullView", &value::AsFullView<value::Matrix>)
        .def("AsFullView", &value::AsFullView<value::Tensor>)
        .def("AsFullView", &value::AsFullView<value::Array>)
        .def(
            "ReduceN",
            [](value::Scalar start, value::Scalar end, value::Scalar step, value::ViewAdapter init, std::function<value::ViewAdapter(value::Scalar, value::ViewAdapter)> forFn) {
                return value::ReduceN(start, end, step, init, std::move(forFn));
            },
            "start"_a,
            "end"_a,
            "step"_a,
            "init"_a,
            "fn"_a)
        .def(
            "Reduce",
            [](value::Array a,
               value::ViewAdapter init,
               std::function<value::ViewAdapter(value::Scalar, value::ViewAdapter)> reduceFn) {
                return value::Reduce(a, init, std::move(reduceFn));
            },
            "data"_a,
            "init"_a,
            "fn"_a)
        .def(
            "MapReduce",
            [](value::Array a,
               value::ViewAdapter init,
               std::function<value::ViewAdapter(value::Scalar)> mapFn,
               std::function<value::ViewAdapter(value::Scalar, value::ViewAdapter)> reduceFn) {
                return value::MapReduce(a, init, std::move(mapFn), std::move(reduceFn));
            },
            "data"_a,
            "init"_a,
            "map_fn"_a,
            "reduce_fn"_a)
        .def("CheckAllClose", &value::CheckAllClose)
        .def("Return", py::overload_cast<value::ViewAdapter>(&value::Return), "view"_a = value::ViewAdapter{})
        .def("GetTime", &value::GetTime);

    auto getFromGPUIndex = [](value::GPUIndex idx, std::string pos) -> value::Scalar {
        if (pos == "x")
        {
            return idx.X();
        }
        else if (pos == "y")
        {
            return idx.Y();
        }
        else
        {
            return idx.Z();
        }
    };

    auto gpu_mod = module.def_submodule("_gpu");
    gpu_mod.def("BlockDim",
                [=](std::string pos) {
                    return getFromGPUIndex(value::GPU::BlockDim(), pos);
                })
        .def("BlockId",
             [=](std::string pos) {
                 return getFromGPUIndex(value::GPU::BlockId(), pos);
             })
        .def("GridDim",
             [=](std::string pos) {
                 return getFromGPUIndex(value::GPU::GridDim(), pos);
             })
        .def("ThreadId",
             [=](std::string pos) {
                 return getFromGPUIndex(value::GPU::ThreadId(), pos);
             })
        .def(
            "Barrier", [=](value::GPU::BarrierScope scope) {
                return value::GPU::Barrier(scope);
            },
            "scope"_a = value::GPU::BarrierScope::Block);
}
} // namespace accera::python::lang
