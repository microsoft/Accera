////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraTypes.h"

#include <value/include/Plan.h>

#include <ir/include/value/ValueEnums.h>

#include <variant>

namespace py = pybind11;
namespace value = accera::value;
namespace ir = accera::ir;
namespace util = accera::utilities;

using namespace pybind11::literals;
namespace accera::python::lang
{
namespace
{
    void DefineExecutionPlanEnums(py::module& module)
    {
        py::enum_<value::CacheIndexing>(module, "CacheIndexing", "An enumeration of cache index mapping types")
            .value("GLOBAL_TO_PHYSICAL", value::CacheIndexing::GlobalToPhysical)
            .value("LOGICAL_TO_PHYSICAL", value::CacheIndexing::LogicalToPhysical)
            .value("LOGICAL_TO_GLOBAL", value::CacheIndexing::LogicalToGlobal)
            .value("NONE", value::CacheIndexing::None);

        py::enum_<value::CacheAllocation>(module, "_CacheAllocation", "An enumeration of cache allocation types")
            .value("AUTO", value::CacheAllocation::Automatic)
            .value("NONE", value::CacheAllocation::None);

        py::enum_<value::MemorySpace>(module, "_MemorySpace", "An enumeration of memory space types")
            .value("NONE", value::MemorySpace::None)
            .value("GLOBAL", value::MemorySpace::Global)
            .value("SHARED", value::MemorySpace::Shared)
            .value("LOCAL", value::MemorySpace::Local);

        py::enum_<ir::value::Processor>(module, "Processor", "An enumeration of processors for loop index mapping")
            .value("BLOCK_X", ir::value::Processor::BlockX)
            .value("BLOCK_Y", ir::value::Processor::BlockY)
            .value("BLOCK_Z", ir::value::Processor::BlockZ)
            .value("THREAD_X", ir::value::Processor::ThreadX)
            .value("THREAD_Y", ir::value::Processor::ThreadY)
            .value("THREAD_Z", ir::value::Processor::ThreadZ)
            .value("SEQUENTIAL", ir::value::Processor::Sequential)
            .export_values();

        py::enum_<value::ParallelizationPolicy>(module, "_ParallelizationPolicy", "Used for configuring the thread scheduling policy")
            .value("STATIC", value::ParallelizationPolicy::Static)
            .value("DYNAMIC", value::ParallelizationPolicy::Dynamic);

        py::enum_<value::ExecutionRuntime>(module, "_ExecutionRuntime", "Used for specifying the execution runtime of the module")
            .value("DEFAULT", value::ExecutionRuntime::Default)
            .value("VULKAN", value::ExecutionRuntime::Vulkan)
            .value("ROCM", value::ExecutionRuntime::Rocm)
            .value("CUDA", value::ExecutionRuntime::CUDA);
    }

    void DefineExecutionPlanStructs(py::module& module)
    {
        py::class_<value::VectorizationInformation>(module, "_VectorizationInfo", "Used for configuring loop vectorization")
            .def(py::init<int, int, bool>(), "vector_bytes"_a = 0, "vector_units"_a = 0, "unroll_only"_a = false)
            .def_readwrite("vector_bytes", &value::VectorizationInformation::vectorBytes)
            .def_readwrite("vector_units", &value::VectorizationInformation::vectorUnitCount)
            .def_readwrite("unroll_only", &value::VectorizationInformation::unrollOnly);

        py::class_<value::targets::Dim3>(module, "_Dim3", "Used for configuring the x, y, and z indices for a GPU processor")
            .def(py::init<int, int, int>(), "x"_a = 0, "y"_a = 0, "z"_a = 0)
            .def_readwrite("x", &value::targets::Dim3::x)
            .def_readwrite("y", &value::targets::Dim3::y)
            .def_readwrite("z", &value::targets::Dim3::z);

        py::class_<value::targets::GPU>(module, "_GPU", "The GPU execution options")
            .def(py::init<value::targets::Dim3, value::targets::Dim3>(), "grid"_a, "block"_a)
            .def_readwrite("grid", &value::targets::GPU::grid)
            .def_readwrite("block", &value::targets::GPU::block);

        py::class_<util::MemoryAffineCoefficients>(module, "_MemoryAffineCoefficients", "Used for mapping Array or Cache dimensions to memory locations")
            .def(py::init<std::vector<int64_t>, int64_t>(), "coefficients"_a, "offset"_a = 0);

        py::class_<util::DimensionOrder>(module, "_DimensionOrder", "Describes the physical order of Array or Cache dimensions as a permutation of the logical order")
            .def(py::init<std::vector<int64_t>>(), "order"_a);
    }

    void DefineExecutionPlanClasses(py::module& module)
    {
        py::class_<value::Cache>(module, "_Cache");

        py::class_<value::Plan>(module, "_ExecutionPlan")
            .def(py::init([](value::Plan& plan) {
                     return value::Plan(std::move(plan));
                 }),
                 py::return_value_policy::move)
            .def(
                "add_cache",
                [](value::Plan& plan,
                   const std::variant<value::ViewAdapter, value::Cache*>& target,
                   const std::optional<value::ScalarIndex>& outermostIncludedSplitIndex,
                   const std::optional<value::ScalarIndex>& triggerIndex,
                   const std::optional<int64_t>& maxElements,
                   value::CacheIndexing indexing,
                   value::CacheAllocation allocation,
                   value::MemorySpace memorySpace,
                   const std::optional<util::MemoryAffineCoefficients>& memoryMap,
                   const std::optional<util::DimensionOrder>& dimOrder) {
                    if (outermostIncludedSplitIndex.has_value())
                    {
                        value::ScalarIndex resolvedTriggerIndex = triggerIndex.has_value() ? *triggerIndex : *outermostIncludedSplitIndex;
                        if (memoryMap.has_value())
                        {
                            return plan.AddCache(target, *outermostIncludedSplitIndex, resolvedTriggerIndex, *memoryMap, indexing, allocation, memorySpace);
                        }
                        else if (dimOrder.has_value())
                        {
                            return plan.AddCache(target, *outermostIncludedSplitIndex, resolvedTriggerIndex, *dimOrder, indexing, allocation, memorySpace);
                        }
                        else
                        {
                            return plan.AddCache(target, *outermostIncludedSplitIndex, indexing, allocation, memorySpace);
                        }
                    }
                    else
                    {
                        if (memoryMap.has_value())
                        {
                            return plan.AddCache(target, *maxElements, *memoryMap, indexing, allocation, memorySpace);
                        }
                        else if (dimOrder.has_value())
                        {
                            return plan.AddCache(target, *maxElements, *dimOrder, indexing, allocation, memorySpace);
                        }
                        else
                        {
                            return plan.AddCache(target, *maxElements, indexing, allocation, memorySpace);
                        }
                    }
                },
                "target"_a,
                "index"_a,
                "trigger_index"_a,
                "max_elements"_a,
                "indexing"_a,
                "allocation"_a,
                "location"_a,
                "memory_map"_a,
                "dim_order"_a)
            .def("emit_runtime_init_packing", py::overload_cast<value::ViewAdapter, const std::string&, const std::string&, value::CacheIndexing>(&value::Plan::EmitRuntimeInitPacking), "target"_a, "packing_func_name"_a, "packed_buf_size_func_name"_a, "indexing"_a = value::CacheIndexing::GlobalToPhysical)
            .def("pack_and_embed_buffer", py::overload_cast<value::ViewAdapter, value::ViewAdapter, const std::string&, const std::string&, value::CacheIndexing>(&value::Plan::PackAndEmbedBuffer), "target"_a, "constant_data_buffer"_a, "wrapper_fn_name"_a, "packed_buffer_name"_a, "indexing"_a = value::CacheIndexing::GlobalToPhysical)
            .def("vectorize", &value::Plan::Vectorize, "i"_a, "vectorization_info"_a)
            .def("parallelize", &value::Plan::Parallelize, "indices"_a, "num_threads"_a, "policy"_a);

        py::class_<value::GPUPlan>(module, "_GPUExecutionPlan")
            .def(py::init([](value::GPUPlan& plan) {
                     return value::GPUPlan(std::move(plan));
                 }),
                 py::return_value_policy::move)
            .def(
                "add_cache", [](value::GPUPlan& plan,
                   const std::variant<value::ViewAdapter, value::Cache*>& target,
                   const std::optional<value::ScalarIndex>& outermostIncludedSplitIndex,
                   const std::optional<value::ScalarIndex>& triggerIndex,
                   const std::optional<int64_t>& maxElements,
                   value::CacheIndexing indexing,
                   value::CacheAllocation allocation,
                   value::MemorySpace memorySpace,
                   const std::optional<util::MemoryAffineCoefficients>& memoryMap,
                   const std::optional<util::DimensionOrder>& dimOrder) {
                        value::ScalarIndex resolvedTriggerIndex = triggerIndex.has_value() ? *triggerIndex : *outermostIncludedSplitIndex;
                        return plan.AddCache(target, *outermostIncludedSplitIndex, resolvedTriggerIndex, *dimOrder, indexing, allocation, memorySpace);
                        //return outermostIncludedSplitIndex.has_value() ? plan.AddCache(target, *outermostIncludedSplitIndex, memorySpace) : plan.AddCache(target, *maxElements, memorySpace);
                },
                "target"_a,
                "index"_a,
                "trigger_index"_a,
                "max_elements"_a,
                "indexing"_a,
                "allocation"_a,
                "location"_a,
                "memory_map"_a,
                "dim_order"_a)
            .def("tensorize", &value::GPUPlan::Tensorize, "indices"_a, "dims"_a)
            .def("map_index_to_processor", &value::GPUPlan::MapIndexToProcessor, "index"_a, "proc"_a);
    }

} // namespace

void DefineExecutionPlanTypes(py::module& module)
{
    DefineExecutionPlanEnums(module);
    DefineExecutionPlanStructs(module);
    DefineExecutionPlanClasses(module);
}
} // namespace accera::python::lang
