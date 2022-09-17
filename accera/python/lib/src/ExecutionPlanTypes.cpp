////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraTypes.h"

#include <utilities/include/Exception.h>
#include <value/include/Plan.h>
#include <value/include/VectorizationInformation.h>

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

        py::enum_<value::CacheStrategy>(module, "_CacheStrategy", "An enumeration of cache strategy types")
            .value("BLOCKED", value::CacheStrategy::Blocked)
            .value("STRIPED", value::CacheStrategy::Striped);

        py::enum_<value::MemorySpace>(module, "_MemorySpace", "An enumeration of memory space types")
            .value("NONE", value::MemorySpace::None)
            .value("GLOBAL", value::MemorySpace::Global)
            .value("SHARED", value::MemorySpace::Shared)
            .value("PRIVATE", value::MemorySpace::Private)
            .value("TENSOR", value::MemorySpace::Tensor);

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
            .value("DEFAULT", value::ExecutionRuntime::DEFAULT)
            .value("VULKAN", value::ExecutionRuntime::VULKAN)
            .value("ROCM", value::ExecutionRuntime::ROCM)
            .value("CUDA", value::ExecutionRuntime::CUDA)
            .value("OPENMP", value::ExecutionRuntime::OPENMP)
            .value("NONE", value::ExecutionRuntime::NONE);

        py::enum_<value::GPU::BarrierScope>(module, "BarrierScope", "An enumeration of barrier scopes")
            .value("BLOCK", value::GPU::BarrierScope::Block)
            .value("WARP", value::GPU::BarrierScope::Warp)
            .value("THREADFENCE", value::GPU::BarrierScope::Threadfence);

        py::enum_<ir::value::MMAShape>(module, "_MMAShape", "Determines the underlying MMA op that will be used")
            .value("M64xN64xK1_B4", ir::value::MMAShape::M64xN64xK1_B4)
            .value("M64xN64xK1_B2", ir::value::MMAShape::M64xN64xK1_B2)
            .value("M32xN32xK2_B1", ir::value::MMAShape::M32xN32xK2_B1)
            .value("M16xN16xK4_B1", ir::value::MMAShape::M16xN16xK4_B1)
            .value("M64xN64xK2_B4", ir::value::MMAShape::M64xN64xK2_B4)
            .value("M64xN64xK2_B2", ir::value::MMAShape::M64xN64xK2_B2)
            .value("M32xN32xK4_B1", ir::value::MMAShape::M32xN32xK4_B1)
            .value("M16xN16xK8_B1", ir::value::MMAShape::M16xN16xK8_B1)
            .value("M64xN64xK4_B4", ir::value::MMAShape::M64xN64xK4_B4)
            .value("M64xN64xK4_B2", ir::value::MMAShape::M64xN64xK4_B2)
            .value("M32xN32xK8_B1", ir::value::MMAShape::M32xN32xK8_B1)
            .value("M16xN16xK16_B1", ir::value::MMAShape::M16xN16xK16_B1)
            .value("M32xN8xK16_B1", ir::value::MMAShape::M32xN8xK16_B1)
            .value("M8xN32xK16_B1", ir::value::MMAShape::M8xN32xK16_B1);

        py::enum_<ir::value::MMASchedulingPolicy>(module, "_MMASchedulingPolicy", "Used for configuring scheduling policy of MMA ops")
            .value("PASS_ORDER", ir::value::MMASchedulingPolicy::PassOrder)
            .value("BLOCK_ORDER", ir::value::MMASchedulingPolicy::BlockOrder);
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
                   const std::optional<util::DimensionOrder>& dimOrder,
                   bool thrifty,
                   bool doubleBuffer,
                   value::MemorySpace doubleBufferMemorySpace,
                   const std::optional<value::VectorizationInformation>& vectorizationInfo,
                   const std::optional<value::ValueType>& elementType,
                   value::CacheStrategy /*not hooked up*/) {
                    if (outermostIncludedSplitIndex.has_value())
                    {
                        value::ScalarIndex resolvedTriggerIndex = triggerIndex.has_value() ? *triggerIndex : *outermostIncludedSplitIndex;
                        if (memoryMap.has_value())
                        {
                            return plan.AddCache(target, *outermostIncludedSplitIndex, resolvedTriggerIndex, *memoryMap, elementType, thrifty, doubleBuffer, vectorizationInfo, indexing, allocation, memorySpace, doubleBufferMemorySpace);
                        }
                        else if (dimOrder.has_value())
                        {
                            return plan.AddCache(target, *outermostIncludedSplitIndex, resolvedTriggerIndex, *dimOrder, elementType, thrifty, doubleBuffer, vectorizationInfo, indexing, allocation, memorySpace, doubleBufferMemorySpace);
                        }
                        else
                        {
                            return plan.AddCache(target, *outermostIncludedSplitIndex, elementType, thrifty, doubleBuffer, vectorizationInfo, indexing, allocation, memorySpace, doubleBufferMemorySpace);
                        }
                    }
                    else
                    {
                        if (memoryMap.has_value())
                        {
                            return plan.AddCache(target, *maxElements, *memoryMap, elementType, thrifty, doubleBuffer, vectorizationInfo, indexing, allocation, memorySpace, doubleBufferMemorySpace);
                        }
                        else if (dimOrder.has_value())
                        {
                            return plan.AddCache(target, *maxElements, *dimOrder, elementType, thrifty, doubleBuffer, vectorizationInfo, indexing, allocation, memorySpace, doubleBufferMemorySpace);
                        }
                        else
                        {
                            return plan.AddCache(target, *maxElements, elementType, thrifty, doubleBuffer, vectorizationInfo, indexing, allocation, memorySpace, doubleBufferMemorySpace);
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
                "dim_order"_a,
                "thrifty"_a,
                "double_buffer"_a,
                "double_buffer_location"_a,
                "vectorization_info"_a,
                "element_type"_a,
                "strategy"_a)
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
                "add_cache",
                [](value::GPUPlan& plan,
                   const std::variant<value::ViewAdapter, value::Cache*>& target,
                   const std::optional<value::ScalarIndex>& outermostIncludedSplitIndex,
                   const std::optional<value::ScalarIndex>& triggerIndex,
                   const std::optional<int64_t>& maxElements,
                   value::CacheIndexing indexing,
                   value::CacheAllocation allocation,
                   value::MemorySpace memorySpace,
                   const std::optional<util::MemoryAffineCoefficients>& memoryMap,
                   const std::optional<util::DimensionOrder>& dimOrder,
                   bool thrifty,
                   bool doubleBuffer,
                   value::MemorySpace doubleBufferMemorySpace,
                   const std::optional<value::VectorizationInformation>& vectorizationInfo,
                   const std::optional<value::ValueType>& elementType,
                   value::CacheStrategy strategy) {
                    value::ScalarIndex resolvedTriggerIndex = triggerIndex.has_value() ? *triggerIndex : *outermostIncludedSplitIndex;
                    if (outermostIncludedSplitIndex.has_value())
                    {
                        return plan.AddCache(target, *outermostIncludedSplitIndex, resolvedTriggerIndex, *dimOrder, elementType, thrifty, doubleBuffer, strategy, vectorizationInfo, indexing, allocation, memorySpace, doubleBufferMemorySpace);
                    }
                    else if (maxElements.has_value())
                    {
                        // TODO : convert all GPUPlan::AddCache() impls to use manual caching rather than automatic, then plumb remaining arguments
                        return plan.AddCache(std::get<value::ViewAdapter>(target), *maxElements, strategy, memorySpace);
                    }
                    else
                    {
                        // TODO : reach parity with GPUPlan::AddCache() and Plan::AddCache() functions
                        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
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
                "dim_order"_a,
                "thrifty"_a,
                "double_buffer"_a,
                "double_buffer_location"_a,
                "vectorization_info"_a,
                "element_type"_a,
                "strategy"_a)
            .def("tensorize", &value::GPUPlan::Tensorize, "indices"_a, "dims"_a, "numTotalPasses"_a, "useStaticOffsets"_a, "numFusedPasses"_a, "schedulingPolicy"_a, "_useRocWMMA"_a)
            .def("_map_index_to_processor", &value::GPUPlan::MapIndicesToProcessor, "indices"_a, "proc"_a);
    }

} // namespace

void DefineExecutionPlanTypes(py::module& module)
{
    DefineExecutionPlanEnums(module);
    DefineExecutionPlanStructs(module);
    DefineExecutionPlanClasses(module);
}
} // namespace accera::python::lang
