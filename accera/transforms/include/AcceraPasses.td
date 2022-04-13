////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ACCERA_CONVERSION_PASSES
#define ACCERA_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

class accModulePass<string passArg> : PassBase<passArg, "::accera::transforms::ModulePass">;

//===----------------------------------------------------------------------===//
// ExecutionPlanCacheRegion
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanCacheRegion : Pass<"convert-exec-plan-cache-region"> {
  let summary = "Lower ExecPlan cache region and mapping ops";
  let constructor = "accera::transforms::executionPlan::createExecutionPlanCacheRegionLoweringPass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanVectorization
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanVectorization : Pass<"convert-exec-plan-vectorize"> {
  let summary = "Match and vectorize ExecPlan ops";
  let constructor = "accera::transforms::executionPlan::createExecutionPlanVectorizationPass()";
  let options = [
    Option<"printVecOpDetails", "print-vec-details", "bool", /*default=*/"false",
           "Print details about op vectorization">
  ];
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanParallelization
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanParallelization : Pass<"convert-exec-plan-parallelize"> {
  let summary = "Match and parallelize ExecPlan ops";
  let constructor = "accera::transforms::executionPlan::createExecutionPlanParallelizationPass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanTensorization
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanTensorization : Pass<"convert-exec-plan-tensorize"> {
  let summary = "Match and tensorize ExecPlan ops";
  let constructor = "accera::transforms::executionPlan::createExecutionPlanTensorizationPass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanMakeCache
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanMakeCache : FunctionPass<"convert-exec-make-cache"> {
  let summary = "Lower ExecPlan make cache op";
  let constructor = "accera::transforms::executionPlan::createExecutionPlanMakeCachePass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanCopyReduce
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanCopyReduce : Pass<"convert-exec-copy-reduce"> {
  let summary = "Lower ExecPlan cache copy and reduce ops";
  let constructor = "accera::transforms::executionPlan::createExecutionPlanCopyReducePass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "accera::ir::loopnest::LoopNestDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanScaleHoisting
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanScaleHoisting : FunctionPass<"hoist-cache-scaling"> {
  let summary = "Hoist scale ops in kernels to cache reduce ops";
  let constructor = "accera::transforms::executionPlan::createExecutionPlanScaleHoistingPass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "accera::ir::loopnest::LoopNestDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// HandleOutOfBoundsAccess
//===----------------------------------------------------------------------===//

def HandleOutOfBoundsAccess : FunctionPass<"handle-out-of-bounds-access"> {
  let summary = "Detect potential out-of-bounds affine loads and replace them with a conditional access and default value";
  let constructor = "accera::transforms::executionPlan::createOutOfBoundsAccessHandlingPass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "accera::ir::loopnest::LoopNestDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ScheduledOperations
//===----------------------------------------------------------------------===//

def ConvertScheduledOperations : FunctionPass<"convert-loopnest-scheduledops"> {
  let summary = "Lower LoopNest ops";
  let constructor = "accera::transforms::loopnest::createScheduledOperationsPass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ScheduleToValue
//===----------------------------------------------------------------------===//

def ConvertScheduleToValue : FunctionPass<"convert-loopnest-to-value"> {
  let summary = "Lower LoopNest ops to Value dialect ops";
  let constructor = "accera::transforms::loopnest::createScheduleToValuePass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LoopNestOpt
//===----------------------------------------------------------------------===//

def ConvertLoopNestOpt : Pass<"loopnest-opt", "accera::ir::value::ValueFuncOp"> {
  let summary = "Optimize LoopNest ops";
  let constructor = "accera::transforms::loopnest::createLoopNestOptPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ValueSimplify
//===----------------------------------------------------------------------===//

def ConvertValueSimplify : Pass<"simplify-value"> {
  let summary = "Simplify Value ops";
  let constructor = "accera::transforms::value::createValueSimplifyPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::math::MathDialect"
  ];
}

//===----------------------------------------------------------------------===//
// RangeValueOptimization
//===----------------------------------------------------------------------===//

def ConvertRangeValueOptimize : Pass<"optimize-range-value"> {
  let summary = "Performing optimization using Range Value analysis";
  let constructor = "accera::transforms::value::createRangeValueOptimizePass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::math::MathDialect",
    "mlir::scf::SCFDialect",
    "mlir::gpu::GPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// BarrierOpt
//===----------------------------------------------------------------------===//

def BarrierOpt : Pass<"optimize-barriers"> {
  let summary = "Optimize Barrier ops";
  let constructor = "accera::transforms::value::createBarrierOptPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"writeBarrierGraph", "barrier-opt-dot", "bool", /*default=*/"false",
           "Write out barrier analysis graph in graphviz format">,
    Option<"barrierGraphFilename", "barrier-opt-dot-filename", "std::string",
           /*default=*/"\"\"",
           "Filename for barrier analysis graph in graphviz format">
  ];
}

//===----------------------------------------------------------------------===//
// ValueToLLVM
//===----------------------------------------------------------------------===//

def ConvertValueToLLVM : accModulePass<"value-to-llvm"> {
  let summary = "Lower Value ops to LLVM";
  let constructor = "accera::transforms::value::createValueToLLVMPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::LLVM::LLVMDialect"
  ];
  // Match std-to-llvm options so we can pass through arguments
  let options = [
    // TODO : evaluate if we want to converge with MLIR and default this to false once the sample emitters catch up
    Option<"useAlignedAlloc", "use-aligned-alloc", "bool", /*default=*/"true",// "false",
           "Use aligned_alloc in place of malloc for heap allocations">,
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"emitCWrappers", "emit-c-wrappers", "bool", /*default=*/"false",
           "Emit wrappers for C-compatible pointer-to-struct memref "
           "descriptors">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">
  ];
}

//===----------------------------------------------------------------------===//
// ValueToStd
//===----------------------------------------------------------------------===//

def ConvertValueToStd : accModulePass<"convert-value-to-std"> {
  let summary = "Lower Value ops to standard MLIR dialects";
  let constructor = "accera::transforms::value::createValueToStdPass()";
  let options = [
    Option<"enableProfiling", "enable-profiling", "bool", /*default=*/"false",
           "Enable profiling">
  ];
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ValueFuncToTarget
//===----------------------------------------------------------------------===//

def ValueFuncToTarget : Pass<"value-func-to-target", "::mlir::ModuleOp"> {
  let constructor = "accera::transforms::value::createValueFuncToTargetPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LoopNestToValueFunc
//===----------------------------------------------------------------------===//

def LoopNestToValueFunc : Pass<"loopnest-to-value-func", "accera::ir::value::ValueFuncOp"> {
  let summary = "Lower loop nests";
  let description = [{
      This pass lowers loop nests into value functions.
    }];
  let constructor = "accera::transforms::loopnest::createLoopNestToValueFuncPass()";
  let options = [
    Option<"printVecOpDetails", "print-vec-details", "bool", /*default=*/"false",
           "Print details about op vectorization">,
    Option<"printLoops", "print-loops", "bool", /*default=*/"false",
           "Print loop structure">
  ];
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::gpu::GPUDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertVulkanLaunchFuncToVulkanCallsWithTiming
//===----------------------------------------------------------------------===//

def ConvertVulkanLaunchFuncToVulkanCallsWithTiming : accModulePass<"launch-func-to-vulkan-with-timing"> {
  let summary = "Convert vulkanLaunch external call to Vulkan runtime external "
                "calls with custom timing configuration runtime calls";
  let constructor = "accera::transforms::vulkan::createConvertVulkanLaunchFuncToVulkanCallsWithTimingPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
  let options = [
    Option<"printTimings", "print-timings", "bool", /*default=*/"true",
           "Print device timestamp averages">,
    Option<"warmupCount", "warmup-count", "unsigned",
           /*default=*/"0",
           "Number of warmup rounds to run when timing">,
    Option<"runCount", "run-count", "unsigned",
           /*default=*/"1",
           "Number of timed rounds to run when timing">
  ];
}

//===----------------------------------------------------------------------===//
// EmitVulkanWrappers
//===----------------------------------------------------------------------===//

def EmitVulkanWrapper : accModulePass<"emit-vulkan-wrappers"> {
  let summary = "Mark vulkanLaunch calls to emit C wrappers";
  let constructor = "accera::transforms::vulkan::createEmitVulkanWrapperPass()";
  let dependentDialects = ["mlir::StandardOpsDialect"];
}

//===----------------------------------------------------------------------===//
// FunctionPointerResolution
//===----------------------------------------------------------------------===//

def FunctionPointerResolution : accModulePass<"resolve-function-pointers"> {
  let summary = "Resolve Accera temporary functions to their fully-realized counterparts";
  let constructor = "accera::transforms::value::createFunctionPointerResolutionPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// SerializeToHSACO
//===----------------------------------------------------------------------===//

def SerializeToHSACO : Pass<"serialize-to-hsaco", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Serializes the GPU kernel to HSACO object (WIP)";
  let constructor = "accera::transforms::createSerializeToHSACOPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::ROCDL::ROCDLDialect"
  ];
  let options = [
    Option<"chip", "chip", "std::string",
          //  TODO: Should this default to something else?
           "\"gfx908\"",
           "The GPU target architecture.">
  ];
}

//===----------------------------------------------------------------------===//
// AcceraToSPIRV
//===----------------------------------------------------------------------===//

def ConvertAcceraToSPIRV : Pass<"convert-accera-to-spirv", "::mlir::ModuleOp"> {
  let summary = "Convert Accera dialects to SPIR-V dialect";
  let constructor = "accera::transforms::createAcceraToSPIRVPass()";
  let dependentDialects = ["mlir::spirv::SPIRVDialect", "mlir::gpu::GPUDialect"];
}

//===----------------------------------------------------------------------===//
// AcceraToRocm
//===----------------------------------------------------------------------===//

def ConvertAcceraToROCDL : Pass<"convert-accera-to-rocdl", "::mlir::ModuleOp"> {
  let summary = "Convert Accera dialects to ROCDL dialect";
  let constructor = "accera::transforms::createAcceraToROCDLPass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::gpu::GPUDialect",
    "mlir::ROCDL::ROCDLDialect"
  ];
}


//===----------------------------------------------------------------------===//
// AcceraToNVVM
//===----------------------------------------------------------------------===//

def ConvertAcceraToNVVM : Pass<"convert-accera-to-nvvm", "::mlir::ModuleOp"> {
  let summary = "Convert Accera dialects to NVVM dialect";
  let constructor = "accera::transforms::createAcceraToNVVMPass()";
  let dependentDialects = [
    "accera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::gpu::GPUDialect",
    "mlir::NVVM::NVVMDialect"
  ];
}

#endif // ACCERA_CONVERSION_PASSES
