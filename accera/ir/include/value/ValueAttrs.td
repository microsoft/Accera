////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ACCERA_accv_ATTRS
#define ACCERA_accv_ATTRS

include "ir/include/Common.td"

def ExecutionTargetCPU : I64EnumAttrCase<"CPU", 0>;
def ExecutionTargetGPU : I64EnumAttrCase<"GPU", 1>;

def ExecutionTargetAttr : I64EnumAttr<"ExecutionTarget", "target for function", [ExecutionTargetCPU, ExecutionTargetGPU]> {
    let cppNamespace = "::accera::ir::value";
}

def ExecutionRuntimeNone : StrEnumAttrCase<"NONE">;
def ExecutionRuntimeCUDA : StrEnumAttrCase<"CUDA">;
def ExecutionRuntimeRocm : StrEnumAttrCase<"ROCM">;
def ExecutionRuntimeVulkan : StrEnumAttrCase<"VULKAN">;
def ExecutionRuntimeOpenMP : StrEnumAttrCase<"OPENMP">;
def ExecutionRuntimeDefault : StrEnumAttrCase<"DEFAULT">;


def ExecutionRuntimeAttr : StrEnumAttr<"ExecutionRuntime", "execution runtime for function",
    [
        ExecutionRuntimeNone,
        ExecutionRuntimeCUDA,
        ExecutionRuntimeRocm,
        ExecutionRuntimeVulkan,
        ExecutionRuntimeOpenMP,
        ExecutionRuntimeDefault
    ]> {
    let cppNamespace = "::accera::ir::value";
    let genSpecializedAttr = 1;
}

def BlockX : I64EnumAttrCase<"BlockX", 0>;
def BlockY : I64EnumAttrCase<"BlockY", 1>;
def BlockZ : I64EnumAttrCase<"BlockZ", 2>;
def ThreadX : I64EnumAttrCase<"ThreadX", 3>;
def ThreadY : I64EnumAttrCase<"ThreadY", 4>;
def ThreadZ : I64EnumAttrCase<"ThreadZ", 5>;
def Sequential : I64EnumAttrCase<"Sequential", 6>;

def ProcessorAttr : I64EnumAttr<"Processor", "processor for loop mapping", [
        BlockX, BlockY, BlockZ, ThreadX, ThreadY, ThreadZ, Sequential]> {
    let cppNamespace = "::accera::ir::value";
}

def MEMORY_ALLOC_GLOBAL : I64EnumAttrCase<"Global", 0>;
def MEMORY_ALLOC_STACK : I64EnumAttrCase<"Stack", 1>;
def MemoryAllocTypeAttr : I64EnumAttr<
        "MemoryAllocType",
        "Describes the memory type in which an allocation resides.",
        [ MEMORY_ALLOC_GLOBAL, MEMORY_ALLOC_STACK]> {
    let cppNamespace = "::accera::ir::value";
}

def MEMORY_SPACE_NONE : I64EnumAttrCase<"None", 0>;
def MEMORY_SPACE_GLOBAL : I64EnumAttrCase<"Global", 1>;
def MEMORY_SPACE_SHARED : I64EnumAttrCase<"Shared", 3>;
def MEMORY_SPACE_PRIVATE : I64EnumAttrCase<"Private", 5>;

def MemorySpaceAttr : I64EnumAttr<
        "MemorySpace",
        "Describes the memory space in which an allocation resides.",
        [ MEMORY_SPACE_NONE, MEMORY_SPACE_GLOBAL, MEMORY_SPACE_SHARED, MEMORY_SPACE_PRIVATE]> {
    let cppNamespace = "::accera::ir::value";
}

def BARRIER_SCOPE_BLOCK : StrEnumAttrCase<"Block", 0>;
def BARRIER_SCOPE_WARP : StrEnumAttrCase<"Warp", 1>;
def BARRIER_SCOPE_THREADFENCE : StrEnumAttrCase<"Threadfence", 2>;

def BarrierScopeAttr : StrEnumAttr<
        "BarrierScope",
        "Describes the scope of the barrier operator.",
        [ BARRIER_SCOPE_BLOCK, BARRIER_SCOPE_WARP, BARRIER_SCOPE_THREADFENCE ]> {
    let cppNamespace = "::accera::ir::value";
    let genSpecializedAttr = 1;
}
#endif // ACCERA_accv_ATTRS
