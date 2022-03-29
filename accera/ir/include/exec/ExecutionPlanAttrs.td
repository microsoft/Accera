////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef EXECUTIONPLAN_ATTRIBUTES
#define EXECUTIONPLAN_ATTRIBUTES

include "ir/include/Common.td"

//
// Custom enum attribute values
//
def CACHE_MAPPING_GLOBAL_TO_PHYSICAL : I64EnumAttrCase<"GlobalToPhysical", 0>;
def CACHE_MAPPING_LOGICAL_TO_PHYSICAL : I64EnumAttrCase<"LogicalToPhysical", 1>;
def CACHE_MAPPING_LOGICAL_TO_GLOBAL : I64EnumAttrCase<"LogicalToGlobal", 2>;
def CACHE_MAPPING_NONE : I64EnumAttrCase<"None", 3>;

def accxp_CacheMappingAttr : I64EnumAttr<
  "CacheIndexing",
  "An attribute containing a cache mapping type enum. This indicates which cache access map to use to access a cache.",
  [CACHE_MAPPING_GLOBAL_TO_PHYSICAL, CACHE_MAPPING_LOGICAL_TO_PHYSICAL, CACHE_MAPPING_LOGICAL_TO_GLOBAL, CACHE_MAPPING_NONE]> {
  let cppNamespace = "::accera::ir::executionPlan";
}

def CACHE_ALLOCATION_AUTOMATIC : I64EnumAttrCase<"Automatic", 0>;
def CACHE_ALLOCATION_NONE : I64EnumAttrCase<"None", 1>;

def accxp_CacheAllocationAttr : I64EnumAttr<
  "CacheAllocation", "An attribute containing a cache allocation type enum",
  [CACHE_ALLOCATION_AUTOMATIC, CACHE_ALLOCATION_NONE]> {
  let cppNamespace = "::accera::ir::executionPlan";
}

def CACHE_COPY_SRC_DIMS : I64EnumAttrCase<"Source", 0>;
def CACHE_COPY_DST_DIMS : I64EnumAttrCase<"Destination", 1>;

def accxp_CacheCopyDimensionsAttr : I64EnumAttr<
  "CacheCopyDimensions", "An attribute containing a cache copy shape dimensions assignment. This indicates which side of the copy is represented by the copy dimension sizes.",
  [CACHE_COPY_SRC_DIMS, CACHE_COPY_DST_DIMS]> {
  let cppNamespace = "::accera::ir::executionPlan";
}

#endif // EXECUTIONPLAN_ATTRIBUTES
