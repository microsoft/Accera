////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LOOPNEST_ATTRS
#define LOOPNEST_ATTRS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "ir/include/nest/LoopNestBase.td"

//
// Attributes
//

def FRAGMENT_FIRST : I64EnumAttrCase<"first", 0>;
def FRAGMENT_LAST : I64EnumAttrCase<"last", 1>;
def FRAGMENT_END_BOUNDARY : I64EnumAttrCase<"endBoundary", 2>;
def FRAGMENT_ALL : I64EnumAttrCase<"all", 3>;
def FRAGMENT_SELECT : I64EnumAttrCase<"select", 4>;
def FRAGMENT_RANGE : I64EnumAttrCase<"range", 5>;

def rcln_FragmentTypeAttr : I64EnumAttr<
  "FragmentType", "An attribute containing a Fragment enum",
  [FRAGMENT_FIRST, FRAGMENT_LAST, FRAGMENT_END_BOUNDARY, FRAGMENT_ALL, FRAGMENT_SELECT, FRAGMENT_RANGE]> {
  let cppNamespace = "::accera::ir::loopnest";
}

def PLACEMENT_BEFORE : I64EnumAttrCase<"before", 0>;
def PLACEMENT_AFTER : I64EnumAttrCase<"after", 1>;

def rcln_PlacementPredicateAttr : I64EnumAttr<
  "PlacementType", "An attribute containing a Placement enum",
  [PLACEMENT_BEFORE, PLACEMENT_AFTER]> {
  let cppNamespace = "::accera::ir::loopnest";
}

def POSITION_PROLOGUE : I64EnumAttrCase<"prologue", 0>;
def POSITION_BODY : I64EnumAttrCase<"body", 1>;
def POSITION_EPILOGUE : I64EnumAttrCase<"epilogue", 2>;

def rcln_PositionAttr : I64EnumAttr<
  "Position", "An attribute containing a Placement enum",
  [POSITION_PROLOGUE, POSITION_BODY, POSITION_EPILOGUE]> {
  let cppNamespace = "::accera::ir::loopnest";
}

def rcln_IndexAttr : DialectAttr<
    LoopNest_Dialect,
    CPred<"$_self.isa<IndexAttr>()">,
    "Symbolic index attribute"> {
  let storageType = "IndexAttr";
  let returnType = "IndexAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def rcln_IndexRangeAttr : DialectAttr<
    LoopNest_Dialect,
    CPred<"$_self.isa<IndexRangeAttr>()">,
    "Index range attribute"> {
  let storageType = "IndexRangeAttr";
  let returnType = "IndexRangeAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def rcln_RangeAttr : DialectAttr<
    LoopNest_Dialect,
    CPred<"$_self.isa<RangeAttr>()">,
    "Range attribute"> {
  let storageType = "RangeAttr";
  let returnType = "RangeAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def rcln_IterationDomainAttr : DialectAttr<
    LoopNest_Dialect,
    CPred<"$_self.isa<IterationDomainAttr>()">,
    "IterationDomain attribute"> {
  let storageType = "IterationDomainAttr";
  let returnType = "IterationDomainAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def rcln_SplitIndexAttr : DialectAttr<
    LoopNest_Dialect,
    CPred<"$_self.isa<SplitIndexAttr>()">,
    "Split index attribute"> {
  let storageType = "SplitIndexAttr";
  let returnType = "SplitIndexAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def rcln_TransformedDomainAttr : DialectAttr<
    LoopNest_Dialect,
    CPred<"$_self.isa<TransformedDomainAttr>()">,
    "TransformedDomain attribute"> {
  let storageType = "TransformedDomainAttr";
  let returnType = "TransformedDomainAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def rcln_OperandIndexAttr : DialectAttr<
    LoopNest_Dialect,
    CPred<"$_self.isa<OperandIndexAttr>()">,
    "Attribute referencing an operand of an op"> {
  let storageType = "OperandIndexAttr";
  let returnType = "OperandIndexAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

#endif // LOOPNEST_ATTRS
