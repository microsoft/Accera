////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LOOPNEST_INTERFACES
#define LOOPNEST_INTERFACES

include "mlir/IR/OpBase.td"

def rcln_EvaluatablePredicateOpInterface : OpInterface<"EvaluatablePredicateOpInterface"> {
  let description = [{
    Evaluatable predicate at emit-time
  }];

  // TODO : generalize the current state
  let methods = [
    InterfaceMethod<[{
        Returns true if the predicate is true given the current state
      }],
      "bool", "evaluate", (ins "const std::vector<Index>&":$definedIndices, "const Index&":$currentIndex, "const Position&":$position)
    >
  ];
}

def rcln_KernelPredicateOpInterface : OpInterface<"KernelPredicateOpInterface"> {
  let description = [{
    Predicate determining if a Kernel should be run
  }];

  let methods = [
    InterfaceMethod<[{
        Evaluates the predicate at compile-time (if possible). Returns std::nullopt if the predicate can't be evaluated at compile-time.
      }],
      "std::optional<bool>", "evaluate", (ins "const TransformedDomain&":$domain, "const LoopIndexSymbolTable&":$indices, "const LoopVisitSchedule&":$schedule)
    >,
    InterfaceMethod<[{
        Simplifies the predicate
      }],

      "Operation*", "simplify", (ins "const TransformedDomain&":$domain, "const LoopIndexSymbolTable&":$indices, "const LoopVisitSchedule&":$schedule)
    >
  ];
}

// TODO : this interface can't surface the size because the op needs to be lowered first, rename it something related to
//        what it does rather than how it will be used
def rcln_EmitTimeSizeOpInterface : OpInterface<"EmitTimeSize"> {
  let description = [{
    Evaluatable size at conversion-time
  }];

  let methods = [
  ];
}

#endif // LOOPNEST_INTERFACES
