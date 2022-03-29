////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LOOPNEST_EXPORTED_INTERFACES
#define LOOPNEST_EXPORTED_INTERFACES

include "mlir/IR/OpBase.td"

//
// Mapping operation interface 
//
def accln_InjectableMappingOpInterface : OpInterface<"InjectableMapping"> {
  let description = [{
    Interface for mapping ops to be injected into a loopnest schedule
  }];

  let methods = [
    InterfaceMethod<[{
        Returns the index to inject at
      }],
      "Index", "index", (ins)
    >,
    InterfaceMethod<[{
        Returns the position to inject at
      }],
      "Position", "position", (ins)
    >,
    InterfaceMethod<[{
        Returns the end region operation
      }],
      "Operation*", "getInjectionEndOp", (ins)
    >,
  ];
}

#endif // LOOPNEST_EXPORTED_INTERFACES
