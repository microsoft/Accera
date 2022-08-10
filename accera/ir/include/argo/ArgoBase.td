////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ARGO_BASE
#define ARGO_BASE

include "mlir/IR/OpBase.td"

def Argo_Dialect : Dialect {
  let name = "argo";
  let cppNamespace = "mlir::argo";
  let description = [{
    The `argo` dialect includes Microsoft extension on top of MLIR's
    `linalg` dialect.
  }];
}

#endif // ARGO_BASE
