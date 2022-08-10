////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ACCERA_OPS
#define ACCERA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Accera_Dialect : Dialect {
  let name = "accera";
  let description = [{The Accera (`accera`) dialect
  }];
  let cppNamespace = "accera::ir::rc";
}

class RC_Op<string mnemonic, list<Trait> traits = []> :
    Op<Accera_Dialect, mnemonic, traits>;

def RC_GemmOp : RC_Op<"GEMM", [NoSideEffect]> {
  let summary = "Accera GEMM operation";
  let description = [{
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16,F32,F64,I32,I64]>, MemRefOf<[F16,F32,F64,I32,I64]>]>:$A,
    AnyTypeOf<[TensorOf<[F16,F32,F64,I32,I64]>, MemRefOf<[F16,F32,F64,I32,I64]>]>:$B,
    AnyTypeOf<[TensorOf<[F16,F32,F64,I32,I64]>, MemRefOf<[F16,F32,F64,I32,I64]>, NoneType]>:$C,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha,
    DefaultValuedAttr<F32Attr, "1.0">:$beta,
    DefaultValuedAttr<I64Attr, "0">:$transA,
    DefaultValuedAttr<I64Attr, "0">:$transB);
  let results = (outs AnyTypeOf<[TensorOf<[F16,F32,F64,I32,I64]>, MemRefOf<[F16,F32,F64,I32,I64]>]>:$Y);
}

#endif // ACCERA_OPS
