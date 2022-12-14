////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ACCERA_intrinsic_OPS
#define ACCERA_intrinsic_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def AcceraIntrinsics_Dialect : Dialect {
  let name = "accintr";
  let cppNamespace = "::accera::ir::intrinsics";
}

// Implements the LLVM_IntrOpBase interface (from mlir/Dialect/LLVMIR/LLVMOpBase.td)
// rather than LLVM_OneResultIntrOp because we don't want to put this op in the llvm dialect.
// Otherwise it will screw up how the conversion is handled later in acc-translate.
// However, we still want the other args to be like those in LLVM_OneResultIntrOp and LLVM_IntrOp
def accintr_VpmaddwdOp : LLVM_IntrOpBase<AcceraIntrinsics_Dialect, // Dialect
                                        "x86.avx2.pmadd.wd", // MLIR op name (will get prefixed with the "accintr." dialect name)
                                        "x86_avx2_pmadd_wd", // LLVM IR C++ enum name (see <llvm-project>/llvm/include/llvm/IR/IntrinsicsX86.td )
                                        [], // overloadedResults
                                        [], // overloadedOperands 
                                        [NoSideEffect], // traits
                                        1>, // num results
                          Arguments<(ins LLVM_Type, LLVM_Type)>;


// TODO : this may not be needed when we have multi-dimensional reductions supporting max/min
def accintr_VmaxpsOp : LLVM_IntrOpBase<AcceraIntrinsics_Dialect, // Dialect
                                       "x86.avx.max.ps.256", // MLIR op name (will get prefixed with the "accintr." dialect name)
                                       "x86_avx_max_ps_256", // LLVM IR C++ enum name (see <llvm-project>/llvm/include/llvm/IR/IntrinsicsX86.td )
                                       [], // overloadedResults
                                       [], // overloadedOperands 
                                       [NoSideEffect], // traits
                                       1>, // num results
                          Arguments<(ins LLVM_Type, LLVM_Type)>;

def accintr_VminpsOp : LLVM_IntrOpBase<AcceraIntrinsics_Dialect, // Dialect
                                       "x86.avx.min.ps.256", // MLIR op name (will get prefixed with the "accintr." dialect name)
                                       "x86_avx_min_ps_256", // LLVM IR C++ enum name (see <llvm-project>/llvm/include/llvm/IR/IntrinsicsX86.td )
                                       [], // overloadedResults
                                       [], // overloadedOperands 
                                       [NoSideEffect], // traits
                                       1>, // num results
                          Arguments<(ins LLVM_Type, LLVM_Type)>;

// TODO : remove after the next llvm update. There is a new math::roundeven op that we can use
def accintr_RoundEvenOp : LLVM_IntrOpBase<AcceraIntrinsics_Dialect, // Dialect
                                       "roundeven", // MLIR op name (will get prefixed with the "accintr." dialect name)
                                       "roundeven", // LLVM IR C++ enum name (see <llvm-project>/llvm/include/llvm/IR/IntrinsicsX86.td )
                                       [], // overloadedResults
                                       [0], // overloadedOperands 
                                       [NoSideEffect, SameOperandsAndResultType], // traits
                                       1>, // num results
                          Arguments<(ins LLVM_Type)>;

def accintr_RoundF32VecAVX2 : LLVM_IntrOpBase<AcceraIntrinsics_Dialect, // Dialect
                                       "x86.avx.cvt.ps2dq.256", // MLIR op name (will get prefixed with the "accintr." dialect name)
                                       "x86_avx_cvt_ps2dq_256", // LLVM IR C++ enum name (see <llvm-project>/llvm/include/llvm/IR/IntrinsicsX86.td )
                                       [], // overloadedResults
                                       [], // overloadedOperands 
                                       [NoSideEffect], // traits
                                       1>, // num results
                          Arguments<(ins LLVM_Type)>;

#endif // ACCERA_intrinsic_OPS
