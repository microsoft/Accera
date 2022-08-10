////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ARGO_OPS
#define ARGO_OPS

// include "mlir/Dialect/Argo/IR/ArgoBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "ir/include/argo/ArgoBase.td"

// Base class for Argo dialect ops
class Argo_Op<string mnemonic, list<Trait> traits = []> :
    Op<Argo_Dialect, mnemonic, traits> {
  // For every argo op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def Argo_YieldOp : Argo_Op<"yield", [NoSideEffect, ReturnLike, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Argo yield operation";
  let description = [{
    `argo.yield` is a special terminator operation for blocks inside regions
    in `argo` opaque ops. It returns values to the immediately enclosing
    `argo` opaque op.

    Example:

    ```mlir
    argo.yield %f0, %f1 : f32, f32
    ```
  }];
}

def Argo_EntryPointOp : Argo_Op<"entry_point", [IsolatedFromAbove, FunctionOpInterface,
                                                HasParent<"ModuleOp">, Symbol]> {
  let summary = "Declare a host entry function for a kernel function.";

  let description = [{
    `argo.entry_point` is a function-declaration-like operation that declares
    an entry point for a kernel function. A kernel function may run on any
    target, including the host where the entry point is declared. For example,
    an entry point may point to a kernel function that runs on the host to call
    mutiple device functions.

    This argo.entry_point op provides an abstraction to
    connect the kernel and the entry while offering certain flexibility to the
    downstream dialects, which will instantiate the entry-kernel interface
    during the lowering process. Moreover, each downstream dialect has its own
    freedom to interpret the arguments. For example, lowering to the GPU dialect
    would convert an `argo.entry_point` op into a host wrapper function that
    calls the `gpu.launch_func`, where the first six arguments are kernel launch
    configs such as `gridSizeX` and `gridSizeY`.

    In the op declaration, the kernel is specified using the `kernel_name`
    symbol attribute. Note that the kernel may sit in a nested sub-module.
    In such a case, its symbol will be qualified by the sub-module name.
    The execution space of the kernel is specified using the
    `kernel_execution_space` string attribute, which could be either "host" or
    "device". The "host" `kernel_execution_space` means that the kernel runs on
    the same target as where the entry point is declared, whereas the "device"
    `kernel_execution_space` denotes that the kernel runs on some other device
    such as GPU.

    Example:
    ```mlir
    argo.entry_point @host_name1(%arg0 : i32, %arg1 : index) -> ()
        attributes {kernel_name = @kernel1, kernel_execution_space = "host"}

    argo.entry_point @host_name2(%arg0 : i32, %arg1 : memref<?xf32, 1>) -> ()
        attributes {kernel_name = @module1::@kernel1,
                    kernel_execution_space = "device"}
    ```
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$entryName,
              "FunctionType":$type, "StringRef":$kernelName,
              "StringRef":$kernelExecSpace)>,
    OpBuilder<(ins "StringRef":$entryName,
              "FunctionType":$type, "StringRef":$moduleName, "StringRef":$kernelName,
              "StringRef":$kernelExecSpace)>
  ];

  let extraClassDeclaration = [{
    /// The kernel function specified by the `kernel_name` attribute
    SymbolRefAttr kernel();

    // The name of this entry_point op
    StringRef getEntryName() { return getName(); }

    // The name of the kernel
    StringRef getKernelName();

    /// The name of the kernel's containing module if any
    StringRef getKernelModuleName();

    /// The execution space of the kernel
    StringRef getKernelExecutionSpace();

    /// Returns the type of this function.
    /// FIXME: We should drive this via the ODS `type` param.
    FunctionType getType() { 
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

    /// Hook for FunctionOpInterface verifier
    LogicalResult verifyType();

    /// The name of the symbol attribute specifying the `kernel_name` attribute
    static StringRef getKernelNameAttrName() { return "kernel_name"; }

    /// The name of the string attribute specifying the
    /// `kernel_execution_space` attribute
    static StringRef getKernelExecutionSpaceAttrName() {
      return "kernel_execution_space";
    }

    /// The string that denotes the `device` kernel execution space
    static StringRef getDeviceExecutionSpace() {
      return "device";
    }

    /// The string that denotes the `host` kernel execution space
    static StringRef getHostExecutionSpace() {
      return "host";
    }
  }];

  let parser = [{ return parseEntryPointOp(parser, result); }];
  let printer = [{ printEntryPointOp(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
}

#endif // ARGO_OPS
