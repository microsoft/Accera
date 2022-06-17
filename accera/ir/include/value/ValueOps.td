////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ACCERA_accv_OPS
#define ACCERA_accv_OPS

include "ir/include/value/ValueBase.td"

include "ir/include/value/ValueAttrs.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"

def accv_ValueLambdaOp : accv_Op<"lambda", [
    SymbolTable,
    Symbol,
    FunctionLike,
    DeclareOpInterfaceMethods<CallableOpInterface>,
    AffineScope]> {
  let summary = "Executable inline function for either CPU or GPU";
  let regions = (region SizedRegion<1>:$body);
  let description = [{}];
  let arguments = (ins StrAttr:$sym_name, ExecutionTargetAttr:$exec_target, Variadic<AnyType>:$args, TypeAttr:$type);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type, "accera::ir::value::ExecutionTarget":$exec_target)>,
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type, "accera::ir::value::ExecutionTargetAttr":$exec_target)>
  ];

  let extraClassDeclaration = [{
    // FunctionLike trait needs access to the functions below.
    friend class OpTrait::FunctionLike<ValueLambdaOp>;

    static StringRef getGPULaunchAttrName() { return "gpu_launch"; }

    static StringRef getExecTargetAttrName() { return "exec_target"; }

    ::accera::ir::value::ExecutionTarget exec_target_value()
    {
        auto attr = exec_targetAttr();
        return static_cast<::accera::ir::value::ExecutionTarget>(attr.getInt());
    }

    /// Hooks for the input/output type enumeration in FunctionLike .
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();
  }];
}

def accv_ValueModuleOp : accv_Op<"module", [
    IsolatedFromAbove,
    SymbolTable,
    Symbol,
    SingleBlockImplicitTerminator<"ModuleTerminatorOp">,
    NoRegionArguments,
    AffineScope]> {
  let summary = "A top level Accera compilation unit";
  let description = [{}];
  let arguments = (ins StrAttr:$sym_name);
  let builders = [OpBuilder<(ins "StringRef":$name)>];
  let regions = (region SizedRegion<1>:$body);

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;

  let hasCanonicalizer = 1;

  let assemblyFormat = "$sym_name $body attr-dict";

  let extraClassDeclaration = [{
    static StringRef getExecRuntimeAttrName() { return "accv.exec_runtime"; }
  }];
}

def accv_ModuleTerminatorOp : accv_Op<"module_terminator", [Terminator, HasParent<"ValueModuleOp">]> {
  let summary = "A pseudo op that marks the end of a gpu.module.";
  let description = [{}];
}

def accv_PrintOp : accv_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The `accv.print` operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyTensor, acc_NumericType]>:$input,
    UnitAttr:$to_stderr
  );
}

def accv_PrintFOp : accv_Op<"printf"> {
  let summary = "printf operation";
  let description = [{
    The `printf` builtin operation prints a scalar value and returns no results.
  }];

  // The printf operation takes an input value to print.
  let arguments = (ins
    StrAttr:$fmt_spec,
    Variadic<AnyType>:$input,
    UnitAttr:$to_stderr
  );

  let builders = [
    OpBuilder<(ins "StringRef":$fmt, "bool":$to_stderr), [{
               build($_builder, $_state, fmt, ValueRange{}, to_stderr); }]>
  ];
}

def accv_GlobalOp : accv_Op<"global", [IsolatedFromAbove, Symbol]> {
  let arguments = (ins
    TypeAttrBase<"MemRefType", "MemRef">:$type,
    UnitAttr:$constant,
    UnitAttr:$external,
    StrAttr:$sym_name,
    OptionalAttr<AnyAttr>:$value,
    DefaultValuedAttr<Confined<I32Attr, [IntNonNegative]>, "0">:$addr_space
  );
  let summary = "Value dialect global.";

  let builders = [
    OpBuilder<(ins "MemRefType":$type, "bool":$isConstant, "StringRef":$name,
              "Attribute":$value, CArg<"unsigned", "0">:$addrSpace, CArg<"bool", "false">:$isExternal)>
  ];

  let extraClassDeclaration = [{
    /// Return the type of the global.
    MemRefType getType() {
      return type().cast<MemRefType>();
    }
    /// Return the initializer attribute if it exists, or a null attribute.
    Attribute getValueOrNull() {
      return value().getValueOr(Attribute());
    }
  }];
}

def accv_ReferenceGlobalOp : accv_Op<"ref_global"> {
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs AnyStaticShapeMemRef);

  let builders = [
    OpBuilder<(ins "MemRefType":$resType, "StringRef":$name),
    [{
      $_state.addAttribute("global_name", $_builder.getSymbolRefAttr(name));
      $_state.addTypes(resType);
    }]>,

    OpBuilder<(ins "GlobalOp":$global),
    [{
      build($_builder, $_state, global.getType(), global.sym_name());
    }]>
  ];

  let extraClassDeclaration = [{
    GlobalOp getGlobal();

    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

  }];
}

def accv_AllocOp : accv_Op<"alloc"> {
  let summary = "Memory allocation operation";
  let description = [{
    The `accv.alloc` operation allocates a region of memory, as specified by its
    memref type. For example:

      ```mlir
      %0 = accv.alloc() : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
      ```

    This operation returns a single ssa value of memref type, which can be used
    by subsequent load and store operations.

    The optional `alignment` attribute may be specified to ensure that the
    region of memory that will be indexed is aligned at the specified byte
    boundary.

      ```mlir
      %0 = accv.alloc() {alignment = 8} : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
      ```
  }];

  let arguments = (ins OptionalAttr<AnyAttr>:$data,
                   Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment,
                   OptionalAttr<MemoryAllocTypeAttr>:$allocType);
  let results = (outs AnyStaticShapeMemRef);

  let builders = [
    OpBuilder<(ins "MemRefType":$memrefType, CArg<"Optional<int64_t>", "llvm::None">:$alignment, CArg<"Optional<MemoryAllocType>", "llvm::None">:$allocType), [{
       $_state.types.push_back(memrefType);
       if (alignment)
         $_state.addAttribute(getAlignmentAttrName(), $_builder.getI64IntegerAttr(*alignment));
       if (allocType)
         $_state.addAttribute(getAllocTypeAttrName(), $_builder.getI64IntegerAttr(static_cast<int64_t>(*allocType)));
     }]>,
    OpBuilder<(ins "MemRefType":$memrefType, "MemoryAllocType":$allocType), [{
        build($_builder, $_state, memrefType, llvm::None, allocType);
     }]>,
    OpBuilder<(ins "Attribute":$data), [{
        build($_builder, $_state, data.getType(), data, IntegerAttr{}, MemoryAllocTypeAttr{});
    }]>];

  let extraClassDeclaration = [{
    static StringRef getDataAttrName() { return "data"; }
    static StringRef getAlignmentAttrName() { return "alignment"; }
    static StringRef getAllocTypeAttrName() { return "allocType"; }

    Attribute getData() { return (*this)->getAttr(getDataAttrName()); }

    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];
}


def accv_BitcastOp : accv_Op<"bitcast_op",
  [NoSideEffect]> {
    let summary = "bitcast operation";
    let description = [{
      The `accv.bitcast_op` operation reinterprets the bits of a value as a value of a different type. The source and destination types must have the same number of bits.
    }];

    let arguments = (ins
      acc_ScalarOrVectorNumericType:$input
    );
    let results = (outs acc_ScalarOrVectorNumericType:$result);
}


def accv_UNARY_OP_NOT : I64EnumAttrCase<"NOT", 0>;

def accv_UnaryOpPredicateAttr : I64EnumAttr<
  "UnaryOpPredicate", "",
  [accv_UNARY_OP_NOT]> {
  let cppNamespace = "::accera::ir::value";
}

def accv_UnaryOp : accv_Op<"unary_op",
  [NoSideEffect]> {
    let summary = "unary operation";
    let description = [{
      The `accv.unary_op` operations performs the unary operation specified by
      the predicate. The result and operands must have the same shape and type.
    }];

    let arguments = (ins
      accv_UnaryOpPredicateAttr:$predicate,
      acc_NumericType:$input
    );
    let results = (outs acc_NumericType:$result);

    let builders = [OpBuilder<(ins "UnaryOpPredicate":$predicate, "Value":$input), [{
        ::buildUnaryOp($_builder, $_state, predicate, input);
    }]>];

    let extraClassDeclaration = [{
      static StringRef getPredicateAttrName() { return "predicate"; }

      UnaryOpPredicate getPredicate() {
        return (UnaryOpPredicate)(*this)->getAttrOfType<IntegerAttr>(getPredicateAttrName())
            .getInt();
      }
    }];
}

def accv_BINARY_OP_ADD : I64EnumAttrCase<"ADD", 0>;
def accv_BINARY_OP_SUB : I64EnumAttrCase<"SUB", 1>;
def accv_BINARY_OP_MUL : I64EnumAttrCase<"MUL", 2>;
def accv_BINARY_OP_DIV : I64EnumAttrCase<"DIV", 3>;
def accv_BINARY_OP_MOD : I64EnumAttrCase<"MOD", 4>;
def accv_BINARY_OP_AND : I64EnumAttrCase<"LOGICAL_AND", 5>;
def accv_BINARY_OP_OR : I64EnumAttrCase<"LOGICAL_OR", 6>;

def accv_BinaryOpPredicateAttr : I64EnumAttr<
  "BinaryOpPredicate", "",
  [accv_BINARY_OP_ADD, accv_BINARY_OP_SUB, accv_BINARY_OP_MUL, accv_BINARY_OP_DIV, accv_BINARY_OP_MOD,
  accv_BINARY_OP_AND, accv_BINARY_OP_OR]> {
  let cppNamespace = "::accera::ir::value";
}

def accv_BinOp : accv_Op<"bin_op",
  [NoSideEffect]> {
    let summary = "binary operation";
    let description = [{
      The `accv.bin_op` operations performs the binary operation specified by the
      predicate. The result and operands must have the same shape and type.
    }];

    let arguments = (ins
      accv_BinaryOpPredicateAttr:$predicate,
      acc_ScalarOrVectorNumericType:$lhs,
      acc_ScalarOrVectorNumericType:$rhs
    );
    let results = (outs acc_ScalarOrVectorNumericType:$result);

    let builders = [OpBuilder<(ins "BinaryOpPredicate":$predicate, "Value":$lhs, "Value":$rhs), [{
        ::buildBinOp($_builder, $_state, predicate, lhs, rhs);
    }]>];

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
      static StringRef getPredicateAttrName() { return "predicate"; }

      BinaryOpPredicate getPredicate() {
        return (BinaryOpPredicate)(*this)->getAttrOfType<IntegerAttr>(getPredicateAttrName())
            .getInt();
      }
    }];
}

def accv_CMP_P_EQ : I64EnumAttrCase<"EQ", 0>;
def accv_CMP_P_NE : I64EnumAttrCase<"NE", 1>;
def accv_CMP_P_LT : I64EnumAttrCase<"LT", 2>;
def accv_CMP_P_LE : I64EnumAttrCase<"LE", 3>;
def accv_CMP_P_GT : I64EnumAttrCase<"GT", 4>;
def accv_CMP_P_GE : I64EnumAttrCase<"GE", 5>;

def accv_CmpOpPredicateAttr : I64EnumAttr<
    "CmpOpPredicate", "",
    [accv_CMP_P_EQ, accv_CMP_P_NE, accv_CMP_P_LT, accv_CMP_P_LE, accv_CMP_P_GT, accv_CMP_P_GE]> {
  let cppNamespace = "::accera::ir::value";
}

def accv_CmpOp : accv_Op<"cmp",
    [NoSideEffect]> {
  let summary = "comparison operation";
  let description = [{
    The `accv.cmp` operation compares its two operands according to the
    predicate specified. The predicate defines the type of comparison:
    (in)equality, less/greater than (or equal to).  The
    operands must have the same type. The result is an i1 or a vector of i1.
  }];

  let arguments = (ins
      accv_CmpOpPredicateAttr:$predicate,
      acc_ScalarOrVectorNumericType:$lhs,
      acc_ScalarOrVectorNumericType:$rhs
  );
  let results = (outs acc_ScalarOrVectorBoolType:$result);

  let builders = [OpBuilder<(ins "CmpOpPredicate":$predicate, "Value":$lhs, "Value":$rhs), [{
      ::buildCmpOp($_builder, $_state, predicate, lhs, rhs);
  }]>];

  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }

    CmpOpPredicate getPredicate() {
      return (CmpOpPredicate)(*this)->getAttrOfType<IntegerAttr>(getPredicateAttrName())
          .getInt();
    }
  }];
}

def accv_CopyOp : accv_Op<"copy"> {
  let description = [{
    Copies the data in the input view into the output view.

    Usage:
      ```mlir
      accv.copy(%arg0, %arg1) : memref<?xf32, stride_specification>,
                                  memref<?xf32, stride_specification>
      ```

    Optionally, can take `input_permutation` and `output_permutation` attributes
    to reorder the dimensions of the input and output views.

    Usage:
      ```mlir
      accv.copy(%arg0, %arg1) {inputPermutation : (i, j, k) -> (i, k, j),
                                 outputPermutation : (i, j, k) -> (k, j, i)} :
        memref<?x?x?xf32, stride_specification>,
        memref<?x?x?xf32, stride_specification>
     ```

    The views are expected to be compatible for correctness but this is not
    enforced at the moment.
  }];

  let arguments = (ins
    // AnyTypeOf<[AnyStridedMemRef,AnyStaticShapeTensor]>:$input,
    AnyType:$input,
    AnyTypeOf<[AnyStridedMemRef]>:$output,
    // AnyType:$output,
    OptionalAttr<AffineMapAttr>:$inputPermutation,
    OptionalAttr<AffineMapAttr>:$outputPermutation);

  let builders = [OpBuilder<(ins "Value":$input, "Value":$output), [{
    return build(
      $_builder, $_state, input, output, AffineMapAttr(), AffineMapAttr());
  }]>];

  let hasCanonicalizer = 1;
}

def accv_IfOp : accv_Op<"if",
      [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "if-then-else operation";
  let description = [{
    The `accv.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
       accv.if %b  {
         ...
       } else {
         ...
       }
    ```

    "accv.if" may also return results that are defined in its regions. The
    values defined are determined by which execution path is taken.
    For example:
    ```mlir
       %x, %y = accv.if %b -> (f32, f32) {
         %x_true = ...
         %y_true = ...
         accv.yield %x_true, %y_true : f32, f32
       } else {
         %x_false = ...
         %y_false = ...
         accv.yield %x_false, %y_false : f32, f32
       }
    ```

    "accv.if" regions are always terminated with "accv.yield". If "accv.if"
    defines no values, the "accv.yield" can be left out, and will be
    inserted implicitly. Otherwise, it must be explicit.
    Also, if "accv.if" defines one or more values, the 'else' block cannot
    be omitted.

    For example:
    ```mlir
       loop.if %b  {
         ...
       }
    ```
  }];
  let arguments = (ins acc_Scalarlike<acc_BoolType>:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$cond, "bool":$withElseRegion)>
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder() {
      assert(!thenRegion().empty() && "Unexpected empty 'then' region.");
      Block &body = thenRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
    OpBuilder getElseBodyBuilder() {
      assert(!elseRegion().empty() && "Unexpected empty 'else' region.");
      Block &body = elseRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
  }];
}

def accv_LoadOp : accv_Op<"load",
    [AllElementTypesMatch<["memref", "result"]>]> {
  let summary = "load operation";
  let description = [{
    The `accv.load` op reads an element from a memref specified by an index list.
    The output of load is a new value with the same type as the elements of the
    memref. The arity of indices is the rank of the memref (i.e., if the memref
    loaded from is of rank 3, then 3 indices are required for the load following
    the memref identifier). For example:

      ```mlir
      %3 = load %0[%1, %1] : memref<4x4xi32>
      ```
  }];

  let arguments = (ins
    AnyTypeOf<[AnyMemRef,AnyStaticShapeTensor]>:$memref,
    Variadic<acc_Indexlike>:$indices);
  let results = (outs acc_NumericType:$result);

  let builders = [OpBuilder<(ins "Value":$data, CArg<"ValueRange", "{}">:$indices), [{
      auto shapedType = data.getType().cast<ShapedType>();
      $_state.addOperands(data);
      $_state.addOperands(indices);
      $_state.types.push_back(MemRefType::get({1}, shapedType.getElementType()));
  }]>];

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }

    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
  }];
}

def accv_GetElementOp : accv_Op<"get_element", [NoSideEffect]> {
  let summary = "load element from memref or tensor";
  let description =[{
    Returns the element at index 0 of a Scalar sized MemRef or Tensor
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);

  let builders = [OpBuilder<(ins "Value":$value), [{
      $_state.addOperands(value);
      auto valueType = value.getType();
      if (auto shapedType = valueType.dyn_cast<ShapedType>())
      {
        $_state.types.push_back(shapedType.getElementType());
      }
      else
      {
        $_state.types.push_back(valueType);
      }
  }]>];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def accv_LaunchFuncOp : accv_Op<"launch_func", [CallOpInterface]> {
  let summary = "call operation for multi-target func ops";
  let description = [{}];

  let arguments = (ins ExecutionTargetAttr:$exec_target, SymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<(ins "ValueFuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", $_builder.getSymbolRefAttr(callee));
      $_state.addAttribute("exec_target", callee.exec_targetAttr());
      if (auto attr = $_builder.getStringAttr(getGPULaunchAttrName()))
      {
        $_state.addAttribute(getGPULaunchAttrName(), attr);
      }
      $_state.addTypes(callee.getType().getResults());
  }]>];

  let extraClassDeclaration = [{
    static StringRef getGPULaunchAttrName() { return "gpu_launch"; }

    ::accera::ir::value::ExecutionTarget exec_target_value()
    {
        auto attr = exec_targetAttr();
        return static_cast<::accera::ir::value::ExecutionTarget>(attr.getInt());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    static StringRef getExecTargetAttrName() { return "exec_target"; }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def accv_CallOp : accv_Op<"call", [CallOpInterface]> {
  let summary = "call operation";
  let description = [{
    The `accv.call` operation represents a direct call to a function that is within
    the same symbol scope as the call.  The operands and result types of the
    call must match the specified function type. The callee is encoded as a
    function attribute named "callee".

      ```mlir
      %2 = call @my_add(%0, %1) : (f32, f32) -> f32
      ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<(ins "ValueFuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", $_builder.getSymbolRefAttr(callee));
      $_state.addTypes(callee.getType().getResults());
  }]>, OpBuilder<(ins "SymbolRefAttr":$callee, "ArrayRef<Type>":$results, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
  }]>, OpBuilder<(ins "StringRef":$callee, "ArrayRef<Type>":$results, CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, $_builder.getSymbolRefAttr(callee), results,
            operands);
  }]>];

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def accv_MemRefCastOp : accv_Op<"memref_cast", [SameOperandsAndResultShape]> {
  let summary = "memref element casting operation";
  let description = [{
    The `accv.memref_cast` operation converts the elements in a memref to another
    element and returns the resulting memref.
  }];

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef);

  let builders = [OpBuilder<(ins "Value":$source, "MemRefType":$destType), [{
      $_state.addOperands(source);
      $_state.types.push_back(destType);
  }]>];

  let extraClassDeclaration = [{
    MemRefType getSourceMemRefType() {
      return source().getType().cast<MemRefType>();
    }

    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];
}

def accv_OffsetOp : accv_Op<"offset", [NoSideEffect]> {
  let summary = "memref offset operation";
  let description = [{
    The `accv.offset` operation converts a memref type to another memref type
    which represents a reduced-size view of the original memref as specified by
    the operation's offsets.

    The SubView operation supports the following arguments:
    *) Memref: the "base" memref on which to create a "view" memref.
    *) Offsets: memref-rank number of dynamic offsets into the "base"
                memref at which to create the "view" memref.
  }];

  let arguments = (ins
    AnyMemRef:$source,
    Variadic<acc_Indexlike>:$offsets
  );
  let results = (outs AnyMemRef);

  let builders = [
    OpBuilder<(ins "Value":$source, "ValueRange":$offsets, "MemRefType":$resultType), [{
        $_state.addOperands(source);
        $_state.addOperands(offsets);
        $_state.types.push_back(resultType);
      }]>,
    OpBuilder<(ins "Value":$source, "MemRefType":$resultType), [{
        $_state.addOperands(source);
        $_state.addOperands(ValueRange{});
        $_state.types.push_back(resultType);
      }]>
  ];

  let extraClassDeclaration = [{
    /// Returns the type of the base memref operand.
    MemRefType getSourceMemRefType() {
      return source().getType().cast<MemRefType>();
    }

    /// The result of a subview is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Returns as integer value the number of offset operands.
    int64_t getNumOffsets() { return llvm::size(offsets()); }
  }];
}

def accv_ViewOp : accv_Op<"view", [NoSideEffect]> {
  let summary = "memref view operation";
  let description = [{
    The `accv.view` operation converts a memref type to another memref type
    which represents a reduced-size view of the original memref as specified by
    the operation's offsets.

    The view operation supports the following arguments:
    *) Memref: the "base" memref on which to create a "view" memref.
    *) Offsets: memref-rank number of dynamic offsets into the "base"
                memref at which to create the "view" memref.
  }];

  let arguments = (ins
    AnyMemRef:$source,
    Variadic<AnyType>:$offsets // TODO: change AnyType to linalg.range
  );
  let results = (outs AnyMemRef);

  let builders = [
    OpBuilder<(ins "Value":$source, "ArrayRef<std::pair<Value, Value>>":$bounds, "MemRefType":$resultType)>,
    OpBuilder<(ins "Value":$source, "ValueRange":$bounds, "MemRefType":$resultType), [{
        $_state.addOperands(source);
        $_state.addOperands(bounds);
        $_state.types.push_back(resultType);
      }]>,
    OpBuilder<(ins "Value":$source, "MemRefType":$resultType), [{
        $_state.addOperands(source);
        $_state.addOperands(ValueRange{});
        $_state.types.push_back(resultType);
      }]>
  ];

  let extraClassDeclaration = [{
    /// Returns the type of the base memref operand.
    MemRefType getSourceMemRefType() {
      return source().getType().cast<MemRefType>();
    }

    /// The result of a subview is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Returns as integer value the number of offset operands.
    int64_t getNumOffsets() { return llvm::size(offsets()); }
  }];
}

def accv_SliceOp : accv_Op<"slice", [NoSideEffect]> {
  let summary = "memref slice operation";
  let description = [{
    The `accv.slice` operation converts a memref type to another memref type
    which represents a reduced-rank view of the original memref as specified by
    the operation's slice dimensions.

    The view operation supports the following arguments:
    *) Memref: the "base" memref on which to create a "view" memref.
    *) SliceDimensions: the dimensions to remove from the memref
    *) Offsets: dynamic offsets into the "base" memref at the given dimensions
  }];

  let arguments = (ins
    AnyMemRef:$source,
    ArrayAttr:$sliceDimensions,
    Variadic<acc_Indexlike>:$offsets
  );
  let results = (outs AnyMemRef:$result);

  let builders = [
    OpBuilder<(ins "Value":$source, "ArrayAttr":$sliceDimensions, "ValueRange":$offsets, "MemRefType":$resultType), [{
        $_state.addOperands(source);
        $_state.addOperands(offsets);
        $_state.addAttribute("sliceDimensions", sliceDimensions);
        $_state.types.push_back(resultType);
      }]>,
    OpBuilder<(ins "Value":$source, "ArrayRef<int64_t>":$sliceDimensions, "ValueRange":$offsets, "MemRefType":$resultType), [{
        $_state.addOperands(source);
        $_state.addOperands(offsets);
        auto dims = $_builder.getI64ArrayAttr(sliceDimensions);
        $_state.addAttribute("sliceDimensions", dims);
        $_state.types.push_back(resultType);
      }]>
  ];

  let extraClassDeclaration = [{
    /// Returns the type of the base memref operand.
    MemRefType getSourceMemRefType() {
      return source().getType().cast<MemRefType>();
    }

    /// The result of a slice is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];

  // For now, the folder is disabled
  // let hasFolder = 1;

  let hasCanonicalizer = 1;
}

def accv_MergeDimOp : accv_Op<"merge_dim", [NoSideEffect]> {
  let summary = "memref merge dimensions operation";
  let description = [{
    The `accv.merge_dim` operation converts a memref type to another memref type
    which represents an reduced-rank view of the original memref by merging
    the given dimensions.
  }];

  let arguments = (ins
    AnyMemRef:$source,
    I64Attr:$dim1,
    I64Attr:$dim2
  );
  let results = (outs AnyMemRef:$result);

  let extraClassDeclaration = [{
    /// Returns the type of the source memref operand.
    MemRefType getSourceMemRefType() {
      return source().getType().cast<MemRefType>();
    }

    /// The result of a split_dim is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];
}

def accv_SplitDimOp : accv_Op<"split_dim", [NoSideEffect]> {
  let summary = "memref split dimension operation";
  let description = [{
    The `accv.split_dim` operation converts a memref type to another memref type
    which represents an expanded-rank view of the original memref by splitting
    the given dimension.
  }];

  let arguments = (ins
    AnyMemRef:$source,
    I64Attr:$dim,
    I64Attr:$size
  );
  let results = (outs AnyMemRef:$result);

  let extraClassDeclaration = [{
    /// Returns the type of the base memref operand.
    MemRefType getSourceMemRefType() {
      return source().getType().cast<MemRefType>();
    }

    /// The result of a split_dim is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];
}

def accv_StoreOp : accv_Op<"store",
    [AllElementTypesMatch<["memref", "value"]>]> {
  let summary = "store operation";
  let description = [{
    The `accv.store` op writes an element to a memref specified by an index list.
    The arity of indices is the rank of the memref (i.e. if the memref being
    stored to is of rank 3, then 3 indices are required for the store following
    the memref identifier). The store operation does not produce a result.

    In the following example, the ssa value '%v' is stored in memref '%A' at
    indices [%i, %j]:

      ```mlir
      store %v, %A[%i, %j] : memref<4x128xf32, (d0, d1) -> (d0, d1), 0>
      ```
  }];

  let arguments = (ins
                   acc_NumericType:$value,
                   AnyMemRef:$memref,
                   Variadic<acc_Indexlike>:$indices);

  let builders = [OpBuilder<(ins "Value":$valueToStore, "Value":$memref), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
  }]>];

  let extraClassDeclaration = [{
      Value getValueToStore() { return getOperand(0); }

      Value getMemRef() { return getOperand(1); }
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return getMemRef().getType().cast<MemRefType>();
      }

      operand_range getIndices() {
        return {operand_begin() + 2, operand_end()};
      }
  }];
}

def accv_StoreToMemRefOp : accv_Op<"store_to_memref", [NoSideEffect]> {
  let summary = "store element in memref";
  let description =[{
    Stores the element at index 0 of a Scalar sizes MemRef
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs acc_MemRefWithShape<[1]>:$result);

  let builders = [OpBuilder<(ins "Value":$value), [{
      auto elementType = value.getType();
      $_state.addOperands(value);
      $_state.types.push_back(MemRefType::get({1}, elementType));
  }]>];
}

def accv_UnsafeOffsetOp : accv_Op<"unsafe_offset"> {
  let summary = "";
  let description = "";
  let arguments = (ins AnyMemRef:$input, acc_Indexlike:$offset);
  let results = (outs AnyMemRef:$result);
}

def accv_ReshapeOp : accv_Op<"reshape"> {
  let summary = "";
  let description = "";
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);

  let extraClassDeclaration = [{
    /// Returns the type of the base memref operand.
    MemRefType getSourceMemRefType() {
      return source().getType().cast<MemRefType>();
    }

    /// The result of a split_dim is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];
}

def accv_ReorderOp : accv_Op<"reorder"> {
  let summary = "";
  let description = "";
  let arguments = (ins
    AnyMemRef:$source,
    I64ArrayAttr:$order
  );
  let results = (outs AnyMemRef:$result);

  let builders = [
    OpBuilder<(ins "Value":$source, "ArrayRef<int64_t>":$order), [{
        auto orderAttr = $_builder.getI64ArrayAttr(order);
        build($_builder, $_state, source, orderAttr);
      }]>,
    OpBuilder<(ins "Value":$source, "ArrayAttr":$order)>];

  let extraClassDeclaration = [{
    /// Returns the type of the base memref operand.
    MemRefType getSourceMemRefType() {
      return source().getType().cast<MemRefType>();
    }

    /// The result of a split_dim is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];
}

def accv_YieldOp : accv_Op<"yield", [Terminator]> {
  let summary = "loop yield and termination operation";
  let description = [{
    The `accv.yield` op yields an SSA value from a loop dialect op region and
    terminates the regions. The semantics of how the values are yielded
    is defined by the parent operation.
    If `accv.yield` has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "accv.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values
    are yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [
    OpBuilder<(ins),
              [{ /* nothing to do */ }]>
  ];
}

def accv_EarlyReturnOp : accv_Op<"early_return", [MemRefsNormalizable,]> {
  let summary = "early return operation";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]
  >];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def accv_ReturnOp : accv_Op<"return", [
        NoSideEffect, MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "return operation";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]
  >];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

// Note: only works on vectors (1-D memrefs) currently
def accv_ReduceOp : accv_Op<"reduce",
      [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Reduction operation";
  let description = "";
  let arguments = (ins Type<Or<[AnyStridedMemRefOfRank<1>.predicate, AnyType.predicate]>>:$input, AnyType:$initArg);
  let regions = (region SizedRegion<1>:$region);
  let results = (outs AnyType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$initArg, CArg<"llvm::function_ref<void(OpBuilder &, Location, Value, Value)>", "nullptr">:$func)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        llvm::function_ref<void(OpBuilder &, Location, Value, Value)>;
    Block *getBody() { return &region().front(); }
    Value getInput() { return input(); }
    Value getInitialValue() { return initArg(); }
    Value getInputValueVar() { return getBody()->getArgument(0); } // 'a' --- the current element of input
    Value getInductionValue() { return getBody()->getArgument(1); } // 'p' --- the value carried over from the previous iteration
    ShapedType getShapedType() {
      return getInput().getType().cast<ShapedType>();
    }
    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
  }];
}

// Note: only works on vectors (1-D memrefs) currently
def accv_MapReduceOp : accv_Op<"map_reduce",
      [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Map-reduce operation";
  let description = "";
  let arguments = (ins AnyStridedMemRefOfRank<1>:$input, AnyType:$initArg);
  let regions = (region SizedRegion<1>:$mapRegion, SizedRegion<1>:$reduceRegion);
  let results = (outs AnyType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
              "Value":$input,
              "Value":$initArg,
              CArg<"llvm::function_ref<void(OpBuilder &, Location, Value)>", "nullptr">:$mapBodyBuilder,
              CArg<"llvm::function_ref<void(OpBuilder &, Location, Value, Value)>", "nullptr">:$reduceBodyBuilder)>
  ];

  let extraClassDeclaration = [{
    using MapBodyBuilderFn =
        llvm::function_ref<void(OpBuilder &, Location, Value)>;
    using ReduceBodyBuilderFn =
        llvm::function_ref<void(OpBuilder &, Location, Value, Value)>;

    Block *getMapBody() { return &mapRegion().front(); }
    OpBuilder getMapBodyBuilder() {
      return OpBuilder(getMapBody(), std::prev(getMapBody()->end()));
    }
    Block *getReduceBody() { return &reduceRegion().front(); }
    OpBuilder getReduceBodyBuilder() {
      return OpBuilder(getReduceBody(), std::prev(getReduceBody()->end()));
    }

    Value getInput() { return input(); }
    Value getInitialValue() { return initArg(); }
    Value getMapInputValueVar() { return getMapBody()->getArgument(0); } // 'a' --- the current element of input
    Value getReduceInputValueVar() { return getReduceBody()->getArgument(0); } // 'a' --- the current element of input
    Value getReduceInductionValue() { return getReduceBody()->getArgument(1); } // 'p' --- the value carried over from the previous iteration

    ShapedType getShapedType() {
      return getInput().getType().cast<ShapedType>();
    }
  }];
}

def accv_ReduceMaxOp : accv_Op<"reduce_max"> {
  let summary = "";
  let description = "";
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyType:$result);
}

def accv_ReduceSumOp : accv_Op<"reduce_sum"> {
  let summary = "";
  let description = "";
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyType:$result);
}

def accv_BarrierOp : accv_Op<"barrier"> {
  let summary = "Block synchronization primitive.";
  let hasCanonicalizer = 1;
  let arguments = (ins BarrierScopeAttr:$scope);
}

def accv_GetTimeOp : accv_Op<"gettime"> {
  let summary = "Get current clock time";
  let results = (outs AnyFloat:$result);
  let builders = [
    OpBuilder<(ins), [{
        build($_builder, $_state, $_builder.getF64Type());
    }]>];
}

def accv_EnterProfileRegionOp : accv_Op<"enter_profile"> {
  let summary = "Enter a profile region";
  let arguments = (ins StrAttr:$regionName);
}

def accv_ExitProfileRegionOp : accv_Op<"exit_profile"> {
  let summary = "Exit a profile region";
  let arguments = (ins StrAttr:$regionName);
}

def accv_PrintProfileResultsOp : accv_Op<"print_profile"> {
  let summary = "Print out a summary of the profile counters";
}


// matrix-fuse-multiply-add

def accv_MMAComputeSyncOp: accv_Op<"wmma_compute_sync", [NoSideEffect, AllTypesMatch<["opC", "result"]>]> {
  let summary = "MFMA Compute";
  let description = [{
    The `accv.wmma_compute_sync` op is an abstraction of TensorCore ops (currently GPU centric).
    cbsz: Defines the number of blocks that can do a broadcast within a group. Legal values = 0-4. The block ID of this group comes from ABID.
    abid: Block ID of block to broadcast during matrix multiply (MFMA ops).
    blgp: “B”-Matrix Lane-Group Pattern. Controls how to swizzle the matrix lane groups (LG) in VGPRs when doing matrix multiplication by controlling the swizzle muxes.
    Documentation: https://developer.amd.com/wp-content/resources/CDNA1_Shader_ISA_14December2020.pdf (Pg. 250)
  }];

  let arguments = (ins
                  Arg<MemRefRankOf<[I8, F16, F32], [1]>>:$opA,
                  Arg<MemRefRankOf<[I8, F16, F32], [1]>>:$opB,
                  Arg<MemRefRankOf<[I32, F32], [1]>>:$opC,
                  I32Attr:$mmaShapeType,
                  I32Attr:$cbsz,
                  I32Attr:$abid,
                  I32Attr:$blgp);

  let results = (outs MemRefRankOf<[I32, F32], [1]>:$result);

  let assemblyFormat = [{
    $opA `,` $opB`,` $opC `,` $cbsz `,` $abid `,` $blgp attr-dict `:` type($opA) `,` type($opB) `,` type($opC) `->` type($result) `[` $mmaShapeType `]`
  }];

  let verifier = [{ return ::verify(*this); }];
}


def accv_MMAFillSyncOp : accv_Op<"wmma_fill_sync",
  [
    NoSideEffect,
    TypesMatchWith<"value type matches element type of mfma_matrix",
                    "result", "value",
                    "$_self.cast<MemRefType>().getElementType()">
  ]>{

  let summary = "creates a constant MFMA matrix";

  let description = [{
    The `accv.wmma_fill_sync` creates a constant MFMA matrix that can be subsequently
    used within an accv.wmma_compute_sync operation.

    This operation takes a constant as its operand. The op returns a memref.
  }];

  let arguments = (ins
    AnyTypeOf<[I8, I32, F16, F32]>:$value,
    I32Attr:$mmaShapeType
  );

  let results = (outs MemRefRankOf<[I8, I32, F16, F32], [1]>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result) `[` $mmaShapeType `]`
  }]; 
  let verifier = [{ return ::verify(*this); }];
}

// load and store submatrix

def accv_MMALoadSyncOp : accv_Op<"wmma_load_sync",
  [
    MemoryEffects<[MemRead]>,
    DeclareOpInterfaceMethods<AffineReadOpInterface>,
    DeclareOpInterfaceMethods<AffineMapAccessInterface>
  ]>{

  let summary = "matrix load";

  let description = [{
    The `accv.wmma_load_sync` operation loads a sub-matrix to be used by the accv.wmma_compute_sync operation.

    This operation takes a memref as its first operand: it is the source matrix
    from which data is to be loaded. The op returns a memref.
  }];

  let arguments = (ins
    Arg<MemRefRankOf<[I8, I32, F16, F32], [2]>, "", [MemRead]>:$memref,
    UI32Attr:$mmaShapeType,
    I8Attr:$operandType,
    Variadic<acc_Indexlike>:$indices,
    AffineMapAttr:$map
  );

  let results = (outs MemRefRankOf<[I8, I32, F16, F32], [1]>:$result);

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$memref, "MMAShape":$mmaShapeType, "MMAOperandType":$operandType, "AffineMap":$map, "ValueRange":$mapOperands), [{
        assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
        $_state.addOperands(memref);
        $_state.addOperands(mapOperands);
        $_state.addAttribute("mmaShapeType", $_builder.getUI32IntegerAttr((uint32_t)mmaShapeType));
        $_state.addAttribute("operandType", $_builder.getI8IntegerAttr((int8_t)operandType));
        $_state.addAttribute(getMapAttrName(), AffineMapAttr::get(map));
        $_state.addTypes(resultType);
    }]>,
    OpBuilder<(ins "Type":$resultType, "Value":$memref, "MMAShape":$mmaShapeType, "MMAOperandType":$operandType, "ValueRange":$indices), [{
        auto memrefType = memref.getType().cast<MemRefType>();
        int64_t rank = memrefType.getRank();
        // Create identity map for memrefs with at least one dimension or () -> ()
        // for zero-dimensional memrefs.
        auto map =
            rank ? $_builder.getMultiDimIdentityMap(rank) : $_builder.getEmptyAffineMap();
        return build($_builder, $_state, resultType, memref, mmaShapeType, operandType, map, indices);
    }]>
  ];

  code extraClassDeclaration = [{
    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return (*this)->getAttr(getMapAttrName()).cast<AffineMapAttr>();
    }

    static StringRef getMapAttrName() { return "map"; }
  }];

  let assemblyFormat = [{
    $memref `[` $indices `]` attr-dict `:` type($memref) `[` type($indices) `]` `->` type($result) `[` $mmaShapeType `]` `[` $operandType `]`
  }]; 
  let verifier = [{ return ::verify(*this); }];
}

def accv_MMAStoreSyncOp : accv_Op<"wmma_store_sync", [
    MemoryEffects<[MemWrite]>,
    DeclareOpInterfaceMethods<AffineWriteOpInterface>,
    DeclareOpInterfaceMethods<AffineMapAccessInterface>
  ]>{

  let summary = "matrix store";

  let description = [{
    The `accv.wmma_store_sync` operation stores an mfma matrix into a matrix.

    This operation takes a source memref containing the data to be stored into the
    destination memref which can be in global or shared memory.
  }];

  let arguments = (ins 
        Arg<MemRefRankOf<[I32, F32], [1]>>:$src,
        Arg<MemRefRankOf<[I8, I32, F16, F32], [2]>, "",[MemWrite]>:$memref,
        UI32Attr:$mmaShapeType,
        Variadic<acc_Indexlike>:$indices,
        AffineMapAttr:$map);

  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$memref, "MMAShape":$mmaShapeType, "AffineMap":$map, "ValueRange":$mapOperands), [{
        assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
        $_state.addOperands(src);
        $_state.addOperands(memref);
        $_state.addOperands(mapOperands);
        $_state.addAttribute("mmaShapeType", $_builder.getUI32IntegerAttr((uint32_t)mmaShapeType));
        $_state.addAttribute(getMapAttrName(), AffineMapAttr::get(map));
    }]>,
    OpBuilder<(ins "Value":$src, "Value":$memref, "MMAShape":$mmaShapeType, "ValueRange":$indices), [{
        auto memrefType = memref.getType().cast<MemRefType>();
        int64_t rank = memrefType.getRank();
        // Create identity map for memrefs with at least one dimension or () -> ()
        // for zero-dimensional memrefs.
        auto map =
            rank ? $_builder.getMultiDimIdentityMap(rank) : $_builder.getEmptyAffineMap();
        return build($_builder, $_state, src, memref, mmaShapeType, map, indices);
    }]>
  ];

  code extraClassDeclaration = [{
    /// Returns the operand index of the value to be stored.
    unsigned getStoredValOperandIndex() { return 0; }

    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 1; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return (*this)->getAttr(getMapAttrName()).cast<AffineMapAttr>();
    }

    static StringRef getMapAttrName() { return "map"; }
  }];

  let assemblyFormat = [{
    $src`,` $memref `[` $indices `]` attr-dict `:` type($src)`,` type($memref) `[` type($indices) `]` `[` $mmaShapeType `]`
  }];

  let verifier = [{ return ::verify(*this); }];
}

#endif // ACCERA_accv_OPS
