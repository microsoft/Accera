////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ACCERA_rcv_OPS
#define ACCERA_rcv_OPS

include "ir/include/value/ValueBase.td"

include "ir/include/value/ValueAttrs.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"

def rcv_ValueLambdaOp : rcv_Op<"lambda", [
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

def rcv_ValueModuleOp : rcv_Op<"module", [
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

def rcv_ModuleTerminatorOp : rcv_Op<"module_terminator", [Terminator, HasParent<"ValueModuleOp">]> {
  let summary = "A pseudo op that marks the end of a gpu.module.";
  let description = [{}];
}

def rcv_PrintOp : rcv_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The `accv.print` operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyTensor, rc_NumericType]>:$input,
    UnitAttr:$to_stderr
  );
}

def rcv_PrintFOp : rcv_Op<"printf"> {
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

def rcv_GlobalOp : rcv_Op<"global", [IsolatedFromAbove, Symbol]> {
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

def rcv_ReferenceGlobalOp : rcv_Op<"ref_global"> {
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

def rcv_AllocOp : rcv_Op<"alloc"> {
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


def rcv_BitcastOp : rcv_Op<"bitcast_op",
  [NoSideEffect]> {
    let summary = "bitcast operation";
    let description = [{
      The `accv.bitcast_op` operation reinterprets the bits of a value as a value of a different type. The source and destination types must have the same number of bits.
    }];

    let arguments = (ins
      rc_ScalarOrVectorNumericType:$input
    );
    let results = (outs rc_ScalarOrVectorNumericType:$result);
}


def rcv_UNARY_OP_NOT : I64EnumAttrCase<"NOT", 0>;

def rcv_UnaryOpPredicateAttr : I64EnumAttr<
  "UnaryOpPredicate", "",
  [rcv_UNARY_OP_NOT]> {
  let cppNamespace = "::accera::ir::value";
}

def rcv_UnaryOp : rcv_Op<"unary_op",
  [NoSideEffect]> {
    let summary = "unary operation";
    let description = [{
      The `accv.unary_op` operations performs the unary operation specified by
      the predicate. The result and operands must have the same shape and type.
    }];

    let arguments = (ins
      rcv_UnaryOpPredicateAttr:$predicate,
      rc_NumericType:$input
    );
    let results = (outs rc_NumericType:$result);

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

def rcv_BINARY_OP_ADD : I64EnumAttrCase<"ADD", 0>;
def rcv_BINARY_OP_SUB : I64EnumAttrCase<"SUB", 1>;
def rcv_BINARY_OP_MUL : I64EnumAttrCase<"MUL", 2>;
def rcv_BINARY_OP_DIV : I64EnumAttrCase<"DIV", 3>;
def rcv_BINARY_OP_MOD : I64EnumAttrCase<"MOD", 4>;
def rcv_BINARY_OP_AND : I64EnumAttrCase<"LOGICAL_AND", 5>;
def rcv_BINARY_OP_OR : I64EnumAttrCase<"LOGICAL_OR", 6>;

def rcv_BinaryOpPredicateAttr : I64EnumAttr<
  "BinaryOpPredicate", "",
  [rcv_BINARY_OP_ADD, rcv_BINARY_OP_SUB, rcv_BINARY_OP_MUL, rcv_BINARY_OP_DIV, rcv_BINARY_OP_MOD,
  rcv_BINARY_OP_AND, rcv_BINARY_OP_OR]> {
  let cppNamespace = "::accera::ir::value";
}

def rcv_BinOp : rcv_Op<"bin_op",
  [NoSideEffect]> {
    let summary = "binary operation";
    let description = [{
      The `accv.bin_op` operations performs the binary operation specified by the
      predicate. The result and operands must have the same shape and type.
    }];

    let arguments = (ins
      rcv_BinaryOpPredicateAttr:$predicate,
      rc_ScalarOrVectorNumericType:$lhs,
      rc_ScalarOrVectorNumericType:$rhs
    );
    let results = (outs rc_ScalarOrVectorNumericType:$result);

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

def rcv_CMP_P_EQ : I64EnumAttrCase<"EQ", 0>;
def rcv_CMP_P_NE : I64EnumAttrCase<"NE", 1>;
def rcv_CMP_P_LT : I64EnumAttrCase<"LT", 2>;
def rcv_CMP_P_LE : I64EnumAttrCase<"LE", 3>;
def rcv_CMP_P_GT : I64EnumAttrCase<"GT", 4>;
def rcv_CMP_P_GE : I64EnumAttrCase<"GE", 5>;

def rcv_CmpOpPredicateAttr : I64EnumAttr<
    "CmpOpPredicate", "",
    [rcv_CMP_P_EQ, rcv_CMP_P_NE, rcv_CMP_P_LT, rcv_CMP_P_LE, rcv_CMP_P_GT, rcv_CMP_P_GE]> {
  let cppNamespace = "::accera::ir::value";
}

def rcv_CmpOp : rcv_Op<"cmp",
    [NoSideEffect]> {
  let summary = "comparison operation";
  let description = [{
    The `accv.cmp` operation compares its two operands according to the
    predicate specified. The predicate defines the type of comparison:
    (in)equality, less/greater than (or equal to).  The
    operands must have the same type. The result is an i1 or a vector of i1.
  }];

  let arguments = (ins
      rcv_CmpOpPredicateAttr:$predicate,
      rc_ScalarOrVectorNumericType:$lhs,
      rc_ScalarOrVectorNumericType:$rhs
  );
  let results = (outs rc_ScalarOrVectorBoolType:$result);

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

def rcv_CopyOp : rcv_Op<"copy"> {
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

def rcv_IfOp : rcv_Op<"if",
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
  let arguments = (ins rc_Scalarlike<rc_BoolType>:$condition);
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

def rcv_LoadOp : rcv_Op<"load",
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
    Variadic<rc_Indexlike>:$indices);
  let results = (outs rc_NumericType:$result);

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

def rcv_GetElementOp : rcv_Op<"get_element", [NoSideEffect]> {
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

def rcv_LaunchFuncOp : rcv_Op<"launch_func", [CallOpInterface]> {
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

def rcv_CallOp : rcv_Op<"call", [CallOpInterface]> {
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

def rcv_MemRefCastOp : rcv_Op<"memref_cast", [SameOperandsAndResultShape]> {
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

def rcv_OffsetOp : rcv_Op<"offset", [NoSideEffect]> {
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
    Variadic<rc_Indexlike>:$offsets
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

def rcv_ViewOp : rcv_Op<"view", [NoSideEffect]> {
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

def rcv_SliceOp : rcv_Op<"slice", [NoSideEffect]> {
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
    Variadic<rc_Indexlike>:$offsets
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

def rcv_MergeDimOp : rcv_Op<"merge_dim", [NoSideEffect]> {
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

def rcv_SplitDimOp : rcv_Op<"split_dim", [NoSideEffect]> {
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

def rcv_StoreOp : rcv_Op<"store",
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
                   rc_NumericType:$value,
                   AnyMemRef:$memref,
                   Variadic<rc_Indexlike>:$indices);

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

def rcv_StoreToMemRefOp : rcv_Op<"store_to_memref", [NoSideEffect]> {
  let summary = "store element in memref";
  let description =[{
    Stores the element at index 0 of a Scalar sizes MemRef
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs rc_MemRefWithShape<[1]>:$result);

  let builders = [OpBuilder<(ins "Value":$value), [{
      auto elementType = value.getType();
      $_state.addOperands(value);
      $_state.types.push_back(MemRefType::get({1}, elementType));
  }]>];
}

def rcv_UnsafeOffsetOp : rcv_Op<"unsafe_offset"> {
  let summary = "";
  let description = "";
  let arguments = (ins AnyMemRef:$input, rc_Indexlike:$offset);
  let results = (outs AnyMemRef:$result);
}

def rcv_ReshapeOp : rcv_Op<"reshape"> {
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

def rcv_ReorderOp : rcv_Op<"reorder"> {
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

def rcv_YieldOp : rcv_Op<"yield", [Terminator]> {
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

def rcv_EarlyReturnOp : rcv_Op<"early_return", [MemRefsNormalizable,]> {
  let summary = "early return operation";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]
  >];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def rcv_ReturnOp : rcv_Op<"return", [
        NoSideEffect, MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "return operation";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]
  >];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

// Note: only works on vectors (1-D memrefs) currently
def rcv_ReduceOp : rcv_Op<"reduce",
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
def rcv_MapReduceOp : rcv_Op<"map_reduce",
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

def rcv_ReduceMaxOp : rcv_Op<"reduce_max"> {
  let summary = "";
  let description = "";
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyType:$result);
}

def rcv_ReduceSumOp : rcv_Op<"reduce_sum"> {
  let summary = "";
  let description = "";
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyType:$result);
}

def rcv_BarrierOp : rcv_Op<"barrier"> {
  let summary = "Block synchronization primitive.";
  let hasCanonicalizer = 1;
  let arguments = (ins BarrierScopeAttr:$scope);
}

def rcv_GetTimeOp : rcv_Op<"gettime"> {
  let summary = "Get current clock time";
  let results = (outs AnyFloat:$result);
  let builders = [
    OpBuilder<(ins), [{
        build($_builder, $_state, $_builder.getF64Type());
    }]>];
}

def rcv_EnterProfileRegionOp : rcv_Op<"enter_profile"> {
  let summary = "Enter a profile region";
  let arguments = (ins StrAttr:$regionName);
}

def rcv_ExitProfileRegionOp : rcv_Op<"exit_profile"> {
  let summary = "Exit a profile region";
  let arguments = (ins StrAttr:$regionName);
}

def rcv_PrintProfileResultsOp : rcv_Op<"print_profile"> {
  let summary = "Print out a summary of the profile counters";
}


// matrix-fuse-multiply-add

// Predicate to check if type is rcv::MMAMatrixType.
def IsMMFAMatrixTypePred : CPred<"$_self.isa<accera::ir::value::MFMAMatrixType>()">;

def rcv_MFMAMatrix : DialectType<Value_Dialect,
                                        IsMMFAMatrixTypePred,
                                        "MFMAMatrix type">;
class MFMAMatrixOf<list<Type> allowedTypes> :
  ContainerType<AnyTypeOf<allowedTypes>, IsMMFAMatrixTypePred,
  "$_self.cast<::accera::ir::value::MFMAMatrixType>().getElementType()",
  "rcv.mfma_matrix", "::accera::ir::value::MFMAMatrixType">;

def rcv_MFMAComputeOp: rcv_Op<"mfma_compute", [NoSideEffect, AllTypesMatch<["opC", "res"]>]> {
  let summary = "MFMA Compute";
  let description = [{
    The `rcv.mfma_compute` op is an abstraction of TensorCore ops (currently GPU centric).
  }];

  let arguments = (ins Arg<MFMAMatrixOf<[F16, F32]>>:$opA,
                  Arg<MFMAMatrixOf<[F16, F32]>>:$opB,
                  Arg<MFMAMatrixOf<[F16, F32]>>:$opC);

  let results = (outs rcv_MFMAMatrix:$res);

  let assemblyFormat = [{
    $opA`,` $opB`,` $opC attr-dict `:` type($opA)`,` type($opB) `->` type($res)
  }];

  let verifier = [{ return ::verify(*this); }];
}

// load and store submatrix

def rcv_MFMALoadMatrixOp : rcv_Op<"mfma_load_matrix", [MemoryEffects<[MemRead]>]>{

  let summary = "matrix load";

  let description = [{
    The `rcv.mfma_load_matrix` operation loads a sub-matrix to be used by the rcv.mfma_compute operation.

    This operation takes a memref as its first operand: it is the source matrix
    from which data is to be loaded. The op returns a `!rcv.mfma_matrix`.
  }];

  let arguments = (ins Arg<MemRefOf<[F16, F32]>, "", [MemRead]>:$srcMemref);

  let results = (outs rcv_MFMAMatrix:$res);

  let assemblyFormat = [{
    $srcMemref attr-dict `:` type($srcMemref) `->` type($res)
  }];

  let verifier = [{ return ::verify(*this); }];
}

def rcv_MFMAStoreMatrixOp : rcv_Op<"subgroup_mma_store_matrix",
    [MemoryEffects<[MemWrite]>]>{

  let summary = "matrix store";

  let description = [{
    The `rcv.mfma_load_matrix` operation loads a sub-matrix.

    This operation takes a `rcv.mfma_matrix` and a memref as operands.
    `!rcv.mma_matrix` is the source value containing the data to be stored into the
    destination memref which can be in global or shared memory.
  }];

  let arguments = (ins Arg<MFMAMatrixOf<[F16, F32]>>:$src,
                  Arg<MemRefOf<[F16, F32]>, "",[MemWrite]>:$dstMemref);

  let assemblyFormat = [{
    $src`,` $dstMemref attr-dict `:` type($src)`,` type($dstMemref)
  }];

  let verifier = [{ return ::verify(*this); }];
}

#endif // ACCERA_rcv_OPS
