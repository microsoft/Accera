////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LOOPNEST_OPS
#define LOOPNEST_OPS

include "ir/include/Common.td"
include "ir/include/nest/LoopNestBase.td"
include "ir/include/nest/LoopNestInterfaces.td"
include "ir/include/nest/LoopNestAttrs.td"
include "ir/include/value/ValueAttrs.td"

//
// Ops
//

// Base class for loopnest dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class rcln_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<LoopNest_Dialect, mnemonic, traits>;

//
// SymbolicIndexOp
//
def rcln_SymbolicIndexOp : rcln_Op<"sym_index", [NoSideEffect, ConstantLike]> {
  let summary = "symbolic loop index";
  let description = [{
    The "accln.sym_index" builtin operation creates a symbolic "placeholder" index for a loop.
  }];

  let arguments = (ins rcln_IndexAttr:$index);
  let results = (outs Index:$result);

  let builders = [
    OpBuilder<(ins "std::string":$name, "int":$id)>,
    OpBuilder<(ins "Index":$index)>
  ];

  let extraClassDeclaration = [{
    static StringRef getIndexAttrName() { return "index"; }

    Index getValue();
  }];
  // Invoke a static verify method to verify this operation.
  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;

  let assemblyFormat = [{
    attr-dict $index
  }];
}

//
// ScheduleOp
//
def rcln_ScheduleOp : rcln_Op<"schedule",
       [SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "loopnest scheduling operation";
  let description = [{
    The "accln.schedule" operation describes a schedule for a loop nest.

    The "accln.schedule" operation has a single region indicating the code at the middle of the loop nest.
  }];

  // TODO: add other attributes?
  let arguments = (ins ArrayAttr:$kernels, Variadic<AnyType>:$rangeOperands);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "NestOp":$nest)>,
  ];

  let extraClassDeclaration = [{

    // Names for attributes
    static StringRef getOrderAttrName() { return "order"; }
    static StringRef getDomainAttrName() { return "domain"; }
    static StringRef getUnrolledIndicesAttrName() { return "unrolled"; }
    static StringRef getSaturatedFlagIndicesAttrName() { return "saturated"; }
    static StringRef getUnrollAndJammedIndicesAttrName() { return "unroll_and_jammed"; }
    static StringRef getKernelsAttrName() { return "kernels"; }
    static StringRef getLoopAttrsName() { return "loopattrs"; }
    static StringRef getIndexAttrKeyName() { return "scheduledIndex"; }
    static StringRef getFusedDomainsAttrName() { return "fused_domains"; }
    static StringRef getFusedIndicesAttrName() { return "fused_indices";  }

    size_t numDimensions();
    size_t numLoops();

    void setOrder(ArrayRef<Index> order);
    void setOrder(ArrayRef<SymbolicIndexOp> order);
    std::vector<Index> getOrder();

    void unroll(Index index, bool unroll=true, std::optional<uint64_t> size=std::nullopt);
    std::optional<uint64_t> getUnrollIfRangeSmallerThan(Index index);

    void setSaturatedFlag(Index index, bool saturated=true);
    bool isSaturated(Index index);

    void unrollAndJam(Index index, uint64_t factor);
    std::optional<uint64_t> getUnrollAndJamFactor(Index index);

    Index pad(Index index, int size, bool padFront);
    SymbolicIndexOp pad(SymbolicIndexOp index, int size, bool padFront);

    Index skew(Index index, Index reference);
    SymbolicIndexOp skew(SymbolicIndexOp index, SymbolicIndexOp reference);

    SplitIndex split(Index index, int splitSize);
    SplitIndexT<SymbolicIndexOp> split(SymbolicIndexOp index, int splitSize);

    TransformedDomainAttr getDomain();
    void setDomain(const TransformedDomain& domain); // for internal use only

    std::vector<IterationDomain> getFusedDomains();
    void setFusedDomains(ArrayRef<IterationDomain> domains);

    std::vector<Index> getFusedIndices(Index targetIndex);
    void setFusedIndices(Index targetIndex, ArrayRef<Index> fusedIndices);

    std::vector<SymbolicIndexOp> getIndices(OpBuilder& builder);

    template <int N>
    std::array<SymbolicIndexOp, N> getIndices(OpBuilder& builder);

    std::vector<std::string> getKernelIds();
    std::vector<ScheduledKernelOp> getKernels();
    ScheduledKernelOp getKernel(llvm::StringRef id);

    void addKernel(StringRef kernelId);
    void addKernel(KernelOp kernel);
    void addKernel(ScheduledKernelOp kernel);

    std::vector<InjectableMapping> getInjectableMappings();
    void injectMapping(mlir::Operation* mappingOp);

    std::vector<AffineValueMap> getIntermediateCompositeIndexMaps(mlir::OpBuilder& builder);

    AffineValueMap getCompositeIndexMap(mlir::OpBuilder& builder);

    SymbolicIndexOp getOrCreateSymbolicIndex(OpBuilder& builder, Index index);

    AffineMap getGlobalDimensionsToInputDimensionsMap(const std::vector<Index>& dims);

    void addLoopAttribute(Index targetIndex, mlir::Identifier name, mlir::Attribute value);
    std::optional<mlir::DictionaryAttr> getLoopAttributes(Index targetIndex);

    bool hasConstantRanges();

    NestOp getNest();
    ExecPlanOp getOrCreateExecPlan();

    OpBuilder getBodyBuilder() {
        return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
}];

  // Invoke a static verify method to verify this operation.
  let verifier = [{ return ::verify(*this); }];
}

//
// NestOp
//
def rcln_NestOp : rcln_Op<"nest",[SingleBlockImplicitTerminator<"TerminatorOp">, SymbolTable]> {
  let summary = "nest operation";
  let description = [{
    The "accln.nest" operation produces a loop nest. Takes a variadic number of values indicating the upper bound for the loops.

    The "accln.nest" operation has a single region indicating the code at the middle of the loop nest.
  }];

  let arguments = (ins SymbolRefArrayAttr:$kernels, rcln_IterationDomainAttr:$domain, OptionalAttr<ExecutionTargetAttr>:$exec_target, Variadic<AnyType>:$rangeOperands);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$loopRanges)>,
    OpBuilder<(ins "ArrayRef<mlir::Value>":$loopRanges)>,
    OpBuilder<(ins "const IterationDomain&":$domain)>
  ];

  let extraClassDeclaration = [{
    size_t numDimensions();

    SymbolicIndexOp getOrCreateSymbolicIndex(OpBuilder& builder, Index index); // internal use only

    std::vector<SymbolicIndexOp> getIndices(OpBuilder& builder);

    template <int N>
    std::array<SymbolicIndexOp, N> getIndices(OpBuilder& builder);

    IterationDomainAttr getDomain();

    std::vector<std::string> getKernelIds();
    std::vector<KernelOp> getKernels();
    KernelOp getKernel(llvm::StringRef id);

    std::vector<int64_t> getLoopRanges();

    ScheduleOp getOrCreateSchedule();

    Block *getBody() { return &body().front(); }

    OpBuilder getBodyBuilder();
  }];

  // Invoke a static verify method to verify this operation.
  let verifier = [{ return ::verify(*this); }];
}

//
// ScheduledLoopOp
//
def rcln_ScheduledLoopOp : rcln_Op<"scheduled_loop",
  [SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "Represents a scheduled loop and separates prologue, body, and epilogue regions";
  let description = [{
    The "accln.ScheduledLoopOp" operation represents a scheduled loop level and separates prolgoue, body, and epilogue regions
  }];

  let arguments = (ins I64Attr:$begin, I64Attr:$end, I64Attr:$step, rcln_IndexAttr:$index, Index:$symbolicIndex, ArrayAttr:$subdomainSize, ArrayAttr:$subdomainIndexOrder);
  let regions = (region AnyRegion:$prologue, AnyRegion:$body, AnyRegion:$epilogue);

  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parseScheduledLoopOp(parser, result); }];

  let builders = [
    OpBuilder<(ins
      "int64_t":$begin,
      "int64_t":$end,
      "int64_t":$step,
      "Value":$symbolicIndex,
      "const std::vector<int64_t>&":$subdomainSize,
      "const std::vector<Index>&":$subdomainIndexOrder)>
  ];

  let extraClassDeclaration = [{
    int64_t getNumIterations();
    Block* getPrologue() {
      return &prologue().front();
    }
    Block* getBody() {
      return &body().front();
    }
    Block* getEpilogue() {
      return &epilogue().front();
    }
    OpBuilder getPrologueBuilder() {
      return OpBuilder(getPrologue(), std::prev(getPrologue()->end()));
    }
    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    OpBuilder getEpilogueBuilder() {
      return OpBuilder(getEpilogue(), std::prev(getEpilogue()->end()));
    }
    mlir::BlockArgument getPrologueArgPlaceholder() {
      return getPrologue()->getArgument(0);
    }
    mlir::BlockArgument getBodyArgPlaceholder() {
      return getBody()->getArgument(0);
    }
    mlir::BlockArgument getEpilogueArgPlaceholder() {
      return getEpilogue()->getArgument(0);
    }
    SymbolicIndexOp getSymbolicIndex();
    void setBegin(int64_t begin)
    {
      mlir::Builder builder(getContext());
      (*this)->setAttr("begin", builder.getI64IntegerAttr(begin));
    }
    void setEnd(int64_t end)
    {
      mlir::Builder builder(getContext());
      (*this)->setAttr("end", builder.getI64IntegerAttr(end));
    }
    void setStep(int64_t step)
    {
      mlir::Builder builder(getContext());
      (*this)->setAttr("step", builder.getI64IntegerAttr(step));
    }
    std::vector<int64_t> getSubdomainSize();
    void setSubdomainSize(const std::vector<int64_t>&);
    std::vector<Index> getSubdomainIndexOrder();
  }];
}


//
// KernelOp
//

//
// %n = nest(M, N, K)
// %i = symbolic_index("i", 1)
// %j = symbolic_index("j", 2)
// %k = symbolic_index("k", 3)
// %kernel = kernel {
//   C(i,j) += A(i,k) * B(k,j);
// }
//
// Then add the kernel to the nest (via n.addKernel(kernel)), which moves %n to the current point and adds an invoke call

// TODO: add CallableOpInterface, FunctionLike ?
def rcln_KernelOp : rcln_Op<"kernel",
       [SingleBlockImplicitTerminator<"TerminatorOp">, Symbol]> {
  let summary = "kernel operation";
  let description = [{
    The "accln.kernel" operation represents the innermost code in a loop nest.
    The "accln.kernel" operation has a single region indicating the code at the middle of the loop nest.
  }];

  let arguments = (ins StrAttr:$sym_name);
  let regions = (region SizedRegion<1>:$region);
  let results = (outs);

  let hasCanonicalizer = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$id)>
  ];

  let extraClassDeclaration = [{
    static StringRef getIdAttrName();

    Block *getBody() { return &region().front(); }

    StringRef getId();

    // TODO: getName

    std::vector<SymbolicIndexOp> getIndices();

    // FunctionLike trait needs access to the functions below.
    // friend class OpTrait::FunctionLike<rcln_KernelOp>;

    // Hooks for the input/output type enumeration in FunctionLike trait
    // unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    // unsigned getNumFuncResults() { return getType().getNumResults(); }

    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
  }];

  // Invoke a static verify method to verify this operation.
  let verifier = [{ return ::verify(*this); }];

  // let assemblyFormat = [{
  //   attr-dict $sym_name $region
  // }];
}

//
// ScheduledKernelOp
//
def rcln_ScheduledKernelOp : rcln_Op<"scheduled_kernel", [Symbol]> {
  let summary = "scheduled kernel operation";
  let description = [{
    The "loopnest.scheduled_kernel" operation associates a predicate with a kernel.
  }];

  let arguments = (ins FlatSymbolRefAttr:$kernel, AnyType:$predicate, Optional<AnyType>:$placementPredicate);
  let results = (outs);

  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$id, "KernelOp":$kernel, "Value":$predicate)>,
    OpBuilder<(ins "StringRef":$id, "KernelOp":$kernel, "Value":$predicate, "Value":$placementPredicate)>,
    OpBuilder<(ins "StringRef":$id, "KernelOp":$kernel, "KernelPredicateOpInterface":$predicate)>,
    OpBuilder<(ins "StringRef":$id, "KernelOp":$kernel, "KernelPredicateOpInterface":$predicate, "KernelPredicateOpInterface":$placementPredicate)>,
    OpBuilder<(ins "StringRef":$id, "KernelOp":$kernel, "EvaluatablePredicateOpInterface":$predicate)>,
  ];

  let extraClassDeclaration = [{
    static StringRef getIdAttrName();
    static StringRef getKernelIdAttrName();
    StringRef getId();
    StringRef getKernel();

    Operation* getPredicate();
    Operation* getPlacementPredicate();
  }];

  // Invoke a static verify method to verify this operation.
  let verifier = [{ return ::verify(*this); }];
}

//
// InvokeKernelOp
//

def rcln_InvokeKernelOp : rcln_Op<"invoke_kernel", []> {
  let summary = "Invoke a kernel";
  let description = [{}];
  let arguments = (ins FlatSymbolRefAttr:$kernel);

  let builders = [
    OpBuilder<(ins
      "ScheduledKernelOp":$kernel
    ),
    [{
      build($_builder, $_state, kernel.getId());
    }]>];

  let extraClassDeclaration = [{
    StringRef getKernel() { return kernel(); }
  }];
}

//
// DimSizeOp
//

def rcln_DimSizeOp : rcln_Op<"dim_size",
    [DeclareOpInterfaceMethods<rcln_EmitTimeSizeOpInterface>]> {
  let summary = "Evaluate the given dimension range in the current subdomain";
  let description = [{
    The "accln.dim_size" op lowers to the constant value of the size of the requested subdomain dimension at the point it is inserted
  }];

  let builders = [
    OpBuilder<(ins "Index":$index)>
  ];

  let extraClassDeclaration = [{
    static StringRef getIndexAttrName();
  }];

  let arguments = (ins rcln_IndexAttr:$dimensionIndex);
  let results = (outs Index:$size);
}

//
// Kernel predicates
//

class rcln_KernelPredicateOp<string mnemonic, list<OpTrait> traits = []> : rcln_Op<!interleave([mnemonic, "pred"], "_"), !listconcat(traits, [IsolatedFromAbove, NoSideEffect])> {
  let results = (outs I1:$result);
}

def rcln_NullPredicateOp : rcln_KernelPredicateOp<"null",
  [DeclareOpInterfaceMethods<rcln_KernelPredicateOpInterface>]> {
  let summary = "";
  let description = [{ blah }];
  let arguments = (ins);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, $_builder.getI1Type());
    }]>];

  let verifier = [{ return ::verify(*this); }];
}

def rcln_ConstantPredicateOp : rcln_KernelPredicateOp<"const",
  [DeclareOpInterfaceMethods<rcln_KernelPredicateOpInterface>]> {
  let arguments = (ins BoolAttr:$value);
  let builders = [
    OpBuilder<(ins "BoolAttr":$value), [{
      build($_builder, $_state, $_builder.getI1Type(), value);
    }]>];

  let verifier = [{ return ::verify(*this); }];
}

def rcln_FragmentTypePredicateOp : rcln_KernelPredicateOp<"frag",
  [DeclareOpInterfaceMethods<rcln_KernelPredicateOpInterface>]> {
  let arguments = (ins rcln_FragmentTypeAttr:$fragment, rcln_IndexAttr:$index, I64ArrayAttr:$indexValues);
  let builders = [
    OpBuilder<(ins
    "IntegerAttr":$fragment,
    "Index":$index,
    CArg<"ArrayRef<int64_t>", "{}">:$indexValues),
    [{
      build($_builder, $_state, $_builder.getI1Type(), fragment.cast<FragmentTypeAttr>(), IndexAttr::get(index, $_builder.getContext()), $_builder.getI64ArrayAttr(indexValues));
    }]>,
    OpBuilder<(ins
    "FragmentType":$fragment,
    "Index":$index,
    CArg<"ArrayRef<int64_t>", "{}">:$indexValues),
    [{
      auto fragmentAttr = $_builder.getI64IntegerAttr(static_cast<int64_t>(fragment)).cast<FragmentTypeAttr>();
      build($_builder, $_state, $_builder.getI1Type(), fragmentAttr, IndexAttr::get(index, $_builder.getContext()), $_builder.getI64ArrayAttr(indexValues));
    }]>,
    OpBuilder<(ins
    "IntegerAttr":$fragment,
    "Value":$index,
    CArg<"ArrayRef<int64_t>", "{}">:$indexValues),
    [{
      build($_builder, $_state, fragment.cast<FragmentTypeAttr>(), cast<SymbolicIndexOp>(index.getDefiningOp()).getValue(), indexValues);
    }]>,
    OpBuilder<(ins
    "FragmentType":$fragment,
    "Value":$index,
    CArg<"ArrayRef<int64_t>", "{}">:$indexValues),
    [{
      auto fragmentAttr = $_builder.getI64IntegerAttr(static_cast<int64_t>(fragment)).cast<FragmentTypeAttr>();
      build($_builder, $_state, fragmentAttr, cast<SymbolicIndexOp>(index.getDefiningOp()).getValue(), indexValues);
    }]>];

  let verifier = [{ return ::verify(*this); }];
  let extraClassDeclaration = [{
    static StringRef getIndexValuesAttrName() { return "index_values"; }
    std::vector<int64_t> getIndexValues();
  }];
}

def rcln_ProloguePredicateOp : rcln_KernelPredicateOp<"prologue",
  [DeclareOpInterfaceMethods<rcln_EvaluatablePredicateOpInterface>]
  > {
  let arguments = (ins rcln_IndexAttr:$index);
  let builders = [
    OpBuilder<(ins "Index":$index), [{
      build($_builder, $_state, $_builder.getI1Type(), IndexAttr::get(index, $_builder.getContext()));
    }]>,
    OpBuilder<(ins "Value":$index), [{
      build($_builder, $_state, cast<SymbolicIndexOp>(index.getDefiningOp()).getValue());
    }]>];

  let verifier = [{ return ::verify(*this); }];
}

def rcln_EpiloguePredicateOp : rcln_KernelPredicateOp<"epilogue",
  [DeclareOpInterfaceMethods<rcln_EvaluatablePredicateOpInterface>]
  > {
  let arguments = (ins rcln_IndexAttr:$index);
  let builders = [
    OpBuilder<(ins "Index":$index), [{
      build($_builder, $_state, $_builder.getI1Type(), IndexAttr::get(index, $_builder.getContext()));
    }]>,
    OpBuilder<(ins "Value":$index), [{
      build($_builder, $_state, cast<SymbolicIndexOp>(index.getDefiningOp()).getValue());
    }]>];

  let verifier = [{ return ::verify(*this); }];
}

def rcln_PlacementPredicateOp : rcln_KernelPredicateOp<"place",
  [DeclareOpInterfaceMethods<rcln_KernelPredicateOpInterface>]> {
  let arguments = (ins rcln_PlacementPredicateAttr:$placement, rcln_IndexAttr:$index);
  let builders = [
    OpBuilder<(ins "IntegerAttr":$placement, "Index":$index), [{
      auto placementAttr = placement.cast<PlacementTypeAttr>();
      build($_builder, $_state, $_builder.getI1Type(), placementAttr, IndexAttr::get(index, $_builder.getContext()));
    }]>,
    OpBuilder<(ins "IntegerAttr":$placement, "Value":$index), [{
      auto placementAttr = placement.cast<PlacementTypeAttr>();
      build($_builder, $_state, placementAttr, cast<SymbolicIndexOp>(index.getDefiningOp()).getValue());
    }]>];

  let verifier = [{ return ::verify(*this); }];
}

def rcln_IndexDefinedPredicateOp : rcln_KernelPredicateOp<"indexdef",
  [DeclareOpInterfaceMethods<rcln_EvaluatablePredicateOpInterface>,
   DeclareOpInterfaceMethods<rcln_KernelPredicateOpInterface>]> {
  let arguments = (ins rcln_IndexAttr:$index);
  let builders = [
    OpBuilder<(ins "Index":$index), [{
      build($_builder, $_state, $_builder.getI1Type(), IndexAttr::get(index, $_builder.getContext()));
    }]>,
    OpBuilder<(ins "Value":$index), [{
      build($_builder, $_state, cast<SymbolicIndexOp>(index.getDefiningOp()).getValue());
    }]>];

  let verifier = [{ return ::verify(*this); }];
}

def rcln_ConjunctionPredicateOp : rcln_KernelPredicateOp<"conj",
  [DeclareOpInterfaceMethods<rcln_KernelPredicateOpInterface>]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [
    OpBuilder<(ins "ValueRange":$values), [{
      build($_builder, $_state, $_builder.getI1Type(), values);
    }]>,

    OpBuilder<(ins "mlir::ArrayRef<KernelPredicateOpInterface>":$preds), [{
      std::vector<mlir::Value> values;
      for(auto p : preds)
      {
        values.push_back(p->getResult(0));
      }
      build($_builder, $_state, $_builder.getI1Type(), values);
    }]>

    ];

  let verifier = [{ return ::verify(*this); }];
}

def rcln_DisjunctionPredicateOp : rcln_KernelPredicateOp<"disj",
  [DeclareOpInterfaceMethods<rcln_KernelPredicateOpInterface>]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [
    OpBuilder<(ins "ValueRange":$values), [{
      build($_builder, $_state, $_builder.getI1Type(), values);
    }]>,

    OpBuilder<(ins "mlir::ArrayRef<KernelPredicateOpInterface>":$preds), [{
      std::vector<mlir::Value> values;
      for(auto p : preds)
      {
        values.push_back(p.getOperation()->getResult(0));
      }
      build($_builder, $_state, $_builder.getI1Type(), values);
    }]>

    ];

  let verifier = [{ return ::verify(*this); }];
}

//
// PrintOp
//
def rcln_PrintOp : rcln_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input to print.
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyTensor]>:$input,
    UnitAttr:$to_stderr
  );

  let builders = [
    OpBuilder<(ins "Value":$input), [{
               build($_builder, $_state, input, false); }]>
  ];

  // Invoke a static verify method to verify this operation.
  let verifier = [{ return ::verify(*this); }];
}

//
// TerminatorOp
//
def rcln_TerminatorOp : rcln_Op<"terminator", [Terminator]> {
  let summary = "cf terminator operation";
  let description = [{
    "accln.terminator" is a special terminator operation for blocks inside
    loops. It terminates the region. This operation does _not_ have a custom
    syntax. However, `std` control operations omit the terminator in their
    custom syntax for brevity.

    ```mlir
       accln.terminator
    ```
  }];

  // Invoke a static verify method to verify this operation.
  let verifier = [{ return ::verify(*this); }];
  let assemblyFormat = [{
    attr-dict
  }];
}

//
// ExecPlanOp
//
def rcln_ExecPlanOp : rcln_Op<"exec_plan", [ParentOneOf<["NestOp", "ScheduleOp"]>]> {
let summary = "loopnest execution plan operation";
let description = [{
The "accln.exec_plan" operation describes the execution plan for a loop nest.

The "accln.exec_plan" operation has a single region, which contains ops directing the execution plan
}];

let arguments = (ins ExecutionTargetAttr:$exec_target);
let results = (outs);
let extraClassDeclaration = [{
static StringRef getGPUProcessorMapAttrName() { return "gpu_processor_map"; }
static StringRef getGPULaunchAttrName() { return "gpu_launch"; }
}];

}

#endif // LOOPNEST_OPS
