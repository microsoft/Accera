////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ARGO_STRUCTURED_OPS
#define ARGO_STRUCTURED_OPS

include "mlir/Interfaces/CopyOpInterface.td"

// #ifndef __ACCERA__
// include "mlir/Dialect/Argo/IR/ArgoBase.td"
// include "mlir/Dialect/Argo/IR/ArgoStructuredOpsInterface.td"
// #else
include "ir/include/argo/ArgoBase.td"
include "ir/include/argo/ArgoStructuredOpsInterface.td"
// #endif // !__ACCERA__

/////////////////////////////////////
/// Traits for Computation pattern
/////////////////////////////////////

class AnyMemRefOfRank<int rank> :
  Type<And<[AnyMemRef.predicate,
            MemRefRankOf<[AnyType], [rank]>.predicate]>,
       AnyMemRef.description # " of rank " # rank>;

// The Argo `NInputs` trait provides the API for ops that are known
// to have a specified number of inputs, all passed as operands.
// See Argo/LinalgTraits.h for implementation details an usage.
class NInputs<int args_in> :
  NativeOpTrait<"argo::NInputs<" # !cast<string>(args_in) # ">::Impl"> {}

// The Argo `NOutputs` trait provides the API for ops that are known
// to have a specified number of outputs, all passed as operands.
// See Argo/LinalgTraits.h for implementation details an usage.
class NOutputs<int args_out> :
  NativeOpTrait<"argo::NOutputs<" # !cast<string>(args_out) # ">::Impl"> {}


def StructuredOpTraits : NativeOpTrait<"argo::StructuredOpTraits">;

// Base Tablegen class for Argo ops.
// Argo ops that correspond to library calls operate on argo::View as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Argo op.
class ArgoStructuredBase_Op<string mnemonic, list<OpTrait> props>
  : Op<Argo_Dialect, mnemonic,
       !listconcat(props, [StructuredOpTraits, ArgoStructuredInterface])> {
}

class ArgoStructured_Op<string mnemonic, list<OpTrait> props>
  : ArgoStructuredBase_Op<mnemonic, props> {
  code libraryCallName = [{
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  }];
  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

/////////////////////////////////////
/// Built-in Memory pattern
/////////////////////////////////////

/////////////////////////////////////
/// CopyOp
/////////////////////////////////////
def CopyOp : ArgoStructured_Op<"copy", [
    CopyOpInterface,
    NInputs<1>,
    NOutputs<1>
  ]> {
  let description = [{
    Copies the data in the input view into the output view.
  }];
  let arguments = (ins
    AnyMemRef:$input,
    AnyMemRef:$output,
    OptionalAttr<AffineMapAttr>:$inputPermutation,
    OptionalAttr<AffineMapAttr>:$outputPermutation);

  let builders = [OpBuilder<
    (ins "Value":$input, "Value":$output), [{
    return build(
      $_builder, $_state, input, output, AffineMapAttr(), AffineMapAttr());
  }]>];
  let extraClassDeclaration = [{
    // Rank-polymorphic.
    //   filling_value -> O(ivs) with parallel iterators.
    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      unsigned nPar = input().getType().cast<ShapedType>().getRank();
      return SmallVector<StringRef, 8>(nPar, getParallelIteratorTypeName());
    }

    // I(input_perm(ivs)) -> O(output_perm(ivs))
    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      mlir::MLIRContext *context = getContext();
      auto maybeInputMap = inputPermutation();
      auto maybeOutputMap = outputPermutation();
      unsigned inputRank = getInputShapedType(0).getRank();
      unsigned outputRank = getOutputShapedType(0).getRank();
      return SmallVector<AffineMap, 8>{
          extractOrIdentityMap(maybeInputMap, inputRank, context),
          extractOrIdentityMap(maybeOutputMap, outputRank, context)};
    }

    Value getSource() { return input();}
    Value getTarget() { return output(); }

    static StringRef getPadValueAttrName() { return "pad_value"; }
    static StringRef getUnpadValueAttrName() { return "unpad"; }
  }];
  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

/////////////////////////////////////
/// FillOp
/////////////////////////////////////
def FillOp : ArgoStructured_Op<"fill", [NInputs<0>, NOutputs<1>]> {
  let description = [{
    Fills the data into the output view.
  }];
  let arguments = (ins
    AnyMemRef:$output,
    AnyAttr:$value);

  let extraClassDeclaration = [{
    // Rank-polymorphic.
    //   filling_value -> O(ivs) with parallel iterators.
    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      unsigned nPar = output().getType().cast<ShapedType>().getRank();
      return SmallVector<StringRef, 8>(nPar, getParallelIteratorTypeName());
    }

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      mlir::MLIRContext *context = getContext();
      unsigned nPar = output().getType().cast<ShapedType>().getRank();
      SmallVector<AffineExpr, 4> result;
      for(unsigned i = 0; i < nPar; ++i) {
        result.push_back(getAffineDimExpr(i, context));
      }
      return SmallVector<AffineMap, 8>{AffineMap::get(nPar, 0, result, context)};
    }

    Value getTarget() { return output(); }

    static StringRef getValueAttrName() { return "value"; }
  }];
  //let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

////////////////////////////////////////////////////////////////////////////////
// Built-in Computation Pattern
////////////////////////////////////////////////////////////////////////////////

def MatmulOp : ArgoStructured_Op<"matmul", [NInputs<2>, NOutputs<1>]> {
  let arguments = (ins AnyMemRefOfRank<2>,
                       AnyMemRefOfRank<2>,
                       AnyMemRefOfRank<2>,
                       DefaultValuedAttr<BoolAttr, "true">:$acc);

  let extraClassDeclaration = [{
    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      SmallVector<StringRef, 8> iters;
      iters.reserve(3);
      iters.append(1, getParallelIteratorTypeName());
      iters.append(1, getParallelIteratorTypeName());
      iters.append(1, getReductionIteratorTypeName());
      return iters;
    }

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      mlir::MLIRContext *context = getContext();
      auto i = getAffineDimExpr(0, context);
      auto j = getAffineDimExpr(1, context);
      auto k = getAffineDimExpr(2, context);
      //   A(i, r_k) * B(r_k, j) -> C(i, j)
      return SmallVector<AffineMap, 8>{AffineMap::get(3, 0, {i, k}, context),
                                       AffineMap::get(3, 0, {k, j}, context),
                                       AffineMap::get(3, 0, {i, j}, context)};
    }
  }];
  //let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def AccOp : ArgoStructured_Op<"acc", [NInputs<1>, NOutputs<1>]> {
  let arguments = (ins AnyMemRefOfRank<2>,
                       AnyMemRefOfRank<2>);

  let extraClassDeclaration = [{
    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      SmallVector<StringRef, 8> iters(2, getParallelIteratorTypeName());
      return iters;
    }

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      mlir::MLIRContext *context = getContext();
      auto i = getAffineDimExpr(0, context);
      auto j = getAffineDimExpr(1, context);
      // A(i, j) -> B(i, j)
      return SmallVector<AffineMap, 8>{AffineMap::get(2, 0, {i, j}, context),
                                       AffineMap::get(2, 0, {i, j}, context)};
    }
  }];

  //let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}



//===----------------------------------------------------------------------===//
// Generic Argo ops.
//===----------------------------------------------------------------------===//
def ArgoOperand: AnyTypeOf<[AnyRankedTensor, AnyMemRef]>;

class ArgoOperandOfRank<int rank>: Type<
  And<[
    ArgoOperand.predicate,
    CPred<"$_self.cast<ShapedType>().getRank() == " # rank>]
  >>;
def OpaqueOp : ArgoStructuredBase_Op<"opaque",
    [SingleBlockImplicitTerminator<"YieldOp">]> {
  let description = [{
    an OpaqueOp wrap other op.
  }];
  let arguments = (ins Variadic<ArgoOperand>:$args,
                   I64Attr:$args_in,
                   I64Attr:$args_out,
                   AffineMapArrayAttr:$indexing_maps,
                   ArrayAttr:$iterator_types,
                   OptionalAttr<StrAttr>:$doc,
                   OptionalAttr<StrAttr>:$library_call);

  let regions = (region AnyRegion:$region);

  let builders = [
    OpBuilder<
      (ins "ValueRange":$args, "int64_t":$argsIn, "int64_t":$argsOut,
      "ArrayRef<AffineMap>":$indexingMaps, "ArrayRef<StringRef>":$iteratorTypes,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>","nullptr">)>
  ];

  let extraClassDeclaration = [{
    SmallVector<StringRef, 8> argoTraitAttrNames() {
      return SmallVector<StringRef, 8>{
        getDocAttrName(),
        getIndexingMapsAttrName(), getLibraryCallAttrName(),
        getIteratorTypesAttrName(),
      };
    }

    unsigned getNumInputs() { return args_in(); }

    unsigned getNumOutputs() { return args_out(); }

    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      llvm_unreachable(
        "No such thing as reference iterator types for a generic op.");
    }

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      llvm_unreachable(
        "No such thing as reference indexing maps for a generic op.");
    }

    std::string getLibraryCallName() {
      return library_call().hasValue() ?
        library_call()->str() : "op_has_no_registered_library_name";
    }
  }];
  let verifier = [{ return ::verify(*this); }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

#endif // ARGO_STRUCTURED_OPS
