//===- CppPrinter.h - The entry class for CppPrinter ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CPP_PRINTER_H_
#define CPP_PRINTER_H_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/raw_ostream.h>

#define RETURN_IF_FAILED(call)                                                 \
  if (failed(call)) {                                                          \
    return failure();                                                          \
  }

namespace mlir {
namespace cpp_printer {

// Forward declarations
struct CppPrinter;
struct DialectCppPrinter;

struct SSANameState {
  friend struct CppPrinter;

  /// Differentiate names
  enum SSANameKind : unsigned {
    Temp = 0,
    Argument,
    Constant,
    Variable,
    LoopIdx
  };

  llvm::StringRef getTempName();

  llvm::StringRef getName(mlir::Value val);

  llvm::StringRef getOrCreateName(mlir::Value val, SSANameKind kind);

  static llvm::StringRef NamePrefix(SSANameKind kind);

  // RAII-style save/restore to allow the printer to number values
  // with block hierarchies
  struct Scope {
    Scope(SSANameState &state)
        : valueID(state.nextValueID), oldValueID(state.nextValueID),
          argID(state.nextArgumentID), oldArgID(state.nextArgumentID),
          tempID(state.nextTempID), oldTempID(state.nextTempID) {}
    ~Scope() {
      valueID = oldValueID;
      argID = oldArgID;
      tempID = oldTempID;
    }

  private:
    unsigned &valueID;
    unsigned oldValueID;

    unsigned &argID;
    unsigned oldArgID;

    unsigned &tempID;
    unsigned oldTempID;
  };

private:
  /// Next ID assigned to an unamed Value
  unsigned nextValueID = 0;

  /// Next ID assigned to an argument
  unsigned nextArgumentID = 0;

  /// Next ID assigned to a temp name that is not associated with any Value
  unsigned nextTempID = 0;

  /// Map from a Value to a unique name
  llvm::DenseMap<mlir::Value, llvm::StringRef> valueNames;

  /// Keep track of the used names in the scope (e.g. function
  /// scope, block scope, etc)
  llvm::ScopedHashTable<llvm::StringRef, char> usedNames;

  /// allocator for name StringRef
  llvm::BumpPtrAllocator nameAllocator;
};

/// Holding the states for the printer such as SSA names, type alias, etc
struct PrinterState {
  explicit PrinterState() {}

  SSANameState nameState;

  // If a FuncOp has conditional macro, it will be placed inside a #ifdef
  // and #endif pair, e.g.
  // #ifdef conditional_macro
  // func_def
  // #endif
  llvm::DenseMap<mlir::Operation *, StringRef> functionDefConditionalMacro;

  // We will not print these ops from the skippedOps set
  llvm::SmallPtrSet<mlir::Operation *, 4> skippedOps;

  // we will generate #pragma unroll for the ops from the set, which must be
  // eiter ForOp or AffineForOp
  llvm::SmallPtrSet<mlir::Operation *, 4> unrolledForOps;

  // contain decls for intrinsics
  llvm::SmallPtrSet<mlir::Operation *, 4> intrinsicDecls;

  // TODO: add more state kinds
};

/// Print the given MLIR into C++ code. Formatting is not a concern
/// since we could always run clang-format on the output.
/// This class implements the print routines for MLIR language constructs
/// that are dialect-agnostic, e.g. Types, Attributes, Blocks, etc.
/// It also holds printer objects for dialects, each of which prints
/// the operations from the corresponding dialect.
struct CppPrinter {
  explicit CppPrinter(llvm::raw_ostream &os, bool cuda)
      : os(os), isCuda(cuda) {}

  void registerAllDialectPrinters();

  LogicalResult runPrePrintingPasses(mlir::Operation *m);

  template <typename ConcreteDialectPrinter>
  void registerDialectPrinter() {
    dialectPrinters.emplace_back(
        std::make_unique<ConcreteDialectPrinter>(this));
  }

  /// Print a floating point value in a way that the parser will be able to
  /// round-trip losslessly.
  LogicalResult printFloatValue(const llvm::APFloat &apValue);

  /// A helper function that prints a variable declaration based on
  /// the return type of the op. If the op returns nothing (i.e.
  /// calling getNumResults returns 0), this function will print nothing.
  LogicalResult printDeclarationForOpResult(Operation *op);

  /// Print Attribute
  LogicalResult printAttribute(Attribute attr);


  /// print VectorType
  LogicalResult printVectorType(VectorType vecType);

  /// Print Type
  LogicalResult printType(Type type);

  /// Print IndexType
  LogicalResult printIndexType(IndexType idxType);

  /// Print IntegerType. When forceSignedness is true, the signed-ness of
  /// the generated IntegerType will be based on isSigned. Otherwise,
  /// isSigned is not used.
  LogicalResult printIntegerType(IntegerType intType,
                                 bool forceSignedness = false,
                                 bool isSigned = false);

  /// Print dereference to an address given by memref (base) and indices
  /// (offset) for read and write
  LogicalResult printMemRefAccess(Value memref, MemRefType memRefType,
                                  Operation::operand_range indices,
                                  bool usePointerOffsets = false);

  /// Print UnrankedMemRefType as a corresponding pointer type
  LogicalResult printUnrankedMemRefType(UnrankedMemRefType unrankedMemRefType);

  /// print an array declaration for the given VectorType
  LogicalResult printVectorTypeArrayDecl(VectorType vecType, StringRef vecVar);

  /// print array of types:
  ///   - If the array is empty, print void type
  ///   - If the size of the array is larger than 1, print types as std::tuple
  ///   - otherwise, print a single type as it
  LogicalResult printTypes(llvm::ArrayRef<Type> types);

  /// check if MemRefType is supported. Currently, we only support a MemRefType
  /// that has only static dimentions and doesn't have an associated AffineMap.
  /// Basically, we treat MemRefType to be C's array type
  LogicalResult checkMemRefType(MemRefType memrefType);

  /// print an array declaration based on the given MemRefType and array name
  LogicalResult printArrayDeclaration(MemRefType memrefType,
                                      StringRef arrayName);

  /// print an array declaration that is decayed into a pointer, e.g.
  /// for an n-d array ``int a[2][3]'', it would become ``int (*a)[3]'';
  LogicalResult printDecayedArrayDeclaration(MemRefType memrefType,
                                             StringRef arrayName);

  /// print BlockArgument
  LogicalResult printBlockArgument(BlockArgument arg);

  /// print the function delcaration for the given FuncOp.
  /// A trailing semiclon will be generated if trailingSemiColon is true.
  LogicalResult printFunctionDeclaration(FuncOp funcOp, bool trailingSemiColon);

  /// print operation
  LogicalResult printOperation(Operation *op, bool *skipped,
                               bool trailingSemiColon);

  /// print operands for an operation
  LogicalResult printOperationOperands(Operation *op);

  /// print the given block
  LogicalResult printBlock(Block *block, bool printParens = true,
                           bool printBlockTerminator = true);

  /// print intrinsic calls
  LogicalResult printIntrinsicCallOp(Operation *callOp, Operation *defFuncOp);

  /// print FuncOp
  LogicalResult printFuncOp(FuncOp funcOp);

  /// print ModuleOp
  LogicalResult printModuleOp(ModuleOp moduleOp);

  /// print Region. Currently we only support Region with a single block
  LogicalResult printRegion(Region &region, bool printParens = true,
                            bool printBlockTerminator = true);

  const char *deviceAttrIfCuda(bool trailingSpace = true) {
    if (isCuda) {
      return trailingSpace ? "__device__ " : "__device__";
    } else {
      return "";
    }
  }

  const char *sharedAttrIfCuda(bool trailingSpace = true) {
    if (isCuda) {
      return trailingSpace ? "__shared__ " : "__shared__";
    } else {
      return "";
    }
  }

  const char *globalAttrIfCuda(bool trailingSpace = true) {
    if (isCuda) {
      return trailingSpace ? "__global__ " : "__global__";
    } else {
      return "";
    }
  }

  const char *float16T() {
    if (isCuda) {
      return "half";
    } else {
      return "UNSUPPRTED_FLOAT16_T";
    }
  }

  const char *float16VecT() {
    if (isCuda) {
      return "half2";
    } else {
      return "UNSUPPRTED_FLOAT16VEC_T";
    }
  }

  const char *pragmaUnroll() { return "#pragma unroll"; }

  llvm::raw_ostream &getOStream() { return os; }

  PrinterState &getPrinterState() { return state; }

  bool forCuda() { return isCuda; }

private:
  llvm::raw_ostream &os;

  // indicate if we are printing cuda code
  bool isCuda;

  PrinterState state;

  std::vector<std::unique_ptr<DialectCppPrinter>> dialectPrinters;
};

struct DialectCppPrinter {
  DialectCppPrinter(CppPrinter *printer)
      : os(printer->getOStream()), isCuda(printer->forCuda()),
        state(printer->getPrinterState()), printer(printer) {}

  virtual ~DialectCppPrinter() {}

  virtual LogicalResult runPrePrintingPasses(Operation *op) {
    return success();
  }

  virtual LogicalResult printPrologue() { return success(); }

  virtual LogicalResult printEpilogue() { return success(); }

  virtual LogicalResult printDialectOperation(Operation *op, bool *skipped,
                                              bool *consumed) = 0;

  virtual LogicalResult printDialectType(Type type, bool *consumed) {
    return success();
  }

  virtual LogicalResult printIntrinsicCallOp(Operation *callOp,
                                             Operation *defFuncOp,
                                             bool *consumed) {
    *consumed = false;
    return success();
  }

  /// print an array declaration for the given VectorType
  virtual LogicalResult printVectorTypeArrayDecl(VectorType vecType,
                                                 StringRef vecVar) {
    return failure();
  }

  virtual void printDialectHeaderFiles() {}

protected:
  CppPrinter *printer;

  llvm::raw_ostream &os;

  bool isCuda;

  PrinterState &state;
};

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STL as functions used on
/// each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
LogicalResult interleaveWithError(ForwardIterator begin, ForwardIterator end,
                                  UnaryFunctor each_fn,
                                  NullaryFunctor between_fn) {
  if (begin == end)
    return success();
  RETURN_IF_FAILED(each_fn(*begin));

  ++begin;
  for (; begin != end; ++begin) {
    between_fn();
    RETURN_IF_FAILED(each_fn(*begin));
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
LogicalResult interleaveWithError(const Container &c, UnaryFunctor each_fn,
                                  NullaryFunctor between_fn) {
  return interleaveWithError(c.begin(), c.end(), each_fn, between_fn);
}

template <typename Container, typename UnaryFunctor>
LogicalResult interleaveCommaWithError(const Container &c, raw_ostream &os,
                                       UnaryFunctor each_fn) {
  return interleaveWithError(c.begin(), c.end(), each_fn,
                             [&]() { os << ", "; });
}

} // namespace cpp_printer
} // namespace mlir

#endif // CPP_PRINTER_H_
