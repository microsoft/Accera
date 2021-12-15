//===- CppPrinter.cpp - Translating MLIR to C++ ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #include "CppPrinter.h"

// #include "AffineDialectCppPrinter.h"
// #include "AcceraDialectCppPrinter.h"
// #include "CppPrinterUtils.h"
// #include "GpuDialectCppPrinter.h"
// #include "ScfDialectCppPrinter.h"
// #include "StdDialectCppPrinter.h"

// #include "llvm/ADT/STLExtras.h"
// #include "llvm/ADT/StringExtras.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringExtras.h>

#include "CppPrinter.h"

#include "AffineDialectCppPrinter.h"
#include "AcceraDialectCppPrinter.h"
#include "CppPrinterUtils.h"
#include "GpuDialectCppPrinter.h"
#include "RocDLDialectCppPrinter.h"
#include "ScfDialectCppPrinter.h"
#include "StdDialectCppPrinter.h"

using namespace llvm;

namespace mlir {
namespace cpp_printer {

StringRef SSANameState::NamePrefix(SSANameKind kind) {
  switch (kind) {
  default:
    return "<<Invalid SSANameKind>>";

  case SSANameKind::Temp:
    return "tmp";
  case SSANameKind::Argument:
    return "arg";
  case SSANameKind::Constant:
    return "const";
  case SSANameKind::Variable:
    return "var";
  case SSANameKind::LoopIdx:
    return "idx";
  }
}

StringRef SSANameState::getName(Value val) {
  auto nameIt = valueNames.find(val);
  if (nameIt != valueNames.end()) {
    return nameIt->second;
  } else {
    return "<<Unknown Value>>";
  }
}

StringRef SSANameState::getOrCreateName(Value val, SSANameKind nameKind) {
  assert(nameKind != SSANameKind::Temp &&
         "Temp name cannot be associated with any Value");

  auto nameIt = valueNames.find(val);
  // Return its name if the value already has one
  if (nameIt != valueNames.end()) {
    return nameIt->second;
  }

  StringRef namePrefix = SSANameState::NamePrefix(nameKind);
  SmallString<16> nameStr("");
  llvm::raw_svector_ostream nameStream(nameStr);
  if (nameKind == SSANameKind::Argument)
    nameStream << namePrefix << nextArgumentID++;
  else
    nameStream << namePrefix << nextValueID++;

  StringRef name;
  if (usedNames.count(nameStr.str())) {
    nameStr.append("<<Non-unique>>");
    name = StringRef(nameStr).copy(nameAllocator);
  } else {
    name = StringRef(nameStr).copy(nameAllocator);
    usedNames.insert(name, char());
    valueNames[val] = name;
  }

  return name;
}

StringRef SSANameState::getTempName() {
  StringRef namePrefix = SSANameState::NamePrefix(SSANameKind::Temp);
  SmallString<16> nameStr("");
  llvm::raw_svector_ostream nameStream(nameStr);
  nameStream << namePrefix << nextTempID++;

  StringRef name;
  if (usedNames.count(nameStr.str())) {
    nameStr.append("<<Non-unique-temp>>");
    name = StringRef(nameStr).copy(nameAllocator);
  } else {
    name = StringRef(nameStr).copy(nameAllocator);
    usedNames.insert(name, char());
  }

  return name;
}

// The function taken from AsmPrinter
LogicalResult CppPrinter::printFloatValue(const APFloat &apValue) {
  // We would like to output the FP constant value in exponential notation,
  // but we cannot do this if doing so will lose precision.  Check here to
  // make sure that we only output it in exponential format if we can parse
  // the value back and get the same value.
  bool isInf = apValue.isInfinity();
  bool isNaN = apValue.isNaN();
  if (!isInf && !isNaN) {
    SmallString<128> strValue;
    apValue.toString(strValue, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                     /*TruncateZero=*/false);

    // Check to make sure that the stringized number is not some string like
    // "Inf" or NaN, that atof will accept, but the lexer will not.  Check
    // that the string matches the "[-+]?[0-9]" regex.
    assert(((strValue[0] >= '0' && strValue[0] <= '9') ||
            ((strValue[0] == '-' || strValue[0] == '+') &&
             (strValue[1] >= '0' && strValue[1] <= '9'))) &&
           "[-+]?[0-9] regex does not match!");

    // Parse back the stringized version and check that the value is equal
    // (i.e., there is no precision loss).
    if (APFloat(apValue.getSemantics(), strValue).bitwiseIsEqual(apValue)) {
      os << strValue;
      return success();
    }

    // If it is not, use the default format of APFloat instead of the
    // exponential notation.
    strValue.clear();
    apValue.toString(strValue);

    // Make sure that we can parse the default form as a float.
    if (StringRef(strValue).contains('.')) {
      os << strValue;
      return success();
    }
  }

  // Print special values in hexadecimal format. The sign bit should be included
  // in the literal.
  SmallVector<char, 16> str;
  APInt apInt = apValue.bitcastToAPInt();
  apInt.toString(str, /*Radix=*/16, /*Signed=*/false,
                 /*formatAsCLiteral=*/true);
  os << str;
  return success();
}

LogicalResult CppPrinter::printDeclarationForOpResult(Operation *op) {
  auto numResults = op->getNumResults();
  if (numResults == 0)
    return success();

  if (numResults == 1) {
    OpResult result = op->getResult(0);
    RETURN_IF_FAILED(printType(result.getType()));
    os << " ";
    os << state.nameState.getOrCreateName(result,
                                          SSANameState::SSANameKind::Variable);
    return success();
  }

  // op returns multiple values
  // TODO: rely on std::tie to retrieve multiple returned values
  return op->emitError() << "<<creating a declaration for an Operation with "
                            "multiple return values "
                            "is not supported yet>>";
}

LogicalResult CppPrinter::printAttribute(Attribute attr) {
  // TODO: handle attribute alias

  auto attrType = attr.getType();
  if (auto boolAttr = attr.dyn_cast<BoolAttr>()) {
    os << (boolAttr.getValue() ? "true" : "false");
  } else if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
    printFloatValue(floatAttr.getValue());
  } else if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
    APInt intVal = intAttr.getValue();
    if (intVal.getBitWidth() == 1) {
      os << (intVal.getBoolValue() ? "true" : "false");
    } else {
      // print intVal as unsigned only if it's explictly
      // unsigned. That being said, signless intVal is printed as signed int
      intVal.print(os, !attrType.isUnsignedInteger());
    }
  } else if (auto strAttr = attr.dyn_cast<StringAttr>()) {
    os << '"';
    printEscapedString(attr.cast<StringAttr>().getValue(), os);
    os << '"';
  } else if (auto arrayAttr = attr.dyn_cast<ArrayAttr>()) {
    os << '{';
    llvm::interleaveComma(attr.cast<ArrayAttr>().getValue(), os,
                          [&](Attribute attr) { printAttribute(attr); });
    os << '}';
  } else if (auto typeAttr = attr.dyn_cast<TypeAttr>()) {
    RETURN_IF_FAILED(printType(attr.cast<TypeAttr>().getValue()));
  } else {
    os << "<<UNSUPPORTED ATTRIBUTE>>";
    return failure();
  }

  return success();
}

LogicalResult CppPrinter::printIndexType(IndexType idxType) {
  int bitCount = getIntTypeBitCount(IndexType::kInternalStorageBitWidth);

  if (bitCount < 0) {
    os << "<<UNSUPPORTED index type witdh: "
       << IndexType::kInternalStorageBitWidth << ">>";
    return failure();
  }

  os << "int" << bitCount << "_t";
  return success();
}

LogicalResult CppPrinter::printIntegerType(IntegerType intType,
                                           bool forceSignedness,
                                           bool isSigned) {
  auto width = intType.getWidth();
  if (width == 1) {
    os << "bool";
    return success();
  }

  int bitCount = getIntTypeBitCount(width);
  if (bitCount < 0) {
    os << "<<UNSUPPORTED integer witdh: " << width << ">>";
    return failure();
  }

  if (!forceSignedness) {
    // when forceSignedness is false, intType is unsigned only if
    // it's explicitly declared as unsigned
    isSigned = !intType.isUnsigned();
  }
  if (!isSigned)
    os << 'u';
  os << "int" << bitCount << "_t";
  return success();
}

LogicalResult
CppPrinter::printUnrankedMemRefType(UnrankedMemRefType unrankedMemRefType) {
  RETURN_IF_FAILED(printType(unrankedMemRefType.getElementType()));
  os << " *";
  return success();
}

LogicalResult CppPrinter::printVectorTypeArrayDecl(VectorType vecType,
                                                   StringRef vecVar) {
  if (isCuda) {
    static GpuDialectCppPrinter *gpuDialectPrinter = nullptr;
    if (!gpuDialectPrinter) {
      for (auto &dialectPrinter : dialectPrinters) {
        if ((gpuDialectPrinter =
                 dynamic_cast<GpuDialectCppPrinter *>(dialectPrinter.get()))) {
          break;
        }
      }
    }
    assert(gpuDialectPrinter);
    RETURN_IF_FAILED(
        gpuDialectPrinter->printVectorTypeArrayDecl(vecType, vecVar));
    return success();
  } else {
    os << "<<non-cuda VectorType is not supported yet>>";
    return failure();
  }
}

LogicalResult CppPrinter::checkMemRefType(MemRefType memrefType) {
  // Currently, do not allow dynamic dimensions
  if (memrefType.getNumDynamicDims() != 0) {
    os << "<<MemRefType with dynamic dimentions is not supported>>";
    return failure();
  }

  // Also disallow affine maps
  if (memrefType.getAffineMaps().size() != 0) {
    os << "<<MemRefType with affine maps is not supported>>";
    return failure();
  }

  return success();
}

LogicalResult CppPrinter::printArrayDeclaration(MemRefType memrefType,
                                                StringRef arrayName) {

  auto shape = memrefType.getShape();
  auto rank = memrefType.getRank();

  RETURN_IF_FAILED(printType(memrefType.getElementType()));
  os << " " << arrayName;

  // We print C-style arrays

  // zero-dimentional memref is treated as array[1] because we
  // can still load and store an element into/from it
  if (rank == 0) {
    os << "[1]";
  } else {
    for (auto d : shape) {
      os << "[" << d << "]";
    }
  }

  return success();
}

LogicalResult CppPrinter::printDecayedArrayDeclaration(MemRefType memRefType,
                                                       StringRef arrayName) {
  RETURN_IF_FAILED(checkMemRefType(memRefType));
  RETURN_IF_FAILED(printType(memRefType.getElementType()));
  auto rank = memRefType.getRank();
  if (rank <= 1) {
    os << " *" << arrayName;
    return success();
  }

  auto shape = memRefType.getShape();
  os << " (*" << arrayName << ")";
  for (auto d : shape.drop_front()) {
    os << "[" << d << "]";
  }
  return success();
}

LogicalResult CppPrinter::printVectorType(VectorType type) {
  RETURN_IF_FAILED(printType(type.getElementType()));
  os << type.getNumElements();
  return success();
}

LogicalResult CppPrinter::printType(Type type) {
  // TODO: handle type alias

  // Check standard type first
  if (type.isa<FunctionType>() || type.isa<OpaqueType>() ||
      type.isa<RankedTensorType>() ||
      type.isa<UnrankedTensorType>() || type.isa<MemRefType>() ||
      type.isa<ComplexType>() || type.isa<TupleType>() ||
      type.isa<NoneType>() || type.isa<Float64Type>()) {
    os << "<<UNSUPPORTED TYPE>>";
    return failure();
  } else if (type.isa<BFloat16Type>()) {
    os << "bfloat16";
    return success();
  } else if (type.isa<Float16Type>()) {
    // Note that individual dialect will print out typedef for
    // the actual float16_t declaration
    os << float16T();
    return success();
  } else if (type.isa<Float32Type>()) {
    os << "float";
    return success();
  } else if (type.isa<IndexType>()) {
    return printIndexType(type.dyn_cast<IndexType>());
  } else if (type.isa<IntegerType>()) {
    return printIntegerType(type.dyn_cast<IntegerType>());
  } else if (type.isa<UnrankedMemRefType>()) {
    return printUnrankedMemRefType(type.dyn_cast<UnrankedMemRefType>());
  } else if (type.isa<VectorType>()) {
    return printVectorType(type.dyn_cast<VectorType>());
  } else {
    // Nothing to do
  }

  for (auto &dialectPrinter : dialectPrinters) {
    bool consumed = false;
    RETURN_IF_FAILED(dialectPrinter->printDialectType(type, &consumed));
    if (consumed)
      return success();
  }

  os << "<<UNSUPPORTED TYPE>>";
  return failure();
}

LogicalResult CppPrinter::printOperationOperands(Operation *op) {
  interleaveComma(op->getOperands(), os, [&](Value operand) {
    os << state.nameState.getName(operand);
  });
  return success();
}

LogicalResult CppPrinter::printMemRefAccess(Value memref, MemRefType memRefType,
                                            Operation::operand_range indices,
                                            bool usePointerOffsets /* = false */) {
  auto rank = memRefType.getRank();
  // early return if rank is 0, i.e. we are accessing an array of size
  // 1
  if (rank == 0) {
    os << "*" << state.nameState.getName(memref);
    return success();
  }

  auto shape = memRefType.getShape();
  SmallVector<int64_t, 5> strides(rank);
  strides[rank - 1] = 1;
  for (int i = static_cast<int>(rank) - 2; i >= 0; i--)
  {
      strides[i] = strides[i + 1] * static_cast<int64_t>(shape[i + 1]);
  }

  std::string offsetStr;
  llvm::raw_string_ostream offsetStream(offsetStr);
  for (int i = 0; i < rank; i++)
  {
      auto idx = state.nameState.getName(indices[i]);
      offsetStream << " + " << idx << " * " << strides[i];
  }

  if (usePointerOffsets)
  {
      os << "*((";
      RETURN_IF_FAILED(printType(memRefType.getElementType()));
      os << "*)" << state.nameState.getName(memref);

      os << offsetStream.str() << ")";
  }
  else
  {
      os << "(";
      RETURN_IF_FAILED(printType(memRefType.getElementType()));
      os << "*)" << state.nameState.getName(memref);

      os << "[" << offsetStream.str() << "]";
  }

  return success();
}

LogicalResult CppPrinter::printBlock(Block *block, bool printParens,
                                     bool printBlockTerminator) {
  SSANameState::Scope scope(state.nameState);
  llvm::ScopedHashTable<StringRef, char>::ScopeTy usedNamesScope(
      state.nameState.usedNames);

  // TODO: support block arguments?
  if (printParens)
    os << "{\n";
  auto i = block->getOperations().begin();
  auto e = block->getOperations().end();
  if (!printBlockTerminator) {
    --e;
  }

  for (; i != e; ++i) {
    Operation *op = &(*i);
    bool skipped = false;
    RETURN_IF_FAILED(printOperation(op, &skipped, /*trailingSemiColon*/ true));
    // doesn't really matter - just beautify the output a bit by skipping
    // an unecessary semicolon
    if (!isa<scf::IfOp>(op) && !isa<scf::ForOp>(op) && !isa<AffineForOp>(op) &&
        !isa<AffineIfOp>(op) && !skipped) {
      os << ";\n";
    }
  }
  if (printParens)
    os << "}\n";

  return success();
}

LogicalResult CppPrinter::printRegion(Region &region, bool printParens,
                                      bool printBlockTerminator) {
  if (!region.empty()) {
    // Currently we only support Region with a single block
    auto &blocks = region.getBlocks();
    if (blocks.size() == 0)
      return success();
    if (blocks.size() != 1) {
      os << "<<does not support regions with multiple blocks>>";
      return failure();
    }
    RETURN_IF_FAILED(
        printBlock(&(blocks.front()), printParens, printBlockTerminator));
  }
  return success();
}

LogicalResult CppPrinter::printTypes(ArrayRef<Type> types) {
  auto typeCnt = types.size();

  if (typeCnt == 0) {
    os << "void";
    return success();
  }

  if (typeCnt == 1)
    return printType(types.front());

  os << "std::tuple<";
  RETURN_IF_FAILED(interleaveCommaWithError(
      types, os, [&](Type type) { return printType(type); }));
  os << ">";

  return success();
}

LogicalResult CppPrinter::printIntrinsicCallOp(Operation *callOp,
                                               Operation *defFuncOp) {
  for (auto &dialectPrinter : dialectPrinters) {
    bool consumed = false;
    RETURN_IF_FAILED(
        dialectPrinter->printIntrinsicCallOp(callOp, defFuncOp, &consumed));
    if (consumed)
      return success();
  }
  return callOp->emitError("unsupported intrinsic call");
}

LogicalResult CppPrinter::printBlockArgument(BlockArgument arg) {
  StringRef argName =
      state.nameState.getOrCreateName(arg, SSANameState::SSANameKind::Argument);
  auto argTy = arg.getType();
  if (auto memRefType = argTy.dyn_cast<MemRefType>()) {
    return printDecayedArrayDeclaration(memRefType, argName);
  }

  RETURN_IF_FAILED(printType(argTy));
  os << " " << argName;
  return success();
}

LogicalResult CppPrinter::printFunctionDeclaration(FuncOp funcOp,
                                                   bool trailingSemiColon) {
  // TODO: This matches what we do in ArgoToGPU pass, where we treat all
  // functions to be CUDA global functions. Will add support to device
  // functions later
  os << globalAttrIfCuda();

  auto resultType = funcOp.getType().getResults();
  if (isCuda && !resultType.empty()) {
    return funcOp.emitOpError() << "<<CUDA kernel must return void>>";
  }

  if (failed(printTypes(funcOp.getType().getResults()))) {
    return funcOp.emitOpError() << "<<Unable to print return type>>";
  }

  os << " " << funcOp.getName();

  os << "(";
  // external function
  if (funcOp.getBlocks().size() == 0) {
    interleaveCommaWithError(
        funcOp.getType().getInputs(), os, [&](Type tp) -> LogicalResult {
          if (auto memRefType = tp.dyn_cast<MemRefType>()) {
            return printDecayedArrayDeclaration(memRefType, /*arrayName*/ "");
          } else {
            return printType(tp);
          }
        });
  } else {
    interleaveCommaWithError(funcOp.getArguments(), os,
                             [&](BlockArgument arg) -> LogicalResult {
                               return printBlockArgument(arg);
                             });
  }
  os << ") ";

  if (trailingSemiColon)
    os << ";";
  return success();
}

LogicalResult CppPrinter::printFuncOp(FuncOp funcOp) {
  SSANameState::Scope scope(state.nameState);
  llvm::ScopedHashTable<StringRef, char>::ScopeTy usedNamesScope(
      state.nameState.usedNames);

  auto &blocks = funcOp.getBlocks();
  auto numBlocks = blocks.size();
  if (numBlocks > 1)
    return funcOp.emitOpError() << "<<only single block functions supported>>";

  // print function declaration
  if (failed(printFunctionDeclaration(funcOp,
                                      /*trailingSemicolon*/ numBlocks == 0))) {
    return funcOp.emitOpError() << "<<failed to print function declaration>>";
  }

  // Just a declaration, so emit a newline and return
  if (numBlocks == 0) {
    os << "\n";
    return success();
  }

  // print function body
  if (failed(printBlock(&(blocks.front()))))
    return funcOp.emitOpError() << "<<failed to print function body>>";

  return success();
}

LogicalResult CppPrinter::printModuleOp(ModuleOp moduleOp) {

  // TODO: print forward declarations
  // TODO: extended possible included header files

  for (auto &dialectPrinter : dialectPrinters) {
    dialectPrinter->printDialectHeaderFiles();
  }

  if (isCuda) {
    os <<
    "#if !defined(HIP_PLATFORM_HCC)\n"
    "#include \"cuda_fp16.h\"\n"
    "#endif // !defined(HIP_PLATFORM_HCC)";
  }

  os << "\n";
  llvm::SmallVector<llvm::StringRef, 2> system_header_files = {"math.h",
                                                               "stdint.h"};
  for (unsigned i = 0, e = system_header_files.size(); i != e; ++i) {
    os << "#include <" << system_header_files[i] << ">\n";
  }

  if (!isCuda) {
    os << "#ifndef __forceinline__\n";
    os << "#if defined(_MSC_VER)\n";
    os << "#define __forceinline__ __forceinlinen\n";
    os << "#else\n";
    os << "#define __forceinline__ __inline__ __attribute__((always_inline))\n";
    os << "#endif // _MSC_VER\n";
    os << "#endif // __forceinline__\n";
  }

  for (auto &dialectPrinter : dialectPrinters) {
    RETURN_IF_FAILED(dialectPrinter->printPrologue());
  }

  for (Operation &op : moduleOp) {
    bool skipped = false;
    RETURN_IF_FAILED(printOperation(&op, &skipped,
                                    /*trailingSemiColon=*/false));
  }

  for (auto &dialectPrinter : dialectPrinters) {
    RETURN_IF_FAILED(dialectPrinter->printEpilogue());
  }
  return success();
}

LogicalResult CppPrinter::printOperation(Operation *op, bool *skipped,
                                         bool trailingSemiColon) {
  if (state.skippedOps.contains(op)) {
    *skipped = true;
    return success();
  }

  if (auto funcOp = dyn_cast<FuncOp>(op)) {
    auto iter = state.functionDefConditionalMacro.find(op);
    bool hasCondition = (iter != state.functionDefConditionalMacro.end());
    if (hasCondition)
      os << "#ifdef " << iter->second << "\n";
    RETURN_IF_FAILED(printFuncOp(funcOp));
    if (hasCondition)
      os << "#endif // " << iter->second << "\n";
    return success();
  }

  for (auto &dialectPrinter : dialectPrinters) {
    bool consumed = false;
    RETURN_IF_FAILED(
        dialectPrinter->printDialectOperation(op, skipped, &consumed));
    if (consumed)
      return success();
  }

  if (trailingSemiColon)
    os << ";";
  os << "\n";

  // if (isa<ModuleTerminatorOp>(op))
  //   return success();

  return op->emitOpError() << "<<unsupported op (" << op->getName()
                           << ") for CppPrinter>>";
}

void CppPrinter::registerAllDialectPrinters() {
  static bool init_once = [&]() {
    registerDialectPrinter<AffineDialectCppPrinter>();
    registerDialectPrinter<AcceraDialectCppPrinter>();
    registerDialectPrinter<GpuDialectCppPrinter>();
    registerDialectPrinter<RocDLDialectCppPrinter>();
    registerDialectPrinter<StdDialectCppPrinter>();
    registerDialectPrinter<ScfDialectCppPrinter>();
    return true;
  }();
  (void)init_once;
}

LogicalResult CppPrinter::runPrePrintingPasses(Operation *m) {
  for (auto &dialectPrinter : dialectPrinters) {
    RETURN_IF_FAILED(dialectPrinter->runPrePrintingPasses(m));
  }
  return success();
}

} // namespace cpp_printer
} // namespace mlir
