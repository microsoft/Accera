////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace accera::ir
{

// mlir-tblgen currently creates files with the assumption that the following
// symbols are present in the current namespace, so we have to import them
// explicitly
using llvm::APFloat;
using llvm::APInt;
using llvm::ArrayRef;
using llvm::iterator_range;
using llvm::Optional;
using llvm::SmallVectorImpl;
using llvm::StringRef;

using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::Attribute;
using mlir::Block;
using mlir::Builder;
using mlir::CallInterfaceCallable;
using mlir::CallOpInterface;
using mlir::DictionaryAttr;
using mlir::FlatSymbolRefAttr;
using mlir::FloatAttr;
using mlir::FloatType;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemoryEffectOpInterface;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::NoneType;
using mlir::Op;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::OwningRewritePatternList;
using mlir::ParseResult;
using mlir::Region;
using mlir::ShapedType;
using mlir::StringAttr;
using mlir::SymbolRefAttr;
using mlir::TensorType;
using mlir::Type;
using mlir::TypeAttr;
using mlir::UnitAttr;
using mlir::Value;
using mlir::ValueRange;
using mlir::VectorType;

using mlir::getElementTypeOrSelf;

namespace SideEffects = mlir::SideEffects;
namespace MemoryEffects = mlir::MemoryEffects;
namespace OpTrait = mlir::OpTrait;

} // namespace accera::ir

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "accera/AcceraDialect.h.inc"
#include "accera/AcceraOps.h.inc"
