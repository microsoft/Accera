////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/STLExtras.h>

#include "ir/include/value/ValueAttributes.h"
#include "ir/include/value/ValueEnums.h"

namespace accera::ir
{

// mlir-tblgen currently creates files with the assumption that the following
// symbols are present in the current namespace, so we have to import them
// explicitly
using llvm::APInt;
using llvm::ArrayRef;
using llvm::DenseMapInfo;
using llvm::Optional;
using llvm::SmallVectorImpl;
using llvm::StringRef;

using llvm::iterator_range;

using mlir::AffineMap;
using mlir::AffineMapAccessInterface;
using mlir::AffineMapAttr;
using mlir::AffineReadOpInterface;
using mlir::AffineWriteOpInterface;
using mlir::ArrayAttr;
using mlir::Attribute;
using mlir::Block;
using mlir::Builder;
using mlir::CallInterfaceCallable;
using mlir::CallOpInterface;
using mlir::DictionaryAttr;
using mlir::FlatSymbolRefAttr;
using mlir::FloatType;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::LoopLikeOpInterface;
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
using mlir::OpFoldResult;
using mlir::ParseResult;
using mlir::Region;
using mlir::ShapedType;
using mlir::StringAttr;
using mlir::SymbolOpInterface;
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

// Duplicated from mlir\lib\Conversion\StandardToLLVM\StandardToLLVM.cpp
const mlir::StringRef CInterfaceAttrName = "llvm.emit_c_interface";

// TODO : move these to ValueFuncOp and set them as part of ValueFuncOp creation
const mlir::StringRef RawPointerAPIAttrName = "accv.emit_raw_pointer_api";
const mlir::StringRef HeaderDeclAttrName = "accv.emit_header_decl";
const mlir::StringRef FunctionTagsAttrName = "accv.function_tags";
const mlir::StringRef NoInlineAttrName = "accv.no_inline";
const mlir::StringRef BaseNameAttrName = "accv.base_name";
const mlir::StringRef DynamicArgSizeReferencesAttrName = "accv.dyn_arg_size_refs";
const mlir::StringRef UsagesAttrName = "accv.usages";

} // namespace accera::ir

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "ValueFuncOp.h"
#include "ValueMMAOp.h"
#include "ValueRangeOp.h"
#include "value/ValueDialect.h.inc"
#include "value/ValueOps.h.inc"
