////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/FoldUtils.h>

#include <llvm/ADT/StringSet.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/raw_ostream.h>

#ifndef __ACCERA__
#include "mlir/Dialect/Argo/IR/ArgoOps.h"
#include "mlir/Dialect/Argo/IR/ArgoTypes.h"
#include "mlir/Dialect/Argo/Utils/Utils.h"
#else
#include "argo/ArgoOps.h"
#include "argo/ArgoTypes.h"
#include "argo/Utils.h"
#endif // !__ACCERA__

using namespace mlir;
using namespace mlir::argo;

/// This is a common class used for patterns of the form
/// ```
///    someop(memrefcast) -> someop
/// ```
/// It folds the source of the memref_cast into the root operation directly.
static LogicalResult foldMemRefCast(Operation* op)
{
    bool folded = false;
    for (OpOperand& operand : op->getOpOperands())
    {
        auto castOp = operand.get().getDefiningOp<memref::CastOp>();
        if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp))
        {
            operand.set(castOp.getOperand());
            folded = true;
        }
    }
    return success(folded);
}

///////////////////// Operations defined with Tablegen /////////////////////////
// For such operations that do not correspond to library calls (i.e. defined in
// ArgoOps.td), we define an overloaded `print` function and a
// parse`className` function.

//===----------------------------------------------------------------------===//
// Built-in Structured Ops
//===----------------------------------------------------------------------===//

[[maybe_unused]] static void printArgoStructuredOp(OpAsmPrinter& p, Operation* op)
{
    assert(op->getAbstractOperation() && "unregistered operation");
    p << op->getName().getStringRef() << "(" << op->getOperands() << ")";
    p.printOptionalAttrDict(op->getAttrs());
    p << " : " << op->getOperandTypes();
}

[[maybe_unused]] static ParseResult parseArgoStructuredOp(OpAsmParser& parser,
                                         OperationState& result)
{
    SmallVector<OpAsmParser::OperandType, 3> ops;
    SmallVector<Type, 3> types;
    return failure(
        parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonTypeList(types) ||
        parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static LogicalResult verify(CopyOp op)
{
    auto outputViewType = op.getOutputShapedType(0);
    auto inputViewType = op.getInputShapedType(0);
    Type outputViewElemType = outputViewType.getElementType();
    Type inputViewElemType = inputViewType.getElementType();

    if (inputViewElemType != outputViewElemType &&
        !isConvertibleTypes(inputViewElemType, outputViewElemType))
    {
        return op.emitOpError("expects views of the same or convertible type");
    }
    if (inputViewType.getRank() != outputViewType.getRank())
        return op.emitOpError("expects views of the same rank");
    auto rank = op.getNumParallelLoops();
    auto inputPermutationMap = op.inputPermutation();
    if (inputPermutationMap)
    {
        if (inputPermutationMap->getNumInputs() != rank)
            return op.emitOpError("expects optional input_permutation map of rank ")
                   << rank;
        if (!inputPermutationMap->isPermutation())
            return op.emitOpError(
                "expects optional input_permutation map to be a permutation");
    }
    auto outputPermutationMap = op.outputPermutation();
    if (outputPermutationMap)
    {
        if (outputPermutationMap->getNumInputs() != rank)
            return op.emitOpError("expects optional output_permutation map of rank ")
                   << rank;
        if (!outputPermutationMap->isPermutation())
            return op.emitOpError(
                "expects optional output_permutation map to be a permutation");
    }
    if (rank == 0 && inputPermutationMap)
        return op.emitOpError("expected no input permutation when rank == 0");
    if (rank == 0 && outputPermutationMap)
        return op.emitOpError("expected no output permutation when rank == 0");
    return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter& p, argo::YieldOp op)
{
    p << op.getOperationName();
    if (op.getNumOperands() > 0)
        p << ' ' << op.getOperands();
    p.printOptionalAttrDict(op->getAttrs());
    if (op.getNumOperands() > 0)
        p << " : " << op.getOperandTypes();
}

static ParseResult parseYieldOp(OpAsmParser& parser, OperationState& result)
{
    SmallVector<OpAsmParser::OperandType, 2> opInfo;
    SmallVector<Type, 2> types;
    llvm::SMLoc loc = parser.getCurrentLocation();
    return failure(parser.parseOperandList(opInfo) ||
                   parser.parseOptionalAttrDict(result.attributes) ||
                   (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                   parser.resolveOperands(opInfo, types, loc, result.operands));
}

// Check the operand number and types must match the types of the
// ArgoOp interface's shaped operands.
static LogicalResult verifyYield(argo::YieldOp op, ArgoOp argoOpInterface)
{

    if (op.getNumOperands() != 0)
        return op.emitOpError("expected number of yield values (")
               << op.getNumOperands()
               << ") to match the number of operands of the enclosing "
               << "ArgoOp (0)";

    return success();
}

static LogicalResult verify(argo::YieldOp op)
{
    auto* parentOp = op->getParentOp();
    if (parentOp->getNumRegions() != 1 || parentOp->getRegion(0).empty())
        return op.emitOpError("expected single non-empty parent region");

    if (auto argoOp = dyn_cast<ArgoOp>(parentOp))
        return verifyYield(op, argoOp);

    return op.emitOpError("expected parent op with ArgoOp interface");
}

//===----------------------------------------------------------------------===//
// OpaqueOp
//===----------------------------------------------------------------------===//

/**
 * @brief All args will become dynamic when passed to the basic block within
 * this op.
 *
 * @param odsBuilder
 * @param odsState
 * @param args
 * @param argsIn Number of input args passed to this opaque op.
 * @param argsOut Number of output args passed to this opaque op.
 * @param indexingMaps Maps used to index this op (this should be equal in
 * number to the total number of args).
 * @param iteratorTypes
 * @param bodyBuild Callback which will be used to construct the body of this
 * op.
 */
void OpaqueOp::build(
    OpBuilder& odsBuilder,
    OperationState& odsState,
    ValueRange args,
    int64_t argsIn,
    int64_t argsOut,
    ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes,
    function_ref<void(OpBuilder& builder, Location bodyLocation, ValueRange blockArgs)>
        bodyBuild)
{

    build(/*odsBuilder=*/odsBuilder,
          /*odsState=*/odsState,
          /*args=*/args,
          /*args_in=*/odsBuilder.getI64IntegerAttr(argsIn),
          /*args_out=*/odsBuilder.getI64IntegerAttr(argsOut),
          /*indexing_maps=*/odsBuilder.getAffineMapArrayAttr(indexingMaps),
          /*iterator_types=*/odsBuilder.getStrArrayAttr(iteratorTypes),
          /*doc=*/nullptr,
          /*library_call=*/nullptr);

    if (!bodyBuild)
    {
        return;
    }

    SmallVector<Type, 4> blockArgTypes;
    for (Value arg : args)
    {
        MemRefType argType = arg.getType().cast<MemRefType>();

        // Convert this arg to have a fully dynamic shape.
        MemRefType::Builder blockArgTypeBuilder(argType);

        SmallVector<int64_t, 4> dynamicShape(argType.getShape().size(),
                                             argo::kDynamicSize);
        blockArgTypeBuilder.setShape(dynamicShape);

        blockArgTypes.push_back(blockArgTypeBuilder);
    }

    OpBuilder::InsertionGuard guard(odsBuilder);
    auto& region = *odsState.regions.front();
    Block* bodyBlock =
        odsBuilder.createBlock(&region, region.end(), blockArgTypes);
    bodyBuild(odsBuilder, odsState.location, bodyBlock->getArguments());
}

static void print(OpAsmPrinter& p, OpaqueOp op)
{
    auto attrNames = op.argoTraitAttrNames();
    llvm::StringSet<> argoTraitAttrsSet;
    argoTraitAttrsSet.insert(attrNames.begin(), attrNames.end());
    SmallVector<NamedAttribute, 8> attrs;
    for (auto attr : op->getAttrs())
        if (argoTraitAttrsSet.count(attr.first.strref()) > 0)
            attrs.push_back(attr);

    auto dictAttr = DictionaryAttr::get(op.getContext(), attrs);
    p << op.getOperationName() << " " << dictAttr;
    p.printOptionalAttrDict(op->getAttrs(), attrNames);
    p << " (" << op.getOperands() << ")";
    if (!op.region().empty())
    {
        p.printRegion(op.region());
    }

    auto inputTypes = op.getOperandTypes();
    if (!inputTypes.empty())
    {
        p << " : " << inputTypes;
    }
}

static ParseResult parseOpaqueOp(OpAsmParser& parser, OperationState& result)
{
    SmallVector<OpAsmParser::OperandType, 8> operandsInfo, regionOperandsInfo;
    DictionaryAttr dictAttr;
    // Parse the core argo traits that must check into a dictAttr.
    // The name is unimportant as we will overwrite result.attributes.
    // The core argo traits must contain the information necessary to pass the
    // verifier.
    if (parser.parseAttribute(dictAttr, "_", result.attributes))
        return failure();
    result.attributes.assign(dictAttr.getValue().begin(),
                             dictAttr.getValue().end());

    // Optional attributes may be added.
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseOperandList(operandsInfo, OpAsmParser::Delimiter::Paren))
    {
        return failure();
    }

    Region& region = *result.addRegion();
    SmallVector<Type, 8> operandTypes, regionTypes;

    if (parser.parseRegion(region, regionOperandsInfo, regionTypes))
    {
        return failure();
    }

    if (parser.parseOptionalColonTypeList(operandTypes))
    {
        return failure();
    }

    return parser.resolveOperands(operandsInfo, operandTypes, parser.getCurrentLocation(), result.operands);
}

//===----------------------------------------------------------------------===//
// EntryPointOp
//===----------------------------------------------------------------------===//

void EntryPointOp::build(OpBuilder& builder, OperationState& result, StringRef entryName, FunctionType type, StringRef kernelName, StringRef kernelExecSpace)
{
    result.addRegion();

    result.addAttribute(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(entryName));
    result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    result.addAttribute(getKernelNameAttrName(),
                        builder.getSymbolRefAttr(kernelName));
    result.addAttribute(getKernelExecutionSpaceAttrName(),
                        builder.getStringAttr(kernelExecSpace));
}

void EntryPointOp::build(OpBuilder& builder, OperationState& result, StringRef entryName, FunctionType type, StringRef moduleName, StringRef kernelName, StringRef kernelExecSpace)
{
    result.addRegion();

    result.addAttribute(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(entryName));
    result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

    auto kernelSymbol = builder.getSymbolRefAttr(
        moduleName, { builder.getSymbolRefAttr(kernelName) });
    result.addAttribute(getKernelNameAttrName(), kernelSymbol);
    result.addAttribute(getKernelExecutionSpaceAttrName(),
                        builder.getStringAttr(kernelExecSpace));
}

/// Parse an Argo entry_point op
/// <operation> ::= `argo.entry_point` symbol-ref-id `(` argument-list `)`
///                 (`->` function-result-list)? function-attributes?
static ParseResult parseEntryPointOp(OpAsmParser& parser,
                                     OperationState& result)
{
    SmallVector<OpAsmParser::OperandType, 8> entryArgs;
    SmallVector<Type, 8> argTypes;
    SmallVector<NamedAttrList, 1> argAttrs;
    SmallVector<NamedAttrList, 1> resultAttrs;
    SmallVector<Type, 1> resultTypes;
    bool isVariadic;

    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes))
    {
        return failure();
    }

    auto signatureLocation = parser.getCurrentLocation();
    if (failed(function_like_impl::parseFunctionSignature(
            parser, /*allowVariadic=*/false, entryArgs, argTypes, argAttrs, isVariadic, resultTypes, resultAttrs)))
    {
        return failure();
    }

    Builder& builder = parser.getBuilder();
    auto funcType = builder.getFunctionType(argTypes, resultTypes);
    result.addAttribute(EntryPointOp::getTypeAttrName(), TypeAttr::get(funcType));

    // Parse attributes.
    if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
        return failure();
    function_like_impl::addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

    [[maybe_unused]] auto endLocation = parser.getCurrentLocation();
    // FunctionLike op requires one region
    auto* body = result.addRegion();
    OptionalParseResult parseResult = parser.parseOptionalRegion(*body);
    if (parseResult.hasValue() && failed(*parseResult))
        return failure();

    if (!body->empty())
    {
        return parser.emitError(signatureLocation) << "cannot have a body";
    }
    return success();
}

static void printEntryPointOp(OpAsmPrinter& p, EntryPointOp op)
{
    p << EntryPointOp::getOperationName() << ' ';
    p.printSymbolName(op.getName());

    FunctionType type = op.getType();
    function_like_impl::printFunctionSignature(p, op.getOperation(), type.getInputs(),
                                               /*isVariadic=*/false,
                                               type.getResults());

    function_like_impl::printFunctionAttributes(p, op.getOperation(), type.getNumInputs(), type.getNumResults());
}

static LogicalResult verify(EntryPointOp op)
{
    auto kernelNameAttr =
        op->getAttrOfType<SymbolRefAttr>(op.getKernelNameAttrName());
    if (!kernelNameAttr)
    {
        return op.emitOpError("must specify a '" + op.getKernelNameAttrName() +
                              "' symbol attribute");
    }

    if (!op.empty())
    {
        return op.emitOpError("cannot have a body");
    }
    return success();
}

SymbolRefAttr EntryPointOp::kernel()
{
    return (*this)->getAttrOfType<SymbolRefAttr>(getKernelNameAttrName());
}

StringRef EntryPointOp::getKernelName()
{
    return kernel().getLeafReference();
}

StringRef EntryPointOp::getKernelModuleName()
{
    return kernel().getRootReference();
}

StringRef EntryPointOp::getKernelExecutionSpace()
{
    return (*this)->getAttrOfType<StringAttr>(getKernelExecutionSpaceAttrName()).getValue();
}

LogicalResult EntryPointOp::verifyType()
{
    Type type = getTypeAttr().getValue();
    if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");

    return success();
}

namespace
{

LogicalResult verifyOpaqueArgs(OpaqueOp op)
{
    if (!llvm::all_of(op.getOperandTypes(),
                      [](Type t) { return t.isa<MemRefType>(); }))
    {
        return op.emitOpError("expected all arguments to be ranked memref types");
    }

    return success();
}

LogicalResult verifyOpaqueBlockArgs(OpaqueOp op, Block& block)
{
    auto nOperands = op.getNumOperands();
    if (block.getNumArguments() != nOperands)
    {
        return op.emitOpError("expected number of block arguments to match number "
                              "of operands");
    }

    for (auto pair : llvm::zip(block.getArgumentTypes(), op.getOperandTypes()))
    {
        Type blockArgType, opArgType;
        std::tie(blockArgType, opArgType) = pair;

        MemRefType blockMemrefType = blockArgType.dyn_cast<MemRefType>();

        if (!blockMemrefType)
        {
            return op.emitOpError("expected ranked memref block argument, but got ")
                   << blockArgType;
        }

        // BB memref arg must be fully dynamic.
        if (blockMemrefType.getNumDynamicDims() != blockMemrefType.getRank())
        {
            return op.emitOpError(
                       "expected memref block argument to be fully dynamic, but got ")
                   << blockMemrefType;
        }

        if (!opArgType.isa<MemRefType>())
        {
            return op.emitOpError("expected ranked memref op argument, but got ")
                   << opArgType;
        }

        // All op/block argument combinations must be directly cast compatible.
        if (!mlir::memref::CastOp::areCastCompatible(opArgType, blockMemrefType))
        {
            return op.emitError("expected block argument with type ")
                   << blockMemrefType
                   << " to be cast compatible with op argument with type "
                   << opArgType;
        }
    }

    return success();
}

} // namespace

static LogicalResult verify(OpaqueOp op)
{
    auto nInputViews = op.getNumInputs();
    auto nLoops = op.getNumLoops();
    auto nInputsAndOutputBuffers = op.getNumInputsAndOutputBuffers();
    if (nInputsAndOutputBuffers != llvm::size(op.args()))
        return op.emitOpError("expected exactly ")
               << nInputsAndOutputBuffers << " inputs and output buffer operands";

    auto& region = op.region();
    if (!llvm::hasSingleElement(region))
    {
        return op.emitOpError("expected region with 1 block");
    }

    if (failed(verifyOpaqueArgs(op)))
    {
        return failure();
    }

    if (failed(verifyOpaqueBlockArgs(op, region.front())))
    {
        return failure();
    }

    auto attr = op->template getAttrOfType<IntegerAttr>("symbol_source");
    int64_t targetRank = 0;
    if (attr)
    {
        unsigned index = attr.getInt();
        if (index >= op.getNumOperands())
            return op.emitOpError("symbol_source index out of range");
        targetRank = op.getShapedType(index).getRank();
    }

    SmallVector<AffineMap, 4> indexingMaps;
    indexingMaps.reserve(op.indexing_maps().size());
    for (auto en : llvm::enumerate(op.indexing_maps()))
    {
        auto idx = en.index();
        auto m = en.value().template cast<AffineMapAttr>().getValue();
        indexingMaps.push_back(m); // Save reference to map for further checks.
        auto view = (idx < nInputViews) ? op.getInputShapedType(idx)
                                        : op.getOutputShapedType(idx - nInputViews);

        if (m.getNumSymbols() != targetRank)
            return op.emitOpError("expected the number of symbols in indexing_map #")
                   << idx << " to match target rank";

        if (m.getNumDims() != nLoops)
            return op.emitOpError("expected indexing_map #")
                   << idx << " to have " << nLoops
                   << " dim(s) to match the number of loops";

        if (m.getNumResults() != view.getRank())
            return op.emitOpError("expected indexing_map #")
                   << idx << " results to match view rank: " << view;
    }

    auto concatMap = concatAffineMaps(indexingMaps);
    // TODO: Bound inference for maps with symbols
    if (!concatMap.getNumSymbols() && !inversePermutation(concatMap))
        return op.emitOpError("expected the concatenation of maps in indexing_map "
                              "to be invertible");

    return success();
}

namespace mlir
{
namespace argo
{

#ifndef __ACCERA__
#include "mlir/Dialect/Argo/IR/ArgoStructuredOpsInterfaces.cpp.inc"
#else
#include "argo/ArgoStructuredOpsInterfaces.cpp.inc"
#endif // !__ACCERA__
} // namespace argo

#define GET_OP_CLASSES
#ifndef __ACCERA__
#include "mlir/Dialect/Argo/IR/ArgoStructuredOps.cpp.inc"
#else
#include "argo/ArgoOps.cpp.inc"
#endif // !__ACCERA__

#define GET_OP_CLASSES
#ifndef __ACCERA__
// #include "mlir/Dialect/Argo/IR/ArgoStructuredOps.cpp.inc"
#else
#include "argo/ArgoStructuredOps.cpp.inc"
#endif // !__ACCERA__

} // namespace mlir

AffineMap mlir::argo::extractOrIdentityMap(Optional<AffineMap> maybeMap,
                                           unsigned rank,
                                           mlir::MLIRContext* context)
{
    if (maybeMap)
        return maybeMap.getValue();
    if (rank == 0)
        return AffineMap::get(context);
    return AffineMap::getMultiDimIdentityMap(rank, context);
}

namespace
{
struct EraseDeadArgoOp : public RewritePattern
{
    EraseDeadArgoOp(MLIRContext* context, PatternBenefit benefit = 1) :
        RewritePattern(MatchAnyOpTypeTag{}, benefit, context) {}

    LogicalResult matchAndRewrite(Operation* op,
                                  PatternRewriter& rewriter) const override
    {
        auto argoOp = dyn_cast<ArgoOp>(op);
        if (!argoOp)
            return failure();
        for (Value v : argoOp.getInputsAndOutputBuffers())
        {
            // Argo "inputs" may be either tensor or memref type.
            // tensor<0xelt_type> is a convention that may not always mean
            // "0 iterations". Only erase in cases we see memref<...x0x...>.
            auto mt = v.getType().dyn_cast<MemRefType>();
            if (!mt)
                continue;
            if (llvm::is_contained(mt.getShape(), 0))
            {
                rewriter.eraseOp(argoOp);
                return success();
            }
        }
        return failure();
    }
};
} // namespace

#define CANONICALIZERS_AND_FOLDERS(ARGOOP)                                      \
    void ARGOOP::getCanonicalizationPatterns(OwningRewritePatternList& results, \
                                             mlir::MLIRContext* context)        \
    {                                                                           \
        results.insert<EraseDeadArgoOp>(context);                               \
    }                                                                           \
                                                                                \
    LogicalResult ARGOOP::fold(ArrayRef<Attribute>,                             \
                               SmallVectorImpl<OpFoldResult>&)                  \
    {                                                                           \
        return foldMemRefCast(*this);                                           \
    }

CANONICALIZERS_AND_FOLDERS(argo::CopyOp)
CANONICALIZERS_AND_FOLDERS(argo::FillOp)
CANONICALIZERS_AND_FOLDERS(argo::MatmulOp)
CANONICALIZERS_AND_FOLDERS(argo::AccOp)
CANONICALIZERS_AND_FOLDERS(argo::OpaqueOp)
