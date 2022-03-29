////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "value/ValueDialect.h"

#include "value/ValueAttributes.h"
#include "value/ValueDialect.cpp.inc"
#include "value/ValueEnums.h"
#include "value/ValueFuncOp.h"

#include "IRUtil.h"

#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/Interfaces/CallInterfaces.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>

#include "value/ValueAttrs.cpp.inc"
#include "value/ValueOpsEnums.cpp.inc"

namespace accera::ir::value
{
void ValueDialect::initialize()
{
    addOperations<
        accera::ir::value::ValueFuncOp,
#define GET_OP_LIST
#include "value/ValueOps.cpp.inc"
        >();
    addTypes<MFMAMatrixType>();
}

mlir::Type ValueDialect::parseType(mlir::DialectAsmParser& parser) const
{
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
        return Type();
    if (keyword != "mfma_matrix")
    {

        parser.emitError(parser.getNameLoc(), "unknown value type: " + keyword);
        return Type();
    }
    llvm::SMLoc beginLoc = parser.getNameLoc();

    if (parser.parseLess())
        return nullptr;

    // Parse the size and elementType.
    llvm::SmallVector<int64_t> shape;
    Type elementType;
    if (parser.parseDimensionList(shape, /*allowDynamic=*/false) ||
        parser.parseType(elementType))
        return nullptr;

    // Parse ','
    if (parser.parseComma())
        return nullptr;

    // Parse operand.
    StringRef operand;
    if (failed(parser.parseOptionalString(&operand)))
        return nullptr;

    // Parse '>'.
    if (parser.parseGreater())
        return nullptr;

    return MFMAMatrixType::getChecked(mlir::detail::getDefaultDiagnosticEmitFn(
                                          parser.getEncodedSourceLoc(beginLoc)),
                                      shape,
                                      elementType,
                                      operand);
}

void ValueDialect::printType(Type type, mlir::DialectAsmPrinter& os) const
{
    mlir::TypeSwitch<Type>(type)
        .Case<MFMAMatrixType>([&](MFMAMatrixType fragTy) {
            os << "mfma_matrix<";
            auto shape = fragTy.getShape();
            for (auto dim = shape.begin(), e = shape.end() - 1; dim != e; ++dim)
                os << *dim << 'x';
            os << shape.back() << 'x' << fragTy.getElementType();
            os << ", \"" << fragTy.getOperand() << "\"" << '>';
        })
        .Default([](Type) { llvm_unreachable("unexpected 'value' type kind"); });
}

} // namespace accera::ir::value

using namespace llvm;
using namespace mlir;
using namespace accera::ir::value;

//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

static void buildCmpOp(OpBuilder& build, OperationState& result, CmpOpPredicate predicate, Value lhs, Value rhs)
{
    result.addOperands({ lhs, rhs });
    auto boolType = build.getI1Type();
    if (auto vectorType = lhs.getType().dyn_cast<VectorType>())
    {
        auto shape = vectorType.getShape();
        auto resultType = mlir::VectorType::get(shape, boolType);

        result.types.push_back(resultType);
    }
    else
    {
        result.types.push_back(boolType);
    }
    result.addAttribute(
        CmpOp::getPredicateAttrName(),
        build.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

static void buildBinOp(OpBuilder& build, OperationState& result, BinaryOpPredicate predicate, Value lhs, Value rhs)
{
    result.addOperands({ lhs, rhs });
    result.types.push_back(lhs.getType());
    result.addAttribute(
        BinOp::getPredicateAttrName(),
        build.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

static void buildUnaryOp(OpBuilder& build, OperationState& result, UnaryOpPredicate predicate, Value input)
{
    result.addOperands({ input });
    result.types.push_back(input.getType().cast<ShapedType>().getElementType());
    result.addAttribute(
        UnaryOp::getPredicateAttrName(),
        build.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

void ValueFuncOp::build(OpBuilder& builder, OperationState& result, StringRef name, FunctionType type, ExecutionTarget target)
{
    result.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    result.addAttribute(getExecTargetAttrName(), ExecutionTargetAttr::get(builder.getContext(), target));
    Region* body = result.addRegion();

    Block* entryBlock = new Block;
    entryBlock->addArguments(type.getInputs());

    body->getBlocks().push_back(entryBlock);
}

void ValueFuncOp::build(OpBuilder& builder, OperationState& result, StringRef name, FunctionType type, ExecutionTarget target, ValueFuncOp::ExternalFuncTag)
{
    build(builder, result, name, type, target);
    result.addAttribute("external", builder.getUnitAttr());
    OpBuilder::InsertionGuard guard{ builder };
    builder.setInsertionPointToEnd(&result.regions[0]->front());
    builder.create<ir::value::ReturnOp>(result.location);
}

/// Hook for FunctionLike verifier.
LogicalResult ValueFuncOp::verifyType()
{
    Type type = getTypeAttr().getValue();
    if (!type.isa<FunctionType>())
    {
        return emitOpError("requires '" + getTypeAttrName() + "' attribute of function type");
    }

    return success();
}

// CallableOpInterface
Region* ValueFuncOp::getCallableRegion()
{
    return isExternal() ? nullptr : &getBody();
}

// CallableOpInterface
ArrayRef<Type> ValueFuncOp::getCallableResults()
{
    return getType().getResults();
}

ParseResult ValueFuncOp::parse(OpAsmParser& parser, OperationState& result)
{
    auto buildFuncType = [](Builder& builder, ArrayRef<Type> argTypes, ArrayRef<Type> results, function_like_impl::VariadicFlag, std::string&) {
        return builder.getFunctionType(argTypes, results);
    };

    return function_like_impl::parseFunctionLikeOp(parser, result, /*allowVariadic=*/false, buildFuncType);
}

void ValueFuncOp::print(OpAsmPrinter& p)
{
    FunctionType fnType = getType();
    function_like_impl::printFunctionLikeOp(p, *this, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

LogicalResult ValueFuncOp::verify()
{
    // If this function is external there is nothing to do.
    if (isExternal())
        return success();

    // Verify that the argument list of the function and the arg list of the entry
    // block line up.  The trait already verified that the number of arguments is
    // the same between the signature and the block.
    auto fnInputTypes = getType().getInputs();
    Block& entryBlock = front();
    for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
        if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
            return emitOpError("type of entry block argument #")
                   << i << '(' << entryBlock.getArgument(i).getType()
                   << ") must match the type of the corresponding argument in "
                   << "function signature(" << fnInputTypes[i] << ')';

    return success();
}

// cf: mlir/lib/IR/Function.cpp
void ValueFuncOp::eraseArguments(ArrayRef<unsigned> argIndices)
{
    auto oldType = getType();
    int originalNumArgs = oldType.getNumInputs();
    llvm::BitVector eraseIndices(originalNumArgs);
    for (auto index : argIndices)
    {
        eraseIndices.set(index);
    }
    auto shouldEraseArg = [&](int i) { return eraseIndices.test(i); };

    // There are 3 things that need to be updated:
    // - Function type.
    // - Arg attrs.
    // - Block arguments of entry block.

    // Update the function type and arg attrs.
    SmallVector<Type, 4> newInputTypes;
    SmallVector<DictionaryAttr, 4> newArgAttrs;
    for (int i = 0; i < originalNumArgs; i++)
    {
        if (shouldEraseArg(i))
        {
            continue;
        }
        newInputTypes.emplace_back(oldType.getInput(i));
        newArgAttrs.emplace_back(getArgAttrDict(i));
    }
    setType(FunctionType::get(getContext(), newInputTypes, oldType.getResults()));
    setAllArgAttrs(newArgAttrs);

    // Update the entry block's arguments.
    // We do this in reverse so that we erase later indices before earlier
    // indices, to avoid shifting the later indices.
    Block& entry = front();
    for (int i = 0; i < originalNumArgs; i++)
    {
        if (shouldEraseArg(originalNumArgs - i - 1))
        {
            entry.eraseArgument(originalNumArgs - i - 1);
        }
    }
}

void ValueLambdaOp::build(OpBuilder& builder, OperationState& result, StringRef name, FunctionType type, ExecutionTarget target)
{
    result.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    result.addAttribute(getExecTargetAttrName(), ExecutionTargetAttr::get(builder.getContext(), target));

    Region* body = result.addRegion();
    Block* entryBlock = new Block;
    entryBlock->addArguments(type.getInputs());

    body->getBlocks().push_back(entryBlock);
}

/// Hook for FunctionLike verifier.
LogicalResult ValueLambdaOp::verifyType()
{
    Type type = getTypeAttr().getValue();
    if (!type.isa<FunctionType>())
    {
        return emitOpError("requires '" + getTypeAttrName() + "' attribute of function type");
    }

    return success();
}

// CallableOpInterface
Region* ValueLambdaOp::getCallableRegion()
{
    return &body();
}

// CallableOpInterface
ArrayRef<Type> ValueLambdaOp::getCallableResults()
{
    return getType().getResults();
}

void ValueModuleOp::build(OpBuilder& builder, OperationState& result, StringRef name)
{
    Region* r = result.addRegion();
    ensureTerminator(*r, builder, result.location);
    result.attributes.push_back(builder.getNamedAttr(::mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

void GlobalOp::build(
    OpBuilder& builder,
    OperationState& result,
    MemRefType type,
    bool isConstant,
    StringRef name,
    Attribute value,
    unsigned addrSpace,
    bool isExternal)
{
    result.addAttribute(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
    result.addAttribute("type", TypeAttr::get(type));
    if (isConstant)
        result.addAttribute("constant", builder.getUnitAttr());
    if (isExternal)
        result.addAttribute("external", builder.getUnitAttr());
    if (value)
        result.addAttribute("value", value);
    if (addrSpace != 0)
        result.addAttribute("addr_space", builder.getI32IntegerAttr(addrSpace));
}

static bool satisfiesModule(Operation* op)
{
    return op->hasTrait<OpTrait::SymbolTable>() &&
           op->hasTrait<OpTrait::IsIsolatedFromAbove>();
}

GlobalOp ReferenceGlobalOp::getGlobal()
{
    Operation* module = (*this)->getParentOp();
    while (module && !satisfiesModule(module))
        module = module->getParentOp();
    assert(module && "unexpected operation outside of a module");
    return dyn_cast_or_null<GlobalOp>(
        mlir::SymbolTable::lookupSymbolIn(module, global_name()));
}

FunctionType CallOp::getCalleeType()
{
    SmallVector<Type, 8> argTypes(getOperandTypes());
    return FunctionType::get(getContext(), argTypes, getResultTypes());
}

OpFoldResult GetElementOp::fold(ArrayRef<Attribute> operands)
{
    if (getOperand().getType() == getType())
        return getOperand();

    return {};
}

void ReorderOp::build(OpBuilder& builder,
                      OperationState& result,
                      Value source,
                      ArrayAttr orderAttr)
{
    auto context = builder.getContext();

    // Compute the result memref type
    // Assume (for now) that source hasn't been permuted
    auto sourceType = source.getType().cast<MemRefType>();
    auto originalSizes = sourceType.getShape();

    // Compute permuted sizes and affine map permutation
    std::vector<int64_t> dimOrder = util::ConvertArrayAttrToIntVector(orderAttr);
    std::vector<int64_t> permutedSizes(dimOrder.size());
    std::vector<unsigned> affineMapOrder(dimOrder.size());

    for (auto en : llvm::enumerate(dimOrder))
    {
        permutedSizes[en.index()] = originalSizes[en.value()];
        affineMapOrder[en.value()] = en.index();
    }
    auto permutationMap = AffineMap::getPermutationMap(affineMapOrder, context);
    assert(permutationMap);

    // Compute permuted strides.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    [[maybe_unused]] auto res = getStridesAndOffset(sourceType, strides, offset);
    assert(succeeded(res));
    auto map = makeStridedLinearLayoutMap(strides, offset, context);
    map = map.compose(permutationMap);

    // Compute result type.
    MemRefType resultType = MemRefType::Builder(sourceType).setShape(permutedSizes).setAffineMaps(map);

    build(builder, result, resultType, source, orderAttr);
}

void ReduceOp::build(OpBuilder& builder, OperationState& result, Value input, Value initArg, BodyBuilderFn bodyBuilder)
{
    [[maybe_unused]] auto elementType = input.getType().cast<ShapedType>().getElementType();
    result.addOperands(input);
    result.addOperands(initArg);
    result.addTypes(initArg.getType());
    Region* bodyRegion = result.addRegion();
    bodyRegion->push_back(new Block);
    Block& bodyBlock = bodyRegion->front();
    bodyBlock.addArgument(initArg.getType());
    bodyBlock.addArgument(initArg.getType());

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    if (bodyBuilder)
        bodyBuilder(builder, result.location, bodyBlock.getArgument(0), bodyBlock.getArgument(1));
}

void MapReduceOp::build(OpBuilder& builder, OperationState& result, Value input, Value initArg, MapBodyBuilderFn mapBodyBuilder, ReduceBodyBuilderFn reduceBodyBuilder)
{
    [[maybe_unused]] auto elementType = input.getType().cast<ShapedType>().getElementType();
    result.addOperands(input);
    result.addOperands(initArg);
    result.addTypes(initArg.getType());

    // Map body
    Region* mapBodyRegion = result.addRegion();
    mapBodyRegion->push_back(new Block);
    Block& mapBodyBlock = mapBodyRegion->front();
    mapBodyBlock.addArgument(initArg.getType());
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&mapBodyBlock);
        if (mapBodyBuilder)
            mapBodyBuilder(builder, result.location, mapBodyBlock.getArgument(0));
    }

    // Reduce body
    Region* reduceBodyRegion = result.addRegion();
    reduceBodyRegion->push_back(new Block);
    Block& reduceBodyBlock = reduceBodyRegion->front();
    reduceBodyBlock.addArgument(initArg.getType());
    reduceBodyBlock.addArgument(initArg.getType());
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&reduceBodyBlock);
        if (reduceBodyBuilder)
            reduceBodyBuilder(builder, result.location, reduceBodyBlock.getArgument(0), reduceBodyBlock.getArgument(1));
    }
}

// //===----------------------------------------------------------------------===//
// // MFMAMatrixType
// //===----------------------------------------------------------------------===//

MFMAMatrixType MFMAMatrixType::get(ArrayRef<int64_t> shape, Type elementType, StringRef operand)
{
    return Base::get(elementType.getContext(), shape, elementType, operand);
}

MFMAMatrixType
MFMAMatrixType::getChecked(llvm::function_ref<InFlightDiagnostic()> emitError,
                           ArrayRef<int64_t> shape,
                           Type elementType,
                           StringRef operand)
{
    return Base::getChecked(emitError, elementType.getContext(), shape, elementType, operand);
}

unsigned MFMAMatrixType::getNumDims() const
{
    return getImpl()->numDims;
}

ArrayRef<int64_t> MFMAMatrixType::getShape() const
{
    return getImpl()->getShape();
}

Type MFMAMatrixType::getElementType() const
{
    return getImpl()->elementType;
}

StringRef MFMAMatrixType::getOperand() const
{
    return getImpl()->getOperand();
}

bool MFMAMatrixType::isValidElementType(Type elementType)
{
    return elementType.isF16() || elementType.isF32();
}

int64_t MFMAMatrixType::getLeadingDim() const {
    return getShape().back();
}

LogicalResult
MFMAMatrixType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<int64_t> shape,
                       Type elementType,
                       StringRef operand)
{
    if (!operand.equals("AOp") && !operand.equals("BOp") &&
        !operand.equals("COp"))
        return emitError() << "operand expected to be one of AOp, BOp or COp";

    if (!MFMAMatrixType::isValidElementType(elementType))
        return emitError() << "MFMAMatrixType elements must be F16 or F32";

    return success();
}

//===----------------------------------------------------------------------===//
// MFMA Ops
//===----------------------------------------------------------------------===//

static const auto kGenericMemorySpace = 0;
static const auto kGlobalMemorySpace = 1;
static const auto kSharedMemorySpace = mlir::gpu::GPUDialect::getWorkgroupAddressSpace();

static LogicalResult verify(MFMAComputeOp op)
{
    enum OperandMap
    {
        A,
        B,
        C
    };
    SmallVector<MFMAMatrixType, 3> opTypes;

    auto populateOpInfo = [&opTypes, &op]() { 
        opTypes.push_back(op.opA().getType().cast<MFMAMatrixType>());
        opTypes.push_back(op.opB().getType().cast<MFMAMatrixType>());
        opTypes.push_back(op.opC().getType().cast<MFMAMatrixType>());
    };
    populateOpInfo();

    if (!opTypes[A].getOperand().equals("AOp") ||
        !opTypes[B].getOperand().equals("BOp") ||
        !opTypes[C].getOperand().equals("COp"))
        return op.emitError("operands must be in the order AOp, BOp, COp");

    ArrayRef<int64_t> aShape, bShape, cShape;
    aShape = opTypes[A].getShape();
    bShape = opTypes[B].getShape();
    cShape = opTypes[C].getShape();

    if (aShape[1] != bShape[0] ||
        aShape[0] != cShape[0] ||
        bShape[1] != cShape[1])
        return op.emitError("operand shapes do not satisfy matmul constraints");

    return success();
}

static LogicalResult verify(MFMAConstantOp op)
{
    auto value = op.value();
    auto valueType = value.getType();
    auto resMatrixType = op.getMFMAMatrixType();
    auto operand = resMatrixType.getOperand();

    if (!operand.equals("AOp") && !operand.equals("BOp") &&
        !operand.equals("COp"))
        return op.emitError("only AOp, BOp and COp can be constant filled");

    if (valueType != resMatrixType.getElementType())
        return op.emitError("value type must match matrix element type");

    return success();
}

static LogicalResult verify(MFMALoadOp op)
{
    auto srcType = op.getMemRefType(); 
    auto resMatrixType = op.getMFMAMatrixType();
    auto operand = resMatrixType.getOperand();
    auto srcMemSpace = srcType.getMemorySpaceAsInt();

    if (srcMemSpace != kGenericMemorySpace && srcMemSpace != kSharedMemorySpace &&
        srcMemSpace != kGlobalMemorySpace)
        return op.emitError(
            "source memorySpace kGenericMemorySpace, kSharedMemorySpace or "
            "kGlobalMemorySpace only allowed");

    if (!operand.equals("AOp") && !operand.equals("BOp") &&
        !operand.equals("COp"))
        return op.emitError("only AOp, BOp and COp can be loaded");

    return success();
}

static LogicalResult verify(MFMAStoreOp op)
{
    auto srcMatrixType = op.getMFMAMatrixType();
    auto dstMemrefType = op.getMemRefType();
    auto dstMemSpace = dstMemrefType.getMemorySpaceAsInt();
    
    if (dstMemSpace != kGenericMemorySpace && dstMemSpace != kSharedMemorySpace &&
        dstMemSpace != kGlobalMemorySpace)
        return op.emitError(
            "destination memorySpace of kGenericMemorySpace, "
            "kGlobalMemorySpace or kSharedMemorySpace only allowed");

    if (!srcMatrixType.getOperand().equals("COp"))
        return op.emitError(
            "expected the operand matrix being stored to have 'COp' operand type");

    return success();
}

// TableGen'd op method definitions
#define GET_OP_CLASSES
#include "value/ValueOps.cpp.inc"
