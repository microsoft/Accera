////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MLIREmitterContext.h"
#include "CompilerOptions.h"
#include "ValueType.h"

#include <ir/include/DialectRegistry.h>
#include <ir/include/IRUtil.h>
#include <ir/include/InitializeAccera.h>
#include <ir/include/TranslateToHeader.h>
#include <ir/include/exec/ExecutionPlanAttributes.h>
#include <ir/include/exec/VectorizationInfo.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueAttributes.h>
#include <ir/include/value/ValueFuncOp.h>

#include <llvm/Support/Casting.h>
#include <transforms/include/value/ValueToStandardLoweringPass.h>

#include <utilities/include/Exception.h>
#include <utilities/include/MemoryLayout.h>
#include <utilities/include/ZipIterator.h>

#include <value/include/Debugging.h>
#include <value/include/MatrixFragment.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVTypes.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>

using namespace accera;
using namespace accera::utilities;
using namespace accera::value;

using ConstantData = accera::value::detail::ConstantData;

namespace
{
const int64_t MemRefPointerShape[1] = { 1 }; // shape of memrefs-of-memrefs
mlir::Type ToLLVMMemrefDescriptorType(mlir::OpBuilder& builder, ::accera::value::Value value); // Forward declare

struct InitAccera
{
    InitAccera(::mlir::MLIRContext* ctx) :
        _ownedContext(ctx ? nullptr : new mlir::MLIRContext),
        _context(ctx ? ctx : _ownedContext.get())

    {
        accera::ir::InitializeAccera();

        // By default, eagerly load all of our registered dialects in our owned MLIRContext
        // Eventually we may want to instead lazily load dialects once we have more
        // dialects to deal with
        _context->appendDialectRegistry(ir::GetDialectRegistry());
        _context->loadAllAvailableDialects();
    }

protected:
    mlir::MLIRContext& context() { return *_context; }

private:
    std::unique_ptr<mlir::MLIRContext> _ownedContext;
    mlir::MLIRContext* _context;
};

MemoryLayout GetSubArrayLayout(const MemoryLayout& originalLayout, const MemoryShape& shape, const std::vector<int64_t>& stridesValue)
{
    return { originalLayout, shape, stridesValue };
}

MemoryLayout GetSliceLayout(const MemoryLayout& originalLayout, std::vector<int64_t> slicedDimensions)
{
    std::sort(slicedDimensions.begin(), slicedDimensions.end(), std::greater<int64_t>());

    MemoryLayout result = originalLayout;
    for (auto dim : slicedDimensions)
    {
        result = result.GetSliceLayout(dim);
    }
    return result;
}

MemoryLayout GetMergeDimLayout(const MemoryLayout& originalLayout, int64_t dim1, int64_t dim2)
{
    return originalLayout.GetMergedDimensionsLayout(dim1, dim2);
}

MemoryLayout GetSplitDimLayout(const MemoryLayout& originalLayout, int64_t dim, int64_t size)
{
    return originalLayout.GetSplitDimensionLayout(dim, size);
}

mlir::MemRefType MemoryLayoutToMemRefType(mlir::OpBuilder& builder, const MemoryLayout& layout, ValueType valueType, bool useDynamicOffset, int pointerLevel)
{
    auto mlirElemType = ValueTypeToMLIRType(builder, valueType);

    if (layout == ScalarLayout)
    {
        // represent pointers as memrefs of memrefs
        assert(pointerLevel <= 1 && "Only pointer levels <= 1 are currently supported for scalar layouts");
        return pointerLevel ? mlir::MemRefType::get(MemRefPointerShape, mlirElemType) : mlir::MemRefType::get({}, mlirElemType);
    }

    assert(pointerLevel <= 2 && "Only pointer levels <= 2 are currently supported for memref layouts");

    auto size = layout.GetActiveSize().ToVector();
    auto strides = layout.GetIncrement().ToVector();
    int64_t offset = useDynamicOffset ? mlir::MemRefType::getDynamicStrideOrOffset() : static_cast<int64_t>(layout.GetFirstEntryOffset());

    auto context = builder.getContext();
    auto stridedMap = mlir::makeStridedLinearLayoutMap(strides, offset, context);

    // strided maps and memory spaces are not supported for variable-sized layouts
    auto type = layout.IsVariableSized() ? mlir::MemRefType::get(size, mlirElemType) : mlir::MemRefType::get(size, mlirElemType, stridedMap, (unsigned)layout.GetMemorySpace());

    // represent pointers as memrefs of memrefs (memrefs start at pointer level 1)
    return (pointerLevel > 1) ? mlir::MemRefType::get(MemRefPointerShape, type) : type;
}

mlir::MemRefType MemoryLayoutToMemRefType(mlir::OpBuilder& builder, const MemoryLayout& layout, ValueType valueType)
{
    return MemoryLayoutToMemRefType(builder, layout, valueType, /*useDynamicOffset=*/false, /*pointerLevel=*/0);
}

auto MemoryLayoutToTensorType(mlir::OpBuilder& builder, const MemoryLayout& layout, ValueType valueType)
{
    // TODO: Figure out whether this assert needs to be active
    // assert(layout.IsCanonicalOrder() && "Can only get a tensor type from a canonically-ordered layout");

    auto mlirElemType = ValueTypeToMLIRType(builder, valueType);
    llvm::SmallVector<int64_t, 4> extents;

    extents.append(layout.GetExtent().begin(), layout.GetExtent().end());

    auto type = mlir::RankedTensorType::get(extents, mlirElemType);

    return type;
}

mlir::Type ToMLIRType(mlir::OpBuilder& builder, Value value)
{
    auto pointerLevel = value.PointerLevel();
    bool pointer = (pointerLevel > 0);
    if (value.IsConstrained())
    {
        auto& layout = value.GetLayout();
        if (!pointer && layout == ScalarLayout)
        {
            return ValueTypeToMLIRType(builder, value.GetBaseType());
        }
        else
        {
            return MemoryLayoutToMemRefType(builder, layout, value.GetBaseType(), /*useDynamicOffset=*/false, pointerLevel);
        }
    }
    else
    {
        auto mlirElemType = ValueTypeToMLIRType(builder, value.GetBaseType());
        auto type = mlir::UnrankedMemRefType::get(mlirElemType, 0);
        if (pointer)
        {
            // represent pointers as memrefs of memrefs
            return mlir::MemRefType::get(MemRefPointerShape, type);
        }
        else
        {
            return type; // casts from mlir::UnrankedMemRefType to mlir::Type
        }
    }
}

mlir::FunctionType ToMLIRType(mlir::OpBuilder& builder, const FunctionDeclaration& decl)
{
    const auto& argValues = decl.GetParameterTypes();
    const auto& returnValue = decl.GetReturnType();

    std::vector<mlir::Type> variableArgTypes(argValues.size());
    if (decl.UseMemRefDescriptorArgs())
    {
        std::transform(argValues.begin(), argValues.end(), variableArgTypes.begin(), [&builder](Value value) {
            return ToLLVMMemrefDescriptorType(builder, value);
        });
    }
    else
    {
        std::transform(argValues.begin(), argValues.end(), variableArgTypes.begin(), [&builder](Value value) {
            return ToMLIRType(builder, value);
        });
    }

    auto fnType = builder.getFunctionType(variableArgTypes, [&returnValue, &builder]() -> llvm::ArrayRef<mlir::Type> {
        if (!returnValue)
        {
            return llvm::None;
        }
        else
        {
            return ToMLIRType(builder, *returnValue);
        }
    }());

    return fnType;
}

[[nodiscard]] mlir::Value ResolveMLIRScalar(mlir::OpBuilder& builder, mlir::Value v)
{
    if (auto type = v.getType(); type.isIntOrIndexOrFloat())
    {
        return v;
    }
    else if (auto shapedType = type.dyn_cast<mlir::ShapedType>();
             shapedType &&
             shapedType.hasStaticShape() &&
             shapedType.getNumElements() == 1 &&
             shapedType.hasRank() &&
             shapedType.getRank() == 0)
    {
        auto loc = builder.getUnknownLoc();
        return builder.create<accera::ir::value::GetElementOp>(loc, v);
    }

    return v;
}

[[nodiscard]] mlir::Value ResolveMLIRIndex(mlir::OpBuilder& builder, mlir::Value v)
{
    v = ResolveMLIRScalar(builder, v);

    auto type = v.getType();
    if (type.isa<mlir::IntegerType>())
    {
        auto loc = builder.getUnknownLoc();
        return builder.create<mlir::arith::IndexCastOp>(loc, v, mlir::IndexType::get(v.getContext()));
    }

    // Index types fall through
    return v;
}

[[nodiscard]] mlir::Value ToMLIRValue(mlir::OpBuilder& builder, const ViewAdapter& view)
{
    auto value = view.GetValue();
    if (value.IsEmpty() || value.IsUndefined())
    {
        return {};
    }
    else
    {
        if (auto emittable = value.Get<Emittable>().GetDataAs<MLIRContext::EmittableInfo*>())
        {
            if (!emittable->isGlobal)
            {
                return mlir::Value::getFromOpaquePointer(emittable->data);
            }
            else
            {
                auto op = ir::value::GlobalOp::getFromOpaquePointer(emittable->data);

                auto insertionBlock = builder.getInsertionBlock();
                auto it = insertionBlock->begin();
                auto end = insertionBlock->end();
                while (it != end && llvm::isa<mlir::arith::ConstantOp,
                                              ir::value::ReferenceGlobalOp>(it))
                {
                    ++it;
                }

                auto loc = builder.getUnknownLoc();
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPoint(insertionBlock, it);
                return builder.create<ir::value::ReferenceGlobalOp>(loc, op);
            }
        }
    }

    return {};
}

[[nodiscard]] mlir::Value ResolveMLIRIndex(mlir::OpBuilder& builder, Scalar s)
{
    return ResolveMLIRIndex(builder, ResolveMLIRScalar(builder, ToMLIRValue(builder, s)));
}

std::vector<mlir::Value> ToMLIRValue(mlir::OpBuilder& builder, std::vector<Value> values)
{
    std::vector<mlir::Value> mlirValues;
    mlirValues.reserve(values.size());
    std::transform(
        values.begin(),
        values.end(),
        std::back_inserter(mlirValues),
        [&builder](Value value) { return ToMLIRValue(builder, value); });
    return mlirValues;
}

[[nodiscard]] mlir::Value ToMLIRIndex(mlir::OpBuilder& builder, Scalar index)
{
    auto mlirValue = ResolveMLIRScalar(builder, ToMLIRValue(builder, index));
    auto indexType = builder.getIndexType();
    if (mlirValue.getType().isIndex())
    {
        return mlirValue;
    }
    else
    {
        auto loc = builder.getUnknownLoc();
        return builder.create<mlir::arith::IndexCastOp>(loc, mlirValue, indexType);
    }
}

mlir::Type ToLLVMMemrefDescriptorType(mlir::OpBuilder& builder, ::accera::value::Value value)
{
    // MemRefDescriptor structs have the following structure:
    // struct MemRefDescriptor
    // {
    //     T* allocated;
    //     T* aligned;
    //     int64_t offset;
    //     int64_t sizes[N];
    //     int64_t strides[N];
    // };

    using namespace mlir;
    auto context = builder.getContext();

    mlir::LLVMTypeConverter llvmTypeConverter(context);

    auto mlirElementType = ValueTypeToMLIRType(builder, value.GetBaseType());
    auto llvmElementType = llvmTypeConverter.convertType(mlirElementType);
    auto rank = value.GetLayout().NumDimensions();

    auto llvmPtrToElementType = LLVM::LLVMPointerType::get(llvmElementType);
    auto int64Type = IntegerType::get(context, 64);
    auto llvmArrayRankElementSizeType = LLVM::LLVMArrayType::get(int64Type, rank);

    auto memRefTy = LLVM::LLVMStructType::getLiteral(context,
                                                     { llvmPtrToElementType, llvmPtrToElementType, int64Type, llvmArrayRankElementSizeType, llvmArrayRankElementSizeType });
    return LLVM::LLVMPointerType::get(memRefTy);
}

void SetOpNameAttr(mlir::Operation* op, std::string name)
{
    if (!name.empty())
    {
        assert(op);

        if (auto symTableOp = mlir::SymbolTable::getNearestSymbolTable(op); !symTableOp)
        {
            llvm::errs() << "Could not find symbol table for operation " << *op << "\n";
        }
        else if (mlir::SymbolTable::lookupSymbolIn(symTableOp, name))
        {
            name += "_" + std::to_string(ir::util::GetUniqueId(op));
        }

        mlir::SymbolTable::setSymbolName(op, name);
    }
}

auto GetConstantDataElementType(const ConstantData& data)
{
    return std::visit(
        [](auto&& data_) {
            using DataType = std::decay_t<decltype(data_)>;
            using ElementType = typename DataType::value_type;

            return GetValueType<ElementType>();
        },
        data);
}

auto ConstantDataToDenseElementAttr(mlir::ShapedType shape, const ConstantData& data)
{
    return std::visit(
        [shape](auto&& data_) -> mlir::DenseElementsAttr {
            using DataType = std::decay_t<decltype(data_)>;
            using ElementType = typename DataType::value_type;

            if constexpr (std::is_same_v<ElementType, Boolean>)
            {
                std::vector<int8_t> boolData(data_.size());
                std::transform(data_.begin(), data_.end(), boolData.begin(), [](Boolean b) { return b ? 1 : 0; });

                return mlir::DenseElementsAttr::get(shape, llvm::makeArrayRef(boolData));
            }
            else if constexpr (std::is_same_v<ElementType, index_t>)
            {
                throw InputException(InputExceptionErrors::invalidArgument, "Can't store an array of index type");
            }
            else if constexpr (std::is_same_v<ElementType, float16_t>)
            {
                using float16_underlying_type = typename float16_t::underlying_type;
                std::vector<float16_underlying_type> fp16Data(data_.size());
                std::transform(data_.begin(), data_.end(), fp16Data.begin(), [](float16_t value) { return value.data; });

                return mlir::DenseElementsAttr::get(shape, llvm::makeArrayRef(fp16Data));
            }
            else if constexpr (std::is_same_v<ElementType, bfloat16_t>)
            {
                using bfloat16_underlying_type = typename bfloat16_t::underlying_type;
                std::vector<bfloat16_underlying_type> bfp16Data(data_.size());
                std::transform(data_.begin(), data_.end(), bfp16Data.begin(), [](bfloat16_t value) { return value.data; });

                return mlir::DenseElementsAttr::get(shape, llvm::makeArrayRef(bfp16Data));
            }
            else
            {
                return mlir::DenseElementsAttr::get(shape, llvm::makeArrayRef(data_));
            }
        },
        data);
}

MemoryLayout InferLayoutFromMLIRValue(mlir::Value value)
{
    auto type = value.getType();
    if (type.isIntOrIndexOrFloat())
    {
        return ScalarLayout;
    }
    else if (auto memRefType = type.dyn_cast<mlir::MemRefType>())
    {
        // bail early for the simple case
        if (memRefType.getNumElements() == 1)
        {
            return ScalarLayout;
        }
        llvm::SmallVector<int64_t, 4> strides;
        int64_t globalOffset;
        if (failed(getStridesAndOffset(memRefType, strides, globalOffset)))
        {
            throw std::logic_error{ "Resource to be filled in must be valid memory" };
        }

        auto rank = memRefType.getRank();
        std::vector<int64_t> offset(rank, 0);

        // Need to sort strides to get permutation order
        std::vector<int64_t> order(rank);
        std::iota(order.begin(), order.end(), 0);

        auto zip1 = llvm::zip(strides, order);
        std::vector<std::tuple<int64_t, int64_t>> stridesAndOrder(zip1.begin(), zip1.end());
        std::sort(stridesAndOrder.begin(), stridesAndOrder.end(), [](auto a, auto b) { return std::get<0>(a) > std::get<0>(b); });
        std::transform(stridesAndOrder.begin(), stridesAndOrder.end(), strides.begin(), [](auto el) { return std::get<0>(el); });
        std::transform(stridesAndOrder.begin(), stridesAndOrder.end(), order.begin(), [](auto el) { return std::get<1>(el); });

        auto shape = memRefType.getShape();
        auto memorySize = strides.front() * shape[order[0]];

        // Compute extents by dividing previous stride by stride (except for the largest dimension, where we just use the size)
        std::vector<int64_t> extent(strides.begin(), strides.end());
        int64_t prevStride = memorySize;
        for (auto& e : extent)
        {
            auto temp = e;
            e = prevStride / e;
            prevStride = temp;
        }

        auto zip2 = llvm::zip(order, extent);
        std::vector<std::tuple<int64_t, int64_t>> permAndExtent(zip2.begin(), zip2.end());
        std::sort(permAndExtent.begin(), permAndExtent.end(), [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); });
        std::transform(permAndExtent.begin(), permAndExtent.end(), extent.begin(), [](auto el) { return std::get<1>(el); });

        auto result = MemoryLayout{ MemoryShape{ shape.vec() }, MemoryShape{ extent }, MemoryShape{ offset }, DimensionOrder{ order } };
        return result;
    }
    else if (auto shapedType = type.dyn_cast<mlir::ShapedType>())
    {
        auto shape = shapedType.getShape();
        return MemoryLayout{ shape.vec() };
    }
    else
    {
        throw std::logic_error("Unknown value type");
    }
}
} // namespace

namespace accera::value
{

bool HasMLIRTypeConversion(ValueType type)
{
    switch (type)
    {
    case ValueType::Boolean:
        [[fallthrough]];
    case ValueType::Byte:
        [[fallthrough]];
    case ValueType::Int8:
        [[fallthrough]];
    case ValueType::Int16:
        [[fallthrough]];
    case ValueType::Int32:
        [[fallthrough]];
    case ValueType::Int64:
        [[fallthrough]];
    case ValueType::Uint16:
        [[fallthrough]];
    case ValueType::Uint32:
        [[fallthrough]];
    case ValueType::Uint64:
        [[fallthrough]];
    case ValueType::Index:
        [[fallthrough]];
    case ValueType::Float16:
        [[fallthrough]];
    case ValueType::BFloat16:
        [[fallthrough]];
    case ValueType::Float:
        [[fallthrough]];
    case ValueType::Double:
        return true;
    case ValueType::Void:
        [[fallthrough]];
    case ValueType::Undefined:
        [[fallthrough]];
    default:
        return false;
    }
}

mlir::Type ValueTypeToMLIRType(mlir::OpBuilder& builder, ValueType type)
{
    switch (type)
    {
    // Signed ints are treated as "signless" (i.e. represented without a sign).
    // Unsigned ints are treated as "non-signless" (i.e. represented with a sign)
    // Non-signless ints must to be converted to signless ints (preserving their
    // underlying values) before calling standard / arithmetic ops
    case ValueType::Boolean:
        return builder.getIntegerType(1);
    case ValueType::Byte: // = Uint8
        return builder.getIntegerType(8, /*isSigned=*/false);
    case ValueType::Int8:
        return builder.getIntegerType(8);
    case ValueType::Int16:
        return builder.getIntegerType(16);
    case ValueType::Int32:
        return builder.getIntegerType(32);
    case ValueType::Int64:
        return builder.getIntegerType(64);
    case ValueType::Uint16:
        return builder.getIntegerType(16, /*isSigned=*/false);
    case ValueType::Uint32:
        return builder.getIntegerType(32, /*isSigned=*/false);
    case ValueType::Uint64:
        return builder.getIntegerType(64, /*isSigned=*/false);
    case ValueType::Index:
        return builder.getIndexType();
    case ValueType::Float16:
        return builder.getF16Type();
    case ValueType::BFloat16:
        return builder.getBF16Type();
    case ValueType::Float:
        return builder.getF32Type();
    case ValueType::Double:
        return builder.getF64Type();
    case ValueType::Void:
        [[fallthrough]];
    case ValueType::Undefined:
        [[fallthrough]];
    default:
        throw LogicException(LogicExceptionErrors::illegalState, "Unknown type conversion");
    }
}

GPUIndex::GPUIndex(std::function<Scalar(const GPUIndexDimension)> fn) :
    _fn(std::move(fn))
{}

Scalar GPUIndex::X()
{
    return _fn(GPUIndexDimension::X);
}
Scalar GPUIndex::Y()
{
    return _fn(GPUIndexDimension::Y);
}
Scalar GPUIndex::Z()
{
    return _fn(GPUIndexDimension::Z);
}

struct MLIRContextBase::Impl : private InitAccera
{
    friend class MLIRContext;

    Impl(const std::string& moduleName) :
        InitAccera(nullptr),
        _ownedModule(mlir::ModuleOp::create(
            mlir::UnknownLoc::get(&context()),
            llvm::StringRef(moduleName))),
        _mlirModule(*_ownedModule),
        builder(&context()),
        sourceMgrHandler(sourceMgr, &context()),
        _valueModuleOp(CreateValueModuleOp(*_ownedModule))
    {
        builder.setInsertionPoint(module().getBody(), module().getBody()->begin());
    }

    Impl(mlir::ModuleOp moduleOp) :
        InitAccera(moduleOp.getContext()),
        _mlirModule(moduleOp),
        builder(&context()),
        sourceMgrHandler(sourceMgr, &context()),
        _valueModuleOp(CreateValueModuleOp(moduleOp))
    {
        builder.setInsertionPoint(module().getBody(), module().getBody()->begin());
    }

    /// Globals are inserted before the first function, if any.
    mlir::OpBuilder::InsertPoint getGlobalInsertPt()
    {
        return ir::util::GetTerminalInsertPoint<
            ir::value::ValueModuleOp,
            ir::value::ModuleTerminatorOp,
            ir::value::ValueFuncOp>(module());
    }

    /// Funcs are inserted before the module terminator
    mlir::OpBuilder::InsertPoint getFunctionInsertPt()
    {
        return ir::util::GetTerminalInsertPoint<
            ir::value::ValueModuleOp,
            ir::value::ModuleTerminatorOp>(module());
    }

    mlir::OpBuilder::InsertionGuard CreateNewScope(mlir::OpBuilder::InsertPoint insertionPoint = {})
    {
        mlir::OpBuilder::InsertionGuard guard(builder);

        if (insertionPoint.isSet())
        {
            builder.restoreInsertionPoint(insertionPoint);
        }

        return guard;
    }

    ir::value::ValueModuleOp CreateValueModuleOp(mlir::ModuleOp moduleOp)
    {
        auto possibleModules = moduleOp.getOps<ir::value::ValueModuleOp>();
        assert((possibleModules.empty() || llvm::hasSingleElement(possibleModules)) && "Multiple value modules is untested");

        ir::value::ValueModuleOp valueModuleOp;
        if (possibleModules.empty())
        {
            auto moduleBody = moduleOp.getBody();
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(moduleBody, moduleBody->end());
            valueModuleOp = builder.create<ir::value::ValueModuleOp>(moduleOp.getLoc(), moduleOp.getName().getValueOr("value_module"));
            assert(valueModuleOp.getOperation()->getNumRegions() == 1);
        }
        else
        {
            valueModuleOp = *possibleModules.begin();
        }
        return valueModuleOp;
    }

    ir::value::ValueModuleOp module() const { return _valueModuleOp; }

    void setDataLayout(const std::string& layout)
    {
        _mlirModule->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(), builder.getStringAttr(layout));
    }

protected:
    mlir::OwningOpRef<mlir::ModuleOp> _ownedModule;
    mlir::ModuleOp _mlirModule;

public:
    mlir::OpBuilder builder;

private:
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler;
    mlir::gpu::GPUModuleOp _gpuModuleOp = nullptr;
    ir::value::ValueModuleOp _valueModuleOp;
};

MLIRContextBase::MLIRContextBase(const std::string& moduleName) :
    _impl(std::make_unique<MLIRContextBase::Impl>(moduleName))
{
}

MLIRContextBase::MLIRContextBase(mlir::ModuleOp& existingModule) :
    _impl(std::make_unique<MLIRContextBase::Impl>(existingModule))
{
}

MLIRContext::EmittableInfo& MLIRContext::StoreGlobalEmittable(EmittableInfo emittable)
{
    emittable.isGlobal = true;

    std::lock_guard lock{ _mutex };

    _globalEmittables.push_front(emittable);
    return _globalEmittables.front();
}

MLIRContext::EmittableInfo& MLIRContext::StoreLocalEmittable(EmittableInfo emittable)
{
    std::lock_guard lock{ _mutex };
    assert(!_localEmittables.empty());
    _localEmittables.top().push_front(emittable);
    return _localEmittables.top().front();
}

MLIRContext::MLIRContext(const std::string& moduleName, const CompilerOptions& options) :
    MLIRContextBase(moduleName),
    EmitterContext(options)
{
    setDataLayout(options);
    setDebugMode(options.debug);
    _localEmittables.push({});
}

MLIRContext::MLIRContext(mlir::ModuleOp& existingModule, const CompilerOptions& options) :
    MLIRContextBase(existingModule),
    EmitterContext(options)
{
    setDataLayout(options);
    setDebugMode(options.debug);
    _localEmittables.push({});
}

MLIRContext::~MLIRContext() = default;

void MLIRContext::save(std::string filename) const
{
    std::error_code ec;
    llvm::raw_fd_ostream s(filename, ec);

    mlir::OwningOpRef<mlir::ModuleOp> cloned = _impl->_mlirModule.clone();
    SaveModule(filename, cloned.get());
}

void MLIRContext::print() const
{
    llvm::raw_os_ostream s(std::cout);
    _impl->_mlirModule.print(s);
}

void MLIRContext::verify() const
{
    (void)_impl->_mlirModule.verify();
}

mlir::OwningOpRef<mlir::ModuleOp> MLIRContext::cloneModule() const
{
    return _impl->_mlirModule.clone();
}

void MLIRContext::writeHeader(std::optional<std::string> filename) const
{
    using llvm::raw_fd_ostream;
    using llvm::raw_ostream;

    std::unique_ptr<raw_fd_ostream> fstream;

    if (filename)
    {
        std::error_code ec;
        fstream = std::make_unique<raw_fd_ostream>(*filename, ec);
    }

    raw_ostream& stream = [&]() -> raw_ostream& {
        if (fstream)
        {
            return *fstream;
        }
        else
            return llvm::outs();
    }();

    (void)ir::TranslateToHeader(_impl->_mlirModule, stream);
}

void MLIRContext::setMetadata(const std::string& key, const accera::ir::MetadataValueType& value)
{
    auto context = _impl->_valueModuleOp.getContext();
    auto newKeyId = mlir::StringAttr::get(context, key);
    auto valueAttr = ir::GetMetadataAttr(value, context);
    auto metadataKeyId = mlir::StringAttr::get(context, accera::ir::value::ValueDialect::getAcceraMetadataAttrName());
    auto metadataDict = _impl->_valueModuleOp->getAttrOfType<mlir::DictionaryAttr>(accera::ir::value::ValueDialect::getAcceraMetadataAttrName());
    if (metadataDict == nullptr)
    {
        mlir::NamedAttrList mutableDict;
        mutableDict.set(newKeyId, valueAttr);
        _impl->_valueModuleOp->setAttr(metadataKeyId, mutableDict.getDictionary(context));
    }
    else
    {
        mlir::NamedAttrList mutableDict(metadataDict);
        mutableDict.set(newKeyId, valueAttr);
        _impl->_valueModuleOp->setAttr(metadataKeyId, mutableDict.getDictionary(context));
    }
}

accera::ir::Metadata MLIRContext::getFullMetadata()
{
    return ir::ParseFullMetadata(_impl->_valueModuleOp->getAttrOfType<mlir::DictionaryAttr>(accera::ir::value::ValueDialect::getAcceraMetadataAttrName()));
}

void MLIRContext::setDataLayout(const CompilerOptions& options)
{
    if (const auto& layout = options.targetDevice.dataLayout; !layout.empty())
    {
        _impl->setDataLayout(layout);
    }
}

void MLIRContext::setDebugMode(bool enable)
{
    // moduleOp-wide debug attribute for generating debug sprintfs in the module header file
    auto& builder = _impl->builder;
    auto context = _impl->_valueModuleOp.getContext();
    auto debugModeId = mlir::StringAttr::get(context, accera::ir::GetDebugModeAttrName());
    if (enable)
    {
        _impl->_valueModuleOp->setAttr(debugModeId, builder.getUnitAttr());
    }
    else
    {
        _impl->_valueModuleOp->removeAttr(debugModeId);
    }
}

Scalar CreateGPUIndexOp(mlir::OpBuilder& builder, accera::ir::value::Processor idxType)
{
    auto loc = builder.getUnknownLoc();
    return Wrap(
        builder.create<mlir::arith::IndexCastOp>(loc,
                                                 accera::ir::util::GetGPUIndex(idxType, builder, loc),
                                                 builder.getI64Type()));
}

template <GPUIndexType type>
GPUIndex MLIRContext::GetGPUIndex()
{
    switch (type)
    {
    case GPUIndexType::BlockDim:
        return GPUIndex{ [this](const GPUIndexDimension dim) {
            switch (dim)
            {
            case GPUIndexDimension::X:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::BlockDimX);
            case GPUIndexDimension::Y:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::BlockDimY);
            case GPUIndexDimension::Z:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::BlockDimZ);
            default:
                llvm_unreachable("Unknown GPU index dimension");
            }
        } };
    case GPUIndexType::BlockId:
        return GPUIndex{ [this](const GPUIndexDimension dim) {
            switch (dim)
            {
            case GPUIndexDimension::X:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::BlockX);
            case GPUIndexDimension::Y:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::BlockY);
            case GPUIndexDimension::Z:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::BlockZ);
            default:
                llvm_unreachable("Unknown GPU index dimension");
            }
        } };
    case GPUIndexType::GridDim:
        return GPUIndex{ [this](const GPUIndexDimension dim) {
            switch (dim)
            {
            case GPUIndexDimension::X:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::GridDimX);
            case GPUIndexDimension::Y:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::GridDimY);
            case GPUIndexDimension::Z:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::GridDimZ);
            default:
                llvm_unreachable("Unknown GPU index dimension");
            }
        } };
    case GPUIndexType::ThreadId:
        return GPUIndex{ [this](const GPUIndexDimension dim) {
            switch (dim)
            {
            case GPUIndexDimension::X:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::ThreadX);
            case GPUIndexDimension::Y:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::ThreadY);
            case GPUIndexDimension::Z:
                return CreateGPUIndexOp(_impl->builder, accera::ir::value::Processor::ThreadZ);
            default:
                llvm_unreachable("Unknown GPU index dimension");
            }
        } };
    default:
        llvm_unreachable("Unknown GPU index type");
    }
}

Value MLIRContext::AllocateImpl(ValueType valueType, MemoryLayout layout, size_t alignment, AllocateFlags flags, const std::vector<ScalarDimension>& runtimeSizes)
{
    auto& b = _impl->builder;
    if (layout.GetMemorySpace() == MemorySpace::None)
    {
        layout = layout.SetMemorySpace(MemorySpace::Shared);
    }
    auto memrefTy = MemoryLayoutToMemRefType(b, layout, valueType);

    mlir::OpBuilder::InsertionGuard guard(b);
    // Place the alloc op at the beginning of the block (after other allocs), unless it depends
    // on runtime sizes that are defined before this
    if (runtimeSizes.empty())
    {
        auto insertionBlock = b.getInsertionBlock();
        auto it = insertionBlock->begin();
        auto end = insertionBlock->end();
        while (it != end && llvm::isa<mlir::arith::ConstantOp,
                                      mlir::memref::AllocOp,
                                      mlir::memref::AllocaOp,
                                      ir::value::ReferenceGlobalOp,
                                      ir::value::AllocOp>(it))
        {
            ++it;
        }
        b.setInsertionPoint(insertionBlock, it);
    }
    auto loc = b.getUnknownLoc();

    std::vector<mlir::Value> sizes;
    std::transform(runtimeSizes.cbegin(), runtimeSizes.cend(), std::back_inserter(sizes), [](ScalarDimension d) { return Unwrap(d); });
    
    mlir::Value result;
    if (layout.IsVariableSized())
    {
        result = b.create<ir::value::AllocOp>(loc,
                                            memrefTy,
                                            alignment
                                                ? llvm::Optional{ (int64_t)alignment }
                                                : llvm::None,
                                            static_cast<bool>(flags & AllocateFlags::Stack)
                                                ? llvm::Optional{ accera::ir::value::MemoryAllocType::Stack }
                                                : llvm::None,
                                            mlir::ValueRange{ sizes});
    }
    else
    {
        result = b.create<ir::value::AllocOp>(loc,
                                            memrefTy,
                                            alignment
                                                ? llvm::Optional{ (int64_t)alignment }
                                                : llvm::None,
                                            static_cast<bool>(flags & AllocateFlags::Stack)
                                                ? llvm::Optional{ accera::ir::value::MemoryAllocType::Stack }
                                                : llvm::None);
    }

    EmittableInfo& emittableInfo = StoreLocalEmittable({ result.getAsOpaquePointer(), { valueType, 1 } });
    Emittable emittable{ &emittableInfo };

    return Value(emittable, layout);
}

std::optional<Value> MLIRContext::GetGlobalValue(GlobalAllocationScope scope, std::string name)
{
    std::string adjustedName = GetScopeAdjustedName(scope, name);
    if (auto it = _globals.find(adjustedName); it != _globals.end())
    {
        return Value(it->second.first, it->second.second);
    }

    return std::nullopt;
}

Value MLIRContext::GlobalAllocateImpl(GlobalAllocationScope allocScope, std::string name, ConstantData data, MemoryLayout layout, AllocateFlags flags)
{
    std::string adjustedName = GetScopeAdjustedName(allocScope, name);

    if (_globals.find(adjustedName) != _globals.end())
    {
        throw InputException(InputExceptionErrors::invalidArgument,
                             "Unexpected collision in global data allocation");
    }

    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    auto scope = _impl->CreateNewScope(_impl->getGlobalInsertPt());

    auto valueElemType = GetConstantDataElementType(data);

    auto memrefType = MemoryLayoutToMemRefType(builder, layout, valueElemType);
    auto dataType = MemoryLayoutToTensorType(builder, layout, valueElemType);
    auto dataAttribute = ConstantDataToDenseElementAttr(dataType, data);

    auto global = builder.create<ir::value::GlobalOp>(loc, memrefType, /*isConstant=*/true, adjustedName, dataAttribute);

    EmittableInfo& emittableInfo = StoreGlobalEmittable({ global, { valueElemType, 1 } });
    Emittable emittable(&emittableInfo);

    _globals[adjustedName] = { emittable, layout };

    return Value(emittable, layout);
}

Value MLIRContext::GlobalAllocateImpl(GlobalAllocationScope allocScope, std::string name, ValueType type, MemoryLayout layout, AllocateFlags flags)
{
    std::string adjustedName = GetScopeAdjustedName(allocScope, name);

    if (_globals.find(adjustedName) != _globals.end())
    {
        throw InputException(InputExceptionErrors::invalidArgument,
                             "Unexpected collision in global data allocation");
    }

    auto scope = _impl->CreateNewScope(_impl->getGlobalInsertPt());
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    auto memrefType = MemoryLayoutToMemRefType(builder, layout, type);

    auto global = builder.create<ir::value::GlobalOp>(loc, memrefType, /*isConstant=*/false, adjustedName, mlir::Attribute{});

    EmittableInfo& emittableInfo = StoreGlobalEmittable({ global, { type, 1 } });
    Emittable emittable(&emittableInfo);

    {
        std::lock_guard lock{ _mutex };
        _globals[adjustedName] = { emittable, layout };
    }

    return Value(emittable, layout);
}

detail::ValueTypeDescription MLIRContext::GetTypeImpl(Emittable emittable)
{
    auto info = emittable.GetDataAs<EmittableInfo*>();
    return info->desc;
}

EmitterContext::DefinedFunction MLIRContext::CreateFunctionImpl(FunctionDeclaration decl, DefinedFunction fn)
{
    {
        std::lock_guard lock{ _mutex };
        if (auto it = _definedFunctions.find(decl); it != _definedFunctions.end())
        {
            return it->second;
        }
    }

    auto& b = _impl->builder;
    auto loc = b.getUnknownLoc();

    auto isPublic = decl.IsPublic();
    auto funcTarget = decl.Target();
    auto funcRuntime = decl.Runtime();
    auto isGpu = std::holds_alternative<targets::GPU>(funcTarget);

    const auto& argValues = decl.GetParameterTypes();
    const auto& returnValue = decl.GetReturnType();

    const auto& fnName = decl.GetFunctionName();
    auto argValuesCopy = argValues;
    auto fnType = ::ToMLIRType(b, decl);

    auto [fnOp, entryBlock] = std::visit(
        [&](auto target) {
            ir::value::ExecutionTarget executionTarget;
            if constexpr (std::is_same_v<decltype(target), targets::CPU>)
            {
                executionTarget = ir::value::ExecutionTarget::CPU;
            }
            else if constexpr (std::is_same_v<decltype(target), targets::GPU>)
            {
                executionTarget = ir::value::ExecutionTarget::GPU;
            }

            mlir::OpBuilder::InsertionGuard guard(b);
            b.restoreInsertionPoint(_impl->getFunctionInsertPt());

            ir::value::ValueFuncOp fnOp = b.create<ir::value::ValueFuncOp>(loc,
                                                                           fnName,
                                                                           fnType,
                                                                           executionTarget);

            mlir::SymbolTable::setSymbolVisibility(fnOp, isPublic ? mlir::SymbolTable::Visibility::Public : mlir::SymbolTable::Visibility::Nested);

            if (decl.EmitsCWrapper())
            {
                // Emit C wrappers for any functions defined in this module
                fnOp->setAttr(ir::CInterfaceAttrName, b.getUnitAttr());
            }
            if (decl.UseRawPointerAPI())
            {
                fnOp->setAttr(ir::RawPointerAPIAttrName, b.getUnitAttr());
            }
            if (decl.EmitsHeaderDecl())
            {
                fnOp->setAttr(ir::HeaderDeclAttrName, b.getUnitAttr());
            }
            if (decl.InlineState() == FunctionInlining::never)
            {
                fnOp->setAttr(ir::NoInlineAttrName, b.getUnitAttr());
            }
            if (auto checkFunctions = decl.GetOutputVerifiers(); !checkFunctions.empty())
            {
                // For each input_output parameter, set its check function
                // TODO: emit these in DebugFunctionPass by plumbing the tolerance and parameter usage
                size_t checkFunctionIdx = 0;
                std::vector<mlir::Attribute> checkFunctionAttrs;
                for (const auto& usage : decl.GetParameterUsages())
                {
                    if (usage == FunctionParameterUsage::inputOutput)
                    {
                        checkFunctionAttrs.push_back(b.getStringAttr(checkFunctions[checkFunctionIdx++]));
                    }
                    else if (usage == FunctionParameterUsage::input)
                    {
                        // input parameter, set an empty string
                        checkFunctionAttrs.push_back(b.getStringAttr(""));
                    }
                    else
                    {
                        // TODO: implement this
                        assert(false && "Output parameter usage is not yet supported");
                    }
                }
                fnOp->setAttr(ir::GetOutputVerifiersAttrName(), b.getArrayAttr(checkFunctionAttrs));
            }
            if (auto argumentsSymbol = decl.GetArgumentsSymbol(); !argumentsSymbol.empty())
            {
                std::vector<mlir::Attribute> argumentsSymbolAttrs;
                for (const auto& arg : argumentsSymbol)
                {
                    argumentsSymbolAttrs.push_back(b.getStringAttr(arg));
                }
                fnOp->setAttr(ir::value::ValueFuncOp::getArgumentsSymbolAttrName(), b.getArrayAttr(argumentsSymbolAttrs));
            }

            // Collect function tags into a dictionary
            auto tags = decl.GetTags();
            std::vector<mlir::NamedAttribute> tagAttrs;
            if (!tags.empty())
            {
                for (const auto& tag : tags)
                {
                    tagAttrs.emplace_back(mlir::NamedAttribute(b.getStringAttr(tag), b.getUnitAttr()));
                }
                fnOp->setAttr(ir::FunctionTagsAttrName, b.getDictionaryAttr(tagAttrs));
            }

            auto baseName = decl.GetBaseName();
            if (!baseName.empty())
            {
                fnOp->setAttr(ir::BaseNameAttrName, b.getStringAttr(baseName));
            }

            if constexpr (std::is_same_v<decltype(target), targets::GPU>)
            {
                if (funcRuntime != ExecutionRuntime::DEFAULT)
                {
                    auto execRuntimeAttrName = ir::value::ValueModuleOp::getExecRuntimeAttrName();
                    auto execRuntimeAttrValue = ir::value::ExecutionRuntimeAttr::get(b.getContext(), (ir::value::ExecutionRuntime)funcRuntime);
                    if (auto mod = fnOp->getParentOfType<mlir::ModuleOp>())
                    {
                        mod->setAttr(execRuntimeAttrName, execRuntimeAttrValue);
                    }
                    if (auto mod = fnOp->getParentOfType<ir::value::ValueModuleOp>())
                    {
                        mod->setAttr(execRuntimeAttrName, execRuntimeAttrValue);
                    }
                }

                fnOp->setAttr(
                    fnOp.getGPULaunchAttrName(),
                    target.ToArrayAttr(b.getContext()));
            }

            return std::pair{ fnOp.getOperation(), &fnOp.body().back() };
        },
        funcTarget);

    {
        auto fnContext = _impl->CreateNewScope({ entryBlock, entryBlock->begin() });
        mlir::OpBuilder::InsertionGuard guard(b);
        b.restoreInsertionPoint({ entryBlock, entryBlock->begin() });

        {
            std::lock_guard lock{ _mutex };
            _localEmittables.push({});
        }

        for (auto zipped : llvm::zip(argValuesCopy, entryBlock->getArguments()))
        {
            Value& value = std::get<0>(zipped);
            EmittableInfo& emittableInfo = StoreLocalEmittable({ std::get<1>(zipped).getAsOpaquePointer(), value.GetType() });
            Emittable emittable(&emittableInfo);
            value.SetData(emittable);
        }

        auto returnValueCopy = returnValue;
        try
        {
            returnValueCopy = fn(argValuesCopy);
            if (returnValueCopy)
            {
                assert(!isGpu);

                (void)b.create<accera::ir::value::ReturnOp>(loc, ToMLIRValue(b, *returnValueCopy));
            }
            else
            {
                (void)b.create<accera::ir::value::ReturnOp>(loc);
            }
        }
        catch (...)
        {
            llvm::errs() << "Error when building function " << fnName << "\n";
            (void)b.create<accera::ir::value::ReturnOp>(loc);
            throw;
        }

        {
            std::lock_guard lock{ _mutex };
            _localEmittables.pop();
        }
    }

    DefinedFunction returnFn = [this, fnOp = fnOp, decl, mlirExpectedValues = argValuesCopy, isGpu](std::vector<Value> args) -> std::optional<Value> {
        const auto& argValues = decl.GetParameterTypes();
        const auto& returnValue = decl.GetReturnType();

        if (!std::equal(args.begin(),
                        args.end(),
                        argValues.begin(),
                        argValues.end(),
                        [](Value suppliedValue, Value fnValue) {
                            return suppliedValue.GetBaseType() == fnValue.GetBaseType();
                        }))
        {
            throw InputException(InputExceptionErrors::invalidArgument, __FILE__ " : " + std::to_string(__LINE__));
        }

        auto& builder = _impl->builder;
        auto loc = builder.getUnknownLoc();
        std::vector<mlir::Value> mlirArgs = ToMLIRValue(builder, args);
        auto returnValueCopy = returnValue;

        mlir::Operation* callOp = builder.create<ir::value::LaunchFuncOp>(loc, mlir::cast<ir::value::ValueFuncOp>(fnOp), mlirArgs);

        if (returnValueCopy)
        {
            assert(callOp->getNumResults() == 1);
            assert(!isGpu);

            EmittableInfo& emittableInfo = StoreLocalEmittable({ callOp->getResult(0).getAsOpaquePointer(), returnValueCopy->GetType() });
            returnValueCopy->SetData(Emittable{ &emittableInfo });
        }
        else
        {
            assert(callOp->getNumResults() == 0);
        }
        return returnValueCopy;
    };

    {
        std::lock_guard lock{ _mutex };
        _definedFunctions[decl] = returnFn;
    }

    return returnFn;
}

EmitterContext::DefinedFunction MLIRContext::DeclareExternalFunctionImpl(FunctionDeclaration decl)
{
    {
        std::lock_guard lock{ _mutex };
        if (auto it = _definedFunctions.find(decl); it != _definedFunctions.end())
        {
            return it->second;
        }
    }

    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    auto name = decl.GetFunctionName();
    auto fnTy = ::ToMLIRType(builder, decl);
    auto isPublic = decl.IsPublic();

    auto insertionBlock = builder.getBlock();
    auto parentOp = insertionBlock->getParentOp();
    auto mod = accera::ir::util::CastOrGetParentOfType<mlir::ModuleOp>(parentOp);
    assert(mod);

    if (auto fnOp = mod.lookupSymbol<ir::value::ValueFuncOp>(name); fnOp)
    {
        throw InputException(InputExceptionErrors::invalidArgument, "Cannot emit an extern decl for a function that is defined in this module");
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.restoreInsertionPoint(_impl->getGlobalInsertPt());
    // insert this in the mlir::ModuleOp outside of the ValueModuleOp
    ir::value::ValueFuncOp fnOp = builder.create<ir::value::ValueFuncOp>(loc,
                                                                         name,
                                                                         fnTy,
                                                                         ir::value::ExecutionTarget::CPU,
                                                                         ir::value::ValueFuncOp::ExternalFuncTag{});

    mlir::SymbolTable::setSymbolVisibility(fnOp, isPublic ? mlir::SymbolTable::Visibility::Public : mlir::SymbolTable::Visibility::Private);

    if (decl.EmitsCWrapper())
    {
        // Emit C wrappers for any functions defined in this module
        fnOp->setAttr(ir::CInterfaceAttrName, builder.getUnitAttr());
    }

    // Add to _definedFunctions so we can call it the normal way

    DefinedFunction returnFn = [this, fnOp = fnOp, decl](std::vector<Value> args) -> std::optional<Value> {
        const auto& argValues = decl.GetParameterTypes();
        const auto& returnValue = decl.GetReturnType();

        if (!std::equal(args.begin(),
                        args.end(),
                        argValues.begin(),
                        argValues.end(),
                        [](Value suppliedValue, Value fnValue) {
                            return suppliedValue.GetBaseType() == fnValue.GetBaseType();
                        }))
        {
            throw InputException(InputExceptionErrors::invalidArgument, __FILE__ " : " + std::to_string(__LINE__));
        }

        auto& builder = _impl->builder;
        auto loc = builder.getUnknownLoc();

        std::vector<mlir::Value> mlirArgs = ToMLIRValue(builder, args);
        auto returnValueCopy = returnValue;

        mlir::Operation* callOp = builder.create<ir::value::LaunchFuncOp>(loc, mlir::cast<ir::value::ValueFuncOp>(fnOp), mlirArgs);

        if (returnValueCopy)
        {
            assert(callOp->getNumResults() == 1);

            EmittableInfo& emittableInfo = StoreLocalEmittable({ callOp->getResult(0).getAsOpaquePointer(), returnValueCopy->GetType() });
            returnValueCopy->SetData(Emittable{ &emittableInfo });
        }
        else
        {
            assert(callOp->getNumResults() == 0);
        }
        return returnValueCopy;
    };
    {
        std::lock_guard lock{ _mutex };
        _definedFunctions[decl] = returnFn;
    }

    return returnFn;
}

bool MLIRContext::IsFunctionDefinedImpl(FunctionDeclaration decl) const
{
    if (std::lock_guard lock{ _mutex }; _definedFunctions.find(decl) != _definedFunctions.end())
    {
        return true;
    }

    return false;
}

Value MLIRContext::StoreConstantDataImpl(ConstantData data, MemoryLayout layout, const std::string& name)
{
    EmittableInfo& emittableInfo = std::visit(
        [this, &layout, name](auto&& data) -> EmittableInfo& {
            assert(!data.empty());

            using DataType = std::decay_t<decltype(data)>;
            using ElementType = typename DataType::value_type;

            auto& b = _impl->builder;
            auto loc = b.getUnknownLoc();

            ValueType valueElemTy = GetValueType<ElementType>();
            auto mlirElemTy = ValueTypeToMLIRType(b, valueElemTy);
            mlir::Value op;

            auto insertionBlock = b.getInsertionBlock();

            auto it = insertionBlock->begin();
            auto end = insertionBlock->end();
            while (it != end && llvm::isa<mlir::arith::ConstantOp>(it))
            {
                ++it;
            }

            mlir::OpBuilder::InsertionGuard guard(b);
            b.setInsertionPoint(insertionBlock, it);

            // if we have one value and we weren't explicitly asked for a higher-ranked layout
            if (data.size() == 1 && layout == ScalarLayout)
            {
                if constexpr (std::is_same_v<ElementType, index_t>)
                {
                    op = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(data[0]));
                }
                else if constexpr (std::is_same_v<ElementType, float16_t>)
                {
                    bool losesInfo = false;
                    auto f = llvm::APFloat(data[0].data);
                    f.convert(llvm::APFloat::IEEEhalf(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
                    op = b.create<mlir::arith::ConstantFloatOp>(loc, f, mlirElemTy.cast<mlir::Float16Type>());
                }
                else if constexpr (std::is_same_v<ElementType, bfloat16_t>)
                {
                    bool losesInfo = false;
                    auto f = llvm::APFloat(data[0].data);
                    f.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
                    op = b.create<mlir::arith::ConstantFloatOp>(loc, f, mlirElemTy.cast<mlir::BFloat16Type>());
                }
                else if constexpr (std::is_integral_v<ElementType> || std::is_same_v<ElementType, Boolean>)
                {
                    auto elem = static_cast<int64_t>(data[0]);
                    if (std::is_unsigned_v<ElementType>)
                    {
                        // ConstantIntOp only can only have signless integer type
                        mlirElemTy = accera::ir::util::ToSignlessMLIRType(b, mlirElemTy);
                    }
                    op = b.create<mlir::arith::ConstantIntOp>(loc, elem, mlirElemTy);
                }
                else if constexpr (std::is_floating_point_v<ElementType>)
                {
                    op = b.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(data[0]), mlirElemTy.cast<mlir::FloatType>());
                }
                else
                {
                    assert(false);
                }

                // TODO: do these need to be marked external as well?
            }
            else
            {
                auto memrefShapeTy = MemoryLayoutToMemRefType(b, layout, valueElemTy);

                auto mlirElemType = ValueTypeToMLIRType(b, valueElemTy);

                auto extents = layout.GetExtent().ToVector();

                // Our usage of the constant data behaves like raw pointer access rather than tensor index access from LLVM's point of view
                // So flatten this buffer shape to enable lowering without raising errors
                auto flattenedTensorShapeTy = mlir::RankedTensorType::get(extents, mlirElemType);

                mlir::DenseElementsAttr dataAttribute;
                if constexpr (std::is_same_v<ElementType, Boolean>)
                {
                    std::vector<int8_t> boolData(data.size());
                    std::transform(data.begin(), data.end(), boolData.begin(), [](Boolean b) { return b ? 1 : 0; });

                    dataAttribute = mlir::DenseElementsAttr::get(flattenedTensorShapeTy, llvm::makeArrayRef(boolData));
                }
                else if constexpr (std::is_same_v<ElementType, index_t>)
                {
                    std::vector<int8_t> indexData(data.size());
                    std::transform(data.begin(), data.end(), indexData.begin(), [](index_t value) { return static_cast<int64_t>(value); });

                    dataAttribute = mlir::DenseElementsAttr::get(flattenedTensorShapeTy, llvm::makeArrayRef(indexData));
                }
                else if constexpr (std::is_same_v<ElementType, float16_t>)
                {
                    using float16_underlying_type = typename float16_t::underlying_type;
                    std::vector<float16_underlying_type> fp16Data(data.size());
                    std::transform(data.begin(), data.end(), fp16Data.begin(), [](float16_t value) { return value.data; });

                    dataAttribute = mlir::DenseElementsAttr::get(flattenedTensorShapeTy, llvm::makeArrayRef(fp16Data));
                }
                else if constexpr (std::is_same_v<ElementType, bfloat16_t>)
                {
                    using bfloat16_underlying_type = typename bfloat16_t::underlying_type;
                    std::vector<bfloat16_underlying_type> bfp16Data(data.size());
                    std::transform(data.begin(), data.end(), bfp16Data.begin(), [](bfloat16_t value) { return value.data; });

                    dataAttribute = mlir::DenseElementsAttr::get(flattenedTensorShapeTy, llvm::makeArrayRef(bfp16Data));
                }
                else
                {
                    dataAttribute = mlir::DenseElementsAttr::get(flattenedTensorShapeTy, llvm::makeArrayRef(data));
                }

                std::string uniquedName = name + "_" + std::to_string(ir::util::GetUniqueId(b.getInsertionBlock()->getParentOp()));

                mlir::MemRefType globalMemrefTy = mlir::MemRefType::Builder{ memrefShapeTy }.setLayout({}); // remove affine maps
                [[maybe_unused]] auto globalOp = b.create<ir::value::GlobalOp>(loc, globalMemrefTy, /* isConstant= */ true, uniquedName, dataAttribute, /*addrSpace*/ 0, /*isExternal*/ true);
                op = b.create<ir::value::ReferenceGlobalOp>(loc, memrefShapeTy, uniquedName);
            }

            return StoreLocalEmittable({ op.getAsOpaquePointer(), { valueElemTy, 1 } });
        },
        data);

    Emittable emittable(&emittableInfo);

    return Value(emittable, layout);
}

bool MLIRContext::IsConstantDataImpl(Value v) const
{
    auto data = Unwrap(v);

    // TODO: Extend this check to handle constant data arrays. Right now, this only works for scalar values
    if (llvm::isa_and_nonnull<mlir::arith::ConstantIntOp, mlir::arith::ConstantIndexOp, mlir::arith::ConstantFloatOp>(data.getDefiningOp()))
    {
        return true;
    }

    return false;
}

Value MLIRContext::ResolveConstantDataReferenceImpl(Value constantDataSource)
{
    auto sourceRefGlobalOp = mlir::Value::getFromOpaquePointer(constantDataSource.Get<Emittable>().GetDataAs<EmittableInfo*>()->data).getDefiningOp();
    auto& builder = _impl->builder;

    auto valueModuleOp = _impl->module();
    auto searchSymName = mlir::dyn_cast<ir::value::ReferenceGlobalOp>(sourceRefGlobalOp).getGlobal().sym_name();

    // TODO: valueModuleOp.lookupSymbol() should be called here to look for an existing symbol, but so far,
    // it doesn't work as expected. So manually walk the top level ops inside the ValueModuleOp to look for the symbol.
    // Replace this workaround with a ValueModuleOp SymbolTable lookup once issues with comparing mlir::Identifiers is resolved.
    bool foundMatch = false;
    for (auto globalOp : valueModuleOp.getOps<ir::value::GlobalOp>())
    {
        if (globalOp.sym_name() == searchSymName)
        {
            foundMatch = true;
            break;
        }
    }

    if (!foundMatch)
    {
        // Clone the GlobalOp at the top of this module and mark it as external
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.restoreInsertionPoint(_impl->getGlobalInsertPt());

        auto sourceGlobalOp = mlir::dyn_cast<ir::value::ReferenceGlobalOp>(sourceRefGlobalOp).getGlobal();
        auto globalOp = mlir::dyn_cast<ir::value::GlobalOp>(builder.clone(*sourceGlobalOp));
        globalOp->setAttr("external", builder.getUnitAttr());
        globalOp->removeAttr("value"); // can't set null attribute (mlir::Attribute()) if existing
    }

    // Clone a ReferenceGlobalOp to refer to the GlobalOp. Since this is a clone, the reference *should* "carry over" without
    // explicitly wrapping globalOp from above
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto insertionBlock = builder.getInsertionBlock();
    auto vFuncOp = ir::util::CastOrGetParentOfType<ir::value::ValueFuncOp>(insertionBlock->getParentOp());
    if (vFuncOp)
    {
        // ensure that the ReferenceGlobalOp is within a function scope, if any
        builder.setInsertionPointToStart(&vFuncOp.body().front());
    }

    auto clonedRefGlobalOp = builder.clone(*sourceRefGlobalOp);
    auto refGlobalOp = mlir::dyn_cast<ir::value::ReferenceGlobalOp>(clonedRefGlobalOp);

    EmittableInfo& emittableInfo = StoreLocalEmittable({ const_cast<void*>(
                                                             refGlobalOp
                                                                 .getResult()
                                                                 .getAsOpaquePointer()),
                                                         { constantDataSource.GetBaseType(), 1 } });
    Emittable emittable{ &emittableInfo };

    return Value(emittable, constantDataSource.GetLayout());
}

void MLIRContext::ForImpl(MemoryLayout layout, std::function<void(std::vector<Scalar>)> fn, const std::string& name)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    auto dim = static_cast<unsigned>(layout.NumDimensions());
    std::vector<mlir::Value>
        LBs(dim, builder.create<mlir::arith::ConstantIndexOp>(loc, 0)),
        UBs;
    std::vector<int64_t> steps(dim, 1);
    for (unsigned i = 0; i < dim; ++i)
    {
        UBs.emplace_back(builder.create<mlir::arith::ConstantIndexOp>(loc, layout.GetActiveSize(i)));
    }

    auto loopSymName = std::string{ "value_loop" };
    if (!name.empty())
    {
        loopSymName += "_" + name;
    }

    mlir::buildAffineLoopNest(
        builder,
        loc,
        mlir::ValueRange{ LBs },
        mlir::ValueRange{ UBs },
        steps,
        [&](mlir::OpBuilder&, mlir::Location, mlir::ValueRange IVs) {
            std::vector<Scalar> logicalIndices(dim);
            for (unsigned i = 0; i < dim; ++i)
            {
                EmittableInfo& emittableInfo = StoreLocalEmittable({ IVs[i].getAsOpaquePointer(), { ValueType::Index, 1 } });
                Emittable emittable{ &emittableInfo };
                logicalIndices[i] = Scalar(Value(emittable, ScalarLayout));
            }
            fn(logicalIndices);
        });
}

void MLIRContext::ForImpl(Scalar start, Scalar stop, Scalar step, std::function<void(Scalar)> fn, const std::string& name)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    auto mlirStart = ToMLIRIndex(builder, start);
    auto mlirStop = ToMLIRIndex(builder, stop);
    auto mlirStep = ToMLIRIndex(builder, step);

    auto loopSymName = std::string{ "value_loop" };
    if (!name.empty())
    {
        loopSymName += "_" + name;
    }
    mlir::scf::buildLoopNest(builder, loc, mlirStart, mlirStop, mlirStep, [&](mlir::OpBuilder&, mlir::Location, mlir::ValueRange IVs) {
        auto iv = IVs[0];
        SetOpNameAttr(iv.getParentRegion()->getParentOp(), loopSymName);
        EmittableInfo& emittableInfo = StoreLocalEmittable({ iv.getAsOpaquePointer(), { ValueType::Index, 1 } });
        Emittable emittable{ &emittableInfo };
        Scalar index(Value(emittable, ScalarLayout));
        fn(index);
    });
}

ViewAdapter MLIRContext::ReduceN(Scalar start, Scalar stop, Scalar step, std::vector<ViewAdapter> initArgs, std::function<ViewAdapter(Scalar, std::vector<ViewAdapter>)> loopFn)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    auto mlirStart = ToMLIRIndex(builder, start);
    auto mlirStop = ToMLIRIndex(builder, stop);
    auto mlirStep = ToMLIRIndex(builder, step);

    auto mlirInitArgs = llvm::to_vector<2>(
        llvm::map_range(
            initArgs,
            [&builder](ViewAdapter view) { return ResolveMLIRScalar(builder, ToMLIRValue(builder, view)); }));
    auto forOp = builder.create<mlir::scf::ForOp>(
        loc,
        mlirStart,
        mlirStop,
        mlirStep,
        mlirInitArgs,
        [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterValues) {
            loc = builder.getFusedLoc({ loc, ir::util::GetLocation(builder, __FILE__, __LINE__) });
            auto wrappedIV = Wrap(iv);

            std::vector<ViewAdapter> iterWrappedValues = Wrap(std::vector<mlir::Value>(iterValues.begin(), iterValues.end()));

            auto result = loopFn(wrappedIV, iterWrappedValues);
            builder.create<mlir::scf::YieldOp>(loc, ToMLIRValue(builder, result));
        });
    assert(forOp.getNumResults() == 1 && "Can't handle multiple results yet");

    return Wrap(forOp.getResult(0));
}

ViewAdapter MLIRContext::Reduce(Array a, std::vector<ViewAdapter> initArgs, std::function<ViewAdapter(ViewAdapter, std::vector<ViewAdapter>)> reduceFn)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    assert(initArgs.size() == 1 && "Can't handle multiple results yet");
    auto mlirInitArgs = llvm::to_vector<2>(llvm::map_range(initArgs, [&builder](ViewAdapter view) { return ResolveMLIRScalar(builder, ToMLIRValue(builder, view)); }));
    auto reduceOp = builder.create<ir::value::ReduceOp>(
        loc,
        ToMLIRValue(builder, a.GetValue()),
        mlirInitArgs[0],
        [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val, mlir::ValueRange iterValues) {
            loc = builder.getFusedLoc({ loc, ir::util::GetLocation(builder, __FILE__, __LINE__) });
            auto wrappedVal = Wrap(val);
            std::vector<ViewAdapter> iterWrappedValues = Wrap(std::vector<mlir::Value>(iterValues.begin(), iterValues.end()));
            auto result = reduceFn(wrappedVal, iterWrappedValues);
            builder.create<ir::value::YieldOp>(loc, ToMLIRValue(builder, result));
        });

    ir::executionPlan::VectorizationInfo vecInfo{ 8, 16 };
    auto vectorizationInfoIdentifier = builder.getStringAttr(ir::executionPlan::VectorizationInfoAttr::getKeyName());
    reduceOp->setAttr(vectorizationInfoIdentifier, ir::executionPlan::VectorizationInfoAttr::get(vecInfo, builder.getContext()));
    return Wrap(reduceOp.getResult());
}

ViewAdapter MLIRContext::MapReduce(Array a, std::vector<ViewAdapter> initArgs, std::function<ViewAdapter(ViewAdapter)> mapFn, std::function<ViewAdapter(ViewAdapter, std::vector<ViewAdapter>)> reduceFn)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    assert(initArgs.size() == 1 && "Can't handle multiple results yet");
    auto mlirInitArgs = llvm::to_vector<2>(llvm::map_range(initArgs, [&builder](ViewAdapter view) { return ResolveMLIRScalar(builder, ToMLIRValue(builder, view)); }));
    auto mapReduceOp = builder.create<ir::value::MapReduceOp>(
        loc,
        ToMLIRValue(builder, a.GetValue()),
        mlirInitArgs[0],
        [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val) {
            loc = builder.getFusedLoc({ loc, ir::util::GetLocation(builder, __FILE__, __LINE__) });
            auto wrappedVal = Wrap(val);
            auto result = mapFn(wrappedVal);
            builder.create<ir::value::YieldOp>(loc, ToMLIRValue(builder, result));
        },
        [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val, mlir::ValueRange iterValues) {
            loc = builder.getFusedLoc({ loc, ir::util::GetLocation(builder, __FILE__, __LINE__) });
            auto wrappedVal = Wrap(val);
            std::vector<ViewAdapter> iterWrappedValues = Wrap(std::vector<mlir::Value>(iterValues.begin(), iterValues.end()));
            auto result = reduceFn(wrappedVal, iterWrappedValues);
            builder.create<ir::value::YieldOp>(loc, ToMLIRValue(builder, result));
        });

    ir::executionPlan::VectorizationInfo vecInfo{ 8, 16 };
    auto vectorizationInfoIdentifier = builder.getStringAttr(ir::executionPlan::VectorizationInfoAttr::getKeyName());
    mapReduceOp->setAttr(vectorizationInfoIdentifier, ir::executionPlan::VectorizationInfoAttr::get(vecInfo, builder.getContext()));
    return Wrap(mapReduceOp.getResult());
}

void MLIRContext::MoveDataImpl(Value& source, Value& destination)
{
    // we treat a move the same as a copy, except we clear out the source
    CopyDataImpl(source, destination);

    // data has been "moved", so clear the source
    source.Reset();
}

void MLIRContext::CopyDataImpl(const Value& source, Value& destination)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    auto src = ToMLIRValue(builder, source);
    auto dst = ToMLIRValue(builder, destination);

    if (!dst)
    {
        // the destination was empty
        assert(source.GetLayout() == ScalarLayout && "Unexpected empty destination for non-Scalar value");

        auto result = ResolveMLIRScalar(builder, src);
        EmittableInfo& emittableInfo = StoreLocalEmittable({ result.getAsOpaquePointer(), { source.GetBaseType(), 1 } });
        Emittable emittable{ &emittableInfo };

        destination.SetData({ emittable, ScalarLayout });
    }
    else if (dst.getType().isIntOrIndexOrFloat() && dst.getType() == src.getType())
    {
        assert(source.GetLayout() == ScalarLayout && destination.GetLayout() == ScalarLayout);
        destination.SetData(source);
    }
    else if (src.getType().isa<mlir::ShapedType>() && source.GetLayout() == ScalarLayout)
    {
        (void)builder.create<accera::ir::value::CopyOp>(loc, ResolveMLIRScalar(builder, src), dst);
    }
    else
    {
        (void)builder.create<accera::ir::value::CopyOp>(loc, src, dst);
    }
}

void MLIRContext::StoreImpl(const Value& sourceValue, Value& destValue, const std::vector<int64_t>& indices)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    // TODO: validation
    auto source = ToMLIRValue(builder, sourceValue);
    auto dest = ToMLIRValue(builder, destValue);

    llvm::SmallVector<mlir::Value, 4> indexValues;
    std::transform(begin(indices), end(indices), std::back_inserter(indexValues), [&](int64_t i) -> mlir::Value {
        return builder.create<mlir::arith::ConstantIndexOp>(loc, i);
    });

    (void)builder.create<ir::value::StoreOp>(loc, source, dest, indexValues);
}

Value MLIRContext::ViewImpl(Value sourceValue, const std::vector<Scalar>& offsetsValue, const MemoryShape& shape, const std::vector<int64_t>& stridesValue)
{
    const MemoryLayout& currentLayout = sourceValue.GetLayout();
    auto destLayout = GetSubArrayLayout(currentLayout, shape, stridesValue);

    llvm::SmallVector<mlir::Value, 4> linalgRanges;
    auto ranges = utilities::MakeZipRange(offsetsValue, shape, stridesValue);
    std::transform(
        begin(ranges),
        end(ranges),
        std::back_inserter(linalgRanges),
        [this](std::tuple<Scalar, Scalar, Scalar> s) -> mlir::Value {
            auto& builder = _impl->builder;
            auto loc = builder.getUnknownLoc();
            auto offset = Cast(std::get<0>(s), ValueType::Index);
            auto size = Cast(std::get<1>(s), ValueType::Index);
            auto stride = Cast(std::get<2>(s), ValueType::Index);
            auto lowerBound = ResolveMLIRIndex(builder, offset);
            auto upperBound = ResolveMLIRIndex(builder, size);
            auto step = ResolveMLIRIndex(builder, stride);
            return builder.create<ir::value::RangeOp>(loc, lowerBound, upperBound, step);
        });

    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    auto resultMemRefType = MemoryLayoutToMemRefType(builder, destLayout, sourceValue.GetBaseType(), /*useDynamicOffset*/ true, /*pointerLevel=*/0);

    auto source = ToMLIRValue(builder, sourceValue);
    mlir::Value result = builder.create<ir::value::ViewOp>(loc, source, linalgRanges, resultMemRefType);

    EmittableInfo& emittableInfo = StoreLocalEmittable({ result.getAsOpaquePointer(), { sourceValue.GetBaseType(), 1 } });
    Emittable emittable{ &emittableInfo };

    return { emittable, destLayout };
}

Value MLIRContext::SliceImpl(Value sourceValue, std::vector<int64_t> slicedDimensions, std::vector<Scalar> sliceOffsets)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    const MemoryLayout& currentLayout = sourceValue.GetLayout();
    auto destLayout = GetSliceLayout(currentLayout, slicedDimensions);

    llvm::SmallVector<mlir::Value, 4> offsets;
    std::transform(
        sliceOffsets.begin(),
        sliceOffsets.end(),
        std::back_inserter(offsets),
        [&builder](Scalar s) { return ResolveMLIRIndex(builder, ResolveMLIRScalar(builder, ToMLIRValue(builder, s))); });

    auto resultMemRefType = MemoryLayoutToMemRefType(builder, destLayout, sourceValue.GetBaseType(), /*useDynamicOffset=*/true, /*pointerLevel=*/0);
    auto source = ToMLIRValue(builder, sourceValue);
    mlir::Value result = builder.create<ir::value::SliceOp>(loc, source, slicedDimensions, offsets, resultMemRefType);

    EmittableInfo& emittableInfo = StoreLocalEmittable({ result.getAsOpaquePointer(), { sourceValue.GetBaseType(), 1 } });
    Emittable emittable{ &emittableInfo };

    return { emittable, destLayout };
}

Value MLIRContext::MergeDimensionsImpl(Value sourceValue, int64_t dim1, int64_t dim2)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    const MemoryLayout& currentLayout = sourceValue.GetLayout();
    ThrowIfNot(currentLayout.IsCanonicalOrder(), InputExceptionErrors::invalidArgument, "MergeDimension requires a canonically-ordered operand");
    auto destLayout = GetMergeDimLayout(currentLayout, dim1, dim2);
    auto resultMemRefType = MemoryLayoutToMemRefType(builder, destLayout, sourceValue.GetBaseType());
    auto source = ToMLIRValue(builder, sourceValue);
    return Wrap(builder.create<ir::value::MergeDimOp>(loc, resultMemRefType, source, dim1, dim2));
}

Value MLIRContext::SplitDimensionImpl(Value sourceValue, int64_t dim, int64_t size)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    // TODO: assert there's no offset
    const MemoryLayout& currentLayout = sourceValue.GetLayout();
    auto destLayout = GetSplitDimLayout(currentLayout, dim, size);
    auto resultMemRefType = MemoryLayoutToMemRefType(builder, destLayout, sourceValue.GetBaseType());
    auto source = ToMLIRValue(builder, sourceValue);
    return Wrap(builder.create<ir::value::SplitDimOp>(loc, resultMemRefType, source, dim, size));
}

Value MLIRContext::ReshapeImpl(Value sourceValue, const MemoryLayout& destLayout)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    // TODO: assert there's no offset
    auto resultMemRefType = MemoryLayoutToMemRefType(builder, destLayout, sourceValue.GetBaseType());
    auto source = ToMLIRValue(builder, sourceValue);
    return Wrap(builder.create<ir::value::ReshapeOp>(loc, resultMemRefType, source));
}

Value MLIRContext::ReorderImpl(Value sourceValue, const DimensionOrder& order)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    // TODO: assert there's no offset
    auto source = ToMLIRValue(builder, sourceValue);

    return Wrap(builder.create<ir::value::ReorderOp>(loc, source, order.ToVector()));
}

namespace
{
    auto Convert(ValueUnaryOperation op)
    {
        using namespace accera::ir::value;

        switch (op)
        {
        case ValueUnaryOperation::LogicalNot:
            return UnaryOpPredicate::NOT;
        }
        llvm_unreachable("Unknown unary operation");
    }
} // namespace

Value MLIRContext::UnaryOperationImpl(ValueUnaryOperation op, Value source)
{
    using namespace accera::ir::value;

    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    auto srcValue = ToMLIRValue(builder, source);
    mlir::Value loadedSrcValue = ResolveMLIRScalar(builder, srcValue);
    mlir::Value result = builder.create<ir::value::UnaryOp>(loc, Convert(op), loadedSrcValue);

    EmittableInfo& emittableInfo = StoreLocalEmittable({ result.getAsOpaquePointer(), { source.GetBaseType(), 1 } });
    Emittable emittable{ &emittableInfo };

    return { emittable, ScalarLayout };
}

namespace
{
    auto Convert(ValueBinaryOperation op)
    {
        using namespace accera::ir::value;
        switch (op)
        {
        case ValueBinaryOperation::add:
            return BinaryOpPredicate::ADD;
        case ValueBinaryOperation::divide:
            return BinaryOpPredicate::DIV;
        case ValueBinaryOperation::logicalAnd:
            return BinaryOpPredicate::LOGICAL_AND;
        case ValueBinaryOperation::logicalOr:
            return BinaryOpPredicate::LOGICAL_OR;
        case ValueBinaryOperation::modulus:
            return BinaryOpPredicate::MOD;
        case ValueBinaryOperation::multiply:
            return BinaryOpPredicate::MUL;
        case ValueBinaryOperation::subtract:
            return BinaryOpPredicate::SUB;
        }
        llvm_unreachable("Unknown binary operation");
    }
} // namespace

Value MLIRContext::BinaryOperationImpl(ValueBinaryOperation op, Value source1, Value source2)
{
    using namespace accera::ir::value;

    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    auto srcValue1 = ToMLIRValue(builder, source1);
    auto srcValue2 = ToMLIRValue(builder, source2);
    auto pred = Convert(op);

    mlir::Value loadedSrcValue1 = ResolveMLIRScalar(builder, srcValue1);
    mlir::Value loadedSrcValue2 = ResolveMLIRScalar(builder, srcValue2);
    mlir::Value result = builder.create<ir::value::BinOp>(loc, pred, loadedSrcValue1, loadedSrcValue2);

    EmittableInfo& emittableInfo = StoreLocalEmittable({ result.getAsOpaquePointer(), { source1.GetBaseType(), 1 } });
    Emittable emittable{ &emittableInfo };

    return { emittable, ScalarLayout };
}

namespace
{
    auto Convert(ValueLogicalOperation op)
    {
        using namespace accera::ir::value;
        switch (op)
        {
        case ValueLogicalOperation::equality:
            return CmpOpPredicate::EQ;
        case ValueLogicalOperation::greaterthan:
            return CmpOpPredicate::GT;
        case ValueLogicalOperation::greaterthanorequal:
            return CmpOpPredicate::GE;
        case ValueLogicalOperation::inequality:
            return CmpOpPredicate::NE;
        case ValueLogicalOperation::lessthan:
            return CmpOpPredicate::LT;
        case ValueLogicalOperation::lessthanorequal:
            return CmpOpPredicate::LE;
        }
        llvm_unreachable("Unknown logical operation");
    }

    std::pair<mlir::Value, mlir::Value> GetCompatibleValueHandles(mlir::OpBuilder& builder, Value val1, Value val2)
    {
        auto val1Handle = ToMLIRValue(builder, val1);
        auto val2Handle = ToMLIRValue(builder, val2);

        auto type1 = val1Handle.getType();
        auto type2 = val2Handle.getType();

        if (type1.isa<mlir::MemRefType>() && type2.isa<mlir::TensorType>())
        {
            Value val2New = Allocate(val2.GetBaseType(), val2.GetLayout());
            val2New = val2;
            auto val2HandleNew = ToMLIRValue(builder, val2New);
            std::swap(val2Handle, val2HandleNew);
        }
        else if (type1.isa<mlir::MemRefType>() && type2.isa<mlir::TensorType>())
        {
            Value val1New = Allocate(val1.GetBaseType(), val1.GetLayout());
            val1New = val1;
            auto val1HandleNew = ToMLIRValue(builder, val1New);
            std::swap(val1Handle, val1HandleNew);
        }

        return { val1Handle, val2Handle };
    }
} // namespace

Value MLIRContext::LogicalOperationImpl(ValueLogicalOperation op, Value source1, Value source2)
{
    using namespace accera::ir::value;
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    auto [src1Handle, src2Handle] = GetCompatibleValueHandles(builder, source1, source2);

    mlir::Value loadedSrc1Handle = ResolveMLIRScalar(builder, src1Handle);
    mlir::Value loadedSrc2Handle = ResolveMLIRScalar(builder, src2Handle);
    mlir::Value result = builder.create<ir::value::CmpOp>(loc, Convert(op), loadedSrc1Handle, loadedSrc2Handle);

    EmittableInfo& emittableInfo = StoreLocalEmittable({ result.getAsOpaquePointer(), { ValueType::Boolean, 1 } });
    Emittable emittable{ &emittableInfo };

    return { emittable, ScalarLayout };
}

Value MLIRContext::MMALoadSyncImpl(const Matrix& source, const int64_t rowOffset, const int64_t colOffset, const MatrixFragment& target)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    auto matValue = ToMLIRValue(builder, source);
    const auto mmaShape = static_cast<ir::value::MMAShape>(target.GetFragmentShape());
    const ir::value::MMAOp mmaType(mmaShape);

    auto rowOff = builder.create<mlir::arith::ConstantIndexOp>(loc, rowOffset);
    auto colOff = builder.create<mlir::arith::ConstantIndexOp>(loc, colOffset);
    const ir::value::MMAOperandType operandType{ static_cast<ir::value::MMAOperandType>(target.GetFragmentType()) };
    const auto isAcc = operandType == ir::value::MMAOperandType::Acc;
    auto elementType = (source.GetValue().IsFloat32() || isAcc) ? builder.getF32Type() : builder.getF16Type();
    auto mmaTileShape = mmaType.getOperandShape(operandType);
    auto vecTy = mlir::MemRefType::get(mmaTileShape, elementType);

    mlir::Value result = builder.create<ir::value::MMAAllocSyncOp>(loc, vecTy, static_cast<uint32_t>(mmaType.getShapeType()), static_cast<uint8_t>(operandType), builder.getBoolAttr(/*TODO*/true));
    builder.create<ir::value::MMALoadSyncOp>(loc, matValue, result, mmaShape, operandType, /*TODO*/true, mlir::ValueRange{ rowOff, colOff });
    EmittableInfo& emittableInfo = StoreLocalEmittable({ result.getAsOpaquePointer(), { source.GetValue().GetBaseType(), 1 } });
    Emittable emittable{ &emittableInfo };

    auto mmaMatLayout = MemoryLayout(mmaTileShape[0], mmaTileShape[1]);
    return Value(emittable, mmaMatLayout);
}

void MLIRContext::MMAStoreSyncImpl(const MatrixFragment& source, Matrix& target, const int64_t rowOffset, const int64_t colOffset)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    auto sourceValue = ToMLIRValue(builder, source);
    auto targetValue = ToMLIRValue(builder, target);
    auto rowOff = builder.create<mlir::arith::ConstantIndexOp>(loc, rowOffset);
    auto colOff = builder.create<mlir::arith::ConstantIndexOp>(loc, colOffset);

    const auto mmaShape = static_cast<ir::value::MMAShape>(source.GetFragmentShape());
    builder.create<ir::value::MMAStoreSyncOp>(loc, sourceValue, targetValue, mmaShape, mlir::ValueRange{ rowOff, colOff });
}

Value MLIRContext::MMAComputeSyncImpl(const MatrixFragment& A, const MatrixFragment& B, const MatrixFragment& C, const uint32_t cbsz, const uint32_t abid, const uint32_t blgp)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    auto aValue = ToMLIRValue(builder, A);
    auto bValue = ToMLIRValue(builder, B);
    auto cValue = ToMLIRValue(builder, C);

    mlir::Value result = builder.create<ir::value::MMAComputeSyncOp>(loc, aValue, bValue, cValue, uint32_t(A.GetFragmentShape()), cbsz, abid, blgp).opC();

    EmittableInfo& emittableInfo = StoreLocalEmittable({ result.getAsOpaquePointer(), { C.GetType(), 1 } });
    Emittable emittable{ &emittableInfo };

    return Value(emittable, C.GetValue().GetLayout());
}

Scalar MLIRContext::CastImpl(Scalar value, ValueType type)
{
    auto& builder = _impl->builder;
    mlir::Value mlirValue = ResolveMLIRScalar(builder, ToMLIRValue(builder, value));
    auto loc = mlirValue.getLoc();
    auto toType = ValueTypeToMLIRType(builder, type);

    mlir::Value castVal = builder.create<ir::value::CastOp>(loc, mlirValue, toType);
    return Scalar(Wrap(castVal));
}

bool MLIRContext::IsImplicitlyCastableImpl(ValueType source, ValueType target) const
{
    auto& builder = _impl->builder;
    if (!HasMLIRTypeConversion(source) || !HasMLIRTypeConversion(target))
    {
        return false;
    }
    auto sourceMlirType = ValueTypeToMLIRType(builder, source);
    auto targetMlirType = ValueTypeToMLIRType(builder, target);
    return accera::ir::util::IsImplicitlyCastable(sourceMlirType, targetMlirType);
}

Scalar MLIRContext::BitcastImpl(Scalar value, ValueType type)
{
    auto& builder = _impl->builder;
    mlir::Value mlirValue = ResolveMLIRScalar(builder, ToMLIRValue(builder, value));

    auto loc = mlirValue.getLoc();
    auto fromType = mlirValue.getType();
    auto toType = ValueTypeToMLIRType(builder, type);
    if (fromType == toType)
    {
        return Wrap(mlirValue);
    }

    if (fromType.isIntOrIndexOrFloat() && toType.isIntOrIndexOrFloat() && fromType.getIntOrFloatBitWidth() == toType.getIntOrFloatBitWidth())
    {
        using namespace accera::ir::value;

        return Wrap(builder.create<BitcastOp>(loc, toType, mlirValue));
    }

    throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Can only bitcast between types of the same size");
}

namespace
{
    mlir::ValueRange CascadingConditionBuilder(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        std::pair<mlir::Value, std::function<void(mlir::OpBuilder&, mlir::Location)>> testCase,
        std::function<void(mlir::OpBuilder&, mlir::Location)> elseCase = nullptr,
        llvm::ArrayRef<std::pair<mlir::Value, std::function<void(mlir::OpBuilder&, mlir::Location)>>> alternates = llvm::None)
    {
        // base case (forward to conditionBuilder)
        if (alternates.empty())
        {
            auto ifOp = builder.create<mlir::scf::IfOp>(
                loc,
                testCase.first,
                testCase.second,
                elseCase
                    ? mlir::function_ref<void(mlir::OpBuilder&, mlir::Location)>{ elseCase }
                    : mlir::function_ref<void(mlir::OpBuilder&, mlir::Location)>{});

            mlir::scf::IfOp::ensureTerminator(ifOp.getThenRegion(), builder, loc);

            if (auto& elseRegion = ifOp.getElseRegion(); !elseRegion.empty())
            {
                mlir::scf::IfOp::ensureTerminator(elseRegion, builder, loc);
            }

            return ifOp.getResults();
        }
        else
        {
            auto ifOp = builder.create<mlir::scf::IfOp>(
                loc,
                testCase.first,
                testCase.second,
                [else_ = std::move(elseCase), alts = std::move(alternates)](mlir::OpBuilder& builder, mlir::Location loc) {
                    auto firstAlt = alts[0];

                    (void)CascadingConditionBuilder(builder, loc, firstAlt, else_, alts.drop_front());
                });

            mlir::scf::IfOp::ensureTerminator(ifOp.getThenRegion(), builder, loc);

            if (auto& elseRegion = ifOp.getElseRegion(); !elseRegion.empty())
            {
                mlir::scf::IfOp::ensureTerminator(elseRegion, builder, loc);
            }
            return ifOp.getResults();
        }
    }
} // namespace

class MLIRContext::IfContextImpl : public EmitterContext::IfContextImpl
{
public:
    IfContextImpl(MLIRContext::Impl& impl, Scalar test, std::function<void()> fn) :
        builder(impl.builder),
        thenPair(ResolveMLIRScalar(builder, ToMLIRValue(builder, test)), [fn = std::move(fn)](mlir::OpBuilder&, mlir::Location) { fn(); })
    {
    }

    ~IfContextImpl() override
    {
        (void)CascadingConditionBuilder(builder, builder.getUnknownLoc(), thenPair, elseFn, elseIfs);
    }

    void ElseIf(Scalar test, std::function<void()> fn) override
    {
        auto testValue = ResolveMLIRScalar(builder, ToMLIRValue(builder, test));
        elseIfs.emplace_back(testValue, [fn = std::move(fn)](mlir::OpBuilder&, mlir::Location) { fn(); });
    }

    void Else(std::function<void()> fn) override
    {
        if (elseFn)
        {
            throw utilities::LogicException(
                utilities::LogicExceptionErrors::illegalState, "There can only be one else clause");
        }
        elseFn = [fn = std::move(fn)](mlir::OpBuilder&, mlir::Location) { fn(); };
    }

private:
    using Fn = std::function<void(mlir::OpBuilder&, mlir::Location)>;
    using CondFnPair = std::pair<mlir::Value, Fn>;
    using CondFnVector = mlir::SmallVector<CondFnPair, 3>;

    mlir::OpBuilder& builder;
    CondFnPair thenPair;
    CondFnVector elseIfs;
    Fn elseFn;
};

EmitterContext::IfContext MLIRContext::IfImpl(Scalar test, std::function<void()> fn)
{
    return { std::make_unique<MLIRContext::IfContextImpl>(*_impl, test, fn) };
}

void MLIRContext::WhileImpl(Scalar test, std::function<void()> fn)
{
    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
}

std::optional<Value> MLIRContext::CallImpl(FunctionDeclaration func, std::vector<Value> args)
{
    if (std::any_of(args.begin(), args.end(), [](const auto& value) { return value.IsEmpty(); }))
    {
        throw InputException(InputExceptionErrors::invalidArgument, __FILE__ " : " + std::to_string(__LINE__));
    }

    {
        std::lock_guard lock{ _mutex };
        if (auto it = _definedFunctions.find(func); it != _definedFunctions.end())
        {
            return it->second(args);
        }
    }

    return EmitExternalCall(func, args);
}

std::optional<Value> MLIRContext::EmitExternalCall(FunctionDeclaration func, std::vector<Value> args)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    std::vector<mlir::Value> mlirValueCallArgs = ToMLIRValue(builder, args);

    auto DeclareFn = [&](const std::string& name, mlir::FunctionType fnTy) -> ir::value::ValueFuncOp {
        auto insertionBlock = builder.getBlock();
        auto parentOp = insertionBlock->getParentOp();
        auto mod = accera::ir::util::CastOrGetParentOfType<mlir::ModuleOp>(parentOp);
        assert(mod);

        if (auto fnOp = mod.lookupSymbol<ir::value::ValueFuncOp>(name); fnOp) return fnOp;

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.restoreInsertionPoint(_impl->getGlobalInsertPt());

        return builder.create<ir::value::ValueFuncOp>(
            loc,
            name,
            fnTy,
            ir::value::ExecutionTarget::CPU);
    };

    accera::ir::value::CallOp callResult = builder.create<ir::value::CallOp>(
        loc,
        DeclareFn(func.GetFunctionName(), ::ToMLIRType(builder, func)),
        mlir::ValueRange{ mlirValueCallArgs });

    if (callResult.getNumResults() > 0)
    {
        // TODO : support multiple returns from an external function
        return Wrap(callResult.getResult(0));
    }
    else
    {
        return std::nullopt;
    }
}

void MLIRContext::ReturnValue(ViewAdapter view)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    auto mlirValue = ToMLIRValue(builder, view);
    (void)builder.create<ir::value::EarlyReturnOp>(loc, mlirValue ? mlir::ValueRange{ mlirValue } : mlir::ValueRange{});
}

Scalar MLIRContext::GetTime()
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    mlir::Value time = builder.create<ir::value::GetTimeOp>(loc);
    return Wrap(time);
}

void MLIRContext::EnterProfileRegion(const std::string& regionName)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    (void)builder.create<accera::ir::value::EnterProfileRegionOp>(loc, regionName);
}

void MLIRContext::ExitProfileRegion(const std::string& regionName)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    (void)builder.create<accera::ir::value::ExitProfileRegionOp>(loc, regionName);
}

void MLIRContext::PrintProfileResults()
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    (void)builder.create<accera::ir::value::PrintProfileResultsOp>(loc);
}

void MLIRContext::PrefetchImpl(Value data, PrefetchType type, PrefetchLocality locality)
{
    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
}

void MLIRContext::PrintImpl(ViewAdapter value, bool toStderr)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    auto mlirValue = ToMLIRValue(builder, value);
    assert(mlirValue);
    [[maybe_unused]] auto op = builder.create<accera::ir::value::PrintOp>(loc, mlirValue, toStderr);
}

void MLIRContext::PrintRawMemoryImpl(ViewAdapter value)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    auto mem = ToMLIRValue(builder, value);
    assert(mem);
    auto type = mem.getType();
    auto memType = type.dyn_cast<mlir::MemRefType>();
    if (!memType)
    {
        throw std::runtime_error{ "Value must have a memref type" };
    }

    // compute the total size
    auto shape = memType.getShape();
    llvm::SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (failed(getStridesAndOffset(memType, strides, offset)))
    {
        throw std::logic_error{ "Resource to be filled in must be valid memory" };
    }

    auto maxStride = std::max_element(strides.begin(), strides.end());
    auto size = (*maxStride) * (shape[maxStride - strides.begin()]);

    auto elemTy = memType.getElementType();
    auto identityLayout = mlir::MemRefLayoutAttrInterface{};

    // cast to a value with type `memref<total_size x elem_type>` (via `memref<* x elem_type>`)
    mlir::Value ptr = builder.create<mlir::memref::CastOp>(loc, mem, mlir::UnrankedMemRefType::get(elemTy, memType.getMemorySpace()));
    mlir::Value mlirValue = builder.create<mlir::memref::CastOp>(loc, ptr, mlir::MemRefType::get({ size }, elemTy, identityLayout, memType.getMemorySpace()));

    [[maybe_unused]] auto op = builder.create<ir::value::PrintOp>(loc, mlirValue, /*toStderr=*/false);
}

void MLIRContext::PrintImpl(const std::string& message, bool toStderr)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    [[maybe_unused]] auto op = builder.create<accera::ir::value::PrintFOp>(loc, message, toStderr);
}

void MLIRContext::DebugBreakImpl()
{
    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
}

void MLIRContext::DebugDumpImpl(Value value, std::string tag, std::ostream& stream) const
{
    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
}

void MLIRContext::DebugDumpImpl(FunctionDeclaration fn, std::string tag, std::ostream& stream) const
{
    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
}

void MLIRContext::DebugPrintImpl(std::string message)
{
    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
}

void MLIRContext::SetNameImpl(const Value& value, const std::string& name)
{
    auto& builder = _impl->builder;
    auto mlirValue = ToMLIRValue(builder, value);
    assert(mlirValue);

    if (auto op = mlirValue.getDefiningOp())
        SetOpNameAttr(op, name);
}

std::string MLIRContext::GetNameImpl(const Value& value) const
{
    auto& builder = _impl->builder;
    auto mlirValue = ToMLIRValue(builder, value);

    if (auto nameAttr = mlirValue.getDefiningOp()->getAttr(mlir::SymbolTable::getSymbolAttrName()))
    {
        if (auto stringAttr = nameAttr.dyn_cast_or_null<mlir::StringAttr>())
        {
            return stringAttr.getValue().str();
        }
    }

    return "";
}

void MLIRContext::ImportCodeFileImpl(std::string)
{
    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
}

Scalar MLIRContext::MaxImpl(Vector input)
{
    auto& builder = _impl->builder;
    auto mlirValue = ToMLIRValue(builder, input);
    assert(mlirValue);
    auto inputType = mlirValue.getType();
    assert(inputType.isa<mlir::MemRefType>() && "Vector input must be a memref");
    auto memRefType = inputType.cast<mlir::MemRefType>();
    auto resultType = memRefType.getElementType();
    auto loc = mlirValue.getLoc();
    auto max = builder.create<ir::value::ReduceMaxOp>(loc, resultType, mlirValue);
    return Wrap(max, ScalarLayout);
}

Scalar MLIRContext::SumImpl(Vector input)
{
    auto& builder = _impl->builder;
    auto mlirValue = ToMLIRValue(builder, input);
    assert(mlirValue);
    auto inputType = mlirValue.getType();
    assert(inputType.isa<mlir::MemRefType>() && "Vector input must be a memref");
    auto memRefType = inputType.cast<mlir::MemRefType>();
    auto resultType = memRefType.getElementType();
    auto loc = mlirValue.getLoc();
    auto sum = builder.create<ir::value::ReduceSumOp>(loc, resultType, mlirValue);
    return Wrap(sum, ScalarLayout);
}

std::string MLIRContext::GetScopeAdjustedName(GlobalAllocationScope scope, std::string name) const
{
    switch (scope)
    {
    case GlobalAllocationScope::Global:
        return GetGlobalScopedName(name);
    case GlobalAllocationScope::Function:
        return GetCurrentFunctionScopedName(name);
    }

    throw LogicException(LogicExceptionErrors::illegalState, __FILE__ " : " + std::to_string(__LINE__));
}

std::string MLIRContext::GetGlobalScopedName(std::string name) const
{
    return _impl->module().getName().str() + "_" + name;
}

std::string MLIRContext::GetCurrentFunctionScopedName(std::string name) const
{
    auto& b = GetMLIRContext().GetOpBuilder();
    auto fnOp = b.getBlock()->getParent()->getParentOfType<ir::value::ValueFuncOp>();
    assert(fnOp);

    return GetGlobalScopedName(fnOp.getName().str() + "_" + name);
}

void MLIRContext::SetLayoutImpl(Value& value, const MemoryLayout& layout)
{}

mlir::OpBuilder& MLIRContext::GetOpBuilder()
{
    return _impl->builder;
}

// Finds a ValueFuncOp matching the given name
ir::value::ValueFuncOp FindValueFuncOp(mlir::ModuleOp mod, const std::string& name)
{
    ir::value::ValueFuncOp fnOp;
    mod->walk([&fnOp, name](ir::value::ValueFuncOp op) {
        if (name == op.sym_name())
        {
            fnOp = op;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return fnOp;
}

mlir::Value Unwrap(ViewAdapter view)
{
    Value v = view;
    auto emittable = v.Get<Emittable>().GetDataAs<MLIRContext::EmittableInfo*>();
    return mlir::Value::getFromOpaquePointer(emittable->data);
}

mlir::Value UnwrapScalar(Scalar s)
{
    auto mlirVal = Unwrap(s);
    return ResolveMLIRScalar(GetMLIRContext().GetOpBuilder(), mlirVal);
}

ValueType MLIRTypeToValueType(mlir::Type ty)
{
    assert(ty.isIntOrIndexOrFloat());
    return llvm::TypeSwitch<mlir::Type, ValueType>(ty)
        .Case<mlir::IndexType>([](mlir::IndexType) {
            return ValueType::Index;
        })
        .Case<mlir::IntegerType>([](mlir::IntegerType iTy) {
            if (iTy.isSigned() || iTy.isSignless())
            {
                switch (iTy.getWidth())
                {
                case 1:
                    return ValueType::Boolean;
                case 8:
                    return ValueType::Int8;
                case 16:
                    return ValueType::Int16;
                case 32:
                    return ValueType::Int32;
                case 64:
                    return ValueType::Int64;
                default:
                    return ValueType::Undefined;
                }
            }
            else
            {
                switch (iTy.getWidth())
                {
                case 1:
                    return ValueType::Boolean;
                case 8:
                    return ValueType::Byte;
                case 16:
                    return ValueType::Uint16;
                case 32:
                    return ValueType::Uint32;
                case 64:
                    return ValueType::Uint64;
                default:
                    return ValueType::Undefined;
                }
            }
        })
        .Case<mlir::IndexType>([](mlir::IndexType idxTy) {
            return ValueType::Index;
        })
        .Case<mlir::FloatType>([](mlir::FloatType fTy) {
            if (fTy.isF16())
                return ValueType::Float16;
            if (fTy.isBF16())
                return ValueType::BFloat16;
            if (fTy.isF32())
                return ValueType::Float;
            if (fTy.isF64())
                return ValueType::Double;

            return ValueType::Undefined;
        })
        .Default([](mlir::Type) {
            return ValueType::Undefined;
        });
}

using accera::utilities::MemoryLayout;
ViewAdapter Wrap(mlir::Value v, std::optional<MemoryLayout> layout /*= std::nullopt*/)
{
    auto& ctx = GetMLIRContext();
    auto type = v.getType();
    if (!layout)
    {
        layout = InferLayoutFromMLIRValue(v);
    }
    if (auto shapedType = type.dyn_cast<mlir::ShapedType>())
    {
        assert(shapedType.hasStaticShape());

        auto eltType = shapedType.getElementType();
        auto eltValueType = MLIRTypeToValueType(eltType);
        assert(eltValueType != ValueType::Undefined);

        MLIRContext::EmittableInfo& emittableInfo =
            ctx.StoreLocalEmittable(
                { v.getAsOpaquePointer(), { eltValueType, 1 } });
        Emittable emittable{ &emittableInfo };

        return Value{ emittable, layout };
    }
    else if (type.isIntOrIndexOrFloat())
    {
        auto eltValueType = MLIRTypeToValueType(type);
        assert(eltValueType != ValueType::Undefined);
        MLIRContext::EmittableInfo& emittableInfo =
            ctx.StoreLocalEmittable(
                { v.getAsOpaquePointer(), { eltValueType, 1 } });
        Emittable emittable{ &emittableInfo };

        return Value{ emittable, layout };
    }
    else
    {
        throw std::logic_error("Unsupported type");
    }
}

std::vector<ViewAdapter> Wrap(std::vector<mlir::Value> values, std::function<MemoryLayout(mlir::Value)> layoutFn /*= nullptr*/)
{
    if (!layoutFn)
    {
        layoutFn = InferLayoutFromMLIRValue;
    }

    std::vector<ViewAdapter> wrappedValues;
    wrappedValues.reserve(values.size());
    llvm::transform(values, std::back_inserter(wrappedValues), [&](mlir::Value v) {
        return Wrap(v, layoutFn(v));
    });

    return wrappedValues;
}

Value ResolveConstantDataReference(Value constantDataSource)
{
    return GetContext().ResolveConstantDataReference(constantDataSource);
}

/*static*/ GPUIndex GPU::BlockDim()
{
    return GetMLIRContext().GetGPUIndex<GPUIndexType::BlockDim>();
}
/*static*/ GPUIndex GPU::BlockId()
{
    return GetMLIRContext().GetGPUIndex<GPUIndexType::BlockId>();
}
/*static*/ GPUIndex GPU::GridDim()
{
    return GetMLIRContext().GetGPUIndex<GPUIndexType::GridDim>();
}
/*static*/ GPUIndex GPU::ThreadId()
{
    return GetMLIRContext().GetGPUIndex<GPUIndexType::ThreadId>();
}

static std::string GPUBarrierScopeToValueIRBarrierScope(GPU::BarrierScope scope)
{
    switch (scope)
    {
    case GPU::BarrierScope::Block:
        return "Block";
    case GPU::BarrierScope::Warp:
        return "Warp";
    case GPU::BarrierScope::Threadfence:
        return "Threadfence";
    default:
        llvm_unreachable("Unhandled case");
    }
}

/*static*/ void GPU::Barrier(GPU::BarrierScope scope)
{
    auto& b = GetMLIRContext().GetOpBuilder();
    auto loc = b.getUnknownLoc();

    (void)ir::util::CreateGPUControlBarrier(b, GPUBarrierScopeToValueIRBarrierScope(scope), loc);
}

// this is declaring externs to reference the fillResource fn's in
// mlir/tools/mlir-vulkan-runner/vulkan-runtime-wrappers.cpp and then proceeds
// to emit calls to these functions
void FillResource(ViewAdapter resourceView, Scalar fillValue)
{
    auto res = Unwrap(resourceView);
    auto resType = res.getType();
    if (auto shapedType = resType.dyn_cast<mlir::MemRefType>(); !shapedType)
    {
        throw std::logic_error{ "Resource to be filled in must be valid memory" };
    }
    else
    {
        auto fill = Unwrap(fillValue);
        auto elemTy = shapedType.getElementType();
        if (elemTy != fill.getType())
        {
            throw std::logic_error{ "Fill value must have same type as resource element type" };
        }

        auto rank = shapedType.getRank();
        if (rank < 1 || rank > 3)
        {
            throw std::logic_error{ "Cannot fill resource with rank less than 1 or greater than 3" };
        }

        auto memorySpace = shapedType.getMemorySpace();
        auto identityLayout = mlir::MemRefLayoutAttrInterface{};
        auto castType = mlir::MemRefType::get(llvm::makeArrayRef(llvm::SmallVector<int64_t, 3>((size_t)rank, -1)), elemTy, identityLayout, memorySpace);

        auto& b = GetMLIRContext().GetOpBuilder();
        auto loc = b.getUnknownLoc();
        mlir::Value memrefCasted = b.create<mlir::memref::CastOp>(
            loc,
            res,
            castType);

        auto DeclareFn = [&](const std::string& name, mlir::FunctionType fnTy) -> ir::value::ValueFuncOp {
            auto mod = res.getParentRegion()->getParentOfType<ir::value::ValueModuleOp>();
            assert(mod);

            if (auto fnOp = mod.lookupSymbol<ir::value::ValueFuncOp>(name); fnOp) return fnOp;

            auto insertPt = ir::util::GetTerminalInsertPoint<
                ir::value::ValueModuleOp,
                ir::value::ModuleTerminatorOp,
                ir::value::ValueFuncOp>(mod);

            mlir::OpBuilder::InsertionGuard guard(b);
            b.restoreInsertionPoint(insertPt);

            ir::value::ValueFuncOp fnOp = b.create<ir::value::ValueFuncOp>(
                loc,
                name,
                fnTy,
                ir::value::ExecutionTarget::CPU,
                ir::value::ValueFuncOp::ExternalFuncTag{});
            fnOp.setPrivate();
            fnOp->setAttr(ir::CInterfaceAttrName, b.getUnitAttr());

            return fnOp;
        };

        auto callSuffix = [&]() -> std::string {
            if (elemTy.isIntOrIndex()) return "DInt";
            if (elemTy.isF32()) return "DFloat";
            throw LogicException(LogicExceptionErrors::illegalState, __FILE__ " : " + std::to_string(__LINE__));
        }();

        (void)b.create<ir::value::LaunchFuncOp>(
            loc,
            DeclareFn(std::string{ "fillResource" } + std::to_string(rank) + callSuffix,
                      b.getFunctionType({ castType, elemTy }, {})),
            mlir::ValueRange{ memrefCasted,
                              fill });
    }
}

// this is declaring externs to reference the fillResource fn's in
// mlir/include/mlir/ExecutionEngine/RunnerUtils.h and then proceeds
// to emit calls to these functions
void PrintMemref(ViewAdapter memView)
{
    auto mem = Unwrap(memView);
    auto memType = mem.getType();
    if (auto shapedType = memType.dyn_cast<mlir::MemRefType>(); !shapedType)
    {
        throw std::logic_error{ "Resource to be filled in must be valid memory" };
    }
    else
    {
        auto& b = GetMLIRContext().GetOpBuilder();
        auto loc = b.getUnknownLoc();
        auto elemTy = shapedType.getElementType();

        if (!(elemTy == b.getI32Type() || elemTy == b.getF32Type()))
        {
            throw std::logic_error{ "Memref to be printed must either have i32 or f32 element type" };
        }
        mlir::Value memrefCasted = b.create<mlir::memref::CastOp>(
            loc,
            mem,
            mlir::UnrankedMemRefType::get(elemTy, shapedType.getMemorySpace()));

        auto DeclareFn = [&](const std::string& name, mlir::FunctionType fnTy) -> ir::value::ValueFuncOp {
            auto mod = mem.getParentRegion()->getParentOfType<ir::value::ValueModuleOp>();
            assert(mod);

            if (auto fnOp = mod.lookupSymbol<ir::value::ValueFuncOp>(name); fnOp) return fnOp;

            auto insertPt = ir::util::GetTerminalInsertPoint<
                ir::value::ValueModuleOp,
                ir::value::ModuleTerminatorOp,
                ir::value::ValueFuncOp>(mod);

            mlir::OpBuilder::InsertionGuard guard(b);
            b.restoreInsertionPoint(insertPt);

            ir::value::ValueFuncOp fnOp = b.create<ir::value::ValueFuncOp>(
                loc,
                name,
                fnTy,
                ir::value::ExecutionTarget::CPU,
                ir::value::ValueFuncOp::ExternalFuncTag{});
            fnOp->setAttr(ir::CInterfaceAttrName, b.getUnitAttr());
            fnOp.setPrivate();
            return fnOp;
        };

        if (elemTy.isF32())
        {
            (void)b.create<ir::value::LaunchFuncOp>(
                loc,
                DeclareFn("print_memref_f32",
                          b.getFunctionType({ memrefCasted.getType() }, {})),
                mlir::ValueRange{ memrefCasted });
        }
        else if (elemTy == b.getI32Type())
        {
            (void)b.create<ir::value::LaunchFuncOp>(
                loc,
                DeclareFn("print_memref_i32",
                          b.getFunctionType({ memrefCasted.getType() }, {})),
                mlir::ValueRange{ memrefCasted });
        }
    }
}

mlir::OwningOpRef<mlir::ModuleOp> GatherModules(const std::string& name, const std::vector<value::MLIRContext*>& contexts, mlir::MLIRContext* context)
{
    auto topLevelModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(context), llvm::StringRef(name));
    mlir::OpBuilder builder(context);
    builder.restoreInsertionPoint(ir::util::GetTerminalInsertPoint<mlir::ModuleOp, mlir::FuncOp>(topLevelModule));
    for (const auto& contextPtr : contexts)
    {
        auto moduleCopy = contextPtr->cloneModule();
        mlir::ModuleOp op = moduleCopy.get();
        builder.clone(*op.getOperation());
    }
    return topLevelModule;
}

void SaveModule(const std::string& filename, mlir::ModuleOp moduleOp)
{
    std::error_code ec;
    llvm::raw_fd_ostream s(filename, ec);
    moduleOp.print(s, mlir::OpPrintingFlags{}.enableDebugInfo(false));
}

void WriteHeaderForModule(const std::string& filename, mlir::ModuleOp moduleOp)
{
    std::error_code ec;
    llvm::raw_fd_ostream fstream(filename, ec);
    (void)ir::TranslateToHeader(moduleOp, fstream);
}

void WriteHeaderForModules(const std::string& filename,
                           const std::string& libraryName,
                           const std::vector<value::MLIRContext*>& contexts)
{
    std::error_code ec;
    llvm::raw_fd_ostream fstream(filename, ec);
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> owningModuleRefs;
    std::vector<mlir::ModuleOp> moduleOps;
    owningModuleRefs.reserve(contexts.size());
    moduleOps.reserve(contexts.size());
    for (const auto& contextPtr : contexts)
    {
        auto moduleCopy = contextPtr->cloneModule();
        moduleOps.push_back(moduleCopy.get());
        owningModuleRefs.push_back(std::move(moduleCopy));
    }
    (void)ir::TranslateToHeader(moduleOps, libraryName, fstream);
}

} // namespace accera::value
