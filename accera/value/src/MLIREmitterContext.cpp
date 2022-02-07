////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MLIREmitterContext.h"
#include "CompilerOptions.h"

#include <ir/include/DialectRegistry.h>
#include <ir/include/IRUtil.h>
#include <ir/include/InitializeAccera.h>
#include <ir/include/TranslateToHeader.h>
#include <ir/include/exec/ExecutionPlanAttributes.h>
#include <ir/include/exec/VectorizationInfo.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueAttributes.h>
#include <ir/include/value/ValueFuncOp.h>

#include <llvm/Support/ErrorHandling.h>
#include <transforms/include/value/ValueToStandardLoweringPass.h>

#include <utilities/include/Exception.h>
#include <utilities/include/MemoryLayout.h>
#include <utilities/include/ZipIterator.h>

#include <value/include/Debugging.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
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
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>

using namespace accera;
using namespace accera::utilities;
using namespace accera::value;
using ConstantData = accera::value::detail::ConstantData;

namespace
{
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

mlir::Type ToMLIRType(mlir::OpBuilder& builder, ValueType type)
{

    switch (type)
    {
    case ValueType::Boolean:
        return builder.getIntegerType(1);
    case ValueType::Byte:
        return builder.getIntegerType(8, false);
    case ValueType::Int8:
        return builder.getIntegerType(8);
    case ValueType::Int16:
        return builder.getIntegerType(16);
    case ValueType::Int32:
        return builder.getIntegerType(32);
    case ValueType::Int64:
        return builder.getIntegerType(64);
    case ValueType::Index:
        return builder.getIndexType();
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

MemoryLayout GetSubArrayLayout(const MemoryLayout& originalLayout, const MemoryShape& shape)
{
    return { shape, originalLayout.GetExtent(), originalLayout.GetOffset() };
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

mlir::MemRefType MemoryLayoutToMemRefType(mlir::OpBuilder& builder, const MemoryLayout& layout, ValueType valueType, bool useDynamicOffset)
{
    auto mlirElemType = ToMLIRType(builder, valueType);

    if (layout == ScalarLayout)
    {
        return mlir::MemRefType::get({}, mlirElemType);
    }

    auto size = layout.GetActiveSize().ToVector();
    auto strides = layout.GetIncrement().ToVector();
    int64_t offset = useDynamicOffset ? mlir::MemRefType::getDynamicStrideOrOffset() : static_cast<int64_t>(layout.GetFirstEntryOffset());

    auto context = builder.getContext();
    auto stridedMap = mlir::makeStridedLinearLayoutMap(strides, offset, context);

    auto type = mlir::MemRefType::get(size, mlirElemType, stridedMap, (unsigned)layout.GetMemorySpace());

    return type;
}

mlir::MemRefType MemoryLayoutToMemRefType(mlir::OpBuilder& builder, const MemoryLayout& layout, ValueType valueType)
{
    return MemoryLayoutToMemRefType(builder, layout, valueType, false);
}

auto MemoryLayoutToTensorType(mlir::OpBuilder& builder, const MemoryLayout& layout, ValueType valueType)
{
    // TODO: Figure out whether this assert needs to be active
    // assert(layout.IsCanonicalOrder() && "Can only get a tensor type from a canonically-ordered layout");

    auto mlirElemType = ToMLIRType(builder, valueType);
    llvm::SmallVector<int64_t, 4> extents;

    extents.append(layout.GetExtent().begin(), layout.GetExtent().end());

    auto type = mlir::RankedTensorType::get(extents, mlirElemType);

    return type;
}

mlir::Type ToMLIRType(mlir::OpBuilder& builder, Value value)
{
    if (value.IsConstrained())
    {
        auto& layout = value.GetLayout();
        if (layout == ScalarLayout)
        {
            return ToMLIRType(builder, value.GetBaseType());
        }
        else
        {
            return MemoryLayoutToMemRefType(builder, layout, value.GetBaseType());
        }
    }
    else
    {
        auto mlirElemType = ToMLIRType(builder, value.GetBaseType());
        auto type = mlir::UnrankedMemRefType::get(mlirElemType, 0);
        return type;
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
        return builder.create<mlir::IndexCastOp>(loc, v, mlir::IndexType::get(v.getContext()));
    }

    // Index types fall through
    return v;
}

[[nodiscard]] mlir::Value ToMLIRValue(mlir::OpBuilder& builder, ViewAdapter view)
{
    auto value = view.GetValue();
    if (value.IsEmpty() || value.IsUndefined() || value.IsConstant())
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
                while (it != end && llvm::isa<mlir::ConstantOp,
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
        return builder.create<mlir::IndexCastOp>(loc, mlirValue, indexType);
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

    auto mlirElementType = ToMLIRType(builder, value.GetBaseType());
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
            name += "_" + std::to_string(ir::util::GetUniqueId());
        }

        mlir::SymbolTable::setSymbolName(op, name);
    }
}

auto GetConstantDataElementType(const ConstantData& data)
{
    return std::visit(
        [](auto&& data) {
            using DataType = std::decay_t<decltype(data)>;
            using ElementType = typename DataType::value_type;

            return GetValueType<ElementType>();
        },
        data);
}

auto ConstantDataToDenseElementAttr(mlir::ShapedType shape, const ConstantData& data)
{
    return std::visit(
        [shape](auto&& data) -> mlir::DenseElementsAttr {
            using DataType = std::decay_t<decltype(data)>;
            using ElementType = typename DataType::value_type;

            if constexpr (std::is_same_v<ElementType, Boolean>)
            {
                std::vector<int8_t> boolData(data.size());
                std::transform(data.begin(), data.end(), boolData.begin(), [](Boolean b) { return b ? 1 : 0; });

                return mlir::DenseElementsAttr::get(shape, llvm::makeArrayRef(boolData));
            }
            else if constexpr (std::is_same_v<ElementType, index_t>)
            {
                throw InputException(InputExceptionErrors::invalidArgument, "Can't store an array of index type");
            }
            else
            {
                return mlir::DenseElementsAttr::get(shape, llvm::makeArrayRef(data));
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

GPUIndex::GPUIndex(std::function<Scalar(const std::string&)> fn) :
    _fn(std::move(fn))
{}

Scalar GPUIndex::X()
{
    return _fn("x");
}
Scalar GPUIndex::Y()
{
    return _fn("y");
}
Scalar GPUIndex::Z()
{
    return _fn("z");
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
    mlir::OwningModuleRef _ownedModule;
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

    mlir::OwningModuleRef cloned = _impl->_mlirModule.clone();
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

mlir::OwningModuleRef MLIRContext::cloneModule() const
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
    auto newKeyId = mlir::Identifier::get(key, context);
    auto valueAttr = ir::GetMetadataAttr(value, context);
    auto metadataKeyId = mlir::Identifier::get(accera::ir::value::ValueDialect::getAcceraMetadataAttrName(), context);
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
    auto& builder = _impl->builder;
    auto context = _impl->_valueModuleOp.getContext();
    auto debugModeId = mlir::Identifier::get(accera::ir::GetDebugModeAttrName(), context);
    if (enable)
    {
        _impl->_valueModuleOp->setAttr(debugModeId, builder.getUnitAttr());
    }
    else
    {
        _impl->_valueModuleOp->removeAttr(debugModeId);
    }
}

void MLIRContext::EmitDebugFunction(const std::string& functionName, const std::vector<std::string>& utilityFunctionNames)
{
    for (const auto& fn : _definedFunctions)
    {
        // Attempt to do a basename comparison instead of the the full name
        if (fn.first.GetFunctionName().compare(0, functionName.length(), functionName) == 0)
        {
            // Do a best effort emitting of the debug function
            EmitNestDebugFunction(fn.first, utilityFunctionNames);
        }
    }
}

template <typename Op>
Scalar CreateGPUIndexOp(mlir::OpBuilder& builder, const std::string& dim)
{
    auto loc = builder.getUnknownLoc();
    return Wrap(
        builder.create<mlir::IndexCastOp>(loc,
                                          builder.create<Op>(
                                              loc,
                                              builder.getIndexType(),
                                              builder.getStringAttr(dim)),
                                          builder.getI64Type()));
}

GPUIndex MLIRContext::GetGPUIndex(GPUIndexType type)
{
    switch (type)
    {
    case GPUIndexType::BlockDim:
        return GPUIndex{ [this](const std::string& dim) { return CreateGPUIndexOp<mlir::gpu::BlockDimOp>(_impl->builder, dim); } };
    case GPUIndexType::BlockId:
        return GPUIndex{ [this](const std::string& dim) { return CreateGPUIndexOp<mlir::gpu::BlockIdOp>(_impl->builder, dim); } };
    case GPUIndexType::GridDim:
        return GPUIndex{ [this](const std::string& dim) { return CreateGPUIndexOp<mlir::gpu::GridDimOp>(_impl->builder, dim); } };
    case GPUIndexType::ThreadId:
        return GPUIndex{ [this](const std::string& dim) { return CreateGPUIndexOp<mlir::gpu::ThreadIdOp>(_impl->builder, dim); } };
    }
    llvm_unreachable("Unknown GPU index type");
}

Value MLIRContext::AllocateImpl(ValueType valueType, MemoryLayout layout, size_t alignment, AllocateFlags flags)
{
    auto& b = _impl->builder;
    if (layout.GetMemorySpace() == MemorySpace::None)
    {
        layout = layout.SetMemorySpace(MemorySpace::Shared);
    }
    auto memrefTy = MemoryLayoutToMemRefType(b, layout, valueType);

    auto insertionBlock = b.getInsertionBlock();
    auto it = insertionBlock->begin();
    auto end = insertionBlock->end();
    while (it != end && llvm::isa<mlir::ConstantOp,
                                  mlir::memref::AllocOp,
                                  mlir::memref::AllocaOp,
                                  ir::value::ReferenceGlobalOp,
                                  ir::value::AllocOp>(it))
    {
        ++it;
    }
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(insertionBlock, it);
    auto loc = b.getUnknownLoc();

    mlir::Value result = b.create<ir::value::AllocOp>(loc,
                                                      memrefTy,
                                                      alignment
                                                          ? llvm::Optional{ (int64_t)alignment }
                                                          : llvm::None,
                                                      static_cast<bool>(flags & AllocateFlags::Stack)
                                                          ? llvm::Optional{ accera::ir::value::MemoryAllocType::Stack }
                                                          : llvm::None);

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
    auto fnType = ToMLIRType(b, decl);

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

            // Collect function tags into a dictionary
            auto tags = decl.GetTags();
            std::vector<mlir::NamedAttribute> tagAttrs;
            if (!tags.empty())
            {
                for (const auto& tag : tags)
                {
                    tagAttrs.emplace_back(b.getIdentifier(tag), b.getUnitAttr());
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
                if (funcRuntime != ExecutionRuntime::Default)
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
                    b.getIndexArrayAttr({
                        target.grid.x,
                        target.grid.y,
                        target.grid.z,
                        target.block.x,
                        target.block.y,
                        target.block.z,
                    }));
            }

            return std::pair{ fnOp.getOperation(), &fnOp.body().back() };
        },
        funcTarget);

    {
        auto fnContext = _impl->CreateNewScope({ entryBlock, entryBlock->begin() });

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
    auto fnTy = ToMLIRType(builder, decl);
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
            auto mlirElemTy = ToMLIRType(b, valueElemTy);
            mlir::Value op;

            auto insertionBlock = b.getInsertionBlock();

            auto it = insertionBlock->begin();
            auto end = insertionBlock->end();
            while (it != end && llvm::isa<mlir::ConstantOp>(it))
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
                    op = b.create<mlir::ConstantIndexOp>(loc, static_cast<int64_t>(data[0]));
                }
                else if constexpr (std::is_integral_v<ElementType> || std::is_same_v<ElementType, Boolean>)
                {
                    auto elem = static_cast<int64_t>(data[0]);
                    op = b.create<mlir::ConstantIntOp>(loc, elem, mlirElemTy);
                }
                else if constexpr (std::is_floating_point_v<ElementType>)
                {
                    op = b.create<mlir::ConstantFloatOp>(loc, llvm::APFloat(data[0]), mlirElemTy.cast<mlir::FloatType>());
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

                auto mlirElemType = ToMLIRType(b, valueElemTy);

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
                else
                {
                    dataAttribute = mlir::DenseElementsAttr::get(flattenedTensorShapeTy, llvm::makeArrayRef(data));
                }

                std::string uniquedName = name + "_" + std::to_string(ir::util::GetUniqueId());

                mlir::MemRefType globalMemrefTy = mlir::MemRefType::Builder{ memrefShapeTy }.setAffineMaps({}); // remove affine maps
                [[maybe_unused]] auto globalOp = b.create<ir::value::GlobalOp>(loc, globalMemrefTy, /* isConstant= */ true, uniquedName, dataAttribute, /*addrSpace*/ 0, /*isExternal*/ true);
                op = b.create<ir::value::ReferenceGlobalOp>(loc, memrefShapeTy, uniquedName);
            }

            return StoreLocalEmittable({ op.getAsOpaquePointer(), { valueElemTy, 1 } });
        },
        data);

    Emittable emittable(&emittableInfo);

    return Value(emittable, layout);
}

Value MLIRContext::ResolveConstantDataReferenceImpl(Value constantDataSource)
{
    auto sourceRefGlobalOp = mlir::Value::getFromOpaquePointer(constantDataSource.Get<Emittable>().GetDataAs<EmittableInfo*>()->data).getDefiningOp();
    auto& builder = _impl->builder;

    {
        // Clone the GlobalOp at the top of this module and mark it as external
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.restoreInsertionPoint(_impl->getGlobalInsertPt());

        auto sourceGlobalOp = mlir::dyn_cast<ir::value::ReferenceGlobalOp>(sourceRefGlobalOp).getGlobal();
        auto globalOp = mlir::dyn_cast<ir::value::GlobalOp>(builder.clone(*sourceGlobalOp));
        globalOp->setAttr("external", builder.getUnitAttr());
        globalOp->removeAttr("value"); // can't set null attribute (mlir::Attribute()) if existing
    }

    // Clone a RefefenceGlobalOp to refer to the GlobalOp. Since this is a clone, the reference *should* "carry over" without
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
        LBs(dim, builder.create<mlir::ConstantIndexOp>(loc, 0)),
        UBs;
    std::vector<int64_t> steps(dim, 1);
    for (unsigned i = 0; i < dim; ++i)
    {
        UBs.emplace_back(builder.create<mlir::ConstantIndexOp>(loc, layout.GetActiveSize(i)));
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
    auto vectorizationInfoIdentifier = builder.getIdentifier(ir::executionPlan::VectorizationInfoAttr::getKeyName());
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
    auto vectorizationInfoIdentifier = builder.getIdentifier(ir::executionPlan::VectorizationInfoAttr::getKeyName());
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

Value MLIRContext::ViewImpl(Value sourceValue, const std::vector<Scalar>& offsetsValue, const MemoryShape& shape, const std::vector<Scalar>& strides_)
{
    const MemoryLayout& currentLayout = sourceValue.GetLayout();
    auto destLayout = GetSubArrayLayout(currentLayout, shape);

    std::vector<Scalar> strides = strides_;
    if (strides.empty())
    {
        strides = std::vector<Scalar>(offsetsValue.size(), Scalar(1));
    }

    llvm::SmallVector<mlir::Value, 4> linalgRanges;
    auto ranges = utilities::MakeZipRange(offsetsValue, shape, strides);
    std::transform(
        begin(ranges),
        end(ranges),
        std::back_inserter(linalgRanges),
        [this](std::tuple<Scalar, Scalar, Scalar> s) -> mlir::Value {
            auto& builder = _impl->builder;
            auto loc = builder.getUnknownLoc();
            auto offset = Scalar(Cast(std::get<0>(s), ValueType::Index));
            auto size = Scalar(Cast(Scalar(std::get<1>(s)), ValueType::Index));
            auto step = Scalar(Cast(Scalar(std::get<2>(s)), ValueType::Index));
            auto lowerBound = ResolveMLIRIndex(builder, offset);
            auto upperBound = ResolveMLIRIndex(builder, size);
            auto stride = ResolveMLIRIndex(builder, step);
            return builder.create<mlir::linalg::RangeOp>(loc, lowerBound, upperBound, stride);
        });

    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    auto resultMemRefType = MemoryLayoutToMemRefType(builder, destLayout, sourceValue.GetBaseType(), true);

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

    auto resultMemRefType = MemoryLayoutToMemRefType(builder, destLayout, sourceValue.GetBaseType(), true);
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
    auto resultMemRefType = MemoryLayoutToMemRefType(builder, destLayout, sourceValue.GetBaseType(), false);
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
    auto resultMemRefType = MemoryLayoutToMemRefType(builder, destLayout, sourceValue.GetBaseType(), false);
    auto source = ToMLIRValue(builder, sourceValue);
    return Wrap(builder.create<ir::value::SplitDimOp>(loc, resultMemRefType, source, dim, size));
}

Value MLIRContext::ReshapeImpl(Value sourceValue, const MemoryLayout& destLayout)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    // TODO: assert there's no offset
    auto resultMemRefType = MemoryLayoutToMemRefType(builder, destLayout, sourceValue.GetBaseType(), false);
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

void MLIRContext::MFMAImpl(Matrix& dest, Matrix A, Matrix B, Matrix C)
{
    using namespace accera::ir::value;

    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();

    auto destValue = ToMLIRValue(builder, dest);
    auto aValue = ToMLIRValue(builder, A);
    auto bValue = ToMLIRValue(builder, B);
    auto cValue = ToMLIRValue(builder, C);

    auto getMatrixTypeOfMemref = [=](mlir::Value val, llvm::StringRef kind) {
        auto memrefType = val.getType().cast<mlir::MemRefType>();
        return MFMAMatrixType::get(
            memrefType.getShape(), memrefType.getElementType(), kind);
    };

    mlir::Value aMatrix = builder.create<ir::value::MFMALoadMatrixOp>(loc, getMatrixTypeOfMemref(aValue, "AOp"), aValue);
    mlir::Value bMatrix = builder.create<ir::value::MFMALoadMatrixOp>(loc, getMatrixTypeOfMemref(aValue, "BOp"), bValue);
    mlir::Value cMatrix = builder.create<ir::value::MFMALoadMatrixOp>(loc, getMatrixTypeOfMemref(aValue, "COp"), cValue);

    auto result = builder.create<ir::value::MFMAComputeOp>(loc, cValue.getType(), aMatrix, bMatrix, cMatrix);

    builder.create<ir::value::MFMAStoreMatrixOp>(loc, result, destValue);

    throw LogicException(LogicExceptionErrors::notImplemented);

    // EmittableInfo& emittableInfo = StoreLocalEmittable({ result.getAsOpaquePointer(), { C.GetBaseType(), 1 } });
    // Emittable emittable{ &emittableInfo };

    // return Value( emittable, C.GetLayout() );
}

Scalar MLIRContext::CastImpl(Scalar value, ValueType type, bool srcSigned)
{
    auto& builder = _impl->builder;
    mlir::Value mlirValue = ResolveMLIRScalar(builder, ToMLIRValue(builder, value));

    auto loc = mlirValue.getLoc();
    auto fromType = mlirValue.getType();
    auto toType = ToMLIRType(builder, type);
    if (fromType == toType)
    {
        return Wrap(mlirValue);
    }

    if (auto fromIntType = fromType.dyn_cast<mlir::IntegerType>(); fromIntType && !fromIntType.isUnsigned()) // signed or signless integer
    {
        if (auto toIntType = toType.dyn_cast<mlir::IntegerType>())
        {
            // int->int
            if (fromIntType.getWidth() > toIntType.getWidth())
            {
                return Wrap(builder.create<mlir::TruncateIOp>(loc, mlirValue, toType));
            }
            else
            {
                if (toIntType.isUnsigned())
                {
                    return Wrap(builder.create<mlir::ZeroExtendIOp>(loc, mlirValue, toType));
                }
                else
                {
                    if (srcSigned)
                    {
                        return Wrap(builder.create<mlir::SignExtendIOp>(loc, mlirValue, toType));
                    }
                    else
                    {
                        return Wrap(builder.create<mlir::ZeroExtendIOp>(loc, mlirValue, toType));
                    }
                }
            }
        }
        else if (auto toIndexType = toType.dyn_cast<mlir::IndexType>())
        {
            // int->index
            return Wrap(builder.create<mlir::IndexCastOp>(loc, mlirValue, toType));
        }
        else if (auto toFloatType = toType.dyn_cast<mlir::FloatType>())
        {
            // int->fp
            return Wrap(builder.create<mlir::SIToFPOp>(loc, mlirValue, toType));
        }

        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
    }
    else if (fromIntType) // explicitly unsigned
    {
        if (auto toIntType = toType.dyn_cast<mlir::IntegerType>())
        {
            // int->int
            if (fromIntType.getWidth() > toIntType.getWidth())
            {
                return Wrap(builder.create<mlir::TruncateIOp>(loc, mlirValue, toType));
            }
            else
            {
                return Wrap(builder.create<mlir::ZeroExtendIOp>(loc, mlirValue, toType));
            }
        }
        else if (auto toIndexType = toType.dyn_cast<mlir::IndexType>())
        {
            // MLIR forbids casting unsigned ints to index
            // TODO: first cast to a signless int
            // int->index
            return Wrap(builder.create<mlir::IndexCastOp>(loc, mlirValue, toType));
        }
        else if (auto toFloatType = toType.dyn_cast<mlir::FloatType>())
        {
            // int->fp
            return Wrap(builder.create<mlir::UIToFPOp>(loc, mlirValue, toType));
        }

        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
    }
    else if (auto fromIndexType = fromType.dyn_cast<mlir::IndexType>())
    {
        if (auto toIntType = toType.dyn_cast<mlir::IntegerType>())
        {
            // index->int
            return Wrap(builder.create<mlir::IndexCastOp>(loc, mlirValue, toType));
        }
        else if (auto toFloatType = toType.dyn_cast<mlir::FloatType>())
        {
            // index->int64
            auto intValue = builder.create<mlir::IndexCastOp>(loc, mlirValue, builder.getI64Type());

            // int64->fp
            return Wrap(builder.create<mlir::SIToFPOp>(loc, intValue, toType));
        }

        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
    }
    else if (auto fromFloatType = fromType.dyn_cast<mlir::FloatType>())
    {
        if (auto toIntType = toType.dyn_cast<mlir::IntegerType>())
        {
            // float->int
            return Wrap(builder.create<mlir::FPToSIOp>(loc, mlirValue, toType));
        }
        else if (auto toFloatType = toType.dyn_cast<mlir::FloatType>())
        {
            // float->float
            if (fromFloatType.getWidth() > toFloatType.getWidth())
            {
                return Wrap(builder.create<mlir::FPTruncOp>(loc, mlirValue, toType));
            }
            else
            {
                return Wrap(builder.create<mlir::FPExtOp>(loc, mlirValue, toType));
            }
        }

        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
    }

    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, __FILE__ " : " + std::to_string(__LINE__));
}

Scalar MLIRContext::CastImpl(Scalar value, ValueType type)
{
    return CastImpl(value, type, true);
}

Scalar MLIRContext::UnsignedCastImpl(Scalar value, ValueType type)
{
    return CastImpl(value, type, false);
}

Scalar MLIRContext::BitcastImpl(Scalar value, ValueType type)
{
    auto& builder = _impl->builder;
    mlir::Value mlirValue = ResolveMLIRScalar(builder, ToMLIRValue(builder, value));

    auto loc = mlirValue.getLoc();
    auto fromType = mlirValue.getType();
    auto toType = ToMLIRType(builder, type);
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

            mlir::scf::IfOp::ensureTerminator(ifOp.thenRegion(), builder, loc);

            if (auto& elseRegion = ifOp.elseRegion(); !elseRegion.empty())
            {
                mlir::scf::IfOp::ensureTerminator(elseRegion, builder, loc);
            }

            return ifOp.results();
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

            mlir::scf::IfOp::ensureTerminator(ifOp.thenRegion(), builder, loc);

            if (auto& elseRegion = ifOp.elseRegion(); !elseRegion.empty())
            {
                mlir::scf::IfOp::ensureTerminator(elseRegion, builder, loc);
            }
            return ifOp.results();
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
        DeclareFn(func.GetFunctionName(), ToMLIRType(builder, func)),
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

    // cast to a value with type `memref<total_size x elem_type>` (via `memref<* x elem_type>`)
    mlir::Value ptr = builder.create<mlir::memref::CastOp>(loc, mem, mlir::UnrankedMemRefType::get(elemTy, memType.getMemorySpace()));
    mlir::Value mlirValue = builder.create<mlir::memref::CastOp>(loc, ptr, mlir::MemRefType::get({ size }, elemTy, {}, memType.getMemorySpace()));

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

// Emit a wrapper function that will invoke the target function with debugging checks
// This is best effort. If there is no ScheduleOp, we will skip the function.
void MLIRContext::EmitNestDebugFunction(FunctionDeclaration targetFunc, const std::vector<std::string>& utilityFunctionNames)
{
    auto& builder = _impl->builder;
    auto loc = builder.getUnknownLoc();
    auto parentOp = builder.getBlock()->getParentOp();
    auto moduleOp = accera::ir::util::CastOrGetParentOfType<mlir::ModuleOp>(parentOp);
    assert(moduleOp);
    auto targetFuncName = targetFunc.GetFunctionName();

    // Find a ValueFuncOp matching the target function name
    if (auto targetFnOp = FindValueFuncOp(moduleOp, targetFuncName))
    {
        // Find the ScheduleOp
        ir::loopnest::ScheduleOp scheduleOp;
        if (auto region = &targetFnOp.body())
        {
            region->walk([&scheduleOp](ir::loopnest::ScheduleOp op) {
                scheduleOp = op;
                return mlir::WalkResult::interrupt();
            });
        }
        if (scheduleOp)
        {
            // Find the LaunchFuncOp that calls target function. This will be used for the debug function name prefix
            // and also for replacement with a new LaunchFuncOp that calls the debug wrapper function.
            // If no LaunchFuncOp exists (because this does not have a raw pointer API wrapper function), fallback to
            // the target function name as the debug function name prefix.
            ir::value::ValueFuncOp targetLaunchFnOp;
            auto dbgFnName = [&targetLaunchFnOp, moduleOp, targetFuncName]() -> std::string {
                moduleOp->walk([&targetLaunchFnOp, targetFuncName](ir::value::LaunchFuncOp op) {
                    if (targetFuncName == op.callee().getLeafReference())
                    {
                        targetLaunchFnOp = accera::ir::util::CastOrGetParentOfType<ir::value::ValueFuncOp>(op);
                        return mlir::WalkResult::interrupt();
                    }
                    return mlir::WalkResult::advance();
                });
                std::string namePrefix = targetLaunchFnOp ? std::string(targetLaunchFnOp.sym_name()) : targetFuncName;
                return std::string("_debug_") + namePrefix;
            }();

            // Create a new function op with the same arguments and return value as the target function
            //      void dbgFnOp(args, ...)
            //      {
            //          Copy output args to output targetFnArgs
            //          Call targetFnOp(targetFnArgs, ...)
            //          Run default schedule impl using (args, ...)
            //          Call utility function to check output args vs output targetFnArgs
            //          Copy output targetFnArgs to output args
            //      }
            // TODO: The last copy can be avoided if we wrap the default schedule impl within its own ValueFuncOp
            auto dbgFnOp = [this, &builder, loc, &targetFnOp, &scheduleOp, dbgFnName]() -> ir::value::ValueFuncOp {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.restoreInsertionPoint(_impl->getFunctionInsertPt());

                auto argTypes = targetFnOp.getType().getInputs().vec();
                auto callingFnType = builder.getFunctionType(argTypes, targetFnOp.getType().getResults());
                auto wrapperFnOp = builder.create<ir::value::ValueFuncOp>(loc, dbgFnName + "_internal", callingFnType, targetFnOp.exec_target());

                builder.setInsertionPointToStart(&wrapperFnOp.body().front());

                // Map target function args to debug function args
                mlir::BlockAndValueMapping valueMap;
                for (auto [fromValue, toValue] : llvm::zip(targetFnOp.getArguments(), wrapperFnOp.getArguments()))
                {
                    valueMap.map(fromValue, toValue);
                }

                // Replicate local allocations (e.g. TEMP arrays) that exist outside of the nest
                // (Assumes that these are needed before the nest)
                targetFnOp->walk([&builder, &valueMap](ir::value::AllocOp op) {
                    auto newOp = mlir::cast<ir::value::AllocOp>(builder.clone(*op.getOperation()));
                    valueMap.map(op.getResult(), newOp.getResult());
                });

                // Create the reference schedule(s)
                auto targetNestOp = scheduleOp.getNest();
                if (auto fusedDomains = scheduleOp.getFusedDomains(); !fusedDomains.empty())
                {
                    // Fusing case: split into multiple schedules (one per kernel), in kernel order
                    auto kernels = targetNestOp.getKernels();

                    // We currently only support 1 kernel per domain (corresponds to a "never-fused-before" schedule)
                    // TODO: remove this limitation when the Python DSL supports adding multiple kernels to a schedule
                    assert(fusedDomains.size() == kernels.size() && "Number of unfused domains != number of unfused kernels");
                    for (auto [targetKernel, fusedDomain] : llvm::zip(kernels, fusedDomains))
                    {
                        auto nest = ir::loopnest::MakeNest(builder, fusedDomain);
                        auto nestBuilder = nest.getBodyBuilder();

                        // Map target symbolic indices to debug symbolic indices
                        auto dims = fusedDomain.GetDimensions();
                        std::unordered_set<ir::loopnest::Index> fusedDomainIndices(dims.begin(), dims.end());

                        targetNestOp.walk([&](ir::loopnest::SymbolicIndexOp fromIndex) {
                            if (!fromIndex.use_empty())
                            {
                                auto sourceIndex = fromIndex.getValue();
                                for (auto fusedIndex : scheduleOp.getFusedIndices(sourceIndex))
                                {
                                    // A reverse mapping of fused index to original index exists AND the original index
                                    // belongs in the unfused domain
                                    if (fusedDomainIndices.find(fusedIndex) != fusedDomainIndices.end())
                                    {
                                        sourceIndex = fusedIndex;
                                        break;
                                    }
                                }
                                auto toIndex = nest.getOrCreateSymbolicIndex(nestBuilder, sourceIndex);
                                valueMap.map(fromIndex.getResult(), toIndex.getResult());
                            }
                        });

                        // Clone the kernel, referencing the re-mapped Values (this creates the symbolic indices)
                        auto kernel = mlir::cast<ir::loopnest::KernelOp>(nestBuilder.clone(*targetKernel.getOperation(), valueMap));

                        // Create the schedule and add the kernels (after the symbolic indices have been inserted into the IR)
                        auto defaultSchedule = nest.getOrCreateSchedule();
                        defaultSchedule.addKernel(kernel);
                    }
                }
                else
                {
                    // Non-fusing case: duplicate the nest with its kernel(s)
                    auto domain = targetNestOp.getDomain().getValue();
                    auto nest = ir::loopnest::MakeNest(builder, domain);
                    auto nestBuilder = nest.getBodyBuilder();

                    // Map target symbolic indices to debug symbolic indices
                    targetNestOp.walk([&nestBuilder, &nest, &valueMap](ir::loopnest::SymbolicIndexOp fromIndex) {
                        if (!fromIndex.use_empty())
                        {
                            auto toIndex = nest.getOrCreateSymbolicIndex(nestBuilder, fromIndex.getValue());
                            valueMap.map(fromIndex.getResult(), toIndex.getResult());
                        }
                    });

                    // Clone the kernels, referencing the re-mapped Values (this creates the symbolic indices)
                    std::vector<ir::loopnest::KernelOp> kernels;
                    auto targetKernels = targetNestOp.getKernels();
                    std::transform(targetKernels.cbegin(), targetKernels.cend(), std::back_inserter(kernels), [&nestBuilder, &valueMap](auto knl) {
                        return mlir::cast<ir::loopnest::KernelOp>(nestBuilder.clone(*knl.getOperation(), valueMap));
                    });

                    // Create the schedule and add the kernels (after the symbolic indices have been inserted into the IR)
                    auto defaultSchedule = nest.getOrCreateSchedule();
                    for (auto& kernel : kernels)
                    {
                        defaultSchedule.addKernel(kernel);
                    }
                }
                return wrapperFnOp;
            }();

            {
                // Collect arguments for calling the target function, then inject a call to the target function
                // Output arguments will be duplicated so that we can compare the results with the reference implementation
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToStart(&dbgFnOp.body().front());

                // Replicate output args at the top of the function and copy the data
                auto targetFnArgs = [this, &builder, loc, targetFunc, &dbgFnOp]() -> std::vector<mlir::Value> {
                    std::string name = GetGlobalScopedName(dbgFnOp.getName().str() + "_target_output_arg");
                    std::vector<mlir::Value> fnArgs;
                    for (auto [blockArg, usage] : llvm::zip(dbgFnOp.getArguments(), targetFunc.GetParameterUsages()))
                    {
                        if (usage == FunctionParameterUsage::inputOutput)
                        {
                            if (auto memrefType = blockArg.getType().dyn_cast<mlir::MemRefType>())
                            {
                                // Simplify any identity affine maps, e.g. (d0, d1) -> (d0 * 256 + d1) can become (d0, d1) -> (d0, d1)
                                // Required by ConvertToLLVMPattern::isConvertibleAndHasIdentityMaps() in GlobalMemrefOpLowering
                                auto argCopy = ir::util::CreateGlobalBuffer(builder, dbgFnOp, mlir::canonicalizeStridedLayout(memrefType), name);

                                // Replace the global-scoped ReferenceGlobalOp with one within the function context
                                auto globalScopeGlobalRef = mlir::dyn_cast_or_null<ir::value::ReferenceGlobalOp>(argCopy.getDefiningOp());
                                auto localScopeGlobalRef = builder.create<accera::ir::value::ReferenceGlobalOp>(loc, globalScopeGlobalRef.getGlobal());
                                CopyData(Wrap(blockArg), Wrap(localScopeGlobalRef));
                                fnArgs.push_back(localScopeGlobalRef);
                                globalScopeGlobalRef.erase();
                            }
                            else
                            {
                                throw std::logic_error{ "Argument is not a memRefType" }; // TODO: support additional function arg types as needed
                            }
                        }
                        else
                        {
                            fnArgs.push_back(blockArg); // pass-through any input args
                        }
                    }
                    return fnArgs;
                }();

                // Make a call to targetFnOp with the collected arguments
                auto msg = std::string("Checking " + targetFuncName + " ...\n");
                Print(msg);

                (void)builder.create<ir::value::LaunchFuncOp>(loc, targetFnOp, targetFnArgs);
                {
                    // Set insertion point past the debug nest
                    mlir::OpBuilder::InsertionGuard guard(builder);
                    builder.setInsertionPointToEnd(&dbgFnOp.body().front());

                    // For each output arg, call its designated utility function to check that the expected values match
                    unsigned utilityFnIndex = 0;
                    for (auto [targetArg, debugArg, usage] : llvm::zip(targetFnArgs, dbgFnOp.getArguments(), targetFunc.GetParameterUsages()))
                    {
                        if (usage == FunctionParameterUsage::inputOutput)
                        {
                            // Expect the number of utility functions to match the number of outputs
                            assert(utilityFnIndex < utilityFunctionNames.size() && "Too few debug utility functions were generated");
                            if (auto utilityFnOp = FindValueFuncOp(moduleOp, utilityFunctionNames[utilityFnIndex++]))
                            {
                                (void)builder.create<ir::value::LaunchFuncOp>(loc, utilityFnOp, mlir::ValueRange{ targetArg, debugArg });
                            }

                            // Set the output arguments of this function so that the caller gets the target result
                            // TODO: This last copy can be avoided if we wrap the default schedule impl within its own ValueFuncOp
                            CopyData(Wrap(targetArg), Wrap(debugArg));
                        }
                    }

                    // Finally, add the terminator
                    assert(dbgFnOp.getNumResults() == 0 && "Nest functions must return no results"); // future work?
                    builder.create<ir::value::ReturnOp>(loc);
                }
            }

            if (targetLaunchFnOp)
            {
                // Replace the original launcher with one that calls the debug wrapper function
                auto newLaunchFnOp = ir::value::CreateRawPointerAPIWrapperFunction(builder, dbgFnOp, targetLaunchFnOp.sym_name());

                // Propagate the base name so that aliases can be created
                if (auto baseName = targetLaunchFnOp->getAttrOfType<mlir::StringAttr>(ir::BaseNameAttrName))
                {
                    newLaunchFnOp->setAttr(ir::BaseNameAttrName, baseName);
                }
                targetLaunchFnOp.erase();
            }
        }
    }
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
                default:
                    return ValueType::Undefined;
                }
            }
        })
        .Case<mlir::IndexType>([](mlir::IndexType idxTy) {
            return ValueType::Index;
        })
        .Case<mlir::FloatType>([](mlir::FloatType fTy) {
            if (fTy.isF32())
                return ValueType::Float;
            else if (fTy.isF64())
                return ValueType::Double;
            else
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
    return GetMLIRContext().GetGPUIndex(GPUIndexType::BlockDim);
}
/*static*/ GPUIndex GPU::BlockId()
{
    return GetMLIRContext().GetGPUIndex(GPUIndexType::BlockId);
}
/*static*/ GPUIndex GPU::GridDim()
{
    return GetMLIRContext().GetGPUIndex(GPUIndexType::GridDim);
}
/*static*/ GPUIndex GPU::ThreadId()
{
    return GetMLIRContext().GetGPUIndex(GPUIndexType::ThreadId);
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
        auto castType = mlir::MemRefType::get(llvm::makeArrayRef(llvm::SmallVector<int64_t, 3>((size_t)rank, -1)), elemTy, {}, memorySpace);

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

mlir::OwningModuleRef GatherModules(const std::string& name, const std::vector<value::MLIRContext*>& contexts, mlir::MLIRContext* context)
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
    std::vector<mlir::OwningModuleRef> owningModuleRefs;
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
