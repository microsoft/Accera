////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/IRUtil.h>
#include <ir/include/intrinsics/AcceraIntrinsicsDialect.h>
#include <ir/include/value/ValueDialect.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Constant.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <transforms/include/util/SnapshotUtilities.h>
#include <value/include/Debugging.h>
#include <value/include/MLIREmitterContext.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/IR/DataLayout.h>
#include <llvm/Support/raw_os_ostream.h>

#include <iostream>

#ifndef _MSC_VER
#include <time.h>
#endif

using namespace mlir;

using namespace accera::ir;
using namespace accera::ir::value;
using namespace accera::transforms;
using namespace accera::transforms::value;

namespace
{
// This is ported from Linux code time.h
//     #define CLOCK_REALTIME   0  // Identifier for system-wide realtime clock.  
//     #define CLOCK_MONOTONIC	1  // Monotonic system-wide clock.
enum class ClockID
{
    ACCERA_CLOCK_REALTIME = 0,
    ACCERA_CLOCK_MONOTONIC = 1,
};

// TODO: Refactor this class and find a better place for this helper class
class LLVMTypeConverterDynMem : public mlir::LLVMTypeConverter
{
public:
    LLVMTypeConverterDynMem(MLIRContext* ctx, const mlir::LowerToLLVMOptions& options) :
        mlir::LLVMTypeConverter(ctx, options)
    {}

    Type convertMemRefToBarePtr(mlir::BaseMemRefType type)
    {
        if (type.isa<mlir::UnrankedMemRefType>())
            // Unranked memref is not supported in the bare pointer calling convention.
            return {};

        Type elementType = convertType(type.getElementType());
        if (!elementType)
            return {};
        return mlir::LLVM::LLVMPointerType::get(elementType, type.getMemorySpaceAsInt());
    }

    Type convertCallingConventionType(Type type)
    {
        auto memrefTy = type.dyn_cast<mlir::BaseMemRefType>();
        if (getOptions().useBarePtrCallConv && memrefTy)
            return convertMemRefToBarePtr(memrefTy);

        return convertType(type);
    }

    LogicalResult barePtrFuncArgTypeConverterDynMem(Type type,
                                                    SmallVectorImpl<Type>& result)
    {
        auto llvmTy = convertCallingConventionType(type);
        if (!llvmTy)
            return mlir::failure();

        result.push_back(llvmTy);
        return mlir::success();
    }

    Type convertFunctionSignature(
        FunctionType funcTy,
        bool isVariadic,
        LLVMTypeConverter::SignatureConversion& result)
    {
        // Select the argument converter depending on the calling convention.
        if (getOptions().useBarePtrCallConv)
        {
            for (auto& en : llvm::enumerate(funcTy.getInputs()))
            {
                Type type = en.value();
                llvm::SmallVector<Type, 8> converted;
                if (failed(barePtrFuncArgTypeConverterDynMem(type, converted)))
                    return {};
                result.addInputs(en.index(), converted);
            }
        }
        else
        {
            for (auto& en : llvm::enumerate(funcTy.getInputs()))
            {
                Type type = en.value();
                llvm::SmallVector<Type, 8> converted;
                if (failed(mlir::structFuncArgTypeConverter(*this, type, converted)))
                    return {};
                result.addInputs(en.index(), converted);
            }
        }

        llvm::SmallVector<Type, 8> argTypes;
        argTypes.reserve(llvm::size(result.getConvertedTypes()));
        for (Type type : result.getConvertedTypes())
            argTypes.push_back(type);

        // If function does not return anything, create the void result type,
        // if it returns on element, convert it, otherwise pack the result types into
        // a struct.
        Type resultType = funcTy.getNumResults() == 0
                              ? mlir::LLVM::LLVMVoidType::get(&getContext())
                              : packFunctionResults(funcTy.getResults());
        if (!resultType)
            return {};
        return mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, isVariadic);
    }
};

static FlatSymbolRefAttr getOrInsertLibraryFunction(PatternRewriter& rewriter,
                                                    std::string libraryFunctionName,
                                                    mlir::Type llvmFnType,
                                                    ModuleOp module,
                                                    LLVM::LLVMDialect* /*llvmDialect*/)
{
    auto* context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(libraryFunctionName))
        return SymbolRefAttr::get(context, libraryFunctionName);

    // Insert the function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), libraryFunctionName, llvmFnType);
    return SymbolRefAttr::get(context, libraryFunctionName);
}

template <typename ConcreteType>
class PrintOpLoweringBase : public OpConversionPattern<ConcreteType>
{
public:
    using OpConversionPattern<ConcreteType>::OpConversionPattern;

    // Return a symbol reference to the printf function, inserting it into the
    // module if necessary.
    static FlatSymbolRefAttr getOrInsertPrintFunction(PatternRewriter& rewriter,
                                                      ModuleOp module,
                                                      LLVM::LLVMDialect* llvmDialect)
    {
        auto* context = module.getContext();
        return getOrInsertLibraryFunction(rewriter, "printf", getPrintfType(context), module, llvmDialect);
    }

    static FlatSymbolRefAttr getOrInsertPrintErrorFunction(PatternRewriter& rewriter,
                                                           ModuleOp module,
                                                           LLVM::LLVMDialect* llvmDialect)
    {
        auto* context = module.getContext();
        return getOrInsertLibraryFunction(rewriter, accera::ir::GetPrintErrorFunctionName(), getPrintfType(context), module, llvmDialect);
    }

    static mlir::Type getPrintfType(mlir::MLIRContext* context)
    {
        // Create a function type for printf, the signature is:
        //   * `i32 (i8*, ...)`
        auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
        return LLVM::LLVMFunctionType::get(getPrintFnReturnType(context), llvmI8PtrTy, /*isVarArg=*/true);
    }

    static mlir::Type getPrintFnReturnType(mlir::MLIRContext* context)
    {
        return IntegerType::get(context, 32);
    }

    // Return a value representing an access into a global string with the given
    // name, creating the string if necessary.
    static mlir::Value getOrCreateGlobalString(Location loc, OpBuilder& builder, StringRef name, StringRef value, ModuleOp module)
    {
        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        auto context = builder.getContext();
        auto i8Ty = IntegerType::get(context, 8);
        if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name)))
        {
            OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto type = LLVM::LLVMArrayType::get(i8Ty, value.size());
            global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name, builder.getStringAttr(value));
        }

        // Get the pointer to the first character in the global string.
        Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
        auto i64Ty = IntegerType::get(context, 64);
        Value cst0 = builder.create<LLVM::ConstantOp>(loc, i64Ty, builder.getIntegerAttr(builder.getIndexType(), 0));
        return builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(i8Ty), globalPtr, ArrayRef<Value>({ cst0, cst0 }));
    }

    static mlir::Value getOrCreateGlobalArray(Location loc, OpBuilder& builder, StringRef name, Type elementType, size_t size, ModuleOp module, LLVM::LLVMDialect* llvmDialect)
    {
        LLVMTypeConverter llvmTypeConverter(builder.getContext());
        auto llvmElementType = llvmTypeConverter.convertType(elementType).dyn_cast<mlir::Type>();

        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name)))
        {
            OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto type = LLVM::LLVMArrayType::get(llvmElementType, size);
            mlir::Attribute valueAttr;
            global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/false, LLVM::Linkage::Internal, name, valueAttr);
        }

        // Get the pointer to the first entry in the global array.
        Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = builder.create<LLVM::ConstantOp>(loc, IntegerType::get(builder.getContext(), 64), builder.getIntegerAttr(builder.getIndexType(), 0));
        return builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(llvmElementType), globalPtr, ArrayRef<Value>({ cst0, cst0 }));
    }
};

class PrintFOpLowering : public PrintOpLoweringBase<PrintFOp>
{
public:
    using PrintOpLoweringBase<PrintFOp>::PrintOpLoweringBase;

    LogicalResult matchAndRewrite(PrintFOp op,
                                  PrintFOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override;
};

template <typename OpTy>
struct ValueLLVMOpConversionPattern : public OpConversionPattern<OpTy>
{
public:
    explicit ValueLLVMOpConversionPattern(LLVMTypeConverter& typeConverter, mlir::MLIRContext* context, PatternBenefit benefit = 1) :
        OpConversionPattern<OpTy>(context, benefit), llvmTypeConverter(typeConverter) {}

protected:
    LLVMTypeConverter& llvmTypeConverter;
};

using ValueCallOp = accera::ir::value::CallOp;
struct CallOpLowering : public ValueLLVMOpConversionPattern<ValueCallOp>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;

    LogicalResult matchAndRewrite(
        ValueCallOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override;
};

struct BitcastOpLowering : public OpConversionPattern<BitcastOp>
{
    using OpConversionPattern<BitcastOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        BitcastOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override;
};

struct GlobalOpToLLVMLowering : public ValueLLVMOpConversionPattern<GlobalOp>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;

    LogicalResult matchAndRewrite(
        GlobalOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override;
};

struct ReferenceGlobalOpLowering : public ValueLLVMOpConversionPattern<ReferenceGlobalOp>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;

    LogicalResult matchAndRewrite(
        ReferenceGlobalOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override;
};

struct CPUEarlyReturnRewritePattern : ValueLLVMOpConversionPattern<EarlyReturnOp>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;
    LogicalResult matchAndRewrite(EarlyReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const final
    {
        if (auto target = util::ResolveExecutionTarget(op); !target || *target != ExecutionTarget::CPU)
        {
            return failure();
        }

        // Get the block the op belongs to
        Block* currentBlock = rewriter.getBlock();
        // Get an iterator pointing to one after the op
        auto position = ++Block::iterator(op);

        // Split the block at this point, so that the early_return op is the last op
        // in the original block and everything after is moved to a new block
        // we don't care about the new block, since we were asked to return early
        // TODO kerha: figure out cleanup semantics
        (void)rewriter.splitBlock(currentBlock, position);

        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

struct GetTimeOpLowering : public ValueLLVMOpConversionPattern<GetTimeOp>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;

    accera::value::TargetDevice deviceInfo;

    GetTimeOpLowering(LLVMTypeConverter& converter, mlir::MLIRContext* context, accera::value::TargetDevice deviceInfo) :
        ValueLLVMOpConversionPattern(converter, context),
        deviceInfo(deviceInfo)
    {}

    LogicalResult matchAndRewrite(
        GetTimeOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override;

    mlir::Value GetTime(ConversionPatternRewriter& rewriter, GetTimeOp& op, ModuleOp& parentModule) const;

    static FlatSymbolRefAttr getOrInsertQueryPerfFrequency(PatternRewriter& rewriter,
                                                           ModuleOp module,
                                                           LLVM::LLVMDialect* llvmDialect)
    {
        auto* context = module.getContext();
        auto llvmFnType = getQueryPerformanceFrequencyFunctionType(context);
        return getOrInsertLibraryFunction(rewriter, "QueryPerformanceFrequency", llvmFnType, module, llvmDialect);
    }

    static FlatSymbolRefAttr getOrInsertQueryPerfCounter(PatternRewriter& rewriter,
                                                         ModuleOp module,
                                                         LLVM::LLVMDialect* llvmDialect)
    {
        auto* context = module.getContext();
        auto llvmFnType = getQueryPerformanceCounterFunctionType(context);
        return getOrInsertLibraryFunction(rewriter, "QueryPerformanceCounter", llvmFnType, module, llvmDialect);
    }

    static FlatSymbolRefAttr getOrInsertClockGetTime(PatternRewriter& rewriter,
                                                     ModuleOp module,
                                                     LLVM::LLVMDialect* llvmDialect,
                                                     size_t numBits)
    {
        auto* context = module.getContext();
        auto llvmFnType = getGetTimeFunctionType(context, numBits);
        return getOrInsertLibraryFunction(rewriter, "clock_gettime", llvmFnType, module, llvmDialect);
    }

    static Type getQueryPerformanceFrequencyFunctionType(mlir::MLIRContext* context)
    {
        // BOOL QueryPerformanceFrequency(LARGE_INTEGER *lpFrequency); // LARGE_INTEGER is a signed 64-bit int
        auto boolTy = IntegerType::get(context, 8);
        auto argTy = LLVM::LLVMPointerType::get(getPerformanceCounterType(context));
        return LLVM::LLVMFunctionType::get(boolTy, { argTy }, /*isVarArg=*/false);
    }

    static Type getQueryPerformanceCounterFunctionType(mlir::MLIRContext* context)
    {
        // BOOL QueryPerformanceCounter(LARGE_INTEGER *lpPerformanceCount); // LARGE_INTEGER is a signed 64-bit int
        auto boolTy = IntegerType::get(context, 8);
        auto argTy = LLVM::LLVMPointerType::get(getPerformanceCounterType(context));
        return LLVM::LLVMFunctionType::get(boolTy, { argTy }, /*isVarArg=*/false);
    }

    static Type getGetTimeFunctionType(mlir::MLIRContext* context, size_t numBits)
    {
        // Create a function type for clock_gettime, the signature is:
        //        int clock_gettime(clockid_t clockid, struct timespec *tp);
        auto returnTy = getIntType(context, numBits);
        auto llvmClockIdTy = getClockIdType(context, numBits);
        auto llvmTimespecTy = getTimeSpecType(context, numBits);
        auto llvmTimespecPtrTy = LLVM::LLVMPointerType::get(llvmTimespecTy);
        return LLVM::LLVMFunctionType::get(returnTy, { llvmClockIdTy, llvmTimespecPtrTy }, /*isVarArg=*/false);
    }

    static Type getPerformanceCounterType(mlir::MLIRContext* context)
    {
        return IntegerType::get(context, 64);
    }

    static Type getClockIdType(mlir::MLIRContext* context, size_t numBits)
    {
        return getIntType(context, numBits);
    }

    static Type getTimeSpecType(mlir::MLIRContext* context, size_t numBits)
    {
        //    struct timespec {
        //        time_t   tv_sec;        /* seconds */
        //        long     tv_nsec;       /* nanoseconds */
        //    };
        auto llvmIntTy = getIntType(context, numBits);
        auto llvmTimespecTy = LLVM::LLVMStructType::getLiteral(context, { llvmIntTy, llvmIntTy }, /* isPacked */ true);
        return llvmTimespecTy;
    }

    static Type getIntType(mlir::MLIRContext* context, size_t numBits)
    {
        auto llvmI32Ty = IntegerType::get(context, 32);
        auto llvmI64Ty = IntegerType::get(context, 64);
        auto llvmIntTy = numBits == 32 ? llvmI32Ty : llvmI64Ty;
        return llvmIntTy;
    }
};

struct RangeOpLowering : public ValueLLVMOpConversionPattern<RangeOp>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;

    LogicalResult matchAndRewrite(
        RangeOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override;
};

// Use a custom lowering pattern instead of using mlir's existing memref::alloc patterns,
// which allows us to add custom allocators in future,
// please keep this class updated with mlir/lib/Conversion/MemRefToLLVM/AllocLikeConversion.cpp.
struct MemrefAllocOpLowering : public ConvertOpToLLVMPattern<memref::AllocOp>
{
    using ConvertToLLVMPattern::createIndexConstant;
    using ConvertToLLVMPattern::getIndexType;
    using ConvertToLLVMPattern::getVoidPtrType;

    MemrefAllocOpLowering(LLVMTypeConverter& converter, mlir::MLIRContext* context) :
        ConvertOpToLLVMPattern(converter)
    {}

    static Value createAligned(ConversionPatternRewriter& rewriter, Location loc, Value input, Value alignment)
    {
        Value one = createIndexAttrConstant(rewriter, loc, alignment.getType(), 1);
        Value bump = rewriter.create<LLVM::SubOp>(loc, alignment, one);
        Value bumped = rewriter.create<LLVM::AddOp>(loc, input, bump);
        Value mod = rewriter.create<LLVM::URemOp>(loc, bumped, alignment);
        return rewriter.create<LLVM::SubOp>(loc, bumped, mod);
    }

    std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter& rewriter, Location loc, Value sizeBytes, Operation* op) const
    {
        // Heap allocations.
        memref::AllocOp allocOp = cast<memref::AllocOp>(op);
        MemRefType memRefType = allocOp.getType();

        Value alignment;
        if (auto alignmentAttr = allocOp.alignment())
        {
            alignment = createIndexConstant(rewriter, loc, *alignmentAttr);
        }
        else if (!memRefType.getElementType().isSignlessIntOrIndexOrFloat())
        {
            alignment = getSizeInBytes(loc, memRefType.getElementType(), rewriter);
        }

        if (alignment)
        {
            // Adjust the allocation size to consider alignment.
            sizeBytes = rewriter.create<LLVM::AddOp>(loc, sizeBytes, alignment);
        }

        // Allocate the underlying buffer and store a pointer to it in the MemRef descriptor.
        Type elementPtrType = this->getElementPtrType(memRefType);
        auto allocFuncOp = LLVM::lookupOrCreateMallocFn(
            allocOp->getParentOfType<ModuleOp>(), getIndexType());
        auto results = createLLVMCall(rewriter, loc, allocFuncOp, { sizeBytes }, getVoidPtrType());
        Value allocatedPtr = rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, results[0]);

        Value alignedPtr = allocatedPtr;
        if (alignment)
        {
            // Compute the aligned type pointer.
            Value allocatedInt = rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), allocatedPtr);
            Value alignmentInt = createAligned(rewriter, loc, allocatedInt, alignment);
            alignedPtr = rewriter.create<LLVM::IntToPtrOp>(loc, elementPtrType, alignmentInt);
        }

        return std::make_tuple(allocatedPtr, alignedPtr);
    }

    LogicalResult match(memref::AllocOp op) const
    {
        if (!op)
            return failure();

        if (auto target = util::ResolveExecutionTarget(op); !target || *target != ExecutionTarget::CPU)
        {
            return failure();
        }

        // TODO: do we need this check?
        MemRefType memRefType = op.getResult().getType().cast<MemRefType>();
        if (!isConvertibleAndHasIdentityMaps(memRefType))
            return failure();

        return success();
    }

    void rewrite(memref::AllocOp allocop, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const
    {
        std::vector<Value> operandsVector;
        mlir::ValueRange operandsRange = adaptor.getOperands();
        for (unsigned i = 0; i < operandsRange.size(); i++)
            operandsVector.push_back(operandsRange[i]);
        ArrayRef<Value> operands(operandsVector);
        Operation* op = allocop.getOperation();

        auto loc = op->getLoc();

        SmallVector<Value, 4> sizes;
        SmallVector<Value, 4> strides;
        Value sizeBytes;
        MemRefType memRefType = allocop.getResult().getType().cast<MemRefType>();
        this->getMemRefDescriptorSizes(loc, memRefType, operands, rewriter, sizes, strides, sizeBytes);

        // Allocate the underlying buffer.
        Value allocatedPtr;
        Value alignedPtr;
        std::tie(allocatedPtr, alignedPtr) = this->allocateBuffer(rewriter, loc, sizeBytes, op);

        // Create the MemRef descriptor.
        auto memRefDescriptor = this->createMemRefDescriptor(
            loc, memRefType, allocatedPtr, alignedPtr, sizes, strides, rewriter);

        // Return the final value of the descriptor.
        rewriter.replaceOp(op, { memRefDescriptor });
    }
};

struct ValueMemRefCastOpLowering : public ValueLLVMOpConversionPattern<MemRefCastOp>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;

    mlir::Value GetLLVMCompatibleSource(mlir::Value src) const
    {
        if (LLVM::isCompatibleType(src.getType()))
        {
            return src;
        }
        else
        {
            // Check if the source is an unrealized conversion cast from a compatible type
            if (auto castOp = src.getDefiningOp<mlir::UnrealizedConversionCastOp>())
            {
                if (castOp.inputs().size() == 1)
                {
                    return GetLLVMCompatibleSource(castOp.inputs()[0]);
                }
            }
            return nullptr;
        }
    }

    LogicalResult matchAndRewrite(
        MemRefCastOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto src = adaptor.source();
        auto srcType = op.getViewSource().getType().cast<MemRefType>();
        auto llvmCompatibleSrc = GetLLVMCompatibleSource(src);
        if (!llvmCompatibleSrc)
            return failure();

        auto dstType = op.getType();
        auto targetMemSpace = dstType.getMemorySpaceAsInt();
        auto targetElementTy = llvmTypeConverter.convertType(dstType.getElementType());
        auto targetStructType = llvmTypeConverter.convertType(dstType).dyn_cast_or_null<LLVM::LLVMStructType>();
        if (!targetStructType)
            return failure();

        auto loc = op.getLoc();

        MemRefDescriptor srcMemrefDesc(llvmCompatibleSrc);
        auto targetMemrefDesc = MemRefDescriptor::undef(rewriter, loc, targetStructType);

        targetMemrefDesc.setAllocatedPtr(
            rewriter,
            loc,
            rewriter.create<LLVM::BitcastOp>(
                loc,
                LLVM::LLVMPointerType::get(
                    targetElementTy,
                    targetMemSpace),
                srcMemrefDesc.allocatedPtr(
                    rewriter,
                    loc)));

        targetMemrefDesc.setAlignedPtr(
            rewriter,
            loc,
            rewriter.create<LLVM::BitcastOp>(
                loc,
                LLVM::LLVMPointerType::get(
                    targetElementTy,
                    targetMemSpace),
                srcMemrefDesc.alignedPtr(
                    rewriter,
                    loc)));

        if (dstType.hasStaticShape())
        {
            targetMemrefDesc.setConstantSize(rewriter, loc, 0, dstType.getDimSize(0));
        }
        else
        {
            auto llvmIndexType = llvmTypeConverter.convertType(rewriter.getIndexType());
            mlir::Value size = rewriter.create<LLVM::ConstantOp>(
                loc, llvmIndexType, rewriter.getI64IntegerAttr(srcType.getElementTypeBitWidth() / dstType.getElementTypeBitWidth()));
            for (auto s : llvm::enumerate(srcType.getShape()))
            {
                if (s.value() != mlir::ShapedType::kDynamicSize)
                {
                    size = rewriter.create<LLVM::MulOp>(loc, size, rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getI64IntegerAttr(s.value())));
                }
                else
                {
                    mlir::Value dimSize = rewriter.create<memref::DimOp>(loc, op.getViewSource(), s.index());
                    auto castOp = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, llvmIndexType, dimSize);
                    mlir::Value intDimSize = castOp.getResult(0);
                    size = rewriter.create<LLVM::MulOp>(loc, size, intDimSize);
                }
            }
            targetMemrefDesc.setSize(rewriter, loc, 0, size);
        }
        targetMemrefDesc.setOffset(rewriter, loc, srcMemrefDesc.offset(rewriter, loc));

        // Unit stride (assuming affine map attached to the cast op will figure out correct byte indexing)
        // The cast op at the Value layer can handle dilations
        targetMemrefDesc.setConstantStride(rewriter, loc, 0, 1);

        rewriter.replaceOp(op, { targetMemrefDesc });

        return success();
    }
};

// TODO : de-dupe these lowerings, all 2-arg-1-result vector intrinsics appear to have the same lowering
struct VpmaddwdOpLowering : public ValueLLVMOpConversionPattern<vpmaddwd>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;

    LogicalResult matchAndRewrite(
        vpmaddwd op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        LLVMTypeConverter llvmTypeConverter(rewriter.getContext());
        auto outputVecType = op.getType().cast<mlir::VectorType>();
        auto outputVecLLVMType = llvmTypeConverter.convertType(outputVecType);
        [[maybe_unused]] auto outputRank = outputVecType.getRank();
        assert(outputRank == 1 && "Vpmaddwd op should have a 1-D result");
        auto elementCount = outputVecType.getShape()[0];
        auto avx512Support = util::ModuleSupportsTargetDeviceFeature(op, "avx512");
        if (elementCount == 8)
        {
            rewriter.replaceOpWithNewOp<intrinsics::VpmaddwdOp>(op, outputVecLLVMType, op.lhs(), op.rhs());
        }
        else if (elementCount == 16 && avx512Support)
        {
            rewriter.replaceOpWithNewOp<intrinsics::VpmaddwdAVX512Op>(op, outputVecLLVMType, op.lhs(), op.rhs());
        }
        else
        {
            assert(false && "Bad vector size for given target's vpmaddwd support");
        }
        return success();
    }
};

struct VmaxpsOpLowering : public ValueLLVMOpConversionPattern<vmaxps>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;

    LogicalResult matchAndRewrite(
        vmaxps op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        LLVMTypeConverter llvmTypeConverter(rewriter.getContext());
        auto outputVecType = op.getType().cast<mlir::VectorType>();
        auto outputVecLLVMType = llvmTypeConverter.convertType(outputVecType);
        rewriter.replaceOpWithNewOp<intrinsics::VmaxpsOp>(op, outputVecLLVMType, op.lhs(), op.rhs());
        return success();
    }
};

struct VminpsOpLowering : public ValueLLVMOpConversionPattern<vminps>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;

    LogicalResult matchAndRewrite(
        vminps op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        LLVMTypeConverter llvmTypeConverter(rewriter.getContext());
        auto outputVecType = op.getType().cast<mlir::VectorType>();
        auto outputVecLLVMType = llvmTypeConverter.convertType(outputVecType);
        rewriter.replaceOpWithNewOp<intrinsics::VminpsOp>(op, outputVecLLVMType, op.lhs(), op.rhs());
        return success();
    }
};

struct RoundOpLowering : public ValueLLVMOpConversionPattern<RoundOp>
{
    using ValueLLVMOpConversionPattern::ValueLLVMOpConversionPattern;

    LogicalResult matchAndRewrite(
        RoundOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        LLVMTypeConverter llvmTypeConverter(rewriter.getContext());
        auto outputType = llvmTypeConverter.convertType(op.getType());

        auto inputType = op.val().getType();
        if (inputType.isa<mlir::VectorType>())
        {
            rewriter.replaceOpWithNewOp<intrinsics::RoundF32VecAVX2>(op, outputType, op.val());
        }
        else
        {
            mlir::Value roundedFPVal = rewriter.create<intrinsics::RoundEvenOp>(op.getLoc(), op.val());

            // Create arithmetic dialect cast ops with the expectation that other arithmetic dialect ops are getting lowered as part of this pass
            auto signlessOutputType = util::ToSignlessMLIRType(rewriter, op.getType());
            mlir::Value roundedSIVal = rewriter.create<mlir::arith::FPToSIOp>(op.getLoc(), signlessOutputType, roundedFPVal);
            rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, op.getType(), roundedSIVal);
        }
        return success();
    }
};

struct ValueToLLVMLoweringPass : public ConvertValueToLLVMBase<ValueToLLVMLoweringPass>
{
    ValueToLLVMLoweringPass(bool useBarePtrCallConv, bool emitCWrappers, unsigned indexBitwidth, bool useAlignedAlloc, llvm::DataLayout dataLayout, accera::value::TargetDevice deviceInfo = {}, const IntraPassSnapshotOptions& snapshotteroptions = {}) :
        _intrapassSnapshotter(snapshotteroptions),
        deviceInfo(deviceInfo)
    {
        this->useBarePtrCallConv = useBarePtrCallConv;
        this->emitCWrappers = emitCWrappers;
        this->indexBitwidth = indexBitwidth;
        // TODO: move to mlir::LowerToLLVMOptions::AllocLowering
        this->useAlignedAlloc = useAlignedAlloc;
        this->dataLayout = dataLayout.getStringRepresentation();
    }

    void runOnModule() final;

private:
    IRSnapshotter _intrapassSnapshotter;
    accera::value::TargetDevice deviceInfo;
};

// Creates a constant Op producing a value of `resultType` from an index-typed
// integer attribute.
Value createIndexAttrConstant(OpBuilder& builder, Location loc, Type resultType, int64_t value)
{
    return builder.create<LLVM::ConstantOp>(
        loc, resultType, builder.getIntegerAttr(builder.getIndexType(), value));
}

// Create an LLVM IR pseudo-operation defining the given index constant.
[[maybe_unused]] Value createIndexConstant(LLVMTypeConverter& converter, ConversionPatternRewriter& builder, Location loc, uint64_t value)
{
    return createIndexAttrConstant(builder, loc, converter.convertType(builder.getIndexType()), value);
}

struct LLVMCallFixupPattern : OpRewritePattern<LLVM::CallOp>
{
    using OpRewritePattern::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const final
    {
        auto optionalCallee = op.getCallee();
        if (!optionalCallee) return failure();

        auto callee = mlir::StringAttr::get(op.getContext(), *optionalCallee);
        auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(op->getParentOfType<ModuleOp>(), callee);
        if (!funcOp) return failure();

        auto numCallArgs = op.getNumOperands();
        auto numFuncArgs = funcOp.getNumArguments();

        return success((numFuncArgs == 0 && numCallArgs == 0) ||
                       (numCallArgs / numFuncArgs == 5));
    }

    void rewrite(LLVM::CallOp callOp, mlir::PatternRewriter& rewriter) const final
    {
        rewriter.updateRootInPlace(callOp, [&] {
            mlir::Operation* op = callOp;
            llvm::SmallVector<mlir::Value, 4> newOperands;
            for (unsigned idx = 1, e = op->getNumOperands(); idx < e; idx += 5)
            {
                newOperands.push_back(op->getOperand(idx));
            }
            op->setOperands(newOperands);
        });
    }
};

struct RawPointerAPIFnConversion : public ConvertOpToLLVMPattern<FuncOp>
{
    using ConvertOpToLLVMPattern<FuncOp>::ConvertOpToLLVMPattern;
    using ConvertToLLVMPattern::getIndexType;

    // cf mlir\lib\Conversion\StandardToLLVM\StandardToLLVM.cpp
    /// Only retain those attributes that are not constructed by
    /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
    /// attributes.
    static void filterFuncAttributes(ArrayRef<NamedAttribute> attrs,
                                     bool filterArgAttrs,
                                     SmallVectorImpl<NamedAttribute>& result)
    {
        for (const auto& attr : attrs)
        {
            if (attr.getName() == SymbolTable::getSymbolAttrName() ||
                attr.getName() == FunctionOpInterface::getTypeAttrName() || attr.getName() == "std.varargs" ||
                (filterArgAttrs && attr.getName() == FunctionOpInterface::getArgDictAttrName()))
                continue;
            result.push_back(attr);
        }
    }

    // cf FuncOpConversionBase in mlir\lib\Conversion\StandardToLLVM\StandardToLLVM.cpp

    // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
    // to this legalization pattern.
    LLVM::LLVMFuncOp convertFuncOpToLLVMFuncOp(FuncOp funcOp, ConversionPatternRewriter& rewriter) const
    {
        // Convert the original function arguments. They are converted using the
        // LLVMTypeConverter provided to this legalization pattern.
        bool isDynamicMem = false;
        auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("std.varargs");
        TypeConverter::SignatureConversion result(funcOp.getNumArguments());
        TypeConverter::SignatureConversion resultDynMem(funcOp.getNumArguments());
        auto llvmType = getTypeConverter()->convertFunctionSignature(
            funcOp.getType(), varargsAttr && varargsAttr.getValue(), result);
        if (!llvmType)
        {
            isDynamicMem = true;
            LLVMTypeConverterDynMem llvmTypeConverterDynMem(&getTypeConverter()->getContext(), getTypeConverter()->getOptions());
            llvmType = llvmTypeConverterDynMem.convertFunctionSignature(funcOp.getType(), false, resultDynMem);
        }

        // Propagate argument attributes to all converted arguments obtained after
        // converting a given original argument.
        SmallVector<NamedAttribute, 4> attributes;
        filterFuncAttributes(funcOp->getAttrs(), /*filterArgAttrs=*/true, attributes);
        if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs())
        {
            SmallVector<Attribute, 4> newArgAttrs(
                llvmType.cast<LLVM::LLVMFunctionType>().getNumParams());
            for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i)
            {
                auto mapping = isDynamicMem ? resultDynMem.getInputMapping(i) : result.getInputMapping(i);
                assert(mapping.hasValue() &&
                       "unexpected deletion of function argument");
                for (size_t j = 0; j < mapping->size; ++j)
                    newArgAttrs[mapping->inputNo + j] = argAttrDicts[i];
            }
            attributes.push_back(
                rewriter.getNamedAttr(FunctionOpInterface::getArgDictAttrName(),
                                      rewriter.getArrayAttr(newArgAttrs)));
        }

        // Create an LLVM function, use external linkage by default until MLIR
        // functions have linkage.
        auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
            funcOp.getLoc(), funcOp.getName(), llvmType, LLVM::Linkage::External, /*dsoLocal*/ false, attributes);

        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
        auto beforeConversion = newFuncOp.getArguments().vec();

        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter, isDynamicMem ? &resultDynMem : &result)))
            return nullptr;

        return newFuncOp;
    }

    // cf BarePtrFuncOpConversion in mlir\lib\Conversion\StandardToLLVM\StandardToLLVM.cpp
    LogicalResult matchAndRewrite(FuncOp funcOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        if (!funcOp->getAttr(RawPointerAPIAttrName))
        {
            // Only match FuncOps with the raw pointer API attribute
            return failure();
        }

        auto llvmIndexTy = getIndexType();

        // Store the type of memref-typed arguments before the conversion so that we
        // can promote them to MemRef descriptor at the beginning of the function.
        SmallVector<Type, 8> oldArgTypes =
            llvm::to_vector<8>(funcOp.getType().getInputs());

        auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
        if (!newFuncOp)
            return failure();
        if (newFuncOp.getBody().empty())
        {
            rewriter.eraseOp(funcOp);
            return success();
        }

        // Promote bare pointers from memref arguments to memref descriptors at the
        // beginning of the function so that all the memrefs in the function have a
        // uniform representation.
        Block* entryBlock = &newFuncOp.getBody().front();
        auto blockArgs = entryBlock->getArguments();
        assert(blockArgs.size() == oldArgTypes.size() &&
               "The number of arguments and types doesn't match");

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        std::vector<mlir::Type> oldArgTypesVec(oldArgTypes.begin(), oldArgTypes.end());
        std::vector<std::vector<int64_t>> dynArgSizeReferences = util::ParseDynamicArgSizeReferences(funcOp, oldArgTypesVec);

        for (auto [blockArg, dynArgSizeRefs, argTy] : llvm::zip(blockArgs, dynArgSizeReferences, oldArgTypes))
        {
            // Unranked memrefs are not supported in the bare pointer calling
            // convention. We should have bailed out before in the presence of
            // unranked memrefs.
            assert(!argTy.isa<UnrankedMemRefType>() && "Unranked memref is not supported");
            auto memrefTy = argTy.dyn_cast<MemRefType>();
            if (!memrefTy)
                continue;

            // Note: this diverges from the MLIR main branch implementation and avoids creating
            //       and UndefOp with a MemRef type as the unrealized cast conversion ops appear
            //       to have a bug where they do not get fully converted for those ops.
            //       Moreover, the MLIR main branch version of this claims that a placeholder undef
            //       op is required to avoid replaceUsesOfBlockArgument() causing the ops that fill out
            //       the MemRefDescriptor to themselves be replaced, however replaceUsesOfBlockArgument()
            //       already accounts for this type of scenario and doesn't perform the replacement on any
            //       ops that preceed the new op that is the old arg is being replaced with.
            Location loc = funcOp.getLoc();

            if (memrefTy.getNumDynamicDims() > 0)
            {
                int64_t offset;
                SmallVector<int64_t, 4> strides;
                [[maybe_unused]] auto res = getStridesAndOffset(memrefTy, strides, offset);
                assert(succeeded(res));

                auto convertedType = getTypeConverter()->convertType(memrefTy);
                auto descr = MemRefDescriptor::undef(rewriter, loc, convertedType);

                descr.setAllocatedPtr(rewriter, loc, blockArg);
                descr.setAlignedPtr(rewriter, loc, blockArg);
                descr.setConstantOffset(rewriter, loc, offset);

                // Fill in sizes
                for (unsigned i = 0, e = memrefTy.getRank(); i != e; ++i)
                {
                    auto dimSize = memrefTy.getDimSize(i);
                    if (dimSize == mlir::ShapedType::kDynamicSize)
                    {
                        auto dimSizeArgIdx = dynArgSizeRefs[i];
                        assert(dimSizeArgIdx != -1 /* sentinel value for static dimension in this context */);
                        auto dimSizeValue = blockArgs[dimSizeArgIdx];
                        descr.setSize(rewriter, loc, i, dimSizeValue);
                    }
                    else
                    {
                        descr.setConstantSize(rewriter, loc, i, dimSize);
                    }
                }

                // Now that we have the sizes, fill in strides as a function of the possibly-dynamic dim sizes
                // Currently assumes a dense FIRST_MAJOR ordering
                //      to get a different ordering we need to detect the
                //      logical->physical dimension permutation from the memref type
                // Examples: { N } has strides {1}
                //           { M x N } has strides {N, 1}
                //           { M x N x K } has strides {N*K, K, 1}
                std::vector<unsigned> minorToMajorDimOrder(memrefTy.getRank(), 0);
                std::iota(minorToMajorDimOrder.begin(), minorToMajorDimOrder.end(), 0); // Now minorToMajorDimOrder contains a last major order = [0, 1, 2, 3, ...]
                std::reverse(minorToMajorDimOrder.begin(), minorToMajorDimOrder.end()); // Now minorToMajorDimOrder contains a first major order = [..., 3, 2, 1, 0]

                mlir::Value strideCumulativeProduct = createIndexAttrConstant(rewriter, loc, llvmIndexTy, 1);
                for (const auto& dimIdx : minorToMajorDimOrder)
                {
                    mlir::Value dimSizeVal = descr.size(rewriter, loc, dimIdx);
                    descr.setStride(rewriter, loc, dimIdx, strideCumulativeProduct);
                    strideCumulativeProduct = rewriter.create<LLVM::MulOp>(loc, llvmIndexTy, strideCumulativeProduct, dimSizeVal);
                }

                rewriter.replaceUsesOfBlockArgument(blockArg, descr);
            }
            else
            {
                Value desc = MemRefDescriptor::fromStaticShape(
                    rewriter, loc, *getTypeConverter(), memrefTy, blockArg);
                rewriter.replaceUsesOfBlockArgument(blockArg, desc);
            }
        }

        rewriter.eraseOp(funcOp);
        return success();
    }
};

// Converts mlir::CallOps that are inside of a RawPointerAPI function and call into a non-raw-pointer-API function
struct RawPointerAPICallOpConversion : public mlir::ConvertOpToLLVMPattern<mlir::CallOp>
{
    using ConvertOpToLLVMPattern<mlir::CallOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(mlir::CallOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        // Only match if this mlir::CallOp is inside of an LLVM::FuncOp with the RawPointerAPI attribute and is calling
        // a function without the RawPointerAPI attribute
        auto callOp = cast<mlir::CallOp>(op);
        auto parentFuncOp = callOp->getParentOfType<LLVM::LLVMFuncOp>();
        if (!parentFuncOp || !parentFuncOp->getAttr(RawPointerAPIAttrName))
        {
            return failure();
        }
        auto parentModule = callOp->getParentOfType<mlir::ModuleOp>();
        auto callee = parentModule.lookupSymbol(callOp.getCallee());
        if (!callee || callee->getAttr(RawPointerAPIAttrName))
        {
            return failure();
        }

        // cf CallOpInterfaceLowering in mlir\lib\Conversion\StandardToLLVM\StandardToLLVM.cpp

        // Pack the result types into a struct.
        Type packedResult = nullptr;
        unsigned numResults = callOp.getNumResults();
        auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

        if (numResults != 0)
        {
            packedResult = getTypeConverter()->packFunctionResults(op->getResultTypes());
            if (!packedResult)
                return failure();
        }

        auto promoted = getTypeConverter()->promoteOperands(
            op->getLoc(), /*opOperands=*/op->getOperands(), adaptor.getOperands(), rewriter);
        auto newOp = rewriter.create<LLVM::CallOp>(
            callOp.getLoc(), packedResult ? TypeRange(packedResult) : TypeRange(), promoted, callOp->getAttrs());

        SmallVector<Value, 4> results;
        if (numResults < 2)
        {
            // If < 2 results, packing did not do anything and we can just return.
            results.append(newOp.result_begin(), newOp.result_end());
        }
        else
        {
            // Otherwise, it had been converted to an operation producing a structure.
            // Extract individual results from the structure and return them as list.
            results.reserve(numResults);
            for (unsigned i = 0; i < numResults; ++i)
            {
                auto type = getTypeConverter()->convertType(op->getResult(i).getType());
                results.push_back(rewriter.create<LLVM::ExtractValueOp>(
                    op->getLoc(), type, newOp.getOperation()->getResult(0), rewriter.getI64ArrayAttr(i)));
            }
        }

        if (this->getTypeConverter()->getOptions().useBarePtrCallConv)
        {
            // For the bare-ptr calling convention, promote memref results to
            // descriptors.
            assert(results.size() == resultTypes.size() &&
                   "The number of arguments and types doesn't match");
            this->getTypeConverter()->promoteBarePtrsToDescriptors(
                rewriter, callOp.getLoc(), resultTypes, results);
        }
        else if (failed(copyUnrankedDescriptors(rewriter, callOp.getLoc(), resultTypes, results, /*toDynamic=*/false)))
        {
            return failure();
        }

        rewriter.replaceOp(op, results);
        return success();
    }
};

struct RawPointerAPIUnusedUndefRemoval : public OpRewritePattern<LLVM::UndefOp>
{
    RawPointerAPIUnusedUndefRemoval(MLIRContext* context) :
        OpRewritePattern(context, 100)
    {}

    LogicalResult match(LLVM::UndefOp op) const final
    {
        auto isMemref = op.getRes().getType().isa<MemRefType>();
        auto hasNoUses = op->use_empty();
        return success(isMemref && hasNoUses);
    }

    void rewrite(LLVM::UndefOp op, mlir::PatternRewriter& rewriter) const final
    {
        rewriter.eraseOp(op);
    }
};

static OpFoldResult getExpandedDimSize(
    OpBuilder &builder, 
    Location loc, 
    Type &llvmIndexType,
    int64_t outDimIndex, ArrayRef<int64_t> outStaticShape,
    MemRefDescriptor &inDesc,
    ArrayRef<int64_t> inStaticShape,
    memref::ExpandShapeOp& reshapeOp,
    DenseMap<int64_t, int64_t> &outDimToInDimMap) 
{
    int64_t outDimSize = outStaticShape[outDimIndex];
    if (!ShapedType::isDynamic(outDimSize))
    {
        return builder.getIndexAttr(outDimSize);
    }

    // Calculate the multiplication of all the out dim sizes except the current dim.
    int64_t inDimIndex = outDimToInDimMap[outDimIndex];
    int64_t otherDimSizesMul = 1;

    auto reassocation = reshapeOp.getReassociationIndices();
    auto blockArgs = builder.getBlock()->getArguments();
    auto splitSizeAttr = reshapeOp->getAttr(GetSplitSizeAttrName()).cast<mlir::IntegerAttr>().getInt();
    mlir::Value splitSize = blockArgs[splitSizeAttr];      

    bool isSplitDim = false;
    // inDimIndex is the dimension to be split
    if (reassocation[inDimIndex].size() > 1) 
    {
        if (outDimIndex == reassocation[inDimIndex][1])
        {
            return splitSize;
        }

        if (outDimIndex == reassocation[inDimIndex][0])
        {
            isSplitDim = true;
        }
    }
   
    for (auto otherDimIndex : reassocation[inDimIndex]) 
    {
        if (otherDimIndex == static_cast<unsigned>(outDimIndex))
        {
            continue;
        }
        otherDimSizesMul *= outStaticShape[otherDimIndex];
    }

    // outDimSize = inDimSize / otherOutDimSizesMul
    int64_t inDimSize = inStaticShape[inDimIndex];
    Value inDimSizeDynamic =
        ShapedType::isDynamic(inDimSize)
            ? inDesc.size(builder, loc, inDimIndex)
            : builder.create<LLVM::ConstantOp>(loc, llvmIndexType, builder.getIndexAttr(inDimSize));

  Value outDimSizeDynamic = builder.create<LLVM::SDivOp>(
      loc, 
      inDimSizeDynamic,
      ShapedType::isDynamic(otherDimSizesMul) && isSplitDim
        ? splitSize
        : builder.create<LLVM::ConstantOp>(loc, llvmIndexType, builder.getIndexAttr(otherDimSizesMul)));

    return outDimSizeDynamic;
}

// Compute a map that for a given dimension of the expanded type gives the
// dimension in the collapsed type it maps to. Essentially its the inverse of the `reassocation` maps.
static DenseMap<int64_t, int64_t> getExpandedDimToOriginalDimMap(ArrayRef<ReassociationIndices> reassociation) 
{
    llvm::DenseMap<int64_t, int64_t> dimMap;
    for (auto &dimArray : enumerate(reassociation)) 
    {
        for (auto dim : dimArray.value())
        {
            dimMap[dim] = dimArray.index();
        }
    }
    return dimMap;
}

static SmallVector<OpFoldResult, 4> getExpandedShape(
    OpBuilder &builder, 
    Location loc, 
    Type &llvmIndexType,
    memref::ExpandShapeOp& reshapeOp,
    ArrayRef<int64_t> inStaticShape,
    MemRefDescriptor &inDesc,
    ArrayRef<int64_t> outStaticShape) 
{
    DenseMap<int64_t, int64_t> outDimToInDimMap = getExpandedDimToOriginalDimMap(reshapeOp.getReassociationIndices());
    return llvm::to_vector<4>(llvm::map_range(
        llvm::seq<int64_t>(0, outStaticShape.size()), [&](int64_t outDimIndex) {
            return getExpandedDimSize(
                builder, 
                loc, 
                llvmIndexType, 
                outDimIndex, 
                outStaticShape, 
                inDesc,
                inStaticShape,
                reshapeOp, 
                outDimToInDimMap);
    }));
}

/// Helper function to convert a vector of `OpFoldResult`s into a vector of `Value`s.
static SmallVector<Value> getResultAsValues(
    OpBuilder &builder, 
    Location loc,
    Type &llvmIndexType,
    ArrayRef<OpFoldResult> valueOrAttrVec) 
{
    return llvm::to_vector<4>(
        llvm::map_range(valueOrAttrVec, [&](OpFoldResult value) -> Value {
        if (auto attr = value.dyn_cast<Attribute>())
        {
            return builder.create<LLVM::ConstantOp>(loc, llvmIndexType, attr);
        }
        return value.get<Value>();
    }));
}

static SmallVector<Value> getDynamicExpandedShape(
    OpBuilder &builder, 
    Location loc, 
    Type &llvmIndexType,
    memref::ExpandShapeOp& reshapeOp,
    ArrayRef<int64_t> inStaticShape, 
    MemRefDescriptor &inDesc,
    ArrayRef<int64_t> outStaticShape) 
{
    return getResultAsValues(
        builder, 
        loc, 
        llvmIndexType,
        getExpandedShape(
            builder, 
            loc, 
            llvmIndexType,
            reshapeOp, 
            inStaticShape,
            inDesc,
            outStaticShape));
}

bool isStrideOrOffsetStatic(int64_t strideOrOffset) 
{
    return !ShapedType::isDynamicStrideOrOffset(strideOrOffset);
}

struct ExpandShapeOpLowering : public ConvertOpToLLVMPattern<memref::ExpandShapeOp> 
{
public:
    using ConvertOpToLLVMPattern<memref::ExpandShapeOp>::ConvertOpToLLVMPattern;
    using ReshapeOpAdaptor = typename memref::ExpandShapeOp::Adaptor;

    LogicalResult matchAndRewrite(
        memref::ExpandShapeOp reshapeOp, 
        ReshapeOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override 
    {
        MemRefType dstType = reshapeOp.getResultType();
        MemRefType srcType = reshapeOp.getSrcType();

        int64_t offset;
        SmallVector<int64_t, 4> strides;
        if (failed(getStridesAndOffset(dstType, strides, offset))) 
        {
            return rewriter.notifyMatchFailure(reshapeOp, "failed to get stride and offset exprs");
        }
        
        MemRefDescriptor srcDesc(adaptor.src());
        Location loc = reshapeOp->getLoc();
        auto dstDesc = MemRefDescriptor::undef(rewriter, loc, this->typeConverter->convertType(dstType));

        dstDesc.setAllocatedPtr(rewriter, loc, srcDesc.allocatedPtr(rewriter, loc));
        dstDesc.setAlignedPtr(rewriter, loc, srcDesc.alignedPtr(rewriter, loc));
        dstDesc.setOffset(rewriter, loc, srcDesc.offset(rewriter, loc));

        ArrayRef<int64_t> srcStaticShape = srcType.getShape();
        ArrayRef<int64_t> dstStaticShape = dstType.getShape();
        Type llvmIndexType = this->typeConverter->convertType(rewriter.getIndexType());

        SmallVector<Value> dstShape = getDynamicExpandedShape(
            rewriter, 
            loc, 
            llvmIndexType, 
            reshapeOp,
            srcStaticShape, 
            srcDesc, 
            dstStaticShape);

        for (auto &shape : llvm::enumerate(dstShape))
        {
            dstDesc.setSize(rewriter, loc, shape.index(), shape.value());
        }

        if (llvm::all_of(strides, isStrideOrOffsetStatic)) 
        {
            for (auto &stride : llvm::enumerate(strides))
            {
                dstDesc.setConstantStride(rewriter, loc, stride.index(), stride.value());
            }
        }
        else if (srcType.getLayout().isIdentity() && dstType.getLayout().isIdentity()) 
        {
            Value stride = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIndexAttr(1));
            for (auto dimIndex : llvm::reverse(llvm::seq<int64_t>(0, dstShape.size()))) 
            {
                dstDesc.setStride(rewriter, loc, dimIndex, stride);
                stride = rewriter.create<LLVM::MulOp>(loc, dstShape[dimIndex], stride);
            }
        } 
        else 
        {
            // There could be mixed static/dynamic strides. For simplicity, we
            // recompute all strides if there is at least one dynamic stride.
            // See comments for computeExpandedLayoutMap in llvm source code 
            // for details on how the strides are calculated.
            for (auto &dimArray : llvm::enumerate(reshapeOp.getReassociationIndices())) 
            {
                auto currentStrideToExpand = srcDesc.stride(rewriter, loc, dimArray.index());
                for (auto dstIndex : llvm::reverse(dimArray.value())) 
                {
                    dstDesc.setStride(rewriter, loc, dstIndex, currentStrideToExpand);
                    Value size = dstDesc.size(rewriter, loc, dstIndex);
                    currentStrideToExpand = rewriter.create<LLVM::MulOp>(loc, size, currentStrideToExpand);
                }
            }

        }
        rewriter.replaceOp(reshapeOp, {dstDesc});
        return success();
    }
};

} // namespace

using namespace accera::transforms::value;

[[maybe_unused]] static Type getPointerIndexType(LLVMTypeConverter& typeConverter)
{
    return IntegerType::get(&typeConverter.getContext(), typeConverter.getPointerBitwidth());
}

LogicalResult GlobalOpToLLVMLowering::matchAndRewrite(
    GlobalOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const
{
    auto type = op.getType();
    assert(type && type.hasStaticShape() && "unexpected type");

    uint64_t numElements = type.getNumElements();

    auto elementType = llvmTypeConverter.convertType(type.getElementType())
                           .cast<Type>();
    LLVM::LLVMArrayType arrayType;

    if (auto denseElementsAttr = op.valueAttr().dyn_cast_or_null<DenseElementsAttr>();
        op.constant() && denseElementsAttr)
    {
        // For tensor / vector constants, the llvm type needs to be nested arrays matching the rank of the constant buffer
        auto shape = denseElementsAttr.getType().getShape().vec();
        assert(!shape.empty());
        arrayType = LLVM::LLVMArrayType::get(elementType, shape.back());
        for (size_t idx = 1; idx < shape.size(); ++idx)
        {
            // Walk from the innermost part of the shape outwards
            size_t currentIdx = shape.size() - idx;
            arrayType = LLVM::LLVMArrayType::get(arrayType, shape[currentIdx]);
        }
    }
    else
    {
        arrayType = LLVM::LLVMArrayType::get(elementType, numElements);
    }

    {
        OpBuilder::InsertionGuard guard(rewriter);

        rewriter.create<LLVM::GlobalOp>(
            op.getLoc(),
            arrayType,
            op.constant(),
            op.external() ? LLVM::Linkage::External : LLVM::Linkage::Internal,
            op.sym_name(),
            op.valueAttr());
    }
    rewriter.eraseOp(op);

    return success();
}

LogicalResult ReferenceGlobalOpLowering::matchAndRewrite(
    ReferenceGlobalOp op,
    OpAdaptor,
    ConversionPatternRewriter& rewriter) const
{
    auto parentValueFuncOp = op->getParentOfType<ValueFuncOp>();
    auto parentFuncOp = op->getParentOfType<mlir::FuncOp>();
    auto parentLLVMFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!parentValueFuncOp && !parentFuncOp && !parentLLVMFuncOp)
    {
        // Global constant buffers are created with a module-level ReferenceGlobalOp as a handle that can be returned
        // However, a module-level ReferenceGlobalOp is not valid in LLVM so remove it here
        rewriter.eraseOp(op);
        return success();
    }

    auto loc = op.getLoc();

    auto i32Type = IntegerType::get(&llvmTypeConverter.getContext(), 32);
    auto zero = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));

    if (auto globalOp = op.getGlobal())
    {
        auto type = op.getType();
        assert(type && type.hasStaticShape() && "unexpected type");
        auto elementType = llvmTypeConverter.convertType(type.getElementType())
                               .cast<Type>();
        auto elementPtrType = LLVM::LLVMPointerType::get(elementType);
        Value address = rewriter.create<LLVM::AddressOfOp>(loc, elementPtrType, op.global_name());
        Value memory = rewriter.create<LLVM::GEPOp>(
            loc,
            elementPtrType,
            address,
            ArrayRef<Value>{ zero, zero });

        auto memrefType = op.getType();
        auto memref = MemRefDescriptor::fromStaticShape(rewriter, loc, llvmTypeConverter, memrefType, memory);
        rewriter.replaceOp(op, { memref });

        return success();
    }

    if (auto globalOp = llvm::dyn_cast_or_null<LLVM::GlobalOp>(
            mlir::SymbolTable::lookupSymbolIn(op->getParentOfType<ModuleOp>(), op.global_name())))
    {
        Value address = rewriter.create<LLVM::AddressOfOp>(loc, globalOp);
        auto elementType = globalOp.getType().cast<LLVM::LLVMArrayType>().getElementType();
        Value memory = rewriter.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(elementType, globalOp.getAddrSpace()), address, ArrayRef<Value>{ zero, zero });

        auto memrefType = op.getType();
        auto memref = MemRefDescriptor::fromStaticShape(rewriter, loc, llvmTypeConverter, memrefType, memory);
        rewriter.replaceOp(op, { memref });

        return success();
    }

    return failure();
}

LogicalResult PrintFOpLowering::matchAndRewrite(PrintFOp op,
                                                PrintFOp::Adaptor operandAdapter,
                                                ConversionPatternRewriter& rewriter) const
{
    auto loc = op.getLoc();

    auto* llvmDialect = op.getContext()->getOrLoadDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    std::string fmt = op.fmt_spec().str();
    auto tag = "fmt_" + std::to_string(llvm::hash_value(fmt));
    Value fmtStr = getOrCreateGlobalString(loc, rewriter, tag, StringRef(fmt.c_str(), fmt.length() + 1), parentModule);

    // The value to print
    auto inputVals = operandAdapter.input();

    std::vector<Value> args{ fmtStr };
    args.insert(args.end(), inputVals.begin(), inputVals.end());

    auto printFnRef = op.to_stderr() ? getOrInsertPrintErrorFunction(rewriter, parentModule, llvmDialect) : getOrInsertPrintFunction(rewriter, parentModule, llvmDialect);
    rewriter.create<LLVM::CallOp>(loc, std::vector<Type>{ getPrintFnReturnType(rewriter.getContext()) }, printFnRef, args);

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
}

using UnsignedTypePair = std::pair<unsigned, Type>;
static void getMemRefArgIndicesAndTypes(
    FunctionType type,
    SmallVectorImpl<UnsignedTypePair>& argsInfo)
{
    argsInfo.reserve(type.getNumInputs());
    for (auto en : llvm::enumerate(type.getInputs()))
    {
        if (en.value().isa<MemRefType>() || en.value().isa<UnrankedMemRefType>())
            argsInfo.push_back({ en.index(), en.value() });
    }
}

// Extract an LLVM IR type from the LLVM IR dialect type.
static Type unwrap(Type type)
{
    if (!type)
        return nullptr;
    auto* mlirContext = type.getContext();
    auto wrappedLLVMType = type.dyn_cast<Type>();
    if (!wrappedLLVMType)
        emitError(UnknownLoc::get(mlirContext),
                  "conversion resulted in a non-LLVM type");
    return wrappedLLVMType;
}

LogicalResult CallOpLowering::matchAndRewrite(
    ValueCallOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const
{
    auto loc = op.getLoc();

    SmallVector<UnsignedTypePair, 4> promotedArgsInfo;
    auto funcType = op.getCalleeType();
    getMemRefArgIndicesAndTypes(funcType, promotedArgsInfo);

    SmallVector<MemRefDescriptor, 4> memrefDescriptors;
    for (auto argInfo : promotedArgsInfo)
    {
        memrefDescriptors.push_back(MemRefDescriptor{ adaptor.getOperands()[argInfo.first] });
    }

    SmallVector<mlir::Value, 4> newCallOperands;
    for (unsigned idx = 0, promotedArgsIdx = 0; idx < funcType.getNumInputs(); ++idx)
    {
        if (promotedArgsIdx < promotedArgsInfo.size() && idx == promotedArgsInfo[promotedArgsIdx].first)
        {
            newCallOperands.push_back(memrefDescriptors[promotedArgsIdx].alignedPtr(rewriter, loc));
            ++promotedArgsIdx;
        }
        else
        {
            newCallOperands.push_back(adaptor.getOperands()[idx]);
        }
    }

    TypeConverter::SignatureConversion result(funcType.getNumInputs());
    [[maybe_unused]] auto llvmType = llvmTypeConverter.convertFunctionSignature(funcType, false, result);

    SmallVector<mlir::Type, 1> resultTypes;
    if (funcType.getNumResults() > 0)
    {
        resultTypes.push_back(unwrap(llvmTypeConverter.packFunctionResults(funcType.getResults())));
    }
    [[maybe_unused]] auto newCallOp = rewriter.create<LLVM::CallOp>(loc, resultTypes, op.calleeAttr(), newCallOperands);
    rewriter.eraseOp(op);

    return success();
}

LogicalResult BitcastOpLowering::matchAndRewrite(
    BitcastOp op,
    OpAdaptor operandAdapter,
    ConversionPatternRewriter& rewriter) const
{
    LLVMTypeConverter llvmTypeConverter(rewriter.getContext());
    auto resultType = llvmTypeConverter.convertType(op.getResult().getType());

    auto arg = operandAdapter.input();
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultType, arg);
    return success();
}

// In order to make the timings as accurate as possible, we need to make sure the operations are ordered appropriately
mlir::Value GetTimeOpLowering::GetTime(ConversionPatternRewriter& rewriter, GetTimeOp& op, ModuleOp& parentModule) const
{
    auto* llvmDialect = rewriter.getContext()->getOrLoadDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    // call the platform-specific time function and convert to seconds
    auto* context = rewriter.getContext();
    auto doubleTy = Float64Type::get(context);
    mlir::Location loc = op.getLoc();

    bool isEnterRegionTimer = false;
    if (auto timerRegionTypeAttr = op->getAttrOfType<mlir::IntegerAttr>(kTimerRegionTypeIdentifier))
    {
        isEnterRegionTimer = static_cast<TimerRegionType>(timerRegionTypeAttr.getInt()) == TimerRegionType::enterRegion;
    }

    if (this->deviceInfo.IsWindows())
    {
        if (isEnterRegionTimer) {
            auto boolTy = IntegerType::get(context, 8);
            auto argTy = getPerformanceCounterType(context);
            LLVMTypeConverter llvmTypeConverter(context);
            Value one = rewriter.create<LLVM::ConstantOp>(loc, llvmTypeConverter.convertType(rewriter.getIndexType()), rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
            
            auto queryPerfFrequencyFn = getOrInsertQueryPerfFrequency(rewriter, parentModule, llvmDialect);
            Value perfFreqPtr = rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(argTy), one);
            auto getFreqCall = rewriter.create<LLVM::CallOp>(loc, std::vector<Type>{ boolTy }, queryPerfFrequencyFn, ValueRange{ perfFreqPtr });

            Value perfFreq = rewriter.create<LLVM::LoadOp>(loc, perfFreqPtr);
            Value freqDoubleVal = rewriter.create<LLVM::SIToFPOp>(loc, doubleTy, perfFreq);

            auto queryPerfCounterFn = getOrInsertQueryPerfCounter(rewriter, parentModule, llvmDialect);
            Value perfCountPtr = rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(argTy), one);
            auto getCounterCall = rewriter.create<LLVM::CallOp>(loc, std::vector<Type>{ boolTy }, queryPerfCounterFn, ValueRange{ perfCountPtr });

            [[maybe_unused]] auto getCountResult = getCounterCall.getResult(0);
            [[maybe_unused]] auto getFreqResult = getFreqCall.getResult(0);

            Value perfCount = rewriter.create<LLVM::LoadOp>(loc, perfCountPtr);
            Value ticksDoubleVal = rewriter.create<LLVM::SIToFPOp>(loc, doubleTy, perfCount);

            Value result = rewriter.create<LLVM::FDivOp>(loc, doubleTy, ticksDoubleVal, freqDoubleVal);
            return result;
        }
        else
        {
            auto queryPerfCounterFn = getOrInsertQueryPerfCounter(rewriter, parentModule, llvmDialect);
            auto queryPerfFrequencyFn = getOrInsertQueryPerfFrequency(rewriter, parentModule, llvmDialect);

            auto boolTy = IntegerType::get(context, 8);
            auto argTy = getPerformanceCounterType(context);
            LLVMTypeConverter llvmTypeConverter(context);
            Value one = rewriter.create<LLVM::ConstantOp>(loc, llvmTypeConverter.convertType(rewriter.getIndexType()), rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
            Value perfCountPtr = rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(argTy), one);
            auto getCounterCall = rewriter.create<LLVM::CallOp>(loc, std::vector<Type>{ boolTy }, queryPerfCounterFn, ValueRange{ perfCountPtr });

            Value perfFreqPtr = rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(argTy), one);
            auto getFreqCall = rewriter.create<LLVM::CallOp>(loc, std::vector<Type>{ boolTy }, queryPerfFrequencyFn, ValueRange{ perfFreqPtr });
            [[maybe_unused]] auto getCountResult = getCounterCall.getResult(0);
            [[maybe_unused]] auto getFreqResult = getFreqCall.getResult(0);

            Value perfCount = rewriter.create<LLVM::LoadOp>(loc, perfCountPtr);
            Value perfFreq = rewriter.create<LLVM::LoadOp>(loc, perfFreqPtr);

            Value ticksDoubleVal = rewriter.create<LLVM::SIToFPOp>(loc, doubleTy, perfCount);
            Value freqDoubleVal = rewriter.create<LLVM::SIToFPOp>(loc, doubleTy, perfFreq);
            Value result = rewriter.create<LLVM::FDivOp>(loc, doubleTy, ticksDoubleVal, freqDoubleVal);
            return result;
        }
    }
    else
    {
        if (isEnterRegionTimer) 
        {
            auto clockGetTimeFn = getOrInsertClockGetTime(rewriter, parentModule, llvmDialect, this->deviceInfo.numBits);

            auto llvmTimespecTy = getTimeSpecType(context, this->deviceInfo.numBits);
            auto clockIdTy = getClockIdType(context, this->deviceInfo.numBits);
            auto intTy = getIntType(context, this->deviceInfo.numBits);
            
            // Get a symbol reference to the gettime function, inserting it if necessary.
            LLVMTypeConverter llvmTypeConverter(context);
            Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvmTypeConverter.convertType(rewriter.getIndexType()), rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
            Value zero32 = rewriter.create<LLVM::ConstantOp>(loc, llvmTypeConverter.convertType(rewriter.getI32Type()), rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
            Value one = rewriter.create<LLVM::ConstantOp>(loc, llvmTypeConverter.convertType(rewriter.getIndexType()), rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
            Value one32 = rewriter.create<LLVM::ConstantOp>(loc, llvmTypeConverter.convertType(rewriter.getI32Type()), rewriter.getIntegerAttr(rewriter.getI32Type(), 1));
            
            Value clockId = rewriter.create<LLVM::ConstantOp>(loc, clockIdTy, rewriter.getI64IntegerAttr(int64_t(ClockID::ACCERA_CLOCK_REALTIME)));

            Value timespecPtr = rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(llvmTimespecTy), one);
            Value secondsPtr = rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(intTy), timespecPtr, ValueRange{ zero, zero32 });
            Value nanosecondsPtr = rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(intTy), timespecPtr, ValueRange{ zero, one32 });

            std::vector<Value> args{ clockId, timespecPtr };
            auto getTimeCall = rewriter.create<LLVM::CallOp>(loc, std::vector<Type>{ getIntType(context, this->deviceInfo.numBits) }, clockGetTimeFn, args);
            [[maybe_unused]] auto getTimeResult = getTimeCall.getResult(0);

            Value secondsIntVal = rewriter.create<LLVM::LoadOp>(loc, secondsPtr);
            Value nanosecondsIntVal = rewriter.create<LLVM::LoadOp>(loc, nanosecondsPtr);
            Value secondsDoubleVal = rewriter.create<LLVM::SIToFPOp>(loc, doubleTy, secondsIntVal);
            Value nanosecondsDoubleVal = rewriter.create<LLVM::UIToFPOp>(loc, doubleTy, nanosecondsIntVal);
            Value divisor = rewriter.create<LLVM::ConstantOp>(loc, doubleTy, rewriter.getF64FloatAttr(1.0e9));
            Value nanoseconds = rewriter.create<LLVM::FDivOp>(loc, doubleTy, nanosecondsDoubleVal, divisor);
            Value totalSecondsDoubleVal = rewriter.create<LLVM::FAddOp>(loc, doubleTy, secondsDoubleVal, nanoseconds);
            return totalSecondsDoubleVal;
        }
        else
        {
            auto clockGetTimeFn = getOrInsertClockGetTime(rewriter, parentModule, llvmDialect, this->deviceInfo.numBits);

            auto llvmTimespecTy = getTimeSpecType(context, this->deviceInfo.numBits);
            auto clockIdTy = getClockIdType(context, this->deviceInfo.numBits);
            auto intTy = getIntType(context, this->deviceInfo.numBits);

            // Get a symbol reference to the gettime function, inserting it if necessary.
            LLVMTypeConverter llvmTypeConverter(context);
            Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvmTypeConverter.convertType(rewriter.getIndexType()), rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
            Value zero32 = rewriter.create<LLVM::ConstantOp>(loc, llvmTypeConverter.convertType(rewriter.getI32Type()), rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
            Value one = rewriter.create<LLVM::ConstantOp>(loc, llvmTypeConverter.convertType(rewriter.getIndexType()), rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
            Value one32 = rewriter.create<LLVM::ConstantOp>(loc, llvmTypeConverter.convertType(rewriter.getI32Type()), rewriter.getIntegerAttr(rewriter.getI32Type(), 1));
            Value clockId = rewriter.create<LLVM::ConstantOp>(loc, clockIdTy, rewriter.getI64IntegerAttr(int64_t(ClockID::ACCERA_CLOCK_REALTIME)));

            Value timespecPtr = rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(llvmTimespecTy), one);

            std::vector<Value> args{ clockId, timespecPtr };
            auto getTimeCall = rewriter.create<LLVM::CallOp>(loc, std::vector<Type>{ getIntType(context, this->deviceInfo.numBits) }, clockGetTimeFn, args);
            [[maybe_unused]] auto getTimeResult = getTimeCall.getResult(0);
            
            Value secondsPtr = rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(intTy), timespecPtr, ValueRange{ zero, zero32 });
            Value nanosecondsPtr = rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(intTy), timespecPtr, ValueRange{ zero, one32 });

            Value secondsIntVal = rewriter.create<LLVM::LoadOp>(loc, secondsPtr);
            Value nanosecondsIntVal = rewriter.create<LLVM::LoadOp>(loc, nanosecondsPtr);
            Value secondsDoubleVal = rewriter.create<LLVM::SIToFPOp>(loc, doubleTy, secondsIntVal);
            Value nanosecondsDoubleVal = rewriter.create<LLVM::UIToFPOp>(loc, doubleTy, nanosecondsIntVal);
            Value divisor = rewriter.create<LLVM::ConstantOp>(loc, doubleTy, rewriter.getF64FloatAttr(1.0e9));
            Value nanoseconds = rewriter.create<LLVM::FDivOp>(loc, doubleTy, nanosecondsDoubleVal, divisor);
            Value totalSecondsDoubleVal = rewriter.create<LLVM::FAddOp>(loc, doubleTy, secondsDoubleVal, nanoseconds);
            return totalSecondsDoubleVal;
        }
    }
}

LogicalResult GetTimeOpLowering::matchAndRewrite(
    GetTimeOp op,
    OpAdaptor,
    ConversionPatternRewriter& rewriter) const
{
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto currentTime = GetTime(rewriter, op, parentModule);
    rewriter.replaceOp(op, { currentTime });
    return success();
}

LogicalResult RangeOpLowering::matchAndRewrite(
    RangeOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const
{
    auto loc = op.getLoc();

    // Convert the given range descriptor type to the LLVMIR dialect.
    // Range descriptor contains the range bounds and the step as 64-bit integers.
    //
    // struct {
    //   int64_t min;
    //   int64_t max;
    //   int64_t step;
    // };
    LLVMTypeConverter llvmTypeConverter(rewriter.getContext());
    auto rangeType = op.getType().cast<RangeType>();
    auto* context = rangeType.getContext();
    auto int64Ty = llvmTypeConverter.convertType(IntegerType::get(context, 64));
    auto rangeDescriptor = LLVM::LLVMStructType::getLiteral(context, { int64Ty, int64Ty, int64Ty });

    // Fill in an aggregate value of the descriptor.
    Value desc = rewriter.create<LLVM::UndefOp>(loc, rangeDescriptor);
    desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, adaptor.min(), rewriter.getI64ArrayAttr(0));
    desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, adaptor.max(), rewriter.getI64ArrayAttr(1));
    desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, adaptor.step(), rewriter.getI64ArrayAttr(2));
    rewriter.replaceOp(op, desc);
    return success();
}

void ValueToLLVMLoweringPass::runOnModule()
{
    llvm::DebugFlag =
#if !defined(NDEBUG) && 0
        true
#else
        false
#endif
        ;

    LLVMConversionTarget target(getContext());

    auto moduleOp = getModule();
    auto snapshotter = _intrapassSnapshotter.MakeSnapshotPipe();
    snapshotter.Snapshot("Initial", moduleOp);

    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<intrinsics::AcceraIntrinsicsDialect>();

    // Set pass parameter values with command line options inherited from ConvertValueToLLVMBase
    mlir::LowerToLLVMOptions options(&getContext());
    options.useBarePtrCallConv = useBarePtrCallConv;
    options.emitCWrappers = emitCWrappers;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
    {
        options.overrideIndexBitwidth(indexBitwidth);
    }
    options.allocLowering = mlir::LowerToLLVMOptions::AllocLowering::AlignedAlloc;
    options.dataLayout = llvm::DataLayout(dataLayout);

    LLVMTypeConverter llvmTypeConverter(&getContext(), options);

    // Create bare pointer llvm options for handling raw-pointer-API function to non-raw-pointer-API function conversion and calls
    mlir::LowerToLLVMOptions barePtrOptions = options;
    barePtrOptions.useBarePtrCallConv = true;
    barePtrOptions.emitCWrappers = false;

    LLVMTypeConverter barePtrTypeConverter(&getContext(), barePtrOptions);

    llvm::SmallVector<Operation*> rawPointerFuncs;

    for (auto it = moduleOp.getOps().begin(), e = moduleOp.getOps().end(); it != e; ++it)
    {
        if (it->hasAttr(RawPointerAPIAttrName))
        {
            rawPointerFuncs.push_back(&*it);
        }
    }

    // Apply targeted Raw / Bare pointer conversions manually
    {
        RewritePatternSet patterns(&getContext());

        patterns.insert<RawPointerAPIFnConversion>(barePtrTypeConverter);
        patterns.insert<RawPointerAPICallOpConversion>(llvmTypeConverter, 100);

        if (failed(applyPartialConversion(rawPointerFuncs, target, std::move(patterns))))
        {
            signalPassFailure();
        }
    }

    snapshotter.Snapshot("BarePtrConversion", moduleOp);

    {
        auto intermediateTarget = target;
        intermediateTarget.addLegalDialect<mlir::arith::ArithmeticDialect>();
        intermediateTarget.addLegalDialect<mlir::BuiltinDialect>();

        RewritePatternSet patterns(&getContext());
        populateValueToLLVMNonMemPatterns(llvmTypeConverter, patterns, this->deviceInfo);

        populateLinalgToLLVMConversionPatterns(llvmTypeConverter, patterns);

        populateVectorToLLVMConversionPatterns(llvmTypeConverter, patterns, /*reassociateFPReductions*/ true);

        // Subset of LowerVectorToLLVMPass patterns
        vector::populateVectorToVectorCanonicalizationPatterns(patterns);
        vector::populateVectorBroadcastLoweringPatterns(patterns);
        vector::populateVectorMaskOpLoweringPatterns(patterns);
        vector::populateVectorShapeCastLoweringPatterns(patterns);
        vector::populateVectorTransposeLoweringPatterns(patterns);
        vector::populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);
        vector::populateVectorContractLoweringPatterns(patterns, vector::VectorTransformsOptions{}.setVectorTransferSplit(mlir::vector::VectorTransferSplit::VectorTransfer));
        vector::populateVectorMaskMaterializationPatterns(patterns, true);

        if (failed(applyPartialConversion(moduleOp, intermediateTarget, std::move(patterns))))
        {
            signalPassFailure();
        }
    }

    snapshotter.Snapshot("ToLLVM_NonMem", moduleOp);

    FrozenRewritePatternSet toLLVMPatterns;
    {
        RewritePatternSet patterns(&getContext());

        populateValueToLLVMMemPatterns(llvmTypeConverter, patterns);
        populateReshapeOpToLLVMMemPatterns(llvmTypeConverter, patterns);
        populateMathToLLVMConversionPatterns(llvmTypeConverter, patterns);
        populateMemRefToLLVMConversionPatterns(llvmTypeConverter, patterns);
        populateStdToLLVMConversionPatterns(llvmTypeConverter, patterns);
        arith::populateArithmeticToLLVMConversionPatterns(llvmTypeConverter, patterns);
        arith::populateArithmeticExpandOpsPatterns(patterns);
        cf::populateControlFlowToLLVMConversionPatterns(llvmTypeConverter, patterns);

        // Subset of LowerVectorToLLVMPass patterns
        vector::populateVectorToVectorCanonicalizationPatterns(patterns);
        vector::populateVectorBroadcastLoweringPatterns(patterns);
        vector::populateVectorMaskOpLoweringPatterns(patterns);
        vector::populateVectorShapeCastLoweringPatterns(patterns);
        vector::populateVectorTransposeLoweringPatterns(patterns);
        vector::populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);

        populateVectorToLLVMConversionPatterns(llvmTypeConverter, patterns, /*reassociateFPReductions*/ true);
        vector::populateVectorContractLoweringPatterns(patterns, vector::VectorTransformsOptions{}.setVectorTransferSplit(mlir::vector::VectorTransferSplit::VectorTransfer));
        vector::populateVectorMaskMaterializationPatterns(patterns, true);

        // cf. mlir\lib\Conversion\OpenMPToLLVM\OpenMPToLLVM.cpp
        target.addDynamicallyLegalOp<mlir::omp::ParallelOp, mlir::omp::WsLoopOp>(
            [&](Operation* op) { return llvmTypeConverter.isLegal(&op->getRegion(0)); });
        target.addLegalOp<mlir::omp::TerminatorOp, mlir::omp::TaskyieldOp, mlir::omp::FlushOp, mlir::omp::BarrierOp, mlir::omp::TaskwaitOp>();

        populateOpenMPToLLVMConversionPatterns(llvmTypeConverter, patterns);

        toLLVMPatterns = std::move(patterns);
        if (failed(applyPartialConversion(moduleOp, target, toLLVMPatterns)))
        {
            signalPassFailure();
        }
    }

    snapshotter.Snapshot("ToLLVM_Mem", moduleOp);

    {
        RewritePatternSet patterns(&getContext());
        patterns.insert<LLVMCallFixupPattern>(&getContext());

        FrozenRewritePatternSet frozen{ std::move(patterns) };
        {
            if (failed(applyPatternsAndFoldGreedily(moduleOp, frozen)))
            {
                signalPassFailure();
            }
        }
    }

    snapshotter.Snapshot("Final", moduleOp);

    llvm::DebugFlag = false;
}

namespace accera::transforms::value
{

void populateGlobalValueToLLVMNonMemPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<GlobalOpToLLVMLowering>(typeConverter, context);
}

void populateLocalValueToLLVMNonMemPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns, accera::value::TargetDevice deviceInfo)
{
    mlir::MLIRContext* context = patterns.getContext();

    patterns.insert<
        CPUEarlyReturnRewritePattern,
        ReferenceGlobalOpLowering,
        BitcastOpLowering,
        CallOpLowering,
        PrintFOpLowering,
        RangeOpLowering,
        VpmaddwdOpLowering,
        VmaxpsOpLowering,
        VminpsOpLowering,
        RoundOpLowering,
        MemrefAllocOpLowering>(typeConverter, context);

    patterns.insert<GetTimeOpLowering>(typeConverter, context, deviceInfo);
}

void populateValueToLLVMNonMemPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns, accera::value::TargetDevice deviceInfo)
{
    populateGlobalValueToLLVMNonMemPatterns(typeConverter, patterns);
    populateLocalValueToLLVMNonMemPatterns(typeConverter, patterns, deviceInfo);
}

void populateValueToLLVMMemPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();

    patterns.insert<ValueMemRefCastOpLowering>(typeConverter, context);
}

void populateReshapeOpToLLVMMemPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns)
{
    patterns.insert<ExpandShapeOpLowering>(typeConverter);
}

const mlir::LowerToLLVMOptions& GetDefaultAcceraLLVMOptions(mlir::MLIRContext* context)
{
    static LowerToLLVMOptions options(context); // statically allocated default we hand out copies to

    // set Accera alterations to the defaults
    options.allocLowering = mlir::LowerToLLVMOptions::AllocLowering::AlignedAlloc;

    return options;
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToLLVMPass(mlir::LowerToLLVMOptions options)
{
    return std::make_unique<ValueToLLVMLoweringPass>(options.useBarePtrCallConv, options.emitCWrappers, options.getIndexBitwidth(), options.allocLowering == mlir::LowerToLLVMOptions::AllocLowering::AlignedAlloc, options.dataLayout);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToLLVMPass(mlir::MLIRContext* context)
{
    return createValueToLLVMPass();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToLLVMPass(bool useBasePtrCallConv,
                                                                           bool emitCWrappers,
                                                                           unsigned indexBitwidth,
                                                                           bool useAlignedAlloc,
                                                                           llvm::DataLayout dataLayout,
                                                                           accera::value::TargetDevice deviceInfo /*  = {} */,
                                                                           const IntraPassSnapshotOptions& options /*  = {} */)
{
    return std::make_unique<ValueToLLVMLoweringPass>(useBasePtrCallConv, emitCWrappers, indexBitwidth, useAlignedAlloc, dataLayout, deviceInfo, options);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToLLVMPass()
{
    // The values here should always match the ones specified by GetDefaultAcceraLLVMOptions
    llvm::DataLayout dataLayout = llvm::DataLayout("");
    return createValueToLLVMPass(/* useBasePtrCallConv = */ false,
                                 /* emitCWrappers = */ false,
                                 /* indexBitwidth = */ kDeriveIndexBitwidthFromDataLayout,
                                 /* useAlignedAlloc = */ true,
                                 /* dataLayout = */ dataLayout);
}

} // namespace accera::transforms::value
