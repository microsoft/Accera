////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gpu/AcceraToGPUPass.h"

#include "AcceraPasses.h"
#include "ir/include/value/ValueEnums.h"
#include "ir/include/value/ValueMFMAOp.h"

#include <ir/include/IRUtil.h>

#include <utilities/include/Exception.h>

#include <llvm/Support/ErrorHandling.h>

#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVEnums.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <optional>
#include <string>

using namespace mlir;
using accera::transforms::populateAcceraToNVVMPatterns;
using accera::transforms::populateAcceraToROCDLPatterns;
using accera::transforms::populateAcceraToSPIRVPatterns;

namespace utilir = accera::ir::util;
namespace vir = accera::ir::value;

namespace
{

// We need to make this greater than 1 to preempt builtin patterns
constexpr unsigned kAcceraGPUPatternBenefit = 10;
const char kPrivateMemoryVarPrefix[] = "__private_mem__";

// cf mlir/lib/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.cpp
/// Returns true if the allocations of type `t` can be lowered to SPIR-V.
static bool isSPIRVFunctionAllocationSupported(MemRefType t)
{
    // Currently only support workgroup local memory allocations with static
    // shape and int or float or vector of int or float element type.
    if (!(t.hasStaticShape() && SPIRVTypeConverter::getMemorySpaceForStorageClass(spirv::StorageClass::Function) == t.getMemorySpaceAsInt()))
        return false;
    Type elementType = t.getElementType();
    if (auto vecType = elementType.dyn_cast<VectorType>())
        elementType = vecType.getElementType();
    return elementType.isIntOrFloat();
}

static std::optional<vir::ExecutionRuntime> getGPURuntimeTarget(mlir::Operation* op)
{
    // TODO: Add tests, verify, enable generic version
#if 1
    return vir::ExecutionRuntime::Rocm;
#else
    auto target = utilir::ResolveExecutionTarget(op);
    if (!target || target != vir::ExecutionTarget::GPU)
    {
        return std::nullopt;
    }
    return utilir::ResolveExecutionRuntime(op);
#endif
}

static std::optional<vir::ExecutionRuntime> getRuntimeTarget(mlir::ModuleOp* op)
{
    auto funOps = op->getOps<FuncOp>();
    for (auto funOp : funOps)
    {
        auto runtime = getGPURuntimeTarget(funOp);
        if (runtime)
        {
            return runtime;
        }
    }
    return std::nullopt;
}

template <vir::ExecutionRuntime Runtime>
static bool hasRuntimeTarget(mlir::Operation* op)
{
    auto runtime = getGPURuntimeTarget(op);
    if (!runtime)
    {
        return false;
    }
    return *runtime == Runtime;
}

static bool hasVulkanRuntimeTarget(mlir::Operation* op)
{
    return hasRuntimeTarget<vir::ExecutionRuntime::Vulkan>(op);
}

static bool hasNVVMRuntimeTarget(mlir::Operation* op)
{
    return hasRuntimeTarget<vir::ExecutionRuntime::CUDA>(op);
}

static bool hasROCDLRuntimeTarget(mlir::Operation* op)
{
    return hasRuntimeTarget<vir::ExecutionRuntime::Rocm>(op);
}

struct PrivateAllocToSPIRVConversion : public OpConversionPattern<memref::AllocOp>
{
    PrivateAllocToSPIRVConversion(SPIRVTypeConverter& typeConverter, MLIRContext* context) :
        OpConversionPattern(typeConverter, context, kAcceraGPUPatternBenefit)
    {}

    LogicalResult matchAndRewrite(memref::AllocOp op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final
    {

        if (!hasVulkanRuntimeTarget(op))
        {
            return failure();
        }

        // cf mlir/lib/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.cpp

        MemRefType allocType = op.getType();
        if (!isSPIRVFunctionAllocationSupported(allocType))
            return failure();

        // Get the SPIR-V type for the allocation.
        Type spirvType = getTypeConverter()->convertType(allocType);

        rewriter.replaceOpWithNewOp<spirv::VariableOp>(op, spirvType, *SPIRVTypeConverter::getStorageClassForMemorySpace(allocType.getMemorySpaceAsInt()), mlir::Value{});
        return success();
    }
};

/// Removes a deallocation if it is a supported allocation
struct PrivateDeallocToSPIRVConversion final : public OpConversionPattern<memref::DeallocOp>
{
    PrivateDeallocToSPIRVConversion(SPIRVTypeConverter& typeConverter, MLIRContext* context) :
        OpConversionPattern(typeConverter, context, kAcceraGPUPatternBenefit)
    {}

    LogicalResult matchAndRewrite(memref::DeallocOp op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final
    {
        if (!hasVulkanRuntimeTarget(op))
        {
            return failure();
        }

        // cf mlir/lib/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.cpp

        MemRefType deallocType = op.memref().getType().cast<MemRefType>();
        if (!isSPIRVFunctionAllocationSupported(deallocType))
        {
            return op.emitError("unhandled deallocation type");
        }
        rewriter.eraseOp(op);
        return success();
    }
};

struct EarlyReturnToSPIRVReturnPattern : public OpConversionPattern<vir::EarlyReturnOp>
{
    using OpConversionPattern<vir::EarlyReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::EarlyReturnOp op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final
    {
        if (!hasVulkanRuntimeTarget(op))
        {
            return failure();
        }

        if (operands.empty())
        {
            rewriter.replaceOpWithNewOp<spirv::ReturnOp>(op);
        }
        else
        {
            assert(operands.size() == 1);
            rewriter.replaceOpWithNewOp<spirv::ReturnValueOp>(op, operands[0]);
        }
        return success();
    }
};

struct EarlyReturnToGPUReturnPattern : public OpRewritePattern<vir::EarlyReturnOp>
{
    using OpRewritePattern<vir::EarlyReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(vir::EarlyReturnOp op, PatternRewriter& rewriter) const final
    {
        if (!hasNVVMRuntimeTarget(op) && !hasROCDLRuntimeTarget(op))
        {
            return failure();
        }

        rewriter.replaceOpWithNewOp<gpu::ReturnOp>(op, op->getOperands());

        return success();
    }
};

struct ValueBarrierToSPIRVBarrierConversion final : public OpConversionPattern<vir::BarrierOp>
{
    ValueBarrierToSPIRVBarrierConversion(SPIRVTypeConverter& typeConverter, MLIRContext* context) :
        OpConversionPattern(typeConverter, context, kAcceraGPUPatternBenefit)
    {}

    LogicalResult matchAndRewrite(vir::BarrierOp op, ArrayRef<Value>, ConversionPatternRewriter& rewriter) const final
    {
        if (!hasVulkanRuntimeTarget(op))
        {
            return failure();
        }
        switch (op.scope())
        {
        case vir::BarrierScope::Block:
            rewriter.replaceOpWithNewOp<spirv::ControlBarrierOp>(
                op,
                /* execution_scope = */ mlir::spirv::Scope::Workgroup,
                /* memory_scope = */ mlir::spirv::Scope::Workgroup,
                /* memory_semantics = */ mlir::spirv::MemorySemantics::AcquireRelease);
            break;
        case vir::BarrierScope::Warp:
            rewriter.replaceOpWithNewOp<spirv::ControlBarrierOp>(
                op,
                /* execution_scope = */ mlir::spirv::Scope::Subgroup,
                /* memory_scope = */ mlir::spirv::Scope::Subgroup,
                /* memory_semantics = */ mlir::spirv::MemorySemantics::AcquireRelease | mlir::spirv::MemorySemantics::SubgroupMemory);
            break;
        default:
            assert(true && "Unhandled barrier scope.");
            return rewriter.notifyMatchFailure(op, "Unhandled barrier scope.");
        }
        return success();
    }
};

struct ValueBarrierToGPUBarrierConversion final : public OpRewritePattern<vir::BarrierOp>
{
    using OpRewritePattern<vir::BarrierOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(vir::BarrierOp op, PatternRewriter& rewriter) const final
    {
        switch (op.scope())
        {
        case vir::BarrierScope::Block:
            rewriter.replaceOpWithNewOp<gpu::BarrierOp>(op);
            break;
        case vir::BarrierScope::Threadfence:
            rewriter.replaceOpWithNewOp<mlir::LLVM::FenceOp>(op, mlir::LLVM::AtomicOrdering::seq_cst, "agent");
            break;
        default:
            assert(true && "Unhandled barrier scope.");
            return rewriter.notifyMatchFailure(op, "Unhandled barrier scope.");
        }
        return success();
    }
};

struct ValueMFMALoadMatrixOpToRocDLConversion final : public OpRewritePattern<vir::MFMALoadMatrixOp>
{
    using OpRewritePattern<vir::MFMALoadMatrixOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(vir::MFMALoadMatrixOp op, PatternRewriter& rewriter) const final
    {
        using namespace accera::utilities;

        throw LogicException(LogicExceptionErrors::notImplemented);

        if (!hasROCDLRuntimeTarget(op))
        {
            return failure();
        }
        auto loc = op.getLoc();
        auto memref = op.srcMemref();
        auto memrefType = memref.getType().cast<MemRefType>();
        auto shape = memrefType.getShape();
        auto elementType = memrefType.getElementType();

        // TODO: Literal constants should be provided by a helper struct (TensorizationInfo?)
        auto vecSize = shape[0] == 16 ? 4 : 16;

        auto vecTy = mlir::VectorType::get({ vecSize }, elementType);

        auto i32Ty = rewriter.getIntegerType(32);
        Value zero = rewriter.create<mlir::ConstantOp>(loc, i32Ty, rewriter.getZeroAttr(i32Ty));

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);

        // TODO: Literal constants should be provided by a helper struct (TensorizationInfo?)
        if (vecSize == 4)
        {
            llvm::SmallVector<int64_t, 4> offsets{ 0, 0 };
            llvm::SmallVector<int64_t, 4> sizes{ 16, 16 };
            llvm::SmallVector<int64_t, 4> strides{ 4, 1 };
            auto rowMemrefTy = MemRefType::get({ 4, 4 }, elementType);
            [[maybe_unused]] auto row = rewriter.create<memref::SubViewOp>(loc, rowMemrefTy, memref, offsets, sizes, strides);
            // auto vec = rewriter.replaceOpWithNewOp<AffineVectorLoadOp>(op, vecTy, row, ValueRange{ rewriter.create<ConstantIndexOp>(loc, 0) });
        }
        else
        {
            return rewriter.notifyMatchFailure(op, "unhandled vector size");
        }

        return success();
    }
};

struct ValueMFMAStoreMatrixOpToRocDLConversion final : public OpRewritePattern<vir::MFMAStoreMatrixOp>
{
    using OpRewritePattern<vir::MFMAStoreMatrixOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(vir::MFMAStoreMatrixOp op, PatternRewriter& rewriter) const final
    {
        using namespace accera::utilities;

        throw LogicException(LogicExceptionErrors::notImplemented);

        return success();
    }
};

struct ValueMFMAComputeToRocDLConversion final : public OpRewritePattern<vir::MFMAComputeOp>
{
    using OpRewritePattern<vir::MFMAComputeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(vir::MFMAComputeOp op, PatternRewriter& rewriter) const final
    {
        using namespace accera::utilities;

        throw LogicException(LogicExceptionErrors::notImplemented);

        if (!hasROCDLRuntimeTarget(op))
        {
            return failure();
        }
        auto adaptor = vir::MFMAComputeOpAdaptor(op);
        auto loc = op.getLoc();

        auto opA = adaptor.opA();
        auto opB = adaptor.opB();
        auto opC = adaptor.opC();
        if (!opA.getType().isa<vir::MFMAMatrixType>())
        {
            return rewriter.notifyMatchFailure(op, "expecting a matrix type for OpA");
        }
        if (!opB.getType().isa<vir::MFMAMatrixType>())
        {
            return rewriter.notifyMatchFailure(op, "expecting a matrix type for OpB");
        }
        if (!opC.getType().isa<vir::MFMAMatrixType>())
        {
            return rewriter.notifyMatchFailure(op, "expecting a matrix type for OpC");
        }

        auto i32Ty = rewriter.getIntegerType(32);
        [[maybe_unused]] Value zero = rewriter.create<mlir::ConstantOp>(loc, i32Ty, rewriter.getZeroAttr(i32Ty));

        // Value accumVecIn;
        // if (accumIn.getType().dyn_cast<MemRefType>())
        // {
        //     auto memRefType = accumIn.getType().cast<MemRefType>();
        //     unsigned rank = memRefType.getRank();
        //     if (rank != 1)
        //     {
        //         return rewriter.notifyMatchFailure(op, "accumulation type for the MFMA op must be a vector (memref rank = 1).");
        //     }
        //     if (!memRefType.hasStaticShape())
        //     {
        //         return rewriter.notifyMatchFailure(op, "accumulation type for the MFMA op must have a static shape.");
        //     }
        //     if (memRefType.getElementType() != rewriter.getF32Type())
        //     {
        //         return rewriter.notifyMatchFailure(op, "accumulation type for the MFMA op must be a floating point number.");
        //     }
        //     auto numElements = memRefType.getNumElements();
        //     if (numElements != 4 && numElements != 16)
        //     {
        //         return rewriter.notifyMatchFailure(op, "accumulation type for the MFMA op must be a floating point vector of size 4 or 16.");
        //     }

        //     auto vecF32Ty = mlir::VectorType::get({ numElements }, rewriter.getF32Type());
        //     accumVecIn = rewriter.create<AffineVectorLoadOp>(loc, vecF32Ty, accumIn, ValueRange{ rewriter.create<ConstantIndexOp>(loc, 0) });
        // }
        // else
        // {
        //     accumVecIn = accumIn;
        // }
        // auto accumVecInTy = accumVecIn.getType().cast<MFMAMatrix>();
        // Operation* newOp;

        // if (accumVecInTy.getNumElements() == 16)
        // {
        //     newOp = rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x2f32>(op, accumVecIn.getType(), ValueRange{ adaptor.opA(), adaptor.opB(), accumVecIn, zero, zero, zero });
        // }
        // else
        // {
        //     newOp = rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x4f32>(op, accumVecIn.getType(), ValueRange{ adaptor.opA(), adaptor.opB(), accumVecIn, zero, zero, zero });
        // }
        // rewriter.setInsertionPointAfter(newOp);

        // assert(op->getNumResults() > 0);
        // auto result = newOp->getResult(0);

        // for (auto user : result.getUsers())
        // {
        //     if (memref::LoadOp loadOp = dyn_cast<memref::LoadOp>(user))
        //     {
        //         mlir::OpBuilder::InsertionGuard guard(rewriter);
        //         rewriter.setInsertionPoint(loadOp);
        //         auto idx =  loadOp.indices().front();
        //         auto idxAsInt = rewriter.create<IndexCastOp>(loadOp->getLoc(), idx, rewriter.getI64Type());
        //         rewriter.replaceOpWithNewOp<vector::ExtractElementOp>(loadOp, result, idxAsInt);
        //     }
        //     else if (memref::StoreOp storeOp = dyn_cast<memref::StoreOp>(user))
        //     {
        //         mlir::OpBuilder::InsertionGuard guard(rewriter);
        //         rewriter.setInsertionPoint(storeOp);
        //         auto value = storeOp.getValueToStore();
        //         auto idx =  loadOp.indices().front();
        //         auto idxAsInt = rewriter.create<IndexCastOp>(loadOp->getLoc(), idx, rewriter.getI64Type());
        //         rewriter.replaceOpWithNewOp<vector::InsertElementOp>(storeOp, value, result, idxAsInt);
        //     } else {
        //         return rewriter.notifyMatchFailure(op, "Unsupported op. Users for the result of the MFMA op must be either store or load instructions.");
        //     }
        // }
        return success();
    }
};

struct AcceraToSPIRVPass : public accera::transforms::ConvertAcceraToSPIRVBase<AcceraToSPIRVPass>
{
    void runOnOperation() final
    {
        ModuleOp module = getOperation();
        auto runtime = getRuntimeTarget(&module);
        if (!runtime || *runtime != vir::ExecutionRuntime::Vulkan)
        {
            return;
        }

        // cf mlir/lib/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.cpp -- GPUToSPIRVPass::runOnOperation
        MLIRContext* context = &getContext();
        SmallVector<Operation*, 1> kernelModules;
        OpBuilder builder(context);
        module.walk([&builder, &kernelModules](gpu::GPUModuleOp moduleOp) {
            // For each kernel module (should be only 1 for now, but that is not a
            // requirement here), clone the module for conversion because the
            // gpu.launch function still needs the kernel module.
            builder.setInsertionPoint(moduleOp.getOperation());
            kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
        });

        auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
        std::unique_ptr<ConversionTarget> target = SPIRVConversionTarget::get(targetAttr);

        SPIRVTypeConverter typeConverter(targetAttr);
        ScfToSPIRVContext scfContext;
        RewritePatternSet patterns(context);
        populateAcceraToSPIRVPatterns(typeConverter, context, patterns);
        populateGPUToSPIRVPatterns(typeConverter, patterns);
        populateSCFToSPIRVPatterns(typeConverter, scfContext, patterns);
        populateStandardToSPIRVPatterns(typeConverter, patterns);

        if (failed(applyFullConversion(kernelModules, *target, std::move(patterns))))
            return signalPassFailure();
    }
};

struct AcceraToROCDLPass : public accera::transforms::ConvertAcceraToROCDLBase<AcceraToROCDLPass>
{
    void runOnOperation() final
    {
        MLIRContext* context = &getContext();
        auto module = getOperation();

        // TODO: Enable querying of module for execution runtime
        // auto runtime = getRuntimeTarget(&module);
        // if (!runtime || *runtime != vir::ExecutionRuntime::Rocm)
        // {
        //     return;
        // }

        RewritePatternSet patterns(context);
        populateAcceraToROCDLPatterns(patterns);

        (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
    }
};

struct AcceraToNVVMPass : public accera::transforms::ConvertAcceraToNVVMBase<AcceraToNVVMPass>
{
    void runOnOperation() final
    {

        MLIRContext* context = &getContext();
        auto module = getOperation();

        // TODO: Enable querying of module for execution runtime
        // auto runtime = getRuntimeTarget(&module);
        // if (!runtime || *runtime != vir::ExecutionRuntime::CUDA)
        // {
        //     return;
        // }

        RewritePatternSet patterns(context);
        populateAcceraToNVVMPatterns(patterns);

        (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
    }
};
} // namespace

namespace accera::transforms
{

void populateAcceraToSPIRVPatterns(mlir::SPIRVTypeConverter& typeConverter, mlir::MLIRContext* context, mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<
        EarlyReturnToSPIRVReturnPattern,
        ValueBarrierToSPIRVBarrierConversion,
        PrivateAllocToSPIRVConversion,
        PrivateDeallocToSPIRVConversion>(typeConverter, context);
}

void populateAcceraToROCDLPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<
        EarlyReturnToGPUReturnPattern,
        ValueBarrierToGPUBarrierConversion,
        ValueMFMALoadMatrixOpToRocDLConversion,
        ValueMFMAComputeToRocDLConversion,
        ValueMFMAStoreMatrixOpToRocDLConversion>(patterns.getContext());
}

void populateAcceraToNVVMPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<
        EarlyReturnToGPUReturnPattern,
        ValueBarrierToGPUBarrierConversion>(patterns.getContext());
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraToSPIRVPass()
{
    return std::make_unique<AcceraToSPIRVPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraToNVVMPass()
{
    return std::make_unique<AcceraToNVVMPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraToROCDLPass()
{
    return std::make_unique<AcceraToROCDLPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraToGPUPass(accera::value::ExecutionRuntime runtime)
{
    using accera::value::ExecutionRuntime;
    switch (runtime)
    {
    case ExecutionRuntime::CUDA:
        return createAcceraToNVVMPass();
    case ExecutionRuntime::Rocm:
        // TODO: default gpu runtime is rocm
        [[fallthrough]];
    case ExecutionRuntime::Default:
        return createAcceraToROCDLPass();
    case ExecutionRuntime::Vulkan:
        return createAcceraToSPIRVPass();
    default:
        llvm::llvm_unreachable_internal("The execution runtime must be specified.");
    }
}

} // namespace accera::transforms
