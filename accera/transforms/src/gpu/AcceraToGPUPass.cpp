////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gpu/AcceraToGPUPass.h"

#include "AcceraPasses.h"
#include "ir/include/value/ValueDialect.h"
#include "ir/include/value/ValueEnums.h"
#include "ir/include/value/ValueMFMAOp.h"

#include <ir/include/IRUtil.h>

#include <utilities/include/Exception.h>

#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVEnums.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

#include <optional>

#define DEBUG_TYPE "accera-to-gpu"

using namespace mlir;
using accera::transforms::populateAcceraToNVVMPatterns;
using accera::transforms::populateAcceraToROCDLPatterns;
using accera::transforms::populateAcceraToSPIRVPatterns;
using accera::transforms::populateGPUSimplificationPatterns;

namespace ir = accera::ir;
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
    // Currently only support workgroup private memory allocations with static
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
    return utilir::ResolveExecutionRuntime(op);
}

template <vir::ExecutionRuntime Runtime>
static bool hasRuntimeTarget(mlir::Operation* op)
{
    auto runtime = getGPURuntimeTarget(op).value_or(vir::ExecutionRuntime::NONE);
    return runtime == Runtime;
}

int dimIndexToInteger(llvm::StringRef dim)
{
    return ::llvm::StringSwitch<int>(dim)
        .Case("x", 0)
        .Case("y", 1)
        .Case("z", 2)
        .Default(-1);
}

struct PrivateAllocToSPIRVConversion : public OpConversionPattern<memref::AllocOp>
{
    PrivateAllocToSPIRVConversion(SPIRVTypeConverter& typeConverter, MLIRContext* context) :
        OpConversionPattern(typeConverter, context, kAcceraGPUPatternBenefit)
    {}

    LogicalResult matchAndRewrite(memref::AllocOp op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final
    {

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

        rewriter.replaceOpWithNewOp<gpu::ReturnOp>(op, op->getOperands());

        return success();
    }
};

// Tries to match to a public facing function that calls another function as its
// sole non-terminator op, which in turn launches a GPU function.
// Once the match is found, renames the GPU function with the name of the top-level function
// plus a suffix of '__gpu__', and updates the launch gpu func op. Updates the runtime used by the
// top-level function.
struct CreateDeviceFuncLauncherPairPattern : public OpRewritePattern<FuncOp>
{
    CreateDeviceFuncLauncherPairPattern(vir::ExecutionRuntime targetRuntime, MLIRContext* context, PatternBenefit benefit = 1) :
        OpRewritePattern(context, benefit), _target(targetRuntime) {}

    LogicalResult matchAndRewrite(FuncOp op, PatternRewriter& rewriter) const final
    {
        if (!op->hasAttr(ir::HeaderDeclAttrName) ||
            !op->hasAttr(ir::RawPointerAPIAttrName)) return failure();

        auto fnBodyOpIterator = op.front().without_terminator();
        if (!llvm::hasSingleElement(fnBodyOpIterator)) return failure();

        if (auto callOp = dyn_cast<CallOp>(fnBodyOpIterator.begin()))
        {
            auto calleeFnOp = dyn_cast_or_null<FuncOp>(SymbolTable::lookupNearestSymbolFrom(op, callOp.callee()));
            if (!calleeFnOp) return failure();

            auto calleeFnBodyOpIterator = calleeFnOp.front().back().getReverseIterator();
            assert(calleeFnBodyOpIterator->hasTrait<OpTrait::IsTerminator>());

            ++calleeFnBodyOpIterator;
            if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(*calleeFnBodyOpIterator))
            {
                auto launchedGPUFnOp = dyn_cast_or_null<gpu::GPUFuncOp>(SymbolTable::lookupNearestSymbolFrom(calleeFnOp, launchOp.kernel()));
                if (!launchedGPUFnOp) return failure();

                auto gpuTargetFuncName = op.getName().str() + "__gpu__";
                if (SymbolTable::lookupNearestSymbolFrom(launchedGPUFnOp, gpuTargetFuncName)) return failure();

                auto context = rewriter.getContext();
                auto execRuntimeAttr = vir::ExecutionRuntimeAttr::get(context, _target);
                auto execTargetAttr = vir::ExecutionTargetAttr::get(context, vir::ExecutionTarget::GPU);
                launchedGPUFnOp->setAttr(vir::ValueModuleOp::getExecRuntimeAttrName(), execRuntimeAttr);
                launchedGPUFnOp->setAttr(vir::ValueFuncOp::getExecTargetAttrName(), execTargetAttr);
                launchedGPUFnOp->setAttr(ir::HeaderDeclAttrName, rewriter.getUnitAttr());
                launchedGPUFnOp->setAttr(ir::RawPointerAPIAttrName, rewriter.getUnitAttr());

                launchedGPUFnOp.setName(gpuTargetFuncName);
                auto kernelSymAttr = launchOp.kernel();
                auto root = kernelSymAttr.getRootReference();
                launchOp.kernelAttr(rewriter.getSymbolRefAttr(root, rewriter.getSymbolRefAttr(gpuTargetFuncName)));

                rewriter.updateRootInPlace(op, [&] {
                    op->setAttr(vir::ValueModuleOp::getExecRuntimeAttrName(), execRuntimeAttr);
                });

                return success();
            }
        }

        return failure();
    }

private:
    vir::ExecutionRuntime _target;
};

struct ValueBarrierToSPIRVBarrierConversion final : public OpConversionPattern<vir::BarrierOp>
{
    ValueBarrierToSPIRVBarrierConversion(SPIRVTypeConverter& typeConverter, MLIRContext* context) :
        OpConversionPattern(typeConverter, context, kAcceraGPUPatternBenefit)
    {}

    LogicalResult matchAndRewrite(vir::BarrierOp op, ArrayRef<Value>, ConversionPatternRewriter& rewriter) const final
    {
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

struct ValueMFMALoadOpToRocDLConversion final : public OpConversionPattern<vir::MFMALoadOp>
{
    using OpConversionPattern<vir::MFMALoadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::MFMALoadOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter) const final
    {
        auto ctx = rewriter.getContext();
        auto loc = op.getLoc();
        vir::MFMALoadOp::Adaptor MFMALoadOpAdaptor(operands, op->getAttrDictionary());
        auto memref = MFMALoadOpAdaptor.memref();
        auto mfmaMatrixType = op.getMFMAMatrixType();
        auto mfmaMatrixOperand = mfmaMatrixType.getOperand();
        auto elementType = mfmaMatrixType.getElementType();

        auto leadingDim = mfmaMatrixType.getLeadingDim();
        if (leadingDim != 16) // 16x16x4
        {
            return rewriter.notifyMatchFailure(op, "unhandled matrix shape");
        }

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);

        const auto warpSize = 64;

        auto strideOffset = rewriter.getAffineSymbolExpr(0);
        auto threadIdxX = rewriter.getAffineSymbolExpr(0);
        auto threadIdxY = rewriter.getAffineSymbolExpr(1);
        auto blockDimX = rewriter.getAffineSymbolExpr(2);
        auto blockTid = threadIdxX + threadIdxY * blockDimX;
        auto warpTid = blockTid % warpSize;
        auto wmmaM = leadingDim;
        auto m = warpTid % wmmaM;
        auto ks = warpTid.floorDiv(wmmaM);

        auto vecSize = 4;
        auto vecTy = mlir::VectorType::get({ vecSize }, elementType);
        // For AOp load from the input memref with a column stride of 4
        //
        // for AOp this transformation is equivalent to:
        // float4 result;
        // memrefView = &memred[loadOperands]
        // for (int i = 0; i < 4; i++) {
        //    result[i] = memrefView[m, ks + 4*i];
        // }
        ////////////////////////////////////////
        // For BOp load from the input memref with a row stride of 4
        //
        // for BOp this transformation is equivalent to:
        // float4 result;
        // memrefView = &memred[loadOperands]
        // for (int i = 0; i < 4; i++) {
        //    result[i] = memrefView[ks + 4*i, m];
        // }
        //
        ////////////////////////////////////////
        // For COp load
        //
        // for COp this transformation is equivalent to:
        // float4 result;
        // memrefView = &memred[loadOperands]
        // for (int i = 0; i < 4; i++) {
        //    result[i] = memrefView[ks * 4 + i, m];
        // }
        //

        auto d0 = rewriter.getAffineDimExpr(0);
        auto d1 = rewriter.getAffineDimExpr(1);
        auto loadAffineMap = MFMALoadOpAdaptor.map().getValue(); // [d0, d1, d2, sa, sb]
        auto offsetAOpMap = AffineMap::get(2, 3, { d0 + m, d1 + ks }, ctx); // [d0, d1, sx, sy, sz]
        auto strideAOpMap = AffineMap::get(2, 1, { d0, d1 + strideOffset * 4 }, ctx); // [d0, d1, s0]
        auto offsetBOpMap = AffineMap::get(2, 3, { d0 + ks, d1 + m }, ctx); // [d0, d1, sx, sy, sz]
        auto strideBOpMap = AffineMap::get(2, 1, { d0 + strideOffset * 4, d1 }, ctx); // [d0, d1, s0]
        auto offsetCOpMap = AffineMap::get(2, 3, { d0 + ks * 4, d1 + m }, ctx); // [d0, d1, sx, sy, sz]
        auto strideCOpMap = AffineMap::get(2, 1, { d0 + strideOffset, d1 }, ctx); // [d0, d1, s0]
        auto matrixLayoutMap = ::llvm::StringSwitch<AffineMap>(mfmaMatrixOperand.str())
                                   .Case("AOp", strideAOpMap.compose(offsetAOpMap))
                                   .Case("BOp", strideBOpMap.compose(offsetBOpMap))
                                   .Case("COp", strideCOpMap.compose(offsetCOpMap))
                                   .Default(/*this is really an error */ AffineMap());
        auto composedMap = matrixLayoutMap.compose(loadAffineMap); // [d0, d1, d2, s0, sx, sy, sz, sa, sb]

        LLVM_DEBUG(llvm::dbgs() << "op: " << *op << "\n"
                                << "loadAffineMap: " << loadAffineMap << "\n"
                                << "offsetAOpMap: " << offsetAOpMap << "\n"
                                << "strideAOpMap: " << strideAOpMap << "\n"
                                << "offsetBOpMap: " << offsetBOpMap << "\n"
                                << "strideBOpMap: " << strideBOpMap << "\n"
                                << "offsetCOpMap: " << offsetCOpMap << "\n"
                                << "strideCOpMap: " << strideCOpMap << "\n"
                                << "matrixLayoutMap: " << matrixLayoutMap << "\n"
                                << "composedMap: " << composedMap << "\n"
                                << "simplify(composedMap): " << simplifyAffineMap(composedMap) << "\n");
        auto indices = MFMALoadOpAdaptor.indices();
        std::vector<Value> mapOperands;
        for (size_t i = 0; i < loadAffineMap.getNumDims(); i++)
        {
            mapOperands.push_back(indices[i]);
        }
        mapOperands.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        mapOperands.push_back(rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), "x"));
        mapOperands.push_back(rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), "y"));
        mapOperands.push_back(rewriter.create<gpu::BlockDimOp>(loc, rewriter.getIndexType(), "x"));
        for (size_t i = loadAffineMap.getNumDims(); i < loadAffineMap.getNumInputs(); i++)
        {
            mapOperands.push_back(indices[i]);
        }

        auto zero = rewriter.create<ConstantOp>(loc, elementType, rewriter.getZeroAttr(elementType));
        mlir::Value vec = rewriter.create<vector::BroadcastOp>(loc, vecTy, zero);

        auto i32Ty = rewriter.getIntegerType(32);
        auto loop = rewriter.replaceOpWithNewOp<AffineForOp>(op, 0, 4, 1, vec);
        auto loopBuilder = utilir::MakeBodyBuilder(loop);
        auto inductionVar = loop.getInductionVar();
        auto regionIterArg = loop.getRegionIterArgs()[0];
        auto laneIndex = loopBuilder.create<mlir::IndexCastOp>(loc, inductionVar, i32Ty);
        mapOperands[loadAffineMap.getNumDims()] = inductionVar; // we override the strideOffset symbol with the current index value

        LLVM_DEBUG(llvm::dbgs() << "mapOperands: ["
                                << "\n";
                   for (auto op
                        : mapOperands) {
                       llvm::dbgs() << "  " << op << "\n";
                   } llvm::dbgs()
                   << "]\n");

        auto mappedOperands = utilir::MultiDimAffineApply(loopBuilder, loc, composedMap, mapOperands);
        auto load = loopBuilder.create<memref::LoadOp>(loc, memref, mappedOperands);
        vec = loopBuilder.create<vector::InsertElementOp>(loc, load, regionIterArg, laneIndex);
        loopBuilder.create<AffineYieldOp>(loc, ValueRange{ vec });

        return success();
    }
};

struct ValueMFMAStoreOpToRocDLConversion final : public OpConversionPattern<vir::MFMAStoreOp>
{
    using OpConversionPattern<vir::MFMAStoreOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::MFMAStoreOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter) const final
    {
        auto ctx = rewriter.getContext();
        auto loc = op.getLoc();
        vir::MFMAStoreOp::Adaptor mfmaStoreOpAdaptor(operands, op->getAttrDictionary());
        auto value = mfmaStoreOpAdaptor.value();
        auto memref = mfmaStoreOpAdaptor.memref();
        auto indices = mfmaStoreOpAdaptor.indices();
        auto mfmaMatrixType = op.getMFMAMatrixType();

        auto leadingDim = mfmaMatrixType.getLeadingDim();
        if (leadingDim != 16) // 16x16x4
        {
            return rewriter.notifyMatchFailure(op, "unhandled matrix shape");
        }

        const auto warpSize = utilir::ResolveWarpSize(op).value();

        auto d0 = rewriter.getAffineDimExpr(0);
        auto d1 = rewriter.getAffineDimExpr(1);
        auto strideOffset = rewriter.getAffineSymbolExpr(0);
        auto threadIdxX = rewriter.getAffineSymbolExpr(1);
        auto threadIdxY = rewriter.getAffineSymbolExpr(2);
        auto blockDimX = rewriter.getAffineSymbolExpr(3);
        auto blockTid = threadIdxY * blockDimX + threadIdxX;
        auto warpTid = blockTid % warpSize;
        auto wmmaM = leadingDim;
        auto m = warpTid % wmmaM;
        auto ks = warpTid.floorDiv(wmmaM);
        auto offsetMap = AffineMap::get(2, 4, { d0 + ks * 4 + strideOffset, d1 + m }, ctx);

        auto storeAffineMap = op.getAffineMap();
        auto composedMap = offsetMap.compose(storeAffineMap);

        std::vector<Value> mapOperands;
        for (size_t i = 0; i < storeAffineMap.getNumDims(); i++)
        {
            mapOperands.push_back(indices[i]);
        }
        mapOperands.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        mapOperands.push_back(rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), "x"));
        mapOperands.push_back(rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), "y"));
        mapOperands.push_back(rewriter.create<gpu::BlockDimOp>(loc, rewriter.getIndexType(), "x"));
        for (size_t i = storeAffineMap.getNumDims(); i < storeAffineMap.getNumInputs(); i++)
        {
            mapOperands.push_back(indices[i]);
        }

        auto i32Ty = rewriter.getIntegerType(32);
        auto loop = rewriter.replaceOpWithNewOp<AffineForOp>(op, 0, 4, 1);
        auto loopBuilder = utilir::MakeBodyBuilder(loop);
        auto inductionVar = loop.getInductionVar();
        auto laneIndex = loopBuilder.create<mlir::IndexCastOp>(loc, inductionVar, i32Ty);
        mapOperands[storeAffineMap.getNumDims()] = inductionVar; // we override the strideOffset symbol with the current index value
        auto mappedOperands = utilir::MultiDimAffineApply(loopBuilder, loc, composedMap, mapOperands);
        auto elem = loopBuilder.create<vector::ExtractElementOp>(loc, value, laneIndex);
        loopBuilder.create<memref::StoreOp>(loc, elem, memref, mappedOperands);

        return success();
    }
};

struct ValueMFMAConstantOpToRocDLConversion final : public OpRewritePattern<vir::MFMAConstantOp>
{
    using OpRewritePattern<vir::MFMAConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(vir::MFMAConstantOp op, PatternRewriter& rewriter) const final
    {

        auto mfmaType = op.getMFMAMatrixType();

        auto leadingDim = mfmaType.getLeadingDim();
        if (leadingDim != 16) // 16x16x4
        {
            return rewriter.notifyMatchFailure(op, "unhandled matrix shape");
        }

        auto vecSize = 4;
        auto vecTy = VectorType::get({ vecSize }, mfmaType.getElementType());

        rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, vecTy, op.value());

        return success();
    }
};

struct ValueMFMAComputeToRocDLConversion final : public OpConversionPattern<vir::MFMAComputeOp>
{
    using OpConversionPattern<vir::MFMAComputeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::MFMAComputeOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter) const final
    {
        using namespace accera::utilities;
        auto loc = op.getLoc();
        vir::MFMAComputeOp::Adaptor mfmaComputeMatrixOpAdaptor(operands, op->getAttrDictionary());
        auto opA = mfmaComputeMatrixOpAdaptor.opA();
        auto opB = mfmaComputeMatrixOpAdaptor.opB();
        auto opC = mfmaComputeMatrixOpAdaptor.opC();
        if (!opA.getType().isa<VectorType>())
        {
            return rewriter.notifyMatchFailure(op, "expecting a vector type for OpA");
        }
        if (!opB.getType().isa<VectorType>())
        {
            return rewriter.notifyMatchFailure(op, "expecting a vector type for OpB");
        }
        if (!opC.getType().isa<VectorType>())
        {
            return rewriter.notifyMatchFailure(op, "expecting a vector type for OpC");
        }

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);

        auto i32Ty = rewriter.getIntegerType(32);
        Value zero = rewriter.create<mlir::ConstantOp>(loc, i32Ty, rewriter.getZeroAttr(i32Ty));

        auto destVectorType = opC.getType().cast<VectorType>();
        auto numElems = destVectorType.getShape()[0];
        if (numElems != 4)
        {
            return rewriter.notifyMatchFailure(op, "expecting a 16x16 matrix type for OpC");
        }

        if (numElems == 4)
        {
            auto numIterations = 4;
            auto result = opC;
            //
            // equivalent to:
            // result = opC;
            // for (int i = 0; i < 4; i++) {
            //    result = mfma_f32_16x16x4f32(opA[i], opB[i], result);
            // }
            //
            auto loop = rewriter.replaceOpWithNewOp<AffineForOp>(op, 0, numIterations, 1, result);
            auto loopBuilder = utilir::MakeBodyBuilder(loop);
            auto inductionVar = loop.getInductionVar();
            auto regionIterArg = loop.getRegionIterArgs()[0];
            auto laneIndex = loopBuilder.create<mlir::IndexCastOp>(loc, inductionVar, i32Ty);
            auto elemA = loopBuilder.create<vector::ExtractElementOp>(loc, opA, laneIndex);
            auto elemB = loopBuilder.create<vector::ExtractElementOp>(loc, opB, laneIndex);
            auto mfmaOp = loopBuilder.create<ROCDL::mfma_f32_16x16x4f32>(loc, result.getType(), ValueRange{ elemA, elemB, regionIterArg, zero, zero, zero });
            loopBuilder.create<AffineYieldOp>(loc, ValueRange{ mfmaOp });
        }
        else
        {
            return rewriter.notifyMatchFailure(op, "Unsupported op size.");
        }
        return success();
    }
};

struct ValueMFMAStoreOpToGPUConversion final : public OpConversionPattern<vir::MFMAStoreOp>
{
    using OpConversionPattern<vir::MFMAStoreOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::MFMAStoreOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter) const final
    {
        return success();
    }
};

struct ValueMFMALoadOpToGPUConversion final : public OpConversionPattern<vir::MFMALoadOp>
{
    using OpConversionPattern<vir::MFMALoadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::MFMALoadOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter) const final
    {
        rewriter.eraseOp(op);
        return success();
    }
};

struct ValueMFMAConstantOpToGPUConversion final : public OpConversionPattern<vir::MFMAConstantOp>
{
    using OpConversionPattern<vir::MFMAConstantOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::MFMAConstantOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter) const final
    {
        rewriter.eraseOp(op);
        return success();
    }
};
struct ValueMFMAComputeToGPUConversion final : public OpConversionPattern<vir::MFMAComputeOp>
{
    using OpConversionPattern<vir::MFMAComputeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::MFMAComputeOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter) const final
    {
        rewriter.replaceOpWithNewOp<gpu::SubgroupMmaComputeOp>(op, operands[2].getType(), operands, op->getAttrs());
        return success();
    }
};

struct ResolveBlockDimPattern final : public OpRewritePattern<gpu::BlockDimOp>
{
    using OpRewritePattern<gpu::BlockDimOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(gpu::BlockDimOp op, PatternRewriter& rewriter) const final
    {
        auto gpuFunc = op->getParentOfType<gpu::GPUFuncOp>();
        if (!gpuFunc)
        {
            return failure();
        }
        auto blockSizeAttr = gpuFunc->getAttrOfType<ArrayAttr>("blockSize");
        auto blockDimIdx = dimIndexToInteger(op.dimension());
        if (!blockSizeAttr || blockDimIdx == -1)
        {
            return failure();
        }
        auto val = blockSizeAttr.getValue()[blockDimIdx].cast<IntegerAttr>().getInt();
        rewriter.replaceOpWithNewOp<mlir::ConstantIndexOp>(op, val);
        return success();
    }
};
struct ConditionalBarrierHoistingPattern : public OpRewritePattern<vir::BarrierOp>
{
    using OpRewritePattern<vir::BarrierOp>::OpRewritePattern;

    mlir::Operation* GetAncestorIfOp(vir::BarrierOp op) const
    {
        mlir::Operation* parentAffineIfOp = utilir::GetHighestAncestorOfType<mlir::AffineIfOp>(op);
        mlir::Operation* parentSCFIfOp = utilir::GetHighestAncestorOfType<mlir::scf::IfOp>(op);

        if (parentAffineIfOp && parentSCFIfOp)
        {
            // There are both affine.if and scf.if parents, so return the highest ancestor between the two
            return parentAffineIfOp->isAncestor(parentSCFIfOp) ? parentAffineIfOp : parentSCFIfOp;
        }
        else
        {
            // Return whichever is nonnull, or return nullptr if both are null
            return parentAffineIfOp == nullptr ? parentSCFIfOp : parentAffineIfOp;
        }
    }

    LogicalResult matchAndRewrite(vir::BarrierOp op, PatternRewriter& rewriter) const final
    {
        // Hoist barrier ops outside of any affine.if or scf.if conditional blocks they are contained inside of

        // As a simple hoist, remove all barriers inside of the conditional and place a barrier before and after the conditional block
        // TODO : instead of hoisting this way, split conditional blocks at the barriers to keep the same relative

        // Get the highest level affine.if or scf.if op that contains this barrier, if one exists
        if (auto ancestorIfOp = GetAncestorIfOp(op))
        {
            // This barrier is contained within a conditional, so clone it before and after the conditional then erase it
            rewriter.setInsertionPoint(ancestorIfOp);
            rewriter.clone(*(op.getOperation()));
            rewriter.setInsertionPointAfter(ancestorIfOp);
            rewriter.clone(*(op.getOperation()));

            rewriter.eraseOp(op);
        }

        return success();
    }
};

struct AcceraToSPIRVPass : public accera::transforms::ConvertAcceraToSPIRVBase<AcceraToSPIRVPass>
{
    void runOnOperation() final
    {
        ModuleOp module = getOperation();

        if (!hasRuntimeTarget<vir::ExecutionRuntime::VULKAN>(module))
        {
            return;
        }

        MLIRContext* context = &getContext();

        {
            RewritePatternSet patterns(context);
            populateGPUSimplificationPatterns(patterns);
            (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
        }

        // cf mlir/lib/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.cpp -- GPUToSPIRVPass::runOnOperation
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
}; // namespace

struct AcceraToROCDLPass : public accera::transforms::ConvertAcceraToROCDLBase<AcceraToROCDLPass>
{
    void runOnOperation() final
    {
        MLIRContext* context = &getContext();
        auto module = getOperation();
        ConversionTarget target(*context);

        if (!hasRuntimeTarget<vir::ExecutionRuntime::ROCM>(module))
        {
            return;
        }

        target.addLegalOp<ModuleOp>();
        target.addIllegalOp<
            vir::EarlyReturnOp,
            vir::MFMAComputeOp,
            vir::MFMAConstantOp,
            vir::MFMALoadOp,
            vir::MFMAStoreOp,
            vir::BarrierOp,
            gpu::BlockDimOp
            >();
        target.addLegalDialect<
            mlir::AffineDialect,
            mlir::BuiltinDialect,
            mlir::gpu::GPUDialect,
            mlir::memref::MemRefDialect,
            mlir::ROCDL::ROCDLDialect,
            mlir::scf::SCFDialect,
            mlir::StandardOpsDialect,
            mlir::vector::VectorDialect,
            omp::OpenMPDialect,
            vir::ValueDialect>();

        {
            RewritePatternSet patterns(context);
            populateGPUSimplificationPatterns(patterns);
            (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
        }
        {
            RewritePatternSet patterns(context);
            patterns.insert<CreateDeviceFuncLauncherPairPattern>(vir::ExecutionRuntime::ROCM, context);
            (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
        }
        {
            RewritePatternSet patterns(context);
            populateAcceraToROCDLPatterns(patterns);
            if (failed(applyFullConversion(module, target, std::move(patterns))))
                signalPassFailure();
        }
    }
};

struct AcceraToNVVMPass : public accera::transforms::ConvertAcceraToNVVMBase<AcceraToNVVMPass>
{
    void runOnOperation() final
    {

        MLIRContext* context = &getContext();
        auto module = getOperation();
        ConversionTarget target(*context);

        if (!hasRuntimeTarget<vir::ExecutionRuntime::CUDA>(module))
        {
            return;
        }

        target.addLegalOp<ModuleOp>();
        target.addIllegalOp<
            vir::EarlyReturnOp,
            vir::MFMAComputeOp,
            vir::MFMAConstantOp,
            vir::MFMALoadOp,
            vir::MFMAStoreOp,
            vir::BarrierOp,
            gpu::BlockDimOp
            >();
        target.addLegalDialect<
            mlir::AffineDialect,
            mlir::BuiltinDialect,
            mlir::gpu::GPUDialect,
            mlir::memref::MemRefDialect,
            mlir::NVVM::NVVMDialect,
            mlir::scf::SCFDialect,
            mlir::StandardOpsDialect,
            mlir::vector::VectorDialect,
            omp::OpenMPDialect,
            vir::ValueDialect>();

        {
            RewritePatternSet patterns(context);
            populateGPUSimplificationPatterns(patterns);
            (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
        }
        {
            RewritePatternSet patterns(context);
            patterns.insert<CreateDeviceFuncLauncherPairPattern>(vir::ExecutionRuntime::CUDA, context);
            (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
        }
        {
            RewritePatternSet patterns(context);
            populateAcceraToNVVMPatterns(patterns);

            if (failed(applyFullConversion(module, target, std::move(patterns))))
                signalPassFailure();
        }
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
        ResolveBlockDimPattern,
        EarlyReturnToGPUReturnPattern,
        ValueBarrierToGPUBarrierConversion,
        ValueMFMALoadOpToRocDLConversion,
        ValueMFMAComputeToRocDLConversion,
        ValueMFMAStoreOpToRocDLConversion,
        ValueMFMAConstantOpToRocDLConversion>(patterns.getContext());
}

void populateAcceraToNVVMPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<
        ResolveBlockDimPattern,
        EarlyReturnToGPUReturnPattern,
        ValueBarrierToGPUBarrierConversion,
        ValueMFMALoadOpToGPUConversion,
        ValueMFMAComputeToGPUConversion,
        ValueMFMAStoreOpToGPUConversion,
        ValueMFMAConstantOpToGPUConversion>(patterns.getContext());
}

void populateGPUSimplificationPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<ConditionalBarrierHoistingPattern>(patterns.getContext());
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
    case ExecutionRuntime::DEFAULT:
        // TODO: default gpu runtime is rocm
        [[fallthrough]];
    case ExecutionRuntime::ROCM:
        return createAcceraToROCDLPass();
    case ExecutionRuntime::CUDA:
        return createAcceraToNVVMPass();
    case ExecutionRuntime::VULKAN:
        return createAcceraToSPIRVPass();
    case ExecutionRuntime::NONE:
        [[fallthrough]];
    case ExecutionRuntime::OPENMP:
        [[fallthrough]];
    default:
        return {};
    }
}

} // namespace accera::transforms
