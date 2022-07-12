////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Abdul Dakkak, Kern Handa, Captain Jack Sparrow
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gpu/AcceraToGPUPass.h"

#include "AcceraPasses.h"
#include "ir/include/value/ValueDialect.h"
#include "ir/include/value/ValueEnums.h"
#include "ir/include/value/ValueMMAOp.h"

#include <ir/include/IRUtil.h>

#include <utilities/include/Exception.h>

#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
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

#include <functional>
#include <numeric>
#include <optional>

#define DEBUG_TYPE "accera-to-gpu"

using namespace mlir;
using accera::transforms::populateAcceraToNVVMPatterns;
using accera::transforms::populateAcceraToROCDLPatterns;
using accera::transforms::populateAcceraToSPIRVPatterns;
using accera::transforms::populateGPUSimplificationPatterns;
using accera::transforms::populateGPUToROCDLPatterns;

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

GPUIndexDimension dimIndexToInteger(llvm::StringRef dim)
{
    return ::llvm::StringSwitch<GPUIndexDimension>(dim)
        .Case("x", GPUIndexDimension::X)
        .Case("y", GPUIndexDimension::Y)
        .Case("z", GPUIndexDimension::Z)
        .Default(GPUIndexDimension::Invalid);
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
            if (utilir::ResolveExecutionRuntime(op).value() == vir::ExecutionRuntime::ROCM)
            {
                rewriter.replaceOpWithNewOp<ROCDL::BarrierOp>(op);
            }
            else
            {
                rewriter.replaceOpWithNewOp<gpu::BarrierOp>(op);
            }
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

template <typename MFMALoadStoreOpTy, typename PostMFMAOp>
void MFMALoadStoreAccumulator(MFMALoadStoreOpTy op, ConversionPatternRewriter& rewriter, const vir::MMAOp& mfmaMatrixType, const int64_t vecSize, AffineMap affineMap, ValueRange indices, PostMFMAOp&& doAfterFn, ValueRange iterArgs = llvm::None)
{
    typename MFMALoadStoreOpTy::Adaptor adaptor(op);

    auto loc = op->getLoc();

    // Get the upper left corner of the tile that should be accessed
    auto accessPosMap = op.map();
    auto accessPosIndices = adaptor.indices();
    std::vector<mlir::Value> accessPosIndicesVec(accessPosIndices.begin(), accessPosIndices.end());
    auto upperLeftCornerPos = utilir::MultiDimAffineApply(rewriter, loc, accessPosMap, accessPosIndicesVec);
    const int externalIndices = upperLeftCornerPos.size() - 2; // the last 2 dimensions are for indexing into the matrix itself

    // Each thread in the MFMA has a disjoint set of elements it will contain the results for
    // Create the offsets for each of these sets
    auto [warpSizeX, warpSizeY] = utilir::ResolveWarpSize(utilir::ResolveExecutionRuntime(op).value()).value();
    const auto warpSize = warpSizeX * warpSizeY;
    constexpr auto subGroupSize = 4; // How many elements are in each chunk - each chunk is a 4x1 region of output
    const auto numBlocks = mfmaMatrixType.getNumBlocks();
    const auto leadingDim = mfmaMatrixType.getM();
    const auto blockWidth = leadingDim / numBlocks;
    const auto warpStride = warpSize / blockWidth;
    const auto rowsPerSet = warpStride * subGroupSize;
    const auto setsPerCol = blockWidth / rowsPerSet;

    // Get per-thread computation handles
    auto upperLeftCornerRowDim = rewriter.getAffineDimExpr(0);
    auto upperLeftCornerColDim = rewriter.getAffineDimExpr(1);
    auto inductionVarDim = rewriter.getAffineDimExpr(2);
    auto warpTidSym = rewriter.getAffineSymbolExpr(0);
    auto staticOffsetIdxSym = rewriter.getAffineSymbolExpr(1);

    auto warpTidVal = utilir::GetCurrentGPUWarpThreadID(rewriter, loc);

    auto loop = rewriter.replaceOpWithNewOp<mlir::AffineForOp>(op, 0, vecSize, 1, iterArgs);
    mlir::OpBuilder loopBuilder = utilir::MakeBodyBuilder(loop); // using auto for the type here isn't compiling on gcc...?
    auto inductionVar = loop.getInductionVar();

    mlir::AffineExpr rowOffsetExpr, colOffsetExpr;
    std::vector<mlir::Value> accessDims(upperLeftCornerPos);
    std::vector<mlir::Value> accessSyms;

    if (SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(loop, vir::MFMAThreadBufferMapName))
    {
        // TODO : can we just create a pass that computes the static offset values from scratch at lowering-time?

        const auto globalBufferType = mfmaMatrixType.GetMFMAThreadOffsetMapType(rewriter.getIntegerType(8)).first;
        mlir::Value globalRef = loopBuilder.create<memref::GetGlobalOp>(loc, globalBufferType, vir::MFMAThreadBufferMapName);

        // Since we have found the symbol this means we need to optimize from the precomputed index maps
        auto offsetMapSize = mfmaMatrixType.getOffsetMapSize();
        auto inductionVarSym_d0 = rewriter.getAffineDimExpr(0); // the induction var is the only dim for this map, so it's in position 0
        auto rowIdx0 = inductionVarSym_d0 % offsetMapSize[0];
        auto subGroupId = warpTidSym.floorDiv(blockWidth);
        auto mfmaPrecompOffsetMap = AffineMap::get(1, 1, { rowIdx0, subGroupId }, rewriter.getContext());
        std::vector<Value> mfmaPrecompOffsetMapOperands{ inductionVar, warpTidVal };
        auto mfmaPrecompOffsetOperands = utilir::MultiDimAffineApply(loopBuilder, loc, mfmaPrecompOffsetMap, mfmaPrecompOffsetMapOperands);

        auto rowOff = loopBuilder.create<memref::LoadOp>(loc, globalRef, mfmaPrecompOffsetOperands);
        auto staticRowOffset = loopBuilder.create<IndexCastOp>(loc, rowOff, rewriter.getIndexType());

        // The static offsets are offsets from the upper left corner of the sub-warp "block" tile (not the GPU logical block tile but the sub-warp computation block for larger MFMAs)
        // Get warp upper left corner memory position
        auto tileRows = mfmaMatrixType.getM();
        auto tileCols = mfmaMatrixType.getN();
        auto blockRowCount = tileRows / numBlocks;
        std::vector<mlir::AffineExpr> warpBlockUpperLeftCornerExprs = { upperLeftCornerRowDim - (upperLeftCornerRowDim % blockRowCount),
                                                                        upperLeftCornerColDim - (upperLeftCornerColDim % tileCols) };

        auto warpBlockUpperLeftCornerMap = mlir::AffineMap::get(2, 0, warpBlockUpperLeftCornerExprs, rewriter.getContext());
        std::vector<mlir::Value> operands = { upperLeftCornerPos.end()[-2], upperLeftCornerPos.end()[-1] };
        auto warpBlockUpperLeftCornerPos = utilir::MultiDimAffineApply(loopBuilder, loc, warpBlockUpperLeftCornerMap, operands);

        rowOffsetExpr = staticOffsetIdxSym;
        colOffsetExpr = inductionVarDim.floorDiv(offsetMapSize[0]) * vecSize;

        accessDims[externalIndices] = warpBlockUpperLeftCornerPos[0]; // Overwrite using the statically computed row index
        accessSyms.push_back(staticRowOffset);
    }
    else
    {
        // offset based on which 4x1 subtile we're looking at
        const auto itemGroup = inductionVarDim.floorDiv(subGroupSize);
        const auto itemOffset = inductionVarDim % subGroupSize;
        const auto itemGroupRowOffset = (itemGroup % setsPerCol) * rowsPerSet;
        const auto itemGroupColOffset = itemGroup.floorDiv(setsPerCol) * blockWidth;

        rowOffsetExpr = itemGroupRowOffset + itemOffset;
        colOffsetExpr = itemGroupColOffset;

        accessSyms.push_back(rewriter.create<mlir::ConstantIndexOp>(loc, 0)); // static offset index
    }
    accessDims.push_back(inductionVar);
    accessDims.push_back(warpTidVal);

    // Concatenate { accessDims..., accessSyms... }
    auto accessOperands = accessDims;
    accessOperands.insert(accessOperands.end(), accessSyms.begin(), accessSyms.end());

    std::vector<mlir::AffineExpr> offsetExprs = { upperLeftCornerRowDim + rowOffsetExpr, upperLeftCornerColDim + colOffsetExpr };
    auto fullOffsetMap = mlir::AffineMap::get(3, 2, offsetExprs, rewriter.getContext());
    auto mfmaExternalDimsMap = mlir::AffineMap::getMultiDimIdentityMap(externalIndices, rewriter.getContext());
    auto fullMatrixAccessMap = utilir::ConcatenateAndShiftAffineDimsAndMaps(rewriter, mfmaExternalDimsMap, fullOffsetMap);
    auto accessPos = utilir::MultiDimAffineApply(loopBuilder, loc, fullMatrixAccessMap, accessOperands);

    doAfterFn(loc, loop, loopBuilder, accessPos);
}

LogicalResult MFMALoadAccumulator(vir::MMALoadSyncOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter)
{
    const vir::MMAOp mfmaOpType{ static_cast<vir::MMAShape>(op.mmaShapeType()) };

    vir::MMALoadSyncOp::Adaptor MMALoadSyncOpAdaptor(operands, op->getAttrDictionary());
    auto memref = MMALoadSyncOpAdaptor.memref();

    auto loc = op.getLoc();
    auto memRefType = op.result().getType().cast<MemRefType>();
    auto memRefShape = memRefType.getShape();
    const auto vecSize = std::accumulate(memRefShape.begin(), memRefShape.end(), 1, std::multiplies<int64_t>());
    auto outputType = memRefType.getElementType();

    // TODO : if we're loading from a private memory buffer, don't create a new buffer here
    mlir::Value vec = rewriter.create<memref::AllocOp>(loc, memRefType);

    MFMALoadStoreAccumulator(
        op, rewriter, mfmaOpType, vecSize, MMALoadSyncOpAdaptor.map().getValue(), MMALoadSyncOpAdaptor.indices(), [&](Location& loc, AffineForOp& loop, OpBuilder& loopBuilder, ValueRange mappedOperands) {
            auto load = loopBuilder.create<memref::LoadOp>(loc, memref, mappedOperands);
            auto destVec = loop.getRegionIterArgs()[0];
            auto inputType = op.memref().getType().cast<MemRefType>().getElementType();
            if (outputType.isF32() && (inputType.isF16() || inputType.isBF16()))
            {
                auto castedElem = loopBuilder.create<mlir::FPExtOp>(loc, load, rewriter.getF32Type());
                loopBuilder.create<memref::StoreOp>(loc, castedElem, destVec, loop.getInductionVar());
            }
            else if (outputType.isInteger(32) && (inputType.isInteger(16) || inputType.isInteger(8)))
            {
                auto castedElem = loopBuilder.create<SignExtendIOp>(loc, load, rewriter.getI32Type());
                loopBuilder.create<memref::StoreOp>(loc, castedElem, destVec, loop.getInductionVar());
            }
            else
            {
                loopBuilder.create<memref::StoreOp>(loc, load, destVec, loop.getInductionVar());
            }
            loopBuilder.create<AffineYieldOp>(loc, destVec);
        },
        vec);

    return success();
}

struct ValueMMALoadSyncOpToRocDLConversion final : public OpConversionPattern<vir::MMALoadSyncOp>
{
    using OpConversionPattern<vir::MMALoadSyncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::MMALoadSyncOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter) const final
    {
        const auto operandType = static_cast<vir::MMAOperandType>(op.operandType());
        if (operandType == vir::MMAOperandType::Acc)
        {
            return MFMALoadAccumulator(op, operands, rewriter);
        }

        const vir::MMAOp mfmaOpType{ static_cast<vir::MMAShape>(op.mmaShapeType()) };

        vir::MMALoadSyncOp::Adaptor adaptor(op);
        auto memref = adaptor.memref();

        auto loc = op->getLoc();
        auto memRefType = op.result().getType().cast<MemRefType>();
        auto memRefShape = memRefType.getShape();
        const auto vecSize = std::accumulate(memRefShape.begin(), memRefShape.end(), 1, std::multiplies<int64_t>());
        mlir::Value vec = rewriter.create<memref::AllocOp>(loc, memRefType);

        // Get the upper left corner of the tile that should be accessed
        auto accessPosMap = op.map();
        auto accessPosIndices = adaptor.indices();
        std::vector<mlir::Value> accessPosIndicesVec(accessPosIndices.begin(), accessPosIndices.end());
        auto upperLeftCornerPos = utilir::MultiDimAffineApply(rewriter, loc, accessPosMap, accessPosIndicesVec);
        const int externalIndices = upperLeftCornerPos.size() - 2; // the last 2 dimensions are for indexing into the matrix itself

        // Constants for the offsettings expressions
        auto [warpSizeX, warpSizeY] = utilir::ResolveWarpSize(utilir::ResolveExecutionRuntime(op).value()).value();
        const auto warpSize = warpSizeX * warpSizeY;
        const auto leadingDim = mfmaOpType.getM();
        const auto warpStride = warpSize / leadingDim;

        // Get per-thread computation handles
        auto inductionVarDim = rewriter.getAffineDimExpr(0);
        auto warpTidSym = rewriter.getAffineSymbolExpr(0);

        mlir::AffineExpr fullRowOffsetExpr, fullColOffsetExpr;
        auto subGroupId = warpTidSym.floorDiv(leadingDim);
        auto ijOffsetExpr = warpTidSym % leadingDim;
        auto kOffsetExpr = subGroupId + inductionVarDim * warpStride;
        switch (operandType)
        {
        case vir::MMAOperandType::A:
            fullRowOffsetExpr = ijOffsetExpr;
            fullColOffsetExpr = kOffsetExpr;
            break;
        case vir::MMAOperandType::B:
            fullRowOffsetExpr = kOffsetExpr;
            fullColOffsetExpr = ijOffsetExpr;
            break;
        default:
            return failure();
        }

        auto warpTidVal = utilir::GetCurrentGPUWarpThreadID(rewriter, loc);

        auto loop = rewriter.replaceOpWithNewOp<mlir::AffineForOp>(op, 0, vecSize, 1, vec);
        auto loopBuilder = utilir::MakeBodyBuilder(loop);
        auto inductionVar = loop.getInductionVar();
        auto destVec = loop.getRegionIterArgs()[0];

        // 1 dim for the induction var, 1 symbol for the warp thread ID
        auto warpTileOffsetMap = mlir::AffineMap::get(1, 1, { fullRowOffsetExpr, fullColOffsetExpr }, rewriter.getContext());

        // Shift the dims in warpTileOffsetMap by the mfma external dim count
        auto mfmaExternalDimsMap = mlir::AffineMap::getMultiDimIdentityMap(externalIndices, loopBuilder.getContext());
        warpTileOffsetMap = utilir::ConcatenateAndShiftAffineDimsAndMaps(loopBuilder, mfmaExternalDimsMap, warpTileOffsetMap);

        // The buffer being accessed may hold the tile data in a physical order different from the logical order,
        // so use the tileAccessMap map attr if it exists to map how offsets within the logical tile translate
        // to offsets within the physical buffer layout
        auto tileAccessMapOpt = op.tileAccessMap();
        if (tileAccessMapOpt.hasValue())
        {
            auto tileAccessMap = tileAccessMapOpt.getValue();
            warpTileOffsetMap = tileAccessMap.compose(warpTileOffsetMap);
        }

        mlir::Value zeroIndex = rewriter.create<mlir::ConstantIndexOp>(loc, 0);
        std::vector<mlir::Value> warpTileOffsetOperands(upperLeftCornerPos.size(), zeroIndex);
        warpTileOffsetOperands[warpTileOffsetOperands.size() - 2] = inductionVar;
        warpTileOffsetOperands[warpTileOffsetOperands.size() - 1] = warpTidVal;
        auto warpTileOffsetPhysicalPos = utilir::MultiDimAffineApply(loopBuilder, loc, warpTileOffsetMap, warpTileOffsetOperands);

        // Now that we have the offsets within the physical buffer for this thread in the warp, add that to the physical upper left corner position
        // To get the full position in the memref

        std::vector<mlir::AffineExpr> matrixAccessExprs;
        for (size_t i = 0; i < upperLeftCornerPos.size(); ++i)
        {
            matrixAccessExprs.emplace_back(loopBuilder.getAffineDimExpr(i) + loopBuilder.getAffineDimExpr(i + upperLeftCornerPos.size()));
        }

        auto fullMatrixAccessMap = mlir::AffineMap::get(2 * upperLeftCornerPos.size(), 0, matrixAccessExprs, loopBuilder.getContext());

        std::vector<mlir::Value> accessOperands(upperLeftCornerPos);
        accessOperands.insert(accessOperands.end(), warpTileOffsetPhysicalPos.begin(), warpTileOffsetPhysicalPos.end());

        auto accessPos = utilir::MultiDimAffineApply(loopBuilder, loc, fullMatrixAccessMap, accessOperands);

        auto load = loopBuilder.create<memref::LoadOp>(loc, memref, accessPos);
        loopBuilder.create<memref::StoreOp>(loc, load, destVec, inductionVar);
        loopBuilder.create<AffineYieldOp>(loc, destVec);

        return success();
    }
};

struct ValueMMAStoreSyncOpToRocDLConversion final : public OpConversionPattern<vir::MMAStoreSyncOp>
{
    using OpConversionPattern<vir::MMAStoreSyncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::MMAStoreSyncOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter) const final
    {
        const vir::MMAOp mfmaOpType{ static_cast<vir::MMAShape>(op.mmaShapeType()) };
        vir::MMAStoreSyncOp::Adaptor MMAStoreSyncOpAdaptor(operands, op->getAttrDictionary());
        auto src = MMAStoreSyncOpAdaptor.src();
        auto memref = MMAStoreSyncOpAdaptor.memref();

        auto srcMemRefType = src.getType().cast<MemRefType>();
        auto srcMemRefShape = srcMemRefType.getShape();
        const auto vecSize = std::accumulate(srcMemRefShape.begin(), srcMemRefShape.end(), 1, std::multiplies<int64_t>());
        MFMALoadStoreAccumulator(op, rewriter, mfmaOpType, vecSize, op.getAffineMap(), MMAStoreSyncOpAdaptor.indices(), [&](Location& loc, AffineForOp& loop, OpBuilder& loopBuilder, ValueRange mappedOperands) {
            auto elem = loopBuilder.create<memref::LoadOp>(loc, src, loop.getInductionVar());

            // Check if we need to cast before storing back the result
            auto srcType = srcMemRefType.getElementType();
            auto dstType = memref.getType().cast<MemRefType>().getElementType();
            if (srcType.isF32() && (dstType.isF16() || dstType.isBF16()))
            {
                auto castedElem = loopBuilder.create<mlir::FPTruncOp>(loc, elem, dstType);
                loopBuilder.create<memref::StoreOp>(loc, castedElem, memref, mappedOperands);
            }
            else if (srcType.isInteger(32) && (dstType.isInteger(16) || dstType.isInteger(8)))
            {
                auto castedElem = loopBuilder.create<mlir::TruncateIOp>(loc, elem, dstType);
                loopBuilder.create<memref::StoreOp>(loc, castedElem, memref, mappedOperands);
            }
            else
            {
                loopBuilder.create<memref::StoreOp>(loc, elem, memref, mappedOperands);
            }
        });

        return success();
    }
};

struct ValueMMAFillSyncOpToRocDLConversion final : public OpRewritePattern<vir::MMAFillSyncOp>
{
    using OpRewritePattern<vir::MMAFillSyncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(vir::MMAFillSyncOp op, PatternRewriter& rewriter) const final
    {
        auto loc = op.getLoc();
        auto memRefType = op.result().getType().cast<MemRefType>();
        auto memRefShape = memRefType.getShape();
        const auto vecSize = std::accumulate(memRefShape.begin(), memRefShape.end(), 1, std::multiplies<int64_t>());
        mlir::Value vec = rewriter.create<memref::AllocOp>(loc, memRefType);
        auto loop = rewriter.replaceOpWithNewOp<AffineForOp>(op, 0, vecSize, 1, vec);
        auto loopBuilder = utilir::MakeBodyBuilder(loop);
        auto inductionVar = loop.getInductionVar();
        auto destVec = loop.getRegionIterArgs()[0];
        loopBuilder.create<memref::StoreOp>(loc, op.value(), destVec, inductionVar);
        loopBuilder.create<AffineYieldOp>(loc, destVec);

        return success();
    }
};

struct ValueMFMAComputeToRocDLConversion final : public OpConversionPattern<vir::MMAComputeSyncOp>
{
    using OpConversionPattern<vir::MMAComputeSyncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(vir::MMAComputeSyncOp op,
                                  ArrayRef<mlir::Value> operands,
                                  ConversionPatternRewriter& rewriter) const final
    {
        using namespace accera::utilities;
        using namespace accera::ir::value;
        auto loc = op.getLoc();
        vir::MMAComputeSyncOp::Adaptor mfmaComputeMatrixOpAdaptor(operands, op->getAttrDictionary());
        auto opA = mfmaComputeMatrixOpAdaptor.opA();
        auto opB = mfmaComputeMatrixOpAdaptor.opB();
        auto opC = mfmaComputeMatrixOpAdaptor.opC();
        auto i32Ty = rewriter.getI32Type();
        auto cbsz = rewriter.create<ConstantOp>(loc, i32Ty, mfmaComputeMatrixOpAdaptor.cbsz());
        auto abid = rewriter.create<ConstantOp>(loc, i32Ty, mfmaComputeMatrixOpAdaptor.abid());
        auto blgp = rewriter.create<ConstantOp>(loc, i32Ty, mfmaComputeMatrixOpAdaptor.blgp());
        if (!opA.getType().isa<MemRefType>())
        {
            return rewriter.notifyMatchFailure(op, "expecting a vector type for OpA");
        }
        if (!opB.getType().isa<MemRefType>())
        {
            return rewriter.notifyMatchFailure(op, "expecting a vector type for OpB");
        }
        if (!opC.getType().isa<MemRefType>())
        {
            return rewriter.notifyMatchFailure(op, "expecting a vector type for OpC");
        }

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);

        vir::MMAShape mfmaShape{ static_cast<vir::MMAShape>(op.mmaShapeType()) };
        vir::MMAOp mfmaType{ mfmaShape };

        const auto outputMemrefType = op.result().getType().cast<MemRefType>();
        const auto outputType = outputMemrefType.getElementType();
        auto&& outputShape = outputMemrefType.getShape();
        const auto outputSize = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());

        // Copy C from memref to vector
        auto zero = rewriter.create<ConstantOp>(loc, outputType, rewriter.getZeroAttr(outputType));
        auto vecTy = mlir::VectorType::get({ outputSize }, outputType);
        mlir::Value vecC = rewriter.create<vector::BroadcastOp>(loc, vecTy, zero);
        auto loopInitC = rewriter.create<AffineForOp>(loc, 0, outputSize, 1, vecC);
        auto loopBuilderInitC = utilir::MakeBodyBuilder(loopInitC);
        auto inductionVarInitC = loopInitC.getInductionVar();
        auto laneIndex = loopBuilderInitC.create<mlir::IndexCastOp>(loc, inductionVarInitC, i32Ty);
        auto elem = loopBuilderInitC.create<memref::LoadOp>(loc, opC, inductionVarInitC);
        vecC = loopBuilderInitC.create<vector::InsertElementOp>(loc, elem, loopInitC.getRegionIterArgs()[0], laneIndex);
        loopBuilderInitC.create<AffineYieldOp>(loc, vecC);

        const auto inputMemrefType = opA.getType().cast<MemRefType>();
        const auto inputType = inputMemrefType.getElementType();
        auto&& inputShape = inputMemrefType.getShape();
        const auto inputSize = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<int64_t>());
        auto [warpSizeX, warpSizeY] = utilir::ResolveWarpSize(utilir::ResolveExecutionRuntime(op).value()).value();
        const auto warpSize = warpSizeX * warpSizeY;
        const auto inElemsPerThread = mfmaType.getInElementsPerThread(warpSize);
        auto loop = rewriter.create<AffineForOp>(loc, 0, inputSize, inElemsPerThread, loopInitC.results());
        auto loopBuilder = utilir::MakeBodyBuilder(loop);
        auto matD = loop.getRegionIterArgs()[0];
        auto inductionVar = loop.getInductionVar();
        if (inputType.isF16() || inputType.isBF16())
        {
            auto vecTy = VectorType::get({ inElemsPerThread }, inputType);
            auto zero = loopBuilder.create<ConstantOp>(loc, inputType, rewriter.getZeroAttr(inputType));
            mlir::Value vecA = loopBuilder.create<vector::BroadcastOp>(loc, vecTy, zero);
            mlir::Value vecB = loopBuilder.create<vector::BroadcastOp>(loc, vecTy, zero);
            auto loadAB = loopBuilder.create<AffineForOp>(loc, 0, inElemsPerThread, 1, ValueRange{ vecA, vecB });
            auto loadABbuilder = utilir::MakeBodyBuilder(loadAB);
            auto iElem = loadABbuilder.create<mlir::IndexCastOp>(loc, loadAB.getInductionVar(), i32Ty);
            auto pos = loadABbuilder.create<AddIOp>(loc, loadAB.getInductionVar(), inductionVar);
            auto elemA = loadABbuilder.create<memref::LoadOp>(loc, opA, pos.result());
            vecA = loadABbuilder.create<vector::InsertElementOp>(loc, elemA, loadAB.getRegionIterArgs()[0], iElem);
            auto elemB = loadABbuilder.create<memref::LoadOp>(loc, opB, pos.result());
            vecB = loadABbuilder.create<vector::InsertElementOp>(loc, elemB, loadAB.getRegionIterArgs()[1], iElem);
            loadABbuilder.create<AffineYieldOp>(loc, ValueRange{ vecA, vecB });
            vecA = loadAB.results()[0];
            vecB = loadAB.results()[1];

            if (inputType.isF16())
            {
                switch (mfmaShape)
                {
                case MMAShape::M64xN64xK4_B4:
                    loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_16x16x4f16>(loc, vecC.getType(), ValueRange{ vecA, vecB, matD, cbsz, abid, blgp }) });
                    break;
                case MMAShape::M64xN64xK4_B2:
                    loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_32x32x4f16>(loc, vecC.getType(), ValueRange{ vecA, vecB, matD, cbsz, abid, blgp }) });
                    break;
                case MMAShape::M32xN32xK8_B1:
                    loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_32x32x8f16>(loc, vecC.getType(), ValueRange{ vecA, vecB, matD, cbsz, abid, blgp }) });
                    break;
                case MMAShape::M16xN16xK16_B1:
                    loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_16x16x16f16>(loc, vecC.getType(), ValueRange{ vecA, vecB, matD, cbsz, abid, blgp }) });
                    break;
                default:
                    return failure();
                }
            }
            else // inputType.isBF16()
            {
                switch (mfmaShape)
                {
                case MMAShape::M64xN64xK2_B4:
                    loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_16x16x2bf16>(loc, vecC.getType(), ValueRange{ vecA, vecB, matD, cbsz, abid, blgp }) });
                    break;
                case MMAShape::M64xN64xK2_B2:
                    loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_32x32x2bf16>(loc, vecC.getType(), ValueRange{ vecA, vecB, matD, cbsz, abid, blgp }) });
                    break;
                case MMAShape::M32xN32xK4_B1:
                    loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_32x32x4bf16>(loc, vecC.getType(), ValueRange{ vecA, vecB, matD, cbsz, abid, blgp }) });
                    break;
                case MMAShape::M16xN16xK8_B1:
                    loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_16x16x8bf16>(loc, vecC.getType(), ValueRange{ vecA, vecB, matD, cbsz, abid, blgp }) });
                    break;
                default:
                    return failure();
                }
            }
        }
        else if (inputType.isInteger(8))
        {
            auto innerLoop = loopBuilder.create<AffineForOp>(loc, 0, inElemsPerThread, 1, matD);
            auto innerLoopbuilder = utilir::MakeBodyBuilder(innerLoop);
            mlir::Value acc = innerLoop.getRegionIterArgs()[0];
            auto pos = innerLoopbuilder.create<AddIOp>(loc, innerLoop.getInductionVar(), inductionVar);
            auto elemA = innerLoopbuilder.create<memref::LoadOp>(loc, opA, pos.result());
            auto elemB = innerLoopbuilder.create<memref::LoadOp>(loc, opB, pos.result());
            switch (mfmaShape)
            {
            case MMAShape::M64xN64xK4_B4:
                innerLoopbuilder.create<AffineYieldOp>(loc, ValueRange{ innerLoopbuilder.create<ROCDL::mfma_i32_16x16x4i8>(loc, vecC.getType(), ValueRange{ elemA, elemB, acc, cbsz, abid, blgp }) });
                break;
            case MMAShape::M64xN64xK4_B2:
                innerLoopbuilder.create<AffineYieldOp>(loc, ValueRange{ innerLoopbuilder.create<ROCDL::mfma_i32_32x32x4i8>(loc, vecC.getType(), ValueRange{ elemA, elemB, acc, cbsz, abid, blgp }) });
                break;
            case MMAShape::M32xN32xK8_B1:
                innerLoopbuilder.create<AffineYieldOp>(loc, ValueRange{ innerLoopbuilder.create<ROCDL::mfma_i32_32x32x8i8>(loc, vecC.getType(), ValueRange{ elemA, elemB, acc, cbsz, abid, blgp }) });
                break;
            case MMAShape::M16xN16xK16_B1:
                innerLoopbuilder.create<AffineYieldOp>(loc, ValueRange{ innerLoopbuilder.create<ROCDL::mfma_i32_16x16x16i8>(loc, vecC.getType(), ValueRange{ elemA, elemB, acc, cbsz, abid, blgp }) });
                break;
            default:
                return failure();
            }

            loopBuilder.create<AffineYieldOp>(loc, innerLoop.results());
        }
        else if (inputType.isF32())
        {
            auto elemA = loopBuilder.create<memref::LoadOp>(loc, opA, inductionVar);
            auto elemB = loopBuilder.create<memref::LoadOp>(loc, opB, inductionVar);
            switch (mfmaShape)
            {
            case MMAShape::M64xN64xK1_B4:
                loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_16x16x1f32>(loc, vecC.getType(), ValueRange{ elemA, elemB, matD, cbsz, abid, blgp }) });
                break;
            case MMAShape::M64xN64xK1_B2:
                loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_32x32x1f32>(loc, vecC.getType(), ValueRange{ elemA, elemB, matD, cbsz, abid, blgp }) });
                break;
            case MMAShape::M32xN32xK2_B1:
                loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_32x32x2f32>(loc, vecC.getType(), ValueRange{ elemA, elemB, matD, cbsz, abid, blgp }) });
                break;
            case MMAShape::M16xN16xK4_B1:
                loopBuilder.create<AffineYieldOp>(loc, ValueRange{ loopBuilder.create<ROCDL::mfma_f32_16x16x4f32>(loc, vecC.getType(), ValueRange{ elemA, elemB, matD, cbsz, abid, blgp }) });
                break;
            default:
                return failure();
            }
        }
        else
        {
            return failure();
        }

        // Copy C back from vector to memref
        mlir::Value result = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
        auto loopCopyC = rewriter.replaceOpWithNewOp<AffineForOp>(op, 0, outputSize, 1, result);
        auto loopBuilderCopyC = utilir::MakeBodyBuilder(loopCopyC);
        auto inductionVarCopyC = loopCopyC.getInductionVar();
        auto resultMemRef = loopCopyC.getRegionIterArgs()[0];
        laneIndex = loopBuilderCopyC.create<mlir::IndexCastOp>(loc, inductionVarCopyC, i32Ty);
        auto item = loopBuilderCopyC.create<vector::ExtractElementOp>(loc, loop.results()[0], laneIndex);
        loopBuilderCopyC.create<memref::StoreOp>(loc, item, resultMemRef, inductionVarCopyC);
        loopBuilderCopyC.create<AffineYieldOp>(loc, resultMemRef);

        return success();
    }
};

LogicalResult BlockDimMatchAndRewrite(Operation* op, PatternRewriter& rewriter, const GPUIndexDimension blockDimIdx, const bool indexType)
{
    auto gpuFunc = op->getParentOfType<gpu::GPUFuncOp>();
    if (!gpuFunc)
    {
        return failure();
    }
    auto blockSizeAttr = gpuFunc->getAttrOfType<ArrayAttr>("blockSize");
    if (!blockSizeAttr || blockDimIdx == GPUIndexDimension::Invalid)
    {
        return failure();
    }
    auto val = blockSizeAttr.getValue()[static_cast<int>(blockDimIdx)].cast<IntegerAttr>().getInt();
    if (indexType)
        rewriter.replaceOpWithNewOp<mlir::ConstantIndexOp>(op, val);
    else // We need this because the ROCDL gpu indices are generated with i32 type,
        // and we need to match that, otherwise we are left with invalid cast ops.
        rewriter.replaceOpWithNewOp<mlir::ConstantIntOp>(op, val, rewriter.getI32Type());
    return success();
}

struct ResolveBlockDimPattern final : public OpRewritePattern<gpu::BlockDimOp>
{
    using OpRewritePattern<gpu::BlockDimOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(gpu::BlockDimOp op, PatternRewriter& rewriter) const final
    {
        return BlockDimMatchAndRewrite(op, rewriter, dimIndexToInteger(op.dimension()), true);
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

// This is copied from: /mlir/lib/Conversion/GPUCommon/IndexIntrinsicsOpLowering.h
// Rewriting that replaces Op with XOp, YOp, or ZOp depending on the dimension
// that Op operates on.
template <typename Op, typename XOp, typename YOp, typename ZOp>
struct GPUIndexIntrinsicOpLowering : public ConvertOpToLLVMPattern<Op>
{
    explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter& typeConverter) :
        ConvertOpToLLVMPattern<Op>(typeConverter) {}

    // Convert the kernel arguments to an LLVM type, preserve the rest.
    LogicalResult
    matchAndRewrite(Op op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override
    {
        auto loc = op->getLoc();
        MLIRContext* context = rewriter.getContext();
        Value newOp;
        switch (dimIndexToInteger(op.dimension()))
        {
        case GPUIndexDimension::X:
            newOp = rewriter.create<XOp>(loc, IntegerType::get(context, 32));
            break;
        case GPUIndexDimension::Y:
            newOp = rewriter.create<YOp>(loc, IntegerType::get(context, 32));
            break;
        case GPUIndexDimension::Z:
            newOp = rewriter.create<ZOp>(loc, IntegerType::get(context, 32));
            break;
        default:
            return failure();
        }

        newOp = rewriter.create<IndexCastOp>(loc, newOp, rewriter.getIndexType());

        rewriter.replaceOp(op, { newOp });
        return success();
    }
};

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
            vir::MMAComputeSyncOp,
            vir::MMAFillSyncOp,
            vir::MMALoadSyncOp,
            vir::MMAStoreSyncOp,
            vir::BarrierOp,
            gpu::BlockDimOp,
            ROCDL::BlockDimXOp,
            ROCDL::BlockDimYOp,
            ROCDL::BlockDimXOp>();
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

struct GPUToROCDLPass : public accera::transforms::ConvertGPUToROCDLBase<GPUToROCDLPass>
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
            gpu::ThreadIdOp,
            gpu::BlockIdOp,
            gpu::BlockDimOp,
            gpu::GridDimOp>();
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
            mlir::LowerToLLVMOptions options(context);
            options.emitCWrappers = true;
            mlir::LLVMTypeConverter converter(context, options);
            RewritePatternSet patterns(context);
            populateGPUToROCDLPatterns(converter, patterns);
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
            vir::MMAComputeSyncOp,
            vir::MMAFillSyncOp,
            vir::MMALoadSyncOp,
            vir::MMAStoreSyncOp,
            vir::BarrierOp,
            gpu::BlockDimOp,
            ROCDL::BlockDimXOp,
            ROCDL::BlockDimYOp,
            ROCDL::BlockDimXOp>();
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
        ValueMMALoadSyncOpToRocDLConversion,
        ValueMFMAComputeToRocDLConversion,
        ValueMMAStoreSyncOpToRocDLConversion,
        ValueMMAFillSyncOpToRocDLConversion>(patterns.getContext());
}

void populateAcceraToNVVMPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<
        ResolveBlockDimPattern,
        EarlyReturnToGPUReturnPattern,
        ValueBarrierToGPUBarrierConversion>(patterns.getContext());
}

void populateGPUToROCDLPatterns(mlir::LLVMTypeConverter& converter, mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<
        GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp, ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>,
        GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, ROCDL::BlockDimXOp, ROCDL::BlockDimYOp, ROCDL::BlockDimZOp>,
        GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp, ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>,
        GPUIndexIntrinsicOpLowering<gpu::GridDimOp, ROCDL::GridDimXOp, ROCDL::GridDimYOp, ROCDL::GridDimZOp>>(converter);
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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGPUToROCDLPass()
{
    return std::make_unique<GPUToROCDLPass>();
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
    default:
        return {};
    }
}

} // namespace accera::transforms
