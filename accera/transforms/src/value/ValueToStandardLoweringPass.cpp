////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"
#include "ir/include/value/ValueEnums.h"
#include "util/VectorizationUtil.h"

#include <ir/include/IRUtil.h>
#include <ir/include/exec/ExecutionPlanAttributes.h>
#include <ir/include/exec/VectorizationInfo.h>
#include <ir/include/value/ValueDialect.h>

#include <utilities/include/MathUtil.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Math/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVAttributes.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/FormatVariadic.h>

#include <algorithm>
#include <unordered_set>

#ifndef RC_FILE_LOC
#define RC_FILE_LOC(rewriter) accera::ir::util::GetLocation(rewriter, __FILE__, __LINE__)
#endif // RC_FILE_LOC

using namespace mlir;
using namespace llvm;
namespace ir = accera::ir;
namespace utilir = accera::ir::util;
namespace vir = accera::ir::value;
namespace tr = accera::transforms;
namespace vtr = accera::transforms::value;

namespace accera::generated
{
using llvm::ArrayRef;
using llvm::SmallVector;
using mlir::Type;
using mlir::Value;
using vir::YieldOp;
namespace
{
#include "value/ValueConversion.inc"
} // namespace
} // namespace accera::generated

namespace irutil = accera::ir::util;
using namespace accera::ir;
using namespace accera::transforms;
using namespace accera::transforms::value;
using namespace accera::utilities;

namespace
{
constexpr auto kDefaultExecutionTarget = vir::ExecutionTarget::CPU;
const char kGlobalOpSymNameFormat[] = "allocated_memref_{0}";

const char kProfileRegionSymNameFormat[] = "profile_region_{0}_{1}";
const char kProfileRegionNameIdentifier[] = "profile_region_name";
const char kProfileRegionTypeIdentifier[] = "profile_region_type";

enum class ProfileCounterType
{
    count = 0,
    time = 1,
    startTime = 2,
};

void InitializeProfileRegions(mlir::ModuleOp module, mlir::OpBuilder& builder)
{
    std::unordered_set<std::string> regionNames;
    module.walk([&](vir::EnterProfileRegionOp op) {
        auto regionName = op.regionName().str();
        regionNames.insert(regionName);
    });

    auto loc = module.getLoc();
    auto body = module.getBody();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(body, body->begin());
    auto int64Type = builder.getI64Type();
    auto doubleType = builder.getF64Type();
    auto executionCounterType = mlir::MemRefType::get({ 1 }, int64Type);
    auto timeType = mlir::MemRefType::get({ 1 }, doubleType);
    auto timeTensorType = mlir::RankedTensorType::get({ 1 }, doubleType);
    // TODO: maybe make a new "profile region" type that holds a reference to the counters?

    for (auto name : regionNames)
    {
        auto nameAttr = builder.getStringAttr(name);

        auto count = builder.create<vir::GlobalOp>(loc, executionCounterType, false, llvm::formatv(kProfileRegionSymNameFormat, name, "count").str(), builder.getI64TensorAttr({ 0 }));
        count->setAttr(kProfileRegionNameIdentifier, nameAttr);
        count->setAttr(kProfileRegionTypeIdentifier, builder.getI32IntegerAttr(static_cast<int>(ProfileCounterType::count)));

        auto time = builder.create<vir::GlobalOp>(loc, timeType, false, llvm::formatv(kProfileRegionSymNameFormat, name, "time").str(), mlir::DenseFPElementsAttr::get(timeTensorType, { 0.0 }));
        time->setAttr(kProfileRegionNameIdentifier, nameAttr);
        time->setAttr(kProfileRegionTypeIdentifier, builder.getI32IntegerAttr(static_cast<int>(ProfileCounterType::time)));

        auto startTime = builder.create<vir::GlobalOp>(loc, timeType, false, llvm::formatv(kProfileRegionSymNameFormat, name, "start").str(), mlir::DenseFPElementsAttr::get(timeTensorType, { 0.0 }));
        startTime->setAttr(kProfileRegionNameIdentifier, nameAttr);
        startTime->setAttr(kProfileRegionTypeIdentifier, builder.getI32IntegerAttr(static_cast<int>(ProfileCounterType::startTime)));
    }
}

struct ProfileCounter
{
    vir::GlobalOp count;
    vir::GlobalOp time;
    vir::GlobalOp startTime;
};

struct ProfileRegions
{
    ProfileRegions(mlir::Operation* op) // must be a module
    {
        auto module = mlir::cast<mlir::ModuleOp>(op);
        module.walk([&](vir::GlobalOp op) {
            if (auto regionNameAttr = op->getAttrOfType<mlir::StringAttr>(kProfileRegionNameIdentifier))
            {
                switch (static_cast<ProfileCounterType>(op->getAttrOfType<mlir::IntegerAttr>(kProfileRegionTypeIdentifier).getInt()))
                {
                case ProfileCounterType::count:
                    counters[regionNameAttr.getValue().str()].count = op;
                    break;
                case ProfileCounterType::time:
                    counters[regionNameAttr.getValue().str()].time = op;
                    break;
                case ProfileCounterType::startTime:
                    counters[regionNameAttr.getValue().str()].startTime = op;
                    break;
                default:
                    op.emitError("Error: bad counter type");
                    break;
                }
            }
        });
    }

    std::map<std::string, ProfileCounter> counters;
};

std::string GetFormatStringForElementType(mlir::Type elementType)
{
    if (elementType.isa<mlir::FloatType>())
    {
        return "%f";
    }
    else if (elementType.isa<mlir::IntegerType>() || elementType.isa<mlir::IndexType>())
    {
        return "%ld";
    }
    assert(false && "Unsupported element type for printing");
    return "";
}

using ValueBinOp = vir::BinOp;
struct BinOpLowering : public OpRewritePattern<ValueBinOp>
{
    using OpRewritePattern<ValueBinOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueBinOp op,
        PatternRewriter& rewriter) const override;
};

using ValueGetElementOp = vir::GetElementOp;
struct GetElementOpLowering : public OpRewritePattern<ValueGetElementOp>
{
    using OpRewritePattern<ValueGetElementOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueGetElementOp op,
        PatternRewriter& rewriter) const override;
};

using ValueGlobalOp = vir::GlobalOp;
struct GlobalOpLowering : public OpRewritePattern<ValueGlobalOp>
{
    using OpRewritePattern<ValueGlobalOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueGlobalOp op,
        PatternRewriter& rewriter) const override;
};

using ValueUnaryOp = vir::UnaryOp;
struct UnaryOpLowering : public OpRewritePattern<ValueUnaryOp>
{
    using OpRewritePattern<ValueUnaryOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueUnaryOp op,
        PatternRewriter& rewriter) const override;
};

using ValueLoadOp = vir::LoadOp;
struct LoadOpLowering : public OpRewritePattern<ValueLoadOp>
{
    using OpRewritePattern<ValueLoadOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueLoadOp op,
        PatternRewriter& rewriter) const override;
};

using ValueStoreOp = vir::StoreOp;
struct StoreOpLowering : public OpRewritePattern<ValueStoreOp>
{
    using OpRewritePattern<ValueStoreOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueStoreOp op,
        PatternRewriter& rewriter) const override;
};

using ValueCmpOp = vir::CmpOp;
struct CmpOpLowering : public OpRewritePattern<ValueCmpOp>
{
    using OpRewritePattern<ValueCmpOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueCmpOp op,
        PatternRewriter& rewriter) const override;
};

using ValueTerminatorOp = vir::YieldOp;
struct TerminatorLowering : public OpRewritePattern<ValueTerminatorOp>
{
    using OpRewritePattern<ValueTerminatorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueTerminatorOp op,
        PatternRewriter& rewriter) const override;
};

using ValueOffsetOp = vir::OffsetOp;
struct OffsetOpLowering : public OpRewritePattern<ValueOffsetOp>
{
    using OpRewritePattern<ValueOffsetOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueOffsetOp op,
        PatternRewriter& rewriter) const override;
};

using ValueViewOp = vir::ViewOp;
struct ViewOpLowering : public OpRewritePattern<ValueViewOp>
{
    using OpRewritePattern<ValueViewOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueViewOp op,
        PatternRewriter& rewriter) const override;
};

using ValueSliceOp = vir::SliceOp;
struct SliceOpLowering : public OpRewritePattern<ValueSliceOp>
{
    using OpRewritePattern<ValueSliceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueSliceOp op,
        PatternRewriter& rewriter) const override;
};

using ValueMergeDimOp = vir::MergeDimOp;
struct MergeDimOpLowering : public OpRewritePattern<ValueMergeDimOp>
{
    using OpRewritePattern<ValueMergeDimOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueMergeDimOp op,
        PatternRewriter& rewriter) const override;
};

using ValueSplitDimOp = vir::SplitDimOp;
struct SplitDimOpLowering : public OpRewritePattern<ValueSplitDimOp>
{
    using OpRewritePattern<ValueSplitDimOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueSplitDimOp op,
        PatternRewriter& rewriter) const override;
};

using ValueReorderOp = vir::ReorderOp;
struct ReorderOpLowering : public OpRewritePattern<ValueReorderOp>
{
    using OpRewritePattern<ValueReorderOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueReorderOp op,
        PatternRewriter& rewriter) const override;
};

using ValueReshapeOp = vir::ReshapeOp;
struct ReshapeOpLowering : public OpRewritePattern<ValueReshapeOp>
{
    using OpRewritePattern<ValueReshapeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueReshapeOp op,
        PatternRewriter& rewriter) const override;
};

using ValuePrintOp = vir::PrintOp;
class PrintOpLowering : public OpRewritePattern<ValuePrintOp>
{
public:
    using OpRewritePattern<ValuePrintOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValuePrintOp op,
        PatternRewriter& rewriter) const override;
};

using ValueReduceOp = vir::ReduceOp;
class ReduceOpVectorization : public OpRewritePattern<ValueReduceOp>
{
public:
    using OpRewritePattern<ValueReduceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueReduceOp op,
        PatternRewriter& rewriter) const override;
};

class ReduceOpLowering : public OpRewritePattern<ValueReduceOp>
{
public:
    using OpRewritePattern<ValueReduceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueReduceOp op,
        PatternRewriter& rewriter) const override;
};

using ValueReferenceGlobalOp = vir::ReferenceGlobalOp;
class ReferenceGlobalOpLowering : public OpRewritePattern<ValueReferenceGlobalOp>
{
public:
    using OpRewritePattern<ValueReferenceGlobalOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueReferenceGlobalOp op,
        PatternRewriter& rewriter) const override;
};

using ValueMapReduceOp = vir::MapReduceOp;
class MapReduceOpVectorization : public OpRewritePattern<ValueMapReduceOp>
{
public:
    using OpRewritePattern<ValueMapReduceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueMapReduceOp op,
        PatternRewriter& rewriter) const override;
};

class MapReduceOpLowering : public OpRewritePattern<ValueMapReduceOp>
{
public:
    using OpRewritePattern<ValueMapReduceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueMapReduceOp op,
        PatternRewriter& rewriter) const override;
};

using ValueReduceMaxOp = vir::ReduceMaxOp;
class ReduceMaxOpLowering : public OpRewritePattern<ValueReduceMaxOp>
{
public:
    using OpRewritePattern<ValueReduceMaxOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueReduceMaxOp op,
        PatternRewriter& rewriter) const override;
};

using ValueReduceSumOp = vir::ReduceSumOp;
class ReduceSumOpLowering : public OpRewritePattern<ValueReduceSumOp>
{
public:
    using OpRewritePattern<ValueReduceSumOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueReduceSumOp op,
        PatternRewriter& rewriter) const override;
};

using vir::EnterProfileRegionOp;
using vir::ExitProfileRegionOp;
using vir::PrintProfileResultsOp;

struct EnterProfileRegionOpLowering : public OpRewritePattern<EnterProfileRegionOp>
{
    using OpRewritePattern::OpRewritePattern;
    EnterProfileRegionOpLowering(MLIRContext* context, bool enableProfiling) :
        OpRewritePattern(context),
        enableProfiling(enableProfiling)
    {}

    LogicalResult matchAndRewrite(EnterProfileRegionOp op, PatternRewriter& rewriter) const final;

    bool enableProfiling = true;
};

struct ExitProfileRegionOpLowering : public OpRewritePattern<ExitProfileRegionOp>
{
    using OpRewritePattern::OpRewritePattern;
    ExitProfileRegionOpLowering(MLIRContext* context, bool enableProfiling) :
        OpRewritePattern(context),
        enableProfiling(enableProfiling)
    {}

    LogicalResult matchAndRewrite(ExitProfileRegionOp op, PatternRewriter& rewriter) const final;

    bool enableProfiling = true;
};

struct PrintProfileResultsOpLowering : public OpRewritePattern<PrintProfileResultsOp>
{
    using OpRewritePattern::OpRewritePattern;
    PrintProfileResultsOpLowering(MLIRContext* context, bool enableProfiling) :
        OpRewritePattern(context),
        enableProfiling(enableProfiling)
    {}
    LogicalResult matchAndRewrite(PrintProfileResultsOp op, PatternRewriter& rewriter) const final;

    bool enableProfiling = true;
};

using ValueAllocOp = vir::AllocOp;
struct AllocOpLowering : public OpRewritePattern<ValueAllocOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(ValueAllocOp op,
                                  PatternRewriter& rewriter) const final
    {
        auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });

        auto execTarget = irutil::ResolveExecutionTarget(op).value_or(kDefaultExecutionTarget);

        switch (execTarget)
        {
        case vir::ExecutionTarget::CPU: {

            auto memrefType = op.getType();
            auto allocType = op.allocType().getValueOr(vir::MemoryAllocType::Global);

            switch (allocType)
            {
            case vir::MemoryAllocType::Global: {
                auto globalOp = irutil::CreateGlobalBufferOp(rewriter, op, MemRefType::Builder{ memrefType }.setAffineMaps({}), kGlobalOpSymNameFormat);
                rewriter.replaceOpWithNewOp<vir::ReferenceGlobalOp>(op, memrefType, globalOp.sym_name());
            }
            break;
            case vir::MemoryAllocType::Stack:
                rewriter.replaceOpWithNewOp<memref::AllocaOp>(op, MemRefType::Builder{ memrefType }.setAffineMaps({}), mlir::ValueRange{}, op.alignmentAttr());
                break;
            default:
                llvm_unreachable("Unknown alloc type");
            }
            break;
        }
        case vir::ExecutionTarget::GPU:
            rewriter.replaceOpWithNewOp<memref::AllocOp>(op, op.getType(), mlir::ValueRange{}, op.alignmentAttr());
            break;
        }

        return success();
    }
};

struct ValueToStdLoweringPass : public ConvertValueToStdBase<ValueToStdLoweringPass>
{
    ValueToStdLoweringPass() = default;
    ValueToStdLoweringPass(bool enableProfiling) :
        ValueToStdLoweringPass()
    {
        this->enableProfiling = enableProfiling;
    }

    void runOnModule() final;
};

struct ValueModuleOpRewritePattern : OpRewritePattern<vir::ValueModuleOp>
{
    using OpRewritePattern::OpRewritePattern;

    void AddGPUAnnotations(ModuleOp module, PatternRewriter& rewriter) const
    {
        module->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
                        rewriter.getUnitAttr());
    }
    void AddRocmAnnotations(vir::ValueModuleOp module, PatternRewriter& rewriter) const
    {
        auto gpuModOps = module.getOps<gpu::GPUModuleOp>();
        for (auto gpuModOp : gpuModOps)
        {
            gpuModOp->setAttr(mlir::gpu::getDefaultGpuBinaryAnnotation(),
                              rewriter.getStringAttr("HSACO"));
            gpuModOp->setAttr(vir::ValueModuleOp::getExecRuntimeAttrName(),
                              ir::value::ExecutionRuntimeAttr::get(getContext(), vir::ExecutionRuntime::ROCM));
        }
    }
    void AddNVVMAnnotations(vir::ValueModuleOp module, PatternRewriter& rewriter) const
    {
        auto gpuModOps = module.getOps<gpu::GPUModuleOp>();
        for (auto gpuModOp : gpuModOps)
        {
            gpuModOp->setAttr(mlir::gpu::getDefaultGpuBinaryAnnotation(),
                              rewriter.getStringAttr("CUBIN"));
            gpuModOp->setAttr(vir::ValueModuleOp::getExecRuntimeAttrName(),
                              ir::value::ExecutionRuntimeAttr::get(getContext(), vir::ExecutionRuntime::CUDA));
        }
    }

    void AddVulkanAnnotations(ModuleOp module, PatternRewriter& rewriter) const
    {
        auto context = module.getContext();
        namespace spirv = mlir::spirv;
        auto triple = spirv::VerCapExtAttr::get(
            spirv::Version::V_1_0,
            { spirv::Capability::Shader },
            // TODO: figure out best way to customize this
            llvm::makeArrayRef(spirv::Extension::SPV_KHR_storage_buffer_storage_class),
            context);
        auto defaultTargetEnvAttr = spirv::getDefaultTargetEnv(context);
        auto targetEnvAttr = spirv::TargetEnvAttr::get(
            triple,
            defaultTargetEnvAttr.getVendorID(),
            defaultTargetEnvAttr.getDeviceType(),
            defaultTargetEnvAttr.getDeviceID(),
            defaultTargetEnvAttr.getResourceLimits());
        module->setAttr(
            mlir::spirv::getTargetEnvAttrName(),
            targetEnvAttr);
        module->setAttr(vir::ValueModuleOp::getExecRuntimeAttrName(),
                        ir::value::ExecutionRuntimeAttr::get(getContext(), vir::ExecutionRuntime::VULKAN));
    }

    LogicalResult matchAndRewrite(vir::ValueModuleOp vModuleOp, PatternRewriter& rewriter) const final
    {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(vModuleOp);

        auto module = vModuleOp->getParentOfType<ModuleOp>();
        if (!vModuleOp.getOps<gpu::GPUModuleOp>().empty())
        {
            AddGPUAnnotations(module, rewriter);

            const auto runtime = utilir::ResolveExecutionRuntime(vModuleOp);
            if (runtime == vir::ExecutionRuntime::VULKAN)
            {
                AddVulkanAnnotations(module, rewriter);
            }
            else if (runtime == vir::ExecutionRuntime::CUDA)
            {
                AddNVVMAnnotations(vModuleOp, rewriter);
            }
            else if (runtime == vir::ExecutionRuntime::ROCM)
            {
                AddRocmAnnotations(vModuleOp, rewriter);
            }
        }

        Operation* modEnd = &(module.getBody()->back());
        rewriter.mergeBlockBefore(vModuleOp.getBody(), modEnd);

        // Erase accv.terminator
        rewriter.eraseOp(modEnd->getPrevNode());
        rewriter.eraseOp(vModuleOp);

        return success();
    }
}; // namespace

constexpr int kLaunchConfigDefaultDimValue = 1;
constexpr int kLocalSizeDimSize = 3;
constexpr size_t kLaunchConfigNumDims = 6;

auto GetGPUModuleBinaryAnnotationAttrName()
{
    return mlir::gpu::getDefaultGpuBinaryAnnotation();
}

auto GetGPUModuleBinaryAnnotationAttrValue(vir::ExecutionRuntime runtime)
{
    switch (runtime)
    {
    // ref: mlir/test/Conversion/GPUToROCm/lower-rocdl-kernel-to-hsaco.mlir
    case vir::ExecutionRuntime::ROCM:
        return "HSACO";

    // ref: mlir/test/Conversion/GPUToCUDA/lower-nvvm-kernel-to-cubin.mlir
    case vir::ExecutionRuntime::CUDA:
        return "CUBIN";

    case vir::ExecutionRuntime::VULKAN:
        [[fallthrough]];
    case vir::ExecutionRuntime::DEFAULT:
        [[fallthrough]];
    default:
        return "";
    }
}

struct GPUTargetedFuncRewritePattern : OpRewritePattern<FuncOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult match(FuncOp op) const final
    {
        auto target = utilir::ResolveExecutionTarget(op);
        return success(target && *target == vir::ExecutionTarget::GPU);
    }

    void rewrite(FuncOp funcOp, PatternRewriter& rewriter) const final
    {
        auto gpuRuntime = utilir::ResolveExecutionRuntime(funcOp).value_or(vir::ExecutionRuntime::NONE);

        auto loc = rewriter.getFusedLoc({ funcOp.getLoc(), RC_FILE_LOC(rewriter) });
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(funcOp);

        auto module = funcOp->getParentOfType<vir::ValueModuleOp>();
        auto newFuncName = funcOp.getName().str();

        auto gpuModule = rewriter.create<gpu::GPUModuleOp>(loc, newFuncName + "_module");
        gpuModule->setAttr(GetGPUModuleBinaryAnnotationAttrName(), rewriter.getStringAttr(GetGPUModuleBinaryAnnotationAttrValue(gpuRuntime)));
        gpuModule->setAttr(vir::ValueModuleOp::getExecRuntimeAttrName(),
                           ir::value::ExecutionRuntimeAttr::get(getContext(), gpuRuntime));
        gpuModule.setVisibility(mlir::SymbolTable::Visibility::Public);

        auto insertPt = utilir::GetTerminalInsertPoint<gpu::GPUModuleOp, gpu::ModuleEndOp>(gpuModule);
        OpBuilder::InsertionGuard gpuScopeGuard(rewriter);
        rewriter.restoreInsertionPoint(insertPt);

        for (auto op : module.getOps<vir::GlobalOp>())
        {
            rewriter.clone(*op);
        }

        SmallVector<mlir::NamedAttribute, 4> fnAttrs;
        SmallVector<int32_t, kLaunchConfigNumDims> launchConfigVec(kLaunchConfigNumDims, kLaunchConfigDefaultDimValue);
        if (auto arrayAttr = funcOp->getAttrOfType<mlir::ArrayAttr>(vir::ValueFuncOp::getGPULaunchAttrName()))
        {
            launchConfigVec = llvm::to_vector<kLaunchConfigNumDims>(llvm::map_range(arrayAttr.getAsRange<IntegerAttr>(), [](IntegerAttr intAttr) { return (int32_t)intAttr.getInt(); }));
        }
        auto launchConfig = llvm::makeArrayRef(launchConfigVec);
        assert(launchConfig.size() == kLaunchConfigNumDims);
        // split out the launch config into the grid and block dimensions, respectively
        [[maybe_unused]] auto gridDimsLaunchConfig = launchConfig.take_front(kLocalSizeDimSize);
        auto blockDimsLaunchConfig = launchConfig.drop_front(kLocalSizeDimSize);

        fnAttrs.emplace_back(rewriter.getIdentifier(mlir::gpu::GPUDialect::getKernelFuncAttrName()),
                             rewriter.getUnitAttr());
        if (gpuRuntime == vir::ExecutionRuntime::VULKAN)
        {
            auto entryPointLocalSize = blockDimsLaunchConfig;
            assert(entryPointLocalSize.size() == kLocalSizeDimSize);
            fnAttrs.emplace_back(
                rewriter.getIdentifier(mlir::spirv::getEntryPointABIAttrName()),
                mlir::spirv::getEntryPointABIAttr(entryPointLocalSize, rewriter.getContext()));
        }
        else
        {
            SmallVector<mlir::Attribute, 4> gridDimsLaunchConfigAttrs, blockDimsLaunchConfigAttrs;
            for (auto dim : gridDimsLaunchConfig)
            {
                gridDimsLaunchConfigAttrs.emplace_back(rewriter.getI32IntegerAttr(dim));
            }
            for (auto dim : blockDimsLaunchConfig)
            {
                blockDimsLaunchConfigAttrs.emplace_back(rewriter.getI32IntegerAttr(dim));
            }
            fnAttrs.emplace_back(
                rewriter.getIdentifier("gridSize"), rewriter.getArrayAttr(gridDimsLaunchConfigAttrs));
            fnAttrs.emplace_back(
                rewriter.getIdentifier("blockSize"), rewriter.getArrayAttr(blockDimsLaunchConfigAttrs));
        }

        auto newFuncOp = rewriter.create<gpu::GPUFuncOp>(
            loc,
            newFuncName,
            funcOp.getType(),
            llvm::None,
            llvm::None,
            fnAttrs);

        rewriter.inlineRegionBefore(funcOp.getBody(), &newFuncOp.back());

        rewriter.eraseOp(funcOp);
    }
};

struct GPUTargetedFuncTerminatorRewritePattern : OpRewritePattern<ReturnOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult match(ReturnOp op) const final
    {
        auto target = utilir::ResolveExecutionTarget(op);
        return success(target && *target == vir::ExecutionTarget::GPU);
    }

    void rewrite(ReturnOp op, PatternRewriter& rewriter) const final
    {
        rewriter.replaceOpWithNewOp<gpu::ReturnOp>(op, op.operands());
    }
};

struct GenericOpTypeConversionPattern : public ConversionPattern
{
    explicit GenericOpTypeConversionPattern(MLIRContext* ctx, TypeConverter& typeConverter, PatternBenefit benefit = 1) :
        ConversionPattern(typeConverter, ConversionPattern::MatchAnyOpTypeTag{}, benefit, ctx)
    {}

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<mlir::Value> operands, ConversionPatternRewriter& rewriter) const final
    {
        NamedAttrList attrDic = op->getAttrDictionary();
        auto typeAttr = attrDic.get(function_like_impl::getTypeAttrName()).dyn_cast_or_null<TypeAttr>();
        if (op->getNumOperands() == 0 && op->getNumResults() == 0 && !typeAttr)
            return failure();

        auto typeConverter = getTypeConverter();
        auto operandTypeRange = ValueTypeRange<ArrayRef<mlir::Value>>{ operands };
        if (!typeConverter->isLegal(operandTypeRange))
        {
            op->emitWarning("TypeConverter cannot legalize operands");
            return failure();
        }
        llvm::SmallVector<mlir::Type, 1> resultTypes;
        if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
        {
            op->emitWarning("TypeConverter cannot convert result types");
            return failure();
        }
        auto loc = op->getLoc();
        auto opName = op->getName().getStringRef();

        auto opTypeName = op->getName().getStringRef().str();
        if (auto funcOp = mlir::dyn_cast<FuncOp>(op))
        {
            auto fnType = funcOp.getType();

            // TODO: support converting functions with one result.
            if (fnType.getNumResults() != 0)
                return failure();

            TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
            for (auto argType : enumerate(funcOp.getType().getInputs()))
            {
                auto convertedType = typeConverter->convertType(argType.value());
                if (!convertedType)
                    return failure();
                signatureConverter.addInputs(argType.index(), convertedType);
            }

            auto newFuncOp = rewriter.create<FuncOp>(
                loc, funcOp.getName(), rewriter.getFunctionType(signatureConverter.getConvertedTypes(), llvm::None));

            // Copy over all attributes other than the function name and type.
            for (const auto& namedAttr : funcOp->getAttrs())
            {
                if (namedAttr.first != function_like_impl::getTypeAttrName() &&
                    namedAttr.first != SymbolTable::getSymbolAttrName())
                    newFuncOp->setAttr(namedAttr.first, namedAttr.second);
            }

            rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
            if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter, &signatureConverter)))
                return failure();
            rewriter.eraseOp(funcOp);
            return success();
        }

        if (typeAttr)
        {
            attrDic.set(rewriter.getIdentifier(function_like_impl::getTypeAttrName()), TypeAttr::get(typeConverter->convertType(typeAttr.getValue())));
        }
        auto attrs = attrDic.getAttrs();

        llvm::SmallVector<std::unique_ptr<Region>, 1> regions;
        regions.reserve(op->getNumRegions());
        for (auto& region : op->getRegions())
        {
            auto newRegion = std::make_unique<Region>();
            newRegion->takeBody(region);
            regions.push_back(std::move(newRegion));
        }

        OperationState newOpState{ loc, opName, operands, resultTypes, attrs, {}, regions };
        auto newOp = rewriter.createOperation(newOpState);
        if (auto numResults = op->getNumResults(); numResults == 0)
        {
            rewriter.eraseOp(op);
        }
        else if (numResults == 1)
        {
            rewriter.replaceOp(op, newOp->getResult(0));
        }
        else
        {
            rewriter.replaceOp(op, newOp->getResults());
        }

        return success();
    }
};

struct ValueLaunchFuncOpRewritePattern : OpRewritePattern<vir::LaunchFuncOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(vir::LaunchFuncOp op, PatternRewriter& rewriter) const final
    {
        auto loc = op.getLoc();
        auto target = op.exec_target();
        auto callee = op.callee().getLeafReference();

        switch (target)
        {
        case vir::ExecutionTarget::CPU:
            rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee, ArrayRef<mlir::Type>{}, ValueRange{ op.operands() });
            return success();
        case vir::ExecutionTarget::GPU:
            auto gpuSymRef = rewriter.getSymbolRefAttr(callee.str() + "_module", rewriter.getSymbolRefAttr(callee));
            auto gpuFuncOp = SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(op, gpuSymRef);
            if (!gpuFuncOp) return failure();

            SmallVector<int64_t, kLaunchConfigNumDims> launchConfig;
            if (auto arrayAttr = op->getAttrOfType<mlir::ArrayAttr>(vir::ValueFuncOp::getGPULaunchAttrName()))
            {
                launchConfig =
                    llvm::to_vector<kLaunchConfigNumDims>(
                        llvm::map_range(
                            arrayAttr.getAsRange<IntegerAttr>(), [](IntegerAttr intAttr) { return intAttr.getInt(); }));
            }
            else
            {
                std::fill_n(std::back_inserter(launchConfig), kLaunchConfigNumDims, kLaunchConfigDefaultDimValue);
            }

            auto gridSize = gpu::KernelDim3{
                rewriter.create<ConstantIndexOp>(loc, launchConfig[(int)vir::Processor::BlockX]),
                rewriter.create<ConstantIndexOp>(loc, launchConfig[(int)vir::Processor::BlockY]),
                rewriter.create<ConstantIndexOp>(loc, launchConfig[(int)vir::Processor::BlockZ]),
            };
            auto blockSize = gpu::KernelDim3{
                rewriter.create<ConstantIndexOp>(loc, launchConfig[(int)vir::Processor::ThreadX]),
                rewriter.create<ConstantIndexOp>(loc, launchConfig[(int)vir::Processor::ThreadY]),
                rewriter.create<ConstantIndexOp>(loc, launchConfig[(int)vir::Processor::ThreadZ]),
            };

            rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(op,
                                                           gpuFuncOp,
                                                           gridSize,
                                                           blockSize,
                                                           op.getOperands());
            return success();
        }
        llvm_unreachable("Unknown target");

        return failure();
    }
};

} // namespace

LogicalResult BinOpLowering::matchAndRewrite(
    ValueBinOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });

    auto lhs = op.lhs();
    auto rhs = op.rhs();

    auto result = [&]() -> mlir::Value {
        using accera::ir::value::BinaryOpPredicate;
        if (auto pred = op.getPredicate(); util::GetElementType(lhs.getType()).isa<FloatType>())
        {
            switch (pred)
            {
            case BinaryOpPredicate::ADD:
                return rewriter.create<AddFOp>(loc, ValueRange{ lhs, rhs }, rewriter.getNamedAttr("RelaxedPrecision", rewriter.getUnitAttr()));
            case BinaryOpPredicate::DIV:
                return rewriter.create<DivFOp>(loc, ValueRange{ lhs, rhs }, rewriter.getNamedAttr("RelaxedPrecision", rewriter.getUnitAttr()));
            case BinaryOpPredicate::MOD:
                return rewriter.create<RemFOp>(loc, ValueRange{ lhs, rhs }, rewriter.getNamedAttr("RelaxedPrecision", rewriter.getUnitAttr()));
            case BinaryOpPredicate::MUL:
                return rewriter.create<MulFOp>(loc, ValueRange{ lhs, rhs }, rewriter.getNamedAttr("RelaxedPrecision", rewriter.getUnitAttr()));
            case BinaryOpPredicate::SUB:
                return rewriter.create<SubFOp>(loc, ValueRange{ lhs, rhs }, rewriter.getNamedAttr("RelaxedPrecision", rewriter.getUnitAttr()));
            default:
                assert(false);
                return {};
            }
        }
        else
        {
            switch (pred)
            {
            case BinaryOpPredicate::ADD:
                return rewriter.create<AddIOp>(loc, lhs, rhs);
            case BinaryOpPredicate::DIV:
                return rewriter.create<SignedDivIOp>(loc, lhs, rhs);
            case BinaryOpPredicate::MOD:
                return rewriter.create<SignedRemIOp>(loc, lhs, rhs);
            case BinaryOpPredicate::MUL:
                return rewriter.create<MulIOp>(loc, lhs, rhs);
            case BinaryOpPredicate::SUB:
                return rewriter.create<SubIOp>(loc, lhs, rhs);
            case BinaryOpPredicate::LOGICAL_AND:
                return rewriter.create<AndOp>(loc, lhs, rhs);
            case BinaryOpPredicate::LOGICAL_OR:
                return rewriter.create<OrOp>(loc, lhs, rhs);
            default:
                assert(false);
                return {};
            }
        }
    }();

    rewriter.replaceOp(op.getOperation(), { result });

    return success();
}

LogicalResult GetElementOpLowering::matchAndRewrite(
    ValueGetElementOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });

    auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
    auto loaded = [&]() -> mlir::Value {
        auto v = op.value();

        if (auto shapedType = v.getType().dyn_cast<ShapedType>())
        {
            // this should probably be moved to the verifier
            assert(shapedType.getNumElements() == 1);

            llvm::SmallVector<mlir::Value, 4> indices{ (size_t)shapedType.getRank(), zero };

            if (shapedType.isa<MemRefType>())
            {
                return rewriter.create<memref::LoadOp>(loc, v, indices);
            }
            else if (shapedType.isa<TensorType>())
            {
                return rewriter.create<tensor::ExtractOp>(loc, v, indices);
            }
            else
            {
                throw std::logic_error("Unknown type of operand");
            }
        }
        else
        {
            return v;
        }
    }();

    rewriter.replaceOp(op.getOperation(), { loaded });

    return success();
}

LogicalResult GlobalOpLowering::matchAndRewrite(
    ValueGlobalOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });

    ValueGlobalOp::Adaptor adaptor(op);
    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        op,
        adaptor.sym_name(),
        rewriter.getStringAttr(op.external() ? "public" : "nested"),
        adaptor.type(),
        adaptor.value(),
        adaptor.constant());

    return success();
}

LogicalResult UnaryOpLowering::matchAndRewrite(
    ValueUnaryOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });

    auto True = rewriter.create<ConstantIntOp>(loc, 1, rewriter.getI1Type());
    auto loaded = op.input();

    auto result = [&]() -> mlir::Value {
        using vir::UnaryOpPredicate;
        switch (op.getPredicate())
        {
        case UnaryOpPredicate::NOT:
            return rewriter.create<XOrOp>(loc, loaded, True);
        default:
            assert(false);
        }
    }();

    rewriter.replaceOp(op.getOperation(), { result });

    return success();
}

LogicalResult LoadOpLowering::matchAndRewrite(
    ValueLoadOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });

    auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
    auto indexType = rewriter.getIndexType();

    llvm::SmallVector<mlir::Value, 4> resolvedIndices;
    for (auto index : op.indices())
    {
        if (index.getType().isIndex())
        {
            resolvedIndices.push_back(index);
        }
        else
        {
            resolvedIndices.push_back(
                rewriter.create<IndexCastOp>(loc, rewriter.create<vir::GetElementOp>(loc, index), indexType));
        }
    }

    auto result = rewriter.create<memref::LoadOp>(loc, op.memref(), resolvedIndices);
    auto allocRes = rewriter.create<memref::AllocaOp>(loc, op.getResult().getType().cast<MemRefType>());
    rewriter.replaceOp(op.getOperation(), { allocRes });
    (void)rewriter.create<memref::StoreOp>(loc, result, allocRes, ValueRange{ zero });

    return success();
}

LogicalResult StoreOpLowering::matchAndRewrite(
    ValueStoreOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });
    auto indexType = rewriter.getIndexType();

    llvm::SmallVector<mlir::Value, 4> resolvedIndices;
    for (auto index : op.indices())
    {
        if (index.getType().isIndex())
        {
            resolvedIndices.push_back(index);
        }
        else
        {
            resolvedIndices.push_back(
                rewriter.create<IndexCastOp>(loc,
                                             rewriter.create<vir::GetElementOp>(loc, index),
                                             indexType));
        }
    }

    (void)rewriter.create<memref::StoreOp>(loc,
                                           rewriter.create<vir::GetElementOp>(loc, op.value()),
                                           op.memref(),
                                           resolvedIndices);
    rewriter.eraseOp(op.getOperation());

    return success();
}

using ValueCmpOpPredicate = vir::CmpOpPredicate;

static ValueCmpOpPredicate NegateCmpOpPredicate(ValueCmpOpPredicate pred)
{
#define MAP_PREDICATE(v1, v2)     \
    case ValueCmpOpPredicate::v1: \
        return ValueCmpOpPredicate::v2

    switch (pred)
    {
        MAP_PREDICATE(EQ, EQ);
        MAP_PREDICATE(GE, LT);
        MAP_PREDICATE(GT, LE);
        MAP_PREDICATE(LE, GT);
        MAP_PREDICATE(LT, GE);
        MAP_PREDICATE(NE, NE);
    default:
        assert(false);
        throw std::logic_error("Unknown CmpOp predicate");
    }
#undef MAP_PREDICATE
}

static CmpFPredicate CmpOpPredicateToCmpFPredicate(ValueCmpOpPredicate pred)
{
#define MAP_PREDICATE(v)         \
    case ValueCmpOpPredicate::v: \
        return CmpFPredicate::U##v

    switch (pred)
    {
        MAP_PREDICATE(EQ);
        MAP_PREDICATE(GE);
        MAP_PREDICATE(GT);
        MAP_PREDICATE(LE);
        MAP_PREDICATE(LT);
        MAP_PREDICATE(NE);
    default:
        assert(false);
    }

#undef MAP_PREDICATE
}

static CmpIPredicate CmpOpPredicateToCmpIPredicate(ValueCmpOpPredicate pred)
{
#define MAP_PREDICATE(v1, v2)     \
    case ValueCmpOpPredicate::v1: \
        return CmpIPredicate::v2

    switch (pred)
    {
        MAP_PREDICATE(EQ, eq);
        MAP_PREDICATE(GE, sge);
        MAP_PREDICATE(GT, sgt);
        MAP_PREDICATE(LE, sle);
        MAP_PREDICATE(LT, slt);
        MAP_PREDICATE(NE, ne);
    default:
        assert(false);
    }

#undef MAP_PREDICATE
}

LogicalResult CmpOpLowering::matchAndRewrite(
    ValueCmpOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });
    auto lhs = op.lhs();
    auto rhs = op.rhs();

    auto result = [&]() -> mlir::Value {
        if (auto pred = op.getPredicate(); util::GetElementType(lhs.getType()).isa<FloatType>())
        {
            return rewriter.create<CmpFOp>(loc, CmpOpPredicateToCmpFPredicate(pred), lhs, rhs);
        }
        else
        {
            return rewriter.create<CmpIOp>(loc, CmpOpPredicateToCmpIPredicate(pred), lhs, rhs);
        }
    }();

    rewriter.replaceOp(op.getOperation(), { result });

    return success();
}

LogicalResult TerminatorLowering::matchAndRewrite(
    ValueTerminatorOp op,
    PatternRewriter& rewriter) const
{
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
    return success();
}

LogicalResult OffsetOpLowering::matchAndRewrite(
    ValueOffsetOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });
    auto indexType = rewriter.getIndexType();
    auto source = op.source();
    auto sourceType = source.getType().cast<mlir::MemRefType>();
    auto shape = sourceType.getShape();

    llvm::SmallVector<mlir::Value, 4> resolvedOffsets;
    llvm::SmallVector<mlir::Value, 4> strides(shape.size(), rewriter.create<ConstantIndexOp>(loc, 1));
    for (auto index : op.offsets())
    {
        if (index.getType().isIndex())
        {
            resolvedOffsets.push_back(index);
        }
        else
        {
            auto indexShape = index.getType().cast<mlir::ShapedType>().getShape();
            if (indexShape.size() == 0 || indexShape.size() == 1)
            {
                resolvedOffsets.push_back(
                    rewriter.create<IndexCastOp>(loc,
                                                 rewriter.create<vir::GetElementOp>(loc, index),
                                                 indexType));
            }
            else
            {
                assert(false && "Unknown index shape for offset op");
            }
        }
    }

    llvm::SmallVector<mlir::Value, 4> sizes;
    for (auto extent : shape)
    {
        sizes.push_back(rewriter.create<ConstantIndexOp>(loc, extent));
    }

    rewriter.replaceOp(op, { rewriter.create<memref::SubViewOp>(loc, op.getType(), source, resolvedOffsets, sizes, strides) });

    return success();
}

LogicalResult ViewOpLowering::matchAndRewrite(
    ValueViewOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });
    auto source = op.source();
    auto sourceType = op.getSourceMemRefType();
    auto shape = sourceType.getShape();

    auto isStaticCanonicalOrder = [](MemRefType t) {
        llvm::SmallVector<int64_t, 4> memRefStrides;
        int64_t globalOffset;
        if (failed(getStridesAndOffset(t, memRefStrides, globalOffset)))
        {
            return false;
        }
        int64_t prevStride = memRefStrides[0];
        for (auto stride : memRefStrides)
        {
            if (mlir::ShapedType::isDynamicStrideOrOffset(stride) || stride > prevStride)
            {
                return false;
            }
            prevStride = stride;
        }
        return true;
    };

    llvm::SmallVector<mlir::Value, 4> offsets, sizes, strides;
    for (auto offset : op.offsets())
    {
        if (auto r = mlir::dyn_cast<linalg::RangeOp>(offset.getDefiningOp()))
        {
            auto min = r.min();
            auto max = r.max();
            auto step = r.step();
            offsets.push_back(min.getDefiningOp()->getResult(0));
            sizes.push_back(max.getDefiningOp()->getResult(0));
            strides.push_back(step.getDefiningOp()->getResult(0));
        }
        else
        {
            return op.emitError("Bad offset operands for ViewOp");
        }
    }

    if (isStaticCanonicalOrder(sourceType))
    {
        bool isStaticShape = true;
        llvm::SmallVector<int64_t, 4> staticOffsets, staticSizes, staticStrides;
        for (auto [size, offset, stride] : llvm::zip(sizes, offsets, strides))
        {
            auto sizeOp = size.getDefiningOp<ConstantIndexOp>();
            auto offsetOp = offset.getDefiningOp<ConstantIndexOp>();
            auto strideOp = stride.getDefiningOp<ConstantIndexOp>();
            if (sizeOp && offsetOp && strideOp)
            {
                staticSizes.push_back(sizeOp.getValue());
                staticOffsets.push_back(offsetOp.getValue());
                staticStrides.push_back(strideOp.getValue());
            }
            else
            {
                isStaticShape = false;
                break;
            }
        }

        if (isStaticShape)
        {
            rewriter.replaceOpWithNewOp<memref::SubViewOp>(op, source, staticOffsets, staticSizes, staticStrides);
        }
        else
        {
            rewriter.replaceOpWithNewOp<memref::SubViewOp>(op, source, offsets, sizes, strides);
        }
    }
    else
    {
        llvm::SmallVector<mlir::Value, 4> sizes;
        llvm::SmallVector<mlir::Value, 4> strides;

        for (auto idx = 0; idx < op.getNumOffsets(); ++idx)
        {
            sizes.push_back(rewriter.create<ConstantIndexOp>(loc, shape[idx]));
            strides.push_back(rewriter.create<ConstantIndexOp>(loc, 1));
        }

        rewriter.replaceOpWithNewOp<memref::SubViewOp>(op, source, offsets, sizes, strides);
    }
    return success();
}

LogicalResult SliceOpLowering::matchAndRewrite(
    ValueSliceOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });
    auto indexType = rewriter.getIndexType();
    auto source = op.source();
    auto sourceType = source.getType().cast<mlir::MemRefType>();
    auto shape = sourceType.getShape();

    // Initialize to a full view (no sliced dimensions)
    llvm::SmallVector<mlir::Value, 4> offsets;
    llvm::SmallVector<mlir::Value, 4> sizes;
    llvm::SmallVector<mlir::Value, 4> strides;
    llvm::SmallVector<mlir::Value, 4> linalgSliceIndexings;
    for (auto extent : shape)
    {
        auto min = rewriter.create<ConstantIndexOp>(loc, 0);
        auto max = rewriter.create<ConstantIndexOp>(loc, extent);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        auto range = rewriter.create<linalg::RangeOp>(loc, min, max, step);
        offsets.push_back(min);
        sizes.push_back(max);
        strides.push_back(step);
        linalgSliceIndexings.push_back(range);
    }

    // Now for the single dimensions
    auto sliceDimensions = op.sliceDimensions().getValue();
    for (auto en : llvm::enumerate(op.offsets()))
    {
        auto i = en.index();
        auto index = en.value();
        auto dim = sliceDimensions[i].cast<IntegerAttr>().getInt();
        if (!index.getType().isIndex())
        {
            auto indexShape = index.getType().cast<mlir::ShapedType>().getShape();
            if (indexShape.size() == 0 || indexShape.size() == 1)
            {
                index = rewriter.create<IndexCastOp>(loc, rewriter.create<vir::GetElementOp>(loc, index), indexType);
            }
            else
            {
                return op.emitError("Unknown offset shape for slice op");
            }
        }
        offsets[dim] = index;
        sizes[dim] = rewriter.create<ConstantIndexOp>(loc, 1);
        linalgSliceIndexings[dim] = rewriter.create<ConstantIndexOp>(loc, 0);
    }

    auto view = rewriter.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
    rewriter.replaceOp(op, { rewriter.create<memref::SubViewOp>(loc, view, linalgSliceIndexings, sizes, strides) });

    return success();
}

LogicalResult MergeDimOpLowering::matchAndRewrite(
    ValueMergeDimOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });
    auto source = op.source();

    auto resultType = op.getSourceMemRefType();
    auto execTarget = irutil::ResolveExecutionTarget(op).value_or(kDefaultExecutionTarget);
    if (execTarget == vir::ExecutionTarget::CPU)
    {
        resultType = MemRefType::Builder(resultType).setMemorySpace(0);
    }

    auto dim1 = static_cast<int64_t>(op.dim1());
    auto dim2 = static_cast<int64_t>(op.dim2());
    std::vector<mlir::ReassociationIndices> mergeIndices;
    auto rank = resultType.getRank();
    for (int64_t i = 0; i < rank; ++i)
    {
        if (i != dim2)
        {
            mlir::ReassociationIndices thisDimIndices;
            thisDimIndices.push_back(i);
            if (i == dim1)
            {
                thisDimIndices.push_back(dim2);
            }
            mergeIndices.push_back(thisDimIndices);
        }
    }

    rewriter.replaceOpWithNewOp<memref::CollapseShapeOp>(op, source, mergeIndices);

    return success();
}

LogicalResult SplitDimOpLowering::matchAndRewrite(
    ValueSplitDimOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });
    auto source = op.source();

    auto sourceType = op.getSourceMemRefType();
    auto resultType = op.getType();
    auto elemTy = sourceType.getElementType();

    // TODO: switch to using a linalg.reshape at some point
    // cast to a value with type `memref<total_size x elem_type>` (via `memref<* x elem_type>`)
    mlir::Value ptr = rewriter.create<memref::CastOp>(loc, source, mlir::UnrankedMemRefType::get(elemTy, sourceType.getMemorySpace()));
    auto result = rewriter.create<memref::CastOp>(loc, ptr, resultType);
    rewriter.replaceOp(op, { result });

    return success();
}

LogicalResult ReorderOpLowering::matchAndRewrite(
    ValueReorderOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });
    auto source = op.source();

    // TODO: switch to using a linalg.transpose at some point
    // just use a memref cast for now
    auto sourceType = op.getSourceMemRefType();
    auto elemTy = sourceType.getElementType();
    auto resultType = op.getType();

    // cast to a value with type `memref<total_size x elem_type>` (via `memref<* x elem_type>`)
    mlir::Value ptr = rewriter.create<memref::CastOp>(loc, source, mlir::UnrankedMemRefType::get(elemTy, sourceType.getMemorySpace()));
    auto result = rewriter.create<memref::CastOp>(loc, ptr, resultType);
    rewriter.replaceOp(op, { result });

    return success();
}

LogicalResult ReshapeOpLowering::matchAndRewrite(
    ValueReshapeOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });
    auto source = op.source();

    auto sourceType = op.getSourceMemRefType();
    auto elemTy = sourceType.getElementType();
    auto resultType = op.getType();

    // cast to a value with type `memref<total_size x elem_type>` (via `memref<* x elem_type>`)
    mlir::Value ptr = rewriter.create<memref::CastOp>(loc, source, mlir::UnrankedMemRefType::get(elemTy, sourceType.getMemorySpace()));
    auto result = rewriter.create<memref::CastOp>(loc, ptr, resultType);
    rewriter.replaceOp(op, { result });

    return success();
}

/// We vectorize a `reduce` op by first turning it into a reduction over vector values,
/// and then performing a final ("horizontal") reduction over the result
///
/// Example: "sum" as a reduction
/// ```
/// A = ...
/// sum = reduce(A, 0)
///   (a, p) {
///     s = add(a, p)
///     yield s
/// }
/// ```
///
/// Abbreviated IR for the result of vectorization:
/// ```
/// Ā = <vector-chunked version of A>
/// s̄ = broadcast(s, vectorSize)
/// x̄ = reduce(Ā, s̄)
///     (ā, p̄) { yield add(ā, p̄) }
/// x = reduce(x̄, s)
///     (a, p) { yield add(a, p) }
/// ```
///
LogicalResult ReduceOpVectorization::matchAndRewrite(
    ValueReduceOp op,
    PatternRewriter& rewriter) const
{
    auto vectorizationInfoIdentifier = rewriter.getIdentifier(ir::executionPlan::VectorizationInfoAttr::getKeyName());
    auto vectorizationInfoAttr = op->getAttrOfType<ir::executionPlan::VectorizationInfoAttr>(vectorizationInfoIdentifier);
    if (!vectorizationInfoAttr)
    {
        return success(); // no vectorization -- done
    }

    [[maybe_unused]] int vectorBytes = vectorizationInfoAttr.getValue().vectorBytes;
    [[maybe_unused]] int vectorUnits = vectorizationInfoAttr.getValue().vectorUnitCount;

    auto loc = op.getLoc();
    auto input = op.input();
    auto inputType = op.getShapedType();

    auto elementBitWidth = inputType.getElementTypeBitWidth();
    auto elementByteWidth = elementBitWidth / 8;
    auto elementsPerVector = vectorBytes / elementByteWidth;

    auto elementType = inputType.getElementType();
    auto rank = inputType.getRank();
    if (rank != 1)
    {
        return op.emitError("Can only reduce a rank-1 memref");
    }

    auto initialValue = op.getInitialValue();
    auto oldInputValue = op.getInputValueVar();
    auto oldInductionValue = op.getInductionValue();

    auto vectorType = mlir::VectorType::get({ elementsPerVector }, elementType);

    // emit "parallel" part
    auto vecInit = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, initialValue);
    auto parallelReduce = rewriter.create<ValueReduceOp>(loc, input, vecInit);
    {
        auto newInputValue = parallelReduce.getInputValueVar();
        auto newInductionValue = parallelReduce.getInductionValue();

        std::vector<BlockAndValueMapping> laneMappings(elementsPerVector);
        VectorizedOpMap vectorizedOps;
        vectorizedOps.Map(oldInputValue, newInputValue);
        vectorizedOps.Map(oldInductionValue, newInductionValue);

        auto body = parallelReduce.getBody();
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(body);

            // Clone reduction op body
            for (auto& op : *op.getBody())
            {
                if (isa<vir::YieldOp>(op))
                {
                    // the vectorizedOps entry is keyed off the original (non-vectorized) op
                    auto yieldValue = vectorizedOps.Lookup(op.getOperand(0).getDefiningOp());
                    if (!yieldValue)
                        return op.emitError("Couldn't find vectorized yield value");

                    rewriter.create<vir::YieldOp>(op.getLoc(), yieldValue->GetVectorResult());
                }
                else
                {
                    auto newOp = VectorizeOp(rewriter,
                                             &op,
                                             vectorizedOps,
                                             laneMappings,
                                             newInductionValue,
                                             1,
                                             elementsPerVector);

                    if (!newOp.has_value() || !newOp->HasVectorType())
                    {
                        llvm_unreachable("Couldn't vectorize reduction op");
                        return op.emitError("Couldn't vectorize reduction op");
                    }
                    vectorizedOps.Map(&op, *newOp);
                }
            }
        }
    }
    parallelReduce->setAttr("parallelReduction", rewriter.getUnitAttr());

    auto horizontalReduce = rewriter.create<ValueReduceOp>(loc, parallelReduce.getResult(), initialValue);
    {
        BlockAndValueMapping operandMap;
        auto newInputValue2 = horizontalReduce.getInputValueVar();
        auto newInductionValue2 = horizontalReduce.getInductionValue();
        operandMap.map(oldInputValue, newInputValue2);
        operandMap.map(oldInductionValue, newInductionValue2);
        auto body = horizontalReduce.getBody();
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(body);

            // Clone reduction op body
            for (auto& op : *op.getBody())
            {
                rewriter.clone(op, operandMap);
            }
        }
    }
    horizontalReduce->setAttr("horizontalReduction", rewriter.getUnitAttr());

    rewriter.replaceOp(op, horizontalReduce.getResult());
    return success();
}

LogicalResult ReduceOpLowering::matchAndRewrite(
    ValueReduceOp op,
    PatternRewriter& rewriter) const
{
    // There are 3 "formats" to deal with during lowering:
    // "normal" scalar result from memref
    // "vectorized" vector result from memref
    // "horizontal" scalar result from vector

    bool isParallelReduction = static_cast<bool>(op->getAttrOfType<UnitAttr>("parallelReduction"));
    bool isHorizontalReduction = static_cast<bool>(op->getAttrOfType<UnitAttr>("horizontalReduction"));

    auto loc = op.getLoc();
    auto input = op.input();
    auto inputType = op.getShapedType();
    auto rank = inputType.getRank();
    if (rank != 1)
    {
        return op.emitError("Can only reduce a rank-1 memref");
    }

    auto initialValue = op.getInitialValue();
    auto initialValueType = initialValue.getType();
    auto vectorSize = 1;
    if (auto vectorType = initialValueType.dyn_cast<ShapedType>())
    {
        vectorSize = vectorType.getShape()[0];
    }
    auto stepValue = isParallelReduction ? vectorSize : 1;

    auto oldInputValue = op.getInputValueVar();
    auto oldInductionValue = op.getInductionValue();
    auto oldTerminator = op.getBody()->getTerminator();
    auto oldYieldValue = oldTerminator->getOperand(0); // TODO: add "get result value" helper to ReduceOp

    // Check for trivial reductions of the form bin_op(arg1, arg2)
    if (isHorizontalReduction)
    {
        // vector reduce ops:
        // fp: add/mul/min/max
        // int: add/mul/min/max/and/or/xor
        if (auto yieldValueOp = oldYieldValue.getDefiningOp())
        {
            auto arg1 = op.getInputValueVar();
            auto arg2 = op.getInductionValue();
            if (auto binOp = dyn_cast<ValueBinOp>(yieldValueOp))
            {
                // Look for sequences like:
                //
                // %4 = "accv.bin_op"(%arg2, %arg3) {predicate = 0 : i64} : (f32, f32) -> f32
                // "accv.yield"(%4) : (f32) -> ()

                if ((binOp.lhs() == arg1 && binOp.rhs() == arg2) || (binOp.lhs() == arg2 && binOp.rhs() == arg1))
                {
                    using accera::ir::value::BinaryOpPredicate;
                    auto pred = binOp.predicate();
                    std::string opName = "";
                    switch (pred)
                    {
                    case BinaryOpPredicate::ADD:
                        opName = "add";
                        break;
                    case BinaryOpPredicate::MUL:
                        opName = "mul";
                        break;
                    case BinaryOpPredicate::SUB:
                        opName = "sub";
                        break;
                    default:
                        break;
                    }

                    if (!opName.empty())
                    {
                        // We can use the init value for floating-point add and mul
                        if (initialValueType.isa<mlir::FloatType>() && (pred == BinaryOpPredicate::ADD || pred == BinaryOpPredicate::MUL))
                        {
                            auto result = rewriter.create<mlir::vector::ReductionOp>(loc, op.result().getType(), rewriter.getStringAttr(opName), op.input(), op.initArg());
                            rewriter.replaceOp(op, { result });
                        }
                        else
                        {
                            auto result = rewriter.create<mlir::vector::ReductionOp>(loc, op.result().getType(), rewriter.getStringAttr(opName), op.input(), llvm::None);
                            rewriter.replaceOp(op, { result });
                        }
                        return success();
                    }
                }
            }
            else if (auto selectOp = dyn_cast<mlir::SelectOp>(yieldValueOp))
            {
                // Look for sequences like:
                //
                // %4 = "accv.cmp"(%arg2, %arg3) {predicate = 4 : i64} : (f32, f32) -> i1
                // %5 = select %4, %arg2, %arg3 : f32
                // "accv.yield"(%5) : (f32) -> ()
                if (auto cmpOp = dyn_cast_or_null<ValueCmpOp>(selectOp.getCondition().getDefiningOp()))
                {
                    if (((cmpOp.lhs() == arg1 && cmpOp.rhs() == arg2) || (cmpOp.lhs() == arg2 && cmpOp.rhs() == arg1)) && ((selectOp.getTrueValue() == arg1 && selectOp.getFalseValue() == arg2) || (selectOp.getTrueValue() == arg2 && selectOp.getFalseValue() == arg1)))
                    {
                        auto pred = cmpOp.getPredicate();
                        if (cmpOp.lhs() == selectOp.getFalseValue())
                        {
                            pred = NegateCmpOpPredicate(pred);
                        }

                        mlir::Value result = {};
                        switch (pred)
                        {
                        case ValueCmpOpPredicate::EQ:
                            llvm::errs() << "Found reduce-trivial-first!\n";
                            break;
                        case ValueCmpOpPredicate::NE:
                            llvm::errs() << "Found reduce-trivial-last!\n";
                            break;
                        case ValueCmpOpPredicate::LT:
                            [[fallthrough]];
                        case ValueCmpOpPredicate::LE:
                            // min
                            result = rewriter.create<mlir::vector::ReductionOp>(loc, op.result().getType(), rewriter.getStringAttr("min"), op.input(), llvm::None);
                            break;
                        case ValueCmpOpPredicate::GT:
                            [[fallthrough]];
                        case ValueCmpOpPredicate::GE:
                            // max
                            result = rewriter.create<mlir::vector::ReductionOp>(loc, op.result().getType(), rewriter.getStringAttr("max"), op.input(), llvm::None);
                            break;
                        }

                        if (result)
                        {
                            rewriter.replaceOp(op, result);
                            return success();
                        }
                    }
                }
            }
        }

        // If we're here, we didn't convert the reduce op to a vector::reduction op
        // TODO: manually unroll the loop or do a logN reduction
    }

    auto size = inputType.getShape()[0];
    auto loopSize = isParallelReduction ? RoundDownToMultiple(size, vectorSize) : size;
    auto remainder = size - loopSize;
    auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<ConstantIndexOp>(loc, loopSize);
    auto step = rewriter.create<ConstantIndexOp>(loc, stepValue);
    auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step, initialValue);
    auto loopBody = loop.getBody();
    {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loopBody);

        // map the "input element value" to "input[i]"
        BlockAndValueMapping operandMap;
        mlir::Value element;
        if (isParallelReduction)
        {
            auto elementType = inputType.getElementType();
            auto zero = rewriter.create<mlir::ConstantOp>(loc, elementType, rewriter.getZeroAttr(elementType));
            auto vectorType = initialValueType;
            element = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, zero);
            for (int64_t i = 0; i < vectorSize; ++i)
            {
                auto offset = rewriter.create<mlir::ConstantIndexOp>(loc, i);
                auto offsetInductionVar = rewriter.create<mlir::AddIOp>(loc, loop.getInductionVar(), offset);
                auto elementLoad = rewriter.create<memref::LoadOp>(loc, input, ValueRange{ offsetInductionVar });
                element = rewriter.create<mlir::vector::InsertElementOp>(loc, elementLoad.getResult(), element, i);
            }
        }
        else if (isHorizontalReduction)
        {
            // extract element from input vector
            auto laneIndex = rewriter.create<mlir::IndexCastOp>(loc, loop.getInductionVar(), rewriter.getI32Type()).getResult();
            element = rewriter.create<mlir::vector::ExtractElementOp>(loc, input, laneIndex).getResult();
        }
        else
        {
            element = rewriter.create<memref::LoadOp>(loc, input, loop.getInductionVar()).getResult();
        }

        auto newInductionValue = loop.getRegionIterArgs()[0];
        operandMap.map(oldInputValue, element);
        operandMap.map(oldInductionValue, newInductionValue);

        // Copy reduction op body
        for (auto& op : op.getBody()->without_terminator())
        {
            rewriter.clone(op, operandMap);
        }

        // now add an appropriate yield operation
        auto newYieldValue = operandMap.lookupOrDefault(oldYieldValue);
        rewriter.create<scf::YieldOp>(loc, newYieldValue);
    }

    mlir::Value result = loop.getResults()[0];

    // Add remainder to value yielded by the vectorized loop
    if (remainder > 0)
    {
        assert(isParallelReduction);

        mlir::Value element = initialValue;
        for (int64_t i = 0; i < remainder; ++i)
        {
            auto offset = rewriter.create<mlir::ConstantIndexOp>(loc, i);
            auto offsetInductionVar = rewriter.create<mlir::AddIOp>(loc, upperBound, offset);
            auto elementLoad = rewriter.create<memref::LoadOp>(loc, input, ValueRange{ offsetInductionVar });
            element = rewriter.create<mlir::vector::InsertElementOp>(loc, elementLoad.getResult(), element, i);
        }

        BlockAndValueMapping operandMap;
        operandMap.map(oldInputValue, element);
        operandMap.map(oldInductionValue, result);

        // Copy reduction op body
        for (auto& op : op.getBody()->without_terminator())
        {
            rewriter.clone(op, operandMap);
        }

        result = operandMap.lookupOrNull(oldYieldValue);
        assert(result);
    }

    rewriter.replaceOp(op, result);
    return success();
}

LogicalResult ReferenceGlobalOpLowering::matchAndRewrite(
    ValueReferenceGlobalOp op,
    PatternRewriter& rewriter) const
{
    auto loc = rewriter.getFusedLoc({ op.getLoc(), RC_FILE_LOC(rewriter) });

    ValueReferenceGlobalOp::Adaptor adaptor(op);

    mlir::Value getGlobalOpValue = rewriter.create<memref::GetGlobalOp>(
        loc,
        static_cast<MemRefType>(MemRefType::Builder{ op.getType() }.setAffineMaps({})),
        adaptor.global_name());

    rewriter.replaceOpWithNewOp<memref::CastOp>(
        op,
        getGlobalOpValue,
        op.getType());

    return success();
}

/// We vectorize a `map_reduce` op by first turning it into a map-reduce over vector values,
/// and then performing a final ("horizontal") reduction over the result
///
/// Example: square the elements of an array and return the sum
/// ```
/// A = ...
/// sum_of_sqares = map_reduce(A, 0)
///   (a) {
///     sq = mul(a, a)
///     yield sq
/// }
///   (a, p) {
///     s = add(a, p)
///     yield s
/// }
/// ```
///
/// Abbreviated IR for the result of vectorization:
/// ```
/// Ā = <vector-chunked version of A>
/// s = 0
/// s̄ = broadcast(s, vectorSize)
/// x̄ = map_reduce(Ā, s̄)
///     (ā) { yield mul(ā, ā) }
///     (ā, p̄) { yield add(ā, p̄) }
/// x = reduce(x̄, s)
///     (a, p) { yield add(a, p) }
/// ```
///
LogicalResult MapReduceOpVectorization::matchAndRewrite(
    ValueMapReduceOp op,
    PatternRewriter& rewriter) const
{
    auto vectorizationInfoIdentifier = rewriter.getIdentifier(ir::executionPlan::VectorizationInfoAttr::getKeyName());
    auto vectorizationInfoAttr = op->getAttrOfType<ir::executionPlan::VectorizationInfoAttr>(vectorizationInfoIdentifier);
    if (!vectorizationInfoAttr)
    {
        return success(); // no vectorization -- done
    }

    [[maybe_unused]] int vectorBytes = vectorizationInfoAttr.getValue().vectorBytes;
    [[maybe_unused]] int vectorUnits = vectorizationInfoAttr.getValue().vectorUnitCount;

    auto loc = op.getLoc();
    auto input = op.input();
    auto inputType = op.getShapedType();

    auto elementBitWidth = inputType.getElementTypeBitWidth();
    auto elementByteWidth = elementBitWidth / 8;
    auto elementsPerVector = vectorBytes / elementByteWidth;

    auto elementType = inputType.getElementType();
    auto rank = inputType.getRank();
    if (rank != 1)
    {
        return op.emitError("Can only reduce a rank-1 memref");
    }

    auto initialValue = op.getInitialValue();
    auto oldReduceInputValue = op.getReduceInputValueVar();
    auto oldInductionValue = op.getReduceInductionValue();

    auto vectorType = mlir::VectorType::get({ elementsPerVector }, elementType);
    auto vecInit = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, initialValue);
    auto parallelReduce = rewriter.create<ValueMapReduceOp>(loc, input, vecInit);

    // emit "parallel" map part
    auto mapBody = parallelReduce.getMapBody();
    {
        auto oldMapInputValue = op.getMapInputValueVar();
        auto oldTerminator = op.getMapBody()->getTerminator();
        auto oldYieldValue = oldTerminator->getOperand(0);

        auto newInputValue = parallelReduce.getMapInputValueVar();

        std::vector<BlockAndValueMapping> laneMappings(elementsPerVector);
        VectorizedOpMap vectorizedOps;
        vectorizedOps.Map(oldMapInputValue, newInputValue);

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mapBody);

        // Clone map op body
        for (auto& subOp : op.getMapBody()->without_terminator())
        {
            auto newOp = VectorizeOp(rewriter,
                                     &subOp,
                                     vectorizedOps,
                                     laneMappings,
                                     nullptr,
                                     1,
                                     elementsPerVector);

            if (!newOp.has_value() || !newOp->HasVectorType())
            {
                llvm_unreachable("Couldn't vectorize map op");
                return op.emitError("Couldn't vectorize map op");
            }

            vectorizedOps.Map(&subOp, *newOp);
        }

        auto yieldValue = vectorizedOps.Lookup(oldYieldValue);
        if (!yieldValue)
        {
            llvm_unreachable("Couldn't find vectorized yield value");
            return op.emitError("Couldn't find vectorized yield value");
        }
        rewriter.create<vir::YieldOp>(op.getLoc(), yieldValue->GetVectorResult());
    }

    // emit "parallel" reduce part
    auto reduceBody = parallelReduce.getReduceBody();
    {
        auto newInputValue = parallelReduce.getReduceInputValueVar();
        auto newInductionValue = parallelReduce.getReduceInductionValue();

        std::vector<BlockAndValueMapping> laneMappings(elementsPerVector);
        VectorizedOpMap vectorizedOps;
        vectorizedOps.Map(oldReduceInputValue, newInputValue);
        vectorizedOps.Map(oldInductionValue, newInductionValue);

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(reduceBody);

        // Clone reduction op body
        for (auto& op : *op.getReduceBody())
        {
            // TODO: can we do this in VectorizeOp?
            if (isa<vir::YieldOp>(op))
            {
                auto yieldValue = vectorizedOps.Lookup(op.getOperand(0));
                if (!yieldValue)
                    return op.emitError("Couldn't find vectorized yield value");

                rewriter.create<vir::YieldOp>(op.getLoc(), yieldValue->GetVectorResult());
            }
            else
            {
                auto newOp = VectorizeOp(rewriter,
                                         &op,
                                         vectorizedOps,
                                         laneMappings,
                                         newInductionValue,
                                         1,
                                         elementsPerVector);

                if (!newOp.has_value() || !newOp->HasVectorType())
                {
                    llvm_unreachable("Couldn't vectorize reduction op");
                    return op.emitError("Couldn't vectorize reduction op");
                }

                vectorizedOps.Map(&op, *newOp);
            }
        }
    }
    parallelReduce->setAttr("parallelReduction", rewriter.getUnitAttr());

    auto horizontalReduce = rewriter.create<ValueReduceOp>(loc, parallelReduce.getResult(), initialValue);
    {
        BlockAndValueMapping operandMap;
        auto newInputValue = horizontalReduce.getInputValueVar();
        auto newInductionValue = horizontalReduce.getInductionValue();
        operandMap.map(oldReduceInputValue, newInputValue);
        operandMap.map(oldInductionValue, newInductionValue);
        auto body = horizontalReduce.getBody();
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(body);

            // Clone reduction op body
            for (auto& op : *op.getReduceBody())
            {
                rewriter.clone(op, operandMap);
            }
        }
    }
    horizontalReduce->setAttr("horizontalReduction", rewriter.getUnitAttr());

    rewriter.replaceOp(op, horizontalReduce.getResult());
    return success();
}

LogicalResult MapReduceOpLowering::matchAndRewrite(
    ValueMapReduceOp op,
    PatternRewriter& rewriter) const
{
    // There are 2 "formats" to deal with during lowering:
    // "normal" scalar result from memref
    // "vectorized" vector result from memref

    bool isParallelReduction = static_cast<bool>(op->getAttrOfType<UnitAttr>("parallelReduction"));

    auto loc = op.getLoc();
    auto input = op.input();
    auto inputType = op.getShapedType();
    auto rank = inputType.getRank();
    if (rank != 1)
    {
        return op.emitError("Can only map-reduce a rank-1 memref");
    }

    auto initialValue = op.getInitialValue();
    auto initialValueType = initialValue.getType();
    auto vectorSize = 1;
    if (auto vectorType = initialValueType.dyn_cast<ShapedType>())
    {
        vectorSize = vectorType.getShape()[0];
    }
    auto stepValue = isParallelReduction ? vectorSize : 1;

    auto size = inputType.getShape()[0];
    auto loopSize = isParallelReduction ? RoundDownToMultiple(size, vectorSize) : size;
    auto remainder = size - loopSize;
    auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<ConstantIndexOp>(loc, loopSize);
    auto step = rewriter.create<ConstantIndexOp>(loc, stepValue);

    // Map loop values
    auto oldMapInputValue = op.getMapInputValueVar();
    auto oldMapTerminator = op.getMapBody()->getTerminator();
    auto oldMapYieldValue = oldMapTerminator->getOperand(0);

    // Reduction loop values
    auto oldReduceInputValue = op.getReduceInputValueVar();
    auto oldInductionValue = op.getReduceInductionValue();
    auto oldReduceTerminator = op.getReduceBody()->getTerminator();
    auto oldReduceYieldValue = oldReduceTerminator->getOperand(0);

    auto mapReduceLoop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step, initialValue);
    auto mapReduceLoopBody = mapReduceLoop.getBody();
    {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mapReduceLoopBody);

        // map the "input element value" to "input[i]"
        BlockAndValueMapping mapOperandMap;
        mlir::Value mapElement;
        if (isParallelReduction)
        {
            auto elementType = inputType.getElementType();
            auto zero = rewriter.create<mlir::ConstantOp>(loc, elementType, rewriter.getZeroAttr(elementType));
            auto vectorType = initialValueType;
            mapElement = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, zero);
            for (int64_t i = 0; i < vectorSize; ++i)
            {
                auto offset = rewriter.create<mlir::ConstantIndexOp>(loc, i);
                auto offsetInductionVar = rewriter.create<mlir::AddIOp>(loc, mapReduceLoop.getInductionVar(), offset);
                auto elementLoad = rewriter.create<memref::LoadOp>(loc, input, ValueRange{ offsetInductionVar });
                mapElement = rewriter.create<mlir::vector::InsertElementOp>(loc, elementLoad.getResult(), mapElement, i);
            }
        }
        else
        {
            mapElement = rewriter.create<memref::LoadOp>(loc, input, mapReduceLoop.getInductionVar()).getResult();
        }

        mapOperandMap.map(oldMapInputValue, mapElement);

        // Clone map op body
        for (auto& op : op.getMapBody()->without_terminator())
        {
            rewriter.clone(op, mapOperandMap);
        }

        auto newMapYieldValue = mapOperandMap.lookupOrDefault(oldMapYieldValue);

        // Store the mapped value back to memory
        if (isParallelReduction)
        {
            for (int64_t i = 0; i < vectorSize; ++i)
            {
                auto element = rewriter.create<mlir::vector::ExtractElementOp>(op.getLoc(), newMapYieldValue, i);
                auto offset = rewriter.create<mlir::ConstantIndexOp>(loc, i);
                auto offsetInductionVar = rewriter.create<mlir::AddIOp>(loc, mapReduceLoop.getInductionVar(), offset);
                rewriter.create<memref::StoreOp>(loc, element, input, ValueRange{ offsetInductionVar });
            }
        }
        else
        {
            rewriter.create<memref::StoreOp>(loc, newMapYieldValue, input, mapReduceLoop.getInductionVar());
        }

        // End of "map" part

        // map the "input element value" to the output of the "map" part
        BlockAndValueMapping reduceOperandMap;
        auto reduceElement = newMapYieldValue;

        auto newInductionValue = mapReduceLoop.getRegionIterArgs()[0];
        reduceOperandMap.map(oldReduceInputValue, reduceElement);
        reduceOperandMap.map(oldInductionValue, newInductionValue);

        // Clone reduction op body
        for (auto& op : op.getReduceBody()->without_terminator())
        {
            rewriter.clone(op, reduceOperandMap);
        }

        // now add an appropriate yield operation
        auto newReduceYieldValue = reduceOperandMap.lookupOrDefault(oldReduceYieldValue);
        rewriter.create<scf::YieldOp>(loc, newReduceYieldValue);
    }

    mlir::Value result = mapReduceLoop.getResults()[0];

    if (remainder > 0)
    {
        assert(isParallelReduction);

        // map the "input element value" to "input[i]"
        BlockAndValueMapping mapOperandMap;
        auto elementType = inputType.getElementType();
        auto zero = rewriter.create<mlir::ConstantOp>(loc, elementType, rewriter.getZeroAttr(elementType));
        auto vectorType = initialValueType;
        mlir::Value mapElement = rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, zero);
        for (int64_t i = 0; i < remainder; ++i)
        {
            auto offset = rewriter.create<mlir::ConstantIndexOp>(loc, i);
            auto offsetInductionVar = rewriter.create<mlir::AddIOp>(loc, upperBound, offset);
            auto elementLoad = rewriter.create<memref::LoadOp>(loc, input, ValueRange{ offsetInductionVar });
            mapElement = rewriter.create<mlir::vector::InsertElementOp>(loc, elementLoad.getResult(), mapElement, i);
        }

        mapOperandMap.map(oldMapInputValue, mapElement);

        // Clone map op body
        for (auto& op : op.getMapBody()->without_terminator())
        {
            rewriter.clone(op, mapOperandMap);
        }

        // TODO: need to zero out the lanes we aren't using (by copying from the init value)
        auto newMapYieldValue = mapOperandMap.lookupOrDefault(oldMapYieldValue);
        auto maskedMapYieldValue = initialValue;
        for (int64_t i = 0; i < remainder; ++i)
        {
            auto element = rewriter.create<mlir::vector::ExtractElementOp>(op.getLoc(), newMapYieldValue, i);
            auto offset = rewriter.create<mlir::ConstantIndexOp>(loc, i);
            auto offsetInductionVar = rewriter.create<mlir::AddIOp>(loc, upperBound, offset);
            rewriter.create<memref::StoreOp>(loc, element, input, ValueRange{ offsetInductionVar });

            maskedMapYieldValue = rewriter.create<mlir::vector::InsertElementOp>(loc, element, maskedMapYieldValue, i);
        }

        // Add remainder to value yielded by the vectorized loop
        mlir::Value reduceElement = maskedMapYieldValue;

        BlockAndValueMapping reduceOperandMap;
        reduceOperandMap.map(oldReduceInputValue, reduceElement);
        reduceOperandMap.map(oldInductionValue, result);

        // Copy reduction op body
        for (auto& op : op.getReduceBody()->without_terminator())
        {
            rewriter.clone(op, reduceOperandMap);
        }

        result = reduceOperandMap.lookupOrNull(oldReduceYieldValue);
        assert(result);
    }

    rewriter.replaceOp(op, result);
    return success();
}

using ValueReduceMaxOp = vir::ReduceMaxOp;
LogicalResult ReduceMaxOpLowering::matchAndRewrite(
    ValueReduceMaxOp op,
    PatternRewriter& rewriter) const
{
    auto loc = op.getLoc();
    auto input = op.input();
    auto type = input.getType();
    assert(type.isa<mlir::MemRefType>() && "Input must be a memref");
    auto memRefType = type.cast<mlir::MemRefType>();
    int64_t rank = memRefType.getRank();
    if (rank != 1)
    {
        return op.emitError("Can only reduce a rank-1 memref");
    }

    mlir::Value memrefToCast = input;
    mlir::Value loadedVector = nullptr;
    if (!memRefType.getAffineMaps().empty())
    {
        auto elementType = memRefType.getElementType();
        auto vectorType = mlir::VectorType::get(memRefType.getShape(), elementType);
        auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
        loadedVector = rewriter.create<mlir::vector::TransferReadOp>(loc, vectorType, memrefToCast, mlir::ValueRange{ zero });
    }
    else
    {
        auto castMemRefVector = rewriter.create<mlir::vector::TypeCastOp>(loc, memrefToCast);
        loadedVector = rewriter.create<memref::LoadOp>(loc, castMemRefVector, llvm::None);
    }
    auto result = rewriter.create<mlir::vector::ReductionOp>(loc, op.result().getType(), rewriter.getStringAttr("max"), loadedVector, llvm::None);
    rewriter.replaceOp(op, { result });
    return success();
}

LogicalResult EnterProfileRegionOpLowering::matchAndRewrite(EnterProfileRegionOp op, PatternRewriter& rewriter) const
{
    if (!enableProfiling)
    {
        // if profiling is disabled, just remove this op
        rewriter.eraseOp(op);
        return success();
    }

    auto loc = op.getLoc();
    auto module = op->getParentOfType<mlir::ModuleOp>();

    ProfileRegions regions(module);
    auto regionName = op.regionName().str();
    if (regions.counters.count(regionName) == 0)
    {
        op.emitError("No counters exist for region");
        return failure();
    }

    auto startTimeGlobal = regions.counters[regionName].startTime;
    mlir::Value startTimeRef = rewriter.create<vir::ReferenceGlobalOp>(loc, startTimeGlobal);

    // get current time and store it in the startTime entry
    mlir::Value currentTime = rewriter.create<vir::GetTimeOp>(loc);
    rewriter.create<vir::CopyOp>(loc, currentTime, startTimeRef);
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ExitProfileRegionOpLowering::matchAndRewrite(ExitProfileRegionOp op, PatternRewriter& rewriter) const
{
    if (!enableProfiling)
    {
        // if profiling is disabled, just remove this op
        rewriter.eraseOp(op);
        return success();
    }

    auto loc = op.getLoc();
    auto module = op->getParentOfType<mlir::ModuleOp>();

    ProfileRegions regions(module);
    auto regionName = op.regionName().str();
    if (regions.counters.count(regionName) == 0)
    {
        op.emitError("No counters exist for region");
        return failure();
    }

    auto startTimeGlobal = regions.counters[regionName].startTime;
    mlir::Value startTimeRef = rewriter.create<vir::ReferenceGlobalOp>(loc, startTimeGlobal);

    auto totalTimeGlobal = regions.counters[regionName].time;
    mlir::Value totalTimeRef = rewriter.create<vir::ReferenceGlobalOp>(loc, totalTimeGlobal);

    auto countGlobal = regions.counters[regionName].count;
    mlir::Value countRef = rewriter.create<vir::ReferenceGlobalOp>(loc, countGlobal);

    mlir::Value startTime = rewriter.create<vir::GetElementOp>(loc, startTimeRef);
    mlir::Value currentTime = rewriter.create<vir::GetTimeOp>(loc);
    mlir::Value duration = rewriter.create<vir::BinOp>(loc, vir::BinaryOpPredicate::SUB, currentTime, startTime);
    mlir::Value prevTotalTime = rewriter.create<vir::GetElementOp>(loc, totalTimeRef);
    mlir::Value totalTime = rewriter.create<vir::BinOp>(loc, vir::BinaryOpPredicate::ADD, prevTotalTime, duration);
    rewriter.create<vir::CopyOp>(loc, totalTime, totalTimeRef);

    mlir::Value prevCount = rewriter.create<vir::GetElementOp>(loc, countRef);
    auto one = rewriter.create<ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    mlir::Value newCount = rewriter.create<vir::BinOp>(loc, vir::BinaryOpPredicate::ADD, prevCount, one);
    rewriter.create<vir::CopyOp>(loc, newCount, countRef);

    rewriter.eraseOp(op);
    return success();
}

LogicalResult PrintProfileResultsOpLowering::matchAndRewrite(PrintProfileResultsOp op, PatternRewriter& rewriter) const
{
    if (!enableProfiling)
    {
        // if profiling is disabled, just remove this op
        rewriter.eraseOp(op);
        return success();
    }

    auto loc = op.getLoc();
    auto module = op->getParentOfType<mlir::ModuleOp>();

    ProfileRegions regions(module);

    // foreach region, print count and number
    for (auto [name, counters] : regions.counters)
    {
        auto totalTimeGlobal = counters.time;
        mlir::Value totalTimeRef = rewriter.create<vir::ReferenceGlobalOp>(loc, totalTimeGlobal);

        auto countGlobal = counters.count;
        mlir::Value countRef = rewriter.create<vir::ReferenceGlobalOp>(loc, countGlobal);

        mlir::Value totalTime = rewriter.create<vir::GetElementOp>(loc, totalTimeRef);
        mlir::Value count = rewriter.create<vir::GetElementOp>(loc, countRef);

        std::string formatStr = name + "\t%ld\t%f\n";
        rewriter.create<vir::PrintFOp>(loc, formatStr, ValueRange{ count, totalTime }, /*toStderr=*/false);
    }

    rewriter.eraseOp(op);
    return success();
}

using ValueReduceSumOp = vir::ReduceSumOp;
LogicalResult ReduceSumOpLowering::matchAndRewrite(
    ValueReduceSumOp op,
    PatternRewriter& rewriter) const
{
    auto loc = op.getLoc();
    auto input = op.input();
    auto type = input.getType();
    assert(type.isa<mlir::MemRefType>() && "Input must be a memref");
    auto memRefType = type.cast<mlir::MemRefType>();
    int64_t rank = memRefType.getRank();
    if (rank != 1)
    {
        return op.emitError("Can only reduce a rank-1 memref");
    }
    mlir::Value memrefToCast = input;
    mlir::Value loadedVector = nullptr;
    if (!memRefType.getAffineMaps().empty())
    {
        auto elementType = memRefType.getElementType();
        auto vectorType = mlir::VectorType::get(memRefType.getShape(), elementType);
        auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
        loadedVector = rewriter.create<mlir::vector::TransferReadOp>(loc, vectorType, memrefToCast, mlir::ValueRange{ zero });
    }
    else
    {
        auto castMemRefVector = rewriter.create<mlir::vector::TypeCastOp>(loc, memrefToCast);
        loadedVector = rewriter.create<memref::LoadOp>(loc, castMemRefVector, llvm::None);
    }
    auto result = rewriter.create<mlir::vector::ReductionOp>(loc, op.result().getType(), rewriter.getStringAttr("add"), loadedVector, llvm::None);
    rewriter.replaceOp(op, { result });
    return success();
}

using ValuePrintFOp = vir::PrintFOp;
LogicalResult PrintOpLowering::matchAndRewrite(
    ValuePrintOp op,
    PatternRewriter& rewriter) const
{
    auto loc = op.getLoc();

    auto input = op.input();
    auto inputType = input.getType();
    auto shapedType = inputType.dyn_cast<ShapedType>();

    auto elementType = shapedType ? shapedType.getElementType() : inputType;
    auto formatStr = GetFormatStringForElementType(elementType);
    auto toStderr = op.to_stderr();

    auto printElement = [&](mlir::Value el) {
        if (elementType.isF32())
        {
            el = rewriter.create<mlir::FPExtOp>(loc, el, rewriter.getF64Type());
        }
        rewriter.create<ValuePrintFOp>(loc, formatStr, ValueRange{ el }, toStderr);
    };

    if (shapedType && shapedType.getRank() > 0)
    {
        formatStr += " ";
        auto inputShape = shapedType.getShape();

        // Create a loop for each of the dimensions within the shape.
        SmallVector<mlir::Value, 4> loopIvs;
        auto rank = shapedType.getRank();

        for (unsigned i = 0; i < rank; ++i)
        {
            auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
            auto upperBound = rewriter.create<ConstantIndexOp>(loc, inputShape[i]);
            auto step = rewriter.create<ConstantIndexOp>(loc, 1);

            auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
            loopIvs.push_back(loop.getInductionVar());
            rewriter.setInsertionPointToStart(loop.getBody());

            // Insert a newline after each of the non-innermost dimensions of the shape,
            // or generate a call to printf for the current element of the loop.
            if (i != rank - 1)
            {
                rewriter.create<ValuePrintFOp>(loc, "\n", toStderr);
            }
            else
            {
                auto elementLoad = rewriter.create<memref::LoadOp>(loc, input, loopIvs);
                printElement(elementLoad);
            }

            // Set the insertion point to the beginning of this loop so that the next loop will get added before the print
            rewriter.setInsertionPointToStart(loop.getBody());
        }
    }
    else // Special case for scalars
    {
        if (shapedType)
        {
            auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
            input = rewriter.create<memref::LoadOp>(loc, input, ValueRange{ zero.getResult() });
        }
        printElement(input);
    }

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
}

void ValueToStdLoweringPass::runOnModule()
{
    auto module = getOperation();
    auto context = module.getContext();

    OpBuilder passBuilder(module);

    if (this->enableProfiling)
    {
        InitializeProfileRegions(module, passBuilder);
    }

    for (auto vModule : make_early_inc_range(module.getOps<vir::ValueModuleOp>()))
    {
        OwningRewritePatternList vecPatterns(context);
        vtr::populateVectorizeValueOpPatterns(vecPatterns);
        (void)applyPatternsAndFoldGreedily(vModule, std::move(vecPatterns));

        OwningRewritePatternList patterns(context);
        vtr::populateValueToStandardPatterns(this->enableProfiling, patterns);
        vtr::populateValueSimplifyPatterns(patterns);
        vtr::populateValueLaunchFuncPatterns(patterns);
        utilir::FillCanonicalPatternsRecursively(vModule, patterns);
        mlir::populateExpandTanhPattern(patterns);
        applyPatternsAndFoldGreedily(vModule, std::move(patterns));
    }

    {
        OwningRewritePatternList valueModRewritePatterns(context);
        vtr::populateValueModuleRewritePatterns(valueModRewritePatterns);

        (void)applyPatternsAndFoldGreedily(module, std::move(valueModRewritePatterns));
    }

    TypeConverter typeConverter{};
    typeConverter.addConversion([](mlir::Type t) { return t; });
    typeConverter.addConversion([](MemRefType memrefTy) -> MemRefType { return MemRefType::Builder{ memrefTy }.setMemorySpace(0); });
    typeConverter.addConversion([](UnrankedMemRefType memrefTy) -> UnrankedMemRefType { return UnrankedMemRefType::get(memrefTy.getElementType(), 0); });

    ConversionTarget target(*context);
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalOp<gpu::GPUModuleOp, ModuleOp, vir::ModuleTerminatorOp>();
    target.markOpRecursivelyLegal<gpu::GPUModuleOp>();
    auto isLegalOperation = [&](Operation* op) {
        if (auto typeAttr = op->getAttrOfType<TypeAttr>(function_like_impl::getTypeAttrName()); typeAttr && !typeConverter.isLegal(typeAttr.getValue()))
        {
            return false;
        }
        return typeConverter.isLegal(op);
    };

    target.addDynamicallyLegalDialect<
        linalg::LinalgDialect,
        vir::ValueDialect,
        StandardOpsDialect,
        AffineDialect,
        math::MathDialect,
        memref::MemRefDialect,
        scf::SCFDialect,
        vector::VectorDialect,
        omp::OpenMPDialect>(
        ConversionTarget::DynamicLegalityCallbackFn(isLegalOperation));
    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(isLegalOperation);
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp fn) {
        return typeConverter.isSignatureLegal(fn.getType()) && typeConverter.isLegal(&fn.getBody());
    });

    OwningRewritePatternList genericTypeConversionPatterns(context);
    genericTypeConversionPatterns.insert<GenericOpTypeConversionPattern>(context, typeConverter);
    if (failed(applyFullConversion(module, target, std::move(genericTypeConversionPatterns))))
    {
        signalPassFailure();
    }
}

namespace accera::transforms::value
{
void populateValueModuleRewritePatterns(mlir::OwningRewritePatternList& patterns)
{
    uint16_t benefit = 1;
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<ValueModuleOpRewritePattern>(context, benefit++);
}

void populateValueLaunchFuncPatterns(mlir::OwningRewritePatternList& patterns)
{
    uint16_t benefit = 1;
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<ValueLaunchFuncOpRewritePattern>(context, benefit++);
}

void populateVectorizeValueOpPatterns(mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    accera::generated::populateWithGenerated(patterns);
    patterns.insert<
        ReduceOpVectorization,
        MapReduceOpVectorization>(context);
}

void populateValueToStandardPatterns(bool enableProfiling, mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    accera::generated::populateWithGenerated(patterns);
    patterns.insert<
        GPUTargetedFuncRewritePattern,
        GPUTargetedFuncTerminatorRewritePattern,
        AllocOpLowering,
        BinOpLowering,
        CmpOpLowering,
        GetElementOpLowering,
        GlobalOpLowering,
        LoadOpLowering,
        MapReduceOpLowering,
        MergeDimOpLowering,
        OffsetOpLowering,
        PrintOpLowering,
        ReduceOpLowering,
        ReferenceGlobalOpLowering,
        ReduceMaxOpLowering,
        ReduceSumOpLowering,
        ReorderOpLowering,
        ReshapeOpLowering,
        SliceOpLowering,
        SplitDimOpLowering,
        StoreOpLowering,
        TerminatorLowering,
        UnaryOpLowering,
        ViewOpLowering>(context);

    patterns.insert<EnterProfileRegionOpLowering,
                    PrintProfileResultsOpLowering,
                    ExitProfileRegionOpLowering>(context, enableProfiling);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToStdPass(bool enableProfiling)
{
    auto pass = std::make_unique<ValueToStdLoweringPass>(enableProfiling);
    return pass;
}
} // namespace accera::transforms::value
