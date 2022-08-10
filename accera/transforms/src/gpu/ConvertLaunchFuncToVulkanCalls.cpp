///////////////////////////////////////////////////////////////////////////////
//
//  Project:  Accera
//  File:     ConvertLaunchFuncToVulkanCalls.cpp (transforms/src/vulkan)
//  Authors:  Mason Remy
//
///////////////////////////////////////////////////////////////////////////////

// Ported from <llvm-project>/mlir/lib/Conversion/GPUToVulkan/ConvertLaunchFuncToVulkanCalls.cpp
// To add calls to set warmup run counts

//===- ConvertLaunchFuncToVulkanCalls.cpp - MLIR Vulkan conversion passes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert vulkan launch call into a sequence of
// Vulkan runtime calls. The Vulkan runtime API surface is huge so currently we
// don't expose separate external functions in IR for each of them, instead we
// expose a few external functions to wrapper libraries which manages Vulkan
// runtime.
//
//===----------------------------------------------------------------------===//

#include "AcceraPasses.h"

#include <value/include/MLIREmitterContext.h>

#include <mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FormatVariadic.h>

using namespace mlir;

static constexpr const char* kCInterfaceVulkanLaunch = "_mlir_ciface_vulkanLaunch";
static constexpr const char* kDeinitVulkan = "deinitVulkan";
static constexpr const char* kRunOnVulkan = "runOnVulkan";
static constexpr const char* kInitVulkan = "initVulkan";
static constexpr const char* kSetBinaryShader = "setBinaryShader";
static constexpr const char* kSetEntryPoint = "setEntryPoint";
static constexpr const char* kSetNumWorkGroups = "setNumWorkGroups";
static constexpr const char* kSPIRVBinary = "SPIRV_BIN";
static constexpr const char* kSPIRVBlobAttrName = "spirv_blob";
static constexpr const char* kSPIRVEntryPointAttrName = "spirv_entry_point";
static constexpr const char* kVulkanLaunch = "vulkanLaunch";

// Custom to acc-vulkan-runtime-wrappers
static constexpr const char* kSetRepeatedRunCharacteristics = "setRepeatedRunCharacteristics";
static constexpr const char* kVulkanRuntimeInstanceHandle = "VulkanRuntime_Instance_Handle";
static constexpr const char* kInitVulkanEmittedFunc = "initVulkanUtilities";
static constexpr const char* kDestroyVulkanEmittedFunc = "destroyVulkanUtilities";
static constexpr const char* kInitializeFuncAttrName = "rc_gpu_init";
static constexpr const char* kDeInitializeFuncAttrName = "rc_gpu_deinit";
static constexpr const char* kVulkanRuntimeHandleAccessor = "getVulkanRuntimeInstance";

namespace
{

/// A pass to convert vulkan launch call op into a sequence of Vulkan
/// runtime calls in the following order:
///
/// * initVulkan                        -- initializes vulkan runtime
/// * bindMemRef                        -- binds memref
/// * setBinaryShader                   -- sets the binary shader data
/// * setEntryPoint                     -- sets the entry point name
/// * setNumWorkGroups                  -- sets the number of a local workgroups
/// * setRepeatedRunCharacteristics     -- sets the warmup and run counts for timing runs
/// * runOnVulkan                       -- runs vulkan runtime
/// * deinitVulkan                      -- deinitializes vulkan runtime
///
class VulkanLaunchFuncToVulkanCallsWithTimingPass
    : public accera::transforms::ConvertVulkanLaunchFuncToVulkanCallsWithTimingBase<VulkanLaunchFuncToVulkanCallsWithTimingPass>
{
private:
    void initializeCachedTypes()
    {
        llvmFloatType = Float32Type::get(&getContext());
        llvmVoidType = LLVM::LLVMVoidType::get(&getContext());
        llvmPointerType =
            LLVM::LLVMPointerType::get(IntegerType::get(&getContext(), 8));
        llvmInt32Type = IntegerType::get(&getContext(), 32);
        llvmInt64Type = IntegerType::get(&getContext(), 64);
    }

    Type getMemRefType(uint32_t rank, Type elemenType)
    {
        // According to the MLIR doc memref argument is converted into a
        // pointer-to-struct argument of type:
        // template <typename Elem, size_t Rank>
        // struct {
        //   Elem *allocated;
        //   Elem *aligned;
        //   int64_t offset;
        //   int64_t sizes[Rank]; // omitted when rank == 0
        //   int64_t strides[Rank]; // omitted when rank == 0
        // };
        auto llvmPtrToElementType = LLVM::LLVMPointerType::get(elemenType);
        auto llvmArrayRankElementSizeType =
            LLVM::LLVMArrayType::get(getInt64Type(), rank);

        // Create a type
        // `!llvm<"{ `element-type`*, `element-type`*, i64,
        // [`rank` x i64], [`rank` x i64]}">`.
        return LLVM::LLVMStructType::getLiteral(
            &getContext(),
            { llvmPtrToElementType, llvmPtrToElementType, getInt64Type(), llvmArrayRankElementSizeType, llvmArrayRankElementSizeType });
    }

    Type getVoidType() { return llvmVoidType; }
    Type getPointerType() { return llvmPointerType; }
    Type getInt32Type() { return llvmInt32Type; }
    Type getInt64Type() { return llvmInt64Type; }

    /// Creates an LLVM global for the given `name`.
    Value createEntryPointNameConstant(StringRef name, Location loc, OpBuilder& builder);

    /// Declares all needed runtime functions.
    void declareVulkanFunctions(Location loc);

    /// Checks whether the given LLVM::CallOp is a vulkan launch call op.
    bool isVulkanLaunchCallOp(LLVM::CallOp callOp)
    {
        return (callOp.getCallee() && callOp.getCallee().getValue() == kVulkanLaunch &&
                callOp.getNumOperands() >= kVulkanLaunchNumConfigOperands);
    }

    /// Checks whether the given LLVM::CallOp is a "ci_face" vulkan launch call
    /// op.
    bool isCInterfaceVulkanLaunchCallOp(LLVM::CallOp callOp)
    {
        return (callOp.getCallee() &&
                callOp.getCallee().getValue() == kCInterfaceVulkanLaunch &&
                callOp.getNumOperands() >= kVulkanLaunchNumConfigOperands);
    }

    /// Checks whether the given LLVM::LLVMFuncOp is marked as an Accera GPU
    /// initialization func
    bool isRCGPUInitializeFunc(LLVM::LLVMFuncOp op)
    {
        auto funcTags = op->getAttrOfType<DictionaryAttr>(accera::ir::FunctionTagsAttrName);
        if (funcTags == nullptr)
        {
            return false;
        }
        return funcTags.get(kInitializeFuncAttrName) != nullptr;
    }

    /// Checks whether the given LLVM::LLVMFuncOp is marked as an Accera GPU
    /// de-initialization func
    bool isRCGPUDeInitializeFunc(LLVM::LLVMFuncOp op)
    {
        auto funcTags = op->getAttrOfType<DictionaryAttr>(accera::ir::FunctionTagsAttrName);
        if (funcTags == nullptr)
        {
            return false;
        }
        return funcTags.get(kDeInitializeFuncAttrName) != nullptr;
    }

    /// Translates the given `vulkanLaunchCallOp` to the sequence of Vulkan
    /// runtime calls.
    void translateVulkanLaunchCall(LLVM::CallOp vulkanLaunchCallOp);

    /// Creates call to `bindMemRef` for each memref operand.
    void createBindMemRefCalls(LLVM::CallOp vulkanLaunchCallOp,
                               Value vulkanRuntime);

    /// Collects SPIRV attributes from the given `vulkanLaunchCallOp`.
    void collectSPIRVAttributes(LLVM::CallOp vulkanLaunchCallOp);

    /// Deduces a rank and element type from the given 'ptrToMemRefDescriptor`.
    LogicalResult deduceMemRefRankAndType(Value ptrToMemRefDescriptor,
                                          uint32_t& rank,
                                          Type& type);
    /// Returns a string representation from the given `type`.
    StringRef stringifyType(Type type)
    {
        if (type.isa<Float32Type>())
            return "Float";
        if (type.isa<Float16Type>())
            return "Half";
        if (auto intType = type.dyn_cast<IntegerType>())
        {
            if (intType.getWidth() == 32)
                return "Int32";
            if (intType.getWidth() == 16)
                return "Int16";
            if (intType.getWidth() == 8)
                return "Int8";
        }

        llvm_unreachable("unsupported type");
    }

    /// Get the pointer to the module-global vulkan runtime handle
    /// Only callable from the GPU utilities module
    mlir::Value getVulkanRuntimeGlobalPtr(mlir::Location loc, mlir::OpBuilder& builder);

    /// Get the value of the module-global vulkan runtime handle
    /// If called from a module other than the GPU utilities module it will ensure
    /// that an extern function decl exists to invoke
    mlir::Value getVulkanRuntimeHandle(mlir::Location loc, mlir::OpBuilder& builder);

    /// Emit the Vulkan Runtime one-time initialization function
    void completeVulkanInitializeFunction(LLVM::LLVMFuncOp op);

    /// Emit the Vulkan Runtime one-time de-initialization function
    void completeVulkanDeInitializeFunction(LLVM::LLVMFuncOp op);

    /// Emit the Vulkan Runtime handle getter so multiple modules can use the same handle
    void emitVulkanRuntimeHandleAccessor();

    /// Finds the vulkanLaunch func symbol and mangles the name, prefixing
    /// it with the module name, and updates uses of the func op to reference
    /// the new mangled name
    void markVulkanLaunchInternal();

public:
    VulkanLaunchFuncToVulkanCallsWithTimingPass(bool printTimings, uint32_t warmupCount, uint32_t runCount)
    {
        this->printTimings = printTimings;
        this->warmupCount = warmupCount;
        this->runCount = runCount;
    }

    void runOnModule() override;

private:
    Type llvmFloatType;
    Type llvmVoidType;
    Type llvmPointerType;
    Type llvmInt32Type;
    Type llvmInt64Type;

    // TODO: Use an associative array to support multiple vulkan launch calls.
    std::pair<StringAttr, StringAttr> spirvAttributes;
    /// The number of vulkan launch configuration operands, placed at the leading
    /// positions of the operand list.
    static constexpr unsigned kVulkanLaunchNumConfigOperands = 3;
};

// matches mlir\lib\Dialect\LLVMIR\IR\LLVMDialect.cpp
static mlir::StringRef getLinkageAttrName()
{
    return "linkage";
}

} // namespace

void VulkanLaunchFuncToVulkanCallsWithTimingPass::runOnModule()
{
    initializeCachedTypes();

    // Collect SPIR-V attributes such as `spirv_blob` and
    // `spirv_entry_point_name`.
    getOperation().walk([this](LLVM::CallOp op) {
        if (isVulkanLaunchCallOp(op))
            collectSPIRVAttributes(op);
    });

    // Convert vulkan launch call op into a sequence of Vulkan runtime calls.
    getOperation().walk([this](LLVM::CallOp op) {
        if (isCInterfaceVulkanLaunchCallOp(op))
        {
            translateVulkanLaunchCall(op);
        }
    });

    // TODO : should this be in a separate pass?
    // Fill out any initialize / de-initialize functions in the module
    // Also create vulkan runtime handle accessor functions for other modules to use
    getOperation().walk([this](LLVM::LLVMFuncOp op) {
        if (isRCGPUInitializeFunc(op))
        {
            completeVulkanInitializeFunction(op);
            emitVulkanRuntimeHandleAccessor();
        }
        else if (isRCGPUDeInitializeFunc(op))
        {
            completeVulkanDeInitializeFunction(op);
        }
    });

    // Mark the  vulkanLaunch function as internal linkage so that multiple modules
    // can each have a unique vulkanLaunch function within a single library
    markVulkanLaunchInternal();

    // It's illegal for the SPIRV attributes to remain behind, according to mlir-translate
    (void)getOperation()->removeAttr(spirv::getTargetEnvAttrName());
}

void VulkanLaunchFuncToVulkanCallsWithTimingPass::collectSPIRVAttributes(
    LLVM::CallOp vulkanLaunchCallOp)
{
    // Check that `kSPIRVBinary` and `kSPIRVEntryPoint` are present in attributes
    // for the given vulkan launch call.
    auto spirvBlobAttr =
        vulkanLaunchCallOp->getAttrOfType<StringAttr>(kSPIRVBlobAttrName);
    if (!spirvBlobAttr)
    {
        vulkanLaunchCallOp.emitError()
            << "missing " << kSPIRVBlobAttrName << " attribute";
        return signalPassFailure();
    }

    auto spirvEntryPointNameAttr =
        vulkanLaunchCallOp->getAttrOfType<StringAttr>(kSPIRVEntryPointAttrName);
    if (!spirvEntryPointNameAttr)
    {
        vulkanLaunchCallOp.emitError()
            << "missing " << kSPIRVEntryPointAttrName << " attribute";
        return signalPassFailure();
    }

    spirvAttributes = std::make_pair(spirvBlobAttr, spirvEntryPointNameAttr);
}

void VulkanLaunchFuncToVulkanCallsWithTimingPass::createBindMemRefCalls(
    LLVM::CallOp cInterfaceVulkanLaunchCallOp,
    Value vulkanRuntime)
{
    if (cInterfaceVulkanLaunchCallOp.getNumOperands() ==
        kVulkanLaunchNumConfigOperands)
        return;
    OpBuilder builder(cInterfaceVulkanLaunchCallOp);
    Location loc = cInterfaceVulkanLaunchCallOp.getLoc();

    // Create LLVM constant for the descriptor set index.
    // Bind all memrefs to the `0` descriptor set, the same way as `GPUToSPIRV`
    // pass does.
    Value descriptorSet = builder.create<LLVM::ConstantOp>(
        loc, getInt32Type(), builder.getI32IntegerAttr(0));

    for (auto en :
         llvm::enumerate(cInterfaceVulkanLaunchCallOp.getOperands().drop_front(
             kVulkanLaunchNumConfigOperands)))
    {
        // Create LLVM constant for the descriptor binding index.
        Value descriptorBinding = builder.create<LLVM::ConstantOp>(
            loc, getInt32Type(), builder.getI32IntegerAttr(en.index()));

        auto ptrToMemRefDescriptor = en.value();
        uint32_t rank = 0;
        Type type;
        if (failed(deduceMemRefRankAndType(ptrToMemRefDescriptor, rank, type)))
        {
            cInterfaceVulkanLaunchCallOp.emitError()
                << "invalid memref descriptor " << ptrToMemRefDescriptor.getType();
            return signalPassFailure();
        }

        auto symbolName =
            llvm::formatv("bindMemRef{0}D{1}", rank, stringifyType(type)).str();
        // Special case for fp16 type. Since it is not a supported type in C we use
        // int16_t and bitcast the descriptor.
        if (type.isa<Float16Type>())
        {
            auto memRefTy = getMemRefType(rank, IntegerType::get(&getContext(), 16));
            ptrToMemRefDescriptor = builder.create<LLVM::BitcastOp>(
                loc, LLVM::LLVMPointerType::get(memRefTy), ptrToMemRefDescriptor);
        }
        // Create call to `bindMemRef`.
        builder.create<LLVM::CallOp>(
            loc, ArrayRef<Type>{}, SymbolRefAttr::get(builder.getContext(), StringRef(symbolName.data(), symbolName.size())), ArrayRef<Value>{ vulkanRuntime, descriptorSet, descriptorBinding, ptrToMemRefDescriptor });
    }
}

LogicalResult VulkanLaunchFuncToVulkanCallsWithTimingPass::deduceMemRefRankAndType(
    Value ptrToMemRefDescriptor,
    uint32_t& rank,
    Type& type)
{
    auto llvmPtrDescriptorTy =
        ptrToMemRefDescriptor.getType().dyn_cast<LLVM::LLVMPointerType>();
    if (!llvmPtrDescriptorTy)
        return failure();

    auto llvmDescriptorTy =
        llvmPtrDescriptorTy.getElementType().dyn_cast<LLVM::LLVMStructType>();
    // template <typename Elem, size_t Rank>
    // struct {
    //   Elem *allocated;
    //   Elem *aligned;
    //   int64_t offset;
    //   int64_t sizes[Rank]; // omitted when rank == 0
    //   int64_t strides[Rank]; // omitted when rank == 0
    // };
    if (!llvmDescriptorTy)
        return failure();

    type = llvmDescriptorTy.getBody()[0]
               .cast<LLVM::LLVMPointerType>()
               .getElementType();
    if (llvmDescriptorTy.getBody().size() == 3)
    {
        rank = 0;
        return success();
    }
    rank = llvmDescriptorTy.getBody()[3]
               .cast<LLVM::LLVMArrayType>()
               .getNumElements();
    return success();
}

void VulkanLaunchFuncToVulkanCallsWithTimingPass::declareVulkanFunctions(Location loc)
{
    ModuleOp module = getOperation();
    auto builder = OpBuilder::atBlockEnd(module.getBody());

    if (!module.lookupSymbol(kSetEntryPoint))
    {
        builder.create<LLVM::LLVMFuncOp>(
            loc, kSetEntryPoint, LLVM::LLVMFunctionType::get(getVoidType(), { getPointerType(), getPointerType() },
                                                             /*isVarArg=*/false));
    }

    if (!module.lookupSymbol(kSetNumWorkGroups))
    {
        builder.create<LLVM::LLVMFuncOp>(
            loc, kSetNumWorkGroups, LLVM::LLVMFunctionType::get(getVoidType(), { getPointerType(), getInt64Type(), getInt64Type(), getInt64Type() },
                                                                /*isVarArg=*/false));
    }

    if (!module.lookupSymbol(kSetBinaryShader))
    {
        builder.create<LLVM::LLVMFuncOp>(
            loc, kSetBinaryShader, LLVM::LLVMFunctionType::get(getVoidType(), { getPointerType(), getPointerType(), getInt32Type() },
                                                               /*isVarArg=*/false));
    }

    if (!module.lookupSymbol(kRunOnVulkan))
    {
        builder.create<LLVM::LLVMFuncOp>(
            loc, kRunOnVulkan, LLVM::LLVMFunctionType::get(getVoidType(), { getPointerType() },
                                                           /*isVarArg=*/false));
    }

    if (!module.lookupSymbol(kSetRepeatedRunCharacteristics))
    {
        builder.create<LLVM::LLVMFuncOp>(
            loc, kSetRepeatedRunCharacteristics, LLVM::LLVMFunctionType::get(getVoidType(), { getPointerType(), getInt32Type(), getInt32Type(), getInt32Type() },
                                                                             /*isVarArg=*/false));
    }

    for (unsigned i = 1; i <= 3; i++)
    {

        SmallVector<Type, 5> types{
            Float32Type::get(&getContext()), IntegerType::get(&getContext(), 32), IntegerType::get(&getContext(), 16), IntegerType::get(&getContext(), 8), Float16Type::get(&getContext())
        };
        for (auto type : types)
        {
            std::string fnName = "bindMemRef" + std::to_string(i) + "D" +
                                 std::string(stringifyType(type));
            if (type.isa<Float16Type>())
                type = IntegerType::get(&getContext(), 16);
            if (!module.lookupSymbol(fnName))
            {
                auto fnType = LLVM::LLVMFunctionType::get(
                    getVoidType(),
                    { getPointerType(), getInt32Type(), getInt32Type(), LLVM::LLVMPointerType::get(getMemRefType(i, type)) },
                    /*isVarArg=*/false);
                builder.create<LLVM::LLVMFuncOp>(loc, fnName, fnType);
            }
        }
    }

    if (!module.lookupSymbol(kInitVulkan))
    {
        builder.create<LLVM::LLVMFuncOp>(
            loc, kInitVulkan, LLVM::LLVMFunctionType::get(getPointerType(), {},
                                                          /*isVarArg=*/false));
    }

    if (!module.lookupSymbol(kDeinitVulkan))
    {
        builder.create<LLVM::LLVMFuncOp>(
            loc, kDeinitVulkan, LLVM::LLVMFunctionType::get(getVoidType(), { getPointerType() },
                                                            /*isVarArg=*/false));
    }
}

Value VulkanLaunchFuncToVulkanCallsWithTimingPass::createEntryPointNameConstant(
    StringRef name,
    Location loc,
    OpBuilder& builder)
{
    SmallString<16> shaderName(name.begin(), name.end());
    // Append `\0` to follow C style string given that LLVM::createGlobalString()
    // won't handle this directly for us.
    shaderName.push_back('\0');

    std::string entryPointGlobalName = (name + "_spv_entry_point_name").str();
    return LLVM::createGlobalString(loc, builder, entryPointGlobalName, shaderName, LLVM::Linkage::Internal);
}

void VulkanLaunchFuncToVulkanCallsWithTimingPass::translateVulkanLaunchCall(
    LLVM::CallOp cInterfaceVulkanLaunchCallOp)
{
    [[maybe_unused]] ModuleOp module = getOperation();
    OpBuilder builder(cInterfaceVulkanLaunchCallOp);
    Location loc = cInterfaceVulkanLaunchCallOp.getLoc();

    // Declare runtime functions.
    declareVulkanFunctions(loc);

    auto vulkanRuntime = getVulkanRuntimeHandle(loc, builder);

    // Create call to `bindMemRef` for each memref operand.
    createBindMemRefCalls(cInterfaceVulkanLaunchCallOp, vulkanRuntime);

    // Create LLVM global with SPIR-V binary data, so we can pass a pointer with
    // that data to runtime call.
    Value ptrToSPIRVBinary = LLVM::createGlobalString(
        loc, builder, kSPIRVBinary, spirvAttributes.first.getValue(), LLVM::Linkage::Internal);

    // Create LLVM constant for the size of SPIR-V binary shader.
    Value binarySize = builder.create<LLVM::ConstantOp>(
        loc, getInt32Type(), builder.getI32IntegerAttr(spirvAttributes.first.getValue().size()));

    // Create call to `setBinaryShader` runtime function with the given pointer to
    // SPIR-V binary and binary size.
    builder.create<LLVM::CallOp>(
        loc, ArrayRef<Type>{}, kSetBinaryShader, ArrayRef<Value>{ vulkanRuntime, ptrToSPIRVBinary, binarySize });

    // Create LLVM global with entry point name.
    Value entryPointName = createEntryPointNameConstant(
        spirvAttributes.second.getValue(), loc, builder);

    // Create call to `setEntryPoint` runtime function with the given pointer to
    // entry point name.
    builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{}, kSetEntryPoint, ArrayRef<Value>{ vulkanRuntime, entryPointName });

    // Create call to setRepeatedRunCharacteristics to set warmup run count and timing run count
    Value printTimingsConstant = builder.create<LLVM::ConstantOp>(loc, getInt32Type(), builder.getI32IntegerAttr(printTimings ? 1 : 0));
    Value warmupCountConstant = builder.create<LLVM::ConstantOp>(loc, getInt32Type(), builder.getI32IntegerAttr(warmupCount));
    Value runCountConstant = builder.create<LLVM::ConstantOp>(loc, getInt32Type(), builder.getI32IntegerAttr(runCount));
    builder.create<LLVM::CallOp>(
        loc,
        ArrayRef<Type>{},
        kSetRepeatedRunCharacteristics,
        ArrayRef<Value>{ vulkanRuntime,
                         printTimingsConstant,
                         warmupCountConstant,
                         runCountConstant });

    // This SetNumWorkGroups call needs to be in whatever function is called with the emit-time-constant dimensions
    // TODO : move the hard-coded workgroup dimensions out of the caller of vulkanLaunch
    builder.create<LLVM::CallOp>(
        loc, ArrayRef<Type>{}, kSetNumWorkGroups, ArrayRef<Value>{ vulkanRuntime, cInterfaceVulkanLaunchCallOp.getOperand(0), cInterfaceVulkanLaunchCallOp.getOperand(1), cInterfaceVulkanLaunchCallOp.getOperand(2) });

    // Create call to `runOnVulkan` runtime function.
    // TODO : surface APIs for moving memory onto and off of the device
    builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{}, kRunOnVulkan, ArrayRef<Value>{ vulkanRuntime });

    cInterfaceVulkanLaunchCallOp.erase();
}

mlir::Value VulkanLaunchFuncToVulkanCallsWithTimingPass::getVulkanRuntimeGlobalPtr(Location loc, mlir::OpBuilder& builder)
{
    ModuleOp module = getOperation();
    MLIRContext* context = builder.getContext();

    // Set up a module global handle to hold the initialized runtime void* pointer
    Operation* globalOp = module.lookupSymbol(kVulkanRuntimeInstanceHandle);
    if (!globalOp)
    {
        // Create the global op if it doesn't already exist
        auto moduleBuilder = OpBuilder::atBlockEnd(module.getBody());
        auto type = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
        globalOp = moduleBuilder.create<mlir::LLVM::GlobalOp>(
            loc, type, /*isConstant=*/false, mlir::LLVM::Linkage::Internal, kVulkanRuntimeInstanceHandle, mlir::Attribute());
    }
    // Get the pointer to the first character in the global string.
    auto llvmGlobalOp = dyn_cast<mlir::LLVM::GlobalOp>(globalOp);
    return builder.create<mlir::LLVM::AddressOfOp>(loc, llvmGlobalOp);
}

void VulkanLaunchFuncToVulkanCallsWithTimingPass::emitVulkanRuntimeHandleAccessor()
{
    ModuleOp module = getOperation();
    Location loc = module.getLoc();
    auto builder = OpBuilder::atBlockEnd(module.getBody());
    [[maybe_unused]] MLIRContext* context = builder.getContext();

    if (!module.lookupSymbol(kVulkanRuntimeHandleAccessor))
    {
        auto vulkanRuntimeHandleAccessor = builder.create<LLVM::LLVMFuncOp>(
            loc, kVulkanRuntimeHandleAccessor, LLVM::LLVMFunctionType::get(getPointerType(), {}, /*isVarArg=*/false));

        OpBuilder::InsertionGuard guard(builder);

        builder.setInsertionPointToStart(vulkanRuntimeHandleAccessor.addEntryBlock());

        auto globalPtr = getVulkanRuntimeGlobalPtr(loc, builder);
        mlir::Value vulkanRuntimeHandle = builder.create<LLVM::LoadOp>(loc, globalPtr);

        builder.create<LLVM::ReturnOp>(loc, vulkanRuntimeHandle);
    }
}

mlir::Value VulkanLaunchFuncToVulkanCallsWithTimingPass::getVulkanRuntimeHandle(Location loc, mlir::OpBuilder& builder)
{
    ModuleOp module = getOperation();
    if (!module.lookupSymbol(kVulkanRuntimeHandleAccessor))
    {
        OpBuilder::InsertionGuard guard(builder);
        auto builder = OpBuilder::atBlockEnd(module.getBody());
        builder.create<LLVM::LLVMFuncOp>(
            loc, kVulkanRuntimeHandleAccessor, LLVM::LLVMFunctionType::get(getPointerType(), {},
                                                                           /*isVarArg=*/false));
    }

    auto getVulkanRuntimeHandleCall = builder.create<LLVM::CallOp>(
        loc, ArrayRef<Type>{ getPointerType() }, kVulkanRuntimeHandleAccessor, ArrayRef<Value>{});

    auto vulkanRuntime = getVulkanRuntimeHandleCall.getResult(0);

    return vulkanRuntime;
}

void VulkanLaunchFuncToVulkanCallsWithTimingPass::completeVulkanInitializeFunction(LLVM::LLVMFuncOp op)
{
    ModuleOp module = getOperation();
    Location loc = module.getLoc();
    OpBuilder builder(op.getBody().back().getTerminator());
    [[maybe_unused]] MLIRContext* context = builder.getContext();

    declareVulkanFunctions(loc);

    // Create call to `initVulkan`.
    auto initVulkanCall = builder.create<LLVM::CallOp>(
        loc, ArrayRef<Type>{ getPointerType() }, kInitVulkan, ArrayRef<Value>{});

    // The result of `initVulkan` function is a pointer to Vulkan runtime, we
    // need to pass that pointer to each Vulkan runtime call.
    auto vulkanRuntime = initVulkanCall.getResult(0);

    // hold onto the vulkanRuntime pointer in the module global instance handle
    auto globalPtr = getVulkanRuntimeGlobalPtr(loc, builder);
    builder.create<LLVM::StoreOp>(loc, vulkanRuntime, globalPtr);
}

void VulkanLaunchFuncToVulkanCallsWithTimingPass::completeVulkanDeInitializeFunction(LLVM::LLVMFuncOp op)
{
    ModuleOp module = getOperation();
    Location loc = module.getLoc();
    OpBuilder builder(op.getBody().back().getTerminator());
    [[maybe_unused]] MLIRContext* context = builder.getContext();

    declareVulkanFunctions(loc);

    auto vulkanRuntime = getVulkanRuntimeHandle(loc, builder);

    // Create call to 'deinitVulkan' runtime function.
    builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{}, kDeinitVulkan, ArrayRef<Value>{ vulkanRuntime });
}

void VulkanLaunchFuncToVulkanCallsWithTimingPass::markVulkanLaunchInternal()
{
    // Mark the vulkanLaunch func op as internal linkage
    // Currently the standard FuncOp does not support linkage so we make this change
    // on the LLVMFuncOp after it has been lowered to override the default External

    auto moduleOp = getOperation();
    SymbolTable symbolTable = SymbolTable::getNearestSymbolTable(moduleOp);
    auto vulkanLaunchFuncOp = dyn_cast_or_null<LLVM::LLVMFuncOp>(symbolTable.lookup(kVulkanLaunch));
    if (vulkanLaunchFuncOp)
    {
        OpBuilder builder(vulkanLaunchFuncOp);
        vulkanLaunchFuncOp->setAttr(getLinkageAttrName(),
                                    builder.getI64IntegerAttr(static_cast<int64_t>(LLVM::Linkage::Internal)));
    }
}

namespace accera::transforms::vulkan
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertVulkanLaunchFuncToVulkanCallsWithTimingPass(const VulkanTimingOptions& options)
{
    return std::make_unique<VulkanLaunchFuncToVulkanCallsWithTimingPass>(options.printTimings, static_cast<uint32_t>(options.warmupCount), static_cast<uint32_t>(options.runCount));
}
} // namespace accera::transforms::vulkan
