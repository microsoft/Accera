////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "InitializeAccera.h"
#include "build/LLVMEmitterTargets.h"
#include "exec/ExecutionPlanOps.h"
#include "nest/LoopNestOps.h"
#include "accera/AcceraOps.h"
#include "value/ValueDialect.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
//#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Dialect.h>

#include <llvm/InitializePasses.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

using namespace mlir;

namespace accera::ir
{

namespace
{
    void InitializeLLVMTargets()
    {
        // This block is part of a X-Macro. LLVM_EMITTER_TARGETS below is
        // defined in build/LLVMEmitterTargets.h at CMake configure time.
        // It is dependent on the value of the CMake variable LLVM_EMITTER_TARGETS.
        // For each LLVM target specified in that variable, EMITTER_TARGET_ACTION
        // below gets called
#define EMITTER_TARGET_ACTION(TargetName)     \
    LLVMInitialize##TargetName##TargetInfo(); \
    LLVMInitialize##TargetName##Target();     \
    LLVMInitialize##TargetName##TargetMC();   \
    LLVMInitialize##TargetName##AsmPrinter(); \
    LLVMInitialize##TargetName##AsmParser();  \
    LLVMInitialize##TargetName##Disassembler();
        LLVM_EMITTER_TARGETS
#undef EMITTER_TARGET_ACTION

        llvm::InitializeNativeTarget();
    }

    [[maybe_unused]] void InitializeGlobalPassRegistry()
    {
        // Get the global pass registry
        llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();

        // Initialize all of the optimization passes (probably unnecessary)
        llvm::initializeCore(*registry);
        llvm::initializeScalarOpts(*registry);
        llvm::initializeVectorization(*registry);
        llvm::initializeIPO(*registry);
        llvm::initializeAnalysis(*registry);
        llvm::initializeTransformUtils(*registry);
        llvm::initializeInstCombine(*registry);
        llvm::initializeAggressiveInstCombine(*registry);
        llvm::initializeInstrumentation(*registry);
        llvm::initializeTarget(*registry);
        llvm::initializeGlobalISel(*registry);

        // For codegen passes, only passes that do IR to IR transformation are
        // supported.
        llvm::initializeExpandMemCmpPassPass(*registry);
        llvm::initializeScalarizeMaskedMemIntrinLegacyPassPass(*registry);
        llvm::initializeCodeGenPreparePass(*registry);
        llvm::initializeAtomicExpandPass(*registry);
        llvm::initializeRewriteSymbolsLegacyPassPass(*registry);
        llvm::initializeWinEHPreparePass(*registry);
        llvm::initializeDwarfEHPrepareLegacyPassPass(*registry);
        llvm::initializeSafeStackLegacyPassPass(*registry);
        llvm::initializeSjLjEHPreparePass(*registry);
        llvm::initializePreISelIntrinsicLoweringLegacyPassPass(*registry);
        llvm::initializeGlobalMergePass(*registry);
        llvm::initializeIndirectBrExpandPassPass(*registry);
        llvm::initializeInterleavedLoadCombinePass(*registry);
        llvm::initializeInterleavedAccessPass(*registry);
        llvm::initializeEntryExitInstrumenterPass(*registry);
        llvm::initializePostInlineEntryExitInstrumenterPass(*registry);
        llvm::initializeUnreachableBlockElimLegacyPassPass(*registry);
        llvm::initializeExpandReductionsPass(*registry);
        llvm::initializeWriteBitcodePassPass(*registry);
    }

    void InitializeLLVM()
    {
        InitializeLLVMTargets();
        InitializeGlobalPassRegistry();
    }

} // namespace

void InitializeAccera()
{
    InitializeLLVM();
}

} // namespace accera::ir
