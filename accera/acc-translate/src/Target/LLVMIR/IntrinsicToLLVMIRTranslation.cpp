////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////


#include "IntrinsicToLLVMIRTranslation.h"

#include <ir/include/intrinsics/AcceraIntrinsicsDialect.h>

#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsX86.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace accera::transforms::intrinsics;

namespace {
class IntrinsicsDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "intrinsics/AcceraIntrinsicsConversions.inc"

    return failure();
  }
};
} // namespace

void accera::transforms::intrinsics::registerIntrinsicsDialectTranslation(DialectRegistry &registry) {
  registry.insert<accera::ir::intrinsics::AcceraIntrinsicsDialect>();
  registry.addDialectInterface<accera::ir::intrinsics::AcceraIntrinsicsDialect,
                               IntrinsicsDialectLLVMIRTranslationInterface>();
}

void accera::transforms::intrinsics::registerIntrinsicsDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerIntrinsicsDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
