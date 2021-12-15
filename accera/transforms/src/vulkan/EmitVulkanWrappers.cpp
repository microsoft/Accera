////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <value/include/MLIREmitterContext.h>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/STLExtras.h>

using namespace mlir;

namespace
{

static constexpr const char* kVulkanLaunch = "vulkanLaunch";

/// A pass to mark the vulkanLaunch function to emit a C wrapper
class EmitVulkanWrapperPass : public accera::transforms::EmitVulkanWrapperBase<EmitVulkanWrapperPass>
{
public:
    void runOnModule() override
    {
        auto moduleOp = getOperation();
        SymbolTable symbolTable = SymbolTable::getNearestSymbolTable(moduleOp);
        auto vulkanLaunchFuncOp = dyn_cast_or_null<mlir::FuncOp>(symbolTable.lookup(kVulkanLaunch));
        if (vulkanLaunchFuncOp)
        {
            OpBuilder builder(vulkanLaunchFuncOp);
            vulkanLaunchFuncOp->setAttr(accera::ir::CInterfaceAttrName, builder.getUnitAttr());
        }
    }
};

} // namespace

namespace accera::transforms::vulkan
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createEmitVulkanWrapperPass()
{
    return std::make_unique<EmitVulkanWrapperPass>();
}
} // namespace accera::transforms::vulkan
