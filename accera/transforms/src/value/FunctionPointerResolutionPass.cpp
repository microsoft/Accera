////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <value/include/FunctionDeclaration.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FormatVariadic.h>

#include <string>

using namespace mlir;

namespace
{

class FunctionPointerResolutionPass
    : public accera::transforms::FunctionPointerResolutionBase<FunctionPointerResolutionPass>
{
public:
    FunctionPointerResolutionPass() {}

    void runOnModule() override;

private:
    StringRef GetFuncSymbolName(LLVM::LLVMFuncOp& op)
    {
        auto symbolNameAttr = op->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
        return symbolNameAttr.getValue();
    }

    bool HasAcceraTemporaryPrefix(LLVM::LLVMFuncOp& op)
    {
        StringRef funcSymbolName = GetFuncSymbolName(op);
        std::string strSymbolName = funcSymbolName.str();
        return strSymbolName.find(accera::value::FunctionDeclaration::GetTemporaryFunctionPointerPrefix(), 0) == 0;
    }

    std::string GetSymbolNameWithoutAcceraTemporaryPrefix(LLVM::LLVMFuncOp& op)
    {
        std::string strSymbolName = GetFuncSymbolName(op).str();
        assert(strSymbolName.find(accera::value::FunctionDeclaration::GetTemporaryFunctionPointerPrefix(), 0) == 0);
        return strSymbolName.substr(accera::value::FunctionDeclaration::GetTemporaryFunctionPointerPrefix().length());
    }
};

} // namespace

void FunctionPointerResolutionPass::runOnModule()
{
    // Find and replace usages of LLVM::LLVMFuncOp's prefixed with the temporary function pointer prefix with their non-prefixed counterparts
    SymbolTable symbolTable = SymbolTable::getNearestSymbolTable(getOperation());
    getOperation().walk([&](LLVM::LLVMFuncOp op) {
        if (HasAcceraTemporaryPrefix(op))
        {
            std::string replacementFuncName = GetSymbolNameWithoutAcceraTemporaryPrefix(op);
            auto replacementFunc = symbolTable.lookup(replacementFuncName);
            auto replacementLLVMFunc = dyn_cast<LLVM::LLVMFuncOp>(replacementFunc);

            [[maybe_unused]] auto ignored = symbolTable.replaceAllSymbolUses(op, GetFuncSymbolName(replacementLLVMFunc), getOperation());
        }
    });
}

namespace accera::transforms::value
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createFunctionPointerResolutionPass()
{
    return std::make_unique<FunctionPointerResolutionPass>();
}
} // namespace accera::transforms::value
