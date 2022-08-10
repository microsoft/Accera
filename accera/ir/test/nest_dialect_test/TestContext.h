////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "IRTestVerification.h"

#include <ir/include/DialectRegistry.h>
#include <ir/include/InitializeAccera.h>

#include <utilities/include/Logger.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/SourceMgr.h>

#include <functional>
#include <string>
#include <vector>

class TestContext
{
public:
    TestContext(std::function<void()> setup) :
        TestContext("test", setup) {}

    TestContext(std::string name, std::function<void()> setup) :
        _context(InitContext()),
        _ownedModule(mlir::ModuleOp::create(
            mlir::UnknownLoc::get(_context),
            llvm::StringRef(name + "Module"))),
        _module(*_ownedModule),
        _sourceMgrHandler(_sourceMgr, _context),
        _builder(_context)
    {
        _fnOp = CreateFunction(_builder, name, {}, {});
        auto entryBlock = _fnOp.addEntryBlock();

        mlir::OpBuilder bodyBuilder(entryBlock->getParent());
        mlir::OpBuilder::InsertionGuard guard(bodyBuilder);
        bodyBuilder.setInsertionPoint(entryBlock, std::prev(entryBlock->end()));
        setup();
        (void)bodyBuilder.create<mlir::ReturnOp>(bodyBuilder.getUnknownLoc());
    }

    TestContext(std::function<std::vector<mlir::Type>()> getArgTypes, std::function<void(std::vector<mlir::Value>)> body) :
        TestContext("test", getArgTypes, body) {}

    TestContext(std::string name, std::function<std::vector<mlir::Type>()> getArgTypes, std::function<void(std::vector<mlir::Value>)> body) :
        _context(InitContext()),
        _ownedModule(mlir::ModuleOp::create(
            mlir::UnknownLoc::get(_context),
            llvm::StringRef(name + "Module"))),
        _module(*_ownedModule),
        _sourceMgrHandler(_sourceMgr, _context),
        _builder(_context)
    {
        auto argTypes = getArgTypes();
        _fnOp = CreateFunction(_builder, name, argTypes, {});
        auto entryBlock = _fnOp.addEntryBlock();
        mlir::OpBuilder bodyBuilder(entryBlock->getParent());
        mlir::OpBuilder::InsertionGuard guard(bodyBuilder);
        bodyBuilder.setInsertionPoint(entryBlock, std::prev(entryBlock->end()));

        std::vector<mlir::Value> args;
        for (auto arg : _fnOp.getArguments())
        {
            args.push_back(arg);
        }
        body(args);
        (void)bodyBuilder.create<mlir::ReturnOp>(bodyBuilder.getUnknownLoc());
    }

    mlir::OwningOpRef<mlir::ModuleOp>& Module() { return _ownedModule; }
    mlir::FuncOp& Func() { return _fnOp; }

private:
    mlir::MLIRContext _ownedContext;
    llvm::SourceMgr _sourceMgr;
    mlir::MLIRContext* _context;
    mlir::OwningOpRef<mlir::ModuleOp> _ownedModule;
    mlir::ModuleOp _module;
    mlir::SourceMgrDiagnosticHandler _sourceMgrHandler;
    mlir::OpBuilder _builder;
    mlir::FuncOp _fnOp;

    mlir::MLIRContext* InitContext()
    {
        accera::ir::InitializeAccera();
        _ownedContext.appendDialectRegistry(accera::ir::GetDialectRegistry());
        _ownedContext.loadAllAvailableDialects();
        return &_ownedContext;
    }

    mlir::FuncOp CreateFunction(mlir::OpBuilder& builder, std::string functionName, llvm::ArrayRef<mlir::Type> argTypes, llvm::ArrayRef<mlir::Type> returnTypes)
    {
        auto fnType = builder.getFunctionType(argTypes, returnTypes);
        auto fnOp = builder.create<mlir::FuncOp>(builder.getUnknownLoc(), functionName, fnType, llvm::None);

        return fnOp;
    }

    mlir::OpBuilder::InsertionGuard CreateNewScope(mlir::OpBuilder& builder, mlir::OpBuilder::InsertPoint insertionPoint = {})
    {
        mlir::OpBuilder::InsertionGuard guard(builder);

        if (insertionPoint.isSet())
        {
            builder.restoreInsertionPoint(insertionPoint);
        }

        return guard;

        // if (insertionPoint.isSet())
        // {
        //     return mlir::edsc::ScopedContext(builder, insertionPoint, builder.getUnknownLoc());
        // }
        // else
        // {
        //     return mlir::edsc::ScopedContext(builder, builder.getUnknownLoc());
        // }
    }
};

inline bool VerifyParse(TestContext& context, bool verbose = false, std::string outputFile = "")
{
    accera::utilities::logging::LogGuard guard(verbose);
    return VerifyParse(context.Module(), context.Func(), outputFile);
}

inline bool VerifyLowerToStd(TestContext& context, bool verbose = false, std::string outputFile = "")
{
    accera::utilities::logging::LogGuard guard(verbose);
    return VerifyLowerToStd(context.Module(), context.Func(), outputFile);
}

inline bool VerifyLowerToLLVM(TestContext& context, bool verbose = false, std::string outputFile = "")
{
    accera::utilities::logging::LogGuard guard(verbose);
    return VerifyLowerToLLVM(context.Module(), context.Func(), outputFile);
}

inline bool VerifyTranslateToLLVMIR(TestContext& context, bool optimize, bool verbose = false, std::string outputFile = "")
{
    accera::utilities::logging::LogGuard guard(verbose);
    return VerifyTranslateToLLVMIR(context.Module(), context.Func(), optimize, outputFile);
}
