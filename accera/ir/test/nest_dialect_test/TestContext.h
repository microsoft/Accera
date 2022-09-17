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
        SetTestBuilder(&_builder);
        _builder.setInsertionPoint(_module.getBody(), _module.getBody()->begin());
        _fnOp = CreateFunction(_builder, name, {}, {});
        auto entryBlock = _fnOp.addEntryBlock();

        mlir::OpBuilder::InsertionGuard guard(_builder);
        _builder.setInsertionPoint(entryBlock, std::prev(entryBlock->end()));
        
        setup();
        
        (void)_builder.create<mlir::ReturnOp>(_builder.getUnknownLoc());
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
        SetTestBuilder(&_builder);
        _builder.setInsertionPoint(_module.getBody(), _module.getBody()->begin());
        auto argTypes = getArgTypes();
        _fnOp = CreateFunction(_builder, name, argTypes, {});
        auto entryBlock = _fnOp.addEntryBlock();
        mlir::OpBuilder::InsertionGuard guard(_builder);
        _builder.setInsertionPoint(entryBlock, std::prev(entryBlock->end()));

        std::vector<mlir::Value> args;
        for (auto arg : _fnOp.getArguments())
        {
            args.push_back(arg);
        }
        body(args);
        (void)_builder.create<mlir::ReturnOp>(_builder.getUnknownLoc());
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
    }
};

inline bool VerifyGenerate(TestContext& context, bool verbose = false, std::string outputFile = "")
{
    accera::utilities::logging::LogGuard guard(verbose);
    return VerifyGenerate(context.Module(), context.Func(), outputFile);
}

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
