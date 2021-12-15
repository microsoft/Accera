////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "value/ValueDialect.h"

#include "IRUtil.h"

namespace accera::ir::value
{

ValueFuncOp CreateRawPointerAPIWrapperFunction(mlir::OpBuilder& builder, ValueFuncOp functionToWrap, mlir::StringRef wrapperFnName)
{
    auto loc = functionToWrap.getLoc();
    mlir::OpBuilder::InsertionGuard insertGuard(builder);

    ValueModuleOp vModuleOp = functionToWrap->getParentOfType<ValueModuleOp>();

    auto insertionPoint = accera::ir::util::GetTerminalInsertPoint<ValueModuleOp, ModuleTerminatorOp>(vModuleOp);
    builder.restoreInsertionPoint(insertionPoint);

    ValueFuncOp apiWrapperFn = builder.create<ValueFuncOp>(loc, wrapperFnName, functionToWrap.getType(), ir::value::ExecutionTarget::CPU );
    apiWrapperFn->setAttr(ir::HeaderDeclAttrName, builder.getUnitAttr());
    apiWrapperFn->setAttr(ir::RawPointerAPIAttrName, builder.getUnitAttr());

    builder.setInsertionPointToStart(&apiWrapperFn.body().front());

    auto launchFuncOp = builder.create<LaunchFuncOp>(loc, functionToWrap, apiWrapperFn.getArguments());

    if (launchFuncOp.getNumResults() > 0)
    {
        builder.create<ReturnOp>(loc, launchFuncOp.getResults() );
    }
    else
    {
        builder.create<ReturnOp>(loc);
    }
    return apiWrapperFn;
}

} // namespace accera::ir::value
