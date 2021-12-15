////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa, Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ConvertToLLVM.h"

#include <ir/include/value/ValueDialect.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

namespace accera
{
namespace ir
{
    mlir::ModuleOp ConvertToLLVM(
        mlir::ModuleOp module,
        std::function<void(mlir::PassManager& pm)> addStdPassesFn,
        std::function<void(mlir::PassManager& pm)> addLLVMPassesFn)
    {
        auto moduleCopy = module.clone();

        mlir::PassManager pm(moduleCopy.getContext());

        // // add custom passes
        addStdPassesFn(pm);

        // canonicalize everything
        // pm.addPass(mlir::createCanonicalizerPass());

        auto& funcOpPM = pm.nest<value::ValueModuleOp>().nest<FuncOp>();

        // linalg -> affine
        funcOpPM.addPass(mlir::createConvertLinalgToAffineLoopsPass());

        // simplify affine
        funcOpPM.addPass(mlir::createSimplifyAffineStructuresPass());

        // affine -> loops
        funcOpPM.addPass(mlir::createLowerAffinePass());

        // loops -> std
        pm.addPass(mlir::createLowerToCFGPass());

        // add custom LLVM passes
        addLLVMPassesFn(pm);

        // linalg -> llvm
        pm.addPass(mlir::createConvertLinalgToLLVMPass());

        // another canonicalization pass
        pm.addPass(mlir::createCanonicalizerPass());

        // std -> llvm
        auto llvmOptions = mlir::LowerToLLVMOptions(moduleCopy.getContext());
        llvmOptions.allocLowering = mlir::LowerToLLVMOptions::AllocLowering::AlignedAlloc;
        pm.addPass(mlir::createLowerToLLVMPass(llvmOptions));

        if (mlir::failed(pm.run(moduleCopy)))
        {
            moduleCopy.dump();
            throw std::runtime_error("failed to lower module to llvm dialect!");
        }

        return moduleCopy;
    }
} // namespace mlirHelpers
} // namespace accera
