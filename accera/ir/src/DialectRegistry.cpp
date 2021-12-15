////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DialectRegistry.h"
#include "exec/ExecutionPlanOps.h"
#include "nest/LoopNestOps.h"
#include "accera/AcceraOps.h"
#include "value/ValueDialect.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/IR/Dialect.h>

#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

using namespace mlir;

namespace accera::ir
{

mlir::DialectRegistry& GetDialectRegistry()
{
    static mlir::DialectRegistry registry;
    [[maybe_unused]] static bool init_once = [&]() {
        registry.insert<value::ValueDialect,
                        loopnest::LoopNestDialect,
                        executionPlan::ExecutionPlanDialect,
                        rc::AcceraDialect,

                        // MLIR dialects
                        StandardOpsDialect,
                        AffineDialect,
                        memref::MemRefDialect,
                        math::MathDialect,
                        gpu::GPUDialect,
                        linalg::LinalgDialect,
                        LLVM::LLVMDialect,
                        spirv::SPIRVDialect,
                        scf::SCFDialect,
                        vector::VectorDialect>();
        mlir::registerLLVMDialectTranslation(registry);
        return true;
    }();
    return registry;
}

} // namespace accera::ir
