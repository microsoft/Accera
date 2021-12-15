////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace mlir
{
class Pass;
class PassManager;
} // namespace mlir

namespace accera::transforms
{
namespace loopnest
{
    std::unique_ptr<mlir::Pass> createScheduledOperationsPass();
    std::unique_ptr<mlir::Pass> createScheduleToValuePass();
    std::unique_ptr<mlir::Pass> createLoopNestOptPass();

    void addLoopNestLoweringPasses(mlir::PassManager& pm);
    void addLoopNestStructureLoweringPasses(mlir::PassManager& pm);
    void addLoopNestFinalLoweringPasses(mlir::PassManager& pm);
    void addLoopNestCleanupLoweringPasses(mlir::PassManager& pm);
} // namespace loopnest
} // namespace accera::transforms
