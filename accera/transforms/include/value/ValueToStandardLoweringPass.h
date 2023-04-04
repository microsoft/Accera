////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class FuncOp;

template <typename OpT>
class OperationPass;

class RewritePatternSet;
} // namespace mlir

namespace
{
struct ProfileRegions;
}

const char kTimerRegionTypeIdentifier[] = "timer_region_type";

enum class TimerRegionType
{
    enterRegion = 0,
    exitRegion = 1,
};

namespace accera::transforms::value
{
inline std::string GetSplitSizeAttrName()
{
    return "split_size";
}
} // namespace accera::transforms::value

namespace accera::transforms::value
{
void populateVectorizeValueOpPatterns(mlir::RewritePatternSet& patterns);
[[maybe_unused]] void populateValueToStandardPatterns(bool enableProfiling, ProfileRegions& profileRegions, mlir::RewritePatternSet& patterns);
void populateValueLaunchFuncPatterns(mlir::RewritePatternSet& patterns);
void populateValueModuleRewritePatterns(mlir::RewritePatternSet& patterns);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToStdPass(bool enableProfiling = false);
} // namespace accera::transforms::value
