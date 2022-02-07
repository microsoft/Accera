////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <value/include/MLIREmitterContext.h>

#include <mlir/Dialect/GPU/Passes.h>

#include <memory>

class SerializeToHsacoPass : public accera::transforms::SerializeToHSACOBase<SerializeToHsacoPass>
{
public:
    void runOnOperation() override
    {
        // noop
    }
};

namespace accera::transforms
{
std::unique_ptr<mlir::OperationPass<mlir::gpu::GPUModuleOp>> createSerializeToHSACOPass()
{
    return std::make_unique<SerializeToHsacoPass>();
}
} // namespace accera::transforms
