////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/value/ValueDialect.h>

#include <utilities/include/TypeTraits.h>

#include <mlir/IR/Visitors.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <mlir/Transforms/Passes.h>

#include <llvm/Support/raw_os_ostream.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <variant>
#include <vector>

using namespace mlir;

namespace
{
using namespace scf;
#include "value/ValueConversion.inc"
} // namespace

using namespace accera::ir;
using namespace accera::transforms;
using namespace accera::ir::value;
using namespace accera::utilities;

using ValueBarrierOp = accera::ir::value::BarrierOp;

struct BarrierOptPass : public BarrierOptBase<BarrierOptPass>
{
    enum class MemoryAccessType
    {
        Read,
        Write,
    };

    // Maybe make this a map from op->operands?
    struct MemoryAccessInfo
    {
        Operation* op;
        mlir::Value baseMemRef;
        mlir::ValueRange indices;
        mlir::AffineMap accessMap;
        MemoryAccessType accessType;
    };

    struct BarrierInfo
    {
        ValueBarrierOp barrier;
        std::vector<MemoryAccessInfo> activeWrites;
        std::vector<MemoryAccessInfo> activeReads;
    };

    using MemoryOpInfo = std::variant<MemoryAccessInfo, BarrierInfo>;

    void runOnOperation() final;

private:
    bool _debug = false;

    std::vector<MemoryOpInfo> GatherMemoryOps(Operation* parentOp)
    {
        std::vector<MemoryOpInfo> memoryOps;

        parentOp->walk<WalkOrder::PreOrder>([&](Operation* op) {
            if (auto barrierOp = dyn_cast<BarrierOp>(op))
            {
                memoryOps.push_back(BarrierInfo{ barrierOp, {}, {} });
            }
            else if (auto memInfo = GetSharedMemoryAccessInfo(op))
            {
                memoryOps.push_back(*memInfo);
            }
        });

        return memoryOps;
    }

    llvm::Optional<MemoryAccessInfo>
    GetSharedMemoryAccessInfo(Operation* op)
    {
        auto getAffineAccessInfo = [](auto affineOp, MemoryAccessType accessType) -> llvm::Optional<MemoryAccessInfo> {
            auto memRefType = affineOp.getMemRefType();
            auto memSpace = memRefType.getMemorySpaceAsInt();
            if (memSpace == gpu::GPUDialect::getWorkgroupAddressSpace())
            {
                MemoryAccessInfo info;
                info.op = affineOp.getOperation();
                info.baseMemRef = affineOp.getMemRef();
                info.indices = affineOp.indices();
                info.accessMap = affineOp.getAffineMap();
                info.accessType = accessType;
                return info;
            }
            return llvm::None;
        };

        if (auto affineLoadOp = dyn_cast<mlir::AffineLoadOp>(op))
        {
            return getAffineAccessInfo(affineLoadOp, MemoryAccessType::Read);
        }
        else if (auto affineStoreOp = dyn_cast<mlir::AffineStoreOp>(op))
        {
            return getAffineAccessInfo(affineStoreOp, MemoryAccessType::Write);
        }

        return llvm::None;
    }
};

void BarrierOptPass::runOnOperation()
{
    auto memoryOps = GatherMemoryOps(getOperation());

    auto usesSameMemory = [&](const MemoryAccessInfo& access1, const MemoryAccessInfo& access2) {
        return access1.baseMemRef == access2.baseMemRef;
    };

    auto contains = [&](const std::vector<MemoryAccessInfo>& activeAccesses, const MemoryAccessInfo& access) {
        return std::find_if(activeAccesses.begin(), activeAccesses.end(), [&](const MemoryAccessInfo& activeAccess) {
                   return usesSameMemory(access, activeAccess);
               }) != activeAccesses.end();
    };

    std::vector<MemoryAccessInfo> activeReads;
    std::vector<MemoryAccessInfo> activeWrites;
    BarrierInfo prevBarrier;

    auto commitPrevBarrier = [&]() {
        prevBarrier = {};
        activeReads.clear();
        activeWrites.clear();
    };

    for (auto memoryOp : memoryOps)
    {
        std::visit(
            VariantVisitor{
                [&](BarrierInfo& barrierInfo) {
                    if (_debug)
                    {
                        auto out = barrierInfo.barrier.emitRemark("Barrier found with ") << activeWrites.size() << " active writes and " << activeReads.size() << " active reads\n";

                        if (activeWrites.size() > 0)
                            out << "Active writes:\n";
                        for (auto& memOpInfo : activeWrites)
                        {
                            out << memOpInfo.op << "\n";
                        }

                        if (activeReads.size() > 0)
                            out << "Active reads:\n";
                        for (auto& memOpInfo : activeReads)
                        {
                            out << memOpInfo.op << "\n";
                        }
                    }

                    if (prevBarrier.barrier)
                    {
                        if (_debug)
                            prevBarrier.barrier.emitRemark("BarrierOpRewrite: removing redundant barrier");
                        prevBarrier.barrier.erase();
                    }
                    auto barrier = barrierInfo.barrier;
                    if (!(isa<LoopLikeOpInterface>(barrier->getParentOp()) && activeWrites.empty() && activeReads.empty()))
                    {
                        prevBarrier = barrierInfo;
                    }
                },
                [&](MemoryAccessInfo& memOpInfo) {
                    if (memOpInfo.accessType == MemoryAccessType::Write)
                    {
                        // If this is a write to memory in the active reads list, we need a barrier
                        if (contains(activeReads, memOpInfo))
                        {
                            if (prevBarrier.barrier)
                            {
                                if (_debug)
                                    prevBarrier.barrier.emitRemark("Barrier needed because of write to memory in active reads");
                                commitPrevBarrier();
                            }
                        }

                        activeWrites.push_back(memOpInfo);
                    }
                    else
                    {
                        // If this is a read to memory in the active writes list, we need a barrier
                        if (contains(activeWrites, memOpInfo))
                        {
                            if (prevBarrier.barrier)
                            {
                                if (_debug)
                                    prevBarrier.barrier.emitRemark("Barrier needed because of read to memory in active writes");
                                commitPrevBarrier();
                            }
                        }

                        activeReads.push_back(memOpInfo);
                    }
                },
            },
            memoryOp);
    }

    // Delete prevBarrier.barrier if necessary
    if (prevBarrier.barrier)
    {
        if (_debug)
            prevBarrier.barrier.emitRemark("BarrierOpRewrite: removing redundant barrier");
        prevBarrier.barrier.erase();
    }
}

namespace accera::transforms::value
{

std::unique_ptr<mlir::Pass> createBarrierOptPass()
{
    return std::make_unique<BarrierOptPass>();
}

} // namespace accera::transforms::value
