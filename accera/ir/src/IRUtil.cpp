////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRUtil.h"
#include "value/ValueAttributes.h"
#include "value/ValueEnums.h"

#include <ir/include/exec/ExecutionOptions.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueDialect.h>

#include <utilities/include/StringUtil.h>

#include <mlir/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <set>

namespace vir = accera::ir::value;

namespace
{
static std::mutex _globalInsertMutex;
}

namespace accera::ir
{
namespace util
{
    void FillCanonicalPatternsRecursively(mlir::Operation* op, mlir::OwningRewritePatternList& patterns)
    {
        std::set<const mlir::AbstractOperation*> s;
        auto context = op->getContext();

        auto abstractOp = op->getAbstractOperation();
        if (s.count(abstractOp) == 0)
        {
            abstractOp->getCanonicalizationPatterns(patterns, context);
            s.insert(abstractOp);
        }
        op->walk([&patterns, &s, context](mlir::Operation* childOp) {
            auto childAbstractOp = childOp->getAbstractOperation();
            if (s.count(childAbstractOp) == 0)
            {
                childAbstractOp->getCanonicalizationPatterns(patterns, context);
                s.insert(childAbstractOp);
            }
        });
    }

    void CanonicalizeGreedily(mlir::Operation* op)
    {
        mlir::OwningRewritePatternList patterns(op->getContext());
        FillCanonicalPatternsRecursively(op, patterns);
        (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
    }

    std::vector<int64_t> ConvertArrayAttrToIntVector(const mlir::ArrayAttr& inputArrayAttr)
    {
        return ArrayAttrToVector<int64_t, mlir::IntegerAttr>(
            inputArrayAttr,
            [](const mlir::IntegerAttr& intAttr) { return intAttr.getInt(); });
    }

    std::vector<loopnest::Index> ConvertArrayAttrToIndexVector(const mlir::ArrayAttr& inputArrayAttr)
    {
        return ArrayAttrToVector<loopnest::Index, loopnest::IndexAttr>(
            inputArrayAttr,
            [](const loopnest::IndexAttr& indexAttr) { return indexAttr.getValue(); });
    }

    mlir::ArrayAttr ConvertIndexVectorToArrayAttr(const std::vector<loopnest::Index>& inputVec, mlir::MLIRContext* context)
    {
        return VectorToArrayAttr<loopnest::Index, loopnest::IndexAttr>(
            inputVec,
            [&](const loopnest::Index& index) { return loopnest::IndexAttr::get(index, context); },
            context);
    }

    mlir::Attribute GetOneAttr(mlir::OpBuilder& builder, mlir::Type type)
    {
        if (type.isa<mlir::FloatType>())
            return builder.getFloatAttr(type, 1.0);
        if (type.isa<mlir::IndexType>())
            return builder.getIndexAttr(1);
        if (auto integerType = type.dyn_cast<mlir::IntegerType>())
            return builder.getIntegerAttr(type, mlir::APInt(type.cast<mlir::IntegerType>().getWidth(), 1));
        if (type.isa<mlir::RankedTensorType, mlir::VectorType>())
        {
            auto vtType = type.cast<mlir::ShapedType>();
            auto element = GetOneAttr(builder, vtType.getElementType());
            if (!element)
                return {};
            return mlir::DenseElementsAttr::get(vtType, element);
        }
        return {};
    }

    mlir::OpBuilder MakeBodyBuilder(mlir::AffineForOp forOp)
    {
        auto& region = forOp.region();
        mlir::Block* front = &region.front();
        return { front, std::prev(front->end()) };
    }

    mlir::Value CreateStackBuffer(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, int64_t alignment)
    {
        auto funcParent = anchorOp->getParentOfType<ir::value::ValueFuncOp>();
        mlir::Block& funcBlock = funcParent.front();

        mlir::OpBuilder::InsertionGuard insertGuard(builder);
        builder.setInsertionPointToStart(&funcBlock);
        auto tmpBuffer = builder.create<mlir::memref::AllocaOp>(anchorOp->getLoc(), bufferType, mlir::ValueRange{}, builder.getI64IntegerAttr(alignment));

        return tmpBuffer;
    }

    mlir::Value CreateGlobalBuffer(mlir::OpBuilder& builder, mlir::MemRefType bufferType, const std::string& namePrefix, const bool constant, Attribute attr, bool isExternal, bool appendUniqueSuffix)
    {
        auto insertionBlock = builder.getInsertionBlock();
        auto parentOp = insertionBlock->getParentOp();
        return CreateGlobalBuffer(builder, parentOp, bufferType, namePrefix, constant, attr, isExternal, appendUniqueSuffix);
    }

    ir::value::GlobalOp CreateGlobalBufferOp(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, std::string globalName, const bool constant, Attribute attr, bool isExternal, bool appendUniqueSuffix)
    {
        auto loc = anchorOp->getLoc();
        if (appendUniqueSuffix)
            globalName += "_" + std::to_string(GetUniqueId());

        mlir::OpBuilder::InsertionGuard guard(builder);

        mlir::Block* body;
        if (auto moduleValue = util::CastOrGetParentOfType<ir::value::ValueModuleOp>(anchorOp))
        {
            body = moduleValue.getBody();
        }
        else if (auto moduleGPU = util::CastOrGetParentOfType<mlir::gpu::GPUModuleOp>(anchorOp))
        {
            body = moduleGPU.getBody();
        }
        else
        {
            auto moduleBase = util::CastOrGetParentOfType<mlir::ModuleOp>(anchorOp);
            assert(moduleBase && "Expected to be inside a ValueModuleOp");
            body = moduleBase.getBody();
        }

        // Lock before accessing the global scope so that multi-threaded lowerings all access the appropriate global insert position
        std::lock_guard<std::mutex> lock(_globalInsertMutex);
        builder.setInsertionPoint(body, body->begin());
        return builder.create<accera::ir::value::GlobalOp>(loc, bufferType, constant, globalName, attr, /*addrSpace*/ 0, isExternal);
    }

    mlir::Value CreateGlobalBuffer(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, const std::string& namePrefix, const bool constant, Attribute attr, bool isExternal, bool appendUniqueSuffix)
    {
        auto globalOp = CreateGlobalBufferOp(builder, anchorOp, bufferType, namePrefix, constant, attr, isExternal, appendUniqueSuffix);

        auto insertionBlock = anchorOp->getBlock();
        auto it = insertionBlock->begin();
        auto end = insertionBlock->end();
        while (it != end && llvm::isa<mlir::ConstantOp,
                                      ir::value::ReferenceGlobalOp>(it))
        {
            ++it;
        }
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(insertionBlock, it);

        auto loc = anchorOp->getLoc();
        auto reference = builder.create<accera::ir::value::ReferenceGlobalOp>(loc, globalOp);
        return reference.getResult();
    }

    mlir::Value CreateSharedBuffer(mlir::OpBuilder& builder, mlir::MemRefType bufferType, const std::string& namePrefix)
    {
        // Todo: implement this method later
        return CreateGlobalBuffer(builder, bufferType, namePrefix);
    }

    mlir::Value CreateSharedBuffer(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, const std::string& namePrefix)
    {
        auto loc = anchorOp->getLoc();
        auto insertionBlock = anchorOp->getBlock();
        auto it = insertionBlock->begin();
        auto end = insertionBlock->end();
        while (it != end && llvm::isa<mlir::ConstantOp,
                                      mlir::memref::AllocOp,
                                      mlir::memref::AllocaOp,
                                      mlir::LLVM::AllocaOp,
                                      ir::value::ReferenceGlobalOp,
                                      ir::value::AllocOp>(it))
        {
            ++it;
        }
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(insertionBlock, it);
        auto op = builder.create<mlir::memref::AllocaOp>(loc, bufferType, mlir::ValueRange{}, builder.getI64IntegerAttr(32));

        return op.getResult();
    }

    mlir::Value CreatePrivateBuffer(mlir::OpBuilder& builder, mlir::MemRefType bufferType, const std::string& namePrefix)
    {
        // Todo: implement this method later
        return CreateGlobalBuffer(builder, bufferType, namePrefix);
    }

    mlir::Value CreatePrivateBuffer(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, const std::string& namePrefix)
    {
        // Todo: implement this method later
        return CreateGlobalBuffer(builder, anchorOp, bufferType, namePrefix);
    }

    mlir::Location GetLocation(mlir::OpBuilder& builder, std::string tag)
    {
        return mlir::FileLineColLoc::get(mlir::Identifier::get(tag, builder.getContext()), 0, 0);
    }

    mlir::Location GetLocation(mlir::OpBuilder& builder, std::string tag, mlir::Location opLocation)
    {
        return builder.getFusedLoc(opLocation, { GetLocation(builder, tag) });
    }

    mlir::Location GetLocation(mlir::OpBuilder& builder, std::string filename, int64_t lineNumber)
    {
        utilities::ReplaceAll(filename, "\\", "/");
        return mlir::FileLineColLoc::get(builder.getIdentifier(filename), lineNumber, 0);
    }

    std::vector<mlir::Value> MultiDimAffineApply(mlir::OpBuilder& builder, mlir::Location loc, mlir::AffineMap map, std::vector<mlir::Value>& operands)
    {
        std::vector<mlir::Value> result;
        result.reserve(map.getNumResults());
        for (unsigned int resultIdx = 0; resultIdx < map.getNumResults(); ++resultIdx)
        {
            auto singleResultSubMap = map.getSubMap({ resultIdx });
            result.push_back(builder.create<mlir::AffineApplyOp>(loc, singleResultSubMap, operands));
        }
        return result;
    }

    AffineMap MakeIdentityAccessMap(mlir::Value val, mlir::MLIRContext* context)
    {
        auto type = val.getType();
        assert(type.isa<MemRefType>() && "Value must be a memref type in order to be accessible");
        auto memRefType = type.cast<MemRefType>();
        auto valRank = memRefType.getRank();
        return AffineMap::getMultiDimIdentityMap(valRank, context);
    }

    mlir::Type GetElementType(mlir::Type type)
    {
        auto result =
            mlir::TypeSwitch<mlir::Type, mlir::Type>(type)
                .Case([&](mlir::ShapedType t) { return t.getElementType(); })
                .Default([&](mlir::Type t) { return t; });
        return result;
    }

    int64_t GetUniqueId()
    {
        static std::atomic<int64_t> nextId = 0;
        return nextId++;
    }

    mlir::Operation* CloneRecursively(mlir::OpBuilder& builder, mlir::Operation* op, mlir::BlockAndValueMapping& mapping)
    {
        for (auto operand : op->getOperands())
        {
            if (!mapping.contains(operand))
            {
                if (operand.isa<mlir::BlockArgument>())
                {
                    mapping.map(operand, operand);
                }
                else
                {
                    auto definingOp = operand.getDefiningOp();
                    auto clonedOp = CloneRecursively(builder, definingOp, mapping);
                    for (unsigned idx = 0, e = clonedOp->getNumResults(); idx != e; ++idx)
                    {
                        mapping.map(definingOp->getResult(idx), clonedOp->getResult(idx));
                    }
                }
            }
        }

        return builder.clone(*op, mapping);
    }

    std::optional<vir::ExecutionTarget> ResolveExecutionTarget(mlir::Operation* op, bool exact /* = false */)
    {
        // modules can define the execution target
        // search if the current module specifies the execution target
        auto getExecTarget = [](Operation* op_) { return op_->getAttrOfType<vir::ExecutionTargetAttr>(vir::ValueFuncOp::getExecTargetAttrName()); };

        Operation* execAwareOp = op;
        auto execTargetAttr = getExecTarget(execAwareOp);
        while (!exact &&
               execAwareOp &&
               !execAwareOp->hasTrait<mlir::OpTrait::FunctionLike>() &&
               !execTargetAttr)
        {
            if ((execAwareOp = execAwareOp->getParentWithTrait<mlir::OpTrait::FunctionLike>()))
            {
                execTargetAttr = getExecTarget(execAwareOp);
            }
        }

        if (execTargetAttr)
        {
            return execTargetAttr.getValue();
        }

        if (!execAwareOp)
        {
            return std::nullopt;
        }

        return mlir::TypeSwitch<Operation*, std::optional<vir::ExecutionTarget>>(execAwareOp)
            .Case([](mlir::gpu::GPUFuncOp) {
                return vir::ExecutionTarget::GPU;
            })
            .Case([](mlir::spirv::FuncOp) {
                return vir::ExecutionTarget::GPU;
            })
            .Case([](mlir::FuncOp) {
                return vir::ExecutionTarget::CPU;
            })
            .Case([](mlir::LLVM::LLVMFuncOp) {
                return vir::ExecutionTarget::CPU;
            })
            .Default([](Operation* op_) {
                op_->emitWarning("Couldn't determine execution environment");
                return std::nullopt;
            });
    }

    std::optional<vir::ExecutionRuntime> ResolveExecutionRuntime(mlir::Operation* op, bool exact /* = false */)
    {
        auto execRuntimeAttrName = ir::value::ValueModuleOp::getExecRuntimeAttrName();

        auto getExecRuntime = [&](Operation* op_) {
            return op_->getAttrOfType<vir::ExecutionRuntimeAttr>(execRuntimeAttrName);
        };

        Operation* moduleLikeOp = op;
        auto execRuntimeAttr = getExecRuntime(moduleLikeOp);
        // if the runtime attribute is not found in the rcv.module, then
        // search the mlir.module for the runtime (using a fully qualified attribute name)
        if (!exact && op && !execRuntimeAttr)
        {
            if ((moduleLikeOp = op->getParentOfType<vir::ValueModuleOp>()))
            {
                execRuntimeAttr = getExecRuntime(moduleLikeOp);
            }
            if (!execRuntimeAttr && (moduleLikeOp = op->getParentOfType<mlir::ModuleOp>()))
            {
                execRuntimeAttr = getExecRuntime(moduleLikeOp);
            }
        }

        // the runtime attribute was not set by the user, so set it to NONE
        if (!execRuntimeAttr)
        {
            return vir::ExecutionRuntime::NONE;
        }

        return execRuntimeAttr.getValue();
    }

    std::optional<std::pair<int, int>> ResolveWarpSize(const vir::ExecutionRuntime runtime)
    {
        if (runtime == vir::ExecutionRuntime::CUDA)
            return std::make_pair(8, 4); // 32

        if (runtime == vir::ExecutionRuntime::ROCM)
            return std::make_pair(8, 8); // 64

        return std::nullopt;
    }

    mlir::Operation* CreateGPUControlBarrier(mlir::OpBuilder& builder, const std::string scope, std::optional<mlir::Location> loc /*= std::nullopt*/)
    {
        auto barrierScope = vir::symbolizeEnum<value::BarrierScope>(scope);
        assert(barrierScope && "Invalid barrier scope");
        return builder.create<vir::BarrierOp>(
            loc.value_or(builder.getUnknownLoc()),
            vir::BarrierScopeAttr::get(builder.getContext(), *barrierScope));
    }

    std::optional<int64_t> GetDimSizeAt(const loopnest::Index& dimensionIndex, mlir::Operation* where)
    {
        assert(where != nullptr);
        mlir::Operation* parentOp = where;

        while ((parentOp = parentOp->getParentOp()) && !mlir::isa<vir::ValueFuncOp>(parentOp) && !mlir::isa<mlir::FuncOp>(parentOp) && !mlir::isa<loopnest::KernelOp>(parentOp))
        {
            if (auto subdomainIndexOrderAttr = parentOp->getAttrOfType<ArrayAttr>("subdomainIndexOrder"))
            {
                auto subdomainIndexOrder = util::ConvertArrayAttrToIndexVector(subdomainIndexOrderAttr);
                auto iter = std::find(subdomainIndexOrder.begin(), subdomainIndexOrder.end(), dimensionIndex);

                if (iter != subdomainIndexOrder.end())
                {
                    if (auto subdomainSizeAttr = parentOp->getAttrOfType<ArrayAttr>("subdomainSize"))
                    {
                        auto subdomainSizes = util::ConvertArrayAttrToIntVector(subdomainSizeAttr);
                        assert(subdomainSizes.size() == subdomainIndexOrder.size() && "subdomainSize and subdomainIndexOrder must have the same number of elements");

                        size_t idx = std::distance(subdomainIndexOrder.begin(), iter);
                        auto dimensionSize = subdomainSizes[idx];
                        return dimensionSize;
                    }
                }
            }
        }

        return {};
    }

    std::vector<std::pair<loopnest::Index, mlir::Value>> ResolveUnrealizedNestIndices(mlir::Operation* where)
    {
        std::vector<std::pair<loopnest::Index, mlir::Value>> result;
        if (auto kernelOp = CastOrGetParentOfType<loopnest::KernelOp>(where))
        {
            auto symbolicIndexOps = kernelOp.getIndices();
            for (auto& symbolicIndexOp : symbolicIndexOps)
            {
                result.emplace_back(std::make_pair(symbolicIndexOp.getValue(), symbolicIndexOp));
            }
        }
        return result;
    }

    std::vector<mlir::Value> GetCurrentIndexIVs(const std::vector<loopnest::Index>& loopIndices, mlir::Operation* where)
    {
        return GetCurrentIndexIVs(loopIndices, where->getBlock());
    }

    std::vector<mlir::Value> GetCurrentIndexIVs(const std::vector<loopnest::Index>& loopIndices, mlir::Block* where)
    {
        std::vector<mlir::Value> ivs(loopIndices.size());

        auto blockParentOp = where->getParentOp();

        std::vector<std::pair<loopnest::Index, mlir::Value>> unrealizedLoopNestIndices = ResolveUnrealizedNestIndices(blockParentOp);

        // First check the unrealizedLoopNestIndices for any loopnest indices that haven't been resolved to full AffineForOps yet
        for (const auto& indexIVPair : unrealizedLoopNestIndices)
        {
            const auto& currentIndex = indexIVPair.first;
            const auto& currentIV = indexIVPair.second;
            auto it = std::find_if(loopIndices.begin(), loopIndices.end(), [&](const loopnest::Index& searchIndex) {
                return (searchIndex == currentIndex) ||
                       (searchIndex.GetId() == loopnest::Index::DefaultID &&
                        searchIndex.GetName() == currentIndex.GetName());
            });
            if (it != loopIndices.end())
            {
                size_t idx = std::distance(loopIndices.begin(), it);
                assert(ivs[idx] == nullptr && "Found same index on multiple loops");
                ivs[idx] = currentIV;
            }
        }

        mlir::AffineForOp currentParentLoop;
        if (mlir::isa<mlir::AffineForOp>(blockParentOp))
        {
            currentParentLoop = mlir::dyn_cast<mlir::AffineForOp>(blockParentOp);
        }
        else
        {
            currentParentLoop = blockParentOp->getParentOfType<mlir::AffineForOp>();
        }

        while (currentParentLoop != nullptr)
        {
            if (auto indexAttr = currentParentLoop->getAttrOfType<loopnest::IndexAttr>("index"))
            {
                auto currentIndex = indexAttr.getValue();

                // If the indices we're looking for have a default ID, then only compare by the name of the index
                // This is to support well-known named loops created internally by Accera
                // If the ID's are not the default, then compare IDs as well
                auto it = std::find_if(loopIndices.begin(), loopIndices.end(), [&](const loopnest::Index& searchIndex) {
                    return (searchIndex == currentIndex) ||
                           (searchIndex.GetId() == loopnest::Index::DefaultID &&
                            searchIndex.GetName() == currentIndex.GetName());
                });

                if (it != loopIndices.end())
                {
                    size_t idx = std::distance(loopIndices.begin(), it);
                    assert(ivs[idx] == nullptr && "Found same index on multiple loops");
                    ivs[idx] = currentParentLoop.getInductionVar();
                }
            }
            currentParentLoop = currentParentLoop->getParentOfType<mlir::AffineForOp>();
        }

        for ([[maybe_unused]] auto iv : ivs)
        {
            assert(iv != nullptr && "Couldn't find all loop indices");
        }

        return ivs;
    }

    std::vector<loopnest::Index> GetIndicesForLoopIVs(const std::vector<mlir::Value>& loopIVs)
    {
        std::vector<loopnest::Index> loopIndices;

        for (const auto& loopIV : loopIVs)
        {
            mlir::AffineForOp loop = mlir::getForInductionVarOwner(loopIV);
            assert(loop != nullptr && "Couldn't find loop with the given IV");
            if (auto indexAttr = loop->getAttrOfType<loopnest::IndexAttr>("index"))
            {
                loopIndices.push_back(indexAttr.getValue());
            }
            else
            {
                assert(false && "Found an AffineForOp with the given IV, but it did not have an IndexAttr");
            }
        }

        return loopIndices;
    }

    mlir::AffineMap ConcatenateAndShiftAffineDimsAndMaps(mlir::OpBuilder& builder, mlir::AffineMap leftMap, mlir::AffineMap rightMap)
    {
        // Differs from mlir::concatAffineMaps in that it shifts the dimensions of the right-hand map
        // and uses all of the dimensions from the two maps independently instead of merging the dimensions

        unsigned dimShift = leftMap.getNumDims();
        std::vector<mlir::AffineExpr> dimReplacements;
        std::vector<mlir::AffineExpr> symReplacements;
        for (unsigned originalDimIdx = 0; originalDimIdx < rightMap.getNumDims(); ++originalDimIdx)
        {
            dimReplacements.push_back(builder.getAffineDimExpr(originalDimIdx + dimShift));
        }
        auto shiftedRightMap = rightMap.replaceDimsAndSymbols(dimReplacements, symReplacements, rightMap.getNumDims() + dimShift, rightMap.getNumSymbols());

        auto leftMapExprs = leftMap.getResults().vec();
        auto concatedExprs = shiftedRightMap.getResults().vec();
        concatedExprs.insert(concatedExprs.begin(), leftMapExprs.begin(), leftMapExprs.end());

        auto concatedMap = mlir::AffineMap::get(shiftedRightMap.getNumDims(), shiftedRightMap.getNumSymbols(), concatedExprs, builder.getContext());

        return concatedMap;
    }

    bool IsSubdomainEmpty(mlir::Operation* where)
    {
        mlir::Operation* parentOp = where;

        while ((parentOp = parentOp->getParentOp()) &&
               !mlir::isa<vir::ValueFuncOp>(parentOp) &&
               !mlir::isa<mlir::FuncOp>(parentOp) &&
               !mlir::isa<loopnest::KernelOp>(parentOp))
        {
            if (auto subdomainSizeAttr = parentOp->getAttrOfType<ArrayAttr>("subdomainSize"))
            {
                auto subdomainSizes = util::ConvertArrayAttrToIntVector(subdomainSizeAttr);
                for (auto subdomainDimSize : subdomainSizes)
                {
                    if (subdomainDimSize == 0)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    void InlineAllRegionOpsBeforeOp(mlir::PatternRewriter& rewriter, mlir::Region& regionToInsert, mlir::Operation* op)
    {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);

        auto parentBlock = op->getBlock();
        auto parentRegion = parentBlock->getParent();

        // Get an iterator to the position of this op in the parent block
        mlir::Block::iterator insertPosition(op);

        // Split the parent block before the op we're inserting before
        auto opContainerBlock = rewriter.splitBlock(parentBlock, insertPosition);

        // Inline the contents of the given block in the spot between blocks that we
        // just created with the splitBlock() call
        rewriter.inlineRegionBefore(regionToInsert, opContainerBlock);

        // Now we have 3 blocks to pay attention to:
        // The original block that we split off from, the predecessorBlock
        // The block we inlined before the target op, the inlinedBlock
        // The block that our target op and anything that came after it is in, the successorBlock == opContainerBlock
        // There may also be blocks before the predecessorBlock and/or after the successorBlock that we shouldn't touch

        // Find the successorBlock with an iterator and step back from it to find the inlinedBlock and the predecessorBlock
        auto successorBlock = opContainerBlock;

        auto blockIter = parentRegion->begin();
        while (blockIter != parentRegion->end() && &(*blockIter) != successorBlock)
        {
            ++blockIter;
        }
        assert(&(*blockIter) == successorBlock && "Failed to find the successorBlock we created as part of lowering");
        // Now step back to find the other blocks
        --blockIter;
        auto& inlinedBlock = *blockIter;
        --blockIter;
        auto& predecessorBlock = *blockIter;

        // We want to erase any terminator in the inlinedBlock that was originally in the regionToInsert's block
        auto inlinedBlockTerminator = inlinedBlock.getTerminator();
        rewriter.eraseOp(inlinedBlockTerminator);

        // Now merge the inlinedBlock into the predecessorBlock
        rewriter.mergeBlocks(&inlinedBlock, &predecessorBlock);

        // Now merge the successorBlock into the predecessorBlock
        rewriter.mergeBlocks(successorBlock, &predecessorBlock);
    }

    mlir::Operation* FindOpWithSymbolName(const llvm::StringRef& id, mlir::Operation* rootOp)
    {
        auto symTableOp = mlir::SymbolTable::getNearestSymbolTable(rootOp);
        auto symbolOp = mlir::SymbolTable::lookupNearestSymbolFrom(symTableOp, id);
        return symbolOp;
    }

    mlir::LogicalResult PromoteIfSingleIteration(mlir::PatternRewriter& rewriter, mlir::AffineForOp forOp)
    {
        // Copied and modified from llvm-project\mlir\lib\Transforms\Utils\LoopUtils.cpp : mlir::promoteIfSingleIteration()
        // Modified to work during a lowering pass (i.e. erase ops via the PatternRewriter rather than erasing the ops directly)
        // and to work within a ValueFuncOp as opposed to a std FuncOp

        llvm::Optional<uint64_t> tripCount = mlir::getConstantTripCount(forOp);
        if (!tripCount || tripCount.getValue() != 1)
            return mlir::failure();

        if (forOp.getLowerBoundMap().getNumResults() != 1)
            return mlir::failure();

        // Replaces all IV uses to its single iteration value.
        auto iv = forOp.getInductionVar();
        auto* parentBlock = forOp->getBlock();
        if (!iv.use_empty())
        {
            if (forOp.hasConstantLowerBound())
            {
                mlir::OpBuilder topBuilder(forOp->getParentOfType<vir::ValueFuncOp>().getBody());
                auto constOp = topBuilder.create<mlir::ConstantIndexOp>(
                    forOp.getLoc(), forOp.getConstantLowerBound());
                iv.replaceAllUsesWith(constOp);
            }
            else
            {
                auto lbOperands = forOp.getLowerBoundOperands();
                auto lbMap = forOp.getLowerBoundMap();
                mlir::OpBuilder builder(parentBlock, mlir::Block::iterator(forOp));
                if (lbMap == builder.getDimIdentityMap())
                {
                    // No need of generating an affine.apply.
                    iv.replaceAllUsesWith(lbOperands[0]);
                }
                else
                {
                    auto affineApplyOp =
                        builder.create<mlir::AffineApplyOp>(forOp.getLoc(), lbMap, lbOperands);
                    iv.replaceAllUsesWith(affineApplyOp);
                }
            }
        }
        // Move the loop body operations, except for its terminator, to the loop's
        // containing block.
        rewriter.eraseOp(forOp.getBody()->getTerminator());

        parentBlock->getOperations().splice(mlir::Block::iterator(forOp),
                                            forOp.getBody()->getOperations());

        rewriter.eraseOp(forOp);
        return mlir::success();
    }

    bool OperationsAreEqual(mlir::Operation* lhs, mlir::Operation* rhs)
    {
        if (lhs == rhs)
        {
            return true;
        }

        // Check that the operations have the same type, operands, and attributes

        // Check op type
        auto abstractLHSOp = lhs->getAbstractOperation();
        auto abstractRHSOp = rhs->getAbstractOperation();
        if (abstractLHSOp->typeID != abstractRHSOp->typeID)
        {
            return false;
        }

        // Check operands
        if (lhs->getNumOperands() != rhs->getNumOperands())
        {
            return false;
        }
        if (!std::equal(lhs->operand_begin(), lhs->operand_end(), rhs->operand_begin(), rhs->operand_end()))
        {
            return false;
        }

        // Check attributes
        auto lhsAttrDict = lhs->getAttrDictionary();
        auto rhsAttrDict = rhs->getAttrDictionary();
        if (lhsAttrDict.size() != rhsAttrDict.size())
        {
            return false;
        }
        for (auto namedAttr : lhsAttrDict.getValue())
        {
            auto lhsAttr = namedAttr.second;
            auto rhsAttr = rhsAttrDict.get(namedAttr.first);
            if (lhsAttr != rhsAttr)
            {
                return false;
            }
        }

        return true;
    }

    mlir::Value CreateConstantRangeForOpIterationCounter(mlir::OpBuilder& builder, mlir::Location loc, mlir::AffineForOp forOp)
    {
        mlir::OpBuilder::InsertionGuard insertGuard(builder);
        builder.setInsertionPointToStart(forOp.getBody());

        assert(forOp.hasConstantBounds() && "AffineForOp must have constant bounds");
        auto lowerBound = forOp.getConstantLowerBound();
        auto step = forOp.getStep();

        // Compute (iv - lowerBound) / step
        auto iterCounterMap = AffineMap::get(1, 0, (builder.getAffineDimExpr(0) - builder.getAffineConstantExpr(lowerBound)).floorDiv(step));
        return builder.create<mlir::AffineApplyOp>(loc, iterCounterMap, mlir::ValueRange{ forOp.getInductionVar() });
    }

    mlir::Operation* GetFirstOp(mlir::Operation* left, mlir::Operation* right)
    {
        assert(left->getBlock() == right->getBlock() && "This utility only supports ops in the same block");
        auto block = left->getBlock();
        auto beginIter = block->begin();
        auto endIter = block->end();
        for (auto iter = beginIter; iter != endIter; ++iter)
        {
            if (&(*iter) == left)
            {
                return left;
            }
            else if (&(*iter) == right)
            {
                return right;
            }
        }
        assert(false && "Neither op found in block");
        return nullptr;
    }

    mlir::AffineMap ComposeAffineMapSequence(const std::vector<mlir::AffineMap>& maps)
    {
        if (maps.empty())
        {
            return mlir::AffineMap();
        }
        else
        {
            auto accessMapComposition = maps.front();
            for (size_t mapIdx = 1; mapIdx < maps.size(); ++mapIdx)
            {
                accessMapComposition = maps[mapIdx].compose(accessMapComposition);
            }
            return accessMapComposition;
        }
    }

    template <typename MemoryOp>
    mlir::AffineMap GetMemRefIndexToMemoryLocationMap(mlir::MLIRContext* context, MemoryOp op)
    {
        auto memRefType = op.memref().getType().template cast<mlir::MemRefType>();
        std::vector<mlir::AffineMap> memRefMaps = memRefType.getAffineMaps().vec();
        if (memRefMaps.empty())
        {
            auto stridedLayout = mlir::makeCanonicalStridedLayoutExpr(memRefType.getShape(), context);
            memRefMaps.push_back(mlir::AffineMap::get(memRefType.getRank(), 0, stridedLayout));
        }
        auto accessMapComposition = ComposeAffineMapSequence(memRefMaps);
        assert(accessMapComposition.getNumResults() == 1);
        return accessMapComposition;
    }

    template <typename AffineMemoryOp>
    mlir::AffineMap GetAffineOpIndexToMemoryLocationMap(mlir::MLIRContext* context, AffineMemoryOp op)
    {
        auto composedMemRefMap = GetMemRefIndexToMemoryLocationMap(context, op);
        mlir::AffineMap affineOpMap = op.getAffineMapAttr().getValue();
        mlir::AffineMap accessMapComposition = composedMemRefMap.compose(affineOpMap);
        assert(accessMapComposition.getNumResults() == 1);
        return accessMapComposition;
    }

    mlir::AffineMap GetIndexToMemoryLocationMap(mlir::MLIRContext* context, mlir::AffineStoreOp op)
    {
        return GetAffineOpIndexToMemoryLocationMap(context, op);
    }

    mlir::AffineMap GetIndexToMemoryLocationMap(mlir::MLIRContext* context, mlir::AffineLoadOp op)
    {
        return GetAffineOpIndexToMemoryLocationMap(context, op);
    }

    mlir::AffineMap GetIndexToMemoryLocationMap(mlir::MLIRContext* context, mlir::memref::StoreOp op)
    {
        return GetMemRefIndexToMemoryLocationMap(context, op);
    }

    mlir::AffineMap GetIndexToMemoryLocationMap(mlir::MLIRContext* context, mlir::memref::LoadOp op)
    {
        return GetMemRefIndexToMemoryLocationMap(context, op);
    }

    TempOpCleanupGuard::TempOpCleanupGuard(std::stack<mlir::Operation*>* opStack, mlir::PatternRewriter& rewriter) :
        _opStack(opStack),
        _rewriter(rewriter)
    {}

    TempOpCleanupGuard::~TempOpCleanupGuard()
    {
        while (!_opStack->empty())
        {
            auto eraseOp = _opStack->top();
            assert(eraseOp->use_empty());
            _rewriter.eraseOp(eraseOp);
            _opStack->pop();
        }
    }

    mlir::Attribute MemorySpaceToAttribute(const value::MemorySpace& memorySpace, mlir::MLIRContext* context)
    {
        return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), static_cast<int64_t>(memorySpace));
    }

    value::MemorySpace AttributeToMemorySpace(mlir::Attribute memorySpaceAttr)
    {
        return static_cast<value::MemorySpace>(memorySpaceAttr.cast<mlir::IntegerAttr>().getInt());
    }

    mlir::AffineMap GetMajorIdentityMap(unsigned dims, unsigned results, mlir::MLIRContext* context)
    {
        assert(dims >= results && "Dimension mismatch");
        auto id = mlir::AffineMap::getMultiDimIdentityMap(dims, context);
        return mlir::AffineMap::get(dims, 0, id.getResults().take_front(results), context);
    }

    void EraseAllOpsInBlock(mlir::PatternRewriter& rewriter, mlir::Block& block)
    {
        for (auto& op : llvm::make_early_inc_range(llvm::reverse(block)))
        {
            assert(op.use_empty() && "expected 'op' to have no uses");
            rewriter.eraseOp(&op);
        }
    }

    mlir::Type ToSignlessMLIRType(mlir::OpBuilder& builder, mlir::Type type)
    {
        if (type.isIntOrFloat())
        {
            if (auto width = type.getIntOrFloatBitWidth(); type.isInteger(width))
            {
                return builder.getIntegerType(width);
            }
        }
        return type; // pass-through, no signless change
    }

    mlir::Value ToSignlessMLIRValue(mlir::OpBuilder& builder, mlir::Value value)
    {
        auto type = value.getType();
        if (auto signlessType = ToSignlessMLIRType(builder, type); signlessType != type)
        {
            // Cast from signed to signless
            // cf. mlir/lib/Conversion/TosaToLinalg/TosaToLinalg.cpp
            return builder.create<mlir::UnrealizedConversionCastOp>(value.getLoc(), signlessType, value).getResult(0);
        }
        return value; // pass-through, no signless change
    }

    mlir::Operation* GetDefiningOpOrForLoop(mlir::Value val)
    {
        if (mlir::isForInductionVar(val)) // AffineForOp
        {
            return mlir::getForInductionVarOwner(val);
        }
        else if (auto scfForOp = mlir::scf::getForInductionVarOwner(val)) // SCFForOp
        {
            return scfForOp;
        }
        else // Arbitrary other op
        {
            return val.getDefiningOp();
        }
    }

    template <typename _TyOp>
    auto GetROCDLGPUIndex(mlir::OpBuilder& builder, mlir::Location& loc)
    {
        return builder.create<mlir::IndexCastOp>(loc, builder.create<_TyOp>(loc, builder.getI32Type()), builder.getIndexType());
    }

    mlir::Value GetGPUIndex(const ir::value::ExecutionRuntime& runtime, value::Processor idxType, mlir::OpBuilder& builder, mlir::Location& loc)
    {
        if (runtime == value::ExecutionRuntime::ROCM)
        {
            switch (idxType)
            {
            case value::Processor::ThreadX:
                return util::GetROCDLGPUIndex<mlir::ROCDL::ThreadIdXOp>(builder, loc);
            case value::Processor::ThreadY:
                return util::GetROCDLGPUIndex<mlir::ROCDL::ThreadIdYOp>(builder, loc);
            case value::Processor::ThreadZ:
                return util::GetROCDLGPUIndex<mlir::ROCDL::ThreadIdZOp>(builder, loc);
            case value::Processor::BlockX:
                return util::GetROCDLGPUIndex<mlir::ROCDL::BlockIdXOp>(builder, loc);
            case value::Processor::BlockY:
                return util::GetROCDLGPUIndex<mlir::ROCDL::BlockIdYOp>(builder, loc);
            case value::Processor::BlockZ:
                return util::GetROCDLGPUIndex<mlir::ROCDL::BlockIdZOp>(builder, loc);
            case value::Processor::BlockDimX:
                return util::GetROCDLGPUIndex<mlir::ROCDL::BlockDimXOp>(builder, loc);
            case value::Processor::BlockDimY:
                return util::GetROCDLGPUIndex<mlir::ROCDL::BlockDimYOp>(builder, loc);
            case value::Processor::BlockDimZ:
                return util::GetROCDLGPUIndex<mlir::ROCDL::BlockDimZOp>(builder, loc);
            case value::Processor::GridDimX:
                return util::GetROCDLGPUIndex<mlir::ROCDL::GridDimXOp>(builder, loc);
            case value::Processor::GridDimY:
                return util::GetROCDLGPUIndex<mlir::ROCDL::GridDimYOp>(builder, loc);
            case value::Processor::GridDimZ:
                return util::GetROCDLGPUIndex<mlir::ROCDL::GridDimZOp>(builder, loc);
            case value::Processor::Sequential:
                [[fallthrough]];
            default:
                llvm_unreachable("Unexpected");
            }
        }
        else
        {
            switch (idxType)
            {
            case value::Processor::ThreadX:
                return builder.create<mlir::gpu::ThreadIdOp>(loc, builder.getIndexType(), "x");
            case value::Processor::ThreadY:
                return builder.create<mlir::gpu::ThreadIdOp>(loc, builder.getIndexType(), "y");
            case value::Processor::ThreadZ:
                return builder.create<mlir::gpu::ThreadIdOp>(loc, builder.getIndexType(), "z");
            case value::Processor::BlockX:
                return builder.create<mlir::gpu::BlockIdOp>(loc, builder.getIndexType(), "x");
            case value::Processor::BlockY:
                return builder.create<mlir::gpu::BlockIdOp>(loc, builder.getIndexType(), "y");
            case value::Processor::BlockZ:
                return builder.create<mlir::gpu::BlockIdOp>(loc, builder.getIndexType(), "z");
            case value::Processor::BlockDimX:
                return builder.create<mlir::gpu::BlockDimOp>(loc, builder.getIndexType(), "x");
            case value::Processor::BlockDimY:
                return builder.create<mlir::gpu::BlockDimOp>(loc, builder.getIndexType(), "y");
            case value::Processor::BlockDimZ:
                return builder.create<mlir::gpu::BlockDimOp>(loc, builder.getIndexType(), "z");
            case value::Processor::GridDimX:
                return builder.create<mlir::gpu::GridDimOp>(loc, builder.getIndexType(), "x");
            case value::Processor::GridDimY:
                return builder.create<mlir::gpu::GridDimOp>(loc, builder.getIndexType(), "y");
            case value::Processor::GridDimZ:
                return builder.create<mlir::gpu::GridDimOp>(loc, builder.getIndexType(), "z");
            case value::Processor::Sequential:
                [[fallthrough]];
            default:
                llvm_unreachable("Unexpected");
            }
        }
    }

    mlir::Value GetGPUIndex(mlir::Operation* op, const value::Processor idxType, mlir::OpBuilder& builder, mlir::Location& loc)
    {
        const auto runtime = ResolveExecutionRuntime(op).value();
        return GetGPUIndex(runtime, idxType, builder, loc);
    }

    value::Processor GetGPUProcessor(mlir::Operation* gpuOp)
    {
        assert(gpuOp != nullptr && "Can't get GPU Proc for null op");
        return mlir::TypeSwitch<mlir::Operation*, value::Processor>(gpuOp)
            .Case([&](mlir::gpu::ThreadIdOp threadIdOp) {
                auto threadStr = threadIdOp.dimension().str();
                if (threadStr == "x")
                {
                    return value::Processor::ThreadX;
                }
                else if (threadStr == "y")
                {
                    return value::Processor::ThreadY;
                }
                else if (threadStr == "z")
                {
                    return value::Processor::ThreadZ;
                }
                else
                {
                    assert(false && "Unrecognized thread dimension");
                    return value::Processor::Sequential;
                }
            })
            .Case([&](mlir::gpu::BlockIdOp blockIdOp) {
                auto blockStr = blockIdOp.dimension().str();
                if (blockStr == "x")
                {
                    return value::Processor::BlockX;
                }
                else if (blockStr == "y")
                {
                    return value::Processor::BlockY;
                }
                else if (blockStr == "z")
                {
                    return value::Processor::BlockZ;
                }
                else
                {
                    assert(false && "Unrecognized block dimension");
                    return value::Processor::Sequential;
                }
            })
            .Case([&](mlir::gpu::BlockDimOp blockDimOp) {
                auto blockStr = blockDimOp.dimension().str();
                if (blockStr == "x")
                {
                    return value::Processor::BlockDimX;
                }
                else if (blockStr == "y")
                {
                    return value::Processor::BlockDimY;
                }
                else if (blockStr == "z")
                {
                    return value::Processor::BlockDimZ;
                }
                else
                {
                    assert(false && "Unrecognized block dimension");
                    return value::Processor::Sequential;
                }
            })
            .Case([&](mlir::gpu::GridDimOp gridDimOp) {
                auto gridStr = gridDimOp.dimension().str();
                if (gridStr == "x")
                {
                    return value::Processor::GridDimX;
                }
                else if (gridStr == "y")
                {
                    return value::Processor::GridDimY;
                }
                else if (gridStr == "z")
                {
                    return value::Processor::GridDimZ;
                }
                else
                {
                    assert(false && "Unrecognized grid dimension");
                    return value::Processor::Sequential;
                }
            })
            .Case([&](mlir::ROCDL::ThreadIdXOp) {
                return value::Processor::ThreadX;
            })
            .Case([&](mlir::ROCDL::ThreadIdYOp) {
                return value::Processor::ThreadY;
            })
            .Case([&](mlir::ROCDL::ThreadIdZOp) {
                return value::Processor::ThreadZ;
            })
            .Case([&](mlir::ROCDL::BlockIdXOp) {
                return value::Processor::BlockX;
            })
            .Case([&](mlir::ROCDL::BlockIdYOp) {
                return value::Processor::BlockY;
            })
            .Case([&](mlir::ROCDL::BlockIdZOp) {
                return value::Processor::BlockZ;
            })
            .Case([&](mlir::ROCDL::BlockDimXOp) {
                return value::Processor::BlockDimX;
            })
            .Case([&](mlir::ROCDL::BlockDimYOp) {
                return value::Processor::BlockDimY;
            })
            .Case([&](mlir::ROCDL::BlockDimZOp) {
                return value::Processor::BlockDimZ;
            })
            .Case([&](mlir::ROCDL::GridDimXOp) {
                return value::Processor::GridDimX;
            })
            .Case([&](mlir::ROCDL::GridDimYOp) {
                return value::Processor::GridDimY;
            })
            .Case([&](mlir::ROCDL::GridDimZOp) {
                return value::Processor::GridDimZ;
            })
            .Case([&](mlir::IndexCastOp castOp) {
                // If this is an index cast, recurse to the arg of the index cast
                auto inputVal = castOp.in();
                return GetGPUProcessor(inputVal.getDefiningOp());
            })
            .Default([&](mlir::Operation*) {
                assert(false && "Unsupported GPU op");
                return value::Processor::Sequential;
            });
    }

    int DimIndexToInteger(llvm::StringRef dim)
    {
        return ::llvm::StringSwitch<int>(dim)
            .Case("x", 0)
            .Case("y", 1)
            .Case("z", 2)
            .Default(-1);
    }

    int GetDimValByDimIndexStr(accera::ir::targets::Dim3 dims, llvm::StringRef dimStr)
    {
        return ::llvm::StringSwitch<int>(dimStr)
            .Case("x", dims.x)
            .Case("y", dims.y)
            .Case("z", dims.z)
            .Default(-1);
    }

    template <typename OpTy>
    accera::ir::targets::GPU GetGPUFuncLaunchHelper(OpTy vFuncOrLambdaOp)
    {
        auto launchAttr = vFuncOrLambdaOp->template getAttrOfType<mlir::ArrayAttr>(vFuncOrLambdaOp.getGPULaunchAttrName());
        assert(launchAttr != nullptr);
        return accera::ir::targets::GPU::FromArrayAttr(launchAttr);
    }

    accera::ir::targets::GPU GetGPUFuncLaunchInfo(mlir::Operation* where)
    {
        return mlir::TypeSwitch<mlir::Operation*, accera::ir::targets::GPU>(where)
            .Case([&](ir::value::ValueFuncOp vFuncOp) { return GetGPUFuncLaunchHelper(vFuncOp); })
            .Case([&](ir::value::ValueLambdaOp vLambdaOp) { return GetGPUFuncLaunchHelper(vLambdaOp); })
            .Default([](mlir::Operation*) {
                assert(false && "Can only resolve gpu launch info for ir::value::ValueFuncOp and ir::value::ValueLambdaOp");
                return accera::ir::targets::GPU{};
            });
    }

    int64_t GetBlockDimSize(mlir::Operation* where, const std::string& dimId)
    {
        if (auto gpuFunc = where->getParentOfType<mlir::gpu::GPUFuncOp>())
        {
            auto blockIdxAttr = gpuFunc->getAttrOfType<ArrayAttr>("blockSize");
            auto blockDimIdx = DimIndexToInteger(dimId);
            assert((blockIdxAttr && blockDimIdx != -1) && "Couldn't resolve block size");
            auto blockDimSize = blockIdxAttr.getValue()[blockDimIdx].cast<IntegerAttr>().getInt();
            return blockDimSize;
        }
        else
        {
            mlir::Operation* vFuncOp = where->getParentOfType<ir::value::ValueFuncOp>();
            mlir::Operation* vLambdaOp = where->getParentOfType<ir::value::ValueLambdaOp>();
            if (vFuncOp == nullptr && vLambdaOp == nullptr)
            {
                assert(false && "Can only resolve block dim size inside of a gpu::GPUFuncOp, ir::value::ValueFuncOp, or ir::value::ValueLambdaOp");
                return -1;
            }
            // Prefer using the ValueLambdaOp as inner loopnests will be a ValueLambdaOp nested inside of a ValueFuncOp
            auto op = vLambdaOp != nullptr ? vLambdaOp : vFuncOp;
            auto gpuParams = GetGPUFuncLaunchInfo(op);
            auto blockDimVal = GetDimValByDimIndexStr(gpuParams.block, dimId);
            assert(blockDimVal != -1 && "Couldn't resolve block size");
            return blockDimVal;
        }
    }

    int64_t GetGridDimSize(mlir::Operation* where, const std::string& dimId)
    {
        if (auto gpuFunc = where->getParentOfType<mlir::gpu::GPUFuncOp>())
        {
            auto gridIdxAttr = gpuFunc->getAttrOfType<ArrayAttr>("gridSize");
            auto gridDimIdx = DimIndexToInteger(dimId);
            assert((gridIdxAttr && gridDimIdx != -1) && "Couldn't resolve grid size");
            auto gridDimSize = gridIdxAttr.getValue()[gridDimIdx].cast<IntegerAttr>().getInt();
            return gridDimSize;
        }
        else
        {
            mlir::Operation* vFuncOp = where->getParentOfType<ir::value::ValueFuncOp>();
            mlir::Operation* vLambdaOp = where->getParentOfType<ir::value::ValueLambdaOp>();
            if (vFuncOp == nullptr && vLambdaOp == nullptr)
            {
                assert(false && "Can only resolve grid dim size inside of a gpu::GPUFuncOp, ir::value::ValueFuncOp, or ir::value::ValueLambdaOp");
                return -1;
            }
            auto op = vLambdaOp != nullptr ? vLambdaOp : vFuncOp;
            auto gpuParams = GetGPUFuncLaunchInfo(op);
            auto gridDimVal = GetDimValByDimIndexStr(gpuParams.grid, dimId);
            assert(gridDimVal != -1 && "Couldn't resolve grid size");
            return gridDimVal;
        }
    }

    int64_t GetBlockDimSize(mlir::gpu::BlockDimOp op)
    {
        return GetBlockDimSize(op, op.dimension().str());
    }

    int64_t GetGridDimSize(mlir::gpu::GridDimOp op)
    {
        return GetGridDimSize(op, op.dimension().str());
    }

} // namespace util
} // namespace accera::ir
