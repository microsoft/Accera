////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRUtil.h"
#include "value/ValueAttributes.h"
#include "value/ValueEnums.h"

#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueDialect.h>

#include <utilities/include/StringUtil.h>

#include <mlir/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
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
            auto abstractOp = childOp->getAbstractOperation();
            if (s.count(abstractOp) == 0)
            {
                abstractOp->getCanonicalizationPatterns(patterns, context);
                s.insert(abstractOp);
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

    mlir::Value CreateGlobalBuffer(mlir::OpBuilder& builder, mlir::MemRefType bufferType, const std::string& namePrefix)
    {
        auto insertionBlock = builder.getInsertionBlock();
        auto parentOp = insertionBlock->getParentOp();
        return CreateGlobalBuffer(builder, parentOp, bufferType, namePrefix);
    }

    ir::value::GlobalOp CreateGlobalBufferOp(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, const std::string& namePrefix)
    {
        int64_t counterVal = GetUniqueId();
        auto loc = anchorOp->getLoc();
        std::string globalName = namePrefix + "_" + std::to_string(counterVal);
        mlir::OpBuilder::InsertionGuard guard(builder);

        auto module = util::CastOrGetParentOfType<ir::value::ValueModuleOp>(anchorOp);
        assert(module && "Expected to be inside a ValueModuleOp");
        auto body = module.getBody();

        // Lock before accessing the global scope so that multi-threaded lowerings all access the appropriate global insert position
        std::lock_guard<std::mutex> lock(_globalInsertMutex);
        builder.setInsertionPoint(body, body->begin());
        return builder.create<accera::ir::value::GlobalOp>(loc, bufferType, /* isConstant= */ false, globalName, Attribute{});
    }

    mlir::Value CreateGlobalBuffer(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, const std::string& namePrefix)
    {
        auto globalOp = CreateGlobalBufferOp(builder, anchorOp, bufferType, namePrefix);

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
        auto getExecTarget = [](Operation* op) { return op->getAttrOfType<vir::ExecutionTargetAttr>(vir::ValueFuncOp::getExecTargetAttrName()); };

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
            .Case([](mlir::gpu::GPUFuncOp op) {
                return vir::ExecutionTarget::GPU;
            })
            .Case([](mlir::spirv::FuncOp op) {
                return vir::ExecutionTarget::GPU;
            })
            .Case([](mlir::FuncOp op) {
                return vir::ExecutionTarget::CPU;
            })
            .Case([](mlir::LLVM::LLVMFuncOp op) {
                return vir::ExecutionTarget::CPU;
            })
            .Default([](Operation* op) {
                op->emitWarning("Couldn't determine execution environment");
                return std::nullopt;
            });
    }

    std::optional<vir::ExecutionRuntime> ResolveExecutionRuntime(mlir::Operation* op, bool exact /* = false */)
    {
        auto execRuntimeAttrName = ir::value::ValueModuleOp::getExecRuntimeAttrName();

        auto getExecRuntime = [&](Operation* op) {
            return op->getAttrOfType<vir::ExecutionRuntimeAttr>(execRuntimeAttrName);
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

    std::optional<std::pair<int, int>> ResolveWarpSize(mlir::Operation* op)
    {
        auto runtime = ResolveExecutionRuntime(op);
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

    std::vector<mlir::Value> GetCurrentIndexIVs(const std::vector<loopnest::Index>& loopIndices, mlir::Operation* where, const std::vector<std::pair<loopnest::Index, mlir::Value>>& unrealizedLoopNestIndices)
    {
        return GetCurrentIndexIVs(loopIndices, where->getBlock(), unrealizedLoopNestIndices);
    }

    std::vector<mlir::Value> GetCurrentIndexIVs(const std::vector<loopnest::Index>& loopIndices, mlir::Block* where, const std::vector<std::pair<loopnest::Index, mlir::Value>>& unrealizedLoopNestIndices)
    {
        std::vector<mlir::Value> ivs(loopIndices.size());

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

        auto blockParentOp = where->getParentOp();
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

        for (auto iv : ivs)
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
        assert(symbolOp && "Op with given symbol name not found");
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
} // namespace util
} // namespace accera::ir
