////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <algorithm>
#include <iterator>
#include <optional>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <type_traits>
#include <vector>

#include "nest/LoopNestAttributes.h"
#include "value/ValueDialect.h"
#include "value/ValueEnums.h"

#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>

#ifndef RC_FILE_LOC
#define RC_FILE_LOC(rewriter) accera::ir::util::GetLocation(rewriter, __FILE__, __LINE__)
#endif // RC_FILE_LOC

namespace mlir
{
class AffineForOp;
class AffineLoadOp;
class AffineStoreOp;
class AffineApplyOp;
class AffineValueMap;
class OpBuilder;

namespace memref
{
    class LoadOp;
    class StoreOp;
} // namespace memref
} // namespace mlir

namespace accera::ir
{
namespace util
{
    void CanonicalizeGreedily(mlir::Operation* op);
    void FillCanonicalPatternsRecursively(mlir::Operation* op, mlir::OwningRewritePatternList& patterns);

    template <typename OpTy, typename... TerminatorOpTys>
    mlir::OpBuilder::InsertPoint GetTerminalInsertPoint(OpTy op)
    {
        assert(op);
        auto modBody = op.getBody();
        // TODO kerha: See if it works (better) with rbegin()
        auto iter = modBody->begin();
        auto end = modBody->end();
        while (iter != end && !llvm::isa<TerminatorOpTys...>(iter))
        {
            ++iter;
        }

        return { modBody, iter };
    }

    template <typename OpTy, typename FirstStartingOpTy, typename... RemainingStartingOpTys>
    mlir::OpBuilder::InsertPoint GetStartingInsertPoint(OpTy op)
    {
        assert(op);
        auto& modBody = op.getBody();
        auto i = modBody.begin();
        while (llvm::isa<FirstStartingOpTy, RemainingStartingOpTys...>(i))
        {
            ++i;
        }

        return { modBody, i };
    }

    template <typename ElementType, typename AttrType>
    std::vector<ElementType> ArrayAttrToVector(const mlir::ArrayAttr& inputArrayAttr, const std::function<ElementType(const AttrType&)>& parsingFn)
    {
        std::vector<ElementType> result;
        result.reserve(inputArrayAttr.size());
        for (size_t attrIdx = 0; attrIdx < inputArrayAttr.size(); ++attrIdx)
        {
            assert(inputArrayAttr[attrIdx].isa<AttrType>());
            auto resultAttr = inputArrayAttr[attrIdx].cast<AttrType>();
            result.push_back(parsingFn(resultAttr));
        }
        return result;
    }

    template <typename ElementType>
    std::vector<ElementType> ArrayAttrToVector(const mlir::ArrayAttr& inputArrayAttr)
    {
        std::vector<ElementType> result;
        result.reserve(inputArrayAttr.size());
        for (size_t attrIdx = 0; attrIdx < inputArrayAttr.size(); ++attrIdx)
        {
            assert(inputArrayAttr[attrIdx].isa<ElementType>());
            auto resultAttr = inputArrayAttr[attrIdx].cast<ElementType>();
            result.push_back(resultAttr);
        }
        return result;
    }

    template <typename ElementType, typename AttrType>
    mlir::ArrayAttr VectorToArrayAttr(const std::vector<ElementType>& inputVec, const std::function<AttrType(const ElementType&)>& conversionFn, mlir::MLIRContext* context)
    {
        std::vector<mlir::Attribute> result;
        result.reserve(inputVec.size());
        std::transform(inputVec.begin(), inputVec.end(), std::back_inserter(result), conversionFn);

        return mlir::ArrayAttr::get(context, result);
    }

    template <typename ElementType>
    mlir::ArrayAttr VectorToArrayAttr(const std::vector<ElementType>& inputVec, mlir::MLIRContext* context)
    {
        std::vector<mlir::Attribute> result;
        result.reserve(inputVec.size());
        std::copy(inputVec.begin(), inputVec.end(), std::back_inserter(result));

        return mlir::ArrayAttr::get(context, result);
    }

    std::vector<int64_t> ConvertArrayAttrToIntVector(const mlir::ArrayAttr& inputArrayAttr);
    std::vector<loopnest::Index> ConvertArrayAttrToIndexVector(const mlir::ArrayAttr& inputArrayAttr);
    mlir::ArrayAttr ConvertIndexVectorToArrayAttr(const std::vector<loopnest::Index>& inputVec, mlir::MLIRContext* context);

    template <typename UserType>
    std::vector<UserType> getUsesOfType(mlir::Value value)
    {
        // Check for immediate uses of this value of the given op type
        std::set<UserType> resultSet;
        std::queue<mlir::OpOperand*> usesToExamine;

        for (auto& use : value.getUses())
        {
            mlir::Operation* useOwner = use.getOwner();
            if (auto useOp = mlir::dyn_cast_or_null<UserType>(useOwner))
            {
                resultSet.insert(useOp);
            }
        }
        return { resultSet.begin(), resultSet.end() };
    }

    template <typename UserType>
    std::vector<UserType> getRecursiveUsesOfType(mlir::Value value)
    {
        // Follow the value uses until we find the type we're looking for or run out of uses
        // Use a std::set for results since the value we're examining may have multiple uses that eventually end up in the same target op
        std::set<UserType> resultSet;
        std::queue<mlir::OpOperand*> usesToExamine;

        for (auto& initialUse : value.getUses())
        {
            usesToExamine.push(&initialUse);
        }
        while (!usesToExamine.empty())
        {
            auto currentUse = usesToExamine.front();
            mlir::Operation* useOwner = currentUse->getOwner();
            if (auto useOp = mlir::dyn_cast_or_null<UserType>(useOwner))
            {
                resultSet.insert(useOp);
            }
            else
            {
                for (auto ownerResult : useOwner->getResults())
                {
                    for (auto& resultUse : ownerResult.getUses())
                    {
                        usesToExamine.push(&resultUse);
                    }
                }
            }
            usesToExamine.pop();
        }
        return { resultSet.begin(), resultSet.end() };
    }

    inline bool hasRecursiveUseOfOp(mlir::Value value, mlir::Operation* op)
    {
        // Follow the value uses until we find the operation we're looking for or run out of uses
        // Use a std::set for results since the value we're examining may have multiple uses that eventually end up in the same target op
        std::queue<mlir::OpOperand*> usesToExamine;
        for (auto& initialUse : value.getUses())
        {
            usesToExamine.push(&initialUse);
        }

        while (!usesToExamine.empty())
        {
            auto currentUse = usesToExamine.front();
            mlir::Operation* useOwner = currentUse->getOwner();
            if (useOwner == op)
            {
                return true;
            }
            usesToExamine.pop();

            for (auto ownerResult : useOwner->getResults())
            {
                for (auto& resultUse : ownerResult.getUses())
                {
                    usesToExamine.push(&resultUse);
                }
            }
        }
        return false;
    }

    template <typename OpType>
    OpType CastOrGetParentOfType(mlir::Operation* op)
    {
        assert(op && "op can't be null");
        if (auto currentOp = mlir::dyn_cast<OpType>(op))
        {
            return currentOp;
        }
        else
        {
            return op->getParentOfType<OpType>();
        }
    }

    mlir::Attribute GetOneAttr(mlir::OpBuilder& builder, mlir::Type type);

    mlir::OpBuilder MakeBodyBuilder(mlir::AffineForOp forOp);

    ir::value::GlobalOp CreateGlobalBufferOp(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, std::string globalName, bool constant = false, Attribute attr = {}, bool isExternal = false, bool appendUniqueSuffix = true);

    mlir::Value CreateStackBuffer(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, int64_t alignment);
    mlir::Value CreateGlobalBuffer(mlir::OpBuilder& builder, mlir::MemRefType bufferType, const std::string& namePrefix, bool constant = false, Attribute attr = {}, bool isExternal = false, bool appendUniqueSuffix = true);
    mlir::Value CreateGlobalBuffer(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, const std::string& namePrefix, bool constant = false, Attribute attr = {}, bool isExternal = false, bool appendUniqueSuffix = true);
    mlir::Value CreateSharedBuffer(mlir::OpBuilder& builder, mlir::MemRefType bufferType, const std::string& namePrefix);
    mlir::Value CreateSharedBuffer(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, const std::string& namePrefix);
    mlir::Value CreatePrivateBuffer(mlir::OpBuilder& builder, mlir::MemRefType bufferType, const std::string& namePrefix);
    mlir::Value CreatePrivateBuffer(mlir::OpBuilder& builder, mlir::Operation* anchorOp, mlir::MemRefType bufferType, const std::string& namePrefix);

    mlir::Location GetLocation(mlir::OpBuilder& builder, std::string tag);
    mlir::Location GetLocation(mlir::OpBuilder& builder, std::string tag, mlir::Location opLocation);
    mlir::Location GetLocation(mlir::OpBuilder& builder, std::string filename, int64_t lineNumber);

    std::vector<mlir::Value> MultiDimAffineApply(mlir::OpBuilder& builder, mlir::Location loc, mlir::AffineMap map, std::vector<mlir::Value>& operands, bool simplify = false);
    mlir::AffineMap MakeIdentityAccessMap(mlir::Value val, mlir::MLIRContext* context);

    mlir::AffineValueMap AffineApplyToAffineValueMap(mlir::AffineApplyOp applyOp);

    mlir::Type GetElementType(mlir::Type type);

    int64_t GetUniqueId();

    mlir::Operation* CloneRecursively(mlir::OpBuilder& builder, mlir::Operation* op, mlir::BlockAndValueMapping& mapping);

    std::optional<ir::value::ExecutionTarget> ResolveExecutionTarget(mlir::Operation* op, bool exact = false);
    std::optional<ir::value::ExecutionRuntime> ResolveExecutionRuntime(mlir::Operation* op, bool exact = false);
    std::optional<std::pair<int, int>> ResolveWarpSize(value::ExecutionRuntime runtime);

    mlir::Operation* CreateGPUControlBarrier(mlir::OpBuilder& builder, const std::string scope, std::optional<mlir::Location> loc = std::nullopt);

    std::optional<int64_t> GetDimSizeAt(const loopnest::Index& dimensionIndex, mlir::Operation* where);

    std::vector<mlir::Value> GetCurrentIndexIVs(const std::vector<loopnest::Index>& loopIndices, mlir::Operation* where);

    std::vector<mlir::Value> GetCurrentIndexIVs(const std::vector<loopnest::Index>& loopIndices, mlir::Block* where);

    std::vector<loopnest::Index> GetIndicesForLoopIVs(const std::vector<mlir::Value>& loopIVs);

    mlir::AffineMap ConcatenateAndShiftAffineDimsAndMaps(mlir::OpBuilder& builder, mlir::AffineMap leftMap, mlir::AffineMap rightMap);

    bool IsSubdomainEmpty(mlir::Operation* where);

    void InlineAllRegionOpsBeforeOp(mlir::PatternRewriter& rewriter, mlir::Region& regionToInline, mlir::Operation* op);

    mlir::Operation* FindOpWithSymbolName(const llvm::StringRef& id, mlir::Operation* rootOp);

    mlir::LogicalResult PromoteIfSingleIteration(mlir::PatternRewriter& rewriter, mlir::AffineForOp forOp);

    bool OperationsAreEqual(mlir::Operation* lhs, mlir::Operation* rhs);

    mlir::Value CreateConstantRangeForOpIterationCounter(mlir::OpBuilder& builder, mlir::Location loc, mlir::AffineForOp forOp);

    mlir::Operation* GetFirstOp(mlir::Operation* left, mlir::Operation* right);

    template <typename OpType>
    bool IsOutermostOpOfType(OpType op, std::optional<std::string> ignoreAttrName = std::nullopt)
    {
        // First check if there is an OpType instance in an ancestor block
        auto currentParentOp = op->getParentOp();
        auto currentBlock = currentParentOp->getBlock();
        while (currentParentOp != nullptr && currentBlock != nullptr)
        {
            for (auto& blockOp : currentBlock->getOperations())
            {
                if (auto outerOp = mlir::dyn_cast_or_null<OpType>(&blockOp))
                {
                    if (ignoreAttrName.has_value())
                    {
                        if (!outerOp->hasAttr(ignoreAttrName.value()))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        return false;
                    }
                }
            }
            currentParentOp = currentBlock->getParentOp();
            if (currentParentOp)
            {
                currentBlock = currentParentOp->getBlock();
            }
        }

        // Now check if there are any OpType siblings to the current one that occur before it in the block
        auto parentBlock = op->getBlock();
        for (auto& siblingOp : parentBlock->getOperations())
        {
            if (&siblingOp == op.getOperation())
            {
                break;
            }
            else if (auto outerOp = mlir::dyn_cast_or_null<OpType>(&siblingOp))
            {
                if (ignoreAttrName.has_value())
                {
                    if (!outerOp->hasAttr(ignoreAttrName.value()))
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }
        }
        return true;
    }

    mlir::AffineMap GetIndexToMemoryLocationMap(mlir::MLIRContext* context, mlir::AffineStoreOp op);
    mlir::AffineMap GetIndexToMemoryLocationMap(mlir::MLIRContext* context, mlir::AffineLoadOp op);
    mlir::AffineMap GetIndexToMemoryLocationMap(mlir::MLIRContext* context, mlir::memref::StoreOp op);
    mlir::AffineMap GetIndexToMemoryLocationMap(mlir::MLIRContext* context, mlir::memref::LoadOp op);

    mlir::AffineMap ComposeAffineMapSequence(const std::vector<mlir::AffineMap>& maps);

    struct TempOpCleanupGuard
    {
        TempOpCleanupGuard(std::stack<mlir::Operation*>* opStack, mlir::PatternRewriter& rewriter);
        ~TempOpCleanupGuard();
        std::stack<mlir::Operation*>* _opStack;
        mlir::PatternRewriter& _rewriter;
    };

    mlir::Attribute MemorySpaceToAttribute(const value::MemorySpace& memorySpace, mlir::MLIRContext* context);
    value::MemorySpace AttributeToMemorySpace(mlir::Attribute memorySpaceAttr);

    // Similar to mlir::AffineMap::getMinorIdentityMap(), but instead this creates the mapping
    // (d0, ..., dn) -> (d0, ... , dk) where n = (num dims - 1) and k = (num results - 1)
    mlir::AffineMap GetMajorIdentityMap(unsigned dims, unsigned results, mlir::MLIRContext* context);

    template <typename OpType>
    mlir::Operation* GetHighestAncestorOfType(mlir::Operation* op)
    {
        mlir::Operation* current = nullptr;
        mlir::Operation* parent = op->getParentOfType<OpType>();
        while (parent != nullptr)
        {
            current = parent;
            parent = current->getParentOfType<OpType>();
        }
        return current;
    }

    void EraseAllOpsInBlock(mlir::PatternRewriter& rewriter, mlir::Block& block);

    mlir::Value ToSignlessMLIRValue(mlir::OpBuilder& builder, mlir::Value value);
    mlir::Type ToSignlessMLIRType(mlir::OpBuilder& builder, mlir::Type type);

    // Returns:
    // a) The AffineForOp if the given value is an AffineForOp induction variable
    // b) The SCFForOp if the given value is an SCFForOp induction variable
    // c) The given value's defining op if it is an op result
    // d) nullptr otherwise (e.g. if it is a block argument for a non-for-loop op)
    mlir::Operation* GetDefiningOpOrForLoop(mlir::Value val);

    mlir::Value GetGPUIndex(value::Processor idxType, mlir::OpBuilder& builder, mlir::Location& loc);

    value::Processor GetGPUProcessor(mlir::Operation* gpuOp);

    int64_t GetBlockDimSize(mlir::gpu::BlockDimOp op);
    int64_t GetGridDimSize(mlir::gpu::GridDimOp op);

    int64_t GetBlockDimSize(mlir::Operation* where, const std::string& dimId);
    int64_t GetGridDimSize(mlir::Operation* where, const std::string& dimId);

    // Gets the flattened thread ID of the current GPU thread within the context of the current block
    mlir::Value GetCurrentGPUBlockThreadID(mlir::OpBuilder& builder, mlir::Location loc);

    // Gets the flattened thread ID of the current GPU thread within the context of the current warp
    mlir::Value GetCurrentGPUWarpThreadID(mlir::OpBuilder& builder, mlir::Location loc);

} // namespace util
} // namespace accera::ir
