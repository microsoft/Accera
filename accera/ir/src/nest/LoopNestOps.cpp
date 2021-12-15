////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRUtil.h"

#include "nest/LoopNestAttributes.h"
#include "nest/LoopNestOps.h"
#include "nest/LoopNestTypes.h"
#include "nest/Util.h"
#include <ir/include/value/ValueDialect.h>
#include <utilities/include/MathUtil.h>

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/RegionUtils.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Support/raw_os_ostream.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>

using namespace accera::ir;
using namespace loopnest;
using namespace mlir;
using namespace accera::utilities;

namespace accera::ir
{
namespace loopnest
{
    // utils
    namespace
    {
        std::string GetIndexName(size_t level)
        {
            static const std::vector<std::string> names = { "i", "j", "k", "l" };
            if (level < names.size())
            {
                return names[level];
            }

            return "idx_" + std::to_string(level);
        }

        ArrayAttr AddOrRemoveIndexId(ArrayAttr& attr, Index::Id id, bool add)
        {
            OpBuilder builder(attr.getContext());
            auto indexType = builder.getIndexType();
            auto idAttr = IntegerAttr::get(indexType, id);

            auto arr = std::vector<Attribute>(attr.getValue());
            auto it = std::find(arr.begin(), arr.end(), idAttr);
            if (add)
            {
                // add if not present
                if (it == arr.end())
                {
                    arr.push_back(idAttr);
                }
            }
            else
            {
                // remove if present
                if (it != arr.end())
                {
                    arr.erase(it);
                }
            }

            return ArrayAttr::get(attr.getContext(), arr);
        }

        bool IsIndexIdPresent(ArrayAttr attr, Index::Id id)
        {
            if (!attr)
                return false;
            OpBuilder builder(attr.getContext());
            auto indexType = builder.getIndexType();
            auto idAttr = IntegerAttr::get(indexType, id);

            auto arr = std::vector<Attribute>(attr.getValue());
            return std::find(arr.begin(), arr.end(), idAttr) != arr.end();
        }

        std::vector<mlir::StringRef> GetKernelIdsInFunction(ir::value::ValueFuncOp func)
        {
            if (!func)
            {
                return {};
            }

            std::vector<mlir::StringRef> ids;
            if (auto region = &func.getBody())
            {
                region->walk([&](Operation* op) {
                    if (auto kernelOp = dyn_cast<KernelOp>(op))
                    {
                        ids.push_back(kernelOp.getId());
                    }
                    else if (auto kernelOp = dyn_cast<ScheduledKernelOp>(op))
                    {
                        ids.push_back(kernelOp.getId());
                    }
                });
            }

            return ids;
        }

        SymbolicIndexOp getSymbolicIndexForOperation(Operation* op, Index index)
        {
            OpBuilder builder(op->getContext());

            // go to start of function
            builder.setInsertionPointToStart(op->getBlock());

            return builder.create<SymbolicIndexOp>(op->getLoc(), index);
        }

        struct LoopNestOpAsmInterface : public OpAsmDialectInterface
        {
            using OpAsmDialectInterface::OpAsmDialectInterface;

            LogicalResult getAlias(Attribute attr, raw_ostream& os) const override
            {
                if (attr.isa<IterationDomainAttr>())
                {
                    os << "domain";
                    return success();
                }
                if (attr.isa<TransformedDomainAttr>())
                {
                    os << "xdomain";
                    return success();
                }
                return failure();
            }
        };

        struct RemoveUnusedKernelOpPattern : public OpRewritePattern<KernelOp>
        {
            using OpRewritePattern<KernelOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(KernelOp op, PatternRewriter& rewriter) const final
            {
                std::string opName = op.getOperation()->getName().getStringRef().str();
                auto loc = util::GetLocation(rewriter, "RemoveUnused_" + opName, op.getLoc());

                // Walk the region looking for schedules or nests that use this kernel
                auto region = op.getOperation()->getParentRegion();
                auto kernelId = op.getId().str();

                bool found = false;
                region->walk([&](Operation* op) {
                    if (auto nestOp = dyn_cast<NestOp>(op))
                    {
                        auto kernels = nestOp.getKernelIds();
                        if (std::find(kernels.begin(), kernels.end(), kernelId) != kernels.end())
                        {
                            found = true;
                            return WalkResult::interrupt();
                        }
                    }
                    else if (auto scheduleOp = dyn_cast<ScheduleOp>(op))
                    {
                        auto kernels = scheduleOp.getKernelIds();
                        if (std::find(kernels.begin(), kernels.end(), kernelId) != kernels.end())
                        {
                            found = true;
                            return WalkResult::interrupt();
                        }

                        // TODO: merge this with the above kernelIds
                        if (auto nestKernelsAttr = scheduleOp->getAttrOfType<ArrayAttr>("nest_kernels"))
                        {
                            found = llvm::any_of(nestKernelsAttr, [&](Attribute attr) {
                                return attr.cast<FlatSymbolRefAttr>().getValue() == kernelId;
                            });

                            if (found)
                            {
                                return WalkResult::interrupt();
                            }
                        }
                    }
                    else if (auto scheduledKernelOp = dyn_cast<ScheduledKernelOp>(op))
                    {
                        if (scheduledKernelOp.getKernel() == kernelId)
                        {
                            found = true;
                            return WalkResult::interrupt();
                        }
                    }
                    return WalkResult::advance();
                });

                if (!found)
                {
                    rewriter.eraseOp(op);
                }

                return success();
            }
        };

        struct RemoveUnusedScheduledKernelOpPattern : public OpRewritePattern<ScheduledKernelOp>
        {
            using OpRewritePattern<ScheduledKernelOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(ScheduledKernelOp op, PatternRewriter& rewriter) const final
            {
                std::string opName = op.getOperation()->getName().getStringRef().str();
                auto loc = util::GetLocation(rewriter, "RemoveUnused_" + opName, op.getLoc());

                // Walk the region looking for schedules or nests that use this kernel
                auto region = op.getOperation()->getParentRegion();
                auto kernelId = op.getId().str();

                bool found = false;
                region->walk([&](Operation* op) {
                    if (auto nestOp = dyn_cast<NestOp>(op))
                    {
                        auto kernels = nestOp.getKernelIds();
                        if (std::find(kernels.begin(), kernels.end(), kernelId) != kernels.end())
                        {
                            found = true;
                            return WalkResult::interrupt();
                        }
                    }
                    else if (auto scheduleOp = dyn_cast<ScheduleOp>(op))
                    {
                        auto kernels = scheduleOp.getKernelIds();
                        if (std::find(kernels.begin(), kernels.end(), kernelId) != kernels.end())
                        {
                            found = true;
                            return WalkResult::interrupt();
                        }
                    }
                    else if (auto scheduledKernelOp = dyn_cast<ScheduledKernelOp>(op))
                    {
                        if (scheduledKernelOp.getKernel() == kernelId)
                        {
                            found = true;
                            return WalkResult::interrupt();
                        }
                    }
                    return WalkResult::advance();
                });

                if (!found)
                {
                    rewriter.eraseOp(op);
                }

                return success();
            }
        };

        void populateKernelCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context)
        {
            patterns.insert<RemoveUnusedKernelOpPattern>(context);
        }

        void populateScheduledKernelCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context)
        {
            patterns.insert<RemoveUnusedScheduledKernelOpPattern>(context);
        }

    } // namespace

    //
    // LoopNestDialect
    //
    void LoopNestDialect::initialize()
    {
        addOperations<
#define GET_OP_LIST
#include "nest/LoopNestOps.cpp.inc"
            >();

        addTypes<ArrayType, KernelType, SymbolicIndexType>();
        addAttributes<IndexAttr, IndexRangeAttr, IterationDomainAttr, RangeAttr, SplitIndexAttr, TransformedDomainAttr>();
        addInterfaces<LoopNestOpAsmInterface>();
    }

    //
    // ScheduleOp
    //

    void ScheduleOp::build(OpBuilder& builder, OperationState& result, NestOp nest)
    {
        ensureTerminator(*result.addRegion(), builder, result.location);

        // Add attributes
        std::vector<Attribute> order;
        auto dims = nest.getDomain().getValue().GetDimensions();
        for (auto i : dims)
        {
            order.push_back(IndexAttr::get(i, builder.getContext()));
        }
        result.addAttribute(getOrderAttrName(), builder.getArrayAttr(order));

        auto domain = nest.getDomain().getValue();
        auto domainAttr = TransformedDomainAttr::get(domain, builder.getContext());
        result.addAttribute(getDomainAttrName(), domainAttr);

        // Copy nest's kernels
        result.addAttribute(getKernelsAttrName(), nest.kernels());

        // Initialize other attributes
        result.addAttribute(getUnrolledIndicesAttrName(), builder.getIndexArrayAttr({}));
        result.addAttribute(getUnrollAndJammedIndicesAttrName(), builder.getDictionaryAttr({}));
        result.addAttribute(getSaturatedFlagIndicesAttrName(), builder.getArrayAttr({}));
        result.addAttribute(getLoopAttrsName(), builder.getArrayAttr({}));
        result.addAttribute(getFusedDomainsAttrName(), builder.getArrayAttr({}));

        // Copy Range operands
        result.addOperands(nest.rangeOperands());
    }

    NestOp ScheduleOp::getNest()
    {
        return (*this)->getParentOfType<NestOp>();
    }

    std::vector<SymbolicIndexOp> ScheduleOp::getIndices(OpBuilder& builder)
    {
        std::vector<SymbolicIndexOp> result;
        for (auto d : getOrder())
        {
            result.push_back(getOrCreateSymbolicIndex(builder, d));
        }

        return result;
    }

    ExecPlanOp ScheduleOp::getOrCreateExecPlan()
    {
        if (auto opRange = this->getOps<ExecPlanOp>(); !opRange.empty())
        {
            assert(std::distance(opRange.begin(), opRange.end()) == 1 && "Expected only one ExecPlanOp inside NestOp body");
            return *opRange.begin();
        }
        if (body().empty())
        {
            body().push_back(new Block);
        }
        return getBodyBuilder().create<ExecPlanOp>(getLoc(), ir::value::ExecutionTarget::CPU);
    }

    size_t ScheduleOp::numDimensions()
    {
        return getDomain().getValue().NumDimensions();
    }

    size_t ScheduleOp::numLoops()
    {
        return getDomain().getValue().NumLoopIndices();
    }

    std::vector<Index> ScheduleOp::getOrder()
    {
        OpBuilder builder(getContext());
        auto orderAttr = (*this)->getAttrOfType<ArrayAttr>(getOrderAttrName()).getValue();
        std::vector<Index> result;
        for (auto elemAttr : orderAttr)
        {
            auto indexAttr = elemAttr.cast<IndexAttr>();
            result.push_back(indexAttr.getValue());
        }
        return result;
    }

    void ScheduleOp::setOrder(ArrayRef<Index> order)
    {
        OpBuilder builder(getContext());
        std::vector<Attribute> indexAttrs;
        for (auto i : order)
        {
            indexAttrs.push_back(IndexAttr::get(i, getContext()));
        }
        auto orderAttr = builder.getArrayAttr(indexAttrs);
        (*this)->setAttr(getOrderAttrName(), orderAttr);
    }

    void ScheduleOp::setOrder(ArrayRef<SymbolicIndexOp> order)
    {
        OpBuilder builder(getContext());
        std::vector<Attribute> indexAttrs;
        for (auto i : order)
        {
            indexAttrs.push_back(IndexAttr::get(i.getValue(), getContext()));
        }
        auto orderAttr = builder.getArrayAttr(indexAttrs);
        (*this)->setAttr(getOrderAttrName(), orderAttr);
    }

    TransformedDomainAttr ScheduleOp::getDomain()
    {
        auto domainAttr = (*this)->getAttrOfType<TransformedDomainAttr>(getDomainAttrName());
        return domainAttr;
    }

    void ScheduleOp::setDomain(const TransformedDomain& domain)
    {
        auto domainAttr = TransformedDomainAttr::get(domain, getContext());
        (*this)->setAttr(getDomainAttrName(), domainAttr);
    }

    std::vector<IterationDomain> ScheduleOp::getFusedDomains()
    {
        auto fusedDomainsAttr = (*this)->getAttrOfType<ArrayAttr>(getFusedDomainsAttrName()).getValue();
        std::vector<IterationDomain> result;
        for (auto elemAttr : fusedDomainsAttr)
        {
            result.push_back(elemAttr.cast<IterationDomainAttr>().getValue());
        }
        return result;
    }

    void ScheduleOp::setFusedDomains(ArrayRef<IterationDomain> domains)
    {
        OpBuilder builder(getContext());
        std::vector<Attribute> domainAttrs;
        for (auto domain : domains)
        {
            auto domainAttr = IterationDomainAttr::get(domain, getContext());
            domainAttrs.push_back(domainAttr.cast<Attribute>());
        }
        (*this)->setAttr(getFusedDomainsAttrName(), builder.getArrayAttr(domainAttrs));
    }

    // 1:Many mapping of targetIndex to domain dimensions that were fused into it
    std::vector<Index> ScheduleOp::getFusedIndices(Index targetIndex)
    {
        std::vector<Index> result;
        if (auto dictAttr = getLoopAttributes(targetIndex))
        {
            OpBuilder builder(getContext());
            auto attrName = builder.getIdentifier(getFusedIndicesAttrName());
            if (auto attr = dictAttr->get(attrName))
            {
                auto fusedIndicesAttr = attr.cast<ArrayAttr>().getValue();
                for (auto elemAttr : fusedIndicesAttr)
                {
                    auto indexAttr = elemAttr.cast<IndexAttr>();
                    result.push_back(indexAttr.getValue());
                }
            }
        }
        return result;
    }

    // 1:Many mapping of targetIndex to domain dimensions that were fused into it
    void ScheduleOp::setFusedIndices(Index targetIndex, ArrayRef<Index> fusedIndices)
    {
        OpBuilder builder(getContext());
        std::unordered_set<Index> uniqueIndices(fusedIndices.begin(), fusedIndices.end());

        std::vector<Attribute> indexAttrs;
        for (auto i : uniqueIndices)
        {
            indexAttrs.push_back(IndexAttr::get(i, getContext()));
        }
        auto fusedIndexAttr = builder.getArrayAttr(indexAttrs);
        auto attrName = builder.getIdentifier(getFusedIndicesAttrName());
        addLoopAttribute(targetIndex, attrName, fusedIndexAttr);
    }

    void ScheduleOp::unroll(Index index, bool val, std::optional<uint64_t> size)
    {
        mlir::Builder b(getContext());
        auto indices = (*this)->getAttrOfType<DictionaryAttr>(getUnrolledIndicesAttrName());

        llvm::SmallVector<NamedAttribute, 4> namedAttrs;
        if (indices && !indices.empty())
        {
            namedAttrs = llvm::to_vector<4>(llvm::make_range(indices.begin(), indices.end()));
        }

        auto id = b.getIdentifier(std::to_string(index.GetId()));
        for (auto it = namedAttrs.begin(), e = namedAttrs.end(); it != e; ++it)
        {
            if (it->first == id)
            {
                namedAttrs.erase(it);
                break;
            }
        }

        if (val)
        {
            auto sizeValue = size ? *size : std::numeric_limits<int64_t>::max(); // to be cast to int64_t

            namedAttrs.emplace_back(id, b.getI64IntegerAttr((int64_t)sizeValue));
        }
        auto dictAttr = b.getDictionaryAttr(namedAttrs);

        (*this)->setAttr(getUnrolledIndicesAttrName(), dictAttr);
    }

    std::optional<uint64_t> ScheduleOp::getUnrollIfRangeSmallerThan(Index index)
    {
        if (auto unrolledIndices = (*this)->getAttrOfType<DictionaryAttr>(getUnrolledIndicesAttrName());
            unrolledIndices && !unrolledIndices.empty())
        {
            if (auto attr = unrolledIndices.get(std::to_string(index.GetId())).dyn_cast_or_null<IntegerAttr>())
            {
                return (uint64_t)attr.getInt();
            }
        }

        return std::nullopt; // not unrolled
    }

    bool ScheduleOp::isSaturated(Index index)
    {
        auto unrolledIndices = (*this)->getAttrOfType<ArrayAttr>(getSaturatedFlagIndicesAttrName());
        if (!unrolledIndices)
            return false;
        return IsIndexIdPresent(unrolledIndices, index.GetId());
    }

    void ScheduleOp::setSaturatedFlag(Index index, bool saturated)
    {
        auto indices = (*this)->getAttrOfType<ArrayAttr>(getSaturatedFlagIndicesAttrName());
        auto newIndices = AddOrRemoveIndexId(indices, index.GetId(), saturated);

        (*this)->setAttr(getSaturatedFlagIndicesAttrName(), newIndices);
    }

    void ScheduleOp::unrollAndJam(Index index, uint64_t factor)
    {
        mlir::Builder b(getContext());
        mlir::NamedAttrList indices = (*this)->getAttrOfType<DictionaryAttr>(getUnrollAndJammedIndicesAttrName());

        if (auto id = b.getIdentifier(std::to_string(index.GetId())); factor == 0)
        {
            indices.erase(id);
        }
        else
        {
            indices.set(id, b.getI64IntegerAttr((int64_t)factor));
        }

        (*this)->setAttr(getUnrollAndJammedIndicesAttrName(), indices.getDictionary(getContext()));
    }

    std::optional<uint64_t> ScheduleOp::getUnrollAndJamFactor(Index index)
    {

        if (auto unrolledIndices = getOperation()->getAttrOfType<DictionaryAttr>(getUnrollAndJammedIndicesAttrName());
            !unrolledIndices.empty())
        {
            if (auto attr = unrolledIndices.get(std::to_string(index.GetId())).dyn_cast_or_null<IntegerAttr>())
            {
                return (uint64_t)attr.getInt();
            }
        }

        return std::nullopt;
    }

    SplitIndex ScheduleOp::split(Index index, int splitSize)
    {
        auto domainAttr = getDomain(); // A TransformedDomainAttr
        auto domain = domainAttr.getValue(); // A TransformedDomain
        auto result = domain.Split(index, splitSize, getContext());

        auto loopSequence = getOrder(); // should be a LoopOrderAttr or something, but it's just a vector<Index>
        auto it = std::find(loopSequence.begin(), loopSequence.end(), index);
        assert(it != loopSequence.end());
        *it = result.outer;
        loopSequence.push_back(result.inner);

        // Update attributes
        setDomain(domain);
        setOrder(loopSequence);

        // Create a SymbolicIndexOp
        OpBuilder builder(getContext());
        builder.setInsertionPoint(*this);
        [[maybe_unused]] auto outerIndexOp = getOrCreateSymbolicIndex(builder, result.outer);
        [[maybe_unused]] auto innerIndexOp = getOrCreateSymbolicIndex(builder, result.inner);

        return result;
    }

    SplitSymbolicIndex ScheduleOp::split(SymbolicIndexOp index, int splitSize)
    {
        auto indices = split(index.getValue(), splitSize);
        OpBuilder builder(getContext());
        builder.setInsertionPoint(*this);
        auto outerIndexOp = getOrCreateSymbolicIndex(builder, indices.outer);
        auto innerIndexOp = getOrCreateSymbolicIndex(builder, indices.inner);
        for (auto op : { outerIndexOp, innerIndexOp })
        {
            op->setAttr("reference", index->getAttr("name"));
        }

        return { outerIndexOp, innerIndexOp };
    }

    Index ScheduleOp::pad(Index index, int size)
    {
        OpBuilder builder(getContext());

        auto domainAttr = getDomain(); // A TransformedDomainAttr
        auto domain = domainAttr.getValue(); // A TransformedDomain
        auto paddedIndex = domain.Pad(index, size, getContext());

        // Replace index with the padded index in the loop sequence order
        auto loopSequence = getOrder();
        auto it = std::find(loopSequence.begin(), loopSequence.end(), index);
        assert(it != loopSequence.end());
        *it = paddedIndex;

        // Update attributes
        setDomain(domain);
        setOrder(loopSequence);

        // Create a SymbolicIndexOp for the padded index
        builder.setInsertionPoint(*this);
        [[maybe_unused]] auto paddedIndexOp = getOrCreateSymbolicIndex(builder, paddedIndex);
        return paddedIndex;
    }

    SymbolicIndexOp ScheduleOp::pad(SymbolicIndexOp index, int size)
    {
        auto paddedIndex = pad(index.getValue(), size);
        OpBuilder builder(getContext());
        builder.setInsertionPoint(*this);

        auto paddedIndexOp = getOrCreateSymbolicIndex(builder, paddedIndex);
        paddedIndexOp->setAttr("reference", index->getAttr("name")); // needed?
        return paddedIndexOp;
    }

    Index ScheduleOp::skew(Index index, Index reference)
    {
        OpBuilder builder(getContext());

        auto domainAttr = getDomain(); // A TransformedDomainAttr
        auto domain = domainAttr.getValue(); // A TransformedDomain
        auto skewedIndex = domain.Skew(index, reference, getContext());

        // Replace index with the skewed index in the loop sequence order
        auto loopSequence = getOrder();
        auto it = std::find(loopSequence.begin(), loopSequence.end(), index);
        assert(it != loopSequence.end());
        *it = skewedIndex;

        // Update attributes
        setDomain(domain);
        setOrder(loopSequence);

        // Create a SymbolicIndexOp for the skewed index
        builder.setInsertionPoint(*this);
        [[maybe_unused]] auto skewedIndexOp = getOrCreateSymbolicIndex(builder, skewedIndex);
        return skewedIndex;
    }

    SymbolicIndexOp ScheduleOp::skew(SymbolicIndexOp index, SymbolicIndexOp reference)
    {
        auto skewedIndex = skew(index.getValue(), reference.getValue());
        OpBuilder builder(getContext());
        builder.setInsertionPoint(*this);

        auto skewedIndexOp = getOrCreateSymbolicIndex(builder, skewedIndex);
        skewedIndexOp->setAttr("reference", index->getAttr("name")); // needed?
        return skewedIndexOp;
    }

    SymbolicIndexOp ScheduleOp::getOrCreateSymbolicIndex(OpBuilder& builder, Index index)
    {
        return getSymbolicIndexForOperation(getOperation(), index);
    }

    std::vector<std::string> ScheduleOp::getKernelIds()
    {
        std::vector<std::string> result;
        for (auto k : kernels())
        {
            result.push_back(std::string{ k.cast<FlatSymbolRefAttr>().getValue() });
        }
        return result;
    }

    std::vector<ScheduledKernelOp> ScheduleOp::getKernels()
    {
        std::vector<ScheduledKernelOp> result;

        for (auto id : getKernelIds())
        {
            auto kernelOp = getKernel(id);
            result.push_back(kernelOp);
        }

        return result;
    }

    ScheduledKernelOp ScheduleOp::getKernel(llvm::StringRef id)
    {
        auto symTableOp = mlir::SymbolTable::getNearestSymbolTable(getOperation());
        auto symbolOp = mlir::SymbolTable::lookupNearestSymbolFrom(symTableOp, id);
        auto kernelOp = mlir::cast<ScheduledKernelOp>(symbolOp);
        assert(kernelOp && "Kernel not found");
        return kernelOp;
    }

    void ScheduleOp::addKernel(StringRef kernelId)
    {
        OpBuilder builder(*this);
        auto kernels = std::vector<Attribute>((*this)->getAttrOfType<ArrayAttr>(getKernelsAttrName()).getValue());
        kernels.push_back(builder.getSymbolRefAttr(kernelId));
        (*this)->setAttr(getKernelsAttrName(), builder.getArrayAttr(kernels));
    }

    void ScheduleOp::addKernel(KernelOp kernel)
    {
        OpBuilder builder(*this);
        KernelPredicateOpInterface nullPred = builder.create<NullPredicateOp>(getLoc());
        std::string id = "scheduled_" + kernel.getId().str();
        auto scheduledKernel = builder.create<ScheduledKernelOp>(getLoc(), id, kernel, nullPred);

        if (Operation* kernelOp = kernel; scheduledKernel.getOperation()->getBlock() != kernelOp->getBlock() || !kernelOp->isBeforeInBlock(scheduledKernel))
        {
            kernelOp->moveBefore(scheduledKernel);
        }

        addKernel(scheduledKernel);
    }

    void ScheduleOp::addKernel(ScheduledKernelOp kernel)
    {
        // TODO: move the original kernel as well?
        if (Operation* scheduleOp = *this; kernel.getOperation()->getBlock() != scheduleOp->getBlock() || scheduleOp->isBeforeInBlock(kernel))
        {
            scheduleOp->moveAfter(kernel);
        }

        addKernel(kernel.getId());
    }

    std::vector<InjectableMapping> ScheduleOp::getInjectableMappings()
    {
        std::vector<InjectableMapping> result;
        for (auto op = operand_begin(); op != operand_end(); ++op)
        {
            if (auto injectableMapping = dyn_cast<InjectableMapping>((*op).getDefiningOp()))
            {
                result.push_back(injectableMapping);
            }
        }

        return result;
    }

    void ScheduleOp::injectMapping(mlir::Operation* op)
    {
        // move scheduleOp to be just after mapping
        assert(mlir::isa<InjectableMapping>(op));
        getOperation()->moveBefore(op);
        op->moveBefore(*this);
        auto injectableMapping = mlir::dyn_cast<InjectableMapping>(op);
        injectableMapping.getInjectionEndOp()->moveBefore(*this);

        auto operands = std::vector<Value>(operand_begin(), operand_end());
        operands.push_back(op->getResult(0));
        getOperation()->setOperands(operands);
    }

    // Example affine maps from getIntermediateCompositeIndexMaps:
    // This was from a 5-dimensional nest that got split a few times
    //
    // hte LHS of the maps represent the actual loop indices
    // the RHS of the maps represent the original nest indices (i, j, k, l, m)
    //
    // (to have a total of 9 loops)
    //
    // Say, i, j, k, l, m are the original dimensions, in the order (i, j, l, m, k)
    //
    // at the beginning, the loop order is: d0=i, d1=j, d2=m, d3=k, d4=l or (i, j, m, k, l)
    //
    // l_outer, l_inner = split(l) -- now LHS is (i, j, m, k, l_outer, l_inner); l = l_outer + l_inner
    // j_outer, j_inner = split(j) -- now LHS is (i, j_outer, m, k, l_outer, l_inner, j_inner); l = l_outer + l_inner, j = j_outer + j_inner
    // k_outer, k_inner = split(k) -- now LHS is (i, j_outer, m, k_outer, l_outer, l_inner, j_inner, k_inner); l = l_outer + l_inner, j = j_outer + j_inner, k = k_outer + k_inner
    // i_outer, i_inner = split(i) -- now LHS is (i_outer, j_outer, m, k_outer, l_outer, l_inner, j_inner, k_inner, i_inner); l = l_outer + l_inner, j = j_outer + j_inner, k = k_outer + k_inner, i = i_outer + i_inner
    //
    // (d0) -> (d0, 0, 0, 0, 0)
    // (d0, d1) -> (d0, d1, 0, 0, 0)
    // (d0, d1, d2) -> (d0, d1, 0, 0, d2)
    // (d0, d1, d2, d3) -> (d0, d1, d3, 0, d2)
    // (d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)
    // (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5 + d4, d2)
    // (d0, d1, d2, d3, d4, d5, d6) -> (d0, d6 + d1, d3, d5 + d4, d2)
    // (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d6 + d1, d7 + d3, d5 + d4, d2)
    // (d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d8 + d0, d6 + d1, d7 + d3, d5 + d4, d2)
    //
    // (i_outer, j_outer, k_outer)

    std::vector<AffineValueMap> ScheduleOp::getIntermediateCompositeIndexMaps(mlir::OpBuilder& builder)
    {
        auto orderedIndices = getOrder();
        std::vector<AffineValueMap> resultMaps;
        resultMaps.reserve(orderedIndices.size());

        auto domain = getDomain().getValue();
        auto dimensions = domain.GetDimensions();
        auto outputDimensions = domain.NumDimensions();

        for (const auto& index : orderedIndices)
        {
            AffineValueMap indexMap;
            for (auto baseIndex : domain.GetBaseIndices(index))
            {
                auto dimIter = std::find(dimensions.begin(), dimensions.end(), baseIndex);
                assert(dimIter != dimensions.end()); // This index should have originated from a dimension
                auto logicalDimensionIdx = std::distance(dimensions.begin(), dimIter);

                auto symbolicIndexValue = getOrCreateSymbolicIndex(builder, index);

                // Each index produces a new map which adds 1 symbol to one of the output dimensions
                // and leaves all of the other output dimensions as 0
                // So copy the previous map and add the symbol where appropriate
                if (resultMaps.empty())
                {
                    // This is the first map, so don't copy a previous map
                    std::vector<AffineExpr> currentLayerAffineExprs(outputDimensions, builder.getAffineConstantExpr(0));
                    currentLayerAffineExprs[logicalDimensionIdx] = builder.getAffineDimExpr(0);
                    AffineMap newMap = AffineMap::get(1, 0, currentLayerAffineExprs, getContext());
                    indexMap.reset(newMap, { symbolicIndexValue });
                }
                else
                {
                    auto& previousAffineValueMap = resultMaps.back();
                    MutableAffineMap previousMapCopy(previousAffineValueMap.getAffineMap());
                    std::vector<Value> operands = previousAffineValueMap.getOperands().vec();
                    operands.push_back(symbolicIndexValue);
                    previousMapCopy.setNumDims(previousMapCopy.getNumDims() + 1);
                    size_t symbolIdx = resultMaps.size(); // We're adding one symbol per affine map in the list
                    auto previousValue = previousMapCopy.getResult(logicalDimensionIdx);
                    AffineExpr resultExpr = builder.getAffineDimExpr(symbolIdx);
                    if (previousValue)
                    {
                        resultExpr = resultExpr + previousValue;
                    }
                    previousMapCopy.setResult(logicalDimensionIdx, resultExpr);
                    AffineMap newMap = previousMapCopy.getAffineMap();
                    indexMap.reset(newMap, operands);
                }
            }
            resultMaps.push_back(indexMap);
        }
        return resultMaps;
    }

    AffineValueMap ScheduleOp::getCompositeIndexMap(mlir::OpBuilder& builder)
    {
        auto intermediateMaps = getIntermediateCompositeIndexMaps(builder);
        return intermediateMaps.back();
    }

    AffineMap ScheduleOp::getGlobalDimensionsToInputDimensionsMap(const std::vector<Index>& indices)
    {
        // Map global dims to input dims
        // e.g. suppose globally we have { I, J, K } dimensions, and an input only
        //      cares about { I, K }, then construct the mapping
        //      (d0, d1, d2) -> (d0, d2)
        // Similarly with order, if an input re-orders dimensions, respect that too
        // e.g. suppose globally we have { I, J, K }, and an input wants { K, J }
        //      then construct the mapping
        //      (d0, d1, d2) -> (d2, d1)

        OpBuilder builder(getContext());
        auto domain = getDomain().getValue();
        auto orderedGlobalDimensions = domain.GetDimensions();
        int numInputDims = orderedGlobalDimensions.size();
        int numResults = indices.size();
        std::vector<AffineExpr> affineExprs;
        affineExprs.reserve(numResults);
        for (const auto& index : indices)
        {
            auto baseDim = domain.GetBaseIndex(index);
            auto globalDimIter = std::find(orderedGlobalDimensions.begin(), orderedGlobalDimensions.end(), baseDim);
            assert(globalDimIter != orderedGlobalDimensions.end());
            auto idx = std::distance(orderedGlobalDimensions.begin(), globalDimIter);
            affineExprs.push_back(builder.getAffineDimExpr(idx));
        }
        AffineMap resultMap = AffineMap::get(numInputDims, 0 /*symbols*/, affineExprs, getContext());
        return resultMap;
    }

    void ScheduleOp::addLoopAttribute(Index targetIndex, mlir::Identifier name, mlir::Attribute value)
    {
        OpBuilder builder(getContext());

        // The index attr attribute is an array of DictionaryAttr's, where each Dictionary element
        // holds a mapping of getIndexAttrKeyName() to the targetIndex as an index attr
        // and then holds all of the other name -> value mappings for that index
        auto currentArrayAttr = (*this)->getAttrOfType<ArrayAttr>(getLoopAttrsName());
        assert(currentArrayAttr != nullptr);

        auto dictionaryAttrs = util::ArrayAttrToVector<mlir::DictionaryAttr>(currentArrayAttr);

        auto dictAttr = getLoopAttributes(targetIndex);
        if (dictAttr)
        {
            // We already have attributes for this index, so just insert this name -> value pair into the current dictionary
            mlir::NamedAttrList mutableDictEntries = *dictAttr;
            mutableDictEntries.set(name, value);

            // Replace the DictionaryAttr in dictionaryAttrs with the copy we've modified
            auto dictAttrIter = std::find(dictionaryAttrs.begin(), dictionaryAttrs.end(), *dictAttr);
            assert(dictAttrIter != dictionaryAttrs.end());
            dictionaryAttrs.erase(dictAttrIter);
            dictionaryAttrs.push_back(mutableDictEntries.getDictionary(getContext()));
        }
        else
        {
            // We didn't have any attributes for this index, so create a new entry in the dictionary
            auto indexAttrKeyIdentifier = builder.getIdentifier(getIndexAttrKeyName());

            mlir::NamedAttrList newDictEntries;
            newDictEntries.set(indexAttrKeyIdentifier, IndexAttr::get(targetIndex, getContext()));
            newDictEntries.set(name, value);
            dictionaryAttrs.push_back(newDictEntries.getDictionary(getContext()));
        }

        auto dictionaryArrayAttr = util::VectorToArrayAttr<mlir::DictionaryAttr>(dictionaryAttrs, getContext());

        (*this)->setAttr(getLoopAttrsName(), dictionaryArrayAttr);
    }

    std::optional<mlir::DictionaryAttr> ScheduleOp::getLoopAttributes(Index targetIndex)
    {
        OpBuilder builder(getContext());

        auto currentArrayAttr = (*this)->getAttrOfType<ArrayAttr>(getLoopAttrsName());
        assert(currentArrayAttr != nullptr);

        auto indexAttrKeyIdentifier = builder.getIdentifier(getIndexAttrKeyName());

        auto dictionaryAttrs = util::ArrayAttrToVector<mlir::DictionaryAttr>(currentArrayAttr);

        for (auto& dictAttr : dictionaryAttrs)
        {
            auto currentIndexValue = dictAttr.get(indexAttrKeyIdentifier);
            assert(currentIndexValue != nullptr && "Found Loop attributes stored without an Index");
            assert(currentIndexValue.isa<IndexAttr>() && "Index attribute in loop attribute mapping is not an IndexAttr");
            auto currentIndexAttr = currentIndexValue.cast<IndexAttr>();
            auto currentIndex = currentIndexAttr.getValue();
            if (currentIndex == targetIndex)
            {
                return dictAttr;
            }
        }
        return std::nullopt;
    }

    bool ScheduleOp::hasConstantRanges()
    {
        auto domain = getDomain().getValue();
        auto dimensionIndices = domain.GetDimensions();
        return std::all_of(dimensionIndices.begin(), dimensionIndices.end(), [&](auto i) {
            return domain.HasConstantDimensionSize(i);
        });
    }

    static mlir::LogicalResult verify(ScheduleOp op)
    {
        // int numDimensions = op.numDimensions();
        auto order = op.getOrder();
        auto numLoops = op.numLoops();
        if (order.size() != numLoops)
        {
            return op.emitOpError("size of order list != numLoops(), numLoops: ") << numLoops << ", order list: " << op.getOrder().size();
        }

        auto loopIndices = op.getDomain().getValue().GetAllLoopIndices();
        if (!std::is_permutation(loopIndices.begin(), loopIndices.end(), order.begin()))
        {
            return op.emitOpError("order isn't a permutation of the loop indices");
        }

        // Find all kernels and make sure they're ScheduledKernelOps

        return mlir::success();
    }

    std::tuple<ScheduleOp, Index> Fuse(mlir::OpBuilder& builder, ScheduleOp schedule1, ScheduleOp schedule2)
    {
        auto domain1 = schedule1.getDomain().getValue();
        auto domain2 = schedule2.getDomain().getValue();
        auto schedule1Indices = domain1.GetAllLoopIndices();
        auto schedule2Indices = domain2.GetAllLoopIndices();
        assert(schedule1Indices.size() == schedule2Indices.size());
        std::vector<std::vector<Index>> indexCorrespondences;
        indexCorrespondences.reserve(schedule1Indices.size());
        std::transform(
            schedule1Indices.begin(),
            schedule1Indices.end(),
            schedule2Indices.begin(),
            std::back_inserter(indexCorrespondences),
            [](Index i1, Index i2) {
                return std::vector{ i1, i2 };
            });

        return Fuse(builder, std::vector{ schedule1, schedule2 }, indexCorrespondences);
    }

    std::tuple<ScheduleOp, Index> Fuse(
        mlir::OpBuilder& builder,
        const std::vector<ScheduleOp>& schedules,
        const std::vector<std::vector<Index>>& indexCorrespondences)
    {
        assert(!schedules.empty());

        if (const auto numSchedules = schedules.size();
            std::any_of(indexCorrespondences.begin(),
                        indexCorrespondences.end(),
                        [&numSchedules](const std::vector<Index>& indices) { return indices.size() != numSchedules; }))
        {
            throw std::logic_error("Index correspondences don't match number of schedules");
        }

        auto schedule1 = schedules[0];
        auto loc = schedule1.getLoc();
        const auto numSchedules = schedules.size();
        const auto numFusedIndices = indexCorrespondences.size();

        mlir::OpBuilder::InsertionGuard insertGuard(builder);
        // assumes nest are declared in fuse order
        auto lastNest = ScheduleOp{ schedules[schedules.size() - 1] }.getNest();
        builder.setInsertionPointAfter(lastNest);

        std::vector<TransformedDomain> domains;
        domains.reserve(numSchedules);
        std::transform(schedules.begin(), schedules.end(), std::back_inserter(domains), [](ScheduleOp op) { return op.getDomain().getValue(); });

        std::vector<std::vector<Index>> scheduleDimensions;
        scheduleDimensions.reserve(numSchedules);
        std::transform(domains.begin(), domains.end(), std::back_inserter(scheduleDimensions), [](const TransformedDomain& domain) { return domain.GetDimensions(); });

        std::vector<IndexRange> dimensionRanges;
        const auto numTotalDimensions = static_cast<size_t>(
            std::accumulate(domains.begin(), domains.end(), 0, [](int64_t x, const TransformedDomain& domain) { return x + domain.NumDimensions(); }));
        dimensionRanges.reserve(numTotalDimensions + 1);

        // Copy non-fused dimension ranges
        for (size_t idx = 0; idx < scheduleDimensions.size(); ++idx)
        {
            for (auto dim : scheduleDimensions[idx])
            {
                if (auto it = std::find_if(
                        indexCorrespondences.begin(),
                        indexCorrespondences.end(),
                        [=](const std::vector<Index>& correspondence) { return correspondence[idx] == dim; });
                    it == indexCorrespondences.end())
                {
                    dimensionRanges.emplace_back(dim, domains[idx].GetIndexRange(dim));
                }
            }
        }

        // Collect the fused domains (for recovering the original schedules in Debug mode)
        // Assumes schedules are declared in fuse order
        std::vector<IterationDomain> allFusedDomains;
        for (auto schedule : schedules)
        {
            auto fusedDomains = schedule.getFusedDomains();
            if (fusedDomains.empty()) // never fused
            {
                fusedDomains.push_back(schedule.getNest().getDomain().getValue());
            }
            allFusedDomains.insert(allFusedDomains.end(), fusedDomains.begin(), fusedDomains.end());
        }

        // Create new loop nest
        IterationDomain fusedNestDomain{ dimensionRanges };
        auto fusedNest = builder.create<NestOp>(loc, fusedNestDomain);
        auto nestBuilder = fusedNest.getBodyBuilder();

        // Create a schedule for the fused nest
        auto fusedSchedule = fusedNest.getOrCreateSchedule();

        // First add "fusing" index (as outermost index)
        Index fusingIndex(std::string{ "f" } + std::to_string(ir::util::GetUniqueId()));
        std::vector<Index> fusedScheduleOrder;
        fusedScheduleOrder.emplace_back(fusingIndex);
        [[maybe_unused]] auto fusingIndexOp = fusedNest.getOrCreateSymbolicIndex(nestBuilder, fusingIndex);

        auto fusedDomain = TransformedDomain::Fuse(domains, indexCorrespondences);
        fusedDomain.AddDimension(fusingIndex, loopnest::Range{ 0, static_cast<int64_t>(schedules.size()) });
        fusedSchedule.setDomain(fusedDomain);
        fusedSchedule.setFusedDomains(allFusedDomains);

        // Then add indices in the correspondence list
        std::map<ScheduleOp, std::map<Index, Index>> schedFusedIndexMap;
        for (const auto& correspondence : indexCorrespondences)
        {
            auto i0 = correspondence[0];
            fusedScheduleOrder.emplace_back(i0);
            for (size_t idx = 1; idx < correspondence.size(); ++idx)
            {
                schedFusedIndexMap[schedules[idx]][correspondence[idx]] = i0;
            }
        }

        auto updateFusedIndicesAttribute = [&](ScheduleOp& oldSchedule, Index oldIndex, ScheduleOp& newSchedule, Index newIndex) {
            auto indices = newSchedule.getFusedIndices(newIndex);
            auto oldIndices = oldSchedule.getFusedIndices(oldIndex);

            // Copy any reverse mappings from oldIndex
            //   From: {fused_indices = [#accln<"index{i,1}">], scheduledIndex = oldIndex">}
            //   To: {fused_indices = [#accln<"index{i,1}">], scheduledIndex = newIndex">}
            indices.insert(indices.end(), oldIndices.begin(), oldIndices.end());

            // Add an entry for oldIndex in the reverse mapping for newIndex:
            //   {fused_indices = [#accln<"index{i,1}">, oldIndex], scheduledIndex = newIndex">}
            if (oldIndex != newIndex)
            {
                indices.push_back(oldIndex);
            }

            if (!indices.empty())
            {
                newSchedule.setFusedIndices(newIndex, indices);
            }
        };

        std::vector<NestOp> nestOps;
        nestOps.reserve(numSchedules);
        std::transform(schedules.begin(), schedules.end(), std::back_inserter(nestOps), [](ScheduleOp op) { return op.getNest(); });

        llvm::SmallVector<mlir::BlockAndValueMapping, 8> valueMap(nestOps.size());
        // Map symbolic index ops in nest1 to ops in the fused nest
        nestOps[0].walk([&](SymbolicIndexOp index) {
            if (!index.use_empty())
            {
                auto newIndex = fusedNest.getOrCreateSymbolicIndex(nestBuilder, index.getValue());
                valueMap[0].map(index.getResult(), newIndex.getResult());

                // Reverse map "newIndex" to its fused indices, for recovering the original schedules in Debug mode
                updateFusedIndicesAttribute(schedule1, index.getValue(), fusedSchedule, newIndex.getValue());
            }
        });

        for (size_t idx = 1; idx < nestOps.size(); ++idx)
        {
            // Map symbolic index ops in nest1..N to ops in the fused nest, replacing them with the corresponding nest1..N index if necessary
            nestOps[idx].walk([&](SymbolicIndexOp index) {
                if (!index.use_empty())
                {
                    auto schedule = schedules[idx];
                    auto& fusedMap = schedFusedIndexMap[schedule];
                    auto indexVal = index.getValue();
                    if (auto it = fusedMap.find(indexVal); it != fusedMap.end())
                    {
                        indexVal = it->second;
                        // TODO: don't add this if it's already been added
                    }
                    auto newIndex = fusedNest.getOrCreateSymbolicIndex(nestBuilder, indexVal);
                    valueMap[idx].map(index.getResult(), newIndex.getResult());

                    // Reverse map "newIndex" to its fused indices, for recovering the original schedules in Debug mode
                    updateFusedIndicesAttribute(schedule, index.getValue(), fusedSchedule, newIndex.getValue());
                }
            });
        }

        // Create the predicates that guarantee correctness
        std::vector<KernelPredicateOpInterface> predicates;
        predicates.reserve(numSchedules);

        for (size_t idx = 0; idx < numSchedules; ++idx)
        {
            // First predicate is on the fusing index to ensure schedule ordering.
            auto predicate = IndexAt(nestBuilder, fusingIndex, static_cast<int64_t>(idx));

            // Next predicate is on the fused index, to ensure that each schedule conforms to the
            // the bounds of its original iteration space.
            for (const auto& correspondence : indexCorrespondences)
            {
                auto originalIndex = correspondence[idx];
                auto originalRange = domains[idx].GetIndexRange(originalIndex);
                auto fusedIndex = correspondence[0];
                auto fusedRange = fusedDomain.GetIndexRange(fusedIndex);
                if (fusedRange != originalRange)
                {
                    predicate = Conjunction(nestBuilder, predicate, InRange(nestBuilder, fusedIndex, originalRange));
                }
            }

            predicates.emplace_back(predicate);
        }

        // TODO: Figure out if we want to support this scenario
        // Next the predicates on explicitly-separate indices (those with a "null" index in either spot in the correspondence list)
        // for (const auto& [i1, i2] : indexCorrespondences)
        // {
        //     if (i1 == Index::none)
        //     {
        //         p1 = p1 && First(i2);
        //     }
        //     else if (i2 == Index::none)
        //     {
        //         p2 = p2 && Last(i1);
        //     }
        // }

        // And finally handle the indices not included in the correspondence list

        std::vector<std::unordered_set<Index>> fusedIndices(schedules.size());
        for (size_t idx = 1; idx < schedules.size(); ++idx)
        {
            auto& m = schedFusedIndexMap[schedules[idx]];
            for (auto& kvp : m)
            {
                fusedIndices[0].insert(kvp.second);

                fusedIndices[idx].insert(kvp.first);
            }
        }

        for (size_t idx = 0; idx < numSchedules; ++idx)
        {
            auto schedule = schedules[idx];
            auto schedIndices = schedule.getOrder();
            std::vector<Index> unfusedIndices;
            unfusedIndices.reserve(schedIndices.size() - numFusedIndices);
            llvm::copy_if(
                schedIndices,
                std::back_inserter(unfusedIndices),
                [&fusedIndices, idx](const Index& i) {
                    return fusedIndices[idx].count(i) == 0;
                });

            for (const auto& index : unfusedIndices)
            {
                if (idx > 0)
                {
                    predicates[idx - 1] = Conjunction(nestBuilder, predicates[idx - 1], First(nestBuilder, index));
                }
                if (idx < numSchedules - 1)
                {
                    predicates[idx + 1] = Conjunction(nestBuilder, predicates[idx + 1], Last(nestBuilder, index));
                }
                fusedScheduleOrder.emplace_back(index);
            }
        }

        std::vector<ScheduledKernelOp> newKernels;
        for (size_t idx = 0; idx < schedules.size(); ++idx)
        {
            for (auto& scheduledKernel : ScheduleOp{ schedules[idx] }.getKernels())
            {
                auto oldPredicateOp = scheduledKernel.getPredicate();
                auto predicate = predicates[idx];
                if (mlir::isa<KernelPredicateOpInterface>(oldPredicateOp) && !mlir::isa<NullPredicateOp>(oldPredicateOp))
                {
                    auto clonedPredicate = mlir::cast<KernelPredicateOpInterface>(ir::util::CloneRecursively(nestBuilder, oldPredicateOp, valueMap[idx]));
                    predicate = Conjunction(nestBuilder, predicate, clonedPredicate);
                }

                auto scheduledKernelId = scheduledKernel.getId();
                auto kernelId = scheduledKernel.getKernel();
                auto oldKernel = nestOps[idx].getKernel(kernelId);
                auto kernel = mlir::cast<KernelOp>(nestBuilder.clone(*oldKernel.getOperation(), valueMap[idx]));
                newKernels.push_back(nestBuilder.create<ScheduledKernelOp>(loc, scheduledKernelId, kernel, predicate));
            }
        }

        // Create a schedule for the fused nest and add the kernels
        for (auto& newKernel : newKernels)
        {
            fusedSchedule.addKernel(newKernel);
        }
        fusedSchedule.setOrder(fusedScheduleOrder);

        for (auto nestOp : nestOps)
        {
            nestOp.erase();
        }

        return { fusedSchedule, fusingIndex };
    }

    std::tuple<ScheduleOp, Index> Fuse(mlir::OpBuilder& builder, ScheduleOp schedule1, ScheduleOp schedule2, const std::vector<std::pair<Index, Index>>& indexCorrespondences)
    {
        std::vector<std::vector<Index>> correspondences;
        correspondences.reserve(indexCorrespondences.size());
        for (const auto& [i1, i2] : indexCorrespondences)
        {
            correspondences.push_back(std::vector{ i1, i2 });
        }

        return Fuse(builder, std::vector{ schedule1, schedule2 }, correspondences);
    }

    //
    // NestOp
    //

    // This is the main build method
    void NestOp::build(OpBuilder& builder, OperationState& result, const IterationDomain& domain)
    {
        auto domainAttr = IterationDomainAttr::get(domain, builder.getContext());
        build(builder, result, builder.getArrayAttr({}), domainAttr, {}, llvm::None);
        ensureTerminator(*result.regions.front(), builder, result.location);
    }

    void NestOp::build(OpBuilder& builder, OperationState& result, ArrayRef<int64_t> loopRanges)
    {
        std::vector<IndexRange> ranges;
        for (size_t i = 0; i < loopRanges.size(); ++i)
        {
            ranges.push_back(IndexRange(GetIndexName(i), Range(0, loopRanges[i], 1)));
        }

        IterationDomain domain(ranges);
        build(builder, result, domain);
    }

    void NestOp::build(OpBuilder& builder, OperationState& result, ArrayRef<mlir::Value> loopRanges)
    {
        std::vector<Range> ranges;

        for (size_t i = 0; i < loopRanges.size(); ++i)
        {
            ranges.push_back(Range(0, OperandIndex(i), 1));
        }

        std::vector<IndexRange> indexRanges;
        for (size_t i = 0; i < loopRanges.size(); ++i)
        {
            indexRanges.push_back(IndexRange("i_" + std::to_string(i), ranges[i]));
        }

        IterationDomain domain(indexRanges);
        auto domainAttr = IterationDomainAttr::get(domain, builder.getContext());
        build(builder, result, builder.getArrayAttr({}), domainAttr, {}, loopRanges);
        ensureTerminator(*result.regions.front(), builder, result.location);
    }

    size_t NestOp::numDimensions()
    {
        return getDomain().getValue().NumDimensions();
    }

    SymbolicIndexOp NestOp::getOrCreateSymbolicIndex(OpBuilder& builder, Index index)
    {
        return builder.create<SymbolicIndexOp>(getLoc(), index);
    }

    std::vector<SymbolicIndexOp> NestOp::getIndices(OpBuilder& builder)
    {
        std::vector<SymbolicIndexOp> result;
        auto domain = getDomain().getValue();
        for (auto d : domain.GetDimensions())
        {
            result.push_back(getOrCreateSymbolicIndex(builder, d));
        }

        return result;
    }

    IterationDomainAttr NestOp::getDomain()
    {
        return domain();
    }

    std::vector<std::string> NestOp::getKernelIds()
    {
        std::vector<std::string> result;
        for (auto k : kernels())
        {
            result.push_back(std::string{ k.cast<FlatSymbolRefAttr>().getValue() });
        }
        return result;
    }

    std::vector<KernelOp> NestOp::getKernels()
    {
        std::vector<KernelOp> result;

        // Kernel ids
        for (auto idAttr : kernels())
        {
            auto id = idAttr.cast<FlatSymbolRefAttr>().getValue();
            auto kernelOp = getKernel(id);
            result.push_back(kernelOp);
        }

        walk([&](KernelOp k) {
            result.push_back(k);
        });
        return result;
    }

    KernelOp NestOp::getKernel(llvm::StringRef id)
    {
        auto symTableOp = mlir::SymbolTable::getNearestSymbolTable(getOperation());
        auto symbolOp = mlir::SymbolTable::lookupNearestSymbolFrom(symTableOp, id);
        auto kernelOp = mlir::cast<KernelOp>(symbolOp);
        assert(kernelOp && "Kernel not found");
        return kernelOp;
    }

    std::vector<int64_t> NestOp::getLoopRanges()
    {
        auto domain = getDomain().getValue();
        auto ranges = domain.GetRanges();
        std::vector<int64_t> result;
        for (auto r : ranges)
        {
            result.push_back(r.Size());
        }
        return result;
    }

    ScheduleOp NestOp::getOrCreateSchedule()
    {
        if (auto opRange = this->getOps<ScheduleOp>(); !opRange.empty())
        {
            assert(std::distance(opRange.begin(), opRange.end()) == 1 && "Expected only one ScheduleOp inside NestOp body");
            return *opRange.begin();
        }

        if (body().empty())
        {
            body().push_back(new Block);
        }
        return getBodyBuilder().create<ScheduleOp>(getLoc(), *this);
    }

    OpBuilder NestOp::getBodyBuilder()
    {
        auto insertPt = util::GetTerminalInsertPoint<NestOp, ScheduleOp, TerminatorOp>(*this);
        return OpBuilder(insertPt.getBlock(), insertPt.getPoint());
    }

    static mlir::LogicalResult verify(NestOp op)
    {
        // auto numOperands = static_cast<int64_t>(op.getOperands().size());
        // int numDimensions = op.numDimensions();
        // int loopRangesSize = op.loopRanges().size();

        // check that the number of loopRanges args == # dimensions
        // if (loopRangesSize != numDimensions)
        // {
        //     return op.emitOpError("number of loop sizes != numDimensions(), numDimensions: ") << numDimensions << ", loopRanges: " << loopRangesSize;
        // }

        // check that the number of results == # dimensions (disabled, since we're not returning anything anymore)
        // if (op.getODSResults(0).size() != numDimensions)
        // {
        //     return op.emitOpError("number of return values != numDimensions(), numDimensions: ") << numDimensions << ", loopRanges: " << op.loopRanges().size();
        // }
        return mlir::success();
    }

    NestOp MakeNest(mlir::OpBuilder& builder, const IterationDomain& domain)
    {
        auto ranges = domain.GetRanges();
        auto nest = builder.create<NestOp>(builder.getUnknownLoc(), domain);
        return nest;
    }

    NestOp MakeNest(mlir::OpBuilder& builder, ArrayRef<int64_t> loopRanges)
    {
        auto nest = builder.create<NestOp>(builder.getUnknownLoc(), loopRanges);
        return nest;
    }

    NestOp MakeNest(mlir::OpBuilder& builder, ArrayRef<mlir::Value> loopRanges)
    {
        auto nest = builder.create<NestOp>(builder.getUnknownLoc(), loopRanges);
        return nest;
    }

    //
    // ScheduledLoopOp
    //
    void ScheduledLoopOp::build(OpBuilder& builder, OperationState& result, int64_t begin, int64_t end, int64_t step, Value symbolicIndex, const std::vector<int64_t>& subdomainSize, const std::vector<Index>& subdomainIndexOrder)
    {
        ArrayAttr subdomainIndexOrderAttr = util::ConvertIndexVectorToArrayAttr(subdomainIndexOrder, builder.getContext());

        auto index = cast<SymbolicIndexOp>(symbolicIndex.getDefiningOp()).index();

        build(builder, result, builder.getI64IntegerAttr(begin), builder.getI64IntegerAttr(end), builder.getI64IntegerAttr(step), index, symbolicIndex, builder.getI64ArrayAttr(subdomainSize), subdomainIndexOrderAttr);

        result.regions.clear();

        Region* prologueRegion = result.addRegion();
        ScheduledLoopOp::ensureTerminator(*prologueRegion, builder, result.location);
        prologueRegion->front().addArgument(builder.getIndexType());

        Region* bodyRegion = result.addRegion();
        ScheduledLoopOp::ensureTerminator(*bodyRegion, builder, result.location);
        bodyRegion->front().addArgument(builder.getIndexType());

        Region* epilogueRegion = result.addRegion();
        ScheduledLoopOp::ensureTerminator(*epilogueRegion, builder, result.location);
        epilogueRegion->front().addArgument(builder.getIndexType());
    }

    int64_t ScheduledLoopOp::getNumIterations()
    {
        return Range(begin(), end(), step()).NumIterations();
    }

    SymbolicIndexOp ScheduledLoopOp::getSymbolicIndex()
    {
        return dyn_cast<SymbolicIndexOp>(symbolicIndex().getDefiningOp());
    }

    std::vector<int64_t> ScheduledLoopOp::getSubdomainSize()
    {
        return util::ConvertArrayAttrToIntVector(subdomainSizeAttr());
    }

    void ScheduledLoopOp::setSubdomainSize(const std::vector<int64_t>& subdomainSize)
    {
        OpBuilder builder(getContext());
        (*this)->setAttr("subdomainSize", builder.getI64ArrayAttr(subdomainSize));
    }

    std::vector<Index> ScheduledLoopOp::getSubdomainIndexOrder()
    {
        return util::ConvertArrayAttrToIndexVector(subdomainIndexOrderAttr());
    }

    static mlir::LogicalResult verify(ScheduledLoopOp op)
    {
        return mlir::success();
    }

    static void print(OpAsmPrinter& p, ScheduledLoopOp op)
    {
        bool printBlockTerminators = false;
        auto indexAttr = op.index();
        p << op.getOperationName() << " " << op.symbolicIndex() << " (" << indexAttr << ")";
        p << " = " << op.begin() << " to " << op.end() << " step " << op.step();

        p << "  prologue(" << op.getPrologue()->getArgument(0) << ")";
        p.printRegion(op.prologue(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/printBlockTerminators);
        p << ",  body(" << op.getBody()->getArgument(0) << ")";
        p.printRegion(op.body(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/printBlockTerminators);
        p << ",  epilogue(" << op.getEpilogue()->getArgument(0) << ")";
        p.printRegion(op.epilogue(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/printBlockTerminators);
        p.printOptionalAttrDict(op->getAttrs(),
                                /*elidedAttrs=*/{ "begin", "end", "step", "index" });
    }

    static ParseResult parseScheduledLoopOp(OpAsmParser& parser, OperationState& result)
    {
        auto& builder = parser.getBuilder();
        Type indexType = builder.getIndexType();
        Type i64Type = builder.getIntegerType(64);

        OpAsmParser::OperandType symbolicIndex;

        // Parse the SSA index variable followed by '(', then the index as an attribute, then ')' and '='
        if (failed(parser.parseOperand(symbolicIndex)) ||
            failed(parser.resolveOperand(symbolicIndex, indexType, result.operands)))
            return failure();

        if (failed(parser.parseLParen()))
            return failure();

        IndexAttr index;
        if (failed(parser.parseAttribute(index, "index", result.attributes)))
            return failure();

        if (failed(parser.parseRParen()))
            return failure();

        if (failed(parser.parseEqual()))
            return failure();

        // Parse loop bounds.
        IntegerAttr boundsAttr;
        if (failed(parser.parseAttribute(boundsAttr, i64Type, "begin", result.attributes)))
            return failure();
        if (failed(parser.parseKeyword("to")))
            return failure();
        if (failed(parser.parseAttribute(boundsAttr, i64Type, "end", result.attributes)))
            return failure();
        if (failed(parser.parseKeyword("step")))
            return failure();
        if (failed(parser.parseAttribute(boundsAttr, i64Type, "step", result.attributes)))
            return failure();

        // TODO: figure this out
        OpAsmParser::OperandType prologueInductionVar;
        OpAsmParser::OperandType bodyInductionVar;
        OpAsmParser::OperandType epilogueInductionVar;

        SmallVector<OpAsmParser::OperandType, 4> regionArgs;
        SmallVector<Type, 4> argTypes;
        argTypes.push_back(indexType);

        // Parse prologue
        if (failed(parser.parseKeyword("prologue")) || failed(parser.parseLParen()) || failed(parser.parseRegionArgument(prologueInductionVar)) || failed(parser.parseRParen()))
            return parser.emitError(parser.getNameLoc(), "Error parsing prologue header");

        regionArgs.push_back(prologueInductionVar);
        Region* prologue = result.addRegion();
        if (parser.parseRegion(*prologue, regionArgs, argTypes))
            return parser.emitError(parser.getNameLoc(), "Error parsing prologue region");
        ScheduledLoopOp::ensureTerminator(*prologue, builder, result.location);

        if (failed(parser.parseComma()))
            return failure();

        // Parse body
        if (failed(parser.parseKeyword("body")) || failed(parser.parseLParen()) || failed(parser.parseRegionArgument(bodyInductionVar)) || failed(parser.parseRParen()))
            return parser.emitError(parser.getNameLoc(), "Error parsing body header");
        regionArgs.clear();
        regionArgs.push_back(bodyInductionVar);
        Region* body = result.addRegion();
        if (parser.parseRegion(*body, regionArgs, argTypes))
            return parser.emitError(parser.getNameLoc(), "Error parsing body region");
        ScheduledLoopOp::ensureTerminator(*body, builder, result.location);

        if (failed(parser.parseComma()))
            return failure();

        // Parse epilogue
        if (failed(parser.parseKeyword("epilogue")) || failed(parser.parseLParen()) || failed(parser.parseRegionArgument(epilogueInductionVar)) || failed(parser.parseRParen()))
            return parser.emitError(parser.getNameLoc(), "Error parsing epilogue header");
        regionArgs.clear();
        regionArgs.push_back(epilogueInductionVar);
        Region* epilogue = result.addRegion();
        if (parser.parseRegion(*epilogue, regionArgs, argTypes))
            return parser.emitError(parser.getNameLoc(), "Error parsing epilogue region");
        ScheduledLoopOp::ensureTerminator(*epilogue, builder, result.location);

        // Parse the optional attribute list.
        if (parser.parseOptionalAttrDict(result.attributes))
            return failure();

        return success();
    }

    //
    // KernelOp
    //
    void KernelOp::build(OpBuilder& builder, OperationState& result, StringRef id)
    {
        result.addAttribute(getIdAttrName(), builder.getStringAttr(id));

        Region* bodyRegion = result.addRegion();
        KernelOp::ensureTerminator(*bodyRegion, builder, result.location);
    }

    StringRef KernelOp::getIdAttrName()
    {
        return mlir::SymbolTable::getSymbolAttrName(); /* was "id" */
    }

    StringRef KernelOp::getId()
    {
        return (*this)->getAttrOfType<StringAttr>(getIdAttrName()).getValue();
    }

    std::vector<SymbolicIndexOp> KernelOp::getIndices()
    {
        llvm::SmallSet<SymbolicIndexOp, 4> result;

        auto body = getBody();

        for (auto& op : body->without_terminator())
        {
            for (auto operand : op.getOperands())
            {
                if (auto symIndex = dyn_cast_or_null<SymbolicIndexOp>(operand.getDefiningOp()))
                {
                    result.insert(symIndex);
                }
            }
        }

        return std::vector(result.begin(), result.end());
    }

    void KernelOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                               MLIRContext* context)
    {
        populateKernelCanonicalizationPatterns(results, context);
    }

    static mlir::LogicalResult verify(KernelOp op)
    {
        return mlir::success();
    }

    // TODO: uniquify id globally
    std::string GetUniqueKernelId(std::string id, mlir::Region* region)
    {
        auto ids = GetKernelIdsInFunction(region->getParentOfType<ir::value::ValueFuncOp>());
        std::set<mlir::StringRef> kernelIds(ids.begin(), ids.end());

        if (kernelIds.count(id) == 0)
            return id;

        int64_t uniqueIdVal = util::GetUniqueId();
        std::string uniquedId = id + "_" + std::to_string(uniqueIdVal);
        if (kernelIds.count(uniquedId) == 0)
        {
            return uniquedId;
        }
        assert(false && "Couldn't find unique ID");
        return "";
    }

    KernelOp MakeKernel(OpBuilder& builder, std::string id, std::function<void(mlir::OpBuilder&, mlir::Location)> body)
    {
        auto loc = builder.getUnknownLoc();
        auto region = builder.getInsertionBlock()->getParent();
        auto uniqueId = GetUniqueKernelId(id, region);
        auto op = builder.create<KernelOp>(loc, uniqueId);
        auto bodyBuilder = op.getBodyBuilder();
        body(bodyBuilder, loc);

        return op;
    }

    KernelOp MakeKernel(OpBuilder& builder, std::function<void(mlir::OpBuilder&, mlir::Location)> body)
    {
        return MakeKernel(builder, "kernel", body);
    }

    //
    // ScheduledKernelOp
    //

    void ScheduledKernelOp::build(OpBuilder& builder, OperationState& result, StringRef id, KernelOp kernel, mlir::Value predicate)
    {
        auto kernelId = kernel.getId();
        result.addAttribute(getIdAttrName(), builder.getStringAttr(id));
        result.addAttribute(getKernelIdAttrName(), builder.getSymbolRefAttr(kernelId));
        result.addOperands(predicate);
    }

    void ScheduledKernelOp::build(OpBuilder& builder, OperationState& result, StringRef id, KernelOp kernel, mlir::Value predicate, mlir::Value placementPredicate)
    {
        auto kernelId = kernel.getId();
        result.addAttribute(getIdAttrName(), builder.getStringAttr(id));
        result.addAttribute(getKernelIdAttrName(), builder.getSymbolRefAttr(kernelId));
        result.addOperands({ predicate, placementPredicate });
    }

    void ScheduledKernelOp::build(OpBuilder& builder, OperationState& result, StringRef id, KernelOp kernel, KernelPredicateOpInterface predicate)
    {
        auto kernelId = kernel.getId();
        result.addAttribute(getIdAttrName(), builder.getStringAttr(id));
        result.addAttribute(getKernelIdAttrName(), builder.getSymbolRefAttr(kernelId));
        result.addOperands(predicate.getOperation()->getResult(0));
    }

    void ScheduledKernelOp::build(OpBuilder& builder, OperationState& result, StringRef id, KernelOp kernel, KernelPredicateOpInterface predicate, KernelPredicateOpInterface placementPredicate)
    {
        auto kernelId = kernel.getId();
        result.addAttribute(getIdAttrName(), builder.getStringAttr(id));
        result.addAttribute(getKernelIdAttrName(), builder.getSymbolRefAttr(kernelId));
        result.addOperands({ predicate.getOperation()->getResult(0), placementPredicate.getOperation()->getResult(0) });
    }

    void ScheduledKernelOp::build(OpBuilder& builder, OperationState& result, StringRef id, KernelOp kernel, EvaluatablePredicateOpInterface predicate)
    {
        auto kernelId = kernel.getId();
        result.addAttribute(getIdAttrName(), builder.getStringAttr(id));
        result.addAttribute(getKernelIdAttrName(), builder.getSymbolRefAttr(kernelId));
        result.addOperands(predicate.getOperation()->getResult(0));
    }

    StringRef ScheduledKernelOp::getIdAttrName()
    {
        return mlir::SymbolTable::getSymbolAttrName(); /* was "id" */
    }

    StringRef ScheduledKernelOp::getKernelIdAttrName()
    {
        return "kernel";
    }

    StringRef ScheduledKernelOp::getId()
    {
        return (*this)->getAttrOfType<StringAttr>(getIdAttrName()).getValue();
    }

    StringRef ScheduledKernelOp::getKernel()
    {
        return (*this)->getAttrOfType<FlatSymbolRefAttr>(getKernelIdAttrName()).getValue();
    }

    Operation* ScheduledKernelOp::getPredicate()
    {
        auto pred = predicate();
        if (pred)
            return pred.getDefiningOp();
        else
            return nullptr;
    }

    Operation* ScheduledKernelOp::getPlacementPredicate()
    {
        auto pred = placementPredicate();
        if (pred)
            return pred.getDefiningOp();
        else
            return nullptr;
    }

    void ScheduledKernelOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                                        MLIRContext* context)
    {
        populateScheduledKernelCanonicalizationPatterns(results, context);
    }

    static mlir::LogicalResult verify(ScheduledKernelOp op)
    {
        return mlir::success();
    }

    ScheduledKernelOp MakeKernel(mlir::OpBuilder& builder, KernelOp kernel, KernelPredicateOpInterface predicate)
    {
        auto loc = builder.getUnknownLoc();
        auto region = builder.getInsertionBlock()->getParent();
        std::string id = GetUniqueKernelId("scheduled_" + kernel.getId().str(), region);
        return builder.create<ScheduledKernelOp>(loc, id, kernel, predicate);
    }

    ScheduledKernelOp MakeKernel(mlir::OpBuilder& builder, KernelOp kernel, KernelPredicateOpInterface predicate, KernelPredicateOpInterface placementPredicate)
    {
        auto loc = builder.getUnknownLoc();
        auto region = builder.getInsertionBlock()->getParent();
        std::string id = GetUniqueKernelId("scheduled_" + kernel.getId().str(), region);
        return builder.create<ScheduledKernelOp>(loc, id, kernel, predicate, placementPredicate);
    }

    ScheduledKernelOp MakeKernel(mlir::OpBuilder& builder, KernelOp kernel, EvaluatablePredicateOpInterface predicate)
    {
        auto loc = builder.getUnknownLoc();
        auto region = builder.getInsertionBlock()->getParent();
        std::string id = GetUniqueKernelId("scheduled_" + kernel.getId().str(), region);
        return builder.create<ScheduledKernelOp>(loc, id, kernel, predicate);
    }

    //
    // DimSizeOp
    //
    void DimSizeOp::build(OpBuilder& builder, OperationState& result, Index index)
    {
        build(builder, result, builder.getIndexType(), IndexAttr::get(index, builder.getContext()));
    }

    StringRef DimSizeOp::getIndexAttrName()
    {
        return "dimensionIndex";
    }

    //
    // SymbolicIndexOp
    //
    void SymbolicIndexOp::build(OpBuilder& builder, OperationState& result, std::string name, int id)
    {
        build(builder, result, { name, id });
    }

    void SymbolicIndexOp::build(OpBuilder& builder, OperationState& result, Index index)
    {
        result.addAttribute(getIndexAttrName(), IndexAttr::get(index, builder.getContext()));
        result.addAttribute("name", builder.getStringAttr(index.GetName()));

        auto indexType = builder.getIndexType();
        result.addTypes({ indexType });
    }

    Index SymbolicIndexOp::getValue()
    {
        return index().getValue();
    }

    mlir::OpFoldResult SymbolicIndexOp::fold(ArrayRef<mlir::Attribute> operands)
    {
        assert(operands.empty() && "sym_index has no operands");
        return index();
    }

    //
    // KernelPredicateOpInterface interface methods
    //

    std::optional<bool> NullPredicateOp::evaluate(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        return {};
    }

    Operation* NullPredicateOp::simplify(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        return getOperation();
    }

    static mlir::LogicalResult verify(NullPredicateOp op)
    {
        return mlir::success();
    }

    std::optional<bool> ConstantPredicateOp::evaluate(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        return value();
    }

    Operation* ConstantPredicateOp::simplify(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        return getOperation();
    }

    static mlir::LogicalResult verify(ConstantPredicateOp op)
    {

        return mlir::success();
    }

    std::optional<bool> FragmentTypePredicateOp::evaluate(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        // TODO: This function assumes the only scheduling operation that creates computed indices is "split". Fix it to work with arbitrary transformations.
        //       Probably, we'll want to transform this predicate into a new predicate that depends only on "loop indices"
        auto condition = fragment();
        if (condition == FragmentType::all)
        {
            return true; // do nothing for 'all' predicates
        }

        auto index = this->index().cast<IndexAttr>().getValue();

        auto loopIndices = domain.GetDependentLoopIndices(index);
        if (loopIndices.empty())
        {
            loopIndices = { index };
        }

        for (auto loopIndex : loopIndices)
        {
            auto fullRange = schedule.GetActiveLoopRange(domain, loopIndex, indices);

            std::vector<int64_t> testRangeVals;
            switch (condition)
            {
            case FragmentType::first:
                testRangeVals.push_back(fullRange.Begin());
                testRangeVals.push_back(testRangeVals[0] + 1); // single test value
                break;
            case FragmentType::select: {
                auto indexValues = getIndexValues();
                assert(indexValues.size() == 1 && "Invalid number of index values for select predicate");
                testRangeVals.push_back(indexValues[0]);
                testRangeVals.push_back(testRangeVals[0] + 1); // single test value
                break;
            }
            case FragmentType::last: {
                auto testVal = fullRange.End() - (fullRange.Size() % fullRange.Increment());
                if (testVal == fullRange.End()) // not a boundary
                {
                    testVal = fullRange.End() - fullRange.Increment();
                }
                testRangeVals.push_back(testVal);
                testRangeVals.push_back(testRangeVals[0] + 1); // single test value
                break;
            }
            case FragmentType::endBoundary: {
                auto testVal = fullRange.End() - (fullRange.Size() % fullRange.Increment());
                if (testVal != fullRange.End()) // ?
                {
                    testRangeVals.push_back(testVal);
                    testRangeVals.push_back(testRangeVals[0] + 1); // single test value
                }
                break;
            }
            case FragmentType::range: {
                auto indexValues = getIndexValues();
                assert(indexValues.size() >= 2 && "Invalid number of index values for range predicate");
                testRangeVals.push_back(indexValues[0]);
                testRangeVals.push_back(indexValues[1]);
                // increment is ignored for now
                break;
            }
            default:
                // throw?
                break;
            }

            if (testRangeVals.size() == 2)
            {
                Range testRange = { testRangeVals[0], testRangeVals[1] };

                // Look up range of the active loop
                auto activeRange = fullRange;
                if (const auto it = indices.find(loopIndex); it != indices.end())
                {
                    if (it->second.state == LoopIndexState::inProgress)
                    {
                        activeRange = it->second.loopRange;
                    }
                }

                // Now check if the test range intersects with the loop's range
                if (activeRange.Increment() == 0) // bad range
                {
                    return {};
                }
                int numIterations = CeilDiv(activeRange.End() - activeRange.Begin(), activeRange.Increment());
                if (numIterations == 0)
                {
                    return {};
                }

                if (!Intersects(activeRange, testRange))
                {
                    return false;
                }
            }
        }

        return true;
    }

    Operation* FragmentTypePredicateOp::simplify(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
#if 0
    if (_condition == Fragment::all)
            {
                return ConstantPredicate(true);
            }

            const auto index = GetIndex();
            const auto condition = GetCondition();

            // Get all index variables dependent on the predicate index
            const auto& domain = schedule.GetLoopNest().GetDomain();
            auto loopIndices = domain.GetDependentLoopIndices(index, true);

            // Evaluate a little equality "sub-predicate" for each dependent variable. All of them must be true for the result to be true.
            for (auto loopIndex : loopIndices)
            {
                // TODO: move `GetLoopRage` somewhere else
                auto fullRange = LoopNestVisitor::GetLoopRange(loopIndex, indices, schedule);

                int testVal = 0;
                bool valid = true;
                switch (condition)
                {
                case Fragment::first:
                    testVal = fullRange.Begin();
                    break;
                case Fragment::last:
                    testVal = fullRange.End() - (fullRange.Size() % fullRange.Increment());
                    if (testVal == fullRange.End()) // not a boundary
                    {
                        testVal = fullRange.End() - fullRange.Increment();
                    }
                    break;
                case Fragment::endBoundary:
                    testVal = fullRange.End() - (fullRange.Size() % fullRange.Increment());
                    if (testVal == fullRange.End()) // not a boundary
                    {
                        valid = false;
                    }
                    break;
                default:
                    valid = false;
                    // throw?
                    break;
                }

                if (valid)
                {
                    // Loop up range of the active loop
                    auto activeRange = fullRange;
                    if (const auto it = indices.find(loopIndex); it != indices.end())
                    {
                        if (it->second.state == LoopIndexState::inProgress)
                        {
                            activeRange = it->second.loopRange;
                        }
                    }

                    // Now check if testVal intersects with the loop's range
                    if (activeRange.Increment() == 0) // bad range
                    {
                        return *this;
                    }
                    int numIterations = CeilDiv(activeRange.End() - activeRange.Begin(), activeRange.Increment());
                    if (numIterations == 0)
                    {
                        return *this;
                    }

                    if (Intersects(activeRange, { testVal, testVal + 1 }))
                    {
                        if (numIterations == 1)
                        {
                            // true -- don't add anything to AND list
                        }
                        else
                        {
                            return *this;
                            // TODO: add index, testVal to AND list, later return a conjunction of equality predicates
                        }
                    }
                    else
                    {
                        return ConstantPredicate(false);
                    }
                }
            }
            return ConstantPredicate(true);
#endif
        return getOperation();
    }

    std::vector<int64_t> FragmentTypePredicateOp::getIndexValues()
    {
        return util::ConvertArrayAttrToIntVector(indexValues());
    }

    static mlir::LogicalResult verify(FragmentTypePredicateOp op)
    {
        // TODO: verify fragment is valid
        // TODO: verify index is valid
        return mlir::success();
    }

    std::optional<bool> PlacementPredicateOp::evaluate(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        throw std::runtime_error("Placement predicate not implemented");
        return {};
    }

    Operation* PlacementPredicateOp::simplify(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        return getOperation();
    }

    static mlir::LogicalResult verify(PlacementPredicateOp op)
    {
        // TODO: verify placement is valid
        // TODO: verify index is valid
        return mlir::success();
    }

    std::optional<bool> IndexDefinedPredicateOp::evaluate(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        throw std::runtime_error("IsDefined predicate not implemented");
        return {};
    }

    Operation* IndexDefinedPredicateOp::simplify(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        return getOperation();
    }

    static mlir::LogicalResult verify(IndexDefinedPredicateOp op)
    {
        // TODO: verify index is valid
        return mlir::success();
    }

    std::optional<bool> ConjunctionPredicateOp::evaluate(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        std::optional<bool> result;
        for (auto part : values())
        {
            if (auto castPredicate = dyn_cast<KernelPredicateOpInterface>(part.getDefiningOp()))
            {
                auto predicateResult = castPredicate.evaluate(domain, indices, schedule);

                // if this term is unknown, return "unknown"
                if (!predicateResult.has_value())
                    return {};

                // get the const value
                bool constResult = *predicateResult;

                // short-circuit on false
                if (!constResult)
                    return false;

                result = result.value_or(true) && constResult;
            }
            else
            {
                // throw?
                return {};
            }
        }
        return result;
    }

    Operation* ConjunctionPredicateOp::simplify(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        auto result = evaluate(domain, indices, schedule);
        if (result.has_value())
        {
            // build a constant op and return it
        }
        return getOperation();
    }

    static mlir::LogicalResult verify(ConjunctionPredicateOp op)
    {
        for (auto child : op.values())
        {
            if (!child.getType().isa<IntegerType>())
                return op.emitOpError("Error: operands must be of type i1");
            if (auto intType = child.getType().cast<IntegerType>(); intType.getWidth() != 1)
                return op.emitOpError("Error: operands must be of type i1");
        }
        return mlir::success();
    }

    std::optional<bool> DisjunctionPredicateOp::evaluate(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        std::optional<bool> result;
        for (auto part : values())
        {
            if (auto castPredicate = dyn_cast<KernelPredicateOpInterface>(part.getDefiningOp()))
            {
                auto predicateResult = castPredicate.evaluate(domain, indices, schedule);

                // if this term is unknown, return "unknown"
                if (!predicateResult.has_value())
                    return {};

                // get the const value
                bool constResult = *predicateResult;

                // short-circuit on true
                if (constResult)
                    return true;

                result = result.value_or(false) || constResult;
            }
            else
            {
                // throw?
                return {};
            }
        }
        return result;
    }

    Operation* DisjunctionPredicateOp::simplify(const TransformedDomain& domain, const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule)
    {
        auto result = evaluate(domain, indices, schedule);
        if (result.has_value())
        {
            // build a constant op and return it
        }

        return getOperation();
    }

    static mlir::LogicalResult verify(DisjunctionPredicateOp op)
    {
        for (auto child : op.values())
        {
            if (!child.getType().isa<IntegerType>())
                return op.emitOpError("Error: operands must be of type i1");
            if (auto intType = child.getType().cast<IntegerType>(); intType.getWidth() != 1)
                return op.emitOpError("Error: operands must be of type i1");
        }
        return mlir::success();
    }

    // EvaluatablePredicate interface methods
    bool ProloguePredicateOp::evaluate(const std::vector<Index>& definedIndices, const Index& currentIndex, const Position& position)
    {
        auto indexValue = index().cast<IndexAttr>().getValue();
        return currentIndex == indexValue && position == Position::prologue;
    }

    bool EpiloguePredicateOp::evaluate(const std::vector<Index>& definedIndices, const Index& currentIndex, const Position& position)
    {
        auto indexValue = index().cast<IndexAttr>().getValue();
        return currentIndex == indexValue && position == Position::epilogue;
    }

    bool IndexDefinedPredicateOp::evaluate(const std::vector<Index>& definedIndices, const Index& currentIndex, const Position& position)
    {
        auto indexValue = index().cast<IndexAttr>().getValue();
        return std::find(definedIndices.begin(), definedIndices.end(), indexValue) != definedIndices.end();
    }

    static mlir::LogicalResult verify(SymbolicIndexOp op)
    {
        return mlir::success();
    }

    //
    // Kernel predicates
    //

    static mlir::LogicalResult verify(ProloguePredicateOp op)
    {
        return mlir::success();
    }

    static mlir::LogicalResult verify(EpiloguePredicateOp op)
    {
        return mlir::success();
    }

    KernelPredicateOpInterface First(mlir::OpBuilder& builder, SymbolicIndexOp index)
    {
        auto firstAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::first));

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(index);

        auto firstIndexPred = builder.create<FragmentTypePredicateOp>(index.getLoc(), firstAttr, index);
        return dyn_cast<KernelPredicateOpInterface>(firstIndexPred.getOperation());
    }

    KernelPredicateOpInterface First(mlir::OpBuilder& builder, Index index)
    {
        auto loc = builder.getUnknownLoc();
        auto firstAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::first));
        auto firstIndexPred = builder.create<FragmentTypePredicateOp>(loc, firstAttr, index);
        return dyn_cast<KernelPredicateOpInterface>(firstIndexPred.getOperation());
    }

    KernelPredicateOpInterface Last(mlir::OpBuilder& builder, SymbolicIndexOp index)
    {
        auto lastAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::last));

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(index);

        auto lastIndexPred = builder.create<FragmentTypePredicateOp>(index.getLoc(), lastAttr, index);
        return dyn_cast<KernelPredicateOpInterface>(lastIndexPred.getOperation());
    }

    KernelPredicateOpInterface Last(mlir::OpBuilder& builder, Index index)
    {
        auto loc = builder.getUnknownLoc();
        auto lastAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::last));
        auto lastIndexPred = builder.create<FragmentTypePredicateOp>(loc, lastAttr, index);
        return dyn_cast<KernelPredicateOpInterface>(lastIndexPred.getOperation());
    }

    KernelPredicateOpInterface IndexAt(mlir::OpBuilder& builder, SymbolicIndexOp index, int64_t value)
    {
        auto selectAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::select));

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(index);

        auto selectIndexPred = builder.create<FragmentTypePredicateOp>(index.getLoc(), selectAttr, index, std::vector<int64_t>{ value });
        return dyn_cast<KernelPredicateOpInterface>(selectIndexPred.getOperation());
    }

    KernelPredicateOpInterface IndexAt(mlir::OpBuilder& builder, Index index, int64_t value)
    {
        auto loc = builder.getUnknownLoc();
        auto selectAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::select));
        auto selectIndexPred = builder.create<FragmentTypePredicateOp>(loc, selectAttr, index, std::vector<int64_t>{ value });
        return dyn_cast<KernelPredicateOpInterface>(selectIndexPred.getOperation());
    }

    KernelPredicateOpInterface EndBoundary(mlir::OpBuilder& builder, SymbolicIndexOp index)
    {
        auto endAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::endBoundary));

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(index);

        auto endIndexPred = builder.create<FragmentTypePredicateOp>(index.getLoc(), endAttr, index);
        return dyn_cast<KernelPredicateOpInterface>(endIndexPred.getOperation());
    }

    KernelPredicateOpInterface EndBoundary(mlir::OpBuilder& builder, Index index)
    {
        auto loc = builder.getUnknownLoc();
        auto endAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::endBoundary));
        auto endIndexPred = builder.create<FragmentTypePredicateOp>(loc, endAttr, index);
        return dyn_cast<KernelPredicateOpInterface>(endIndexPred.getOperation());
    }

    KernelPredicateOpInterface Before(mlir::OpBuilder& builder, SymbolicIndexOp index)
    {
        auto beforeAttr = builder.getI64IntegerAttr(static_cast<int64_t>(PlacementType::before));

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(index);

        auto beforeIndexPred = builder.create<PlacementPredicateOp>(index.getLoc(), beforeAttr, index);
        return dyn_cast<KernelPredicateOpInterface>(beforeIndexPred.getOperation());
    }

    KernelPredicateOpInterface Before(mlir::OpBuilder& builder, Index index)
    {
        auto loc = builder.getUnknownLoc();
        auto beforeAttr = builder.getI64IntegerAttr(static_cast<int64_t>(PlacementType::before));
        auto beforeIndexPred = builder.create<PlacementPredicateOp>(loc, beforeAttr, index);
        return dyn_cast<KernelPredicateOpInterface>(beforeIndexPred.getOperation());
    }

    KernelPredicateOpInterface After(mlir::OpBuilder& builder, SymbolicIndexOp index)
    {
        auto afterAttr = builder.getI64IntegerAttr(static_cast<int64_t>(PlacementType::after));

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(index);

        auto afterIndexPred = builder.create<PlacementPredicateOp>(index.getLoc(), afterAttr, index);
        return dyn_cast<KernelPredicateOpInterface>(afterIndexPred.getOperation());
    }

    KernelPredicateOpInterface After(mlir::OpBuilder& builder, Index index)
    {
        auto loc = builder.getUnknownLoc();
        auto afterAttr = builder.getI64IntegerAttr(static_cast<int64_t>(PlacementType::after));
        auto afterIndexPred = builder.create<PlacementPredicateOp>(loc, afterAttr, index);
        return dyn_cast<KernelPredicateOpInterface>(afterIndexPred.getOperation());
    }

    KernelPredicateOpInterface IsDefined(mlir::OpBuilder& builder, SymbolicIndexOp index)
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(index);

        auto pred = builder.create<IndexDefinedPredicateOp>(index.getLoc(), index);
        return dyn_cast<KernelPredicateOpInterface>(pred.getOperation());
    }

    KernelPredicateOpInterface IsDefined(mlir::OpBuilder& builder, Index index)
    {
        auto loc = builder.getUnknownLoc();
        auto pred = builder.create<IndexDefinedPredicateOp>(loc, index);
        return dyn_cast<KernelPredicateOpInterface>(pred.getOperation());
    }

    KernelPredicateOpInterface InRange(mlir::OpBuilder& builder, SymbolicIndexOp index, Range range)
    {
        auto rangeAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::range));
        auto rangeIndexPred = builder.create<FragmentTypePredicateOp>(index.getLoc(), rangeAttr, index, std::vector<int64_t>{ range.Begin(), range.End(), range.Increment() });
        return dyn_cast<KernelPredicateOpInterface>(rangeIndexPred.getOperation());
    }

    KernelPredicateOpInterface InRange(mlir::OpBuilder& builder, Index index, Range range)
    {
        auto loc = builder.getUnknownLoc();
        auto rangeAttr = builder.getI64IntegerAttr(static_cast<int64_t>(FragmentType::range));
        auto rangeIndexPred = builder.create<FragmentTypePredicateOp>(loc, rangeAttr, index, std::vector<int64_t>{ range.Begin(), range.End(), range.Increment() });
        return dyn_cast<KernelPredicateOpInterface>(rangeIndexPred.getOperation());
    }

    KernelPredicateOpInterface Conjunction(mlir::OpBuilder& builder, KernelPredicateOpInterface lhs, KernelPredicateOpInterface rhs)
    {
        assert(lhs->getBlock() == rhs->getBlock() && "Conjunction operands must be in the same block");

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(lhs->isBeforeInBlock(rhs) ? rhs : lhs);

        auto conjPred = builder.create<ConjunctionPredicateOp>(lhs.getLoc(), std::vector<KernelPredicateOpInterface>{ lhs, rhs });
        return dyn_cast<KernelPredicateOpInterface>(conjPred.getOperation());
    }

    KernelPredicateOpInterface Disjunction(mlir::OpBuilder& builder, KernelPredicateOpInterface lhs, KernelPredicateOpInterface rhs)
    {
        assert(lhs->getBlock() == rhs->getBlock() && "Conjunction operands must be in the same block");

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(lhs->isBeforeInBlock(rhs) ? rhs : lhs);

        auto disjPred = builder.create<DisjunctionPredicateOp>(lhs.getLoc(), std::vector<KernelPredicateOpInterface>{ lhs, rhs });
        return dyn_cast<KernelPredicateOpInterface>(disjPred.getOperation());
    }

    //
    // PrintOp
    //
    static mlir::LogicalResult verify(PrintOp op)
    {
        return mlir::success();
    }

    //
    // TerminatorOp
    //
    static mlir::LogicalResult verify(TerminatorOp op)
    {
        return mlir::success();
    }

    Operation* LoopNestDialect::materializeConstant(OpBuilder& builder, Attribute value, Type type, Location loc)
    {
        if (auto indexAttr = value.dyn_cast<IndexAttr>())
        {
            auto op = builder.create<SymbolicIndexOp>(loc, type, indexAttr);
            op->setAttr("name", builder.getStringAttr(indexAttr.getValue().GetName()));
            return op;
        }
        else
        {
            return builder.create<ConstantOp>(loc, type, value);
        }
    }

    //
    // Parse and print overloads
    //

    // Parse an instance of an attribute registered to the loopnest dialect.
    mlir::Attribute LoopNestDialect::parseAttribute(mlir::DialectAsmParser& parser, mlir::Type type) const
    {
        // Parse the main keyword for the attribute.
        StringRef keyword;
        if (failed(parser.parseKeyword(&keyword)))
            return {};

        if (keyword == "index")
        {
            return parseIndex(parser);
        }
        if (keyword == "indexrange")
        {
            return parseIndexRange(parser);
        }
        if (keyword == "idomain")
        {
            return parseIterationDomain(parser);
        }
        else if (keyword == "range")
        {
            return parseRange(parser);
        }
        else if (keyword == "splitindex")
        {
            return parseSplitIndex(parser);
        }
        else if (keyword == "xfdomain")
        {
            return parseTransformedDomain(parser);
        }

        parser.emitError(parser.getNameLoc(), "unknown loopnest attribute: " + keyword);
        return {};
    }

    // Print an instance of a type registered to the loopnest dialect.
    void LoopNestDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer) const
    {
        if (auto castAttr = attr.dyn_cast<IndexAttr>())
        {
            print(castAttr, printer);
        }
        else if (auto castAttr = attr.dyn_cast<IndexRangeAttr>())
        {
            print(castAttr, printer);
        }
        else if (auto castAttr = attr.dyn_cast<IterationDomainAttr>())
        {
            print(castAttr, printer);
        }
        else if (auto castAttr = attr.dyn_cast<RangeAttr>())
        {
            print(castAttr, printer);
        }
        else if (auto castAttr = attr.dyn_cast<SplitIndexAttr>())
        {
            print(castAttr, printer);
        }
        else if (auto castAttr = attr.dyn_cast<TransformedDomainAttr>())
        {
            print(castAttr, printer);
        }
    }

    mlir::Type LoopNestDialect::parseType(mlir::DialectAsmParser& parser) const
    {
        // Parse the main keyword for the attribute.
        StringRef keyword;
        if (failed(parser.parseKeyword(&keyword)))
            return {};

        if (keyword == "array")
        {
            return parseArrayType(parser);
        }
        else if (keyword == "kernel")
        {
            return parseKernelType(parser);
        }
        else if (keyword == "symbolic_index")
        {
            return parseSymbolicIndexType(parser);
        }

        parser.emitError(parser.getNameLoc(), "unknown loopnest type: " + keyword);
        return {};
    }

    void LoopNestDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const
    {
        if (auto arrayType = type.dyn_cast<ArrayType>())
        {
            print(arrayType, printer);
        }
        else if (auto kernelType = type.dyn_cast<KernelType>())
        {
            print(kernelType, printer);
        }
        else if (auto symbolicIndexType = type.dyn_cast<SymbolicIndexType>())
        {
            print(symbolicIndexType, printer);
        }
    }

} // namespace loopnest
} // namespace accera::ir

//
// TableGen'd op method definitions
//

#define GET_OP_CLASSES
#include "nest/LoopNestExportedInterfaces.cpp.inc"
#include "nest/LoopNestInterfaces.cpp.inc"
#include "nest/LoopNestOps.cpp.inc"
