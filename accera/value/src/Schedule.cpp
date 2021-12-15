////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Schedule.h"
#include "Kernel.h"
#include "KernelPredicate.h"
#include "MLIREmitterContext.h"
#include "Nest.h"
#include "Plan.h"

#include <ir/include/nest/LoopNestOps.h>

#include <utilities/include/Exception.h>

#include <mlir/IR/BlockAndValueMapping.h>

#include <cassert>
#include <functional>
#include <ir/include/IRUtil.h>
#include <llvm/ADT/SetVector.h>

using namespace accera::ir::loopnest;

namespace accera
{
using namespace utilities;

namespace value
{
    // Implementation class
    class ScheduleImpl
    {
    public:
        ScheduleImpl(NestOp nestOp)
        {
            // auto& builder = GetBuilder();
            _op = nestOp.getOrCreateSchedule();
        }

        std::vector<ScalarIndex> GetIndices()
        {
            [[maybe_unused]] auto context = dynamic_cast<MLIRContext*>(&GetContext());
            assert(context && "Nest only works with MLIRContext");
            auto builder = GetBuilder();
            auto indices = _op.getIndices(builder);
            std::vector<ScalarIndex> result;
            for (auto i : indices)
            {
                result.push_back(MakeScalarIndex(i));
            }
            return result;
        }

        SplitIndex Split(Index i, int factor)
        {
            return _op.split(i, factor);
        }

        ScalarIndexPair Split(ScalarIndex i, int factor)
        {
            auto indices = _op.split(GetIndexOp(i), factor);
            ScalarIndexPair result;
            result.first = MakeScalarIndex(indices.outer);
            result.second = MakeScalarIndex(indices.inner);

            return result;
        }

        Index Pad(Index i, int size)
        {
            return _op.pad(i, size);
        }

        ScalarIndex Pad(ScalarIndex i, int size)
        {
            auto paddedIndex = _op.pad(GetIndexOp(i), size);
            return MakeScalarIndex(paddedIndex);
        }

        Index Skew(Index i, Index reference)
        {
            return _op.skew(i, reference);
        }

        ScalarIndex Skew(ScalarIndex i, ScalarIndex reference)
        {
            auto skewedIndex = _op.skew(GetIndexOp(i), GetIndexOp(reference));
            return MakeScalarIndex(skewedIndex);
        }

        void Unroll(Index i, std::optional<uint64_t> size)
        {
            _op.unroll(i, /*unroll=*/true, size);
        }

        void Unroll(ScalarIndex i, std::optional<uint64_t> size)
        {
            _op.unroll(GetValueIndex(i), /*unroll=*/true, size);
        }

        void InterleavedUnroll(Index i, uint64_t interleaveFactor)
        {
            _op.unrollAndJam(i, interleaveFactor);
        }

        void InterleavedUnroll(ScalarIndex i, uint64_t interleaveFactor)
        {
            InterleavedUnroll(GetValueIndex(i), interleaveFactor);
        }

        void SetOrder(std::vector<Index> order)
        {
            _op.setOrder(order);
        }

        void SetOrder(std::vector<ScalarIndex> order)
        {
            std::vector<Index> indexOrder;
            for (auto i : order)
            {
                indexOrder.push_back(GetValueIndex(i));
            }
            SetOrder(indexOrder);
        }

        void AddKernel(KernelOp kernel)
        {
            auto builder = GetBuilder();

            kernel.getOperation()->moveBefore(_op);
            auto loc = kernel.getLoc();
            KernelPredicateOpInterface nullPred = builder.create<NullPredicateOp>(loc);
            std::string id = "scheduled_" + kernel.getId().str();
            auto scheduledKernel = builder.create<ScheduledKernelOp>(loc, id, kernel, nullPred);

            _op.addKernel(scheduledKernel);
        }

        void AddKernel(KernelOp kernel, const KernelPredicate& pred)
        {
            auto builder = GetBuilder();

            kernel.getOperation()->moveBefore(_op);

            auto loc = kernel.getLoc();
            std::string id = "scheduled_" + kernel.getId().str();
            mlir::BlockAndValueMapping mapping;
            auto predicate = ir::util::CloneRecursively(builder, pred.GetOp(), mapping);
            auto scheduledKernel = builder.create<ScheduledKernelOp>(loc, id, kernel, mlir::dyn_cast<KernelPredicateOpInterface>(predicate));

            _op.addKernel(scheduledKernel);
        }

        void AddKernel(KernelOp kernel, const KernelPredicate& pred, const KernelPredicate& placement)
        {
            auto builder = GetBuilder();

            kernel.getOperation()->moveBefore(_op);

            auto loc = kernel.getLoc();
            std::string id = "scheduled_" + kernel.getId().str();
            mlir::BlockAndValueMapping mapping;
            auto predicate = ir::util::CloneRecursively(builder, pred.GetOp(), mapping);
            auto placementPredicate = ir::util::CloneRecursively(builder, placement.GetOp(), mapping);
            auto scheduledKernel = builder.create<ScheduledKernelOp>(loc, id, kernel, predicate, placementPredicate);

            _op.addKernel(scheduledKernel);
        }

        ScalarIndex Fuse(std::vector<ScheduleImpl>& others, std::vector<std::vector<ScalarIndex>> correspondences)
        {
            auto nest = GetOp()->getParentOfType<NestOp>();
            mlir::OpBuilder builder(nest);

            std::vector<std::vector<Index>> indexCorrespondences;
            indexCorrespondences.reserve(correspondences.size());

            std::transform(
                correspondences.begin(),
                correspondences.end(),
                std::back_inserter(indexCorrespondences),
                [this](const std::vector<ScalarIndex>& correspondence) {

                    std::vector<Index> valueIndices;
                    valueIndices.reserve(correspondence.size());

                    std::transform(
                        correspondence.begin(),
                        correspondence.end(),
                        std::back_inserter(valueIndices),
                        [this](const ScalarIndex& idx) {
                            return GetValueIndex(idx);
                            });

                    return valueIndices;
                });

            std::vector<ScheduleOp> schedules{ GetOp() };
            schedules.reserve(others.size() + 1);
            std::transform(
                others.begin(),
                others.end(),
                std::back_inserter(schedules),
                [](ScheduleImpl& other) { return other.GetOp(); });

            auto [newOp, fusingIndex] = ir::loopnest::Fuse(builder, schedules, indexCorrespondences);

            for (auto& other : others)
            {
                other._op = {};
            }
            _op = newOp;
            auto indexOp = _op.getOrCreateSymbolicIndex(builder, fusingIndex);
            return MakeScalarIndex(indexOp);
        }

        ScheduleOp GetOp() const
        {
            return _op;
        }

        void dump()
        {
            _op.dump();
        }

        Index GetValueIndex(ScalarIndex val)
        {
            return GetOpIndex(GetIndexOp(val));
        }

        ScalarIndex GetIndexValue(Index index)
        {
            auto nest = GetOp()->getParentOfType<NestOp>();
            mlir::OpBuilder builder(nest);
            auto indexOp = _op.getOrCreateSymbolicIndex(builder, index);
            return MakeScalarIndex(indexOp);
        }

        void SetSaturatedFlag(Index i)
        {
            _op.setSaturatedFlag(i);
        }

        void SetSaturatedFlag(ScalarIndex i)
        {
            _op.setSaturatedFlag(GetValueIndex(i));
        }

        ScheduleImpl(ScheduleOp op) :
            _op(op)
        {
        }

    private:
        mlir::OpBuilder GetBuilder()
        {
            return mlir::OpBuilder(_op);
        }

        SymbolicIndexOp GetIndexOp(ScalarIndex val)
        {
            if (val.GetValue().IsUndefined())
            {
                return {}; // return a null index
            }

            auto mlirValue = mlir::Value::getFromOpaquePointer(val.GetValue().Get<Emittable>().GetDataAs<MLIRContext::EmittableInfo*>()->data);
            auto op = mlirValue.getDefiningOp();
            assert(op);

            auto indexOp = llvm::dyn_cast_or_null<SymbolicIndexOp>(op);
            assert(indexOp);
            return indexOp;
        }

        static ScalarIndex MakeScalarIndex(SymbolicIndexOp i)
        {
            return Wrap(i.getResult(), ScalarLayout);
        }

        Index GetOpIndex(SymbolicIndexOp op)
        {
            if (!op)
            {
                return Index::none;
            }
            return op.getValue();
        }

        ScheduleOp _op;
    };

    //
    // Main class implementation
    //

    Schedule::Schedule(const Schedule& other) :
        Schedule(other.GetOp())
    {
    }

    Schedule::Schedule(Schedule&&) noexcept = default;

    Schedule& Schedule::operator=(const Schedule& other) {

        return *this;
    }

    Schedule& Schedule::operator=(Schedule&& other) noexcept {

        return *this;
    }

    Schedule::Schedule(Nest& nest) :
        _impl(std::make_unique<ScheduleImpl>(nest.GetOp()))
    {}

    Schedule::Schedule(ScheduleOp op) :
        _impl(std::make_unique<ScheduleImpl>(op))
    {}

    Schedule::~Schedule() = default;

    ScheduleImpl& Schedule::GetImpl()
    {
        return *_impl;
    }

    std::vector<ScalarIndex> Schedule::GetIndices()
    {
        return _impl->GetIndices();
    }

    SplitIndex Schedule::Split(Index i, int factor)
    {
        return _impl->Split(i, factor);
    }

    ScalarIndexPair Schedule::Split(ScalarIndex i, int factor)
    {
        return _impl->Split(i, factor);
    }

    Index Schedule::Pad(Index i, int size)
    {
        return _impl->Pad(i, size);
    }
    ScalarIndex Schedule::Pad(ScalarIndex i, int size)
    {
        return _impl->Pad(i, size);
    }

    Index Schedule::Skew(Index i, Index reference)
    {
        return _impl->Skew(i, reference);
    }
    ScalarIndex Schedule::Skew(ScalarIndex i, ScalarIndex reference)
    {
        return _impl->Skew(i, reference);
    }

    void Schedule::Unroll(Index i, std::optional<uint64_t> size)
    {
        return _impl->Unroll(i, size);
    }
    void Schedule::Unroll(ScalarIndex i, std::optional<uint64_t> size)
    {
        return _impl->Unroll(i, size);
    }

    void Schedule::InterleavedUnroll(Index i, uint64_t factor)
    {
        return _impl->InterleavedUnroll(i, factor);
    }

    void Schedule::InterleavedUnroll(ScalarIndex i, uint64_t factor)
    {
        return _impl->InterleavedUnroll(i, factor);
    }

    void Schedule::SetSaturatedFlag(Index i)
    {
        return _impl->SetSaturatedFlag(i);
    }

    void Schedule::SetSaturatedFlag(ScalarIndex i)
    {
        return _impl->SetSaturatedFlag(i);
    }

    void Schedule::SetOrder(std::vector<Index> order)
    {
        return _impl->SetOrder(order);
    }

    void Schedule::SetOrder(std::vector<ScalarIndex> order)
    {
        return _impl->SetOrder(order);
    }

    void Schedule::AddKernel(const Kernel& kernel)
    {
        return _impl->AddKernel(kernel.GetOp());
    }

    void Schedule::AddKernel(const Kernel& kernel, const KernelPredicate& predicate)
    {
        return _impl->AddKernel(kernel.GetOp(), predicate);
    }

    void Schedule::AddKernel(const Kernel& kernel, const KernelPredicate& predicate, const KernelPredicate& placement)
    {
        return _impl->AddKernel(kernel.GetOp(), predicate, placement);
    }

    ScalarIndex Schedule::Fuse(std::vector<Schedule>& others, const std::vector<std::vector<ScalarIndex>>& correspondences)
    {
        std::vector<ScheduleImpl> impls;
        impls.reserve(others.size());

        std::transform(others.begin(), others.end(), std::back_inserter(impls),[](Schedule& other){return other.GetImpl();});

        return _impl->Fuse(impls, correspondences);
    }

    Plan Schedule::CreatePlan()
    {
        return { *this };
    }

    GPUPlan Schedule::CreateGPUPlan(targets::GPU gpuOptions)
    {
        return { gpuOptions, *this };
    }

    ScheduleOp Schedule::GetOp() const
    {
        return _impl->GetOp();
    }

    void Schedule::dump()
    {
        _impl->dump();
    }

    Index Schedule::ResolveIndex(ScalarIndex index)
    {
        return _impl->GetValueIndex(index);
    }

    ScalarIndex Schedule::LookUpIndex(Index index)
    {
        return _impl->GetIndexValue(index);
    }

} // namespace value
} // namespace accera
