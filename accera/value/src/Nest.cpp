////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Nest.h"
#include "Kernel.h"
#include "MLIREmitterContext.h"
#include "Schedule.h"

#include <ir/include/IRUtil.h>
#include <ir/include/nest/LoopNestOps.h>

#include <utilities/include/Exception.h>

#include <mlir/IR/Builders.h>

#include <cassert>
#include <functional>

using namespace accera::ir::loopnest;

namespace accera
{
using namespace utilities;

namespace value
{
    namespace
    {
        std::string GetIndexName(int level)
        {
            static const std::vector<std::string> names = { "i", "j", "k", "l" };
            if (level < static_cast<int>(names.size()))
            {
                return names[level];
            }

            return "idx_" + std::to_string(level);
        }

        IterationDomain MakeDomain(const MemoryShape& sizes, std::string indexPrefix)
        {
            std::vector<IndexRange> indexRanges;
            int n = 0;
            for (auto s : sizes)
            {
                indexRanges.push_back({ indexPrefix + GetIndexName(n), { 0, s } });
                ++n;
            }
            return { indexRanges };
        }

        IterationDomain MakeDomain(const std::vector<Range>& ranges, std::string indexPrefix)
        {
            std::vector<IndexRange> indexRanges;
            int n = 0;
            for (auto r : ranges)
            {
                indexRanges.push_back({ indexPrefix + GetIndexName(n), r });
                ++n;
            }
            return { indexRanges };
        }

    } // namespace

    // Implementation class
    class NestImpl
    {
    public:
        NestImpl(mlir::OpBuilder& builder, const IterationDomain& domain, const std::vector<ScalarDimension>& runtimeSizes)
        {
            std::vector<mlir::Value> sizes;
            std::transform(runtimeSizes.cbegin(), runtimeSizes.cend(), std::back_inserter(sizes), [](ScalarDimension d) { return Unwrap(d); });

            _op = MakeNest(builder, domain, sizes);
        }

        ScalarIndex GetIndex(int pos)
        {
            [[maybe_unused]] auto context = dynamic_cast<MLIRContext*>(&GetContext());
            assert(context && "Nest only works with MLIRContext");
            auto builder = GetBuilder();
            auto indices = _op.getIndices(builder);
            return MakeScalarIndex(indices[pos]);
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

        IterationDomain GetDomain() const
        {
            return const_cast<NestOp&>(_op).getDomain().getValue();
        }

        void Set(KernelOp kernelOp)
        {
            _op.getOrCreateSchedule().addKernel(kernelOp);
        }

        void dump()
        {
            _op.dump();
        }

        NestOp GetOp()
        {
            return _op;
        }

        mlir::OpBuilder GetBuilder()
        {
            return _op.getBodyBuilder();
        }

    private:
        static Index GetIndexFromValue(ScalarIndex val)
        {
            if (val.GetValue().IsEmpty())
            {
                return {};
            }
            auto mlirValue = Unwrap(val);
            auto op = mlirValue.getDefiningOp();
            auto indexOp = llvm::dyn_cast_or_null<SymbolicIndexOp>(op);
            if (indexOp)
            {
                return indexOp.getValue();
            }
            return {};
        }

        ScalarIndex MakeScalarIndex(SymbolicIndexOp i)
        {
            return Wrap(i.getResult(), ScalarLayout);
        }

        NestOp _op;
    };

    //
    // Main class implementation
    //

    Nest::Nest(const utilities::MemoryShape& sizes, const std::vector<ScalarDimension>& runtimeSizes) :
        Nest(MakeDomain(sizes, ""), runtimeSizes)
    {}

    Nest::Nest(const std::vector<Range>& ranges, const std::vector<ScalarDimension>& runtimeSizes) :
        Nest(MakeDomain(ranges, ""), runtimeSizes)
    {
    }

    Nest::Nest(const IterationDomain& domain, const std::vector<ScalarDimension>& runtimeSizes) :
        _impl(std::make_unique<NestImpl>(::accera::value::GetMLIRContext().GetOpBuilder(), domain, runtimeSizes))
    {
    }

    Nest::Nest(Nest&& other) = default;

    Nest::~Nest() = default;

    ScalarIndex Nest::GetIndex(int pos)
    {
        return _impl->GetIndex(pos);
    }

    std::vector<ScalarIndex> Nest::GetIndices()
    {
        return _impl->GetIndices();
    }

    IterationDomain Nest::GetDomain() const
    {
        return _impl->GetDomain();
    }

    void Nest::Set(std::function<void()> kernelFn)
    {
        int64_t count = accera::ir::util::GetUniqueId();
        Kernel k("body_" + std::to_string(count), kernelFn);
        CreateSchedule().AddKernel(k);
    }

    Schedule Nest::CreateSchedule()
    {
        return { *this };
    }

    void Nest::dump()
    {
        _impl->dump();
    }

    NestOp Nest::GetOp()
    {
        return _impl->GetOp();
    }

} // namespace value
} // namespace accera
