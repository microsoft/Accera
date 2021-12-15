////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "KernelPredicate.h"
#include "MLIREmitterContext.h"

#include <ir/include/nest/LoopNestOps.h>

#include <utilities/include/Exception.h>

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
        SymbolicIndexOp GetIndexOp(ScalarIndex val)
        {
            auto mlirValue = mlir::Value::getFromOpaquePointer(val.GetValue().Get<Emittable>().GetDataAs<MLIRContext::EmittableInfo*>()->data);
            auto op = mlirValue.getDefiningOp();
            assert(op);

            auto indexOp = llvm::dyn_cast<SymbolicIndexOp>(op);
            assert(indexOp);
            return indexOp;
        }

    } // namespace

    // Implementation class
    class KernelPredicateImpl
    {
    public:
        KernelPredicateImpl()
        {
            auto builder = ::accera::value::GetMLIRContext().GetOpBuilder();
            auto loc = builder.getUnknownLoc();
            _op = mlir::dyn_cast<accera::ir::loopnest::KernelPredicateOpInterface>(builder.create<NullPredicateOp>(loc).getOperation());
        }

        KernelPredicateImpl(accera::ir::loopnest::KernelPredicateOpInterface predicate)
        {
            _op = predicate;
        }

        void dump()
        {
            _op.dump();
        }

        accera::ir::loopnest::KernelPredicateOpInterface GetOp() const
        {
            return _op;
        }

    private:
        mlir::OpBuilder GetBuilder()
        {
            return mlir::OpBuilder(_op);
        }

        accera::ir::loopnest::KernelPredicateOpInterface _op;
    };

    //
    // Main class implementation
    //

    KernelPredicate::KernelPredicate() :
        _impl(std::make_unique<KernelPredicateImpl>())
    {
    }

    KernelPredicate::KernelPredicate(const accera::ir::loopnest::KernelPredicateOpInterface& predicate) :
        _impl(std::make_unique<KernelPredicateImpl>(predicate))
    {
    }

    KernelPredicate::KernelPredicate(KernelPredicate&& other) = default;

    KernelPredicate::~KernelPredicate() = default;

    void KernelPredicate::dump()
    {
        _impl->dump();
    }

    accera::ir::loopnest::KernelPredicateOpInterface KernelPredicate::GetOp() const
    {
        return _impl->GetOp();
    }

    //
    // Helper functions
    //
    KernelPredicate First(ScalarIndex index)
    {
        // Get the SymbolicIndexOp held by `index`
        auto& builder = ::accera::value::GetMLIRContext().GetOpBuilder();
        auto indexOp = GetIndexOp(index);
        auto pred = First(builder, indexOp);
        return { pred };
    }

    KernelPredicate Last(ScalarIndex index)
    {
        // Get the SymbolicIndexOp held by `index`
        auto& builder = ::accera::value::GetMLIRContext().GetOpBuilder();
        auto indexOp = GetIndexOp(index);
        auto pred = Last(builder, indexOp);
        return { pred };
    }

    KernelPredicate EndBoundary(ScalarIndex index)
    {
        // Get the SymbolicIndexOp held by `index`
        auto& builder = ::accera::value::GetMLIRContext().GetOpBuilder();
        auto indexOp = GetIndexOp(index);
        auto pred = EndBoundary(builder, indexOp);
        return { pred };
    }

    KernelPredicate Before(ScalarIndex index)
    {
        // Get the SymbolicIndexOp held by `index`
        auto& builder = ::accera::value::GetMLIRContext().GetOpBuilder();
        auto indexOp = GetIndexOp(index);
        auto pred = Before(builder, indexOp);
        return { pred };
    }

    KernelPredicate After(ScalarIndex index)
    {
        // Get the SymbolicIndexOp held by `index`
        auto& builder = ::accera::value::GetMLIRContext().GetOpBuilder();
        auto indexOp = GetIndexOp(index);
        auto pred = After(builder, indexOp);
        return { pred };
    }

    KernelPredicate IsDefined(ScalarIndex index)
    {
        // Get the SymbolicIndexOp held by `index`
        auto& builder = ::accera::value::GetMLIRContext().GetOpBuilder();
        auto indexOp = GetIndexOp(index);
        auto pred = IsDefined(builder, indexOp);
        return { pred };
    }

    KernelPredicate operator&&(const KernelPredicate& lhs, const KernelPredicate& rhs)
    {
        auto& builder = ::accera::value::GetMLIRContext().GetOpBuilder();
        return { Conjunction(builder, lhs.GetOp(), rhs.GetOp()) };
    }

    KernelPredicate operator||(const KernelPredicate& lhs, const KernelPredicate& rhs)
    {
        auto& builder = ::accera::value::GetMLIRContext().GetOpBuilder();
        return { Disjunction(builder, lhs.GetOp(), rhs.GetOp()) };
    }

} // namespace value
} // namespace accera
