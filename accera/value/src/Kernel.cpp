////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Kernel.h"
#include "MLIREmitterContext.h"
#include "ScalarIndex.h"

#include <ir/include/nest/LoopNestOps.h>

#include <utilities/include/Exception.h>

using namespace accera::ir::loopnest;

namespace accera
{
using namespace utilities;

namespace value
{
    // Implementation class
    class KernelImpl
    {
    public:
        KernelImpl(std::string id, std::function<void()> kernelFn)
        {
            auto& builder = ::accera::value::GetMLIRContext().GetOpBuilder();
            auto realKernelFn = [&](mlir::OpBuilder& bodyBuilder, mlir::Location) {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());
                kernelFn();
            };

            _op = MakeKernel(builder, id, realKernelFn);
        }

        void dump()
        {
            _op.dump();
        }

        std::vector<ScalarIndex> GetIndices()
        {
            auto indices = _op.getIndices();

            return MakeScalarIndices(indices);
        }

        KernelOp GetOp() const
        {
            return _op;
        }

    private:
        mlir::OpBuilder GetBuilder()
        {
            return mlir::OpBuilder(_op);
        }

        std::vector<ScalarIndex> MakeScalarIndices(std::vector<SymbolicIndexOp> indices)
        {
            auto context = dynamic_cast<MLIRContext*>(&GetContext());
            assert(context && "Nest only works with MLIRContext");
            std::vector<ScalarIndex> scalarIndices;
            scalarIndices.reserve(indices.size());
            for (auto index : indices)
            {
                scalarIndices.emplace_back(Wrap(index.getResult(), ScalarLayout));
                scalarIndices.back().SetName(index.getValue().GetName());
            }

            return scalarIndices;
        }

        KernelOp _op;
    };

    //
    // Main class implementation
    //

    Kernel::Kernel(std::string id, std::function<void()> kernelFn) :
        _impl(std::make_unique<KernelImpl>(id, kernelFn))
    {
    }

    Kernel::Kernel(Kernel&& other) = default;

    Kernel::~Kernel() = default;

    void Kernel::dump()
    {
        _impl->dump();
    }

    KernelOp Kernel::GetOp() const
    {
        return _impl->GetOp();
    }

    std::vector<ScalarIndex> Kernel::GetIndices() const
    {
        return _impl->GetIndices();
    }

} // namespace value
} // namespace accera
