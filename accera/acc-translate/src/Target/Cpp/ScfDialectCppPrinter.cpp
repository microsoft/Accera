////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ScfDialectCppPrinter.h"

using namespace mlir::scf;

namespace mlir
{
namespace cpp_printer
{

    LogicalResult ScfDialectCppPrinter::printIfOp(IfOp ifOp)
    {
        Operation* op = ifOp.getOperation();

        auto numResults = op->getNumResults();
        // IfOp returns values from its regions, so we create variables
        // before outside of the blocks to hold the return values
        if (numResults)
        {
            for (auto ret : op->getResults())
            {
                RETURN_IF_FAILED(printer->printType(ret.getType()));
                os << " "
                   << state.nameState.getOrCreateName(ret,
                                                      SSANameState::SSANameKind::Variable)
                   << ";\n";
            }
        }

        // We control the parentheses instead of passing down to printRegion,
        // because we want to include yield-to-retval assignments within
        // the ``then'' and ``else'' blocks.
        os << "if (" << state.nameState.getName(ifOp.condition()) << ") {\n";
        auto& thenRegion = ifOp.thenRegion();
        // If we have yield block terminators, we will treat them specially
        // because we will need to assign yielded values to retNames that
        // we just created. So, we ask printBlock to skip the terminators.
        RETURN_IF_FAILED(printer->printRegion(thenRegion, /*printParens*/ false,
                                              /*printBlockTerminator*/ false));
        // We take the op's results if there are any
        if (numResults)
        {
            auto e = thenRegion.getBlocks().front().getOperations().end();
            RETURN_IF_FAILED(printYieldOp<Operation::result_range>(
                dyn_cast<YieldOp>(--e), op->getResults()));
        }
        os << "}\n";

        auto& elseRegion = ifOp.elseRegion();
        if (!elseRegion.empty())
        {
            os << "else {\n";
            RETURN_IF_FAILED(printer->printRegion(elseRegion, /*printParens*/ false,
                                                  /*printBlockTerminator*/ false));
            if (numResults)
            {
                auto e = elseRegion.getBlocks().front().getOperations().end();
                RETURN_IF_FAILED(printYieldOp<Operation::result_range>(
                    dyn_cast<YieldOp>(--e), op->getResults()));
            }
            os << "}\n";
        }

        return success();
    }

    LogicalResult ScfDialectCppPrinter::printForOp(ForOp forOp)
    {
        // Create variables for loop-carried variables if they exist.
        // In fact, we create two sets of variables. One set is for holding
        // the actual values, and the other just has references to the first
        // set. Basically, we introduce this indirect reference to make SSA
        // name-loopup easy, i.e. we can simplify look for the argument inside
        // the loop instead of indirectly searching for ret values through the
        // argument.
        // For example, for a ForOp such as:
        //   %ret = scf.for %i = %0 to %n step %s iter_args(%p = %v) -> (f32)
        // we would end up generating some C++ code like below:
        //   float ret = v;
        //   float &loop_p = ret;
        //   for (int64_t i = v0; i < v_n; i += v_s) {
        //     ...
        //     loop_p = xyz; // where xyz is from the corresponding YeildOp
        //   }
        if (forOp.hasIterOperands())
        {
            auto results = forOp.getResults();
            auto initValues = forOp.getIterOperands();
            auto args = forOp.getRegionIterArgs();
            for (auto e : llvm::zip(results, args, initValues))
            {
                auto ret = std::get<0>(e);
                auto tp = ret.getType();

                StringRef retName = state.nameState.getOrCreateName(
                    ret, SSANameState::SSANameKind::Variable);
                StringRef argName = state.nameState.getOrCreateName(
                    std::get<1>(e), SSANameState::SSANameKind::Variable);
                StringRef valName = state.nameState.getName(std::get<2>(e));

                RETURN_IF_FAILED(printer->printType(tp));
                os << " " << retName << " = " << valName << ";\n";

                RETURN_IF_FAILED(printer->printType(tp));
                os << " &" << argName << " = " << retName << ";\n";
            }
        }

        auto idx = forOp.getInductionVar();
        auto idxType = idx.getType();
        StringRef idxName =
            state.nameState.getOrCreateName(idx, SSANameState::SSANameKind::LoopIdx);
        StringRef lowerBoundName = state.nameState.getOrCreateName(
            forOp.lowerBound(), SSANameState::SSANameKind::Variable);
        StringRef upperBoundName = state.nameState.getOrCreateName(
            forOp.upperBound(), SSANameState::SSANameKind::Variable);
        StringRef stepName = state.nameState.getOrCreateName(
            forOp.step(), SSANameState::SSANameKind::Variable);

        // The ForOp has been canonicalized to allow us to safely generate
        // canonicalized for loops as well, like below:
        //   for (int64_t i = lower; i < upper; i += step) { ... }
        os << "for (";
        RETURN_IF_FAILED(printer->printType(idxType));
        os << " " << idxName << " = " << lowerBoundName << "; ";
        os << idxName << " < " << upperBoundName << "; ";
        os << idxName << " += " << stepName << ") {\n";

        auto& loopRegion = forOp.region();
        // We generate parentheses here instead of passinig down to printRegion,
        // because we will append assignments for YieldOp. Similarly, we ask
        // printRegion to skip block terminators, i.e. YieldOp
        RETURN_IF_FAILED(printer->printRegion(loopRegion, /*printParens*/ false,
                                              /*printBlockTerminator*/ false));
        // assign yielded values to loop-carried variables if there are any
        if (forOp.hasIterOperands())
        {
            auto e = loopRegion.getBlocks().front().getOperations().end();
            RETURN_IF_FAILED(printYieldOp<Block::BlockArgListType>(
                dyn_cast<YieldOp>(--e), forOp.getRegionIterArgs()));
        }

        // end of the for loop block
        os << "}\n";
        return success();
    }

    template <typename RangeT>
    LogicalResult ScfDialectCppPrinter::printYieldOp(YieldOp yieldOp,
                                                     RangeT retValues)
    {
        if (!yieldOp)
        {
            os << "Not a YieldOp!";
            return failure();
        }

        auto numOperands = yieldOp.getNumOperands();
        if (numOperands != retValues.size())
        {
            return yieldOp.emitOpError()
                   << "Number of yielded values doesn't match the number of retNames!";
        }

        for (auto e : llvm::zip(retValues, yieldOp.getOperands()))
        {
            auto& ret = std::get<0>(e);
            auto& yielded = std::get<1>(e);
            // make sure yielded value's type matches its receiver's
            if (ret.getType() != yielded.getType())
            {
                return yieldOp.emitOpError()
                       << "yielded value's type doesn't match its receiver";
            }

            os << state.nameState.getName(ret) << " = "
               << state.nameState.getName(yielded) << ";\n";
        }

        return success();
    }

    LogicalResult ScfDialectCppPrinter::printDialectOperation(Operation* op,
                                                              bool* skipped,
                                                              bool* consumed)
    {
        *consumed = true;

        if (auto forOp = dyn_cast<ForOp>(op))
        {
            *skipped = true;
            return printForOp(forOp);
        }

        if (auto ifOp = dyn_cast<IfOp>(op))
        {
            *skipped = true;
            return printIfOp(ifOp);
        }

        *consumed = false;
        return success();
    }

} // namespace cpp_printer
} // namespace mlir
