////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AffineDialectCppPrinter.h"

#include <llvm/ADT/Sequence.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LogicalResult.h>

#include <numeric>
namespace
{

using namespace mlir;

struct AffineMapVisitor
{
    explicit AffineMapVisitor(llvm::DenseMap<AffineMap, std::string>& mapToFunc) :
        affineMapToFuncBaseName(mapToFunc) {}

    void visit(Type type);

    void visit(Attribute attr);

    llvm::DenseMap<AffineMap, std::string>& affineMapToFuncBaseName;

    DenseSet<Type> visitedTypes;

    DenseSet<Attribute> visitedAttrs;
};

void AffineMapVisitor::visit(Type type)
{
    if (visitedTypes.count(type))
        return;
    visitedTypes.insert(type);

    if (auto funcType = type.dyn_cast<FunctionType>())
    {
        for (auto input : funcType.getInputs())
            visit(input);
        for (auto result : funcType.getResults())
            visit(result);
    }
    else if (auto memRefType = type.dyn_cast<MemRefType>())
    {
        for (auto m : memRefType.getAffineMaps())
        {
            visit(AffineMapAttr::get(m));
        }
    }
    else if (auto shapedType = type.dyn_cast<ShapedType>())
    {
        visit(shapedType.getElementType());

        if (auto memRefType = type.dyn_cast<MemRefType>())
        {
            visit(memRefType);
        }
    }
}

std::string makeAffineMapFuncName()
{
    static int cnt = 0;
    return mlir::cpp_printer::AffineDialectCppPrinter::affineMapFuncPrefix +
           std::to_string(cnt++);
}

void AffineMapVisitor::visit(Attribute attr)
{
    if (visitedAttrs.count(attr))
        return;
    visitedAttrs.insert(attr);

    if (auto arrayAttr = attr.dyn_cast<ArrayAttr>())
    {
        for (Attribute element : arrayAttr.getValue())
        {
            visit(element);
        }
    }
    else if (auto typeAttr = attr.dyn_cast<TypeAttr>())
    {
        visit(typeAttr.getValue());
    }
    else if (auto mapAttr = attr.dyn_cast<AffineMapAttr>())
    {
        affineMapToFuncBaseName.try_emplace(mapAttr.getValue(),
                                            makeAffineMapFuncName());
    }
}

} // end anonymous namespace

namespace mlir
{
namespace cpp_printer
{

    static std::string makeAffineDimName(int idx)
    {
        return "d" + std::to_string(idx);
    }

    static std::string makeAffineSymName(int idx)
    {
        return "s" + std::to_string(idx);
    }

    LogicalResult AffineDialectCppPrinter::printAffineExpr(AffineExpr affineExpr)
    {
        if (auto symExpr = affineExpr.dyn_cast<AffineSymbolExpr>())
        {
            os << makeAffineSymName(symExpr.getPosition());
            return success();
        }

        if (auto dimExpr = affineExpr.dyn_cast<AffineDimExpr>())
        {
            os << makeAffineDimName(dimExpr.getPosition());
            return success();
        }

        if (auto constExpr = affineExpr.dyn_cast<AffineConstantExpr>())
        {
            os << constExpr.getValue();
            return success();
        }

        auto binOp = affineExpr.cast<AffineBinaryOpExpr>();
        AffineExpr lhs = binOp.getLHS();
        AffineExpr rhs = binOp.getRHS();
        auto kind = binOp.getKind();

        if (kind == AffineExprKind::CeilDiv)
        {
            os << affineCeilDivStr << "(";
            RETURN_IF_FAILED(printAffineExpr(lhs));
            os << ", ";
            RETURN_IF_FAILED(printAffineExpr(rhs));
            os << ")";
            return success();
        }

        const char* binOpStr = nullptr;
        switch (kind)
        {
        case AffineExprKind::Add:
            binOpStr = "+";
            break;
        case AffineExprKind::Mul:
            binOpStr = "*";
            break;
        case AffineExprKind::FloorDiv:
            binOpStr = "/";
            break;
        case AffineExprKind::Mod:
            binOpStr = "%";
            break;
        default:
            llvm_unreachable("Unexpected");
        }

        os << "(";
        RETURN_IF_FAILED(printAffineExpr(lhs));
        os << " " << binOpStr << " ";
        RETURN_IF_FAILED(printAffineExpr(rhs));
        os << ")";

        return success();
    }

    // print functions that compute the actual indices by applying affine indices
    // to the affine expressions from the AffineMap
    LogicalResult
    AffineDialectCppPrinter::printAffineMapFunc(AffineMap map,
                                                StringRef funcBaseName)
    {
        // We generate one function for each computed index
        for (int resIdx = 0; resIdx < (int)(map.getNumResults()); resIdx++)
        {
            os << "__forceinline__ " << printer->deviceAttrIfCuda() << "\n";
            RETURN_IF_FAILED(printer->printIndexType());
            os << " " << makeAffineIdxFuncName(funcBaseName, resIdx)
               << "(";
            int numDims = (int)(map.getNumDims());
            int numSyms = (int)(map.getNumSymbols());
            if (numDims > 0)
            {
                int dimIdx = 0;
                for (; dimIdx < numDims - 1; dimIdx++)
                {
                    RETURN_IF_FAILED(printer->printIndexType());
                    os << " " << makeAffineDimName(dimIdx) << ", ";
                }
                RETURN_IF_FAILED(printer->printIndexType());
                os << " " << makeAffineDimName(dimIdx);
                if (numSyms > 0)
                    os << ", ";
            }

            if (numSyms > 0)
            {
                int symIdx = 0;
                for (; symIdx < numSyms - 1; symIdx++)
                {
                    RETURN_IF_FAILED(printer->printIndexType());
                    os << " " << makeAffineSymName(symIdx) << ", ";
                }
                RETURN_IF_FAILED(printer->printIndexType());
                os << " " << makeAffineSymName(symIdx);
            }

            os << ") {\n";

            const char* idxName = "idx";
            RETURN_IF_FAILED(printer->printIndexType());
            os << " " << idxName << " = ";
            RETURN_IF_FAILED(printAffineExpr(map.getResult(resIdx)));
            os << ";\n";
            os << "return " << idxName << ";\n";

            os << "}\n\n";
        }
        return success();
    }

    LogicalResult AffineDialectCppPrinter::printDeclarations()
    {
        // print affine-map funcs in order (i.e. from 0 to the end)
        llvm::SmallVector<std::pair<AffineMap, StringRef>, 0> sortedMapFuncs;
        sortedMapFuncs.resize(affineMapToFuncBaseName.size());
        unsigned prefixLen = strlen(AffineDialectCppPrinter::affineMapFuncPrefix);
        for (auto const& e : affineMapToFuncBaseName)
        {
            StringRef idxStr = StringRef(e.second).drop_front(prefixLen);
            size_t idx;
            bool failed = idxStr.getAsInteger(/*radix*/ 10, idx);
            assert(!failed && "invalid map func name");
            assert(idx < sortedMapFuncs.size());
            if (failed || idx >= sortedMapFuncs.size())
                return failure();
            sortedMapFuncs[idx] = { e.first, e.second };
        }

        for (auto p : sortedMapFuncs)
        {
            RETURN_IF_FAILED(printAffineMapFunc(p.first, p.second));
        }
        return success();
    }

    // generate the actual indices used for accessing the memref for the given
    // AffineMap and its input indices.
    void AffineDialectCppPrinter::printAffineMapResultIndices(
        AffineMap map,
        Operation::operand_range origIndices,
        SmallVector<StringRef, 4>& memIdxVars)
    {
        // make the string for constructing the affine-map function arguments
        std::string affineFuncArgs;
        llvm::raw_string_ostream tmpOs(affineFuncArgs);
        interleaveComma(origIndices, tmpOs, [&](Value operand) {
            tmpOs << state.nameState.getName(operand);
        });

        StringRef funcBaseName = getFuncBaseName(map);
        // making a call for computing each result index
        for (size_t idx = 0; idx < map.getNumResults(); idx++)
        {
            std::string idxFuncName = makeAffineIdxFuncName(funcBaseName, idx);
            auto idxVarName = state.nameState.getTempName();
            THROW_IF_FAILED(printer->printIndexType());
            os << " " << idxVarName << " = " << idxFuncName << "("
               << affineFuncArgs << ");\n";
            memIdxVars.push_back(idxVarName);
        }
    }

    // print out the pointer that point to the MemRef location indexed by memIdxVars
    LogicalResult AffineDialectCppPrinter::printMemRefAccessPtr(
        Value memRef,
        const SmallVector<StringRef, 4>& memIdxVars,
        std::string& memRefPtr)
    {
        StringRef memRefName = state.nameState.getName(memRef);
        auto memRefElemType = memRef.getType().cast<MemRefType>().getElementType();
        memRefPtr = llvm::Twine(memRefName)
                        .concat("_")
                        .concat(state.nameState.getTempName())
                        .concat("_ptr")
                        .str();

        RETURN_IF_FAILED(printer->printType(memRefElemType));
        os << " *" << memRefPtr << " = &" << memRefName;
        for (const auto& memIdx : memIdxVars)
        {
            os << "[" << memIdx << "]";
        }
        os << ";\n";

        return success();
    }

    // print out accessing a MemRef location indexed by memIdxVars
    LogicalResult AffineDialectCppPrinter::printMemRefAccessValue(
        Value memRef,
        const SmallVector<StringRef, 4>& memIdxVars,
        std::string& memRefVal)
    {
        StringRef memRefName = state.nameState.getName(memRef);
        auto memRefElemType = memRef.getType().cast<MemRefType>().getElementType();
        memRefVal = llvm::Twine(memRefName)
                        .concat("_")
                        .concat(state.nameState.getTempName())
                        .str();

        RETURN_IF_FAILED(printer->printType(memRefElemType));
        os << " " << memRefVal << " = " << memRefName;
        for (const auto& memIdx : memIdxVars)
        {
            os << "[" << memIdx << "]";
        }
        os << ";\n";

        return success();
    }

    LogicalResult
    AffineDialectCppPrinter::printAffineApplyOp(AffineApplyOp affineApplyOp)
    {
        SmallVector<StringRef, 4> memIdxVars;
        printAffineMapResultIndices(affineApplyOp.getAffineMap(),
                                    affineApplyOp.getMapOperands(),
                                    memIdxVars);

        assert(memIdxVars.size() == 1);

        RETURN_IF_FAILED(
            printer->printDeclarationForOpResult(affineApplyOp.getOperation()));
        os << " = " << memIdxVars[0];
        return success();
    }

    LogicalResult
    AffineDialectCppPrinter::printAffineLoadOp(AffineLoadOp affineLoadOp)
    {
        SmallVector<StringRef, 4> memIdxVars;
        printAffineMapResultIndices(affineLoadOp.getAffineMap(),
                                    affineLoadOp.getMapOperands(),
                                    memIdxVars);

        std::string srcMemRefVal;
        RETURN_IF_FAILED(printMemRefAccessValue(affineLoadOp.getMemRef(), memIdxVars, srcMemRefVal));

        RETURN_IF_FAILED(
            printer->printDeclarationForOpResult(affineLoadOp.getOperation()));
        os << " = " << srcMemRefVal;
        return success();
    }

    LogicalResult
    AffineDialectCppPrinter::printAffineStoreOp(AffineStoreOp affineStoreOp)
    {
        SmallVector<StringRef, 4> memIdxVars;
        printAffineMapResultIndices(affineStoreOp.getAffineMap(),
                                    affineStoreOp.getMapOperands(),
                                    memIdxVars);

        std::string dstMemRefPtr;
        RETURN_IF_FAILED(printMemRefAccessPtr(affineStoreOp.getMemRef(), memIdxVars, dstMemRefPtr));

        auto varName = state.nameState.getName(affineStoreOp.value());
        os << "*" << dstMemRefPtr << " = " << varName;
        return success();
    }

    LogicalResult AffineDialectCppPrinter::printAffineVectorLoadOp(
        AffineVectorLoadOp affineVecLoadOp)
    {
        SmallVector<StringRef, 4> memIdxVars;
        printAffineMapResultIndices(affineVecLoadOp.getAffineMap(),
                                    affineVecLoadOp.getMapOperands(),
                                    memIdxVars);

        std::string srcMemRefPtr;
        RETURN_IF_FAILED(printMemRefAccessPtr(affineVecLoadOp.getMemRef(), memIdxVars, srcMemRefPtr));

        // print out vector declaration for the given vector type
        VectorType vecType = affineVecLoadOp.getVectorType();
        std::string vecVar = state.nameState
                                 .getOrCreateName(affineVecLoadOp.getResult(),
                                                  SSANameState::SSANameKind::Variable)
                                 .str();
        RETURN_IF_FAILED(printer->printVectorTypeArrayDecl(vecType, vecVar));
        os << ";\n";

        // add a comment to indicate the vector is loaded from the memory
        os << "// memcpy(&" << vecVar << ", "
           << srcMemRefPtr << ","
           << "sizeof(";
        RETURN_IF_FAILED(printer->printType(vecType));
        os << "))\n";

        int numElems = vecType.getNumElements();
        llvm::interleave(
            llvm::seq(0, numElems),
            os,
            [&](auto idx) {
                os << vecVar << "[" << idx << "] = " << srcMemRefPtr << "[" << idx << "]";
            },
            ";\n");

        return success();
    }

    LogicalResult AffineDialectCppPrinter::printAffineVectorStoreOp(
        AffineVectorStoreOp affineVecStoreOp)
    {
        SmallVector<StringRef, 4> memIdxVars;
        printAffineMapResultIndices(affineVecStoreOp.getAffineMap(),
                                    affineVecStoreOp.getMapOperands(),
                                    memIdxVars);

        std::string dstMemRefPtr;
        RETURN_IF_FAILED(printMemRefAccessPtr(affineVecStoreOp.getMemRef(),
                                              memIdxVars,
                                              dstMemRefPtr));

        auto vecVar = state.nameState.getName(affineVecStoreOp.value());

        VectorType vecType = affineVecStoreOp.getVectorType();

        // print a comment for the vector store
        os << "// memcpy(" << dstMemRefPtr << ", &" << vecVar << ", "
           << "sizeof(";
        RETURN_IF_FAILED(printer->printType(vecType));
        os << "))\n";

        int numElems = vecType.getNumElements();
        llvm::interleave(
            llvm::seq(0, numElems),
            os,
            [&](auto idx) {
                os << dstMemRefPtr << "[" << idx << "] = " << vecVar << "[" << idx << "]";
            },
            ";\n");

        return success();
    }

    LogicalResult
    AffineDialectCppPrinter::printAffineForOp(AffineForOp affineForOp)
    {
        if (!affineForOp.getResults().empty())
        {
            if (affineForOp.getIterOperands().size() != affineForOp.getResults().size())
            {
                os << "AffineForOp: number of iter operands is different from the number of results.\n";
                return failure();
            }

            assert(affineForOp.getIterOperands().size() == affineForOp.getNumRegionIterArgs());

            for (auto&& [resultVar, iterVar, initVal] : llvm::zip(affineForOp.getResults(), affineForOp.getRegionIterArgs(), affineForOp.getIterOperands()))
            {
                StringRef iterVarName = state.nameState.getOrCreateName(
                    iterVar, SSANameState::SSANameKind::Variable);

                if (auto memRefType = resultVar.getType().dyn_cast<MemRefType>())
                {
                    RETURN_IF_FAILED(printer->printDecayedArrayDeclaration(memRefType, iterVarName));
                }
                else
                {
                    RETURN_IF_FAILED(printer->printType(resultVar.getType()));
                    os << " " << iterVarName;
                }
                os << " = " << state.nameState.getName(initVal) << ";\n";
            }
        }

        if (!affineForOp.hasConstantLowerBound())
        {
            affineForOp.emitError("non-const lower-bound is not supported yet");
        }
        if (!affineForOp.hasConstantUpperBound())
        {
            affineForOp.emitError("non-const upper-bound is not supported yet");
        }

        int64_t lowerBound = affineForOp.getConstantLowerBound();
        int64_t upperBound = affineForOp.getConstantUpperBound();
        int64_t step = affineForOp.getStep();

        Value idxVal = affineForOp.getInductionVar();

        StringRef idxName = state.nameState.getOrCreateName(
            idxVal, SSANameState::SSANameKind::LoopIdx);

        if (state.unrolledForOps.contains(affineForOp.getOperation()))
        {
            os << printer->pragmaUnroll() << "\n";
        }

        auto idxType = idxVal.getType();
        os << "for (";
        RETURN_IF_FAILED(printer->printType(idxType));
        os << " " << idxName << " = " << lowerBound << "; ";
        os << idxName << " < " << upperBound << "; ";
        os << idxName << " += " << step << ") {\n";

        auto& loopRegion = affineForOp.region();
        RETURN_IF_FAILED(printer->printRegion(loopRegion, /*printParens*/ false,
                                              /*printBlockTerminator*/ true));
        os << "}\n";

        if (!affineForOp.getResults().empty())
        {
            for (auto&& [resultVar, iterVar] : llvm::zip(affineForOp.getResults(), affineForOp.getRegionIterArgs()))
            {
                StringRef resultName = state.nameState.getOrCreateName(
                    resultVar, SSANameState::SSANameKind::Variable);
                if (auto memRefType = resultVar.getType().dyn_cast<MemRefType>())
                {
                    RETURN_IF_FAILED(printer->printDecayedArrayDeclaration(memRefType, resultName));
                }
                else
                {
                    RETURN_IF_FAILED(printer->printType(resultVar.getType()));
                    os << " " << resultName;
                }
                os << " = " << state.nameState.getName(iterVar) << ";\n";
            }
        }

        return success();
    }

    LogicalResult AffineDialectCppPrinter::printAffineYieldOp(AffineYieldOp affineYieldOp)
    {
        if (affineYieldOp.getNumOperands() == 0)
        {
            return success();
        }

        auto affineForOp = affineYieldOp->getParentOfType<AffineForOp>();
        for (auto i = 0u; i < affineForOp.getNumRegionIterArgs(); ++i)
        {
            auto iterVar = affineForOp.getRegionIterArgs()[i];
            auto result = affineYieldOp.getOperand(i);

            os << state.nameState.getName(iterVar) << " = " << state.nameState.getName(result) << ";\n";
        }
        return success();
    }

    LogicalResult AffineDialectCppPrinter::printDialectOperation(Operation* op,
                                                                 bool* skipped,
                                                                 bool* consumed)
    {
        *consumed = true;

        if (auto affineApplyOp = dyn_cast<AffineApplyOp>(op))
            return printAffineApplyOp(affineApplyOp);

        if (auto affineLoadOp = dyn_cast<AffineLoadOp>(op))
            return printAffineLoadOp(affineLoadOp);

        if (auto affineStoreOp = dyn_cast<AffineStoreOp>(op))
            return printAffineStoreOp(affineStoreOp);

        if (auto affineVectorLoadOp = dyn_cast<AffineVectorLoadOp>(op))
            return printAffineVectorLoadOp(affineVectorLoadOp);

        if (auto affineVectorStoreOp = dyn_cast<AffineVectorStoreOp>(op))
            return printAffineVectorStoreOp(affineVectorStoreOp);

        if (auto affineYieldOp = dyn_cast<AffineYieldOp>(op))
            return printAffineYieldOp(affineYieldOp);

        if (auto affineForOp = dyn_cast<AffineForOp>(op))
        {
            *skipped = true;
            return printAffineForOp(affineForOp);
        }

        *consumed = false;
        return success();
    }

    void AffineDialectCppPrinter::checkAffineMemCpyPass(Operation* op)
    {
        auto walkResult = op->walk([&](Operation* subOp) {
            if (isa<AffineVectorLoadOp, AffineVectorStoreOp>(subOp))
            {
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if (walkResult.wasInterrupted())
        {
            needAffineMemCpy = true;
        }
    }

    // TODO: Skipping dead affine ops is particularly useful because it can greatly
    // reduce the number of affine maps to be printed out. Later, if we think this
    // would benefit more senarios, we could consider to move it to CppPrinter.
    // But, let's narrow its scope for now.
    void AffineDialectCppPrinter::checkDeadAffineOpsPass(Operation* op)
    {
        op->walk([&](Operation* subOp) {
            if (isa<AffineApplyOp, AffineLoadOp, AffineMaxOp, AffineMinOp, AffineVectorLoadOp>(subOp))
            {
                bool unused = llvm::all_of(subOp->getResults(),
                                           [&](Value res) { return res.use_empty(); });
                if (unused)
                {
                    state.skippedOps.insert(subOp);
                }
            }
        });
    }

    void AffineDialectCppPrinter::collectAffineMapsPass(Operation* op)
    {
        AffineMapVisitor visitor(getAffineMapToFuncBaseName());

        op->walk([&](Operation* subOp) {
            for (auto type : subOp->getOperandTypes())
            {
                visitor.visit(type);
            }

            for (auto type : subOp->getResultTypes())
            {
                visitor.visit(type);
            }

            for (auto& region : subOp->getRegions())
            {
                for (auto& block : region)
                {
                    for (auto arg : block.getArguments())
                    {
                        visitor.visit(arg.getType());
                    }
                }
            }

            for (auto attr : subOp->getAttrs())
            {
                visitor.visit(attr.second);
            }
        });
    }

    LogicalResult AffineDialectCppPrinter::runPrePrintingPasses(Operation* op)
    {
        checkAffineMemCpyPass(op);
        checkDeadAffineOpsPass(op);
        collectAffineMapsPass(op);
        return success();
    }

} // namespace cpp_printer
} // namespace mlir
