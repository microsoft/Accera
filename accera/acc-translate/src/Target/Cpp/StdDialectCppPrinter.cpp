////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "StdDialectCppPrinter.h"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/GPU/GPUDialect.h>

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir
{
namespace cpp_printer
{

    static bool isBinaryOp(Operation* op)
    {
        return isa<
            // KEEP THIS SORTED
            arith::AddFOp,
            arith::AddIOp,
            arith::AndIOp,
            arith::CmpFOp,
            arith::CmpIOp,
            arith::DivFOp,
            arith::DivSIOp,
            arith::DivUIOp,
            arith::MulFOp,
            arith::MulIOp,
            arith::OrIOp,
            arith::RemFOp,
            arith::RemSIOp,
            arith::RemUIOp,
            arith::ShLIOp,
            arith::ShRSIOp,
            arith::ShRUIOp,
            arith::SubFOp,
            arith::SubIOp,
            arith::XOrIOp>(op);
    }

    // Simple CastOps are those that we don't care about the signed-ness of the
    // operands, i.e. we can directly generate a cast expression based on the
    // result type of the op
    static bool isSimpleCastOp(Operation* op)
    {
        return isa<
            // KEEP THIS SORTED
            arith::ExtFOp,
            arith::SIToFPOp,
            arith::TruncFOp,
            arith::TruncIOp>(op);
    }

    static bool isSupportedCudaMemSpace(unsigned memspace)
    {
        return (memspace != gpu::GPUDialect::getPrivateAddressSpace()) ||
               (memspace != gpu::GPUDialect::getWorkgroupAddressSpace());
    }

    LogicalResult StdDialectCppPrinter::printAllocOp(memref::AllocOp allocOp)
    {
        MemRefType memrefType = allocOp.getType();
        RETURN_IF_FAILED(printer->checkMemRefType(memrefType));

        if (!memrefType.hasRank())
        {
            return allocOp.emitOpError("<<Unranked MemRefType for AllocOp>>");
        }

        auto varName = state.nameState.getOrCreateName(
            allocOp.getResult(), SSANameState::SSANameKind::Variable);

        Type elemType = memrefType.getElementType();
        unsigned numElems = memrefType.getNumElements();
        unsigned memspace = memrefType.getMemorySpaceAsInt();
        // only support default memspace (a.k.a 0) for non-cuda cases
        if (!state.hasRuntime(Runtime::CUDA))
        {
            if (memspace != 0)
            {
                allocOp.emitOpError("<<Unsupported memspace " + std::to_string(memspace) +
                                    " for AllocOp>>");
            }
            RETURN_IF_FAILED(printer->printType(elemType));
            os << " *" << varName << " = ";
            os << "new ";
            RETURN_IF_FAILED(printer->printType(elemType));
            os << "[" << numElems << "]";
        }
        else
        {
            if (!isSupportedCudaMemSpace(memspace))
            {
                allocOp.emitOpError("<<Unsupported CUDA memspace " +
                                    std::to_string(memspace) + " for AllocOp>>");
            }

            if (memspace == gpu::GPUDialect::getWorkgroupAddressSpace() && allocOp.alignment())
            {
                auto shape = memrefType.getShape();
                std::string idxAccess;
                for (int i = 1; i < memrefType.getRank(); ++i)
                {
                    idxAccess += "[" + std::to_string(shape[i]) + "]";
                }
                RETURN_IF_FAILED(printer->printType(elemType));
                os << "(*" << varName << ")" << idxAccess;
                os << " = reinterpret_cast<";
                RETURN_IF_FAILED(printer->printType(elemType));
                os << "(*)" << idxAccess << ">(reinterpret_cast<";
                RETURN_IF_FAILED(printer->printType(elemType));
                os << "*>(sharedMemBaseAddr) + " << *allocOp.alignment() << ")";
            }
            else
            {
                if (memspace == gpu::GPUDialect::getWorkgroupAddressSpace())
                    os << printer->sharedAttrIfCuda();
                RETURN_IF_FAILED(printer->printArrayDeclaration(memrefType, varName));
            }
        }

        return success();
    }

    LogicalResult StdDialectCppPrinter::printAllocaOp(memref::AllocaOp allocaOp)
    {
        // Note that we do not allow to directly print MemRefType, which should
        // be handled along with each dialect's array type (if it has any).
        // In AllocaOp case, we need to create std::array or C-style arrays for
        // double buffers.
        MemRefType memrefType = allocaOp.getType();
        RETURN_IF_FAILED(printer->checkMemRefType(memrefType));

        auto varName = state.nameState.getOrCreateName(
            allocaOp.getResult(), SSANameState::SSANameKind::Variable);
        return printer->printArrayDeclaration(memrefType, varName);
    }

    LogicalResult StdDialectCppPrinter::printCallOp(CallOp callOp)
    {
        Operation* op = callOp.getOperation();
        CallOpInterface call = dyn_cast<CallOpInterface>(op);
        assert(call);
        Operation* defOp = call.resolveCallable();
        assert(defOp && "cannot find defining func op!");

        // we are calling an intrinsic
        if (state.intrinsicDecls.count(defOp))
        {
            return printer->printIntrinsicCallOp(op, defOp);
        }

        (void)printer->printDeclarationForOpResult(op);
        if (op->getNumResults() > 0)
            os << " = ";

        os << callOp.getCallee() << "(";
        RETURN_IF_FAILED(printer->printOperationOperands(callOp));
        os << ")";
        return success();
    }

    LogicalResult StdDialectCppPrinter::printConstantOp(arith::ConstantOp constOp)
    {
        if (isConstantScalarOp(constOp))
        {
            state.nameState.addConstantValue(constOp.getResult(), constOp.getValue());
        }
        else
        {
            RETURN_IF_FAILED(printer->printType(constOp.getType()));
            os << " "
               << state.nameState.getOrCreateName(constOp.getResult(),
                                                  SSANameState::SSANameKind::Constant);
            // Now print out the constant value
            os << " = ";
            if (failed(printer->printAttribute(constOp.getValue())))
                return constOp.emitOpError("<<unable to print constant value>>");
        }
        return success();
    }

    LogicalResult StdDialectCppPrinter::printMaxFOp(arith::MaxFOp maxfOp)
    {
        auto lhs = state.nameState.getOrCreateName(maxfOp.getLhs(), SSANameState::SSANameKind::Variable);
        auto rhs = state.nameState.getOrCreateName(maxfOp.getRhs(), SSANameState::SSANameKind::Variable);
        RETURN_IF_FAILED(printer->printType(maxfOp.getType()));
        os << " " << state.nameState.getOrCreateName(maxfOp.getResult(), SSANameState::SSANameKind::Variable);
        os << " = " << lhs << " > " << rhs << " ? " << lhs << " : " << rhs;
        return success();
    }

    LogicalResult StdDialectCppPrinter::printDeallocOp(memref::DeallocOp deallocOp,
                                                       bool* skipped)
    {
        auto ty = deallocOp.getOperand().getType();
        MemRefType memrefType = ty.dyn_cast<MemRefType>();
        assert(memrefType && "not a MemRefType?");
        RETURN_IF_FAILED(printer->checkMemRefType(memrefType));

        // make sure we have a var for both cuda and non-cuda cases
        auto varName = state.nameState.getName(deallocOp.getOperand());
        if (state.hasRuntime(Runtime::CUDA))
        {
            // we have nothing to do except for checking if the memspace is either
            // shared or private
            unsigned memspace = memrefType.getMemorySpaceAsInt();
            if (!isSupportedCudaMemSpace(memspace))
            {
                deallocOp.emitOpError("<<Unsupported CUDA memspace " +
                                      std::to_string(memspace) + " for Dealloc>>");
            }
            *skipped = true;
        }
        else
        {
            os << "delete []" << varName << "\n";
        }
        return success();
    }

    LogicalResult StdDialectCppPrinter::printDimOp(memref::DimOp dimOp)
    {
        auto operandTy = dimOp.getOperand(0).getType().dyn_cast<ShapedType>();
        if (!operandTy)
            return dimOp.emitOpError("<<DimOp's operand must be of ShapedType>>");

        // TODO: support retrieving dynamic dimensions for DimOp
        if (operandTy.getNumDynamicDims() != 0)
        {
            return dimOp.emitOpError(
                "<<Dynamic dimension for DimOp is not supported yet>>");
        }

        (void)printer->printDeclarationForOpResult(dimOp.getOperation());
        auto idx = dimOp.getConstantIndex();
        assert(idx.hasValue());
        os << " = " << operandTy.getShape()[idx.getValue()];

        return success();
    }

    LogicalResult StdDialectCppPrinter::printExpOp(math::ExpOp expOp)
    {
        RETURN_IF_FAILED(printer->printDeclarationForOpResult(expOp.getOperation()));
        os << " = ";

        StringRef operandName = state.nameState.getName(expOp.getOperand());
        auto ty = expOp.getType();
        if (ty.isF32())
        {
            os << "expf(" << operandName << ")";
        }
        else if (ty.isF16())
        {
            if (!state.hasRuntime(Runtime::CUDA))
            {
                return expOp.emitOpError("<<fp16 is supported only for CUDA>>");
            }
            os << "hexp(" << operandName << ")";
        }
        else
        {
            return expOp.emitOpError("<<unsupported type for expOp>>");
        }
        return success();
    }

    LogicalResult StdDialectCppPrinter::printLoadOp(memref::LoadOp loadOp)
    {
        // make sure we don't accidentally load a value from an unsupported
        // MemRefType
        MemRefType memRefType = loadOp.getMemRefType();
        RETURN_IF_FAILED(printer->checkMemRefType(memRefType));
        auto indices = loadOp.getIndices();
        size_t rank = memRefType.getRank();
        if (indices.size() != rank)
        {
            return loadOp.emitOpError() << "<<Indices do not match rank>>";
        }
        Value memref = loadOp.getMemRef();
        return printer->printMemRefLoadOrStore(true, memref, memRefType, indices, loadOp.getResult());
    }

    LogicalResult StdDialectCppPrinter::printCastToIntegerOp(Operation* op,
                                                             bool isSigned)
    {
        auto toTy = op->getResult(0).getType();
        auto intTy = toTy.dyn_cast<IntegerType>();
        if (!intTy)
        {
            return op->emitError() << "<<toTy is not an Integer type>>";
        }

        auto printIntType = [&]() {
            RETURN_IF_FAILED(
                printer->printIntegerType(intTy, /*forceSignedness*/ true, isSigned));
            return success();
        };

        auto printDecl = [&]() {
            RETURN_IF_FAILED(printIntType());
            os << " ";
            os << state.nameState.getOrCreateName(op->getResult(0),
                                                  SSANameState::SSANameKind::Variable);
            return success();
        };

        os << "#if defined(__cplusplus)\n";
        RETURN_IF_FAILED(printDecl());
        os << " = static_cast<";
        RETURN_IF_FAILED(printIntType());
        os << ">(" << state.nameState.getName(op->getOperand(0)) << ");\n";
        os << "#else\n";
        RETURN_IF_FAILED(printDecl());
        os << " = (";
        RETURN_IF_FAILED(printIntType());
        os << ")(" << state.nameState.getName(op->getOperand(0)) << ");\n";
        os << "#endif // __cplusplus\n";

        return success();
    }

    LogicalResult StdDialectCppPrinter::printIndexCastOp(arith::IndexCastOp op)
    {
        state.nameState.addNameAlias(op.getResult(), op.getIn());
        return success();
    }

    LogicalResult StdDialectCppPrinter::printSimpleCastOp(Operation* op)
    {
        auto toTy = op->getResult(0).getType();
        if (toTy.dyn_cast<VectorType>())
        {
            return op->emitError() << "<<casting on VectorType is not supported yet>>";
        }

        if (isa<arith::ExtFOp, arith::TruncFOp>(op) && state.hasRuntime(Runtime::ROCM))
        {
            auto fromTy = op->getOperand(0).getType();
            if (fromTy.isBF16() || toTy.isBF16())
            {
                RETURN_IF_FAILED(printer->printDeclarationForOpResult(op));
                os << " = cast(" << state.nameState.getName(op->getOperand(0)) << ")";
                return success();
            }
        }

        os << "#if defined(__cplusplus)\n";
        RETURN_IF_FAILED(printer->printDeclarationForOpResult(op));
        os << " = static_cast<";
        RETURN_IF_FAILED(printer->printType(toTy));
        os << ">(" << state.nameState.getName(op->getOperand(0)) << ");\n";
        os << "#else\n";
        RETURN_IF_FAILED(printer->printDeclarationForOpResult(op));
        os << " = (";
        RETURN_IF_FAILED(printer->printType(toTy));
        os << ")(" << state.nameState.getName(op->getOperand(0)) << ");\n";
        os << "#endif // __cplusplus\n";
        return success();
    }

    LogicalResult StdDialectCppPrinter::printReturnOp(ReturnOp returnOp)
    {
        auto numOperands = returnOp.getNumOperands();
        os << "return";

        if (numOperands == 0)
        {
            // Nothing to do
        }
        else if (numOperands == 1)
        {
            os << " " << state.nameState.getName(returnOp.getOperand(0));
        }
        else
        {
            return returnOp.emitOpError() << "<<Returning tuple is not supported yet>>";
        }

        return success();
    }

    LogicalResult StdDialectCppPrinter::printSelectOp(SelectOp selectOp)
    {
        if (selectOp.getNumOperands() != 3)
        {
            return selectOp.emitOpError()
                   << "<<Invalid SelectOp: incorrect number of operands>>";
        }

        RETURN_IF_FAILED(
            printer->printDeclarationForOpResult(selectOp.getOperation()));
        os << " = ";

        OperandRange operands = selectOp.getOperands();
        os << "(" << state.nameState.getName(operands[0]) << " ? ";
        os << state.nameState.getName(operands[1]) << " : "
           << state.nameState.getName(operands[2]) << ")";
        return success();
    }

    LogicalResult StdDialectCppPrinter::printGetGlobalOp(memref::GetGlobalOp getGlobalOp)
    {
        auto globalName = getGlobalOp.nameAttr();
        auto result = getGlobalOp.getResult();

        auto symTableOp = mlir::SymbolTable::getNearestSymbolTable(getGlobalOp);
        auto globalOp = dyn_cast<memref::GlobalOp>(mlir::SymbolTable::lookupNearestSymbolFrom(symTableOp, globalName));

        // if the global was nested, then create the instance of the global here
        if (globalOp && globalOp.sym_visibilityAttr() == "nested")
        {
            MemRefType memrefType = globalOp.type().dyn_cast<MemRefType>();

            if (memrefType.getMemorySpaceAsInt() == gpu::GPUDialect::getWorkgroupAddressSpace())
                os << printer->sharedAttrIfCuda();

            auto varName = state.nameState.getOrCreateName(
                result, SSANameState::SSANameKind::Variable);
            return printer->printArrayDeclaration(memrefType, varName);
        }

        os << "auto";
        os << " ";
        os << state.nameState.getOrCreateName(result,
                                              SSANameState::SSANameKind::Variable);
        os << " = " << globalName.getValue();

        return success();
    }

    LogicalResult StdDialectCppPrinter::printStoreOp(memref::StoreOp storeOp)
    {
        MemRefType memRefType = storeOp.getMemRefType();
        RETURN_IF_FAILED(printer->checkMemRefType(memRefType));

        Value memref = storeOp.getMemRef();
        auto indices = storeOp.getIndices();
        return printer->printMemRefLoadOrStore(false, memref, memRefType, indices, storeOp.getValueToStore());
    }

    LogicalResult StdDialectCppPrinter::printReinterpretCastOp(memref::ReinterpretCastOp reinterpretCastOp)
    {
        // Just cast it to a (type*) for the given element type. At this stage, any specific size data may have been removed
        auto resultElementType = reinterpretCastOp.getType().getElementType();
        auto src = reinterpretCastOp.source();
        // Create an unranked memreftype so that we generate a basic "type*" decl rather than allocating a buffer for the memref
        auto customType = mlir::UnrankedMemRefType::get(resultElementType, reinterpretCastOp.getType().getMemorySpace());
        RETURN_IF_FAILED(printer->printDeclarationForValue(reinterpretCastOp.result(), customType));
        os << " = ";
        os << "(";
        RETURN_IF_FAILED(printer->printType(resultElementType));
        os << "*) ";
        os << state.nameState.getName(src) << ";\n";
        return success();
    }

    static StringRef getCmpIOpString(arith::CmpIPredicate predicate)
    {
        switch (predicate)
        {
        default:
            return "<<Invalid CmpIOp>>";
        case arith::CmpIPredicate::eq:
            return "==";
        case arith::CmpIPredicate::ne:
            return "!=";

        case arith::CmpIPredicate::slt: // Fall-through
        case arith::CmpIPredicate::ult:
            return "<";

        case arith::CmpIPredicate::sle: // Fall-through
        case arith::CmpIPredicate::ule:
            return "<=";

        case arith::CmpIPredicate::sgt: // Fall-through
        case arith::CmpIPredicate::ugt:
            return ">";

        case arith::CmpIPredicate::sge: // Fall-through
        case arith::CmpIPredicate::uge:
            return ">=";
        }
    }

    static StringRef getCmpFOpString(arith::CmpFPredicate predicate)
    {
        using arith::CmpFPredicate;

        switch (predicate)
        {
        default:
            return "<<Invalid CmpFOp>>";

        case CmpFPredicate::OEQ: // Fall-through
        case CmpFPredicate::UEQ:
            return "==";

        case CmpFPredicate::OGT: // Fall-through
        case CmpFPredicate::UGT:
            return ">";

        case CmpFPredicate::OGE: // Fall-through
        case CmpFPredicate::UGE:
            return ">=";

        case CmpFPredicate::OLT: // Fall-through
        case CmpFPredicate::ULT:
            return "<";

        case CmpFPredicate::OLE: // Fall-through
        case CmpFPredicate::ULE:
            return "<=";

        case CmpFPredicate::ONE: // Fall-through
        case CmpFPredicate::UNE:
            return "!=";

        case CmpFPredicate::ORD:
        case CmpFPredicate::UNO: // Fall-through
            return "<<Unsupported CmpFPredicate>>";
        }
    }

    // TODO: ArithDialectCppPrinter?
    LogicalResult StdDialectCppPrinter::printBinaryOp(Operation* binOp)
    {
        if (binOp->getNumOperands() != 2)
            return binOp->emitError("<<Invalid binOp Operands>>");
        if (binOp->getNumResults() != 1)
            return binOp->emitError("<<Invalid binOp Results>>");

        RETURN_IF_FAILED(printer->printDeclarationForOpResult(binOp));

        os << " = ";
        os << state.nameState.getName(binOp->getOperand(0)) << " ";

        // FIXME: We need to add cast to force unsigned/signed conversion for those
        // signed/unsigned ops

        // FIXME: handle CmpFPPredicate::TURE specially

        // TODO: invoke fp functions for fp ops such as remf, exp, etc
        llvm::TypeSwitch<Operation*, void>(binOp)
            .Case<arith::AddIOp, arith::AddFOp>([&](auto&) { os << "+"; })
            .Case<arith::AndIOp>([&](auto&) { os << "&"; })
            .Case<arith::CmpFOp>(
                [&](arith::CmpFOp op) { os << getCmpFOpString(op.getPredicate()); })
            .Case<arith::CmpIOp>(
                [&](arith::CmpIOp op) { os << getCmpIOpString(op.getPredicate()); })
            .Case<arith::DivUIOp, arith::DivSIOp, arith::DivFOp>([&](auto&) { os << "/"; })
            .Case<arith::MulIOp, arith::MulFOp>([&](auto&) { os << "*"; })
            .Case<arith::OrIOp>([&](auto&) { os << "|"; })
            .Case<arith::RemUIOp, arith::RemSIOp>([&](auto&) { os << "%"; })
            .Case<arith::ShLIOp>([&](auto&) { os << "<<"; })
            .Case<arith::ShRUIOp, arith::ShRSIOp>(
                [&](auto&) { os << ">>"; })
            .Case<arith::SubIOp, arith::SubFOp>([&](auto&) { os << "-"; })
            .Case<arith::XOrIOp>([&](auto&) { os << "^"; })
            .Default([&](auto&) { os << "<<unknown ArithmeticOp>>"; });

        os << " " << state.nameState.getName(binOp->getOperand(1));

        return success();
    }

    LogicalResult
    StdDialectCppPrinter::printMemRefCastOp(memref::CastOp memRefCastOp)
    {
        // note that Unrank to Unrank cast is not support in the MLIR's Standard
        // Dialect

        Type srcType = memRefCastOp.getOperand().getType();
        Type dstType = memRefCastOp.getType();

        MemRefType srcMemRefType = srcType.dyn_cast<MemRefType>();
        if (srcMemRefType)
        {
            RETURN_IF_FAILED(printer->checkMemRefType(srcMemRefType));
        }
        MemRefType dstMemRefType = dstType.dyn_cast<MemRefType>();
        if (dstMemRefType)
        {
            RETURN_IF_FAILED(printer->checkMemRefType(dstMemRefType));
        }

        if (dstMemRefType)
        {
            RETURN_IF_FAILED(printer->printType(dstMemRefType.getElementType()));
        }
        else
        {
            RETURN_IF_FAILED(printer->printType(
                dstType.dyn_cast<UnrankedMemRefType>().getElementType()));
        }

        os << " *"
           << state.nameState.getOrCreateName(memRefCastOp.getResult(),
                                              SSANameState::SSANameKind::Variable)
           << " = ";
        auto srcName = state.nameState.getName(memRefCastOp.getOperand());
        // For UnrankedMemRefType to MemRefType case, because we just simply
        // create an assignment at the moment. It should be fine because
        // MemRefType is treated to be C's array
        if (srcType.isa<UnrankedMemRefType>())
        {
            os << srcName;
            return success();
        }

        // Cast from MemRefType to UnrankedMemRefType or MemRefType
        os << "(";
        RETURN_IF_FAILED(printer->printType(srcMemRefType.getElementType()));
        os << "*)" << srcName;

        return success();
    }

    LogicalResult StdDialectCppPrinter::printMemRefTransposeOp(memref::TransposeOp memRefCastOp)
    {
        // This is metadata transposition only, so this is identical to CastOp
        Type srcType = memRefCastOp.in().getType();
        MemRefType srcMemRefType = srcType.dyn_cast<MemRefType>();
        Type dstType = memRefCastOp.getType();
        RETURN_IF_FAILED(printer->printType(dstType.dyn_cast<BaseMemRefType>().getElementType()));

        os << " *"
           << state.nameState.getOrCreateName(memRefCastOp.getResult(),
                                              SSANameState::SSANameKind::Variable)
           << " = ";
        auto srcName = state.nameState.getName(memRefCastOp.getOperand());
        assert(!srcType.isa<UnrankedMemRefType>());

        os << "(";
        RETURN_IF_FAILED(printer->printType(srcMemRefType.getElementType()));
        os << "*)" << srcName;

        return success();
    }

    LogicalResult StdDialectCppPrinter::printDialectOperation(Operation* op,
                                                              bool* skipped,
                                                              bool* consumed)
    {
        // TODO: support more ops, UnaryOp, Tensor, ExtractElementOp, etc

        *consumed = true;

        if (auto allocOp = dyn_cast<memref::AllocOp>(op))
            return printAllocOp(allocOp);

        if (auto allocaOp = dyn_cast<memref::AllocaOp>(op))
            return printAllocaOp(allocaOp);

        // Binary Ops
        if (isBinaryOp(op))
            return printBinaryOp(op);

        if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(op))
            return printIndexCastOp(indexCastOp);

        if (isSimpleCastOp(op))
            return printSimpleCastOp(op);

        if (auto callOp = dyn_cast<CallOp>(op))
            return printCallOp(callOp);

        if (auto constOp = dyn_cast<arith::ConstantOp>(op))
            return printConstantOp(constOp);

        if (auto maxFOp = dyn_cast<arith::MaxFOp>(op))
            return printMaxFOp(maxFOp);

        if (auto deallocOp = dyn_cast<memref::DeallocOp>(op))
            return printDeallocOp(deallocOp, skipped);

        if (auto dimOp = dyn_cast<memref::DimOp>(op))
            return printDimOp(dimOp);

        if (auto expOp = dyn_cast<math::ExpOp>(op))
            return printExpOp(expOp);

        if (auto loadOp = dyn_cast<memref::LoadOp>(op))
            return printLoadOp(loadOp);

        if (auto memRefCastOp = dyn_cast<memref::CastOp>(op))
            return printMemRefCastOp(memRefCastOp);

        if (auto returnOp = dyn_cast<ReturnOp>(op))
            return printReturnOp(returnOp);

        if (auto selectOp = dyn_cast<SelectOp>(op))
            return printSelectOp(selectOp);

        if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(op))
            return printGetGlobalOp(getGlobal);

        if (auto storeOp = dyn_cast<memref::StoreOp>(op))
            return printStoreOp(storeOp);

        if (auto reinterpretCastOp = dyn_cast<memref::ReinterpretCastOp>(op))
            return printReinterpretCastOp(reinterpretCastOp);

        if (isa<arith::ExtSIOp>(op))
            return printCastToIntegerOp(op, /*isSigned*/ true);

        if (isa<arith::ExtUIOp>(op))
            return printCastToIntegerOp(op, /*isSigned*/ false);

        if (auto transposeOp = dyn_cast<memref::TransposeOp>(op))
            return printMemRefTransposeOp(transposeOp);

        *consumed = false;
        return success();
    }

    LogicalResult StdDialectCppPrinter::printHeaderFiles()
    {
        return success();
    }

    LogicalResult StdDialectCppPrinter::printPrologue()
    {
        if (!state.hasRuntime(Runtime::CUDA))
        {
            os << R"STD(

#ifndef __forceinline__
#if defined(_MSC_VER)
#define __forceinline__ __forceinline
#else
#define __forceinline__ __inline__ __attribute__((always_inline))
#endif // _MSC_VER
#endif // __forceinline__

)STD";
        }

        return success();
    }

} // namespace cpp_printer
} // namespace mlir
