////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "exec/ExecutionPlanAttributes.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>

#include <iostream>
#include <stdexcept>

namespace accera::ir
{
namespace executionPlan
{
    //
    // DSL type printers
    //
    mlir::DialectAsmPrinter& operator<<(mlir::DialectAsmPrinter& printer, VectorizationInfo vectorizationInfo)
    {
        printer << "{" << vectorizationInfo.vectorBytes << "," << vectorizationInfo.vectorUnitCount << "," << (vectorizationInfo.unrollOnly ? 1 : 0) << '}';
        return printer;
    }

    mlir::DialectAsmPrinter& operator<<(mlir::DialectAsmPrinter& printer, ParallelizationInfo parallelizationInfo)
    {
        printer << "{" << (parallelizationInfo.isDynamicPolicy ? 1 : 0) << "," << parallelizationInfo.numThreads << '}';
        return printer;
    }

    mlir::DialectAsmPrinter& operator<<(mlir::DialectAsmPrinter& printer, TensorizationInfo tensorizationInfo)
    {
        printer << "{{" << (int)tensorizationInfo.dim << "}," << tensorizationInfo.numTotalPasses << "," << tensorizationInfo.useStaticOffsets << "," << tensorizationInfo.numFusedPasses << "," << (int)tensorizationInfo.schedulingPolicy << "}";
        return printer;
    }

    mlir::DialectAsmPrinter& operator<<(mlir::DialectAsmPrinter& printer, InPlaceUnrollInfo inPlaceUnrollInfo)
    {
        printer << "{" << (inPlaceUnrollInfo.loopUnrollFactor) << "}";
        return printer;
    }

    //
    // VectorizationInfoAttr
    //
    VectorizationInfoAttr VectorizationInfoAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    VectorizationInfo VectorizationInfoAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    VectorizationInfoAttr parseVectorizationInfo(mlir::DialectAsmParser& parser)
    {
        // Parse a vectorization info attribute in the following form:
        //   vectorization-info-attr ::= `{` vectorBytes `,` vectorUnitCount `}`

        // NOTE: All MLIR parser function return a ParseResult. This is a
        // specialization of LogicalResult that auto-converts to a `true` boolean
        // value on failure to allow for chaining, but may be used with explicit
        // `mlir::failed/mlir::succeeded` as desired.

        if (failed(parser.parseLBrace()))
            return {};

        int vectorBytes;
        if (failed(parser.parseInteger(vectorBytes)))
            return {};

        if (failed(parser.parseComma()))
            return {};

        int vectorUnitCount;
        if (failed(parser.parseInteger(vectorUnitCount)))
            return {};

        int unrollOnly = 0;
        if (succeeded(parser.parseOptionalComma()))
        {
            if (failed(parser.parseInteger(unrollOnly)))
                return {};
        }
        if (failed(parser.parseRBrace()))
            return {};

        return VectorizationInfoAttr::get(VectorizationInfo{ vectorBytes, vectorUnitCount, static_cast<bool>(unrollOnly) }, parser.getBuilder().getContext());
    }

    void print(VectorizationInfoAttr attr, mlir::DialectAsmPrinter& printer)
    {
        printer << "vectorizationinfo";
        auto vectorizationInfo = attr.cast<VectorizationInfoAttr>().getValue();
        printer << vectorizationInfo;
    }

    //
    // ParallelizationInfoAttr
    //
    ParallelizationInfoAttr ParallelizationInfoAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    ParallelizationInfo ParallelizationInfoAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    ParallelizationInfoAttr parseParallelizationInfo(mlir::DialectAsmParser& parser)
    {
        // Parse a parallelization info attribute in the following form:
        //   parallelization-info-attr ::= `{` isDynamicPolicy `,` numThreads `}`

        if (failed(parser.parseLBrace()))
            return {};

        int isDynamicPolicy = 0;
        if (failed(parser.parseInteger(isDynamicPolicy)))
            return {};

        if (failed(parser.parseComma()))
            return {};

        int numThreads;
        if (failed(parser.parseInteger(numThreads)))
            return {};

        if (failed(parser.parseRBrace()))
            return {};

        return ParallelizationInfoAttr::get(ParallelizationInfo{ static_cast<int64_t>(numThreads), static_cast<bool>(isDynamicPolicy) }, parser.getBuilder().getContext());
    }

    void print(ParallelizationInfoAttr attr, mlir::DialectAsmPrinter& printer)
    {
        printer << "parallelizationinfo";
        auto parallelizationInfo = attr.cast<ParallelizationInfoAttr>().getValue();
        printer << parallelizationInfo;
    }

    //
    // TensorizationInfoAttr
    //
    TensorizationInfoAttr TensorizationInfoAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    TensorizationInfo TensorizationInfoAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    TensorizationInfoAttr parseTensorizationInfo(mlir::DialectAsmParser& parser)
    {
        int dim;
        bool useStaticOffsets;
        int numFusedPasses;
        int numTotalPasses;
        int schedulingPolicy;
        if (failed(parser.parseLBrace()))
            return {};
        if (failed(parser.parseLBrace()))
            return {};
        if (failed(parser.parseInteger(dim)))
            return {};
        if (failed(parser.parseRBrace()))
            return {};
        if (failed(parser.parseComma()))
            return {};
        if (failed(parser.parseInteger(numTotalPasses)))
            return {};
        if (failed(parser.parseComma()))
            return {};
        if (failed(parser.parseInteger(useStaticOffsets)))
            return {};
        if (failed(parser.parseComma()))
            return {};
        if (failed(parser.parseInteger(numFusedPasses)))
            return {};
        if (failed(parser.parseComma()))
            return {};
        if (failed(parser.parseInteger(schedulingPolicy)))
            return {};
        if (failed(parser.parseRBrace()))
            return {};
        if (useStaticOffsets != 0 && useStaticOffsets != 1)
            return {};
        return TensorizationInfoAttr::get(TensorizationInfo{ accera::ir::value::MMAShape{ dim }, numTotalPasses, useStaticOffsets, numFusedPasses, accera::ir::value::MMASchedulingPolicy{ schedulingPolicy } }, parser.getBuilder().getContext());
    }

    void print(TensorizationInfoAttr attr, mlir::DialectAsmPrinter& printer)
    {
        printer << "tensorizationinfo";
        auto tensorizelInfo = attr.cast<TensorizationInfoAttr>().getValue();
        printer << tensorizelInfo;
    }

    //
    // InPlaceUnrollInfoAttr
    //
    InPlaceUnrollInfoAttr InPlaceUnrollInfoAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    InPlaceUnrollInfo InPlaceUnrollInfoAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    InPlaceUnrollInfoAttr parseInPlaceUnrollInfo(mlir::DialectAsmParser& parser)
    {
        // Parse a InPlaceUnroll info attribute in the following form:
        //   in-place-unroll-info-attr ::= `{` loopUnrollFactor `}`

        if (failed(parser.parseLBrace()))
            return {};

        int loopUnrollFactor = 0;
        if (failed(parser.parseInteger(loopUnrollFactor)))
            return {};

        if (failed(parser.parseRBrace()))
            return {};

        return InPlaceUnrollInfoAttr::get(InPlaceUnrollInfo{ static_cast<int64_t>(loopUnrollFactor) }, parser.getBuilder().getContext());
    }

    void print(InPlaceUnrollInfoAttr attr, mlir::DialectAsmPrinter& printer)
    {
        printer << "inplaceunrollinfo";
        auto inPlaceUnrollInfo = attr.cast<InPlaceUnrollInfoAttr>().getValue();
        printer << inPlaceUnrollInfo;
    }

    //
    // Hash functions for storage key types
    //
    llvm::hash_code hash_value(const VectorizationInfo& vectorizationInfo)
    {
        return llvm::hash_combine(vectorizationInfo.vectorBytes, vectorizationInfo.vectorUnitCount);
    }

    llvm::hash_code hash_value(const ParallelizationInfo& parallelizationInfo)
    {
        return llvm::hash_combine(parallelizationInfo.numThreads, parallelizationInfo.isDynamicPolicy);
    }

    llvm::hash_code hash_value(const TensorizationInfo& tensorizationInfo)
    {
        return llvm::hash_combine(tensorizationInfo.dim, tensorizationInfo.numTotalPasses, tensorizationInfo.useStaticOffsets, tensorizationInfo.numFusedPasses, tensorizationInfo.schedulingPolicy);
    }

    llvm::hash_code hash_value(const InPlaceUnrollInfo& inPlaceUnrollInfo)
    {
        return llvm::hash_combine(inPlaceUnrollInfo.loopUnrollFactor);
    }
} // namespace executionPlan
} // namespace accera::ir
