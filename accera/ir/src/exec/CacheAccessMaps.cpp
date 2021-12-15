////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "exec/CacheAccessMaps.h"
#include "IRUtil.h"

namespace accera::ir
{
namespace executionPlan
{
    static const std::string RelevantIndicesToActiveElementCacheName = "relevantIndicesToActiveElementCache";
    static const std::string RelevantIndicesToInputName = "relevantIndicesToInput";
    static const std::string InputIndicesToActiveBlockCacheName = "inputIndicesToActiveBlockCache";
    static const std::string MemoryAffineCoefficientsName = "manualCacheMemoryAffineCoefficients";
    static const std::string DimOrderName = "manualCacheDimOrder";

    static const std::string CoefficientsName = "coefficients";
    static const std::string OffsetName = "offset";

    mlir::DictionaryAttr SerializeMemoryAffineCoefficients(mlir::OpBuilder& builder, const utilities::MemoryAffineCoefficients& coefficients)
    {
        std::vector<mlir::NamedAttribute> mappings;

        mappings.emplace_back(std::make_pair(
            builder.getIdentifier(CoefficientsName),
            builder.getI64ArrayAttr(coefficients.coefficients)));

        mappings.emplace_back(std::make_pair(
            builder.getIdentifier(OffsetName),
            builder.getI64IntegerAttr(coefficients.offset)));

        return mlir::DictionaryAttr::get(builder.getContext(),mappings);
    }

    utilities::MemoryAffineCoefficients DeserializeMemoryAffineCoefficients(mlir::DictionaryAttr dictAttr)
    {
        utilities::MemoryAffineCoefficients coefficients;

        auto coefficientsAttr = dictAttr.get(CoefficientsName);
        assert(coefficientsAttr.isa<mlir::ArrayAttr>());
        coefficients.coefficients = util::ConvertArrayAttrToIntVector(coefficientsAttr.cast<mlir::ArrayAttr>());

        auto offsetAttr = dictAttr.get(OffsetName);
        assert(offsetAttr.isa<mlir::IntegerAttr>());
        coefficients.offset = offsetAttr.cast<mlir::IntegerAttr>().getInt();

        return coefficients;
    }

    mlir::DictionaryAttr CacheAccessMaps::ToAttr(mlir::OpBuilder& builder)
    {
        std::vector<mlir::NamedAttribute> mappings;

        if (relevantIndicesToActiveElementCache)
        {
            mappings.emplace_back(std::make_pair(
                builder.getIdentifier(RelevantIndicesToActiveElementCacheName),
                mlir::AffineMapAttr::get(relevantIndicesToActiveElementCache)));
        }
        if (relevantIndicesToInput)
        {
            mappings.emplace_back(std::make_pair(
                builder.getIdentifier(RelevantIndicesToInputName),
                mlir::AffineMapAttr::get(relevantIndicesToInput)));
        }
        if (inputIndicesToActiveBlockCache)
        {
            mappings.emplace_back(std::make_pair(
                builder.getIdentifier(InputIndicesToActiveBlockCacheName),
                mlir::AffineMapAttr::get(inputIndicesToActiveBlockCache)));
        }
        if (!coefficients.coefficients.empty())
        {
            mappings.emplace_back(std::make_pair(
                builder.getIdentifier(MemoryAffineCoefficientsName),
                SerializeMemoryAffineCoefficients(builder, coefficients)));
        }
        auto dimOrderVec = dimOrder.ToVector();
        if (!dimOrderVec.empty())
        {
            mappings.emplace_back(std::make_pair(
                builder.getIdentifier(DimOrderName),
                builder.getI64ArrayAttr(dimOrderVec)));
        }

        return mlir::DictionaryAttr::get(builder.getContext(), mappings);
    }

    CacheAccessMaps CacheAccessMaps::FromAttr(mlir::DictionaryAttr dictAttr)
    {
        mlir::AffineMap relevantIndicesToActiveElementCache;
        mlir::AffineMap relevantIndicesToInput;
        mlir::AffineMap inputIndicesToActiveBlockCache;
        utilities::MemoryAffineCoefficients coefficients;
        utilities::DimensionOrder dimOrder;

        if (auto relevantIndicesToActiveElementCacheAttr = dictAttr.get(RelevantIndicesToActiveElementCacheName))
        {
            assert(relevantIndicesToActiveElementCacheAttr.isa<mlir::AffineMapAttr>() && "Non-affine-map found in cache access maps dictionary");
            relevantIndicesToActiveElementCache = relevantIndicesToActiveElementCacheAttr.cast<mlir::AffineMapAttr>().getValue();
        }
        if (auto relevantIndicesToInputAttr = dictAttr.get(RelevantIndicesToInputName))
        {
            assert(relevantIndicesToInputAttr.isa<mlir::AffineMapAttr>() && "Non-affine-map found in cache access maps dictionary");
            relevantIndicesToInput = relevantIndicesToInputAttr.cast<mlir::AffineMapAttr>().getValue();
        }
        if (auto inputIndicesToActiveBlockCacheAttr = dictAttr.get(InputIndicesToActiveBlockCacheName))
        {
            assert(inputIndicesToActiveBlockCacheAttr.isa<mlir::AffineMapAttr>() && "Non-affine-map found in cache access maps dictionary");
            inputIndicesToActiveBlockCache = inputIndicesToActiveBlockCacheAttr.cast<mlir::AffineMapAttr>().getValue();
        }

        if (auto memoryAffineCoefficientsAttr = dictAttr.get(MemoryAffineCoefficientsName))
        {
            assert(memoryAffineCoefficientsAttr.isa<mlir::DictionaryAttr>() && "Memory affine coefficients attr must be a mlir::DictionaryAttr");
            coefficients = DeserializeMemoryAffineCoefficients(memoryAffineCoefficientsAttr.cast<mlir::DictionaryAttr>());
        }
        if (auto dimOrderAttr = dictAttr.get(DimOrderName))
        {
            assert(dimOrderAttr.isa<mlir::ArrayAttr>() && "DimOrder attr must be an ArrayAttr of integers");
            dimOrder = util::ConvertArrayAttrToIntVector(dimOrderAttr.cast<mlir::ArrayAttr>());
        }
        return CacheAccessMaps(relevantIndicesToActiveElementCache,
                               relevantIndicesToInput,
                               inputIndicesToActiveBlockCache,
                               coefficients,
                               dimOrder);
    }

} // namespace executionPlan
} // namespace accera::ir
