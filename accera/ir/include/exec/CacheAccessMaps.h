////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <utilities/include/MemoryLayout.h>

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace accera::ir
{
namespace executionPlan
{
    class CacheAccessMaps
    {
    public:
        CacheAccessMaps() = default;

        CacheAccessMaps(mlir::AffineMap relevantIndicesToActiveElementCacheMap,
                        mlir::AffineMap relevantIndicesToInputMap,
                        mlir::AffineMap inputIndicesToActiveBlockCacheMap,
                        const utilities::MemoryAffineCoefficients& cacheMemCoefficients,
                        const utilities::DimensionOrder& cacheDimOrder) :
            relevantIndicesToActiveElementCache(relevantIndicesToActiveElementCacheMap),
            relevantIndicesToInput(relevantIndicesToInputMap),
            inputIndicesToActiveBlockCache(inputIndicesToActiveBlockCacheMap),
            coefficients(cacheMemCoefficients),
            dimOrder(cacheDimOrder)
        {}

        mlir::DictionaryAttr ToAttr(mlir::OpBuilder& builder);
        static CacheAccessMaps FromAttr(mlir::DictionaryAttr dictAttr);

        mlir::AffineMap relevantIndicesToActiveElementCache;
        mlir::AffineMap relevantIndicesToInput;
        mlir::AffineMap inputIndicesToActiveBlockCache;
        utilities::MemoryAffineCoefficients coefficients;
        utilities::DimensionOrder dimOrder;

    private:
        friend inline bool operator==(const CacheAccessMaps& cam1, const CacheAccessMaps& cam2)
        {
            return (cam1.relevantIndicesToActiveElementCache == cam2.relevantIndicesToActiveElementCache) &&
                   (cam1.relevantIndicesToInput == cam2.relevantIndicesToInput) &&
                   (cam1.inputIndicesToActiveBlockCache == cam2.inputIndicesToActiveBlockCache) &&
                   (cam1.coefficients == cam2.coefficients) &&
                   (cam1.dimOrder == cam2.dimOrder);
        }
        friend inline bool operator!=(const CacheAccessMaps& cam1, const CacheAccessMaps& cam2)
        {
            return !(cam1 == cam2);
        }
    };

} // namespace executionPlan
} // namespace accera::ir
