////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Metadata.h"
#include "IRUtil.h"

#include <utilities/include/Exception.h>
#include <utilities/include/TypeTraits.h>


using namespace accera::utilities;

namespace accera
{
namespace ir
{
    Metadata ParseFullMetadata(mlir::DictionaryAttr metadataDictionary)
    {
        Metadata metadataMap;
        if (metadataDictionary == nullptr)
        {
            return metadataMap; // treat a null dictionary as an empty dictionary
        }
        for (auto dictPair : metadataDictionary)
        {
            auto key = dictPair.first;
            auto keyStr = key.str();
            auto attr = dictPair.second;
            if (auto boolAttr = attr.dyn_cast_or_null<mlir::BoolAttr>())
            {
                metadataMap.emplace(keyStr, boolAttr.getValue());
            }
            else if (auto intAttr = attr.dyn_cast_or_null<mlir::IntegerAttr>())
            {
                metadataMap.emplace(keyStr, intAttr.getInt());
            }
            else if (auto strAttr = attr.dyn_cast_or_null<mlir::StringAttr>())
            {
                metadataMap.emplace(keyStr, strAttr.getValue().str());
            }
            else if (auto arrayAttr = attr.dyn_cast_or_null<mlir::ArrayAttr>())
            {
                if (arrayAttr[0].isa<mlir::BoolAttr>())
                {
                    auto boolVec = util::ArrayAttrToVector<utilities::Boolean, mlir::BoolAttr>(arrayAttr, [&](const mlir::BoolAttr& boolAttr) {
                        return boolAttr.getValue();
                    });
                    metadataMap.emplace(keyStr, boolVec);
                }
                else if (arrayAttr[0].isa<mlir::IntegerAttr>())
                {
                    auto intVec = util::ArrayAttrToVector<int64_t, mlir::IntegerAttr>(arrayAttr, [&](const mlir::IntegerAttr& intAttr) {
                        return intAttr.getInt();
                    });
                    metadataMap.emplace(keyStr, intVec);
                }
                else if (arrayAttr[0].isa<mlir::StringAttr>())
                {
                    auto strVec = util::ArrayAttrToVector<std::string, mlir::StringAttr>(arrayAttr, [&](const mlir::StringAttr& strAttr) {
                        return strAttr.getValue().str();
                    });
                    metadataMap.emplace(keyStr, strVec);
                }
            }
            else
            {
                throw InputException(InputExceptionErrors::invalidArgument, "Unsupported metadata value type");
            }
        }
        return metadataMap;
    }

    mlir::Attribute GetMetadataAttr(const MetadataValueType& value, mlir::MLIRContext* context)
    {
        return std::visit(
            VariantVisitor{
                [&](bool boolVal) {
                    return mlir::BoolAttr::get(context, boolVal).cast<mlir::Attribute>();
                },
                [&](int intVal) {
                    return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), intVal).cast<mlir::Attribute>();
                },
                [&](int64_t intVal) {
                    return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), intVal).cast<mlir::Attribute>();
                },
                [&](const std::string& stringVal) {
                    return mlir::StringAttr::get(context, stringVal).cast<mlir::Attribute>();
                },
                [&](const std::vector<utilities::Boolean>& boolVec) {
                    return util::VectorToArrayAttr<utilities::Boolean, mlir::BoolAttr>(
                               boolVec, [&](const utilities::Boolean& boolVal) {
                                   return mlir::BoolAttr::get(context, boolVal);
                               },
                               context)
                        .cast<mlir::Attribute>();
                },
                [&](const std::vector<int>& intVec) {
                    return util::VectorToArrayAttr<int, mlir::IntegerAttr>(
                               intVec, [&](const int& intVal) {
                                   return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), intVal);
                               },
                               context)
                        .cast<mlir::Attribute>();
                },
                [&](const std::vector<int64_t>& intVec) {
                    return util::VectorToArrayAttr<int64_t, mlir::IntegerAttr>(
                               intVec, [&](const int64_t& intVal) {
                                   return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), intVal);
                               },
                               context)
                        .cast<mlir::Attribute>();
                },
                [&](const std::vector<std::string>& strVec) {
                    return util::VectorToArrayAttr<std::string, mlir::StringAttr>(
                               strVec, [&](const std::string& stringVal) {
                                   return mlir::StringAttr::get(context, stringVal);
                               },
                               context)
                        .cast<mlir::Attribute>();
                } },
            value);
    }
} // namespace ir
} // namespace accera
