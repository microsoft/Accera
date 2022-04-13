////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AMDGPU.h"

#include <llvm/ADT/Optional.h>
#include <mlir/IR/BuiltinTypes.h>
#include <sstream>

using namespace mlir;

namespace mlir
{
namespace cpp_printer
{
    std::pair<mlir::Type, int64_t> GetElementTypeAndCount(const mlir::Type& type)
    {
        if (auto ty = type.dyn_cast<VectorType>())
        {
            return { ty.getElementType(), ty.getNumElements() };
        }

        if (auto ty = type.dyn_cast<MemRefType>())
        {
            return { ty.getElementType(), ty.getNumElements() };
        }

        assert(false && "invalid output type. Must be either an vector or a memref");
        return {};
    }

    auto GetTypeStr(const mlir::Type& type)
    {
        if (type.isInteger(8))
        {
            return "i8";
        }

        if (type.isInteger(32))
        {
            return "i32";
        }

        if (type.isF16())
        {
            return "f16";
        }

        if (type.isBF16())
        {
            return "bf16";
        }

        if (type.isF32())
        {
            return "f32";
        }

        assert(false && "invalid type. Must be either an i8, f16, f32, bf16, or i32");
        return "";
    }

    llvm::Optional<std::string> GetAMDMFMAOpName(const mlir::Type& aTy, const mlir::Type& bTy, const mlir::Type& cTy, const mlir::Type& resTy)
    {
        std::stringstream ss;

        ss << "__builtin_amdgcn_mfma_";

        const auto& [aElemType, aElemCount] = GetElementTypeAndCount(aTy);
        [[maybe_unused]] const auto& [bElemType, bElemCount] = GetElementTypeAndCount(bTy);
        [[maybe_unused]] const auto& [cElemType, cElemCount] = GetElementTypeAndCount(cTy);
        const auto& [rElemType, rElemCount] = GetElementTypeAndCount(resTy);

        assert(aElemType == bElemType && "A and B cannot have different types!");
        assert(aElemCount == bElemCount && "A and B cannot have different sizes!");
        assert(cElemType == rElemType && "C and return value cannot have different types!");
        assert(cElemCount == rElemCount && "C and return value cannot have different sizes!");
        assert((aElemType.isF32() || aElemType.isInteger(8) || aElemType.isF16() || aElemType.isBF16()) && "invalid input type. Must be either an i8, f16, bf16, or f32");
        assert((rElemType.isF32() || rElemType.isInteger(32)) && "invalid output type. Must be either an f32 or i32");

        ss << GetTypeStr(rElemType);

        switch (rElemCount)
        {
        case 16:
            if (aElemType.isInteger(8) || aElemType.isF16())
            {
                ss << "_32x32x8";
            }
            else if (aElemType.isBF16())
            {
                ss << "_32x32x4";
            }
            else if (aElemType.isF32())
            {
                ss << "_32x32x2";
            }
            break;
        case 4:
            if (aElemType.isInteger(8) || aElemType.isF16())
            {
                ss << "_16x16x16";
            }
            else if (aElemType.isBF16())
            {
                ss << "_16x16x8";
            }
            else if (aElemType.isF32())
            {
                ss << "_16x16x4";
            }
            break;
        default:
            assert(false && "invalid input type.");
        }

        ss << GetTypeStr(aElemType);

        return ss.str();
    }

} // namespace cpp_printer
} // namespace mlir