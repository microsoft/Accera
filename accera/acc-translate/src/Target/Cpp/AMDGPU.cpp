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

    llvm::Optional<std::string> GetAMDMFMAOpName(mlir::Type outTy, mlir::Type inTy)
    {
        std::stringstream ss;

        ss << "__builtin_amdgcn_mfma_";
        mlir::Type outElementTy, inElementTy;
        size_t outNumElems, inNumElems;

        if (auto ty = outTy.dyn_cast<VectorType>())
        {
            outElementTy = ty.getElementType();
            outNumElems = ty.getNumElements();
        }
        else if (auto ty = outTy.dyn_cast<MemRefType>())
        {
            outElementTy = ty.getElementType();
            outNumElems = ty.getNumElements();
        }
        else
        {
            assert(true && "invalid output type. Must be either an vector or a memref");
        }

        if (auto ty = inTy.dyn_cast<VectorType>())
        {
            inElementTy = ty.getElementType();
            inNumElems = ty.getNumElements();
        }
        else if (auto ty = inTy.dyn_cast<MemRefType>())
        {
            inElementTy = ty.getElementType();
            inNumElems = ty.getNumElements();
        }
        else
        {
            assert(true && "invalid input type. Must be either an vector or a memref");
        }

        if (outElementTy.isF32())
        {
            ss << "f32";
        }
        else if (outElementTy.isInteger(32))
        {
            ss << "i32";
        }
        else
        {
            assert(true && "invalid output type. Must be either an f32 or i32");
        }

        assert(outNumElems == inNumElems);

        if (inNumElems == 2)
        {
            ss << "_32x32x2";
        }
        else if (inNumElems == 4)
        {
            ss << "_16x16x4";
        }
        else
        {
            assert(true && "invalid input type. Must be either an 2 or 4 element vector");
        }

        if (inElementTy.isInteger(8))
        {
            ss << "i8";
        }
        else if (inElementTy.isF16())
        {
            ss << "f16";
        }
        else if (inElementTy.isBF16())
        {
            ss << "bf16";
        }
        else if (inElementTy.isF32())
        {
            ss << "f32";
        }
        else
        {
            assert(true && "invalid input type. Must be either an i8, f16, bf16, or i32");
        }

        return ss.str();
    }

} // namespace cpp_printer
} // namespace mlir