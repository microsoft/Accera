////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/LoopNestTypes.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/BuiltinTypes.h>

namespace accera::ir
{
namespace loopnest
{

    //
    // ArrayType
    //
    ArrayType ArrayType::get(mlir::MLIRContext* context, llvm::ArrayRef<int64_t> shape, mlir::Type elementType)
    {
        return Base::get(context, shape, elementType);
    }

    ArrayType ArrayType::get(mlir::MLIRContext* context, mlir::ArrayAttr shapeAttr, mlir::Type elementType)
    {
        std::vector<int64_t> shape;
        for (auto s : shapeAttr)
        {
            shape.push_back(s.cast<mlir::IntegerAttr>().getInt());
        }
        return Base::get(context, shape, elementType);
    }

    ArrayType ArrayType::get(llvm::ArrayRef<int64_t> shape, mlir::Type elementType)
    {
        return Base::get(elementType.getContext(), shape, elementType);
    }

    ArrayType ArrayType::get(mlir::ArrayAttr shapeAttr, mlir::Type elementType)
    {
        std::vector<int64_t> shape;
        for (auto s : shapeAttr)
        {
            shape.push_back(s.cast<mlir::IntegerAttr>().getInt());
        }
        return Base::get(elementType.getContext(), shape, elementType);
    }

    bool ArrayType::isValidElementType(mlir::Type type)
    {
        return mlir::TensorType::isValidElementType(type); // punt to Tensor for now
    }

    mlir::LogicalResult ArrayType::verifyConstructionInvariants(mlir::Location location, mlir::ArrayRef<int64_t> shape, mlir::Type elementType)
    {
        for (int64_t s : shape)
        {
            if (s < -1)
                return emitError(location, "invalid tensor dimension size");
        }

        if (!mlir::TensorType::isValidElementType(elementType))
            return emitError(location, "invalid tensor element type");
        return mlir::success();
    }

    llvm::ArrayRef<int64_t> ArrayType::getShape() const
    {
        return getImpl()->getShape();
    }

    mlir::Type ArrayType::getElementType() const
    {
        return getImpl()->getElementType();
    }

    mlir::Type parseArrayType(mlir::DialectAsmParser& parser)
    {
        llvm::SmallVector<int64_t, 4> dimensions;

        if (failed(parser.parseLess()))
            return {};

        if (failed(parser.parseDimensionList(dimensions, false)))
            return {};

        // Parse the element type.
        auto elementTypeLoc = parser.getCurrentLocation();
        mlir::Type elementType;
        if (failed(parser.parseType(elementType)))
        {
            return {};
        }
        if (failed(parser.parseGreater()))
            return {};

        if (!ArrayType::isValidElementType(elementType))
        {
            parser.emitError(elementTypeLoc, "invalid tensor element type");
            return {};
        }

        return ArrayType::get(std::vector<int64_t>(dimensions.begin(), dimensions.end()), elementType);
    }

    void print(ArrayType type, mlir::DialectAsmPrinter& printer)
    {
        auto shape = type.getShape();

        printer << "array<"; // TODO: print size and element type as well
        for (auto dim : shape)
        {
            if (dim < 0)
                printer << '?';
            else
                printer << dim;
            printer << 'x';
        }
        printer << "f64>"; // TODO: print a real element type
    }

    //
    // KernelType
    //
    KernelType KernelType::get(mlir::MLIRContext* context)
    {
        return Base::get(context);
    }

    mlir::Type parseKernelType(mlir::DialectAsmParser& parser)
    {
        return KernelType::get(parser.getBuilder().getContext());
    }

    void print(KernelType type, mlir::DialectAsmPrinter& printer)
    {
        printer << "kernel";
    }

    //
    // SymbolicIndexType
    //
    SymbolicIndexType SymbolicIndexType::get(mlir::MLIRContext* context)
    {
        return Base::get(context);
    }

    mlir::Type parseSymbolicIndexType(mlir::DialectAsmParser& parser)
    {
        return SymbolicIndexType::get(parser.getBuilder().getContext());
    }

    void print(SymbolicIndexType type, mlir::DialectAsmPrinter& printer)
    {
        printer << "symbolic_index";
    }

} // namespace loopnest
} // namespace accera::ir
