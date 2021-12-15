////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>

namespace accera::ir
{
namespace loopnest
{

    namespace detail
    {
        class ArrayTypeStorage : public mlir::TypeStorage
        {
        public:
            using KeyTy = std::pair<llvm::ArrayRef<int64_t>, mlir::Type>;

            ArrayTypeStorage(llvm::ArrayRef<int64_t> shape, mlir::Type elementType) :
                _shape(shape),
                _elementType(elementType) {}

            bool operator==(const KeyTy& key) const
            {
                return key == KeyTy(getShape(), getElementType());
            }

            static ArrayTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                               const KeyTy& key)
            {
                // Copy the shape into the bump pointer.
                auto shape = allocator.copyInto(key.first);

                // Initialize the memory using placement new.
                return new (allocator.allocate<ArrayTypeStorage>())
                    ArrayTypeStorage(shape, key.second);
            }

            llvm::ArrayRef<int64_t> getShape() const { return _shape; }

            mlir::Type getElementType() const { return _elementType; }

        private:
            llvm::ArrayRef<int64_t> _shape;
            mlir::Type _elementType;
        };
    } // namespace detail

    class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type, detail::ArrayTypeStorage>
    {
    public:
        using Base::Base;

        static ArrayType get(mlir::MLIRContext* context, llvm::ArrayRef<int64_t> sizes, mlir::Type elementType);
        static ArrayType get(mlir::MLIRContext* context, mlir::ArrayAttr sizes, mlir::Type elementType);
        static ArrayType get(llvm::ArrayRef<int64_t> sizes, mlir::Type elementType);
        static ArrayType get(mlir::ArrayAttr sizes, mlir::Type elementType);

        static bool isValidElementType(mlir::Type type);
        static mlir::LogicalResult verifyConstructionInvariants(mlir::Location location, mlir::ArrayRef<int64_t> shape, mlir::Type elementType);

        llvm::ArrayRef<int64_t> getShape() const;
        mlir::Type getElementType() const;
    };

    class KernelType : public mlir::Type::TypeBase<KernelType, mlir::Type, mlir::DefaultTypeStorage>
    {
    public:
        using Base::Base;

        static KernelType get(mlir::MLIRContext* context);
    };

    class SymbolicIndexType : public mlir::Type::TypeBase<SymbolicIndexType, mlir::Type, mlir::DefaultTypeStorage>
    {
    public:
        using Base::Base;

        static SymbolicIndexType get(mlir::MLIRContext* context);
    };

    // Parse and print functions
    mlir::Type parseArrayType(mlir::DialectAsmParser& os);
    mlir::Type parseSymbolicIndexType(mlir::DialectAsmParser& os);
    mlir::Type parseKernelType(mlir::DialectAsmParser& os);

    void print(ArrayType type, mlir::DialectAsmPrinter& os);
    void print(SymbolicIndexType type, mlir::DialectAsmPrinter& os);
    void print(KernelType type, mlir::DialectAsmPrinter& os);

} // namespace loopnest
} // namespace accera::ir
