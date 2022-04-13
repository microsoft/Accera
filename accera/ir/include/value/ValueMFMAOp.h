////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ValueAttributes.h"
#include "ValueEnums.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/TypeSupport.h>

namespace accera::ir::value
{
using llvm::ArrayRef;
using llvm::StringRef;

using mlir::LogicalResult;
using mlir::Type;

/// MFMAMatrixType storage and uniquing. Array is uniqued based on its shape,
/// type, and operand.
struct MFMAMatrixStorageType : public mlir::TypeStorage
{
    MFMAMatrixStorageType(unsigned numDims_, const int64_t* dimShapes_, Type elementType_, StringRef operand_) :
        dimShapes(dimShapes_), numDims(numDims_), elementType(elementType_), operand(operand_) {}

    /// The hash key for uniquing.
    using KeyTy = std::tuple<ArrayRef<int64_t>, Type, StringRef>;
    bool operator==(const KeyTy& key) const
    {
        return key == KeyTy(getShape(), elementType, operand);
    }

    /// Construction.
    static MFMAMatrixStorageType* construct(mlir::TypeStorageAllocator& allocator,
                                            const KeyTy& key)
    {
        ArrayRef<int64_t> shape = allocator.copyInto(std::get<0>(key));
        StringRef operand = allocator.copyInto(std::get<2>(key));

        return new (allocator.allocate<MFMAMatrixStorageType>())
            MFMAMatrixStorageType(shape.size(), shape.data(), std::get<1>(key), operand);
    }

    ArrayRef<int64_t> getShape() const
    {
        return ArrayRef<int64_t>(dimShapes, numDims);
    }

    StringRef getOperand() const { return operand; }

    /// Reference to the shape of the MMA matrix.
    const int64_t* dimShapes;

    /// Number of dimensions in the MMA matrix.
    unsigned numDims;

    /// Element type of elements held in the MMA matrix.
    Type elementType;

    /// MMA operand that this MFMAMatrix holds. The general form of operation this
    /// type supports is given by the equation C += A*B. This field specifies
    /// which operand in the given equation is held by this type. The valid values
    /// are "AOp", "BOp" and "COp".
    StringRef operand;
};

/// MFMAMatrix represents a matrix held by for matrix-matrix multiply
/// accumulate operations. MFMAMatrices are taken as direct operands by these
/// operations and are also produced as results. These matrices are meant to
/// reside in the registers. A limited number of pointwise operations can be
/// performed on these matrices, i.e., operations which operate uniformly on
/// all the elements in the matrix and do not change the order of matrix
/// elements. The above conditions exist because the layout of matrix elements
/// inside the matrix is opaque i.e., the elements may be present in the
/// matrix in any order. The general usage of this type is shown as follows:-
///
///   %0 = accv.mfma_load_matrix %arg0[%c0, %c0] {leadDimension = 16 :
///           index} : memref<16x16xf16> -> !accv.mfma_matrix<16x16xf16, "AOp">
///
/// The MFMAMatrixType describes the shape of the matrix being loaded and the
/// operand being loaded too. The operand needs to be specified to aid the
/// lowering of this type to dialects such as NVVM where each workitem may
/// hold different amount of elements depending on the elementType of the
/// matrix. For e.g., Each workitem holds 4 vector<2xf16>s for f16 data type
/// and 8 f32s for f32 data type of MFMAMatrix. Some other instances of usage
/// are:-
///
///   %3 = accv.mfma_compute %0, %1, %2 :
///   !accv.mfma_matrix<16x16xf16, "AOp">, !accv.mfma_matrix<16x16xf16, "BOp">
///    -> !accv.mfma_matrix<16x16xf32, "COp">
///
///
///   accv.mfma_store_matrix %3, %arg22[%c0, %c0] {leadDimension = 16
///           : index}: !accv.mfma_matrix<16x16xf32, "COp">, memref<16x16xf32>
class MFMAMatrixType
    : public Type::TypeBase<MFMAMatrixType, Type, MFMAMatrixStorageType>
{
public:
    using Base::Base;

    enum class Shape
    {
        T4x16x64,
        T2x32x64,
        T4x4x32,
        T2x2x16,
        Invalid
    };

    /// Get MFMAMatrixType and verify construction Invariants.
    static MFMAMatrixType get(ArrayRef<int64_t> shape, Type elementType, StringRef operand);

    /// Get MFMAMatrixType at a particular location and verify construction
    /// Invariants.
    static MFMAMatrixType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                     ArrayRef<int64_t> shape,
                                     Type elementType,
                                     StringRef operand);

    /// Check if a type is valid a MFMAMatrixType elementType.
    static bool isValidElementType(Type elementType);

    /// Verify that shape and elementType are actually allowed for the
    /// MFMAMatrixType.
    static LogicalResult verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                ArrayRef<int64_t> shape,
                                Type elementType,
                                StringRef operand);

    /// Get number of dims.
    unsigned getNumDims() const;

    /// Get shape of the matrix.
    ArrayRef<int64_t> getShape() const;

    /// Get elementType of a single element.
    Type getElementType() const;

    /// The general form of operation this type supports is given by the equation
    /// C += A*B. This function returns which operand in the given equation is
    /// held by this type. String returned can be one of"AOp", "BOp" and "COp".
    StringRef getOperand() const;

    /// The stride between consecutive rows of the MFMA matrix.
    int64_t getLeadingDim() const;

    Shape getShapeType() const;

    int64_t getThreadTileSize() const;

    bool isValidShape() const;

    int64_t getNumBlocks() const;

    int64_t getTileFactor() const;

private:
    static Shape getShapeType(const ArrayRef<int64_t>& mfmaShape);
};
} // namespace accera::ir::value
