////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Matrix.h"

namespace accera
{
namespace value
{
    /// <summary> A special type Matrix which is used for tensor core operations on the GPU. </summary>
    class MatrixFragment : public Matrix
    {
    public:
        enum class Shape
        {
            M64xN64xK1_B4,
            M64xN64xK1_B2,
            M32xN32xK2_B1,
            M16xN16xK4_B1,

            M64xN64xK2_B4,
            M64xN64xK2_B2,
            M32xN32xK4_B1,
            M16xN16xK8_B1,

            M64xN64xK4_B4,
            M64xN64xK4_B2,
            M32xN32xK8_B1,
            M16xN16xK16_B1,

            M32xN8xK16_B1,
            M8xN32xK16_B1,

            Invalid
        };

        enum class Type
        {
            A,
            B,
            Acc
        };

        /// <summary> Constructor used to declare a matrix fragment with a particular shape and type, owned by a warp of threads. </summary>
        /// <param name="shape"> The shape of the matrix fragment </param>
        /// <param name="type"> The type of the matrix fragment </param>
        MatrixFragment(Shape shape, Type type);

        MatrixFragment(const MatrixFragment&) = default;
        MatrixFragment(MatrixFragment&&) noexcept = default;
        MatrixFragment& operator=(const MatrixFragment&);
        MatrixFragment& operator=(MatrixFragment&&);

        /// <summary> Load the data from the source into the matrix fragment. Loading is collaboratively performed by a warp of threads. </summary>
        /// <param name="sourceMatrix"> The source from which to load the data. </param>
        /// <param name="rowOffset"> The row offset within the source matrix. </param>
        /// <param name="colOffset"> The column offset within the source matrix. </param>
        void LoadSync(const Matrix& sourceMatrix, int64_t rowOffset = 0, int64_t colOffset = 0);

        /// <summary> Calculate the matrix multiplication of the current matrix fragment (A) with B and then add the result with matrix fragment C the matrix fragment.
        ///           Matrix multiplication is collaboratively performed by a warp of threads. </summary>
        /// <param name="B"> The RHS operand of the matrix multiplication. </param>
        /// <param name="C"> The intial value of the matrix multiplication accumulator. </param>
        /// <param name="cbsz"> (AMD specific) Defines the number of blocks that can do a broadcast within a group. Legal values = 0-4. The block ID of this group comes from ABID. </param>
        /// <param name="abid"> (AMD specific) Block ID of block to broadcast during matrix multiply (MMA ops). </param>
        /// <param name="blgp"> (AMD specific) “B”-Matrix Lane-Group Pattern. Controls how to swizzle the matrix lane groups (LG) in VGPRs when doing matrix multiplication by controlling the swizzle muxes. </param>
        /// <returns> The result of A * B + C. </returns>
        MatrixFragment MultiplyAccumulateSync(const MatrixFragment& B, const MatrixFragment& C, uint32_t cbsz = 0, uint32_t abid = 0, uint32_t blgp = 0) const;

        /// <summary> Store the data from the matrix fragment into the target. Storing is collaboratively performed by a warp of threads. </summary>
        /// <param name="targetMatrix"> The target to which to store the data. </param>
        /// <param name="rowOffset"> The row offset within the target matrix. </param>
        /// <param name="colOffset"> The column offset within the target matrix. </param>
        void StoreSync(Matrix& targetMatrix, int64_t rowOffset = 0, int64_t colOffset = 0) const;

        /// <summary> Get the shape of the tensor operation. </summary>
        Shape GetFragmentShape() const;

        /// <summary> Get the type of the matrix fragment. </summary>
        Type GetFragmentType() const;

    private:
        MatrixFragment(Value value, Shape shape, Type type);

        Type _type;
        Shape _shape;
    };
} // namespace value
} // namespace accera