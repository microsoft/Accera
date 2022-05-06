////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MatrixFragment.h"
#include "EmitterContext.h"

namespace accera
{
namespace value
{
    MatrixFragment::MatrixFragment(const Shape shape, const Type type) :
        _type{ type }, _shape{ shape }
    {
    }

    MatrixFragment::MatrixFragment(Value value, const Shape shape, const Type type) :
        Matrix{ value }, _type{ type }, _shape{ shape }
    {
    }

    MatrixFragment& MatrixFragment::operator=(const MatrixFragment& other)
    {
        if (this != &other)
        {
            Matrix::operator=(other);
            _type = other._type;
            _shape = other._shape;
        }
        return *this;
    }

    MatrixFragment& MatrixFragment::operator=(MatrixFragment&& other)
    {
        if (this != &other)
        {
            Matrix::operator=(std::move(other));
            _type = other._type;
            _shape = other._shape;
        }
        return *this;
    }

    void MatrixFragment::LoadSync(const Matrix& sourceMatrix, const int64_t rowOffset, const int64_t colOffset)
    {
        SetValue(GetContext().MMALoadSync(sourceMatrix, rowOffset, colOffset, *this));
    }

    MatrixFragment MatrixFragment::MultiplyAccumulateSync(const MatrixFragment& B, const MatrixFragment& C, uint32_t cbsz, uint32_t abid, uint32_t blgp) const
    {
        auto val = GetContext().MMAComputeSync(*this, B, C, cbsz, abid, blgp);
        return { val, _shape, Type::Acc };
    }

    void MatrixFragment::StoreSync(Matrix& targetMatrix, int64_t rowOffset, int64_t colOffset) const
    {
        GetContext().MMAStoreSync(*this, targetMatrix, rowOffset, colOffset);
    }

    MatrixFragment::Shape MatrixFragment::GetFragmentShape() const
    {
        return _shape;
    }

    MatrixFragment::Type MatrixFragment::GetFragmentType() const
    {
        return _type;
    }
} // namespace value
} // namespace accera
