////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/OperandIndex.h"

namespace accera::ir
{
namespace loopnest
{
    OperandIndex::OperandIndex(int64_t index) :
        _idx(index)
    {
    }

    int64_t OperandIndex::GetIndex() const
    {
        return _idx;
    }

    std::ostream& operator<<(std::ostream& os, const OperandIndex& index)
    {
        os << index.GetIndex();
        return os;
    }

} // namespace loopnest
} // namespace accera::ir

using namespace accera::ir::loopnest;

std::hash<OperandIndex>::result_type std::hash<OperandIndex>::operator()(const argument_type& element) const
{
    return static_cast<size_t>(std::hash<int64_t>()(element.GetIndex()));
}
