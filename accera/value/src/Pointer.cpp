////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Pointer.h"

#include <utilities/include/Exception.h>

namespace accera
{
using namespace utilities;

namespace value
{
    Pointer::Pointer() = default;

    Pointer::Pointer(Value value, const std::string& name) :
        _value(value)
    {
        if (!_value.IsDefined() || !_value.IsConstrained())
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must be defined and have a memory layout");
        }
        if (_value.PointerLevel() < 1)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must have a pointer level >= 1");
        }
        if (!name.empty())
        {
            SetName(name);
        }
    }

    Pointer::~Pointer() = default;
    Pointer::Pointer(const Pointer&) = default;
    Pointer::Pointer(Pointer&&) noexcept = default;

    Value Pointer::GetValue() const { return _value; }

    utilities::MemoryLayout Pointer::GetDataLayout() const { return _value.GetLayout(); }
    ValueType Pointer::GetDataType() const { return _value.GetBaseType(); }

    void Pointer::Store(ViewAdapter data)
    {
        GetContext().Store(data.GetValue(), _value, std::vector<int64_t>{ 0 });
    }

    void Pointer::SetName(const std::string& name) { _value.SetName(name); }
    std::string Pointer::GetName() const { return _value.GetName(); }

} // namespace value
} // namespace accera