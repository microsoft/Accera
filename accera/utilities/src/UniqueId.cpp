////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "UniqueId.h"

namespace accera
{
namespace utilities
{
    size_t UniqueId::_nextId = 1000;

    UniqueId::UniqueId()
    {
        _id = std::to_string(_nextId);
        ++_nextId;
    }

    UniqueId::UniqueId(const std::string& idString)
    {
        _id = idString;
    }

    bool UniqueId::operator==(const UniqueId& other) const
    {
        return _id == other._id;
    }

    bool UniqueId::operator!=(const UniqueId& other) const
    {
        return !(other == *this);
    }

    bool UniqueId::operator<(const UniqueId& other) const
    {
        return _id < other._id;
    }

    bool UniqueId::operator>(const UniqueId& other) const
    {
        return _id > other._id;
    }

    std::ostream& operator<<(std::ostream& stream, const UniqueId& id)
    {
        stream << id._id;
        return stream;
    }

    std::string to_string(const UniqueId& id)
    {
        return id._id;
    }
} // namespace utilities
} // namespace accera

std::hash<accera::utilities::UniqueId>::result_type std::hash<accera::utilities::UniqueId>::operator()(const argument_type& id) const
{
    return std::hash<std::string>()(id._id);
}
