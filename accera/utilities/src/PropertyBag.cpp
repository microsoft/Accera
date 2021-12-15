////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "PropertyBag.h"
#include "Exception.h"
#include "TypeTraits.h"

#include <algorithm>

namespace accera
{
namespace utilities
{

    //
    // PropertyBag
    //
    const std::any& PropertyBag::GetEntry(const std::string& key) const
    {
        return _metadata.at(key);
    }

    std::any& PropertyBag::operator[](const std::string& key)
    {
        return _metadata[key];
    }

    bool PropertyBag::IsEmpty() const
    {
        if (_metadata.empty())
        {
            return true;
        }

        return !std::any_of(_metadata.begin(), _metadata.end(), [](const std::any& any) { return any.has_value(); });
    }

    std::any PropertyBag::RemoveEntry(const std::string& key)
    {
        Variant result;
        auto keyIter = _metadata.find(key);
        if (keyIter != _metadata.end())
        {
            result = keyIter->second;
            _metadata.erase(keyIter);
        }
        return result;
    }

    bool PropertyBag::HasEntry(const std::string& key) const
    {
        auto keyIter = _metadata.find(key);
        return (keyIter != _metadata.end()) && (keyIter->second.has_value());
    }

    std::vector<std::string> PropertyBag::Keys() const
    {
        std::vector<std::string> result;
        result.reserve(_metadata.size());
        for (const auto& keyValue : _metadata)
        {
            if (keyValue.second.has_value())
            {
                result.push_back(keyValue.first);
            }
        }
        return result;
    }

} // namespace utilities
} // namespace accera
