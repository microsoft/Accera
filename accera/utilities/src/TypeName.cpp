////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TypeName.h"

#include <string>
#include <vector>

namespace accera
{
namespace utilities
{
    std::string GetCompositeTypeName(std::string baseType, const std::vector<std::string>& subtypes)
    {
        if (subtypes.size() == 0)
        {
            return baseType;
        }
        std::string result = baseType + "<";
        for (size_t index = 0; index < subtypes.size(); ++index)
        {
            if (index != 0)
            {
                result += ",";
            }
            result += subtypes[index];
        }
        result += ">";
        return result;
    }
} // namespace utilities
} // namespace accera
