////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CppPrinterUtils.h"

namespace mlir
{
namespace cpp_printer
{

    int getIntTypeBitCount(int width)
    {
        int bitCount = -1;
        if (width <= 8)
        {
            bitCount = 8;
        }
        else if (width <= 16)
        {
            bitCount = 16;
        }
        else if (width <= 32)
        {
            bitCount = 32;
        }
        else if (width <= 64)
        {
            bitCount = 64;
        }
        else
        {
        }
        return bitCount;
    }

} // namespace cpp_printer
} // namespace mlir
