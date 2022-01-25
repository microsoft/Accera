////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CPP_PRINTER_UTILS_H_
#define CPP_PRINTER_UTILS_H_

namespace mlir
{
namespace cpp_printer
{

    // Return the round-up number of bits that are valid for integer types, e.g.
    // 8, 16, 32, and 64
    int getIntTypeBitCount(int width);

} // namespace cpp_printer
} // namespace mlir

#endif
