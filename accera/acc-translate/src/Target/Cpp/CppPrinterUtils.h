//===- CppPrinterUtils.h - common utility functions -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CPP_PRINTER_UTILS_H_
#define CPP_PRINTER_UTILS_H_

namespace mlir {
namespace cpp_printer {

// Return the round-up number of bits that are valid for integer types, e.g.
// 8, 16, 32, and 64
int getIntTypeBitCount(int width);

} // namespace cpp_printer
} // namespace mlir

#endif
