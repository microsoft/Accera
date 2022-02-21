////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TranslateToCpp.h"
#include "CppPrinter.h"

using namespace llvm;

namespace mlir
{

LogicalResult translateModuleToCpp(Operation* m, raw_ostream& os)
{
    cpp_printer::CppPrinter printer(os);

    return printer.process(m);
}

} // namespace mlir
