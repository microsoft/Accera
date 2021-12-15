////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LOOPNEST_TO_VALUE
#define LOOPNEST_TO_VALUE

include "ir/include/nest/LoopNestOps.td"
include "ir/include/value/ValueOps.td"

def : Pat<(rcln_PrintOp $input, $to_stderr), (rcv_PrintOp $input, $to_stderr)>;

#endif // LOOPNEST_TO_VALUE
