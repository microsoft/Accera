####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(LLVM_TARGET_DEFINITIONS AcceraPasses.td)
mlir_tablegen(AcceraPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(AcceraConversionIncGen)
