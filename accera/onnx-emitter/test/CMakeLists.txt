####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(target onnx_emitter_tests)
set(dest ${CMAKE_BINARY_DIR}/onnx-emitter)

configure_file(${target}.py ${dest}/${target}.py @ONLY)
add_test(NAME ${target} COMMAND python ${dest}/${target}.py)